import torch
from logzero import logger

from transformers import AutoTokenizer
from flash_vstream import VStreamLlamaForCausalLM

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV


class FlashVStream_ReKV(VStreamLlamaForCausalLM, Abstract_ReKV):
    def __init__(self, config, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        VStreamLlamaForCausalLM.__init__(self, config)
        Abstract_ReKV.__init__(self, None, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}ASSISTANT:"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):  # (Nv, 3, H, W)
        video_features = self.encode_images(pixel_values_videos)  # (Nv, 256, 1024)
        video_features = self.compress_spatial_features(video_features, 8)  # (Nv, 64, 1024)
        video_features = self.get_model().mm_projector(video_features)  # (Nv, 64, 3584)
        video_features = video_features.flatten(0, 1).unsqueeze(0)  # (1, Nv*64, 3584)
        return video_features

    def _encode_video_chunk(self, video_chunk):  # (Nv, H, W, 3)
        pixel_values_videos = self.processor.preprocess(video_chunk, return_tensors="pt").pixel_values.to(self.device, self.dtype)  # (Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*64, D)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=16):  # video: (Nv, H, W, 3)
        super().encode_video(video, encode_chunk_size)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids[1:]  # [1:] remove <s>
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:  # retrieval mode
            layer_kv.set_retrieval()
        
        if retrieved_indices is None:  # Internal retrieval
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        else:  # External retrieval
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids[1:]  # [1:] remove <s>
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = self.lm_head(out['last_hidden_state'])
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = self.lm_head(out['last_hidden_state'])
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            
            # greedy
            # token = torch.argmax(last_token_logits).item()
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output

    # NOTE: currently only for calculating GFLOPs
    def streaming_vqa(self, video, inputs):
        cur_t = 0
        for prompt, next_t in inputs:
            if next_t > cur_t:
                video_clip = video[cur_t:next_t]
                self.encode_video(video_clip)
                cur_t = next_t
            self.question_answering(prompt)

    # # NOTE: currently only for calculating GFLOPs
    # def streaming_vqa_with_clip(self, clip_model, video, inputs):
    #     cur_t = 0
    #     for prompt, next_t in inputs:
    #         if next_t > cur_t:
    #             video_clip = video[cur_t:next_t]

    #             self.encode_video(video_clip)


   
    #             cur_t = next_t
    #         self.question_answering(prompt)


def load_model(model_path='/home/shangzhedi/shangzhedi/Flash-VStream-Base/checkpoints-finetune/base-7b-finetune-uniform-16/checkpoint-5900',
               n_init=None, n_local=4000, topk=16, chunk_size=1):
    n_frame_tokens = 64
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    """
    "<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <unk>\nQuestion: Where did I put the dog fur?\nOptions:\n(A) on the sofa\n(B) on the floor\n(C) on the table\n(D) in the trash\nAnswer with the option's letter from the given choices directly and only give the best option. ASSISTANT:"
    [    1,   319, 13563,  1546,   263, 12758,  1404,   322,   385, 23116,
         21082, 20255, 29889,   450, 20255,  4076,  8444, 29892, 13173, 29892,
           322,  1248,   568,  6089,   304,   278,  1404, 29915, 29879,  5155,
         29889,  3148,  1001, 29901, 29871,  -200, 29871,    13, 16492, 29901,
          6804,  1258,   306,  1925,   278, 11203,  3261, 29973,    13,  5856,
         29901,    13, 29898, 29909, 29897,   373,   278,   577,  5444,    13,
         29898, 29933, 29897,   373,   278, 11904,    13, 29898, 29907, 29897,
           373,   278,  1591,    13, 29898, 29928, 29897,   297,   278,   534,
          1161,    13, 22550,   411,   278,  2984, 29915, 29879,  5497,   515,
           278,  2183, 19995,  4153,   322,   871,  2367,   278,  1900,  2984,
         29889,   319,  1799,  9047, 13566, 29901]
    """
    init_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
    init_prompt_ids = tokenizer(init_prompt).input_ids
    inf_llm_config = {
        'n_init': len(init_prompt_ids) if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = FlashVStream_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    processor = vision_tower.image_processor
    processor.tokenizer = tokenizer
    model.processor = processor

    model = patch_hf(model, **inf_llm_config)
    model.language_model = model.model
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor


if __name__ == '__main__':
    import numpy as np
    from calflops import calculate_flops

    model, processor = load_model(n_local=2048, topk=16)

    FPS = 0.5
    video = np.load("data/VStream-QA/ego4d_videos/9198b9a4-8d8f-4ba6-9924-4c86982d890a.npy")
    num_frames = len(video)
    frame_idx = np.linspace(0, num_frames-1, int(num_frames*FPS), dtype=int).tolist()  # 0.5 FPS
    video = video[frame_idx]
    video_tensor = torch.from_numpy(video)  # (1800, H, W, 3)
    print(video_tensor.shape)

    model.clear_cache()
    model.encode_init_prompt()

    question = "What task is being performed with vegetables?"
    prompt = model.get_prompt(question)
    inputs = [({'question': question, 'prompt': prompt}, (i+1)*9) for i in range(200)]  # 9, 18, ..., 1800

    flops, macs, params = calculate_flops(
        model, 
        forward_mode="streaming_vqa", 
        kwargs={'video': video_tensor, 'inputs': inputs}
    )
    print(f'FLOPs: {flops}, MACs: {macs}, Params: {params}')
