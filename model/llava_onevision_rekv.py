import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.config import get_config
from model.prune import *
from model.custom_siglip import *
import torch.distributed as dist


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)
        ##########################################################
        register_cache_by_key_Siglip(self.vision_tower)
        update_token_ratio=0.25
        chunk_idx=0
        cache1 = STC_CACHE.new_instance(chunk_idx,update_token_ratio)
        
        self.stc_pruner = STC_Pruner()
        self.stc_pruner.past_memory_mean_token =[]
        self.past_memory_mean_token = self.stc_pruner.past_memory_mean_token
        ###############################################
        
    def get_vision_tower(self):
        return self.vision_tower

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

        
        
    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]
        frames=selected_video_feature.shape[0]
        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        
        reshaped_video_tensor=video_features.reshape(-1, video_features.size(-1))  
        ###############################################
        token_per_frame = get_config().model.token_per_frame
        video_features = self.stc_pruner.compress(reshaped_video_tensor)
        rank = dist.get_rank()
        if rank == 0:
        
            logger.info(f"LLM | Vocab size: 196, Tokens to retained: {token_per_frame}")
        
        #############################################
        frames=video_features.shape[0]//token_per_frame
        
        video_features = video_features.reshape(batch_size, frames * token_per_frame, -1)
        return video_features


    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False
        if isinstance(input_text, str):
            question_text = input_text
            prompt_text = input_text
        else:
            question_text = input_text['question']
            prompt_text = input_text['prompt']
            
        input_ids = self.processor.tokenizer(question_text).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        
        for layer_kv in self.kv_cache:  # activate retrieval mode
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
                input_ids = self.processor.tokenizer(prompt_text).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]
            if i == 0 and token in stop_token_ids:   # 第一步就算 eos 也继续
                token = tokens[1] if len(tokens) > 1 else 1 
            


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


def load_model(model_path='llava-hf/llava-onevision-qwen2-7b-ov-hf',device=None,
                       n_init=None, n_local=15000, topk=64, chunk_size=1):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    token_per_frame = get_config().model.token_per_frame
    n_frame_tokens =int(token_per_frame)
    
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map={"": device}, # <--- 核心修改：禁止 "auto"，强制指定
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )

    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    ######################################################################
    rank = dist.get_rank()
    if rank == 0:
        for k, v in inf_llm_config.items():
            logger.info(f'{k}: {v}')
        logger.info(f'n_frame_tokens: {n_frame_tokens}')
    ######################################################################
    model.eval()

    return model, processor
