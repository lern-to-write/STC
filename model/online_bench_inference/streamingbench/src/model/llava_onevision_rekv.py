import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.vidcom2_for_rekv import *

import time
import transformers.models.siglip.modeling_siglip as modeling_siglip
from model.mmg_vid import *
# 用你自己的类替换掉原来的类
from model.config import get_config
import os
from model.custom_siglip import *

class Enhanced_LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    """
    增强版本的LLaVA OneVision ReKV模型，支持奇偶数chunk缓存
    """
    
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)
        ##########################################################
        register_cache_Siglip(self.vision_tower)
        update_token_ratio=0.25
        chunk_idx=0
        acc_time=0
        max_mem=0
        cache1 = STC_CACHE.new_instance(chunk_idx,update_token_ratio,acc_time,max_mem)
        ###############################################
        
    def get_vision_tower(self):
        """获取vision tower，供增强抽象基类使用"""
        return self.vision_tower

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt
    
    def dynamic_processor(self,reshaped_video_tensor,retention_ratio):
        prune_strategy = os.environ.get("PRUNE_STRATEGY", "full_tokens")
        # --------------------------------------------------------------------
        
        print(f"Using prune_strategy (from env): {prune_strategy}")
        if prune_strategy == "vidcom2_orignal":
            return vidcom2_compression(self,reshaped_video_tensor,base_scale=retention_ratio,token_score_mode="negtive_video_mean_and_global_mean")
        elif prune_strategy == "vidcom2_two_score":
            return vidcom2_compression(self,reshaped_video_tensor,base_scale=retention_ratio,token_score_mode="negtive_frame_mean_and_update_memory_mean")
        elif prune_strategy == "vidcom2_only_memory":
            return vidcom2_compression(self,reshaped_video_tensor,base_scale=retention_ratio,token_score_mode="negtive_update_memory_mean")
        elif prune_strategy == "dart":
            return vidcom2_compression(self,reshaped_video_tensor,base_scale=retention_ratio,token_score_mode="negtive_frame_mean")
        elif prune_strategy == "mmg_sparse":
            return mmg_sparse(self,reshaped_video_tensor,retention_ratio)
        elif prune_strategy == "dycoke":
            return dycoke_easy(reshaped_video_tensor)
        elif prune_strategy == "full_tokens":
            return  full_tokens(reshaped_video_tensor)
        else:
            return  full_tokens(reshaped_video_tensor)
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
        token_per_frame = get_config().model.token_per_frame
        
        retention_ratio = float(token_per_frame / 196)
        video_features = self.dynamic_processor(reshaped_video_tensor, retention_ratio)
        video_features = video_features.reshape(batch_size, frames * token_per_frame, -1)

        return video_features


    @torch.inference_mode()
    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        # Handle both string input and dict input
        if isinstance(input_text, str):
            question_text = input_text
            prompt_text = input_text
        else:
            # Original dict-based approach
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


def load_enhanced_model(model_path='/mnt/data0/public/back/huggingface/hub/models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068',
                       n_init=None, n_local=15000, topk=64, chunk_size=1):
    """
    加载增强版本的LLaVA OneVision ReKV模型，支持奇偶数chunk缓存
    """
    device = 'cuda'
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
    model = Enhanced_LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
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
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor


# 向后兼容
def load_model(model_path='llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
               n_init=None, n_local=None, topk=64, chunk_size=1):
    return load_enhanced_model(model_path, n_init, n_local, topk, chunk_size)

# 向后兼容的类名别名
LlavaOneVision_ReKV = Enhanced_LlavaOneVision_ReKV