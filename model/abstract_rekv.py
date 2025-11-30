import torch
from logzero import logger

import os
from model.config import *
from model.cache import *
class Abstract_ReKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size
        self.ram_usage=0
        self.total_cuda_time=0
        self.max_mem=0
        self.total_llm_pre_time=0

    def clear_cache(self):
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        pass

    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        

 
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)

        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'
        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        

        self.kv_cache = output.past_key_values
    


    @torch.inference_mode()
    
    def encode_video(self, video):  # video: (Nv, H, W, 3)
        
        encode_chunk_size=get_config().model.encode_chunk_size
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            # 选择相似度最低的token作为需要更新的token
            update_token_ratio=get_config().cache.update_token_ratio
            ###############################################
            cache1 = STC_CACHE.new_instance(chunk_idx,update_token_ratio)
            cache_strategy = get_config().cache.strategy

            if cache_strategy=="none":
                cache1 = STC_CACHE.new_instance(0,update_token_ratio)
                
            #################################################
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')
            



        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)
        
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory


