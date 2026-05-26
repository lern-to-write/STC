import copy
import torch
from typing import Optional

from .kv_cache_manager import ContextManager
from .dot_production_attention import get_multi_stage_dot_production_attention
# from ..video_qa.run_eval import TimeRecord


# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import DBSCAN

import math
import torch.nn.functional as F
import os



def random_keep_half_tokens_v1(k):
    """
    方法1: 使用torch.randperm随机选择一半token
    """
    batch_size, num_heads, token_number, channel = k.shape
    
    # 计算需要保留的token数量
    keep_num = (token_number-13) // 2
    
    # 生成随机索引
    indices = torch.randperm(token_number)[:keep_num]
    
    # 按索引选择token
    k_kept = k[:, :, indices, :]
    
    return k_kept, indices


def _from_group_kv(key):
    # tensor: (batch_size, n_head_kv, length, dim_head)
    query_head_number=28
    batch, k_head_number, length, dim_head = key.shape
    num_group = query_head_number // k_head_number
    grouped_key = key.view(
        (batch, k_head_number, 1, length, dim_head)
    )  # (batch_size, n_head_kv, 1, length, dim_head)
    grouped_key = grouped_key.expand(
        (batch, k_head_number, num_group, length, dim_head)
    ).reshape(
        (batch,length, dim_head*query_head_number)
    )  # (batch_size, n_head, length, dim_head)
    return grouped_key



def filter_tokens_simple(video_tensor, memory_mean_token, token_per_frame):
    """简化版本"""
    batch_size, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame
    
    # 计算余弦相似度
    video_flat = video_tensor.view(-1, channel)  # [token_number, channel]
    memory_flat = memory_mean_token.expand(token_number, -1)  # [token_number, channel]
    cosine_sim = F.cosine_similarity(video_flat, memory_flat, dim=1)  # [token_number]
    
    # 分帧处理并保留每帧一半相似度小的token
    kept_indices = []
    
    for i in range(frame_number):
        start_idx = i * token_per_frame
        end_idx = start_idx + token_per_frame
        frame_sim = cosine_sim[start_idx:end_idx]
        
        # 保留相似度最小的一半
        num_keep = token_per_frame // 2
        _, top_indices = torch.topk(frame_sim, num_keep, largest=False)  # 最小的num_keep个
        kept_indices.append(top_indices + start_idx)
    
    # 合并索引并选择token
    final_indices = torch.cat(kept_indices, dim=0)
    # filtered_tokens = video_tensor[:, final_indices, :]
    
    return final_indices


def filter_tokens_random(video_tensor, memory_mean_token, token_per_frame):
    """随机保留每帧一半的token"""
    batch_size, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame
    
    kept_indices = []
    
    for i in range(frame_number):
        start_idx = i * token_per_frame
        end_idx = start_idx + token_per_frame
        
        # 随机选择一半的token
        num_keep = token_per_frame // 2
        frame_indices = torch.randperm(token_per_frame)[:num_keep]
        kept_indices.append(frame_indices + start_idx)
    
    final_indices = torch.cat(kept_indices, dim=0)
    return final_indices

def filter_tokens_magnitude(video_tensor, memory_mean_token, token_per_frame):
    """基于token的L2范数大小进行过滤，保留范数较小的一半"""
    batch_size, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame
    
    # 计算每个token的L2范数
    token_magnitude = torch.norm(video_tensor.squeeze(0), dim=1)  # [token_number]
    
    kept_indices = []
    
    for i in range(frame_number):
        start_idx = i * token_per_frame
        end_idx = start_idx + token_per_frame
        frame_magnitude = token_magnitude[start_idx:end_idx]
        
        # 保留范数较小的一半
        num_keep = token_per_frame // 2
        _, top_indices = torch.topk(frame_magnitude, num_keep, largest=False)
        kept_indices.append(top_indices + start_idx)
    
    final_indices = torch.cat(kept_indices, dim=0)
    return final_indices

def filter_tokens_euclidean_distance(video_tensor, memory_mean_token, token_per_frame):
    """基于欧几里得距离进行过滤，保留距离memory mean token较近的一半"""
    batch_size, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame
    
    # 计算欧几里得距离
    video_flat = video_tensor.view(-1, channel)  # [token_number, channel]
    memory_flat = memory_mean_token.expand(token_number, -1)  # [token_number, channel]
    euclidean_dist = torch.norm(video_flat - memory_flat, dim=1)  # [token_number]
    
    kept_indices = []
    
    for i in range(frame_number):
        start_idx = i * token_per_frame
        end_idx = start_idx + token_per_frame
        frame_dist = euclidean_dist[start_idx:end_idx]
        
        # 保留距离较小的一半（距离越小越相似）
        num_keep = token_per_frame // 2
        _, top_indices = torch.topk(frame_dist, num_keep, largest=False)
        kept_indices.append(top_indices + start_idx)
    
    final_indices = torch.cat(kept_indices, dim=0)
    return final_indices

def filter_tokens_inverse_cosine(video_tensor, memory_mean_token, token_per_frame):
    """基于余弦相似度的倒数进行过滤（相似度越大，倒数越小）"""
    batch_size, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame
    
    # 计算余弦相似度
    video_flat = video_tensor.view(-1, channel)
    memory_flat = memory_mean_token.expand(token_number, -1)
    cosine_sim = F.cosine_similarity(video_flat, memory_flat, dim=1)
    
    # 添加小的epsilon避免除零
    epsilon = 1e-8
    inverse_cosine = 1.0 / (torch.abs(cosine_sim) + epsilon)
    
    kept_indices = []
    
    for i in range(frame_number):
        start_idx = i * token_per_frame
        end_idx = start_idx + token_per_frame
        frame_inverse = inverse_cosine[start_idx:end_idx]
        
        # 保留倒数较小的一半（即原相似度绝对值较大的）
        num_keep = token_per_frame // 2
        _, top_indices = torch.topk(frame_inverse, num_keep, largest=False)
        kept_indices.append(top_indices + start_idx)
    
    final_indices = torch.cat(kept_indices, dim=0)
    return final_indices

def filter_tokens_percentile(video_tensor, memory_mean_token, token_per_frame):
    """基于余弦相似度的百分位数进行过滤"""
    batch_size, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame
    
    # 计算余弦相似度
    video_flat = video_tensor.view(-1, channel)
    memory_flat = memory_mean_token.expand(token_number, -1)
    cosine_sim = F.cosine_similarity(video_flat, memory_flat, dim=1)
    
    kept_indices = []
    
    for i in range(frame_number):
        start_idx = i * token_per_frame
        end_idx = start_idx + token_per_frame
        frame_sim = cosine_sim[start_idx:end_idx]
        
        # 计算25%分位数，保留小于该分位数的token
        threshold = torch.quantile(frame_sim, 0.25)
        below_threshold = frame_sim < threshold
        selected_indices = torch.nonzero(below_threshold, as_tuple=True)[0]
        
        # 如果选中的token数量超过一半，只取前一半
        num_keep = token_per_frame // 2
        if len(selected_indices) > num_keep:
            # 按相似度排序，取最小的num_keep个
            selected_sim = frame_sim[selected_indices]
            _, sorted_idx = torch.sort(selected_sim)
            selected_indices = selected_indices[sorted_idx[:num_keep]]
        
        kept_indices.append(selected_indices + start_idx)
    
    final_indices = torch.cat(kept_indices, dim=0)
    return final_indices

def filter_tokens_top_half(video_tensor, memory_mean_token, token_per_frame):
    """保留每帧相似度最大的一半token（与原方法相反）"""
    batch_size, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame
    
    # 计算余弦相似度
    video_flat = video_tensor.view(-1, channel)
    memory_flat = memory_mean_token.expand(token_number, -1)
    cosine_sim = F.cosine_similarity(video_flat, memory_flat, dim=1)
    
    kept_indices = []
    
    for i in range(frame_number):
        start_idx = i * token_per_frame
        end_idx = start_idx + token_per_frame
        frame_sim = cosine_sim[start_idx:end_idx]
        
        # 保留相似度最大的一半
        num_keep = token_per_frame // 2
        _, top_indices = torch.topk(frame_sim, num_keep, largest=True)  # 改为True
        kept_indices.append(top_indices + start_idx)
    
    final_indices = torch.cat(kept_indices, dim=0)
    return final_indices

def dynamic_processor(video_tensor, memory_mean_token):
    retrieved_kv_compression_strategy = os.getenv("retrieved_KV_COMPRESSION_STRATEGY", "full_kv")  # 默认值
    
    # 直接使用if-else避免预执行
    token_per_frame = int(os.getenv("TOKEN_PER_FRAME", 196))
    
    if retrieved_kv_compression_strategy == "filter_tokens_simple":
        return filter_tokens_simple(video_tensor, memory_mean_token, token_per_frame)
    elif retrieved_kv_compression_strategy == "filter_tokens_random":
        return filter_tokens_random(video_tensor, memory_mean_token, token_per_frame)
    elif retrieved_kv_compression_strategy == "filter_tokens_magnitude":
        return filter_tokens_magnitude(video_tensor, memory_mean_token, token_per_frame)
    elif retrieved_kv_compression_strategy == "filter_tokens_euclidean_distance":
        return filter_tokens_euclidean_distance(video_tensor, memory_mean_token, token_per_frame)
    elif retrieved_kv_compression_strategy == "filter_tokens_inverse_cosine":
        return filter_tokens_inverse_cosine(video_tensor, memory_mean_token, token_per_frame)
    elif retrieved_kv_compression_strategy == "filter_tokens_percentile":
        return filter_tokens_percentile(video_tensor, memory_mean_token, token_per_frame)
    elif retrieved_kv_compression_strategy == "filter_tokens_top_half":
        return filter_tokens_top_half(video_tensor, memory_mean_token, token_per_frame)
    
    else:
        # 默认策略
        raise ValueError(f"Invalid processor_type: {retrieved_kv_compression_strategy}")
def rekv_attention_forward(
    n_local, n_init, topk, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    async_global_stream=True,
    pin_memory=False,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
    ):

        """ 1. Project QKV """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                n_init, n_local, 
                block_size, max_cached_block, topk, chunk_size, exc_block_size,
                fattn,
                async_global_stream,
                pin_memory,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        # NOTE: Question-answering, fall back to sliding-window attention (infinite_lm)
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
            if type(past_key_value) is ContextManager:  # retrieval
                
                ########################################################################################################
                # torch.cuda.reset_peak_memory_stats()
                # gen_start_event = torch.cuda.Event(enable_timing=True)
                # gen_end_event = torch.cuda.Event(enable_timing=True)
                # torch.cuda.synchronize()
                # gen_start_event.record()
                ##################################################################################
                if past_key_value.retrieved_block_indices is None:  # retrieve based on global_q (question's query)
                    past_k, past_v = past_key_value.get_retrieved_kv(global_q)

                else:  # retrieve based on pre-computed retrieved_block_indices
                    past_k, past_v = past_key_value.get_retrieved_kv()
                # past_k, indices=random_keep_half_tokens_v1(past_k)    
                # past_v = past_v[:, :, indices, :]
                
                prune_retrieved_kv = os.getenv("PRUNE_RETIREVED_KV", "no") 
                if prune_retrieved_kv == "yes":
                    memory_tokens = past_key_value.origin_block_k[0].data[: past_key_value.length].float()# [token_number, channel][batch]
                    memory_mean_token = memory_tokens.mean(dim=0, keepdim=True)  # (1, channel)
                    grouped_past_k=_from_group_kv(past_k)# (batch,token_number,channel)
                    image_k=grouped_past_k[:,13:,:]  # 去掉前13个文本token
                    final_indices= dynamic_processor(image_k, memory_mean_token)
                    global_indices=final_indices+13
                    past_k = past_k[:,:, global_indices, :]
                    past_v = past_v[:,:, global_indices, :]
               # past_k.shape  ([1, 4, 12557, 128])   12544
                
                
                
                
                #############################################################
                # gen_end_event.record()
                # torch.cuda.synchronize()
                # gen_time = gen_start_event.elapsed_time(gen_end_event) / 1000.0  # 秒
                # gen_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

                # TimeRecord.total_cuda_time += gen_time
                # TimeRecord.max_mem=max(gen_max_mem,past_key_value.max_mem)
                # print("total_time",TimeRecord.total_cuda_time,"max_mem",TimeRecord.max_mem)
                ###########################################################
                updata_kv_cache = False  # We do not update KV cache with the input KV (h_k, h_v) because we only use it for retrieval
            else:  # sliding-window attention
                past_k = past_key_value[0]
                past_v = past_key_value[1]
                updata_kv_cache = True

            """ 2. Update KV w/ past KV cache """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            

            len_k += past_k.shape[2]

            """ 3. Update KV cache """
            if updata_kv_cache:
                if len_k <= n_local + n_init:
                    h_k_cache = h_k
                    h_v_cache = h_v
                else:
                    h_k_cache = torch.cat([h_k[:,:, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                    h_v_cache = torch.cat([h_v[:,:, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                current_key_value = (h_k_cache, h_v_cache)
            else:
                current_key_value = (past_k, past_v)

            """ 4. Get local QKV and apply RoPE to local QK """
            h_q_, h_k_, h_v_ = h_q, h_k, h_v
            if len_q + n_local < h_k_.size(-2):
                h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local:, :]
                h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local:, :]

            local_h_q, local_h_k = position_bias(h_q_, h_k_)
            local_h_v = h_v_

            """ 5. Get init QKV and apply RoPE to init Q (Infinite-LM assigns the same position_ids to initial tokens) """
            if len_k > n_local:
                init_h_q = position_bias.apply_rotary_pos_emb_one_angle(
                    h_q, n_local
                )
                init_h_k = h_k
                init_h_v = h_v
                init_h_k = init_h_k[:, :, :n_init, :].contiguous()
                init_h_v = init_h_v[:, :, :n_init, :].contiguous()

            else:
                init_h_q = h_q
                init_h_k = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_k.device,
                    dtype=h_k.dtype
                )
                init_h_v = torch.empty(
                    (batch_size, num_heads_kv, 0, dim_head),
                    device=h_v.device,
                    dtype=h_v.dtype
                )

            """ 6. Sliding Window Attention """

            
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
            attn.append(init_h_q, init_h_k, init_h_v, end=True, sliding_window=(len_k - len_q, n_local), complement_sliding_window=True)
            score, _ = attn.get_result()

            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
            score = score.reshape(batch_size, len_q, num_heads * dim_head) # (batch, len_q, num_heads * dim_head)
            score = attention_out(score)

            return score, current_key_value

        # NOTE: Encode video, managed by the KVCacheManager
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            o = attention_out(o)

            return o, past_key_value

    return forward