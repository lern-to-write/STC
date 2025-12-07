

import math
import torch.nn.functional as F
import torch
# def dpc_knn_frame_wise(image_k, k=15, token_per_frame=196, use_global=False):
#     """
#     使用DPC-KNN算法计算每个token的分数
    
#     Args:
#         image_k: [batch, num_heads, token_number, dim_channel]
#         k: KNN中的邻居数量
#         token_per_frame: 每帧的token数量
#         use_global: 是否在整个token序列上使用DPC-KNN (True) 或仅在帧内使用 (False)
    
#     Returns:
#         scores: [batch, num_heads, token_number] 每个token的分数
#     """
#     batch_size, num_heads, token_number, dim_channel = image_k.shape
#     frame_number = token_number // token_per_frame
    
#     if use_global:
#         # 全局DPC-KNN: 在整个token序列上计算
#         scores = torch.zeros(batch_size, num_heads, token_number, device=image_k.device)
        
#         for b in range(batch_size):
#             for h in range(num_heads):
#                 # 获取当前batch和head的所有token
#                 tokens = image_k[b, h]  # [token_number, dim_channel]
                
#                 # 计算DPC-KNN分数
#                 token_scores = compute_dpc_knn_scores(tokens, k=k)
#                 scores[b, h] = token_scores
                
#     else:
#         # 帧内DPC-KNN: 在每帧内分别计算
#         scores = torch.zeros(batch_size, num_heads, token_number, device=image_k.device)
        
#         for b in range(batch_size):
#             for h in range(num_heads):
#                 for f in range(frame_number):
#                     # 获取当前帧的token
#                     start_idx = f * token_per_frame
#                     end_idx = (f + 1) * token_per_frame
#                     frame_tokens = image_k[b, h, start_idx:end_idx]  # [token_per_frame, dim_channel]
                    
#                     # 计算DPC-KNN分数
#                     frame_scores = compute_dpc_knn_scores(frame_tokens, k=k)
#                     scores[b, h, start_idx:end_idx] = frame_scores
    
#     return scores

# def compute_dpc_knn_scores(tokens, k=15):
#     """
#     使用DPC-KNN算法计算token分数
    
#     Args:
#         tokens: [n_tokens, dim] token特征
#         k: KNN邻居数量
    
#     Returns:
#         scores: [n_tokens] 每个token的分数
#     """
#     n_tokens, dim = tokens.shape
    
#     # 转换为numpy进行计算
#     tokens_np = tokens.detach().cpu().numpy()
    
#     # 1. 计算KNN距离
#     nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(tokens_np)
#     distances, indices = nbrs.kneighbors(tokens_np)
    
#     # 2. 计算局部密度 (使用高斯核)
#     dc = np.mean(distances[:, 1:])  # 平均距离作为截断距离
#     rho = np.zeros(n_tokens)
    
#     for i in range(n_tokens):
#         # 使用高斯核计算局部密度
#         rho[i] = np.sum(np.exp(-((distances[i, 1:] / dc) ** 2)))
    
#     # 3. 计算到更高密度点的距离delta
#     delta = np.zeros(n_tokens)
#     # 找到密度最大的点
#     max_rho_idx = np.argmax(rho)
#     delta[max_rho_idx] = np.max(distances[max_rho_idx, 1:])  # 最大密度点的距离设为最大k距离
    
#     # 对于其他点
#     for i in range(n_tokens):
#         if i != max_rho_idx:
#             # 找到密度比当前点高的点中距离最近的
#             higher_rho_indices = np.where(rho > rho[i])[0]
#             if len(higher_rho_indices) > 0:
#                 min_dist = np.min(distances[i, 1:][np.isin(indices[i, 1:], higher_rho_indices)])
#                 delta[i] = min_dist
#             else:
#                 # 如果没有更高密度的点，则设为到最远点的距离
#                 delta[i] = np.max(distances[i, 1:])
    
#     # 4. 计算综合分数 gamma = rho * delta
#     gamma = rho * delta
    
#     # 转换回tensor
#     scores = torch.tensor(gamma, dtype=tokens.dtype, device=tokens.device)
    
#     return scores
def compute_image_attention_scores(
    local_h_q: torch.Tensor, 
    image_k: torch.Tensor
):

    len_q = local_h_q.size(-2)
    len_k = image_k.size(-2)
    
    # 处理多头注意力头数不匹配的情况
    num_heads_q = local_h_q.size(1)
    num_heads_k = image_k.size(1)
    
    k_expanded = image_k
    if num_heads_q != num_heads_k:
        shape = list(image_k.shape)
        num_group = num_heads_q // num_heads_k
        k_expanded = image_k[:, :, None, :, :].expand(
            shape[0], shape[1], num_group, shape[2], shape[3]
        )
        k_expanded = k_expanded.reshape(shape[0], num_heads_q, shape[2], shape[3])
    
    # 计算注意力分数
    logits = torch.matmul(local_h_q, k_expanded.transpose(-1, -2))
    
    # 缩放
    logits.mul_(1 / math.sqrt(local_h_q.size(-1)))
    
        # 先应用softmax得到注意力权重
    attention_weights = F.softmax(logits, dim=-1)
    # 然后计算每个image token的重要性
    image_score = attention_weights[0].mean(dim=0).mean(dim=0)  # [token_number_k]
    
    return image_score






def get_kept_token_indices(token_scores, keep_ratios):
    """
    根据每帧的保留率获取最终保留的token索引
    
    Args:
        token_scores: Tensor of shape [token_number], 每个token的分数
        keep_ratios: list of length frame_number, 每帧的保留率
    
    Returns:
        kept_indices: Tensor of kept token indices in the original token_tensor
    """
    token_number = len(token_scores)
    frame_number = len(keep_ratios)
    token_per_frame = 196
    
    # 验证输入
    assert token_number % frame_number == 0, f"token_number ({token_number}) should be divisible by frame_number ({frame_number})"
    
    kept_indices = []
    
    # 对每一帧处理
    for frame_idx in range(frame_number):
        # 计算当前帧的token范围
        start_idx = frame_idx * token_per_frame
        end_idx = (frame_idx + 1) * token_per_frame
        
        # 获取当前帧的token分数
        frame_scores = token_scores[start_idx:end_idx]
        
        # 计算需要保留的token数量
        keep_count = max(1, int(token_per_frame * keep_ratios[frame_idx]))
        
        # 获取分数最高的token索引（在当前帧内）
        _, top_indices = torch.topk(frame_scores, keep_count)
        
        # 转换为全局索引
        global_indices = top_indices + start_idx
        
        kept_indices.append(global_indices)
    
    # 合并所有帧的保留索引
    if kept_indices:
        kept_indices = torch.cat(kept_indices, dim=0)
    else:
        kept_indices = torch.tensor([], dtype=torch.long)
    
    return kept_indices