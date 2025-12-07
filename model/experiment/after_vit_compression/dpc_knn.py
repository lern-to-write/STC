import torch
import torch.nn.functional as F


def dpc_knn_token_reducer_strict(input_tensor: torch.Tensor, k: int = 15, center_ratio: float = 0.15, output_ratio: float = 0.5) -> torch.Tensor:
    """
    使用 DPC-KNN 算法减少每帧的 token 数量。

    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 [frame_number, token_per_frame, channel]。
        k (int): 用于计算密度的 KNN 参数。
        center_ratio (float): 每帧聚类中心占原始 token 数的比例 (例如 0.15 表示 15%)。
        output_ratio (float): 每帧最终输出 token 数占原始 token 数的比例 (例如 0.5 表示 50%)。

    Returns:
        torch.Tensor: 输出张量，形状为 [frame_number, int(token_per_frame * output_ratio), channel]。
    """
    frame_number, token_per_frame, channel = input_tensor.shape
    num_centers = max(1, int(token_per_frame * center_ratio)) # 至少保留一个中心
    num_output_tokens = max(1, int(token_per_frame * output_ratio))
    
    if num_output_tokens < num_centers:
        raise ValueError("output_ratio must be greater than or equal to center_ratio.")

    # 初始化输出张量
    output_tensor = torch.zeros((frame_number, num_output_tokens, channel), device=input_tensor.device, dtype=input_tensor.dtype)

    for f in range(frame_number):
        frame_tokens = input_tensor[f] # Shape: [token_per_frame, channel]
        N = token_per_frame

        # Step 1: 计算距离矩阵 (使用平方欧氏距离)
        dist_matrix = torch.cdist(frame_tokens, frame_tokens, p=2.0) ** 2 # Shape: [N, N]

        # Step 2: 计算密度 rho (使用 KNN 距离)
        _, knn_indices = torch.topk(dist_matrix, k=k+1, largest=False, sorted=True) # Shape: [N, k+1]
        knn_indices = knn_indices[:, 1:] # 排除自己，Shape: [N, k]
        knn_distances = torch.gather(dist_matrix, 1, knn_indices) # Shape: [N, k]
        rho = -torch.sum(knn_distances, dim=1) # Shape: [N] # 密度

        # Step 3: 计算距离 delta
        delta = torch.zeros(N, device=input_tensor.device, dtype=input_tensor.dtype) # Shape: [N]
        for i in range(N):
            mask_higher_rho = rho > rho[i]
            if mask_higher_rho.any():
                min_dist = torch.min(dist_matrix[i][mask_higher_rho])
                delta[i] = min_dist
            else:
                # 密度最高的点，计算到所有点的平均距离
                delta[i] = torch.mean(dist_matrix[i]) if N > 1 else 0.0

        # Step 4: 选择聚类中心 (选择 rho * delta 最大的 center_ratio% 个点)
        gamma = rho * delta # Shape: [N]
        _, center_indices_tmp = torch.topk(gamma, k=num_centers, largest=True, sorted=True) # Shape: [num_centers]
        center_mask = torch.zeros(N, dtype=torch.bool, device=input_tensor.device) # Shape: [N]
        center_mask[center_indices_tmp] = True
        
        # Step 5: 分配其他 token 到最近的聚类中心
        # 计算所有点到中心的距离
        dist_to_centers = dist_matrix[:, center_indices_tmp] # Shape: [N, num_centers]
        # 找到每个点最近的中心索引 (在 center_indices_tmp 中的索引)
        _, nearest_center_local_indices = torch.min(dist_to_centers, dim=1) # Shape: [N]
        # 转换为原始 token 的中心索引
        nearest_center_global_indices = center_indices_tmp[nearest_center_local_indices] # Shape: [N]

        # Step 6: 选择最终输出的 token
        # 6a. 中心点是优先保留的
        selected_indices = center_indices_tmp.tolist() # List of indices
        
        # 6b. 如果还需要更多 token，则从非中心点中选择
        num_additional_needed = num_output_tokens - num_centers
        if num_additional_needed > 0:
            non_center_mask = ~center_mask # Shape: [N]
            if non_center_mask.any():
                non_center_indices = torch.where(non_center_mask)[0] # Shape: [num_non_centers]
                
                # 计算非中心点到其分配中心的距离，作为重要性的一个衡量（距离越近可能越重要，越优先保留）
                dist_to_assigned_center = dist_matrix[non_center_indices, nearest_center_global_indices[non_center_indices]] # Shape: [num_non_centers]
                
                # 根据距离排序，选择距离最近的 num_additional_needed 个点
                 # argsort 返回升序索引，取前 num_additional_needed 个
                _, sorted_non_center_local_indices = torch.sort(dist_to_assigned_center, descending=False) 
                selected_additional_local_indices = sorted_non_center_local_indices[:num_additional_needed]
                selected_additional_global_indices = non_center_indices[selected_additional_local_indices]
                
                selected_indices.extend(selected_additional_global_indices.tolist())
        
        # 确保数量正确（理论上应该正确，但以防万一）
        selected_indices = selected_indices[:num_output_tokens] 
        
        # 生成输出
        output_tensor[f] = frame_tokens[selected_indices]

    return output_tensor