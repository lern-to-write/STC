def frame_wise_token_reduction(video_tensor, reduction_ratio=0.5, dc=2.0, coef_pruned=1.5, print_debug=False):
    """
    基于DBDPC算法对每帧进行token减少
    
    Args:
        video_tensor (torch.Tensor): 输入张量，形状为 [frame_number, token_per_frame, channel]
        reduction_ratio (float): 减少比例，0.5表示保留一半token
        dc (float): DBDPC算法的截断距离参数
        coef_pruned (float): 聚类距离的系数
        print_debug (bool): 是否打印调试信息
    
    Returns:
        reduced_tensor (torch.Tensor): 减少后的张量，形状为 [frame_number, reduced_tokens, channel]
        masks (torch.Tensor): 每帧的mask，形状为 [frame_number, token_per_frame]，表示哪些token被保留
        weights (torch.Tensor): 每帧的权重，形状为 [frame_number, token_per_frame]，用于比例注意力
    """
    
    frame_number, token_per_frame, channel = video_tensor.shape
    device = video_tensor.device
    
    # 计算每帧保留的token数量
    target_tokens_per_frame = int(token_per_frame * reduction_ratio)
    
    # 初始化输出张量
    reduced_tensors = []
    all_masks = []
    all_weights = []
    
    if print_debug:
        print(f"Processing {frame_number} frames, reducing from {token_per_frame} to ~{target_tokens_per_frame} tokens per frame")
    
    # 对每一帧单独处理
    for frame_idx in range(frame_number):
        if print_debug and frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{frame_number}")
            
        # 获取当前帧的tokens
        current_frame = video_tensor[frame_idx]  # [token_per_frame, channel]
        
        # 使用DBDPC算法进行聚类
        frame_reduced, frame_mask, frame_weights = _apply_dbdpc_to_frame(
            current_frame, dc, coef_pruned, target_tokens_per_frame, print_debug and frame_idx == 0
        )
        
        reduced_tensors.append(frame_reduced)
        all_masks.append(frame_mask)
        all_weights.append(frame_weights)
    
    # 堆叠结果
    # 由于每帧保留的token数量可能不完全相同，我们需要填充到统一长度
    max_kept_tokens = max(tensor.shape[0] for tensor in reduced_tensors)
    
    # 创建最终的输出张量
    final_reduced = torch.zeros(frame_number, max_kept_tokens, channel, device=device, dtype=video_tensor.dtype)
    final_masks = torch.zeros(frame_number, token_per_frame, dtype=torch.bool, device=device)
    final_weights = torch.zeros(frame_number, token_per_frame, dtype=torch.float32, device=device)
    
    for frame_idx in range(frame_number):
        kept_tokens = reduced_tensors[frame_idx].shape[0]
        final_reduced[frame_idx, :kept_tokens] = reduced_tensors[frame_idx]
        final_masks[frame_idx] = all_masks[frame_idx]
        final_weights[frame_idx] = all_weights[frame_idx]
    
    if print_debug:
        avg_kept = sum(tensor.shape[0] for tensor in reduced_tensors) / frame_number
        print(f"Average tokens kept per frame: {avg_kept:.1f}/{token_per_frame} ({avg_kept/token_per_frame*100:.1f}%)")
    
    return final_reduced, final_masks, final_weights

def _apply_dbdpc_to_frame(frame_tokens, dc, coef_pruned, target_tokens, print_debug=False):
    """
    对单帧应用DBDPC算法
    
    Args:
        frame_tokens (torch.Tensor): 单帧的tokens，形状为 [token_per_frame, channel]
        dc (float): 截断距离
        coef_pruned (float): 系数
        target_tokens (int): 目标保留的token数量
        print_debug (bool): 是否打印调试信息
    
    Returns:
        reduced_frame (torch.Tensor): 减少后的帧tokens
        mask (torch.Tensor): 保留的token mask
        weights (torch.Tensor): token权重
    """
    
    device = frame_tokens.device
    N, D = frame_tokens.shape
    

    
    # 标准化tokens用于距离计算
    normalized_tokens = torch.nn.functional.normalize(frame_tokens, p=2.0, dim=1)
    
    # 计算成对距离矩阵
    distance_matrix = _compute_cosine_distance_matrix(normalized_tokens)
    

    # 计算局部密度 rho
    rho = torch.exp(-(distance_matrix / dc) ** 2).sum(dim=1)
    
    # 为了数值稳定性，使用排名代替原始密度值
    sorted_indices = torch.argsort(rho, descending=True)
    ranks = torch.empty_like(sorted_indices, dtype=torch.float32, device=device)
    ranks[sorted_indices] = torch.arange(len(rho), dtype=torch.float32, device=device)
    rho = ranks

    
    # 找到聚类中心
    cluster_centers = _find_cluster_centers_iterative(
        distance_matrix, rho, dc, target_tokens, print_debug
    )
    
    if print_debug:
        print(f"Cluster centers found: {len(cluster_centers)}")

    
    # 将所有token分配到最近的聚类中心
    distances_to_centers = distance_matrix[:, cluster_centers]
    nearest_center_idx = torch.argmin(distances_to_centers, dim=1)
    labels = cluster_centers[nearest_center_idx]
    
    # 确保聚类中心自己的标签正确
    labels[cluster_centers] = cluster_centers
    

    
    # 合并聚类
    reduced_frame, mask, weights = _merge_clusters_for_frame(
        frame_tokens, labels, cluster_centers, print_debug
    )
    

    
    return reduced_frame, mask, weights

def _compute_cosine_distance_matrix(normalized_tokens):
    """计算余弦距离矩阵"""
    dot_product = torch.mm(normalized_tokens, normalized_tokens.t())
    distance_matrix = torch.clamp(1 - dot_product, min=0)
    distance_matrix.fill_diagonal_(0)
    return distance_matrix

def _find_cluster_centers_iterative(distance_matrix, rho, dc, target_tokens, print_debug=False):
    """迭代寻找聚类中心"""
    device = distance_matrix.device
    N = distance_matrix.shape[0]
    cluster_centers = []
    unassigned_mask = torch.ones(N, dtype=torch.bool, device=device)
    
    max_iterations = 10  # 防止无限循环
    iteration = 0
    
    while len(cluster_centers) < target_tokens and iteration < max_iterations:
        iteration += 1
        
        unassigned_indices = torch.where(unassigned_mask)[0]
        if len(unassigned_indices) == 0:
            break
            
        # 计算未分配点的delta值
        rho_unassigned = rho[unassigned_indices]
        dist_unassigned = distance_matrix[unassigned_indices][:, unassigned_indices]
        
        # 计算delta：到更高密度点的最小距离
        delta = _compute_delta(rho_unassigned, dist_unassigned)
        
        # 找到新的聚类中心（delta > dc的点）
        new_centers_mask = delta > dc
        new_centers_local = torch.where(new_centers_mask)[0]
        
        if len(new_centers_local) == 0:
            # 如果没有找到满足条件的中心，选择剩余的最高密度点
            remaining_needed = target_tokens - len(cluster_centers)
            if remaining_needed > 0:
                rho_values = rho_unassigned
                _, top_indices = torch.topk(rho_values, min(remaining_needed, len(rho_values)))
                new_centers_local = top_indices
        
        new_centers_global = unassigned_indices[new_centers_local]
        cluster_centers.extend(new_centers_global.tolist())
        
        # 更新未分配掩码：移除新中心和它们附近的点
        if len(new_centers_global) > 0:
            distances_to_new_centers = distance_matrix[unassigned_indices][:, new_centers_global]
            within_cutoff = (distances_to_new_centers <= dc).any(dim=1)
            points_to_remove = unassigned_indices[within_cutoff]
            unassigned_mask[points_to_remove] = False
            unassigned_mask[new_centers_global] = False
        
        # 如果新找到的中心数量很少，切换到顺序方法
        if len(new_centers_local) < 5:
            break
    
    # 确保我们有足够的聚类中心
    if len(cluster_centers) < target_tokens:
        remaining_indices = torch.where(unassigned_mask)[0]
        remaining_needed = target_tokens - len(cluster_centers)
        
        if len(remaining_indices) > 0:
            rho_remaining = rho[remaining_indices]
            _, top_indices = torch.topk(rho_remaining, min(remaining_needed, len(rho_remaining)))
            additional_centers = remaining_indices[top_indices]
            cluster_centers.extend(additional_centers.tolist())
    
    return torch.tensor(cluster_centers[:target_tokens], device=device)

def _compute_delta(rho_unassigned, dist_unassigned):
    """计算delta值：到更高密度点的最小距离"""
    num_points = len(rho_unassigned)
    delta = torch.full((num_points,), float('inf'), device=rho_unassigned.device)
    
    # 扩展密度矩阵用于比较
    rho_expand = rho_unassigned.unsqueeze(1).expand(num_points, num_points)
    
    # 创建掩码：只考虑密度更高的点
    higher_density_mask = rho_expand.t() > rho_expand
    higher_density_mask.fill_diagonal_(False)
    
    # 创建条件距离矩阵
    inf_mask = torch.full_like(dist_unassigned, float('inf'))
    conditioned_dist = torch.where(higher_density_mask, dist_unassigned, inf_mask)
    
    # 计算到更高密度点的最小距离
    delta_values, _ = torch.min(conditioned_dist, dim=1)
    
    # 处理最高密度的点
    max_rho = rho_unassigned.max()
    max_rho_mask = (rho_unassigned == max_rho)
    delta_values[max_rho_mask] = float('inf')
    
    return delta_values

def _merge_clusters_for_frame(frame_tokens, labels, cluster_centers, print_debug=False):
    """合并聚类得到最终的减少结果"""
    device = frame_tokens.device
    N, D = frame_tokens.shape
    
    # 创建聚类字典
    clusters = {}
    for center in cluster_centers:
        center_item = center.item()
        cluster_mask = (labels == center)
        clusters[center_item] = torch.where(cluster_mask)[0].tolist()
    
    # 计算每个聚类的平均值
    reduced_tokens = []
    cluster_indices = []
    weights_list = []
    
    for center_idx, member_indices in clusters.items():
        if len(member_indices) > 0:
            cluster_tokens = frame_tokens[member_indices]
            cluster_mean = cluster_tokens.mean(dim=0)
            
            reduced_tokens.append(cluster_mean)
            cluster_indices.append(center_idx)
            weights_list.append(len(member_indices))
    
    # 创建输出张量
    reduced_frame = torch.stack(reduced_tokens) if reduced_tokens else torch.empty(0, D, device=device)
    
    # 创建mask和weights
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    weights = torch.zeros(N, dtype=torch.float32, device=device)
    
    if cluster_indices:
        cluster_indices_tensor = torch.tensor(cluster_indices, device=device)
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=device)
        
        mask[cluster_indices_tensor] = True
        weights[cluster_indices_tensor] = weights_tensor
    
    return reduced_frame, mask, weights

