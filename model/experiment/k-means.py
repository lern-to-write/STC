import torch
import torch_kmeans
from torch.nn.functional import pairwise_distance

def reduce_tokens_with_kmeans_selective(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    使用 K-means 聚类将 Tensor 中每帧的 token 数量减少一半，
    并通过选择距离簇中心最近的原始 token 来保留更具代表性的 token。

    Args:
        input_tensor (torch.Tensor): 输入 Tensor，形状为 [frame_number, token_per_frame, channel]。

    Returns:
        torch.Tensor: 输出 Tensor，形状为 [frame_number, token_per_frame//2, channel]。
                      每帧保留了 token_per_frame//2 个原始 token。
                      注意：token_per_frame 应该是偶数。
    """
    frame_number, token_per_frame, channel = input_tensor.shape

    if token_per_frame % 2 != 0:
        raise ValueError("token_per_frame should be an even number to allow halving.")

    k = token_per_frame // 2
    reduced_frames = []

    for i in range(frame_number):
        frame_data = input_tensor[i] # Shape: [token_per_frame, channel]

        # 执行 K-means 聚类
        kmeans_model = torch_kmeans.KMeans(n_clusters=k, mode='euclidean', verbose=0)
        labels = kmeans_model.fit_predict(frame_data.unsqueeze(0)).long() # Shape: [1, token_per_frame]
        labels = labels.squeeze(0) # Shape: [token_per_frame], 每个 token 对应的簇标签
        cluster_centers = kmeans_model.centers.squeeze(0) # Shape: [k, channel], 簇中心

        # 选择每个簇中距离中心最近的原始 token
        selected_tokens = []
        for cluster_idx in range(k):
            # 找到属于当前簇的所有 token 的索引
            indices_in_cluster = torch.where(labels == cluster_idx)[0] # Shape: [num_points_in_cluster]

            if len(indices_in_cluster) > 0:
                # 获取这些 token 的数据
                tokens_in_cluster = frame_data[indices_in_cluster] # Shape: [num_points_in_cluster, channel]
                # 获取当前簇的中心
                center = cluster_centers[cluster_idx] # Shape: [channel]
                # 计算这些 token 到簇中心的距离
                # unsqueeze(0) for center to allow broadcasting
                distances = pairwise_distance(tokens_in_cluster, center.unsqueeze(0), p=2) # Shape: [num_points_in_cluster]
                # 找到距离最小的 token 的索引 (相对于 tokens_in_cluster)
                closest_idx_in_cluster = torch.argmin(distances)
                # 获取该 token 在原始 frame_data 中的索引
                original_idx = indices_in_cluster[closest_idx_in_cluster]
                # 选择这个原始 token
                selected_tokens.append(frame_data[original_idx]) # Shape: [channel]
            # else: 理论上 K-means 不应该产生空簇，但为健壮性可加处理

        # 将选中的 token 堆叠成一帧
        if selected_tokens:
             # List of [channel] tensors -> [k, channel] tensor
            reduced_frame = torch.stack(selected_tokens, dim=0)
        else:
            # 如果所有簇都为空（理论上不会发生），返回零张量或报错
            reduced_frame = torch.zeros((k, channel), dtype=frame_data.dtype, device=frame_data.device)

        reduced_frames.append(reduced_frame)

    # 将所有帧堆叠起来
    output_tensor = torch.stack(reduced_frames, dim=0) # Shape: [frame_number, k, channel]
    return output_tensor