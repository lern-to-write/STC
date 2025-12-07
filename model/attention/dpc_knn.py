import torch
import torch.nn.functional as F
import torch.nn.functional as F

import torch


def _dpc_knn_single(video_tensor: torch.Tensor, k: int, n_to_keep: int) -> torch.Tensor:
    """
    (内部帮助函数) 根据DPC-KNN算法从单个视频样本中选择关键token。

    Args:
        video_tensor (torch.Tensor): 输入的单个视频token张量，形状为 [token_number, channel]。
        k (int): 用于计算局部密度的K值。
        n_to_keep (int): 希望保留的token数量。

    Returns:
        torch.Tensor: 一个包含保留的token索引的一维张量。
    """
    n_tokens, _ = video_tensor.shape
    device = video_tensor.device

    # 1. 计算所有token之间的欧氏距离矩阵
    dist_matrix = torch.cdist(video_tensor, video_tensor, p=2)

    # 2. 计算局部密度 (ρ)
    sorted_dists, _ = torch.sort(dist_matrix, dim=1)
    k_nearest_dists = sorted_dists[:, 1 : k + 1]
    rho = -k_nearest_dists.mean(dim=1)

    # 3. 计算相对距离 (δ)
    sorted_rho_indices = torch.argsort(rho, descending=True)
    delta = torch.full((n_tokens,), float("inf"), device=device)

    for i in range(1, n_tokens):
        current_idx = sorted_rho_indices[i]
        higher_density_indices = sorted_rho_indices[:i]
        delta[current_idx] = dist_matrix[current_idx, higher_density_indices].min()

    highest_rho_idx = sorted_rho_indices[0]
    # 避免在只有一个token时出错
    if n_tokens > 1:
        delta[highest_rho_idx] = dist_matrix[highest_rho_idx].max()
    else:
        delta[highest_rho_idx] = 0.0  # 如果只有一个token，delta为0

    # 4. 计算决策值 (γ = ρ * δ)
    # 为避免不同量纲影响，先对ρ和δ进行归一化
    rho_norm = (rho - rho.min()) / (rho.max() - rho.min() + 1e-8)
    delta_norm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-8)
    gamma = rho_norm * delta_norm

    # 5. 选择top-N的token
    top_indices = torch.argsort(gamma, descending=True)

    return top_indices[:n_to_keep]


def dpc_knn_select_tokens_batched(
    video_batch_tensor: torch.Tensor, k: int, n_to_keep: int
) -> torch.Tensor:
    """
    根据DPC-KNN算法从一批视频token中为每个样本选择一部分关键token。

    该函数会遍历batch中的每个样本，并独立地应用DPC-KNN算法。

    Args:
        video_batch_tensor (torch.Tensor): 输入的视频token批次张量，
                                          形状为 [batch, token_number, channel]。
        k (int): 用于计算局部密度时的最近邻数量 (K-Nearest Neighbors)。
        n_to_keep (int): 希望为每个样本最终保留的token数量。

    Returns:
        torch.Tensor: 一个包含每个样本保留token索引的二维张量，形状为 [batch, n_to_keep]。

    Example:
        >>> # 创建一个示例输入批次张量
        >>> batch_size = 4
        >>> tokens = 100
        >>> channels = 64
        >>> video_batch = torch.randn(batch_size, tokens, channels)
        >>> # 为每个样本选择10个最重要的token
        >>> selected_indices_batch = dpc_knn_select_tokens_batched(video_batch, k=5, n_to_keep=10)
        >>> print(selected_indices_batch.shape)
        torch.Size([4, 10])
    """
    if not isinstance(video_batch_tensor, torch.Tensor):
        raise TypeError("输入必须是一个PyTorch张量。")
    if video_batch_tensor.dim() != 3:
        raise ValueError("输入张量的形状必须是 [batch, token_number, channel]。")

    batch_size, n_tokens, _ = video_batch_tensor.shape

    if n_to_keep > n_tokens:
        raise ValueError(
            f"希望保留的token数量 ({n_to_keep}) 不能超过总token数量 ({n_tokens})。"
        )
    if k >= n_tokens:
        raise ValueError(f"K值 ({k}) 必须小于总token数量 ({n_tokens})。")

    # 遍历batch中的每个样本，分别计算并收集结果
    results_list = []
    for i in range(batch_size):
        single_video_tensor = video_batch_tensor[i]
        selected_indices = _dpc_knn_single(single_video_tensor, k, n_to_keep)
        results_list.append(selected_indices)

    # 将结果列表堆叠成一个张量
    return torch.stack(results_list, dim=0)