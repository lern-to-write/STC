import torch
import torch.nn.functional as F
from typing import List

def mae_cosine(
    batch_tensor: torch.Tensor,
    moving_avg_window: int = 5,
    threshold_factor: float = 2.0,
) -> List[torch.Tensor]:
    """
    分析一个批次的视频帧张量，为每个视频找出发生突变的帧的索引。

    该函数通过以下步骤并行处理批次中的每个序列：
    1.  计算每个序列中，每一帧与其前一帧之间的余弦相似度。
    2.  将相似度转换为“变化得分”(1 - similarity)，得分越高代表变化越大。
    3.  使用1D卷积（作为滑动平均）并行平滑每个序列的变化得分。
    4.  为每个序列独立计算其平滑得分的平均值和标准差。
    5.  为每个序列设定一个独立的动态阈值。
    6.  找出每个序列中，变化得分超过其对应阈值的帧，并返回结果列表。

    Args:
        batch_tensor (torch.Tensor): 输入的批处理张量，形状为 [batch, token_number, channel]。
        moving_avg_window (int, optional): 用于平滑变化得分的滑动平均窗口大小。默认为 5。
        threshold_factor (float, optional): 用于定义突变阈值的标准差倍数。默认为 2.0。

    Returns:
        List[torch.Tensor]: 一个列表，其中每个元素都是一个一维张量，
                            包含了对应批次中那个序列的突变帧索引。

    Raises:
        ValueError: 如果输入张量不合法或参数不正确。
    """
    # --- 1. 输入验证 ---
    if not isinstance(batch_tensor, torch.Tensor):
        raise TypeError("输入必须是一个PyTorch张量。")
    if batch_tensor.dim() != 3:
        raise ValueError("输入张量的形状必须是 [batch, token_number, channel]。")

    batch_size, token_number, _ = batch_tensor.shape
    if token_number < 2:
        # 如果所有序列的帧数都太少，无法比较
        return [torch.tensor([], dtype=torch.long) for _ in range(batch_size)]
    if moving_avg_window > token_number:
        raise ValueError("滑动窗口大小不能超过序列的token数量。")

    device = batch_tensor.device

    # --- 2. 计算帧间余弦相似度 (在批处理上并行) ---
    # `tokens_t` 形状: [batch, token_number - 1, channel]
    tokens_t = batch_tensor[:, :-1, :]
    # `tokens_t_plus_1` 形状: [batch, token_number - 1, channel]
    tokens_t_plus_1 = batch_tensor[:, 1:, :]

    # 沿着 channel 维度(dim=2)计算相似度
    # `similarities` 形状: [batch, token_number - 1]
    similarities = F.cosine_similarity(tokens_t, tokens_t_plus_1, dim=2)

    # --- 3. 计算“变化得分” ---
    # `change_scores` 形状: [batch, token_number - 1]
    change_scores = 1.0 - similarities

    # --- 4. 使用滑动平均并行平滑所有序列的得分 ---
    if token_number - 1 < moving_avg_window:
        smoothed_scores = change_scores
    else:
        # 准备输入形状 for conv1d: [batch, in_channels, length]
        scores_for_conv = change_scores.unsqueeze(1)  # -> [batch, 1, token_number - 1]

        # 创建平均核
        kernel = torch.ones(1, 1, moving_avg_window, device=device) / moving_avg_window

        # 应用卷积
        padding = (moving_avg_window - 1) // 2
        # `smoothed_scores` 形状: [batch, token_number - 1]
        smoothed_scores = F.conv1d(
            scores_for_conv, kernel.to(scores_for_conv.dtype), padding=padding
        ).squeeze(1)

    # --- 5. 为每个序列独立确定阈值并找出突变点 ---
    # 沿 token_number 维度(dim=1)计算每个序列的均值和标准差
    # 使用 keepdim=True 以便进行广播
    mean_scores = smoothed_scores.mean(dim=1, keepdim=True)  # -> [batch, 1]
    std_dev_scores = smoothed_scores.std(dim=1, keepdim=True)  # -> [batch, 1]

    # 为每个序列计算独立的阈值
    # `thresholds` 形状: [batch, 1]
    thresholds = mean_scores + threshold_factor * std_dev_scores

    # --- 6. 收集结果 ---
    results = []
    for i in range(batch_size):
        # 找出当前序列中，得分超过其对应阈值的点的索引
        indices_in_batch = torch.where(smoothed_scores[i] > thresholds[i])[0]
        # 修正索引（得分i对应第i+1帧的突变）并添加到结果列表
        results.append(indices_in_batch + 1)

    return results