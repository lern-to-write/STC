import torch
import heapq
from typing import List, Tuple


def adaptive_keyframe_sampling(
    scores: torch.Tensor,
    max_frames: int = 64,
    t1: float = 0.8,
    t2: float = -100.0,
    max_depth: int = 5,
) -> List[List[int]]:
    """
    基于AKS算法的自适应关键帧采样

    Args:
        scores: 形状为 [batch, frame_number] 的分数张量
        max_frames: 最大选择的帧数
        t1: 相关性阈值（relevance threshold）
        t2: 方差阈值（variance threshold）
        max_depth: 最大递归深度

    Returns:
        List[List[int]]: 每个batch中选中的帧索引列表
    """
    batch_size, total_frames = scores.shape
    device = scores.device

    results = []

    for batch_idx in range(batch_size):
        batch_scores = scores[batch_idx].cpu().numpy()
        frame_indices = list(range(total_frames))

        # 标准化分数到[0,1]
        if batch_scores.max() != batch_scores.min():
            normalized_scores = (batch_scores - batch_scores.min()) / (
                batch_scores.max() - batch_scores.min()
            )
        else:
            normalized_scores = batch_scores

        # 递归采样
        selected_indices = _recursive_sampling(
            normalized_scores, frame_indices, max_frames, t1, t2, max_depth, depth=0
        )

        # 确保选择的帧数不超过max_frames
        if len(selected_indices) > max_frames:
            # 如果超出，按分数排序并取top-k
            score_index_pairs = [(batch_scores[i], i) for i in selected_indices]
            score_index_pairs.sort(reverse=True, key=lambda x: x[0])
            selected_indices = [pair[1] for pair in score_index_pairs[:max_frames]]

        # 按时间顺序排序
        selected_indices.sort()
        results.append(selected_indices)

    return results


def _recursive_sampling(
    scores: List[float],
    indices: List[int],
    target_frames: int,
    t1: float,
    t2: float,
    max_depth: int,
    depth: int,
) -> List[int]:
    """
    递归采样核心函数
    """
    if len(scores) <= target_frames or depth >= max_depth:
        # 如果帧数不足或达到最大深度，直接返回所有索引
        return indices

    # 计算统计信息
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance**0.5

    # 找到top-n分数
    top_n = min(target_frames, len(scores))
    top_indices_in_scores = heapq.nlargest(
        top_n, range(len(scores)), key=lambda i: scores[i]
    )
    top_scores = [scores[i] for i in top_indices_in_scores]
    top_mean = sum(top_scores) / len(top_scores)

    # 计算相关性差异
    mean_diff = top_mean - mean_score

    # 判断是否直接采样
    if mean_diff > t1 and std_score > t2:
        # 条件满足，直接采样top frames
        selected_indices = [indices[i] for i in top_indices_in_scores]
        return selected_indices
    else:
        # 条件不满足，分割成两部分递归处理
        mid_point = len(scores) // 2

        # 分割分数和索引
        left_scores = scores[:mid_point]
        right_scores = scores[mid_point:]
        left_indices = indices[:mid_point]
        right_indices = indices[mid_point:]

        # 计算每部分应该分配的帧数
        left_frames = int(target_frames * len(left_scores) / len(scores))
        right_frames = target_frames - left_frames

        # 递归处理左右两部分
        left_selected = _recursive_sampling(
            left_scores, left_indices, left_frames, t1, t2, max_depth, depth + 1
        )
        right_selected = _recursive_sampling(
            right_scores, right_indices, right_frames, t1, t2, max_depth, depth + 1
        )

        return left_selected + right_selected


# 使用示例
if __name__ == "__main__":
    # 示例：批量大小为2，每个视频100帧
    batch_size, frame_count = 2, 100

    # 模拟帧分数（随机生成，实际使用中是BLIP/CLIP等模型的输出）
    scores = torch.randn(batch_size, frame_count)
    scores = torch.softmax(scores, dim=1)  # 模拟概率分布

    # 执行自适应关键帧采样
    selected_frames = adaptive_keyframe_sampling(
        scores=scores,
        max_frames=32,  # 选择32帧
        t1=0.8,  # 相关性阈值
        t2=-100,  # 方差阈值（设置为负值使其更容易满足条件）
        max_depth=5,  # 最大递归深度
    )

    print(f"Batch 0 选中的帧索引: {selected_frames[0]}")
    print(f"Batch 1 选中的帧索引: {selected_frames[1]}")
    print(f"选中帧数: {len(selected_frames[0])}, {len(selected_frames[1])}")
