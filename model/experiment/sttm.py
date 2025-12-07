import torch
import torch.nn.functional as F
import math
from typing import Tuple, List

def spatial_token_merging_with_budget(
    video_tokens: torch.Tensor, 
    token_budget_per_frame: int,
    similarity_threshold: float = 0.85
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于四叉树的空间Token合并，支持固定token预算
    
    Args:
        video_tokens: 输入视频tokens，形状为 [frame_number, token_per_frame, channel]
        token_budget_per_frame: 每帧保留的token数量预算
        similarity_threshold: 相似度阈值，用于决定是否合并
    
    Returns:
        merged_tokens: 合并后的tokens，形状为 [total_merged_tokens, channel]
        token_positions: 每个token在原始帧中的位置信息，形状为 [total_merged_tokens, 4] (t, y, x, scale_level)
    """
    
    T, N, C = video_tokens.shape
    H = W = int(math.sqrt(N))  # 假设是正方形
    assert H * W == N, f"Token数量 {N} 不是完美平方数"
    
    # 重塑为 (T, H, W, C)
    video_features = video_tokens.view(T, H, W, C)
    
    all_merged_tokens = []
    all_token_positions = []
    
    # 对每一帧进行空间Token合并
    for t in range(T):
        frame_features = video_features[t]  # [H, W, C]
        
        # 构建多级特征金字塔
        feature_pyramid = build_feature_pyramid(frame_features)
        
        # 使用四叉树进行空间合并
        merged_frame_tokens, frame_positions = quadtree_spatial_merge_with_budget(
            feature_pyramid, token_budget_per_frame, similarity_threshold, t
        )
        
        all_merged_tokens.append(merged_frame_tokens)
        all_token_positions.append(frame_positions)
    
    # 合并所有帧的结果
    merged_tokens = torch.cat(all_merged_tokens, dim=0)
    token_positions = torch.cat(all_token_positions, dim=0)
    
    return merged_tokens


def build_feature_pyramid(frame_features: torch.Tensor) -> List[torch.Tensor]:
    """
    构建特征金字塔，用于四叉树分析
    
    Args:
        frame_features: 单帧特征，形状为 [H, W, C]
    
    Returns:
        feature_pyramid: 不同尺度的特征列表，从粗到精
    """
    H, W, C = frame_features.shape
    device = frame_features.device
    
    # 转换为 [C, H, W] 用于池化操作
    features = frame_features.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    
    pyramid = [features]
    current_features = features
    
    # 构建由粗到精的金字塔
    while current_features.shape[-1] > 2:  # 直到最小尺寸
        # 使用平均池化下采样
        if current_features.shape[-1] % 2 == 0 and current_features.shape[-2] % 2 == 0:
            pooled = F.avg_pool2d(current_features, kernel_size=2, stride=2)
        else:
            # 处理奇数尺寸
            pooled = adaptive_avg_pool_odd_size(current_features)
        
        pyramid.insert(0, pooled)  # 插入到前面，保持从粗到精的顺序
        current_features = pooled
    
    # 转换回 [H, W, C] 格式
    pyramid = [feat.squeeze(0).permute(1, 2, 0) for feat in pyramid]
    return pyramid


def adaptive_avg_pool_odd_size(features: torch.Tensor) -> torch.Tensor:
    """处理奇数尺寸的自适应平均池化"""
    _, _, H, W = features.shape
    new_H, new_W = math.ceil(H / 2), math.ceil(W / 2)
    
    # 使用自适应平均池化
    return F.adaptive_avg_pool2d(features, (new_H, new_W))


def quadtree_spatial_merge_with_budget(
    feature_pyramid: List[torch.Tensor], 
    token_budget: int, 
    similarity_threshold: float,
    frame_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用四叉树进行空间Token合并，支持token预算约束
    
    Args:
        feature_pyramid: 特征金字塔，从粗到精
        token_budget: token预算
        similarity_threshold: 相似度阈值
        frame_idx: 帧索引
    
    Returns:
        merged_tokens: 合并后的tokens
        positions: token位置信息
    """
    device = feature_pyramid[0].device
    n_levels = len(feature_pyramid)
    
    # 存储所有候选节点及其信息
    candidate_nodes = []
    
    # 从最粗层开始遍历
    coarsest_level = feature_pyramid[0]  # 最粗层
    H_coarse, W_coarse, C = coarsest_level.shape
    
    # 初始化根节点
    for y in range(H_coarse):
        for x in range(W_coarse):
            # 递归评估每个位置的四叉树
            evaluate_quadtree_node(
                feature_pyramid, 0, y, x, 
                similarity_threshold, candidate_nodes, frame_idx
            )
    
    # 根据相似度分数排序，选择最佳的token组合
    selected_tokens, selected_positions = select_tokens_by_budget(
        candidate_nodes, token_budget
    )
    
    return selected_tokens, selected_positions


def evaluate_quadtree_node(
    feature_pyramid: List[torch.Tensor],
    level: int,
    y: int, 
    x: int,
    similarity_threshold: float,
    candidate_nodes: List,
    frame_idx: int
) -> None:
    """
    递归评估四叉树节点
    
    Args:
        feature_pyramid: 特征金字塔
        level: 当前层级
        y, x: 当前位置
        similarity_threshold: 相似度阈值
        candidate_nodes: 候选节点列表
        frame_idx: 帧索引
    """
    current_level_features = feature_pyramid[level]
    H, W, C = current_level_features.shape
    
    # 检查是否是最精细层或超出边界
    if level >= len(feature_pyramid) - 1 or y >= H or x >= W:
        # 添加叶子节点
        if y < H and x < W:
            token = current_level_features[y, x]  # [C]
            candidate_nodes.append({
                'token': token,
                'position': torch.tensor([frame_idx, y, x, level], dtype=torch.long),
                'score': 1.0,  # 叶子节点分数为1
                'area': 1,
                'level': level
            })
        return
    
    # 获取当前节点特征
    parent_token = current_level_features[y, x]  # [C]
    
    # 获取下一层对应的4个子节点
    next_level_features = feature_pyramid[level + 1]
    H_next, W_next, _ = next_level_features.shape
    
    # 计算子节点坐标
    child_coords = []
    child_tokens = []
    
    for dy in range(2):
        for dx in range(2):
            child_y = y * 2 + dy
            child_x = x * 2 + dx
            
            if child_y < H_next and child_x < W_next:
                child_coords.append((child_y, child_x))
                child_tokens.append(next_level_features[child_y, child_x])
    
    if len(child_tokens) == 0:
        # 没有有效子节点，添加当前节点
        candidate_nodes.append({
            'token': parent_token,
            'position': torch.tensor([frame_idx, y, x, level], dtype=torch.long),
            'score': 1.0,
            'area': 2 ** (len(feature_pyramid) - 1 - level),
            'level': level
        })
        return
    
    # 计算子节点与父节点的相似度
    child_tokens_tensor = torch.stack(child_tokens)  # [num_children, C]
    parent_token_expanded = parent_token.unsqueeze(0).expand_as(child_tokens_tensor)
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(
        parent_token_expanded.float(), 
        child_tokens_tensor.float(), 
        dim=-1
    )
    
    # 计算平均相似度作为该区域的分数
    avg_similarity = similarities.mean().item()
    
    # 根据相似度决定是否继续细分
    if avg_similarity >= similarity_threshold:
        # 相似度高，使用父节点代表整个区域
        candidate_nodes.append({
            'token': parent_token,
            'position': torch.tensor([frame_idx, y, x, level], dtype=torch.long),
            'score': avg_similarity,
            'area': len(child_tokens),
            'level': level
        })
    else:
        # 相似度低，递归处理子节点
        for child_y, child_x in child_coords:
            evaluate_quadtree_node(
                feature_pyramid, level + 1, child_y, child_x,
                similarity_threshold, candidate_nodes, frame_idx
            )


def select_tokens_by_budget(
    candidate_nodes: List, 
    token_budget: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据token预算选择最优的token组合
    
    Args:
        candidate_nodes: 候选节点列表
        token_budget: token预算
    
    Returns:
        selected_tokens: 选中的tokens
        selected_positions: 选中的位置信息
    """
    if len(candidate_nodes) <= token_budget:
        # 如果候选节点少于预算，全部选择
        tokens = torch.stack([node['token'] for node in candidate_nodes])
        positions = torch.stack([node['position'] for node in candidate_nodes])
        return tokens, positions
    
    # 按照分数和覆盖面积的综合指标排序
    # 优先选择高相似度（可以用粗粒度表示）且覆盖面积大的节点
    for node in candidate_nodes:
        node['priority'] = node['score'] * node['area']
    
    # 按优先级降序排序
    candidate_nodes.sort(key=lambda x: x['priority'], reverse=True)
    
    # 贪心选择，避免重叠
    selected_nodes = []
    covered_positions = set()
    
    for node in candidate_nodes:
        if len(selected_nodes) >= token_budget:
            break
            
        # 检查是否与已选择的节点重叠
        pos = node['position']
        t, y, x, level = pos[0].item(), pos[1].item(), pos[2].item(), pos[3].item()
        
        # 计算该节点覆盖的区域
        scale_factor = 2 ** (len(candidate_nodes) - 1 - level) if len(candidate_nodes) > 1 else 1
        node_region = set()
        for dy in range(scale_factor):
            for dx in range(scale_factor):
                node_region.add((t, y * scale_factor + dy, x * scale_factor + dx))
        
        # 检查重叠
        if not node_region.intersection(covered_positions):
            selected_nodes.append(node)
            covered_positions.update(node_region)
    
    # 如果选择的节点不足，补充剩余的节点
    while len(selected_nodes) < token_budget and len(selected_nodes) < len(candidate_nodes):
        for node in candidate_nodes:
            if node not in selected_nodes:
                selected_nodes.append(node)
                if len(selected_nodes) >= token_budget:
                    break
    
    # 提取tokens和positions
    tokens = torch.stack([node['token'] for node in selected_nodes])
    positions = torch.stack([node['position'] for node in selected_nodes])
    
    return tokens,positions


