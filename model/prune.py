import torch
import torch.nn.functional as F
import os
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass

# 假设外部 config 存在
from model.config import get_config

@dataclass
class ModelSpec:
    tokens_per_frame: int
    index_mapper_type: str 

MODEL_SPECS = {
    "llava_ov":  ModelSpec(tokens_per_frame=196, index_mapper_type="flat"),
    "llava_vid": ModelSpec(tokens_per_frame=169, index_mapper_type="grid_13x13"),
    "clip":      ModelSpec(tokens_per_frame=144, index_mapper_type="flat"),
}

class ScoreCalculator:
    @staticmethod
    def gaussian_similarity(
        features: torch.Tensor, 
        target: torch.Tensor, 
        alphas: Optional[List[float]] = None
    ) -> torch.Tensor:

        if alphas is None:
            alphas = [2**k for k in range(-3, 2)]
        diff = features - target  # Broadcasting occurs here
        l2_dist_sq = torch.sum(diff ** 2, dim=-1) # [B, N]
        scores = sum(torch.exp(-l2_dist_sq / (2 * alpha)) for alpha in alphas)
        return scores

    @staticmethod
    def compute_scores(
        reshaped_features: torch.Tensor, 
        memory_mean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一次性计算 Frame, Video, Memory 三种分数"""
        # 统一归一化
        features_norm = F.normalize(reshaped_features, dim=-1)
        
        # 1. Frame Mean: 当前帧的均值
        frame_means = features_norm.mean(dim=1, keepdim=True) # [Frames, 1, D]
        frame_scores = ScoreCalculator.gaussian_similarity(features_norm, frame_means)
        
        # 2. Video Mean: 整个视频片段的均值
        video_mean = features_norm.mean(dim=(0, 1), keepdim=True) # [1, 1, D]
        video_scores = ScoreCalculator.gaussian_similarity(features_norm, video_mean)
        
        # 3. Memory Mean: 历史记忆均值
        memory_mean_norm = F.normalize(memory_mean, dim=-1).view(1, 1, -1)
        memory_scores = ScoreCalculator.gaussian_similarity(features_norm, memory_mean_norm)
        
        return frame_scores, video_scores, memory_scores



# --- 3. 索引处理工具 (Index Utils) ---

class IndexMapper:
    @staticmethod
    def map_indices(model_spec: ModelSpec, local_indices: List[torch.Tensor], 
                    device: torch.device, original_features: torch.Tensor) -> torch.Tensor:
        
        if model_spec.index_mapper_type == "flat":
            return IndexMapper._map_flat(local_indices, model_spec.tokens_per_frame, device)
        elif model_spec.index_mapper_type == "grid_13x13":
            return IndexMapper._map_grid(local_indices, 13, device)
        else:
            raise NotImplementedError(f"Mapper {model_spec.index_mapper_type} not implemented")

    @staticmethod
    def _map_flat(indices_list: List[torch.Tensor], tokens_per_frame: int, device: torch.device) -> torch.Tensor:
        num_frames = len(indices_list)
        offsets = torch.arange(num_frames, device=device).unsqueeze(1) * tokens_per_frame
        global_indices = torch.cat([idx + off for idx, off in zip(indices_list, offsets)])
        return global_indices

    @staticmethod
    def _map_grid(indices_list: List[torch.Tensor], size: int, device: torch.device) -> torch.Tensor:
        """针对 LLaVA-Video 等需要处理空间 Grid 结构的映射"""
        H, W = size, size
        W_new = W + 1 # 考虑到某些特殊的 grid token 布局
        
        all_global_indices = []
        for frame_idx, local_idx in enumerate(indices_list):
            rows, cols = torch.div(local_idx, W, rounding_mode='floor'), local_idx % W
            frame_start = frame_idx * (H * W_new)
            feat_global = frame_start + (rows * W_new + cols)
            all_global_indices.append(feat_global)
            row_markers = torch.arange(H, device=device) * W_new + W
            all_global_indices.append(frame_start + row_markers)
            
        return torch.cat(all_global_indices, dim=0)

class STC_Pruner:
    def __init__(self):
        self.past_memory_mean_token: List[torch.Tensor] = []

    def _update_memory(self, current_features: torch.Tensor) -> torch.Tensor:
        current_chunk_mean = current_features.mean(dim=(0, 1), keepdim=True) # [1, 1, Dim]
        self.past_memory_mean_token.append(current_chunk_mean)
        history = self.past_memory_mean_token 
        return torch.mean(torch.cat(history, dim=0), dim=0)

    def select_feature_channel(self, tensor: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
        channel_var = tensor.var(dim=0, unbiased=False)
        k = int(channel_var.shape[0] * keep_ratio)
        _, indices = torch.topk(channel_var, k=k, largest=False)
        return tensor[:, indices]

    def compress(self, 
                 flattened_features: torch.Tensor, 
                 model_name: str = "llava_ov",
                 raw_image_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}")
        spec = MODEL_SPECS[model_name]
        if model_name == "llava_vid" and raw_image_features is None:
            raise ValueError("llava_vid requires raw_image_features")
        selected_features = self.select_feature_channel(flattened_features)
        num_frames = selected_features.shape[0] // spec.tokens_per_frame
        reshaped_features = selected_features.view(num_frames, spec.tokens_per_frame, -1)
        memory_mean_token = self._update_memory(reshaped_features)
        frame_score, video_score, memory_score = ScoreCalculator.compute_scores(
            reshaped_features, memory_mean_token
        )
        combined_score = memory_score + frame_score
        kept_indices_list = []
        token_cfg_per_frame = int(get_config().model.token_per_frame) 

        for i in range(num_frames):
            k = token_cfg_per_frame 
            _, idx = torch.topk(combined_score[i], k=k, largest=False)
            kept_indices_list.append(idx.sort().values)
        final_indices = IndexMapper.map_indices(
            spec, kept_indices_list, flattened_features.device, flattened_features
        )
        if model_name == "llava_vid":
            return raw_image_features[final_indices]
        else:
            return flattened_features[final_indices]



