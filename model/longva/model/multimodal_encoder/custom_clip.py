import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import CLIPVisionModel, CLIPVisionConfig

from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPEncoder, CLIPConfig
from transformers.modeling_outputs import BaseModelOutput

from typing import Optional, Tuple,Union


# ✅ 简单的全局重置标志
RESET_CACHE_FLAG = False

    
class LayerRatioAllocator:
    """
    简化版层级 skip ratio 分配器。支持:
    - uniform: 各层同一比例
    - linear_increasing: 由浅到深线性递增（默认）
    """
    def __init__(self, num_layers: int, target_ratio: float = 0.3):
        self.num_layers = num_layers
        self.target_ratio = float(target_ratio)
        self.layer_ratios = self._initialize_layer_ratios()

    def _initialize_layer_ratios(self):
        strategy = os.getenv("LAYER_RATIO_STRATEGY", "uniform")
        if strategy == "uniform":
            return [self.target_ratio] * self.num_layers
        # linear_increasing
        ratios = []
        for i in range(self.num_layers):
            ratio = self.target_ratio * (0.2 + 1.6 * (i / max(self.num_layers - 1, 1)))
            ratios.append(ratio)
        avg = sum(ratios) / len(ratios)
        if avg > 0:
            ratios = [r * (self.target_ratio / avg) for r in ratios]
        return ratios

    def get_layer_ratio(self, layer_idx: int) -> float:
        if layer_idx >= self.num_layers:
            return self.target_ratio
        return float(self.layer_ratios[layer_idx])


class TokenLevelCacheCLIPEncoderLayer(CLIPEncoderLayer):
    """
    参考 custom_siglip 的结构，为 CLIP 的每层增加：
    - 偶数 chunk：全量计算，缓存最后一帧 pre-LN2 以及 MLP 输出
    - 奇数 chunk：LN1+Attention 全量，LN2+MLP 阶段按相似度跳过部分 token，直接复用缓存结果
    说明：
    - 这里将 batch 维度视作“帧数” F（对图像也一样工作，只是 F=批大小）
    - token_per_frame = 序列长度（patch 数+cls），通常含 cls；若你只对 patch 操作，可在上游做裁剪
    """
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.layer_idx: Optional[int] = None
        self.ratio_allocator: Optional[LayerRatioAllocator] = None

        # 缓存
        self.reference_frame_pre_ln2 = None  # [T, C]
        self.reference_frame_mlp_post = None  # [T, C]

        # 统计
        self.total_tokens_processed = 0
        self.total_tokens_skipped = 0

        # 配置
        self.base_skip_token_ratio = float(os.getenv("SKIP_TOKEN_RATIO", "0.8"))
        self._current_chunk_idx = 0

    # ------- 基础工具 -------
    def set_chunk_index(self, chunk_idx: int):
        self._current_chunk_idx = int(chunk_idx)
        if self._current_chunk_idx == 0:
            self.clear_cache()

    def get_chunk_index(self) -> int:
        return int(getattr(self, "_current_chunk_idx", 0))

    def clear_cache(self):
        self.reference_frame_pre_ln2 = None
        self.reference_frame_mlp_post = None
        self.total_tokens_processed = 0
        self.total_tokens_skipped = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_cache_stats(self):
        total = max(self.total_tokens_processed, 1)
        return {
            "layer_idx": self.layer_idx,
            "total_tokens_processed": int(self.total_tokens_processed),
            "total_tokens_skipped": int(self.total_tokens_skipped),
            "actual_skip_ratio": float(self.total_tokens_skipped) / total,
        }

    def get_layer_skip_ratio(self) -> float:
        if self.ratio_allocator is not None:
            return float(self.ratio_allocator.get_layer_ratio(self.layer_idx))
        return float(self.base_skip_token_ratio)

    # ------- 相似度与索引选择 -------
    @staticmethod
    def _cosine_sim(hidden_sel: torch.Tensor, ref_sel: torch.Tensor) -> torch.Tensor:
        # hidden_sel: [F, T, C], ref_sel: [T, C]
        return F.cosine_similarity(hidden_sel, ref_sel.unsqueeze(0), dim=-1, eps=1e-8)

    def _compute_preln_skip_indices(
        self, hidden_states_after_attn_residual: torch.Tensor
    ):
        """
        输入 residual2，也就是 Attention 段之后、进入 LN2 前的特征:
        hidden_states_after_attn_residual: [F, T, C]
        返回: skip_indices [F, S], compute_indices [F, K]
        """
        Fn, T, C = hidden_states_after_attn_residual.shape

        # 没有参考帧则不跳过
        if self.reference_frame_pre_ln2 is None:
            all_idx = torch.arange(T, device=hidden_states_after_attn_residual.device)[None, :].expand(Fn, -1)
            return (
                torch.empty(Fn, 0, dtype=torch.long, device=hidden_states_after_attn_residual.device),
                all_idx,
            )

        with torch.no_grad():
            ref = self.reference_frame_pre_ln2  # [T, C]
            sim = self._cosine_sim(hidden_states_after_attn_residual, ref)  # [F, T]

            # 层级 skip ratio
            use_layer_ratio = os.getenv("LAYER_RATIO_ENABLED", "0").lower() in ("1", "true", "yes")
            skip_ratio = self.get_layer_skip_ratio() if use_layer_ratio else self.base_skip_token_ratio
            num_skip = int(max(0, min(T, int(T * skip_ratio))))
            num_comp = T - num_skip

            if num_skip > 0:
                skip_indices = torch.topk(sim, k=num_skip, dim=1, largest=True).indices  # [F, S]
            else:
                skip_indices = torch.empty(Fn, 0, dtype=torch.long, device=sim.device)

            if num_comp > 0:
                all_idx = torch.arange(T, device=sim.device)[None, :].expand(Fn, -1)
                comp_mask = torch.ones_like(all_idx, dtype=torch.bool).scatter(1, skip_indices, False)
                compute_indices = all_idx[comp_mask].view(Fn, num_comp)  # [F, K]
            else:
                compute_indices = torch.empty(Fn, 0, dtype=torch.long, device=sim.device)

        return skip_indices, compute_indices

    # ------- 两种执行路径 -------
    def _forward_preln_token_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ):
        """
        偶数 chunk：全量计算并更新缓存
        奇数 chunk：LN1+Attn 全量；LN2+MLP 使用缓存按 token 跳过
        """
        os.environ['RESET_CLIP_CACHE'] = '0'
        
        chunk_idx = self.get_chunk_index() 
        Fn, T, C = hidden_states.shape

        # LN1
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)

        # Attention 全量
        attn_out, attn_weights = self.self_attn(
            hidden_states=hidden_states_ln1,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual1 + attn_out

        # 进入 LN2/MLP
        residual2 = hidden_states

        is_even = (chunk_idx % 2 == 0)
        if is_even:
            # 全量 LN2 + MLP
            hs_ln2 = self.layer_norm2(hidden_states)
            mlp_out_full = self.mlp(hs_ln2)
            hidden_states = residual2 + mlp_out_full

            # 更新缓存（取最后一“帧”作为参考）
            with torch.no_grad():
                self.reference_frame_pre_ln2 = residual2[-1].detach()           # [T, C]
                self.reference_frame_mlp_post = mlp_out_full[-1].detach()       # [T, C]

            self.total_tokens_processed += Fn * T
            return (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        else:
            # 奇数 chunk：只在 LN2+MLP 阶段跳过
            skip_indices, compute_indices = self._compute_preln_skip_indices(residual2)

            if self.reference_frame_mlp_post is None or compute_indices.shape[1] == T:
                # 无缓存或无可跳过：走全量
                hs_ln2 = self.layer_norm2(hidden_states)
                mlp_out_full = self.mlp(hs_ln2)
                hidden_states = residual2 + mlp_out_full
            else:
                # 仅计算 compute 的 token，其余用参考帧 MLP 输出
                out = self.reference_frame_mlp_post.unsqueeze(0).expand(Fn, -1, -1).clone()  # [F, T, C]
                num_comp = compute_indices.shape[1] if compute_indices.numel() > 0 else 0
                if num_comp > 0:
                    idx_comp = compute_indices.unsqueeze(-1).expand(-1, -1, C)             # [F, K, C]
                    tokens_to_ln2 = hidden_states.gather(1, idx_comp)                      # [F, K, C]
                    tokens_ln2 = self.layer_norm2(tokens_to_ln2)                           # [F, K, C]
                    tokens_mlp = self.mlp(tokens_ln2)                                      # [F, K, C]
                    out.scatter_(1, idx_comp, tokens_mlp)
                hidden_states = residual2 + out

            # 统计
            num_skip = skip_indices.shape[1] if skip_indices.numel() > 0 else 0
            print("num_skip",num_skip)
            
            self.total_tokens_processed += Fn * T
            self.total_tokens_skipped += Fn * num_skip

            return (hidden_states, attn_weights) if output_attentions else (hidden_states,)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal_attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.FloatTensor]:

        """
        默认行为保持与 CLIP 一致；当 CACHE_STRATEGY=token_level_cache_preln 时启用缓存路径
        """
        cache_strategy = os.getenv("CACHE_STRATEGY", "none").lower()
        if cache_strategy == "token_level_cache_preln":
            return self._forward_preln_token_cache(hidden_states, attention_mask, output_attentions)

        # 原始路径（不缓存）
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

def patch_clip_with_token_cache(model, skip_token_ratio=0.3):
    """
    将传入的 CLIPVisionModel 的 encoder.layers 替换为 TokenLevelCacheCLIPEncoderLayer，
    并完成权重拷贝和层级 ratio 分配。
    """
    if not isinstance(model, CLIPVisionModel):
        raise TypeError("patch_clip_with_token_cache expects a transformers.CLIPVisionModel")

    # 取到 encoder 层列表
    layers = model.vision_model.encoder.layers
    
    num_layers = len(layers)
    allocator = LayerRatioAllocator(num_layers=num_layers, target_ratio=float(skip_token_ratio))

    new_layers = nn.ModuleList()
    for i, old_layer in enumerate(layers):
        new_layer = TokenLevelCacheCLIPEncoderLayer(model.config)  # CLIPVisionConfig
        new_layer.load_state_dict(old_layer.state_dict())
        new_layer.layer_idx = i
        new_layer.ratio_allocator = allocator
        new_layers.append(new_layer)

    # 替换
    model.vision_model.encoder.layers = new_layers

    # 给 model 添加便捷方法（可选）
    def _set_chunk_index_all(idx: int):
        for lyr in model.vision_model.encoder.layers:
            if hasattr(lyr, "set_chunk_index"):
                lyr.set_chunk_index(int(idx))

    def _clear_all_cache():
        for lyr in model.vision_model.encoder.layers:
            if hasattr(lyr, "clear_cache"):
                lyr.clear_cache()

    def _get_all_cache_stats():
        stats = {}
        for lyr in model.vision_model.encoder.layers:
            if hasattr(lyr, "get_cache_stats"):
                stats[f"layer_{lyr.layer_idx}"] = lyr.get_cache_stats()
        return stats

    model.set_chunk_index = _set_chunk_index_all
    model.clear_all_cache = _clear_all_cache
    model.get_all_cache_stats = _get_all_cache_stats
    
    return model

