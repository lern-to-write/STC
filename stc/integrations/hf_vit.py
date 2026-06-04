"""Patch HuggingFace pre-LN ViT vision towers (CLIP / SigLIP) with STC-Cacher.

The two HF ViT-style vision towers the original code supported (CLIP and
SigLIP) only differ in their layer ``forward`` signature: CLIP layers accept
an additional ``causal_attention_mask``.  Instead of duplicating the patch
helper, we expose a single :func:`register_stc_cacher` that takes a ``kind``
argument.

Contract for the vision tower being patched:

* ``vision_tower.vision_model.encoder.layers`` – iterable of pre-LN encoder
  layers.
* Each layer exposes ``layer_norm1``, ``layer_norm2``, ``mlp`` and
  ``self_attn`` with ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj``.

After registration:

* ``layer.forward`` is replaced with the selective-recompute forward.
* ``layer.stc_attention`` is bound to the SDPA wrapper.
* ``layer._stc_cache`` and ``layer._stc_config`` hold the active state and
  config (defaults are filled in if not provided).
* ``layer._stc_old_forward`` holds the original method for unregistration.
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from typing import Literal, Optional

from torch import nn

from stc.cacher.reference_forward import (
    forward_with_selective_key_recompute,
    forward_with_selective_key_recompute_clip,
    stc_sdpa_attention,
)
from stc.cacher.state import STCCache, default_cache
from stc.config import CacheConfig, default_config

Kind = Literal["clip", "siglip"]


def _encoder_layers(vision_tower: nn.Module) -> Iterable[nn.Module]:
    try:
        return vision_tower.vision_model.encoder.layers
    except AttributeError as exc:
        raise AttributeError(
            "Expected a HF pre-LN ViT vision tower with "
            "vision_model.encoder.layers"
        ) from exc


def _select_forward(kind: Kind):
    if kind == "siglip":
        return forward_with_selective_key_recompute
    if kind == "clip":
        return forward_with_selective_key_recompute_clip
    raise ValueError(f"Unsupported kind: {kind!r} (expected 'clip' or 'siglip')")


def register_stc_cacher(
    vision_tower: nn.Module,
    *,
    kind: Kind = "siglip",
    cache: Optional[STCCache] = None,
    config: Optional[CacheConfig] = None,
    overwrite: bool = False,
) -> nn.Module:
    """Patch every encoder layer of ``vision_tower`` with STC-Cacher.

    The same ``cache`` / ``config`` objects are bound to every layer; the
    forwards read them from the layer at runtime so the global STC config is
    never consulted in the hot path.
    """

    resolved_cache = cache if cache is not None else default_cache()
    resolved_config = config if config is not None else default_config().cache
    forward_fn = _select_forward(kind)

    for idx, layer in enumerate(_encoder_layers(vision_tower)):
        if hasattr(layer, "_stc_old_forward") and not overwrite:
            continue
        layer._stc_old_forward = layer.forward
        layer._stc_cache = resolved_cache
        layer._stc_config = resolved_config
        # 仅第一层在 share_selection 模式下负责计算"重算哪些 token"，其余层复用它。
        layer._stc_is_selector = idx == 0
        layer.forward = types.MethodType(forward_fn, layer)
        layer.stc_attention = types.MethodType(stc_sdpa_attention, layer)

    # 可选：用 CUDA graph 回放 selective 帧的整塔前向，消掉分配 + launch 开销
    # （详见 stc.cacher.graph）。失败会自动回退 eager，永不致命。
    if getattr(resolved_config, "cuda_graph", False) and not hasattr(
        vision_tower, "_stc_old_vt_forward"
    ):
        from stc.cacher.graph import SelectiveCUDAGraphRunner

        vision_tower._stc_old_vt_forward = vision_tower.forward
        vision_tower.forward = SelectiveCUDAGraphRunner(
            vision_tower._stc_old_vt_forward, resolved_cache, resolved_config
        )
    return vision_tower


def unregister_stc_cacher(vision_tower: nn.Module) -> nn.Module:
    """Restore a vision tower previously patched by :func:`register_stc_cacher`."""

    if hasattr(vision_tower, "_stc_old_vt_forward"):
        vision_tower.forward = vision_tower._stc_old_vt_forward
        delattr(vision_tower, "_stc_old_vt_forward")

    for layer in _encoder_layers(vision_tower):
        if hasattr(layer, "_stc_old_forward"):
            layer.forward = layer._stc_old_forward
            delattr(layer, "_stc_old_forward")
        for attr in ("stc_attention", "_stc_cache", "_stc_config", "_stc_is_selector"):
            if hasattr(layer, attr):
                delattr(layer, attr)
    return vision_tower
