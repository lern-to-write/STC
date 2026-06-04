"""Per-frame streaming wrapper for STC-Cacher on batched vision towers.

STC-Cacher needs the vision tower to be called **once per frame, in order**, so
the per-frame ``chunk_idx`` advances (``0`` = full forward that caches the
reference; later frames reuse it via selective recompute) and resets at each new
video.  Models like StreamForest / Dispider instead push *all* frames through the
tower in a single batched call, so a naive ``chunk_idx += 1`` per call only
advances once per video and selective recompute never fires.

:func:`enable_streaming_cacher` registers the cacher on the tower's inner HF
model and wraps the *outer* tower's ``forward`` so a batched ``(N, C, H, W)``
input (or a list of frames) is processed frame-by-frame, advancing ``chunk_idx``
per frame and concatenating the results back — no change to the model's
``encode_video`` / ``extract_images`` code.  Call :func:`reset_streaming_cacher`
once per video.

The selective references live on the patched HF layers (not in the cache), and
``reset_default_cache`` only clears the cache's feature buffer, so resetting
before every frame is safe.  The optional CUDA-graph runner wraps the *inner* HF
model and keys on input shape, so per-frame ``B=1`` calls are captured once and
replayed — this wrapper composes with it transparently.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
from torch import nn

from stc.cacher.state import reset_default_cache
from stc.config import CacheConfig, default_config
from stc.integrations.hf_vit import Kind, register_stc_cacher


def _streaming_forward(self, images, *args, **kwargs):
    """Outer-tower forward that advances ``chunk_idx`` once per frame."""
    ratio = self._stc_update_token_ratio
    old_forward = self._stc_old_tower_forward

    def step(frame):
        reset_default_cache(self._stc_chunk_idx, ratio)
        self._stc_chunk_idx += 1
        return old_forward(frame, *args, **kwargs)

    if isinstance(images, (list, tuple)):
        # Match the original per-image list path: return a list of features.
        return [step(img.unsqueeze(0)) for img in images]
    if torch.is_tensor(images) and images.dim() == 4:
        # (N, C, H, W) → encode each frame with B=1, then re-stack to (N, …).
        outs = [step(images[n : n + 1]) for n in range(images.shape[0])]
        return torch.cat(outs, dim=0)
    # Unknown layout: keep behaviour, advance chunk_idx once.
    return step(images)


def enable_streaming_cacher(
    outer_tower: nn.Module,
    kind: Kind = "siglip",
    config: Optional[CacheConfig] = None,
) -> nn.Module:
    """Apply STC-Cacher to ``outer_tower`` (a wrapper exposing ``.vision_tower``,
    the HF CLIP/SigLIP model) and make its ``forward`` per-frame streaming.

    Idempotent: a second call leaves the existing wrapper in place.
    """
    cfg = config if config is not None else default_config().cache
    register_stc_cacher(outer_tower.vision_tower, kind=kind, config=cfg)

    outer_tower._stc_chunk_idx = 0
    outer_tower._stc_update_token_ratio = cfg.update_token_ratio
    if not hasattr(outer_tower, "_stc_old_tower_forward"):
        outer_tower._stc_old_tower_forward = outer_tower.forward
        outer_tower.forward = types.MethodType(_streaming_forward, outer_tower)
    reset_default_cache(0, cfg.update_token_ratio)
    return outer_tower


def reset_streaming_cacher(outer_tower: Optional[nn.Module]) -> None:
    """Reset per-frame streaming state — call once at the start of each video."""
    if outer_tower is None or not hasattr(outer_tower, "_stc_update_token_ratio"):
        return
    outer_tower._stc_chunk_idx = 0
    reset_default_cache(0, outer_tower._stc_update_token_ratio)
