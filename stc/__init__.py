"""Streaming Token Compression (STC).

Top-level public API.  The package has four logical layers:

* :mod:`stc.config` — dataclasses (``CacheConfig`` / ``ModelConfig`` / ``STCConfig``)
  and the legacy process-wide singleton accessors.
* :mod:`stc.core` — pure algorithms (token similarity, anchors, scoring,
  index mapping, layer ratios).  No HF or torch-distributed dependency
  beyond ``torch`` itself.
* :mod:`stc.cacher` — STC-Cacher: per-stream :class:`STCCache` plus the
  selective-recompute layer forwards.
* :mod:`stc.pruner` — STC-Pruner: :class:`STCPruner` for visual token
  compression before LLM prefill.
* :mod:`stc.integrations` — adapters that bind the cacher into specific
  vision-tower libraries.  Currently :func:`register_stc_cacher` for
  HuggingFace pre-LN ViT (CLIP / SigLIP).

Typical usage::

    import stc

    cfg = stc.STCConfig()                    # or stc.default_config()
    cache = stc.STCCache()
    stc.register_stc_cacher(vision_tower, kind="siglip",
                            cache=cache, config=cfg.cache)

    pruner = stc.STCPruner(cfg.model)
    out = pruner.compress(features, model="llava_ov")

    # advance to the next chunk between encodings
    cache.reset_for_chunk(chunk_idx=1, update_token_ratio=0.25)
"""

from stc.cacher import STCCache
from stc.cacher.state import default_cache, reset_default_cache, set_default_cache
from stc.config import (
    CacheConfig,
    GlobalConfig,
    ModelConfig,
    STCConfig,
    default_config,
    get_config,
)
from stc.core import (
    LayerRatioAllocator,
    select_dynamic_token_indices,
    token_similarity,
)
from stc.integrations.hf_vit import register_stc_cacher, unregister_stc_cacher
from stc.pruner import (
    MODEL_SPECS,
    AnchorMemory,
    IndexMapper,
    ModelSpec,
    STCPruner,
    ScoreCalculator,
    dual_anchor_scores,
    spatial_anchor,
)

__all__ = [
    # Config
    "CacheConfig",
    "GlobalConfig",
    "ModelConfig",
    "STCConfig",
    "default_config",
    "get_config",
    # Cacher
    "STCCache",
    "default_cache",
    "reset_default_cache",
    "set_default_cache",
    # Pruner
    "MODEL_SPECS",
    "ModelSpec",
    "STCPruner",
    # Core algorithms
    "AnchorMemory",
    "IndexMapper",
    "LayerRatioAllocator",
    "ScoreCalculator",
    "dual_anchor_scores",
    "select_dynamic_token_indices",
    "spatial_anchor",
    "token_similarity",
    # Integrations
    "register_stc_cacher",
    "unregister_stc_cacher",
]
