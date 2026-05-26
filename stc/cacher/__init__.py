"""STC-Cacher: reference-frame selective recomputation for vision encoders.

The two pieces here are:

* :class:`STCCache` — runtime per-stream state (chunk index, update ratio,
  optional layer feature cache).  Kept as a normal class; pass an instance
  explicitly to ``register_stc_cacher`` or use :func:`stc.default_cache`.
* :func:`stc_sdpa_attention` and the reference-frame forwards — the actual
  selective forward used after a HF vision tower has been patched.

Layer-shape-specific code (which assumes a HuggingFace pre-LN ViT) lives in
``stc.integrations.hf_vit``; this subpackage stays algorithm-agnostic.
"""

from stc.cacher.reference_forward import (
    forward_with_selective_key_recompute,
    forward_with_selective_key_recompute_clip,
    stc_sdpa_attention,
)
from stc.cacher.state import STCCache

__all__ = [
    "STCCache",
    "forward_with_selective_key_recompute",
    "forward_with_selective_key_recompute_clip",
    "stc_sdpa_attention",
]
