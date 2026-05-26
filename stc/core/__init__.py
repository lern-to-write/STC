"""Cross-cutting algorithms used by multiple STC components.

Only primitives that are reused across the cacher, the pruner, or external
integrations live here.  Pruner-internal algorithms (scoring, anchors,
index mapping) stay inside :mod:`stc.pruner`.
"""

from stc.core.layer_ratio import LayerRatioAllocator
from stc.core.selectors import select_dynamic_token_indices, token_similarity

__all__ = [
    "LayerRatioAllocator",
    "select_dynamic_token_indices",
    "token_similarity",
]
