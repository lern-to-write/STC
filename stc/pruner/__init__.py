"""STC-Pruner: query-agnostic visual token pruning.

Compresses a flat ``[tokens, channels]`` tensor of vision-tower features down
to ``token_per_frame`` tokens per frame before LLM prefill.

Algorithm building blocks live next to the orchestrator class:

* :mod:`stc.pruner.scoring` — Gaussian and dual-anchor score functions.
* :mod:`stc.pruner.anchors` — :class:`AnchorMemory` and ``spatial_anchor``.
* :mod:`stc.pruner.index_mapper` — map per-frame token indices back into
  flattened LLaVA-style token streams (with optional row markers).
* :mod:`stc.pruner.specs` — :data:`MODEL_SPECS` describing supported layouts.
"""

from stc.pruner.anchors import AnchorMemory, spatial_anchor
from stc.pruner.index_mapper import IndexMapper
from stc.pruner.pruner import STCPruner
from stc.pruner.scoring import ScoreCalculator, dual_anchor_scores
from stc.pruner.specs import MODEL_SPECS, ModelSpec

__all__ = [
    "AnchorMemory",
    "IndexMapper",
    "MODEL_SPECS",
    "ModelSpec",
    "ScoreCalculator",
    "STCPruner",
    "dual_anchor_scores",
    "spatial_anchor",
]
