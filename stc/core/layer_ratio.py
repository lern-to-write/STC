"""Per-layer skip-ratio allocation for cacher variants.

Extracted from the longva ``custom_clip.py`` reimplementation.  Some
STC-Cacher variants spend a different fraction of compute on each layer of
the vision tower (deeper layers tend to recompute fewer tokens).  This
helper centralises that allocation so any future cacher backend can opt in.
"""

from __future__ import annotations

from typing import Literal

Strategy = Literal["uniform", "linear_increasing"]


class LayerRatioAllocator:
    """Compute per-layer skip ratios that average to ``target_ratio``.

    ``uniform`` returns ``target_ratio`` for every layer.  ``linear_increasing``
    grows the ratio linearly from 0.2*target at layer 0 to 1.8*target at the
    deepest layer, then renormalises so the mean equals ``target_ratio``.
    """

    def __init__(
        self,
        num_layers: int,
        target_ratio: float = 0.3,
        strategy: Strategy = "uniform",
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= target_ratio <= 1.0:
            raise ValueError("target_ratio must be in [0, 1]")
        self.num_layers = num_layers
        self.target_ratio = float(target_ratio)
        self.strategy = strategy
        self.layer_ratios = self._build_ratios()

    def _build_ratios(self) -> list[float]:
        if self.strategy == "uniform":
            return [self.target_ratio] * self.num_layers
        if self.strategy == "linear_increasing":
            denom = max(self.num_layers - 1, 1)
            ratios = [
                self.target_ratio * (0.2 + 1.6 * (i / denom))
                for i in range(self.num_layers)
            ]
            mean = sum(ratios) / len(ratios)
            if mean > 0:
                ratios = [r * (self.target_ratio / mean) for r in ratios]
            return ratios
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def get_layer_ratio(self, layer_idx: int) -> float:
        if layer_idx < 0:
            raise ValueError("layer_idx must be >= 0")
        if layer_idx >= self.num_layers:
            return self.target_ratio
        return self.layer_ratios[layer_idx]
