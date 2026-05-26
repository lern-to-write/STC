"""Temporal and spatial anchors for streaming token pruning."""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch


class AnchorMemory:
    """Maintain historical chunk means for the temporal context anchor."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self.max_size = max_size
        self._items: deque[torch.Tensor] = deque(maxlen=max_size)

    def reset(self) -> None:
        self._items.clear()

    def append(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=(0, 1), keepdim=True)
        self._items.append(mean.detach())
        return self.temporal_anchor()

    def temporal_anchor(self) -> torch.Tensor:
        if not self._items:
            raise RuntimeError("AnchorMemory is empty")
        return torch.mean(torch.cat(list(self._items), dim=0), dim=0)

    @property
    def items(self) -> list[torch.Tensor]:
        return list(self._items)


def spatial_anchor(features: torch.Tensor) -> torch.Tensor:
    """Return the current-frame/chunk anchor."""

    return features.mean(dim=1, keepdim=True)
