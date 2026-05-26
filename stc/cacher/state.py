"""Runtime state for STC-Cacher.

A single ``STCCache`` instance corresponds to one streaming inference session.
It tracks which chunk we are encoding so the layer forwards know whether to
do a full recompute (refreshing the reference frame) or a selective update.

This class is **not** a singleton.  Use :func:`stc.default_cache` if you want
the legacy process-wide instance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


class STCCache:
    """Per-stream cache state used by the selective-recompute forwards."""

    def __init__(
        self,
        chunk_idx: int = 1,
        update_token_ratio: float = 0.25,
    ) -> None:
        if chunk_idx < 0:
            raise ValueError("chunk_idx must be >= 0")
        if not 0.0 < update_token_ratio <= 1.0:
            raise ValueError("update_token_ratio must be in (0, 1]")
        self.chunk_idx: int = chunk_idx
        self.update_token_ratio: float = update_token_ratio
        self.acc_time: int = 0
        self.max_mem: int = 0
        self._features: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Per-chunk lifecycle
    # ------------------------------------------------------------------

    def reset_for_chunk(
        self,
        chunk_idx: int,
        update_token_ratio: float | None = None,
    ) -> "STCCache":
        """Advance to the next streaming chunk.

        ``update_token_ratio`` can be overridden per-chunk; otherwise the
        previously configured value is kept.
        """

        if chunk_idx < 0:
            raise ValueError("chunk_idx must be >= 0")
        self.chunk_idx = chunk_idx
        if update_token_ratio is not None:
            if not 0.0 < update_token_ratio <= 1.0:
                raise ValueError("update_token_ratio must be in (0, 1]")
            self.update_token_ratio = update_token_ratio
        return self

    def clear(self) -> None:
        """Drop layer feature caches and free CUDA memory."""

        self._features.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Optional per-layer feature scratch space
    # ------------------------------------------------------------------

    def set_feature(self, layer_id: int, name: str, value: torch.Tensor) -> None:
        self._features[layer_id][name] = value

    def get_feature(self, layer_id: int, name: str) -> torch.Tensor:
        return self._features[layer_id][name]

    def has_feature(self, layer_id: int, name: str) -> bool:
        return name in self._features.get(layer_id, {})

    def __repr__(self) -> str:
        return (
            f"STCCache(chunk_idx={self.chunk_idx}, "
            f"update_token_ratio={self.update_token_ratio})"
        )


# ----------------------------------------------------------------------
# Process-wide default (legacy compatibility)
# ----------------------------------------------------------------------

_DEFAULT_CACHE: STCCache | None = None


def default_cache() -> STCCache:
    """Return the lazy module-level :class:`STCCache` used by legacy code."""

    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = STCCache()
    return _DEFAULT_CACHE


def set_default_cache(cache: STCCache | None) -> None:
    """Override or clear the process-wide default cache (mostly for tests)."""

    global _DEFAULT_CACHE
    _DEFAULT_CACHE = cache


def reset_default_cache(
    chunk_idx: int = 1,
    update_token_ratio: float = 0.25,
) -> STCCache:
    """Reset the legacy default cache to a fresh instance.

    This is what the old ``STC_CACHE.new_instance(chunk_idx, update_token_ratio)``
    call did.  Returned for parity with that API.
    """

    cache = default_cache()
    cache.reset_for_chunk(chunk_idx, update_token_ratio)
    cache.clear()
    return cache
