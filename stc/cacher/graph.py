"""CUDA-graph runner for the STC-Cacher selective vision-tower path.

A streaming frame is either a *refresh* frame (``chunk_idx % cache_interval ==
0``; runs the full eager forward and updates references in place) or a
*selective* frame (reuses references — identical op graph every frame).  The
selective path is made of many small ops (token selection, gather/scatter,
sliced projections); on fast GPUs the per-frame allocation + kernel-launch
overhead can exceed the FLOPs it saves, making it slower than a dense forward.

Capturing the all-selective forward once and replaying a CUDA graph removes
that overhead (allocate + launch once, replay with no host work), exposing the
real FLOP savings.  Requires references stored *in place* (see
``_store_reference`` in ``reference_forward``) so a refresh updates the same
addresses the captured graph reads.

Always on for the cacher (``CacheConfig.cuda_graph``, default ``True``; not
user-configurable).  On any capture failure it disables itself and falls back to
eager selective, so it can never make a run fail.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

import torch

from stc.cacher.state import STCCache
from stc.config import CacheConfig

logger = logging.getLogger(__name__)


class SelectiveCUDAGraphRunner:
    """Replace a vision tower's ``forward`` to replay a captured graph on
    selective frames and run eager on refresh frames."""

    def __init__(
        self,
        orig_forward: Callable[..., Any],
        cache: STCCache,
        config: CacheConfig,
        warmup_iters: int = 3,
    ) -> None:
        self.orig_forward = orig_forward
        self.cache = cache
        self.config = config
        self.warmup_iters = warmup_iters
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_in: Optional[torch.Tensor] = None
        self.static_out: Any = None
        self.captured_key: Optional[Tuple] = None
        self.disabled = False

    # ------------------------------------------------------------------
    def _is_refresh(self) -> bool:
        if self.config.strategy == "none":
            return True
        interval = max(1, int(getattr(self.config, "cache_interval", 1)))
        chunk_idx = int(getattr(self.cache, "chunk_idx", 1))
        return chunk_idx % interval == 0

    def __call__(self, pixel_values, *args, **kwargs):
        # 只对 selective 帧、且无额外位置参数的 CUDA 张量走图；其余一律 eager。
        if (
            self.disabled
            or args
            or not isinstance(pixel_values, torch.Tensor)
            or not pixel_values.is_cuda
            or self._is_refresh()
        ):
            return self.orig_forward(pixel_values, **kwargs)

        key = (tuple(pixel_values.shape), pixel_values.dtype)
        if self.graph is None or self.captured_key != key:
            if not self._capture(pixel_values, kwargs, key):
                return self.orig_forward(pixel_values, **kwargs)

        self.static_in.copy_(pixel_values)
        self.graph.replay()
        return self.static_out

    # ------------------------------------------------------------------
    def _capture(self, pixel_values, kwargs, key) -> bool:
        try:
            self.static_in = pixel_values.clone()
            # 侧流预热：跑通 cudnn 选择 / 填充 allocator，再正式捕获。
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.warmup_iters):
                    self.orig_forward(self.static_in, **kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.static_out = self.orig_forward(self.static_in, **kwargs)
            self.graph = graph
            self.captured_key = key
            logger.info(
                "STC-Cacher: captured selective CUDA graph for shape=%s dtype=%s",
                key[0], key[1],
            )
            return True
        except Exception as exc:  # noqa: BLE001 — never let capture break a run
            logger.warning(
                "STC-Cacher: CUDA graph capture failed (%s: %s); "
                "falling back to eager selective.",
                type(exc).__name__, exc,
            )
            self.graph = None
            self.static_in = None
            self.static_out = None
            self.disabled = True
            return False
