"""Small distributed helpers that are safe outside torch.distributed."""

from __future__ import annotations

import torch.distributed as dist


def get_rank(default: int = 0) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return default


def is_main_process() -> bool:
    return get_rank() == 0

