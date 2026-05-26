"""Token selection utilities used by STC-Cacher.

Given a ``current`` tensor of shape ``[frames, tokens, channels]`` and a
``reference`` of shape ``[tokens, channels]``, ``select_dynamic_token_indices``
returns the indices of the most "dynamic" tokens per frame, i.e. the tokens
least similar to the reference frame.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

Metric = Literal["cosine", "l1", "l2", "dot"]


def token_similarity(
    current: torch.Tensor,
    reference: torch.Tensor,
    metric: Metric = "cosine",
) -> torch.Tensor:
    """Score ``current`` tokens against a single ``reference`` frame.

    For distance metrics (``l1``, ``l2``) the negated distance is returned so
    that *lower* scores always indicate "more dynamic" tokens regardless of
    metric.
    """

    if current.ndim != 3:
        raise ValueError("current must have shape [frames, tokens, channels]")
    if reference.ndim != 2:
        raise ValueError("reference must have shape [tokens, channels]")
    if current.shape[1:] != reference.shape:
        raise ValueError(
            "current token/channel dimensions must match reference dimensions"
        )

    ref = reference.unsqueeze(0)
    if metric == "cosine":
        return F.cosine_similarity(current, ref, dim=-1)
    if metric == "dot":
        return (current * ref).sum(dim=-1)
    if metric == "l1":
        return -(current - ref).abs().sum(dim=-1)
    if metric == "l2":
        return -torch.linalg.vector_norm(current - ref, ord=2, dim=-1)
    raise ValueError(f"Unknown selector metric: {metric}")


def select_dynamic_token_indices(
    current: torch.Tensor,
    reference: torch.Tensor,
    update_ratio: float,
    metric: Metric = "cosine",
) -> torch.Tensor:
    """Pick the ``update_ratio`` fraction of most-dynamic tokens per frame."""

    if not 0.0 < update_ratio <= 1.0:
        raise ValueError("update_ratio must be in (0, 1]")

    scores = token_similarity(current, reference, metric=metric)
    token_count = scores.shape[1]
    keep_count = max(1, min(int(token_count * update_ratio), token_count))
    return torch.topk(scores, k=keep_count, dim=1, largest=False).indices
