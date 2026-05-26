"""Scoring functions for STC-Pruner."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


class ScoreCalculator:
    """Gaussian-similarity scorer matching the original STC release.

    The pruner keeps tokens with the highest combined frame+memory similarity,
    so the kept tokens are those that best summarise both the current frame
    and the running historical anchor.
    """

    @staticmethod
    def gaussian_similarity(
        features: torch.Tensor,
        target: torch.Tensor,
        alphas: Optional[Iterable[float]] = None,
    ) -> torch.Tensor:
        if alphas is None:
            alphas = (2**k for k in range(-3, 2))
        diff = features - target
        l2_dist_sq = torch.sum(diff**2, dim=-1)
        return sum(torch.exp(-l2_dist_sq / (2 * alpha)) for alpha in alphas)

    @staticmethod
    def compute_scores(
        reshaped_features: torch.Tensor,
        memory_mean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features_norm = F.normalize(reshaped_features, dim=-1)

        frame_means = features_norm.mean(dim=1, keepdim=True)
        frame_scores = ScoreCalculator.gaussian_similarity(features_norm, frame_means)

        video_mean = features_norm.mean(dim=(0, 1), keepdim=True)
        video_scores = ScoreCalculator.gaussian_similarity(features_norm, video_mean)

        memory_mean_norm = F.normalize(memory_mean, dim=-1).view(1, 1, -1)
        memory_scores = ScoreCalculator.gaussian_similarity(
            features_norm, memory_mean_norm
        )
        return frame_scores, video_scores, memory_scores


def dual_anchor_scores(
    features: torch.Tensor,
    temporal_anchor: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute paper-style SCA/TCA novelty scores.

    Higher scores are more novel and should be retained.  ``alpha`` blends
    the temporal-context anchor (history) with the spatial-content anchor
    (current frame mean).
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    features_norm = F.normalize(features, dim=-1)
    spatial = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
    temporal = F.normalize(temporal_anchor, dim=-1).view(1, 1, -1)

    spatial_distance = 1.0 - F.cosine_similarity(features_norm, spatial, dim=-1)
    temporal_distance = 1.0 - F.cosine_similarity(features_norm, temporal, dim=-1)
    return alpha * temporal_distance + (1.0 - alpha) * spatial_distance
