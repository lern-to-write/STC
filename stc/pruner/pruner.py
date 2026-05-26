"""STC-Pruner: query-agnostic visual token pruning."""

from __future__ import annotations

from typing import Optional

import torch

from stc.config import ModelConfig, PruneStrategy, default_config
from stc.pruner.index_mapper import IndexMapper
from stc.pruner.scoring import ScoreCalculator, dual_anchor_scores
from stc.pruner.specs import MODEL_SPECS


class STCPruner:
    """Compress visual tokens before LLM prefilling.

    The default ``gaussian`` strategy preserves the behavior of the released
    STC code.  ``dual_anchor`` follows the paper pseudocode more directly
    by retaining tokens that are dissimilar to both the historical temporal
    anchor and the current spatial anchor.

    Configuration is taken from the explicit ``config`` argument; if not
    provided, the process-wide :func:`stc.default_config` is used.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self._config = config
        self.past_memory_mean_token: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        return self._config if self._config is not None else default_config().model

    def reset(self) -> None:
        self.past_memory_mean_token = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_memory(self, current_features: torch.Tensor) -> torch.Tensor:
        current_chunk_mean = current_features.mean(dim=(0, 1), keepdim=True)
        self.past_memory_mean_token.append(current_chunk_mean.detach())
        return torch.mean(torch.cat(self.past_memory_mean_token, dim=0), dim=0)

    def _select_feature_channels(
        self,
        tensor: torch.Tensor,
        keep_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        ratio = keep_ratio if keep_ratio is not None else self.config.channel_keep_ratio
        if not 0.0 < ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")

        channel_var = tensor.var(dim=0, unbiased=False)
        kept_channels = max(1, int(channel_var.shape[0] * ratio))
        _, indices = torch.topk(channel_var, k=kept_channels, largest=False)
        return tensor[:, indices]

    def _resolve_token_budget(
        self,
        configured_tokens: Optional[int],
        source_tokens_per_frame: int,
    ) -> int:
        budget = configured_tokens if configured_tokens is not None else self.config.token_per_frame
        return max(1, min(int(budget), source_tokens_per_frame))

    def _score(
        self,
        reshaped_features: torch.Tensor,
        memory_mean_token: torch.Tensor,
        strategy: PruneStrategy,
    ) -> tuple[torch.Tensor, bool]:
        """Return ``(scores, largest)`` where ``largest`` is the topk argument
        used to keep the highest-scoring tokens.
        """

        if strategy == "gaussian":
            frame_score, _, memory_score = ScoreCalculator.compute_scores(
                reshaped_features, memory_mean_token
            )
            return memory_score + frame_score, False
        if strategy == "dual_anchor":
            return (
                dual_anchor_scores(
                    reshaped_features,
                    memory_mean_token,
                    self.config.spatial_temporal_alpha,
                ),
                True,
            )
        raise ValueError(f"Unknown pruner score strategy: {strategy}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        flattened_features: torch.Tensor,
        model: str = "llava_ov",
        raw_image_features: Optional[torch.Tensor] = None,
        tokens_per_frame: Optional[int] = None,
        score_strategy: Optional[PruneStrategy] = None,
    ) -> torch.Tensor:
        """Return a compressed visual token sequence.

        Args:
            flattened_features: Tensor shaped ``[tokens, channels]``.
            model: Key in ``MODEL_SPECS`` describing the source token layout.
            raw_image_features: Original features used by ``llava_vid``-style
                layouts that include row marker tokens.  Required for that
                layout.
            tokens_per_frame: Override the output budget for this call.
            score_strategy: Override the configured scoring strategy for
                this call.
        """

        if model not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model}")
        if flattened_features.ndim != 2:
            raise ValueError("flattened_features must have shape [tokens, channels]")

        spec = MODEL_SPECS[model]
        if model == "llava_vid" and raw_image_features is None:
            raise ValueError("llava_vid requires raw_image_features")
        if flattened_features.shape[0] % spec.tokens_per_frame != 0:
            raise ValueError(
                "flattened_features token count must be divisible by "
                f"{spec.tokens_per_frame} for {model}"
            )

        selected_features = self._select_feature_channels(flattened_features)
        num_frames = selected_features.shape[0] // spec.tokens_per_frame
        reshaped_features = selected_features.view(
            num_frames, spec.tokens_per_frame, -1
        )

        memory_mean_token = self._update_memory(reshaped_features)
        strategy: PruneStrategy = score_strategy or self.config.prune_strategy
        scores, largest = self._score(reshaped_features, memory_mean_token, strategy)

        token_budget = self._resolve_token_budget(tokens_per_frame, spec.tokens_per_frame)
        kept_indices_list: list[torch.Tensor] = []
        for frame_idx in range(num_frames):
            _, indices = torch.topk(
                scores[frame_idx],
                k=token_budget,
                largest=largest,
            )
            kept_indices_list.append(indices.sort().values)

        final_indices = IndexMapper.map_indices(
            spec,
            kept_indices_list,
            flattened_features.device,
            flattened_features,
        )
        if model == "llava_vid":
            return raw_image_features[final_indices]
        return flattened_features[final_indices]
