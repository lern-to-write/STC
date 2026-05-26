"""Map per-frame token indices back to flattened VideoLLM token layouts."""

from __future__ import annotations

from typing import Sequence

import torch

from stc.pruner.specs import ModelSpec


class IndexMapper:
    @staticmethod
    def map_indices(
        model_spec: ModelSpec,
        local_indices: Sequence[torch.Tensor],
        device: torch.device,
        original_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del original_features
        if model_spec.index_mapper_type == "flat":
            return IndexMapper._map_flat(
                local_indices, model_spec.tokens_per_frame, device
            )
        if model_spec.index_mapper_type == "grid_13x13":
            return IndexMapper._map_grid(local_indices, 13, device)
        raise NotImplementedError(
            f"Mapper {model_spec.index_mapper_type} not implemented"
        )

    @staticmethod
    def _map_flat(
        indices_list: Sequence[torch.Tensor],
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = torch.arange(len(indices_list), device=device).unsqueeze(1)
        offsets = offsets * tokens_per_frame
        return torch.cat(
            [indices + offset for indices, offset in zip(indices_list, offsets)]
        )

    @staticmethod
    def _map_grid(
        indices_list: Sequence[torch.Tensor],
        size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Map a 13x13 grid into layouts with row marker tokens.

        LLaVA-Video style features insert one marker token after each grid
        row.  The pruner selects patch tokens on a 13x13 grid and this mapper
        preserves those row markers when expanding to the full token stream.
        """

        height, width = size, size
        width_with_marker = width + 1

        all_global_indices: list[torch.Tensor] = []
        for frame_idx, local_idx in enumerate(indices_list):
            rows = torch.div(local_idx, width, rounding_mode="floor")
            cols = local_idx % width
            frame_start = frame_idx * (height * width_with_marker)
            patch_indices = frame_start + (rows * width_with_marker + cols)
            all_global_indices.append(patch_indices)

            row_markers = torch.arange(height, device=device) * width_with_marker + width
            all_global_indices.append(frame_start + row_markers)

        return torch.cat(all_global_indices, dim=0)
