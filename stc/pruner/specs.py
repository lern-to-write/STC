"""Model-specific token layout metadata used by STC-Pruner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    tokens_per_frame: int
    index_mapper_type: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "llava_ov": ModelSpec(tokens_per_frame=196, index_mapper_type="flat"),
    "llava_vid": ModelSpec(tokens_per_frame=169, index_mapper_type="grid_13x13"),
    "clip": ModelSpec(tokens_per_frame=144, index_mapper_type="flat"),
}

