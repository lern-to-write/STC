"""Configuration dataclasses for Streaming Token Compression.

The cacher and pruner each have a small dataclass of options.  ``STCConfig``
bundles them and provides:

* a process-wide default singleton (``default_config()`` / ``GlobalConfig`` –
  the latter is a backward-compatible alias for entrypoints that already
  call ``GlobalConfig.initialize_from_args(args)``);
* an :py:meth:`STCConfig.initialize_from_args` classmethod that maps argparse
  attributes into the dataclasses, accepting both bare names
  (``token_per_frame``) and namespaced ones (``model_token_per_frame``).

Internal hot-path code does **not** read this singleton.  ``register_stc_cacher``
and :class:`STCPruner` accept explicit ``CacheConfig`` / ``ModelConfig``
objects and only fall back to the default when the caller passes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional

CacheStrategy = Literal["none", "selective"]
PruneStrategy = Literal["gaussian", "dual_anchor"]
SelectorMetric = Literal["cosine", "l1", "l2", "dot"]


# Strategies recognised on input but normalised away.  ``cacher`` was the
# legacy name for ``selective``; ``full_tokens`` was a no-op alias for
# ``gaussian`` that the pruner silently rewrote.
_CACHE_STRATEGY_ALIASES = {"cacher": "selective"}
_PRUNE_STRATEGY_ALIASES = {"full_tokens": "gaussian"}


@dataclass
class CacheConfig:
    """Options used by STC-Cacher during vision encoding."""

    strategy: CacheStrategy = "selective"
    update_token_ratio: float = 0.25
    cache_interval: int = 2
    selector_metric: SelectorMetric = "cosine"

    def __post_init__(self) -> None:
        if self.strategy in _CACHE_STRATEGY_ALIASES:
            self.strategy = _CACHE_STRATEGY_ALIASES[self.strategy]  # type: ignore[assignment]
        if self.strategy not in ("none", "selective"):
            raise ValueError(f"Unknown cache strategy: {self.strategy}")
        if self.cache_interval < 1:
            raise ValueError("cache_interval must be >= 1")
        if not 0.0 < self.update_token_ratio <= 1.0:
            raise ValueError("update_token_ratio must be in (0, 1]")


@dataclass
class ModelConfig:
    """Options used by STC-Pruner before LLM prefilling."""

    token_per_frame: int = 60
    prune_strategy: PruneStrategy = "gaussian"
    encode_chunk_size: int = 1
    channel_keep_ratio: float = 0.5
    spatial_temporal_alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.prune_strategy in _PRUNE_STRATEGY_ALIASES:
            self.prune_strategy = _PRUNE_STRATEGY_ALIASES[self.prune_strategy]  # type: ignore[assignment]
        if self.prune_strategy not in ("gaussian", "dual_anchor"):
            raise ValueError(f"Unknown prune_strategy: {self.prune_strategy}")
        if self.token_per_frame < 1:
            raise ValueError("token_per_frame must be >= 1")
        if self.encode_chunk_size < 1:
            raise ValueError("encode_chunk_size must be >= 1")
        if not 0.0 < self.channel_keep_ratio <= 1.0:
            raise ValueError("channel_keep_ratio must be in (0, 1]")
        if not 0.0 <= self.spatial_temporal_alpha <= 1.0:
            raise ValueError("spatial_temporal_alpha must be in [0, 1]")


@dataclass
class STCConfig:
    """Aggregate STC configuration (cacher + pruner)."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Process-wide default. Kept on the class for backwards compatibility
    # with the historical ``GlobalConfig.get_instance()`` call style.
    _instance: Optional["STCConfig"] = None

    # ------------------------------------------------------------------
    # Singleton helpers (legacy ergonomics)
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "STCConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> "STCConfig":
        cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize_from_args(cls, args: Any) -> "STCConfig":
        """Update the default config from an argparse-style namespace.

        Both flat names (``token_per_frame``) and namespaced names
        (``model_token_per_frame`` / ``cache_strategy``) are accepted.
        """

        instance = cls.get_instance()
        for section_name in ("cache", "model"):
            section = getattr(instance, section_name)
            for item in fields(section):
                if hasattr(args, item.name):
                    setattr(section, item.name, getattr(args, item.name))
                namespaced = f"{section_name}_{item.name}"
                if hasattr(args, namespaced):
                    setattr(section, item.name, getattr(args, namespaced))
            section.__post_init__()
        return instance

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update(self, **kwargs: Any) -> "STCConfig":
        """Update fields by keyword.  Routes to ``cache`` or ``model`` based on
        which dataclass owns the field."""

        for key, value in kwargs.items():
            if hasattr(self.cache, key):
                setattr(self.cache, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            else:
                raise KeyError(f"Unknown STC config field: {key}")
        self.cache.__post_init__()
        self.model.__post_init__()
        return self

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "cache": {
                "strategy": self.cache.strategy,
                "update_token_ratio": self.cache.update_token_ratio,
                "cache_interval": self.cache.cache_interval,
                "selector_metric": self.cache.selector_metric,
            },
            "model": {
                "token_per_frame": self.model.token_per_frame,
                "prune_strategy": self.model.prune_strategy,
                "encode_chunk_size": self.model.encode_chunk_size,
                "channel_keep_ratio": self.model.channel_keep_ratio,
                "spatial_temporal_alpha": self.model.spatial_temporal_alpha,
            },
        }

    def __str__(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Module-level convenience accessors
# ---------------------------------------------------------------------------


def default_config() -> STCConfig:
    """Return the process-wide :class:`STCConfig` (lazy-initialised)."""

    return STCConfig.get_instance()


def get_config() -> STCConfig:
    """Backward-compatible alias for :func:`default_config`."""

    return default_config()


# Backward-compatible name used by entrypoints.
GlobalConfig = STCConfig
