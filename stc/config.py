"""Configuration dataclasses for Streaming Token Compression.

The cacher and pruner each have a small dataclass of options.  ``STCConfig``
bundles them and provides:

* a process-wide default singleton (``default_config()`` / ``GlobalConfig`` –
  the latter is a backward-compatible alias for entrypoints that already
  call ``GlobalConfig.initialize_from_args(args)``);
* an :py:meth:`STCConfig.initialize_from_env` classmethod that maps ``STC_*``
  environment variables into the dataclasses for the ReKV entrypoints;
* an :py:meth:`STCConfig.initialize_from_args` classmethod kept for older
  callers that still pass argparse attributes.

Internal hot-path code does **not** read this singleton.  ``register_stc_cacher``
and :class:`STCPruner` accept explicit ``CacheConfig`` / ``ModelConfig``
objects and only fall back to the default when the caller passes nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Mapping, Optional

CacheStrategy = Literal["none", "selective"]
PruneStrategy = Literal["gaussian", "dual_anchor"]
SelectorMetric = Literal["cosine", "l1", "l2", "dot"]


# Strategies recognised on input but normalised away.  ``cacher`` was the
# legacy name for ``selective``; ``full_tokens`` was a no-op alias for
# ``gaussian`` that the pruner silently rewrote.
_CACHE_STRATEGY_ALIASES = {"cacher": "selective"}
_PRUNE_STRATEGY_ALIASES = {"full_tokens": "gaussian"}

_TRUE_ENV_VALUES = {"1", "true", "yes", "on", "selective", "cacher"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off", "none"}


def _coerce_env_value(env_name: str, raw_value: str, default: Any) -> Any:
    value = raw_value.strip()
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a float, got {raw_value!r}") from exc
    return value.lower()


def env_flag(
    name: str,
    default: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Read a boolean-like environment flag."""

    source = os.environ if env is None else env
    raw_value = source.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in _TRUE_ENV_VALUES:
        return True
    if value in _FALSE_ENV_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, on/off, "
        f"selective/cacher, or none; got {raw_value!r}"
    )


def stc_patch_vision_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether ReKV should patch the HF vision tower with STC-Cacher."""

    return env_flag("STC_PATCH_VISION", default=False, env=env)


_ENV_DEFAULTS: dict[str, dict[str, Any]] = {
    "cache": {
        "update_token_ratio": 1.0,
        "cache_interval": 2,
    },
    "model": {
        "token_per_frame": 196,
    },
}

_ENV_NAMES: dict[tuple[str, str], str] = {
    ("cache", "update_token_ratio"): "STC_UPDATE_TOKEN_RATIO",
    ("cache", "cache_interval"): "STC_CACHE_INTERVAL",
    ("model", "token_per_frame"): "STC_TOKEN_PER_FRAME",
}


@dataclass
class CacheConfig:
    """Options used by STC-Cacher during vision encoding."""

    strategy: CacheStrategy = "selective"
    update_token_ratio: float = 0.25
    cache_interval: int = 2
    selector_metric: SelectorMetric = "cosine"
    cuda_graph: bool = False
    share_selection: bool = False

    def __post_init__(self) -> None:
        if self.strategy in _CACHE_STRATEGY_ALIASES:
            self.strategy = _CACHE_STRATEGY_ALIASES[self.strategy]  # type: ignore[assignment]
        if self.strategy not in ("none", "selective"):
            raise ValueError(f"Unknown cache strategy: {self.strategy}")
        if self.selector_metric not in ("cosine", "l1", "l2", "dot"):
            raise ValueError(f"Unknown selector_metric: {self.selector_metric}")
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

    @classmethod
    def initialize_from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
    ) -> "STCConfig":
        """Update the default config from ``STC_*`` environment variables.

        Missing variables use the ReKV baseline defaults: no vision-tower
        caching and full per-frame token retention.
        """

        source = os.environ if env is None else env
        instance = cls.get_instance()
        instance.cache.strategy = (
            "selective" if stc_patch_vision_enabled(source) else "none"
        )
        instance.cache.selector_metric = "cosine"
        instance.cache.cuda_graph = env_flag("STC_CUDA_GRAPH", default=False, env=source)
        instance.cache.share_selection = env_flag(
            "STC_SHARE_SELECTION", default=False, env=source
        )
        instance.model.prune_strategy = "gaussian"
        instance.model.encode_chunk_size = 1
        instance.model.channel_keep_ratio = 0.5
        instance.model.spatial_temporal_alpha = 0.5
        for section_name, defaults in _ENV_DEFAULTS.items():
            section = getattr(instance, section_name)
            for field_name, default_value in defaults.items():
                env_name = _ENV_NAMES[(section_name, field_name)]
                raw_value = source.get(env_name)
                value = (
                    default_value
                    if raw_value is None
                    else _coerce_env_value(env_name, raw_value, default_value)
                )
                setattr(section, field_name, value)
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
                "cuda_graph": self.cache.cuda_graph,
                "share_selection": self.cache.share_selection,
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
