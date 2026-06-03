# STC: Streaming Token Compression

Standalone implementation of **STC-Cacher** (vision encoder selective
recomputation) and **STC-Pruner** (query-agnostic visual token pruning),
factored out of the original ReKV codebase so it can be reused across video
LLMs.

## Layout

```
STC_new/
├── stc/                Python package
│   ├── config.py           Dataclasses + process-wide default config
│   ├── core/               Cross-cutting algorithms
│   │   ├── selectors.py        Token similarity / dynamic-token selection
│   │   └── layer_ratio.py      Per-layer skip-ratio allocator
│   ├── cacher/             STC-Cacher
│   │   ├── state.py            STCCache (per-stream state, no singleton)
│   │   └── reference_forward.py  Selective ViT-layer forwards
│   ├── pruner/             STC-Pruner
│   │   ├── pruner.py           STCPruner orchestrator
│   │   ├── scoring.py          Gaussian / dual-anchor score functions
│   │   ├── anchors.py          Temporal/spatial anchors
│   │   ├── index_mapper.py     Per-frame -> flat token index mapping
│   │   └── specs.py            MODEL_SPECS for supported layouts
│   ├── integrations/
│   │   └── hf_vit.py           register_stc_cacher() for HF CLIP / SigLIP
│   └── utils/distributed.py
├── models/             Vendored research checkpoints + entrypoints
│   ├── rekv/               ReKV streaming inference (consumer of stc)
│   ├── livecc/             LiveCC
│   ├── Dispider/
│   ├── StreamForest/
│   └── reproduce_smoke.py
├── benchmarks/         Offline / OvoBench / StreamingBench data
├── scripts/eval_rekv/  Eval shell drivers
└── results/            Run outputs
```

## Install

```bash
pip install -e .            # core only
pip install -e .[hf]        # add transformers for HF integrations
```

## Quick start

```python
import stc

cache  = stc.STCCache()
config = stc.STCConfig()                       # or stc.default_config()
stc.register_stc_cacher(
    vision_tower,
    kind="siglip",                             # or "clip"
    cache=cache,
    config=config.cache,
)

pruner = stc.STCPruner(config.model)
out    = pruner.compress(features, model="llava_ov")

# Advance per-chunk state during streaming inference
cache.reset_for_chunk(chunk_idx=1, update_token_ratio=0.25)
```

Legacy entrypoints (`GlobalConfig.initialize_from_args(args)`,
`get_config()`, `default_cache()`, `reset_default_cache(...)`) remain
available for the existing ReKV scripts under `models/rekv/`.

## What changed (vs. original ReKV-bundled STC code)

- `stc/cacher.py`, `stc/prune.py`, `stc/custom_siglip.py` legacy alias
  modules removed.
- `stc/cache/` renamed `stc/cacher/`. `STC_CACHE` (singleton) renamed
  `STCCache` (regular class) — global default still reachable via
  `stc.default_cache()`.
- `stc.integrations.{clip,siglip}` merged into a single
  `stc.integrations.hf_vit.register_stc_cacher(..., kind=...)` helper.
- `stc.integrations.{rekv,livecc,dispider,streamforest}` empty namespace
  packages removed.
- Selective-forward hot path no longer reads the global config / cache;
  `register_stc_cacher` binds the active state to each layer instead.
- `STC_Pruner` alias dropped (use `STCPruner`); duplicate
  `register_cache_by_key_*` casing variants dropped (use
  `register_stc_cacher`).
- Pruner-internal modules (anchors / scoring / index_mapper / specs)
  moved out of `stc.core` into `stc.pruner` so the dependency direction
  matches actual usage.
