# Speed Benchmark

Runtime-instrumented latency benchmarks for STC. No changes to model code. One
script set per framework.

```
speed_benchmark/
├── benchmark_rekv.py         / run_rekv.sh          ReKV: ViT encoding + LLM pre-fill
├── benchmark_streamforest.py / run_streamforest.sh  StreamForest (SigLIP): ViT encoding
└── benchmark_dispider.py     / run_dispider.sh      Dispider (CLIP): ViT encoding
```

ReKV applies both STC components, so its benchmark times ViT encoding **and** LLM
pre-fill. StreamForest and Dispider are cacher-only, so theirs time ViT encoding.
Each `run_*.sh` runs the baseline and +STC configurations and prints the
reduction; STC is toggled via `STC_PATCH_VISION`.

## Usage

```bash
# ReKV (default 16 frames)
GPU=0 bash speed_benchmark/run_rekv.sh

# StreamForest (SigLIP)
GPU=0 bash speed_benchmark/run_streamforest.sh

# Dispider (CLIP) — point CLIP_CKPT_PATH at your checkpoint if not in the HF cache
GPU=0 CLIP_CKPT_PATH=/path/to/clip-vit-large-patch14 bash speed_benchmark/run_dispider.sh
```

Pass a mode to run a single configuration: `rekv`/`rekv_stc`, `sf`/`sf_stc`,
`dispider`/`dispider_stc`.

## Notes

- Times stages with CUDA events. Pin a dedicated GPU (`GPU=...`) and read
  **min / median** over repeats: clock scaling only adds time, so the minimum
  approximates the uncontended floor. Absolute latency is GPU-dependent; the
  **reduction ratio** is what reproduces.
- Weights do not affect latency, so the SigLIP / CLIP towers load from public
  checkpoints (`google/siglip-so400m-patch14-384`, `openai/clip-vit-large-patch14`)
  — no full model needed.
- CUDA-graph replay and per-frame shared token selection are always on in `stc`
  (not user-configurable); they make selective recompute faster than dense
  encoding and fall back to eager if graph capture is unsupported.

## Environment knobs

`STC_PY` (interpreter), `HF_HOME`, `STC_EXTRA_SITE` (extra `PYTHONPATH` entry),
`GPU`, `NUM_FRAMES`, `REPEATS`, `WARMUP`, `CLIP_CKPT_PATH` / `SIGLIP_CKPT`.
