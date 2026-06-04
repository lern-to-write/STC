# Dispider evaluation

Self-contained OVO-Bench driver for Dispider (CLIP backbone) with **STC-Cacher**
toggled on/off by a mode argument. STC on CLIP is **cacher-only** (no pruner);
CUDA-graph replay + per-frame shared token selection are always on inside `stc`.
The repo root is added to `PYTHONPATH` so `import stc` works.

```bash
export MODEL_PATH=/path/Dispider
export CLIP_CKPT_PATH=/path/clip-vit-large-patch14
export ANNO_PATH=/path/ovo_bench_new.json
export CHUNKED_DIR=/path/chunked_videos

# baseline vs + STC-Cacher
bash scripts/eval_dispider/eval_dispider_ovobench.sh dispider
bash scripts/eval_dispider/eval_dispider_ovobench.sh dispider_stc

# 1-sample smoke
MAX_SAMPLES=1 TASKS=EPM bash scripts/eval_dispider/eval_dispider_ovobench.sh dispider_stc

# multi-GPU
NUM_GPUS=8 NUM_CHUNKS=8 bash scripts/eval_dispider/eval_dispider_ovobench.sh dispider_stc
```

Honored env: `MODEL_PATH`, `CLIP_CKPT_PATH`, `ANNO_PATH`, `CHUNKED_DIR`,
`RESULT_DIR`, `NUM_GPUS`, `NUM_CHUNKS`, `MAX_SAMPLES`, `TASKS`, `RUN_SCORE`,
`DISPIDER_ENV`, `CUDA_HOME`.
