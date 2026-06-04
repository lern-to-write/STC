# StreamForest evaluation

Self-contained `lmms_eval` driver for StreamForest (SigLIP backbone) with
**STC-Cacher** toggled on/off by a mode argument. STC on SigLIP is **cacher-only**
(no pruner); CUDA-graph replay + per-frame shared token selection are always on
inside `stc`. The script sources StreamForest's env-config (`streamforest_env.sh`,
which also puts the repo root on `PYTHONPATH`) and runs `accelerate ... lmms_eval`
directly — one task per loop iteration.

```bash
# baseline vs + STC-Cacher
TASKS=ovobench bash scripts/eval_streamforest/eval_streamforest.sh sf
TASKS=ovobench bash scripts/eval_streamforest/eval_streamforest.sh sf_stc

# 1-sample smoke
LIMIT=1 TASKS=ovobench bash scripts/eval_streamforest/eval_streamforest.sh sf_stc

# multiple tasks, multi-GPU
TASKS="streamingbench,videomme" NUM_GPUS=8 bash scripts/eval_streamforest/eval_streamforest.sh sf_stc
```

Honored env: `TASKS` (default = StreamForest's 8-benchmark set), `LIMIT`,
`NUM_GPUS`, `MAX_FRAMES`, `TIME_MSG`, `BATCH_SIZE`, `STREAMFOREST_CKPT_PATH`,
and everything in `models/StreamForest/scripts/env/streamforest_env.sh`.
