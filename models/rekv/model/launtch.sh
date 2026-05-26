#!/bin/bash
set -euo pipefail

REKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REKV_ROOT/../.." && pwd)"
export PYTHONPATH="$REKV_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

python -m model.video_qa.run_eval \
    --num_chunks 1 \
    --model llava_ov_7b \
    --dataset qaego4d \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64




EXPERIMENT=tome python -m model.video_qa.run_eval \
    --num_chunks 1 \
    --model llava_ov_7b \
    --dataset egoschema \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64



python -m model.video_qa.run_eval \
    --num_chunks 1 \
    --model llava_ov_7b \
    --dataset videomme \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64 \
    --prune_strategy "vidcom2_orignal" \
    --token_per_frame 49 
