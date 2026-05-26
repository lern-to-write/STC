#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/models/rekv:$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face}"
cd "$PROJECT_ROOT/models/rekv/model/online_bench_inference/ovobench"
# ReKV 特定配置
MODEL_NAME="rekv"
ANNO_PATH="${ANNO_PATH:-$PROJECT_ROOT/benchmarks/ovobench/ovo_bench_new.json}"
VIDEO_DIR="${VIDEO_DIR:-$HF_HOME/OVO-Bench/src_videos}"
CHUNKED_DIR="${CHUNKED_DIR:-$HF_HOME/OVO-Bench/chunked_videos}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/ovobench_rekv_raw}"
MODE="offline"

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_PROCESSES="${TOTAL_PROCESSES:-8}"
PROCESSES_PER_GPU=$((TOTAL_PROCESSES / NUM_GPUS))

RETRIEVE_SIZE="${RETRIEVE_SIZE:-64}"
CACHE_STRATEGY="${CACHE_STRATEGY:-none}"
PRUNE_STRATEGY="${PRUNE_STRATEGY:-full_tokens}"
TOKEN_PER_FRAME="${TOKEN_PER_FRAME:-196}"
UPDATE_TOKEN_RATIO="${UPDATE_TOKEN_RATIO:-1.0}"

# 任务列表
TASKS="${TASKS:-EPM ASI HLD OCR ACR ATR STU FPD OJR REC SSR CRR}"


echo "=========================================="
echo "Starting ReKV Distributed Inference"
echo "Number of GPUs: $NUM_GPUS"
echo "Processes per GPU: $PROCESSES_PER_GPU"
echo "Total Processes: $TOTAL_PROCESSES"
echo "Retrieve Size: $RETRIEVE_SIZE"
echo "Cache strategy: $CACHE_STRATEGY"
echo "Prune strategy: $PRUNE_STRATEGY"
echo "Token per frame: $TOKEN_PER_FRAME"
echo "=========================================="

for TASK in $TASKS; do
    echo "=========================================="
    echo "Processing task: $TASK"
    echo "=========================================="
    
    # 使用总进程数运行
    python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$TOTAL_PROCESSES \
        inference_distributed.py \
        --model $MODEL_NAME \
        --anno_path $ANNO_PATH \
        --video_dir $VIDEO_DIR \
        --chunked_dir $CHUNKED_DIR \
        --result_dir $RESULT_DIR \
        --mode $MODE \
        --task $TASK \
        --retrieve_size $RETRIEVE_SIZE \
        --cache_strategy "$CACHE_STRATEGY" \
        --prune_strategy "$PRUNE_STRATEGY" \
        --token_per_frame "$TOKEN_PER_FRAME" \
        --update_token_ratio "$UPDATE_TOKEN_RATIO" \
        --save_results True \
        --global_seed 42 \
        --tf32
done

echo "=========================================="
echo "ReKV inference completed!"
echo "Results: $RESULT_DIR/$MODEL_NAME/"
echo "=========================================="
