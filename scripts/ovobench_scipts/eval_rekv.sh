#!/bin/bash

###############################################################################
# ReKV 模型分布式推理专用脚本 - 每张卡多进程版本
###############################################################################
cd model/online_bench_inference/ovobench
# ReKV 特定配置
MODEL_NAME="rekv"
ANNO_PATH="data/ovo_bench_new.json"
VIDEO_DIR="data/src_videos"
CHUNKED_DIR="/mnt/data0/public/back/huggingface/hub/datasets--JoeLeelyf--OVO-Bench/snapshots/fec29e3385747b5642d995370143ba92d2819bd2/videos/chunked_videos"
RESULT_DIR="results"
MODE="offline"

# 多进程配置
NUM_GPUS=8              # 实际GPU数量
PROCESSES_PER_GPU=6     # 每张卡运行的进程数
TOTAL_PROCESSES=$((NUM_GPUS * PROCESSES_PER_GPU))  # 总进程数 = 48

RETRIEVE_SIZE=64

# 任务列表
TASKS="EPM STU"


echo "=========================================="
echo "Starting ReKV Distributed Inference"
echo "Number of GPUs: $NUM_GPUS"
echo "Processes per GPU: $PROCESSES_PER_GPU"
echo "Total Processes: $TOTAL_PROCESSES"
echo "Retrieve Size: $RETRIEVE_SIZE"
echo "=========================================="

# 遍历每个任务并单独运行
for TASK in $TASKS; do
    echo "=========================================="
    echo "Processing task: $TASK"
    echo "=========================================="
    
    # 使用总进程数运行
    torchrun \
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
        --save_results True \
        --global_seed 42 \
        --tf32
done

echo "=========================================="
echo "ReKV inference completed!"
echo "Results: $RESULT_DIR/$MODEL_NAME/"
echo "=========================================="