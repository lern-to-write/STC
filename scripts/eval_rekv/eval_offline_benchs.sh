#!/bin/bash
DATASET="mlvu"
MODEL="llava_ov_0.5b"
NUM_PROCESSES=1
NUM_GPUS=1
SAVE_DIR="results/torchrun"
MASTER_PORT=29500
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/models/rekv:$PROJECT_ROOT:${PYTHONPATH:-}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 打印配置
echo "=========================================="
echo "🚀 TorchRun 分布式评估"
echo "=========================================="
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "GPU数量: $NUM_GPUS"
echo "进程数量: $NUM_PROCESSES"
echo "每GPU进程数: $((NUM_PROCESSES / NUM_GPUS))"
echo "Master Port: $MASTER_PORT"
echo "输出目录: $SAVE_DIR"
echo "=========================================="

# 使用当前 Python 环境启动分布式评估
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=$NUM_PROCESSES \
    --master_port=$MASTER_PORT \
    -m model.video_qa.run_distributed \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --save_dir "$SAVE_DIR"

echo "✅ TorchRun评估完成！"
