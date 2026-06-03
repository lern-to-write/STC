


#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/models/rekv:$PROJECT_ROOT:${PYTHONPATH:-}"
export STC_PATCH_VISION="${STC_PATCH_VISION:-0}"
export STC_TOKEN_PER_FRAME="${STC_TOKEN_PER_FRAME:-196}"
export STC_UPDATE_TOKEN_RATIO="${STC_UPDATE_TOKEN_RATIO:-1.0}"
cd "$PROJECT_ROOT/models/rekv/model/online_bench_inference/streamingbench"

EVAL_MODEL="rekv"
CONTEXT_TIME=-1
TASK="real"
DATA_FILE="src/data/questions_${TASK}.json"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
OUTPUT_FILE="src/data/${TASK}${EVAL_MODEL}_${TIMESTAMP}.json"
BENCHMARK="Streaming"

echo "开始执行评估..."
echo "模型: $EVAL_MODEL"
echo "任务: $TASK"
echo "数据文件: $DATA_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "STC_PATCH_VISION: $STC_PATCH_VISION"
echo "STC_TOKEN_PER_FRAME: $STC_TOKEN_PER_FRAME"
echo "STC_UPDATE_TOKEN_RATIO: $STC_UPDATE_TOKEN_RATIO"
# ReKV is physically integrated under STC_new/models/rekv.

# 使用改进的启动命令
python src/eval.py \
    --model_name "$EVAL_MODEL" \
    --benchmark_name "$BENCHMARK" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --context_time "$CONTEXT_TIME"

echo "评估完成!"
