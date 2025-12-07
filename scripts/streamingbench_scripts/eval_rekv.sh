


#!/bin/bash

cd model/online_bench_inference/streamingbench

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
# export PYTHONPATH="path/to/your/project:$PYTHONPATH"

# 使用改进的启动命令
python src/eval.py \
    --model_name "$EVAL_MODEL" \
    --benchmark_name "$BENCHMARK" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --context_time "$CONTEXT_TIME"

echo "评估完成!"

