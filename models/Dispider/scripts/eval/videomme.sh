# !/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python  ./dispider/eval/model_videomme_long.py \
        --model-path YOUR_MODEL_PATH \
        --image-folder YOUR_VIDEO_FOLDER_PATH \
        --chat_conversation_output_folder YOUR_OUTPUT_PATH/${CHUNKS}_${IDX}.json \
        --Eval_QA_root ./playground/data/videomme.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode qwen &
done


wait

python  ./dispider/eval/eval_videomme.py \
        --results_file YOUR_OUTPUT_PATH
