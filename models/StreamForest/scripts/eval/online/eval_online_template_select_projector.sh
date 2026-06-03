#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env/streamforest_env.sh"

MAX_NUM_FRAMES="${MAX_FRAMES:-512}"
MODEL_NAME="${MODEL_NAME:-streamforest}"
TIME_MSG="${TIME_MSG:-short_online_v2}"
REPLACE_PROJECTOR="${REPLACE_PROJECTOR:-ablation_woSTFW_PEMF}"
CKPT_PATH="${STREAMFOREST_PROJECTOR_CKPT_PATH:-${STREAMFOREST_CKPT_PATH}}"
TASK="${TASK:-ovbench}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-$((18000 + RANDOM % 1000))}"
TASK_SUFFIX="${TASK//,/_}"
JOB_NAME="$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")"
CKPT_TAG="$(basename "${CKPT_PATH}")"
RESULT_DIR="${STREAMFOREST_OUTPUT_DIR}/eval/${CKPT_TAG}/${MAX_NUM_FRAMES}_${TASK_SUFFIX}_${REPLACE_PROJECTOR}"
mkdir -p "${RESULT_DIR}"

MODEL_ARGS="pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES},time_msg=${TIME_MSG},mm_projector_type=${REPLACE_PROJECTOR}"
OUTPUT_PATH="${RESULT_DIR}/response__${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}"
LOG_PATH="${RESULT_DIR}/log_${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

LAUNCH_PREFIX=()
if [[ "${STREAMFOREST_USE_SLURM:-0}" == "1" ]]; then
  PARTITION="${PARTITION:-videopp1}"
  CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
  LAUNCH_PREFIX=(
    srun -p "${PARTITION}"
    --job-name="${JOB_NAME}"
    --ntasks=1
    --gres="gpu:${NUM_GPUS}"
    --ntasks-per-node=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --kill-on-bad-exit=1
  )
fi

echo "Checkpoint: ${CKPT_PATH}"
echo "Max frames: ${MAX_NUM_FRAMES}"
echo "Task: ${TASK}"
echo "Model: ${MODEL_NAME}"
echo "Time message: ${TIME_MSG}"
echo "Projector: ${REPLACE_PROJECTOR}"
echo "Python: ${PYTHON_EXECUTABLE:-python}"
echo "Output: ${OUTPUT_PATH}"

"${LAUNCH_PREFIX[@]}" \
"${PYTHON_EXECUTABLE:-python}" -m accelerate.commands.launch --num_processes "${NUM_GPUS}" --main_process_port "${MASTER_PORT}" -m lmms_eval \
  --model "${MODEL_NAME}" \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASK}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --log_samples \
  --log_samples_suffix "${TASK_SUFFIX}" \
  --output_path "${OUTPUT_PATH}" \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"
