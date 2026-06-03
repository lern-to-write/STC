#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env/streamforest_env.sh"

CKPT_PATH="${STREAMFOREST_CKPT_PATH}"
MAX_NUM_FRAMES="${MAX_FRAMES:-2048}"
TASK="${TASK:-}"
MODEL_NAME="${MODEL_NAME:-streamforest}"
TIME_MSG="${TIME_MSG:-short_online}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt_path) CKPT_PATH="$2"; shift 2 ;;
    --max_frames) MAX_NUM_FRAMES="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --time_msg) TIME_MSG="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  echo "Missing --task or TASK env var" >&2
  exit 1
fi

MASTER_PORT="${MASTER_PORT:-$((18000 + RANDOM % 1000))}"
NUM_GPUS="${NUM_GPUS:-1}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2}"
TASK_SUFFIX="${TASK//,/_}"
JOB_NAME="$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")"
CKPT_TAG="$(basename "${CKPT_PATH}")"
RUN_OUTPUT_DIR="${STREAMFOREST_OUTPUT_DIR}/eval/${CKPT_TAG}"
mkdir -p "${RUN_OUTPUT_DIR}"

MODEL_ARGS="pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES},time_msg=${TIME_MSG}"
OUTPUT_PATH="${RUN_OUTPUT_DIR}/response__${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}"
LOG_PATH="${RUN_OUTPUT_DIR}/log_${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}_${JOB_NAME}.log"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

EXTRA_ARGS=()
if [[ -n "${LMMS_EVAL_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${LMMS_EVAL_EXTRA_ARGS}"
fi

LAUNCH_PREFIX=()
if [[ "${STREAMFOREST_USE_SLURM:-0}" == "1" ]]; then
  PARTITION="${PARTITION:-videop1}"
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

CMD=(
  "${PYTHON_EXECUTABLE:-python}"
  -m accelerate.commands.launch
  --num_processes "${NUM_GPUS}"
  --main_process_port "${MASTER_PORT}"
  -m lmms_eval
  --model "${MODEL_NAME}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASK}"
  --batch_size "${BATCH_SIZE:-1}"
  --log_samples
  --log_samples_suffix "${TASK_SUFFIX}"
  --output_path "${OUTPUT_PATH}"
  "${LIMIT_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

echo "Checkpoint: ${CKPT_PATH}"
echo "Max frames: ${MAX_NUM_FRAMES}"
echo "Task: ${TASK}"
echo "Model: ${MODEL_NAME}"
echo "Time message: ${TIME_MSG}"
echo "Python: ${PYTHON_EXECUTABLE:-python}"
echo "Output: ${OUTPUT_PATH}"
echo "Log: ${LOG_PATH}"

"${LAUNCH_PREFIX[@]}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
