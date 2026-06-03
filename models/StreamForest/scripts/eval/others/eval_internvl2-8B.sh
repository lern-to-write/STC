#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../env/streamforest_env.sh"

MASTER_PORT="${MASTER_PORT:-$((18000 + RANDOM % 1000))}"
CKPT_PATH="${INTERNVL2_CKPT_PATH:-${STREAMFOREST_INTERNVL2_CKPT_PATH:-InternVL2-8B}}"
MODEL_NAME="${MODEL_NAME:-internvl2}"
CONV_TEMPLATE="${CONV_TEMPLATE:-internlm}"
MAX_NUM_FRAMES="${MAX_FRAMES:-24}"
NUM_GPUS="${NUM_GPUS:-1}"
TASK="${TASK:-odvbench}"
TASK_SUFFIX="${TASK//,/_}"
JOB_NAME="$(basename "$0" .sh)_$(date +"%Y%m%d_%H%M%S")"
CKPT_TAG="$(basename "${CKPT_PATH}")"
LOG_DIR="${STREAMFOREST_OUTPUT_DIR}/eval/${CKPT_TAG}"
mkdir -p "${LOG_DIR}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

LAUNCH_PREFIX=()
if [[ "${STREAMFOREST_USE_SLURM:-0}" == "1" ]]; then
  PARTITION="${PARTITION:-video5}"
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

"${LAUNCH_PREFIX[@]}" \
"${PYTHON_EXECUTABLE:-python}" -m accelerate.commands.launch --num_processes "${NUM_GPUS}" --main_process_port "${MASTER_PORT}" -m lmms_eval \
  --model "${MODEL_NAME}" \
  --model_args "pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES}" \
  --tasks "${TASK}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --log_samples \
  --log_samples_suffix "${TASK_SUFFIX}" \
  --output_path "${LOG_DIR}/log_result/${JOB_NAME}_f${MAX_NUM_FRAMES}" \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/${JOB_NAME}_f${MAX_NUM_FRAMES}.log"
