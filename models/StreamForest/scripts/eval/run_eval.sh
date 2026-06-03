#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env/streamforest_env.sh"

MAX_FRAMES="${MAX_FRAMES:-2048}"
TIME_MSG="${TIME_MSG:-short_online_v2}"
MODEL_NAME="${MODEL_NAME:-streamforest}"
CKPT_PATH="${STREAMFOREST_CKPT_PATH}"

DEFAULT_TASKS=(
  "odvbench"
  "streamingbench"
  "ovbench"
  "ovobench"
  "videomme"
  "mlvu_mc"
  "mvbench"
  "perceptiontest_val_mc"
)

if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_ARRAY <<< "${TASKS//,/ }"
else
  TASK_ARRAY=("${DEFAULT_TASKS[@]}")
fi

echo "StreamForest root: ${STREAMFOREST_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "Data root: ${STREAMFOREST_DATA_ROOT}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Output root: ${STREAMFOREST_OUTPUT_DIR}"

for TASK in "${TASK_ARRAY[@]}"; do
  echo "============================"
  echo "Running benchmark: ${TASK}"
  echo "============================"

  bash scripts/eval/online/eval_online_template.sh \
    --ckpt_path "${CKPT_PATH}" \
    --max_frames "${MAX_FRAMES}" \
    --model_name "${MODEL_NAME}" \
    --time_msg "${TIME_MSG}" \
    --task "${TASK}"
done
