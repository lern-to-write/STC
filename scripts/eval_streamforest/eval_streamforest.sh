#!/usr/bin/env bash
# StreamForest (lmms_eval) — baseline vs + STC-Cacher. Self-contained (no wrapper).
#
#   TASKS=ovobench bash scripts/eval_streamforest/eval_streamforest.sh sf_stc
#   TASKS="streamingbench,videomme" NUM_GPUS=8 bash scripts/eval_streamforest/eval_streamforest.sh sf
#   smoke: LIMIT=1 TASKS=ovobench bash scripts/eval_streamforest/eval_streamforest.sh sf_stc
#
# STC on SigLIP is cacher-only. graph + shared-selection are always on inside stc.
set -euo pipefail

MODE="${1:-sf_stc}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SF_DIR="$PROJECT_ROOT/models/StreamForest"

case "$MODE" in
  sf)
    export STC_PATCH_VISION=0
    ;;
  sf_stc)
    export STC_PATCH_VISION=1
    export STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ;;
  *)
    echo "Usage: $0 [sf|sf_stc]   (set TASKS, e.g. TASKS=ovobench)" >&2; exit 2 ;;
esac

# Source StreamForest's env-config only (checkpoint / anno / output paths + repo
# root on PYTHONPATH so `import stc` works) — not an eval driver.
source "$SF_DIR/scripts/env/streamforest_env.sh"
cd "$SF_DIR"

CKPT_PATH="${STREAMFOREST_CKPT_PATH}"
MODEL_NAME="${MODEL_NAME:-streamforest}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2}"
MAX_NUM_FRAMES="${MAX_FRAMES:-2048}"
TIME_MSG="${TIME_MSG:-short_online_v2}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-$((18000 + RANDOM % 1000))}"
DEFAULT_TASKS="odvbench streamingbench ovbench ovobench videomme mlvu_mc mvbench perceptiontest_val_mc"
read -r -a TASK_ARRAY <<< "${TASKS:-$DEFAULT_TASKS}"
TASK_ARRAY=("${TASK_ARRAY[@]//,/ }")

CKPT_TAG="$(basename "$CKPT_PATH")"
RUN_OUTPUT_DIR="${STREAMFOREST_OUTPUT_DIR}/eval/${CKPT_TAG}"
mkdir -p "$RUN_OUTPUT_DIR"

echo "StreamForest | mode=$MODE | STC_PATCH_VISION=$STC_PATCH_VISION | ckpt=$CKPT_PATH | tasks=${TASK_ARRAY[*]}"

for TASK in "${TASK_ARRAY[@]}"; do
  [[ -z "$TASK" ]] && continue
  TASK_SUFFIX="${TASK//,/_}"
  MODEL_ARGS="pretrained=${CKPT_PATH},conv_template=${CONV_TEMPLATE},max_frames_num=${MAX_NUM_FRAMES},time_msg=${TIME_MSG}"
  OUTPUT_PATH="${RUN_OUTPUT_DIR}/response__${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}"
  LOG_PATH="${RUN_OUTPUT_DIR}/log_${TASK_SUFFIX}_${MODEL_NAME}_F${MAX_NUM_FRAMES}.log"

  CMD=(
    "${PYTHON_EXECUTABLE:-python}" -m accelerate.commands.launch
    --num_processes "$NUM_GPUS" --main_process_port "$MASTER_PORT"
    -m lmms_eval
    --model "$MODEL_NAME"
    --model_args "$MODEL_ARGS"
    --tasks "$TASK"
    --batch_size "${BATCH_SIZE:-1}"
    --log_samples --log_samples_suffix "$TASK_SUFFIX"
    --output_path "$OUTPUT_PATH"
  )
  [[ -n "${LIMIT:-}" ]] && CMD+=(--limit "$LIMIT")

  echo "=== task=$TASK -> $OUTPUT_PATH ==="
  "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
done
