#!/usr/bin/env bash
# Dispider OVO-Bench — baseline vs + STC-Cacher. Self-contained (no wrapper).
#
#   MODEL_PATH=/path/Dispider CLIP_CKPT_PATH=/path/clip-vit-large-patch14 \
#   ANNO_PATH=/path/ovo_bench_new.json CHUNKED_DIR=/path/chunked_videos \
#     bash scripts/eval_dispider/eval_dispider_ovobench.sh dispider_stc
#
# Smoke: add MAX_SAMPLES=1 TASKS=EPM. Multi-GPU: NUM_GPUS=8 NUM_CHUNKS=8.
set -euo pipefail

MODE="${1:-dispider_stc}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISPIDER_DIR="$PROJECT_ROOT/models/Dispider"

# --- STC-Cacher toggle (CLIP, cacher-only). graph + shared-selection are always on inside stc. ---
case "$MODE" in
  dispider)
    export STC_PATCH_VISION=0
    ;;
  dispider_stc)
    export STC_PATCH_VISION=1
    export STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ;;
  *)
    echo "Usage: $0 [dispider|dispider_stc]" >&2; exit 2 ;;
esac

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -n "${DISPIDER_ENV:-}" ]]; then
  source "$DISPIDER_ENV/bin/activate"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
# repo root on PYTHONPATH so `import stc` works; DISPIDER_DIR for dispider.*
export PYTHONPATH="$PROJECT_ROOT:$DISPIDER_DIR:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Mar2Ding/Dispider}"
CLIP_CKPT_PATH="${CLIP_CKPT_PATH:-openai/clip-vit-large-patch14}"
export DISPIDER_CLIP_CKPT_PATH="${DISPIDER_CLIP_CKPT_PATH:-$CLIP_CKPT_PATH}"
ANNO_PATH="${ANNO_PATH:-}"
CHUNKED_DIR="${CHUNKED_DIR:-}"
RESULT_DIR="${RESULT_DIR:-$DISPIDER_DIR/results/ovobench}"
TASKS="${TASKS:-EPM ASI HLD OCR ACR ATR STU FPD OJR REC SSR CRR}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_CHUNKS="${NUM_CHUNKS:-$NUM_GPUS}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RUN_SCORE="${RUN_SCORE:-0}"
OVO_SCORE_ROOT="${OVO_SCORE_ROOT:-$PROJECT_ROOT/models/rekv/model/online_bench_inference/ovobench}"

: "${ANNO_PATH:?Set ANNO_PATH to OVO-Bench ovo_bench_new.json}"
: "${CHUNKED_DIR:?Set CHUNKED_DIR to OVO-Bench chunked_videos directory}"

echo "Dispider OVO-Bench | mode=$MODE | STC_PATCH_VISION=$STC_PATCH_VISION"
echo "  model: $MODEL_PATH | clip: $CLIP_CKPT_PATH | chunks: $NUM_CHUNKS | tasks: $TASKS"

mkdir -p "$RESULT_DIR/Dispider"

for IDX in $(seq 0 $((NUM_CHUNKS - 1))); do
  GPU_ID=$((IDX % NUM_GPUS))
  CMD=(
    python "$DISPIDER_DIR/dispider/eval/model_ovobench.py"
    --model-path "$MODEL_PATH"
    --clip-ckpt-path "$CLIP_CKPT_PATH"
    --anno-path "$ANNO_PATH"
    --chunked-dir "$CHUNKED_DIR"
    --result-dir "$RESULT_DIR"
    --tasks $TASKS
    --num-chunks "$NUM_CHUNKS"
    --chunk-idx "$IDX"
  )
  if [[ -n "$MAX_SAMPLES" ]]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}" &
done

wait

if [[ "$RUN_SCORE" == "1" ]]; then
  if [[ -f "$OVO_SCORE_ROOT/score.py" ]]; then
    (cd "$OVO_SCORE_ROOT" && python score.py --model Dispider --mode offline --result_dir "$RESULT_DIR")
  else
    echo "Skip scoring: score.py not found under $OVO_SCORE_ROOT"
  fi
fi
