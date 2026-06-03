#!/usr/bin/env bash
set -euo pipefail

DISPIDER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd "$DISPIDER_DIR/../.." && pwd)"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -n "${DISPIDER_ENV:-}" ]]; then
  source "$DISPIDER_ENV/bin/activate"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$DISPIDER_DIR:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-Mar2Ding/Dispider}"
CLIP_CKPT_PATH="${CLIP_CKPT_PATH:-openai/clip-vit-large-patch14}"
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

echo "Dispider OVO-Bench"
echo "  python:     $(command -v python)"
echo "  cuda:       ${CUDA_HOME:-system/default}"
echo "  model:      $MODEL_PATH"
echo "  clip:       $CLIP_CKPT_PATH"
echo "  anno:       $ANNO_PATH"
echo "  chunked:    $CHUNKED_DIR"
echo "  result_dir: $RESULT_DIR"
echo "  tasks:      $TASKS"
echo "  chunks:     $NUM_CHUNKS"

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
