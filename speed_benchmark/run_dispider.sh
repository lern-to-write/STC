#!/bin/bash
# ViT-encoding latency for Dispider (CLIP): baseline vs + STC-Cacher.
#
#   bash speed_benchmark/run_dispider.sh                 # both, then a comparison
#   bash speed_benchmark/run_dispider.sh dispider        # baseline only
#   bash speed_benchmark/run_dispider.sh dispider_stc    # +STC only
#   GPU=0 CLIP_CKPT_PATH=/path/to/clip-vit-large-patch14 bash speed_benchmark/run_dispider.sh
#
# Requires a GPU. Override the interpreter / cache with STC_PY, HF_HOME.
set -euo pipefail

MODE="${1:-both}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

PYTHON="${STC_PY:-python}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$PROJECT_ROOT/models/Dispider:$PROJECT_ROOT${STC_EXTRA_SITE:+:$STC_EXTRA_SITE}:${PYTHONPATH:-}"
export DISPIDER_CLIP_CKPT_PATH="${DISPIDER_CLIP_CKPT_PATH:-${CLIP_CKPT_PATH:-openai/clip-vit-large-patch14}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

NUM_FRAMES="${NUM_FRAMES:-16}"
REPEATS="${REPEATS:-20}"
WARMUP="${WARMUP:-5}"
OUT_DIR="$PROJECT_ROOT/results/speed"
BENCH="$HERE/benchmark_dispider.py"
mkdir -p "$OUT_DIR"

run_mode() {
  local mode="$1"
  case "$mode" in
    dispider)     export STC_PATCH_VISION=0 ;;
    dispider_stc) export STC_PATCH_VISION=1 STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4 ;;
    *) echo "Usage: $0 [dispider|dispider_stc|both]" >&2; exit 2 ;;
  esac
  echo ">>> mode=$mode  frames=$NUM_FRAMES  gpu=${CUDA_VISIBLE_DEVICES:-default}"
  "$PYTHON" "$BENCH" --num-frames "$NUM_FRAMES" --repeats "$REPEATS" \
    --warmup "$WARMUP" --label "$mode" --out "$OUT_DIR/${mode}.json"
}

case "$MODE" in
  dispider|dispider_stc) run_mode "$MODE" ;;
  both)
    run_mode dispider
    run_mode dispider_stc
    "$PYTHON" - "$OUT_DIR/dispider.json" "$OUT_DIR/dispider_stc.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1])); stc = json.load(open(sys.argv[2]))
b = base["vit_encode_ms"]["min"]; s = stc["vit_encode_ms"]["min"]
print("=" * 60)
print(f" Dispider ViT encode (min over {base['repeats']} reps) | frames={base['num_frames']}")
print(f"   baseline {b:8.1f} ms  ->  +STC {s:8.1f} ms   x{b/s:4.2f}   ({(1-s/b)*100:5.1f}% down)")
print("=" * 60)
PY
    ;;
  *) echo "Usage: $0 [dispider|dispider_stc|both]" >&2; exit 2 ;;
esac
