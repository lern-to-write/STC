#!/bin/bash
# Latency benchmark for ReKV: baseline vs ReKV+STC, printing the ViT-encode and
# LLM-prefill reduction.
#
#   bash speed_benchmark/run_rekv.sh                 # both, then a comparison table
#   bash speed_benchmark/run_rekv.sh rekv            # baseline only
#   bash speed_benchmark/run_rekv.sh rekv_stc        # +STC only
#   GPU=0 NUM_FRAMES=16 REPEATS=20 bash speed_benchmark/run_rekv.sh
#   VIDEO=/path/clip.mp4 bash speed_benchmark/run_rekv.sh
#
# Requires a GPU. Override the interpreter / cache with STC_PY, HF_HOME.
set -euo pipefail

MODE="${1:-both}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

PYTHON="${STC_PY:-python}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$PROJECT_ROOT/models/rekv:$PROJECT_ROOT${STC_EXTRA_SITE:+:$STC_EXTRA_SITE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# The selective path allocates many short-lived tensors; expandable_segments
# avoids repeated cudaMalloc/Free and stabilizes timings. (Both var names cover
# different torch versions.)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

MODEL="${REKV_MODEL:-llava_ov_7b}"
NUM_FRAMES="${NUM_FRAMES:-16}"
REPEATS="${REPEATS:-20}"
WARMUP="${WARMUP:-5}"
OUT_DIR="$PROJECT_ROOT/results/speed"
BENCH="$HERE/benchmark_rekv.py"
VIDEO_ARG=(); [[ -n "${VIDEO:-}" ]] && VIDEO_ARG=(--video "$VIDEO")
mkdir -p "$OUT_DIR"

run_mode() {
  local mode="$1"
  case "$mode" in
    rekv)     export STC_PATCH_VISION=0 STC_TOKEN_PER_FRAME=196 STC_UPDATE_TOKEN_RATIO=1.0  STC_CACHE_INTERVAL=4 ;;
    rekv_stc) export STC_PATCH_VISION=1 STC_TOKEN_PER_FRAME=64  STC_UPDATE_TOKEN_RATIO=0.25 STC_CACHE_INTERVAL=4 ;;
    *) echo "Usage: $0 [rekv|rekv_stc|both]" >&2; exit 2 ;;
  esac
  echo ">>> mode=$mode  frames=$NUM_FRAMES  repeats=$REPEATS  gpu=${CUDA_VISIBLE_DEVICES:-default}"
  "$PYTHON" "$BENCH" --model "$MODEL" --num-frames "$NUM_FRAMES" \
    --repeats "$REPEATS" --warmup "$WARMUP" --label "$mode" \
    --out "$OUT_DIR/${mode}.json" "${VIDEO_ARG[@]}"
}

case "$MODE" in
  rekv|rekv_stc) run_mode "$MODE" ;;
  both)
    run_mode rekv
    run_mode rekv_stc
    "$PYTHON" - "$OUT_DIR/rekv.json" "$OUT_DIR/rekv_stc.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1])); stc = json.load(open(sys.argv[2]))
def row(name, b, s):
    print(f"  {name:<13}{b:9.1f}  ->{s:9.1f} ms    x{b/s:4.2f}   ({(1-s/b)*100:5.1f}% down)")
print("=" * 66)
print(f" ReKV vs ReKV+STC (min over {base['repeats']} reps)  |  frames={base['num_frames']}")
print("-" * 66)
print(f"  {'stage':<13}{'baseline':>9}    {'+STC':>9}        speedup  reduction")
row("ViT encode",  base['vit_encode_ms']['min'],  stc['vit_encode_ms']['min'])
row("LLM prefill", base['llm_prefill_ms']['min'], stc['llm_prefill_ms']['min'])
print("-" * 66)
print(" paper (ReKV, Table 1): ViT 24.5% down, LLM prefill 45.3% down")
print(" note: LLM-prefill reduction grows with sequence length; try NUM_FRAMES=64.")
print("=" * 66)
PY
    ;;
  *) echo "Usage: $0 [rekv|rekv_stc|both]" >&2; exit 2 ;;
esac
