#!/bin/bash
# STC 加速比复现：跑 baseline(ReKV) 与 ReKV+STC，打印 ViT 编码 / LLM prefill 的降幅。
#
#   bash speed_benchmark/run.sh                 # 两档都跑并对比（默认 16 帧）
#   bash speed_benchmark/run.sh rekv            # 只跑 baseline
#   bash speed_benchmark/run.sh rekv_stc        # 只跑 ReKV+STC
#   GPU=4 NUM_FRAMES=16 REPEATS=20 bash speed_benchmark/run.sh
#   VIDEO=/path/x.mp4 bash speed_benchmark/run.sh           # 用真实视频替代合成帧
#
# ⚠️ 必须在带 GPU 的容器里跑。环境要求见 speed_benchmark/README.md。
set -euo pipefail

MODE="${1:-both}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# --- 环境（可用 env 覆盖；默认值适配本机容器，详见 README）---
PYTHON="${STC_PY:-/apdcephfs_tj5/share_303570626/yiyuwang/envs/lmms-streamforest-py312-tf446/bin/python}"
EXTRA_SITE="${STC_EXTRA_SITE:-/apdcephfs_tj5/share_303570626/yiyuwang/envs/_stc_extra_site}"
export HF_HOME="${HF_HOME:-/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face}"
export PYTHONPATH="$EXTRA_SITE:$PROJECT_ROOT/models/rekv:$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# STC-Cacher 的 selective 路径会建大量临时张量；默认 caching allocator 会反复
# cudaMalloc/Free 直连驱动（极慢且抖动大）。expandable_segments 能把 selective
# 中位耗时从 40~99ms 压到 ~24ms。新旧两个变量名都设，兼容不同 torch 版本。
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

MODEL="${REKV_MODEL:-llava_ov_7b}"
NUM_FRAMES="${NUM_FRAMES:-16}"          # 固定 16 帧（论文 ViT 延迟基准）
REPEATS="${REPEATS:-20}"
WARMUP="${WARMUP:-5}"
OUT_DIR="$PROJECT_ROOT/results/speed"
BENCH="$HERE/benchmark.py"
VIDEO_ARG=(); [[ -n "${VIDEO:-}" ]] && VIDEO_ARG=(--video "$VIDEO")
mkdir -p "$OUT_DIR"

run_mode() {
  local mode="$1"
  case "$mode" in
    rekv)     export STC_PATCH_VISION=0 STC_TOKEN_PER_FRAME=196 STC_UPDATE_TOKEN_RATIO=1.0  STC_CACHE_INTERVAL=2 ;;
    rekv_stc) export STC_PATCH_VISION=1 STC_TOKEN_PER_FRAME=64  STC_UPDATE_TOKEN_RATIO=0.25 \
                     STC_CACHE_INTERVAL="${STC_CACHE_INTERVAL:-2}" \
                     STC_CUDA_GRAPH="${STC_CUDA_GRAPH:-1}" \
                     STC_SHARE_SELECTION="${STC_SHARE_SELECTION:-1}" ;;   # graph 回放 + 每帧共享 token 选择

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
    print(f"  {name:<13}{b:9.1f}  ->{s:9.1f} ms    x{b/s:4.2f}   (↓{(1-s/b)*100:5.1f}%)")
print("=" * 66)
print(f" ReKV vs ReKV+STC (min over {base['repeats']} reps)  |  frames={base['num_frames']}")
print(f" baseline token/frame={base['config']['token_per_frame']}  "
      f"stc token/frame={stc['config']['token_per_frame']}  "
      f"update_ratio={stc['config']['update_token_ratio']}  N={stc['config']['cache_interval']}")
print("-" * 66)
print(f"  {'stage':<13}{'baseline':>9}    {'+STC':>9}        speedup  reduction")
row("ViT encode",  base['vit_encode_ms']['min'],  stc['vit_encode_ms']['min'])
row("LLM prefill", base['llm_prefill_ms']['min'], stc['llm_prefill_ms']['min'])
print("-" * 66)
print(" paper (ReKV, Table 1): ViT ↓24.5%,  LLM prefill ↓45.3%")
print(" note: LLM-prefill reduction grows with sequence length; try NUM_FRAMES=64.")
print("=" * 66)
PY
    ;;
  *) echo "Usage: $0 [rekv|rekv_stc|both]" >&2; exit 2 ;;
esac
