#!/bin/bash
set -euo pipefail

MODE="${1:-rekv}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${HF_HOME:-/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face}"
export PYTHONPATH="$PROJECT_ROOT/models/rekv:$PROJECT_ROOT:${PYTHONPATH:-}"
MODEL="${REKV_MODEL:-llava_ov_7b}"

case "$MODE" in
  rekv)
    export STC_PATCH_VISION=0
    export STC_TOKEN_PER_FRAME=196
    export STC_UPDATE_TOKEN_RATIO=1.0
    ;;
  rekv_stc|rekv+stc|stc)
    export STC_PATCH_VISION=1
    export STC_TOKEN_PER_FRAME=64
    export STC_UPDATE_TOKEN_RATIO=0.25
    ;;
  *)
    echo "Usage: $0 [rekv|rekv_stc]" >&2
    exit 2
    ;;
esac

mkdir -p "$PROJECT_ROOT/results/smoke_${MODE}"
cd "$PROJECT_ROOT"

python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port="${MASTER_PORT:-29500}" \
  -m model.video_qa.run_distributed \
  --dataset smoke \
  --model "$MODEL" \
  --save_dir "results/smoke_${MODE}" \
  --sample_fps 0.5 \
  --n_local 15000 \
  --retrieve_size 16
