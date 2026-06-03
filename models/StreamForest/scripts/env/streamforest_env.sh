#!/usr/bin/env bash

STREAMFOREST_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STREAMFOREST_ROOT="${STREAMFOREST_ROOT:-$(cd "${STREAMFOREST_ENV_DIR}/../.." && pwd)}"
export STREAMFOREST_PROJECT_ROOT="${STREAMFOREST_PROJECT_ROOT:-$(cd "${STREAMFOREST_ROOT}/../.." && pwd)}"

export HF_HOME="${HF_HOME:-/apdcephfs_tj5/share_303570626/yiyuwang/hugging_face}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export STREAMFOREST_DATA_ROOT="${STREAMFOREST_DATA_ROOT:-${HF_HOME}}"
export STREAMFOREST_OUTPUT_DIR="${STREAMFOREST_OUTPUT_DIR:-${STREAMFOREST_ROOT}/results}"
export STREAMFOREST_CKPT_ROOT="${STREAMFOREST_CKPT_ROOT:-${STREAMFOREST_ROOT}/ckpt}"
if [[ -z "${STREAMFOREST_ANNO_ROOT:-}" ]]; then
  STREAMFOREST_PROJECT_ANNO_ROOT="${STREAMFOREST_PROJECT_ROOT}/benchmarks/streamforest/eval"
  STREAMFOREST_HF_ANNODATA_ROOT=""
  STREAMFOREST_HF_ANNODATA_SNAPSHOTS="${HF_HOME}/hub/datasets--MCG-NJU--StreamForest-Annodata/snapshots"
  if [[ -d "${STREAMFOREST_HF_ANNODATA_SNAPSHOTS}" ]]; then
    for STREAMFOREST_HF_ANNODATA_CANDIDATE in "${STREAMFOREST_HF_ANNODATA_SNAPSHOTS}"/*/eval; do
      if [[ -e "${STREAMFOREST_HF_ANNODATA_CANDIDATE}/OVOBench/json/backward_tracking.json" ]]; then
        STREAMFOREST_HF_ANNODATA_ROOT="${STREAMFOREST_HF_ANNODATA_CANDIDATE}"
        break
      fi
    done
  fi

  if [[ -e "${STREAMFOREST_PROJECT_ANNO_ROOT}/OVOBench/json/backward_tracking.json" ]]; then
    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_PROJECT_ROOT}/benchmarks/streamforest/eval"
  elif [[ -n "${STREAMFOREST_HF_ANNODATA_ROOT}" ]]; then
    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_HF_ANNODATA_ROOT}"
  elif [[ -d "${STREAMFOREST_PROJECT_ANNO_ROOT}" ]]; then
    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_PROJECT_ANNO_ROOT}"
  else
    export STREAMFOREST_ANNO_ROOT="${STREAMFOREST_ROOT}/anno/eval"
  fi
  unset STREAMFOREST_PROJECT_ANNO_ROOT
  unset STREAMFOREST_HF_ANNODATA_ROOT
  unset STREAMFOREST_HF_ANNODATA_SNAPSHOTS
  unset STREAMFOREST_HF_ANNODATA_CANDIDATE
fi

if [[ -z "${STREAMFOREST_CKPT_PATH:-}" ]]; then
  if [[ -d "${HF_HOME}/StreamForest-Qwen2-7B" ]]; then
    export STREAMFOREST_CKPT_PATH="${HF_HOME}/StreamForest-Qwen2-7B"
  else
    export STREAMFOREST_CKPT_PATH="MCG-NJU/StreamForest-Qwen2-7B"
  fi
fi

if [[ -z "${STREAMFOREST_DRIVE_CKPT_PATH:-}" ]]; then
  if [[ -d "${HF_HOME}/StreamForest-Drive-Qwen2-7B" ]]; then
    export STREAMFOREST_DRIVE_CKPT_PATH="${HF_HOME}/StreamForest-Drive-Qwen2-7B"
  else
    export STREAMFOREST_DRIVE_CKPT_PATH="MCG-NJU/StreamForest-Drive-Qwen2-7B"
  fi
fi

case ":${PYTHONPATH:-}:" in
  *":${STREAMFOREST_ROOT}:"*) ;;
  *) export PYTHONPATH="${STREAMFOREST_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

cd "${STREAMFOREST_ROOT}"
