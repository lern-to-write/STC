#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env/streamforest_env.sh"

export TASKS="${TASKS:-ovobench_backward_tracking}"
export LIMIT="${LIMIT:-1}"
export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_FRAMES="${MAX_FRAMES:-8}"

bash "${SCRIPT_DIR}/run_eval.sh"
