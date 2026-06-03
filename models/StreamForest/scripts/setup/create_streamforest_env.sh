#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env/streamforest_env.sh"

ENV_ROOT="${STREAMFOREST_ENV_ROOT:-/apdcephfs_tj5/share_303570626/yiyuwang/envs}"
ENV_NAME="${STREAMFOREST_ENV_NAME:-streamforest-py310}"
ENV_PREFIX="${STREAMFOREST_ENV_PREFIX:-${ENV_ROOT}/${ENV_NAME}}"
EXISTING_ENV_PREFIX="${STREAMFOREST_EXISTING_ENV_PREFIX:-${ENV_ROOT}/lmms-streamforest-py312-tf446}"
PYTHON_VERSION="${STREAMFOREST_PYTHON_VERSION:-3.10}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

mkdir -p "${ENV_ROOT}"

if [[ "${STREAMFOREST_FORCE_CREATE:-0}" != "1" && -x "${EXISTING_ENV_PREFIX}/bin/python" ]]; then
  ENV_PREFIX="${EXISTING_ENV_PREFIX}"
  echo "Using existing verified environment: ${ENV_PREFIX}"
elif [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Using existing environment: ${ENV_PREFIX}"
elif command -v conda >/dev/null 2>&1; then
  echo "Creating conda environment: ${ENV_PREFIX}"
  conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
elif command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
  echo "Creating venv environment: ${ENV_PREFIX}"
  "python${PYTHON_VERSION}" -m venv "${ENV_PREFIX}"
else
  echo "Neither conda nor python${PYTHON_VERSION} was found. Enter the taiji container and rerun this script." >&2
  exit 1
fi

PYTHON="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"

if [[ "${ENV_PREFIX}" != "${EXISTING_ENV_PREFIX}" || "${STREAMFOREST_INSTALL_REQUIREMENTS:-0}" == "1" ]]; then
  "${PYTHON}" -m pip install --upgrade pip setuptools wheel
  "${PIP}" install ${PIP_EXTRA_ARGS} -r "${STREAMFOREST_ROOT}/requirements.txt"
else
  echo "Skipping pip install for existing environment. Set STREAMFOREST_INSTALL_REQUIREMENTS=1 to reinstall."
fi

if [[ "${STREAMFOREST_DOWNLOAD_HF:-0}" == "1" ]]; then
  "${PYTHON}" "${STREAMFOREST_ROOT}/download_hf.py"
fi

"${PYTHON}" - <<'PY'
import importlib
import sys

modules = ["torch", "transformers", "accelerate", "av", "decord", "lmms_eval", "llava"]
failed = []
for name in modules:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: ok {getattr(mod, '__version__', '')}")
    except Exception as exc:
        failed.append((name, repr(exc)))

if failed:
    for name, exc in failed:
        print(f"{name}: FAIL {exc}", file=sys.stderr)
    sys.exit(1)
PY

echo "Environment ready: ${ENV_PREFIX}"
echo "Activate with: source ${ENV_PREFIX}/bin/activate"
