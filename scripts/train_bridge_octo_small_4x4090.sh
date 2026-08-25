#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG="${PROJECT_ROOT}/configs/octo_small_bridge_v2_4x4090.yaml"
PYENV_BIN="${PYENV_BIN:-/home/dwb/.pyenv/bin/pyenv}"
export PYENV_VERSION="${PYENV_VERSION:-miniconda3-3.12-25.11.1-1}"

if [[ ! -x "${PYENV_BIN}" ]]; then
  printf 'error: pyenv executable is not executable: %s\n' "${PYENV_BIN}" >&2
  exit 2
fi

if [[ " $* " == *" --preflight-only "* ]]; then
  exec "${PYENV_BIN}" exec python -m octo_small_bridge.cli --config "${CONFIG}" "$@"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec "${PYENV_BIN}" exec torchrun --standalone --nproc_per_node=4 \
  -m octo_small_bridge.cli --config "${CONFIG}" "$@"
