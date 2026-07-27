#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ " $* " == *" --preflight-only "* ]]; then
  exec python3 -m octo_small_libero.cli \
    --config "${PROJECT_ROOT}/configs/octo_small_libero_4x4090.yaml" \
    "$@"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec torchrun --standalone --nproc_per_node=4 \
  -m octo_small_libero.cli \
  --config "${PROJECT_ROOT}/configs/octo_small_libero_4x4090.yaml" \
  "$@"
