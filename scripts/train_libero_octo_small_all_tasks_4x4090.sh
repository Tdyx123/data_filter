#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

training_args=(
  --config "${PROJECT_ROOT}/configs/octo_small_libero_4x4090.yaml"
  --all-tasks
)
target_only=false
custom_prior=false
for argument in "$@"; do
  case "${argument}" in
    --target-only)
      target_only=true
      ;;
    --prior-relcore-manifest|--prior-relcore-manifest=*)
      custom_prior=true
      ;;
    --prior-quality-filter-scores|--prior-quality-filter-scores=*)
      custom_prior=true
      ;;
  esac
done

if [[ "${target_only}" == false ]]; then
  training_args+=(
    --sample-weights 3 1
  )
  if [[ "${custom_prior}" == false ]]; then
    training_args+=(
      --prior-prefiltered-scores /data/dwb/libero90_sqcn/filter/top10pct/scores.csv
    )
  fi
fi

if [[ " $* " == *" --preflight-only "* ]]; then
  exec python3 -m octo_small_libero.cli \
    "${training_args[@]}" \
    "$@"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec torchrun --standalone --nproc_per_node=4 \
  -m octo_small_libero.cli \
  "${training_args[@]}" \
  "$@"
