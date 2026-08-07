#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

training_args=(
  --config "${PROJECT_ROOT}/configs/qwen3_vl_4b_groot_libero_4x4090.yaml"
  --all-tasks
)
target_only=false
custom_weights=false
custom_prior=false
for argument in "$@"; do
  case "${argument}" in
    --target-only)
      target_only=true
      ;;
    --sample-weights|--sample-weights=*)
      custom_weights=true
      ;;
    --prior-top-percent|--prior-top-percent=*|--prior-scores|--prior-scores=*|--prior-prefiltered-scores|--prior-prefiltered-scores=*|--prior-relcore-manifest|--prior-relcore-manifest=*|--prior-quality-filter-scores|--prior-quality-filter-scores=*)
      custom_prior=true
      ;;
  esac
done

if [[ "${target_only}" == false ]]; then
  if [[ "${custom_weights}" == false ]]; then
    training_args+=(--sample-weights 1 1)
  fi
  if [[ "${custom_prior}" == false ]]; then
    training_args+=(
      --prior-prefiltered-scores /data/dwb/libero90_sqcn/filter/top10pct/scores.csv
    )
  fi
fi

exec python -m qwen3_vl_groot.cli launch \
  "${training_args[@]}" \
  "$@"
