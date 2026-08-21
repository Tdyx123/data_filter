#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
starvla_launcher="${project_root}/scripts/evaluate_simpler_starvla.sh"
fixed_checkpoint="/data/dwb/models/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
model_dir="$(dirname "$(dirname "${fixed_checkpoint}")")"

usage() {
  cat <<EOF
Usage: scripts/evaluate_simpler_qwen3vl_groot_rt1.sh [launcher and evaluation options]

Fixed checkpoint:
  ${fixed_checkpoint}

Other options are forwarded unchanged to evaluate_simpler_starvla.sh;
--checkpoint and --model-dir cannot override the fixed checkpoint.
EOF
}

fail() {
  printf 'error: %s\n' "$1" >&2
  exit 2
}

for argument in "$@"; do
  case "${argument}" in
    --checkpoint|--checkpoint=*|--model-dir|--model-dir=*)
      fail "fixed checkpoint cannot be overridden with ${argument%%=*}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

[[ -f "${fixed_checkpoint}" ]] ||
  fail "fixed checkpoint does not exist: ${fixed_checkpoint}"

exec "${starvla_launcher}" --model-dir "${model_dir}" "$@"
