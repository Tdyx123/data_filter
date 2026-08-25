#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pyenv_bin="/home/dwb/.pyenv/bin/pyenv"
source_checkpoint="/data/dwb/models/octo-small"
output_checkpoint="/data/dwb/models/octo-small-pytorch-official"
source_step="270000"

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${pyenv_bin}" exec python -m octo_small_official_pytorch.convert_checkpoint \
  --source "${source_checkpoint}" \
  --output "${output_checkpoint}" \
  --step "${source_step}" \
  "$@"

