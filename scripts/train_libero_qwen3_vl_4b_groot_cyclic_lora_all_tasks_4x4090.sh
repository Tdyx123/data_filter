#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh" \
  --lora-freeze-steps 5000 \
  --lora-cycle-steps 100 \
  --lora-active-steps 10 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --no-compile-qwen-backbone \
  --episode-cache-size 2 \
  "$@"
