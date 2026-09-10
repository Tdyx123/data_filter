#!/usr/bin/env bash

run_ablation() {
  if [ "$#" -lt 1 ]; then
    echo "用法：bash run_ablation.sh <实验名称> [消融参数...]" >&2
    return 2
  fi

  local experiment="$1"
  shift

  (
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

    # 基准参数、共享缓存和输出路径统一由独立配置提供。
    # 末尾传入的参数用于覆盖本次实验的消融项。
    python -m cocore_ablation run \
      --config cocore_ablation/config_libero90.yaml \
      --force \
      --subfolder-name "$experiment" \
      "$@"
  )
}

run_ablation "$@"
