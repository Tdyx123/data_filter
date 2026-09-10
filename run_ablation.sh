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
    local -a args=(
      run
      --config cocore_ablation/config_libero90.yaml
      --force
      --subfolder-name "$experiment"
      "$@"
    )
    # stdin 仅用于路径预览；实际流水线必须通过模块入口启动以支持 spawn。
    python - "${args[@]}" <<'PY'
import sys
from pathlib import Path

from cocore_ablation.cli import build_parser
from cocore_ablation.config import load_config

argv = sys.argv[1:]
args = build_parser().parse_args(argv)
config = load_config(args.config)
output_dir = args.output_dir if args.output_dir is not None else config["output"]["directory"]
manifest = (
    Path(output_dir).expanduser()
    / args.subfolder_name
    / "select"
    / "selected_manifest.jsonl"
).resolve()
print(f"selected_manifest_path={manifest}", flush=True)
PY
    local preview_status=$?
    if [ "$preview_status" -ne 0 ]; then
      exit "$preview_status"
    fi
    exec python -m cocore_ablation "${args[@]}"
  )
}

run_ablation "$@"
