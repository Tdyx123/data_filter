from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_config, resolved_paths, save_resolved_config


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--dit-layers", type=int)
    parser.add_argument("--dit-hidden-size", type=int)
    parser.add_argument("--deepspeed-stage", type=int, choices=(2, 3))


def _overrides(namespace: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "model_path",
        "dataset_path",
        "output_dir",
        "gpu_count",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "dit_layers",
        "dit_hidden_size",
        "deepspeed_stage",
    )
    return {key: getattr(namespace, key, None) for key in keys}


def _resolve_config(arguments: argparse.Namespace) -> dict[str, Any]:
    config = apply_overrides(load_config(arguments.config), _overrides(arguments))
    paths = resolved_paths(config)
    config["paths"]["model"] = str(paths["model"])
    config["paths"]["dataset"] = str(paths["dataset"])
    config["paths"]["output"] = str(paths["output"])
    if getattr(arguments, "smoke_test", False):
        config["train"].update(
            {
                "max_steps": 20,
                "log_every_steps": 1,
                "eval_every_steps": 20,
                "save_every_steps": 20,
                "validation_batches": 2,
                "keep_last_checkpoints": 1,
            }
        )
        config["data"]["num_workers"] = min(int(config["data"]["num_workers"]), 1)
    return config


def launch(arguments: argparse.Namespace) -> None:
    from .preflight import run_preflight

    config = _resolve_config(arguments)
    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    runtime_config = output / "run_config.yaml"
    save_resolved_config(config, runtime_config)

    run_preflight(config, memory_probe=True)
    if arguments.preflight_only:
        return

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(config["train"]["gpu_count"]),
        "-m",
        "qwen3_vl_groot.cli",
        "train",
        "--config",
        str(runtime_config),
    ]
    if arguments.resume:
        command.extend(["--resume", arguments.resume])
    print("Launching:", " ".join(command), flush=True)
    environment = os.environ.copy()
    project_src = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = (
        project_src
        if not environment.get("PYTHONPATH")
        else project_src + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(command, check=True, env=environment)


def distributed_train(arguments: argparse.Namespace) -> None:
    from .training import train

    config = load_config(arguments.config)
    paths = resolved_paths(config)
    for key in ("model", "dataset", "output"):
        config["paths"][key] = str(paths[key])
    train(config, resume=arguments.resume)


def inspect_data(arguments: argparse.Namespace) -> None:
    import json

    from .preflight import validate_paths_and_data

    config = _resolve_config(arguments)
    result = validate_paths_and_data(config, decode_samples=not arguments.no_decode)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-vl-groot",
        description="Train a full-36-layer Qwen3-VL Bridge VLA with a GR00T-style action head.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser(
        "launch", help="preflight and launch distributed training"
    )
    launch_parser.add_argument("--config", required=True)
    launch_parser.add_argument("--preflight-only", action="store_true")
    launch_parser.add_argument("--smoke-test", action="store_true")
    launch_parser.add_argument("--resume", help="latest or a DeepSpeed checkpoint path")
    _add_override_arguments(launch_parser)
    launch_parser.set_defaults(function=launch)

    train_parser = subparsers.add_parser(
        "train", help=argparse.SUPPRESS
    )
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume")
    train_parser.set_defaults(function=distributed_train)

    inspect_parser = subparsers.add_parser(
        "inspect-data", help="validate metadata and first/middle/last episodes"
    )
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--no-decode", action="store_true")
    _add_override_arguments(inspect_parser)
    inspect_parser.set_defaults(function=inspect_data)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()

