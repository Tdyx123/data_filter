from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_config, resolved_paths, save_resolved_config


def positive_finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def parse_gpu_ids(value: str) -> list[int]:
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError("GPU IDs must look like: 0,1,2,3")
    try:
        result = [int(piece) for piece in pieces]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from error
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return result


def configure_visible_gpus(config: dict[str, Any]) -> str | None:
    gpu_ids = config["train"].get("gpu_ids")
    if gpu_ids is None:
        return None
    visible = ",".join(str(value) for value in gpu_ids)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--deepspeed-stage", type=int, choices=(2, 3))
    parser.add_argument("--lora-learning-rate", type=positive_finite_float)
    parser.add_argument("--action-head-learning-rate", type=positive_finite_float)
    parser.add_argument("--lora-freeze-steps", type=int)
    parser.add_argument(
        "--compile-qwen-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-action-head",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def _overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_path": getattr(arguments, "model_path", None),
        "dataset_path": getattr(arguments, "dataset_path", None),
        "output_dir": getattr(arguments, "output_dir", None),
        "gpu_count": getattr(arguments, "gpu_count", None),
        "gpu_ids": getattr(arguments, "gpu_ids", None),
        "micro_batch_size": getattr(arguments, "micro_batch_size", None),
        "gradient_accumulation_steps": getattr(arguments, "gradient_accumulation_steps", None),
        "max_steps": getattr(arguments, "max_steps", None),
        "lora_rank": getattr(arguments, "lora_rank", None),
        "lora_alpha": getattr(arguments, "lora_alpha", None),
        "lora_dropout": getattr(arguments, "lora_dropout", None),
        "deepspeed_stage": getattr(arguments, "deepspeed_stage", None),
        "lora_learning_rate": getattr(arguments, "lora_learning_rate", None),
        "action_head_learning_rate": getattr(arguments, "action_head_learning_rate", None),
        "lora_freeze_steps": getattr(arguments, "lora_freeze_steps", None),
        "compile_qwen_backbone": getattr(arguments, "compile_qwen_backbone", None),
        "compile_action_head": getattr(arguments, "compile_action_head", None),
    }


def _resolve_config(arguments: argparse.Namespace) -> dict[str, Any]:
    config = apply_overrides(load_config(arguments.config), _overrides(arguments))
    for name, path in resolved_paths(config).items():
        if name in {"model", "dataset", "output"}:
            config["paths"][name] = str(path)
    if getattr(arguments, "smoke_test", False):
        config["train"].update(
            {
                "max_steps": 20,
                "log_every_steps": 1,
                "eval_every_steps": 20,
                "save_every_steps": 20,
                "validation_batches": 2,
            }
        )
        config["data"]["num_workers"] = min(int(config["data"]["num_workers"]), 1)
    return config


def launch(arguments: argparse.Namespace) -> None:
    config = _resolve_config(arguments)
    warm_start: str | None = None
    if arguments.warm_start_checkpoint:
        from .checkpointing import inspect_compact_checkpoint

        inspected = inspect_compact_checkpoint(arguments.warm_start_checkpoint, config=config)
        output = Path(config["paths"]["output"])
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Warm-start output must be absent or empty: {output}")
        warm_start = str(inspected.path)
    visible = configure_visible_gpus(config)
    from .preflight import run_preflight

    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    runtime_config = output / "run_config.yaml"
    save_resolved_config(config, runtime_config)
    if visible is not None:
        print(f"Using physical GPU IDs: {visible}", flush=True)
    report = run_preflight(config, memory_probe=not arguments.skip_memory_probe)
    temporary = output / "preflight.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output / "preflight.json")
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
        "qwen_vl_oft.cli",
        "train",
        "--config",
        str(runtime_config),
    ]
    if warm_start is not None:
        command.extend(["--warm-start-checkpoint", warm_start])
    environment = os.environ.copy()
    project_src = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = (
        project_src
        if not environment.get("PYTHONPATH")
        else project_src + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(command, check=True, env=environment)


def distributed_train(arguments: argparse.Namespace) -> None:
    config = load_config(arguments.config)
    for name, path in resolved_paths(config).items():
        if name in {"model", "dataset", "output"}:
            config["paths"][name] = str(path)
    configure_visible_gpus(config)
    from .training import train

    train(config, warm_start_checkpoint=arguments.warm_start_checkpoint)


def inspect_data(arguments: argparse.Namespace) -> None:
    from .preflight import validate_paths_and_data

    print(
        json.dumps(
            validate_paths_and_data(
                _resolve_config(arguments),
                decode_samples=not arguments.no_decode,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-vl-oft",
        description="Train StarVLA-compatible Qwen3-VL-4B OFT on BridgeData V2.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--config", required=True)
    launch_parser.add_argument("--preflight-only", action="store_true")
    launch_parser.add_argument("--skip-memory-probe", action="store_true")
    launch_parser.add_argument("--smoke-test", action="store_true")
    launch_parser.add_argument("--warm-start-checkpoint")
    _add_override_arguments(launch_parser)
    launch_parser.set_defaults(function=launch)

    train_parser = subparsers.add_parser("train", help=argparse.SUPPRESS)
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--warm-start-checkpoint")
    train_parser.set_defaults(function=distributed_train)

    inspect_parser = subparsers.add_parser("inspect-data")
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
