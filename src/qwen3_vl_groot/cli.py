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
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def parse_gpu_ids(value: str) -> list[int]:
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError("GPU IDs must look like: 0,1,2,3")
    try:
        gpu_ids = [int(piece) for piece in pieces]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from error
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return gpu_ids


def configure_visible_gpus(config: dict[str, Any]) -> str | None:
    gpu_ids = config["train"].get("gpu_ids")
    if gpu_ids is None:
        return None
    visible_devices = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return visible_devices


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--lerobot-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        help="comma-separated physical GPU numbers, for example 2,3,6,7",
    )
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--qwen-context-forward",
        dest="context_forward",
        choices=("causal_lm", "backbone"),
    )
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
    parser.add_argument("--episode-cache-size", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--dit-layers", type=int)
    parser.add_argument("--dit-hidden-size", type=int)
    parser.add_argument("--deepspeed-stage", type=int, choices=(2, 3))
    parser.add_argument("--lora-learning-rate", type=positive_finite_float)
    parser.add_argument("--action-head-learning-rate", type=positive_finite_float)
    parser.add_argument("--lora-freeze-steps", type=int)
    parser.add_argument("--lora-cycle-steps", type=int)
    parser.add_argument("--lora-active-steps", type=int)
    parser.add_argument("--all-tasks", action="store_true", default=None)
    parser.add_argument("--target-only", action="store_true", default=None)
    parser.add_argument(
        "--sample-weights",
        type=positive_finite_float,
        nargs=2,
        metavar=("TARGET", "PRIOR"),
    )
    parser.add_argument("--prior-top-percent", type=positive_finite_float)
    parser.add_argument("--prior-scores")
    parser.add_argument("--prior-prefiltered-scores")
    parser.add_argument("--prior-relcore-manifest")
    parser.add_argument("--prior-quality-filter-scores")


def _overrides(namespace: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "model_path",
        "dataset_path",
        "lerobot_path",
        "output_dir",
        "gpu_count",
        "gpu_ids",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "context_forward",
        "compile_qwen_backbone",
        "compile_action_head",
        "episode_cache_size",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "dit_layers",
        "dit_hidden_size",
        "deepspeed_stage",
        "lora_learning_rate",
        "action_head_learning_rate",
        "lora_freeze_steps",
        "lora_cycle_steps",
        "lora_active_steps",
        "target_all_tasks",
        "target_only",
        "sample_weights",
        "prior_top_percent",
        "prior_scores",
        "prior_prefiltered_scores",
        "prior_relcore_manifest",
        "prior_quality_filter_scores",
    )
    values = {key: getattr(namespace, key, None) for key in keys}
    values["target_all_tasks"] = getattr(namespace, "all_tasks", None)
    return values


def _resolve_config(arguments: argparse.Namespace) -> dict[str, Any]:
    prior_values = (
        getattr(arguments, "prior_top_percent", None),
        getattr(arguments, "prior_scores", None),
        getattr(arguments, "prior_prefiltered_scores", None),
        getattr(arguments, "prior_relcore_manifest", None),
        getattr(arguments, "prior_quality_filter_scores", None),
    )
    if getattr(arguments, "target_only", False) and (
        getattr(arguments, "sample_weights", None) is not None
        or any(value is not None for value in prior_values)
    ):
        raise ValueError("--target-only cannot be combined with --sample-weights or --prior-*")
    selected_prior_modes = sum(
        value is not None
        for value in (
            getattr(arguments, "prior_top_percent", None),
            getattr(arguments, "prior_prefiltered_scores", None),
            getattr(arguments, "prior_relcore_manifest", None),
            getattr(arguments, "prior_quality_filter_scores", None),
        )
    )
    if selected_prior_modes > 1:
        raise ValueError("only one --prior-* selection mode may be used")
    config = apply_overrides(load_config(arguments.config), _overrides(arguments))
    paths = resolved_paths(config)
    for key in ("model", "dataset", "lerobot", "output"):
        if key in paths:
            config["paths"][key] = str(paths[key])
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
    visible_devices = configure_visible_gpus(config)
    # Importing preflight imports torch. CUDA_VISIBLE_DEVICES must be set first so
    # both the memory probe and torchrun see the requested physical GPUs.
    from .preflight import run_preflight

    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    runtime_config = output / "run_config.yaml"
    save_resolved_config(config, runtime_config)

    if visible_devices is not None:
        print(f"Using physical GPU IDs: {visible_devices}", flush=True)
    preflight_report = run_preflight(
        config,
        memory_probe=not arguments.skip_memory_probe,
    )
    temporary_report = output / "preflight.json.tmp"
    temporary_report.write_text(
        json.dumps(preflight_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(output / "preflight.json")
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
    config = load_config(arguments.config)
    paths = resolved_paths(config)
    for key in ("model", "dataset", "lerobot", "output"):
        if key in paths:
            config["paths"][key] = str(paths[key])
    configure_visible_gpus(config)
    from .training import train

    train(config)


def inspect_data(arguments: argparse.Namespace) -> None:
    import json

    from .preflight import validate_paths_and_data

    config = _resolve_config(arguments)
    result = validate_paths_and_data(config, decode_samples=not arguments.no_decode)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-vl-groot",
        description="Train a Qwen3-VL or Qwen3.5 VLA with a GR00T-style action head.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser(
        "launch", help="preflight and launch distributed training"
    )
    launch_parser.add_argument("--config", required=True)
    launch_parser.add_argument("--preflight-only", action="store_true")
    launch_parser.add_argument("--skip-memory-probe", action="store_true")
    launch_parser.add_argument("--smoke-test", action="store_true")
    _add_override_arguments(launch_parser)
    launch_parser.set_defaults(function=launch)

    train_parser = subparsers.add_parser(
        "train", help=argparse.SUPPRESS
    )
    train_parser.add_argument("--config", required=True)
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
