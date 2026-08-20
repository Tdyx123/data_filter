"""Command-line entry point for Octo-small BridgeData V2 fine-tuning."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence

from .config import apply_overrides, load_config, resolved_paths
from .preflight import run_preflight


def _gpu_ids(value: str) -> list[int]:
    try:
        result = [int(piece.strip()) for piece in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from error
    if not result or any(gpu_id < 0 for gpu_id in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("GPU IDs must be unique non-negative integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Octo-small directly on BridgeData V2 LeRobot data"
    )
    parser.add_argument(
        "--config", default="configs/octo_small_bridge_v2_4x4090.yaml"
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", type=_gpu_ids)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = build_parser().parse_args(argv)
    if arguments.max_steps is not None and arguments.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if arguments.learning_rate is not None and (
        not math.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0
    ):
        raise SystemExit("--learning-rate must be a positive finite number")
    if arguments.warmup_steps is not None and arguments.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    max_steps = 2 if arguments.smoke_test else arguments.max_steps
    config = apply_overrides(
        config,
        dataset_path=arguments.dataset_path,
        model_path=arguments.model_path,
        output_dir=arguments.output_dir,
        gpu_ids=arguments.gpu_ids,
        max_steps=max_steps,
        learning_rate=arguments.learning_rate,
        warmup_steps=arguments.warmup_steps,
    )
    if arguments.smoke_test:
        config["train"]["log_every_steps"] = 1
        config["train"]["save_every_steps"] = 2
    visible = ",".join(str(value) for value in config["train"]["gpu_ids"])
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    paths = resolved_paths(config)
    if int(os.environ.get("RANK", "0")) == 0:
        run_preflight(config, paths)
    if arguments.preflight_only:
        return
    from .training import train

    train(config, paths, resume=arguments.resume)


if __name__ == "__main__":
    main()
