from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from .config import apply_overrides, load_config, resolved_paths
from .libero10_tasks import LIBERO_10_TASK_COUNT
from .preflight import run_preflight


def _gpu_ids(value: str) -> list[int]:
    try:
        result = [int(piece.strip()) for piece in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from error
    if not result or any(gpu_id < 0 for gpu_id in result):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    return result


def _task_index(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Task index must be an integer") from error
    if not 0 <= result < LIBERO_10_TASK_COUNT:
        raise argparse.ArgumentTypeError(
            f"Task index must be in [0, {LIBERO_10_TASK_COUNT - 1}]"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Octo-small LIBERO LeRobot v2 co-training route"
    )
    parser.add_argument(
        "--config",
        default="configs/octo_small_libero_4x4090.yaml",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--lerobot-path")
    parser.add_argument("--target-dataset")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--task-index", type=_task_index)
    target.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", type=_gpu_ids)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--sample-weights",
        type=float,
        nargs=2,
        metavar=("TARGET", "PRIOR"),
    )
    parser.add_argument("--prior-top-percent", type=float)
    parser.add_argument("--prior-scores")
    parser.add_argument("--prior-prefiltered-scores")
    parser.add_argument("--prior-relcore-manifest")
    parser.add_argument("--target-only", action="store_true", default=None)
    parser.add_argument("--resume")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.prior_prefiltered_scores is not None and (
        arguments.prior_top_percent is not None
        or arguments.prior_scores is not None
    ):
        parser.error(
            "--prior-prefiltered-scores cannot be combined with "
            "--prior-top-percent or --prior-scores"
        )
    if arguments.prior_relcore_manifest is not None and any(
        value is not None
        for value in (
            arguments.prior_top_percent,
            arguments.prior_scores,
            arguments.prior_prefiltered_scores,
        )
    ):
        parser.error(
            "--prior-relcore-manifest cannot be combined with "
            "--prior-top-percent, --prior-scores, or --prior-prefiltered-scores"
        )
    if arguments.target_only and any(
        value is not None
        for value in (
            arguments.sample_weights,
            arguments.prior_top_percent,
            arguments.prior_scores,
            arguments.prior_prefiltered_scores,
            arguments.prior_relcore_manifest,
        )
    ):
        parser.error(
            "--target-only cannot be combined with --sample-weights or --prior-* options"
        )
    return arguments


def main() -> None:
    arguments = parse_arguments()

    config = load_config(arguments.config)
    max_steps = 2 if arguments.smoke_test else arguments.max_steps
    config = apply_overrides(
        config,
        model_path=arguments.model_path,
        lerobot_path=arguments.lerobot_path,
        target_dataset=arguments.target_dataset,
        target_task_index=arguments.task_index,
        target_all_tasks=arguments.all_tasks,
        target_only=arguments.target_only,
        output_dir=arguments.output_dir,
        gpu_ids=arguments.gpu_ids,
        batch_size=arguments.batch_size,
        max_steps=max_steps,
        sample_weights=arguments.sample_weights,
        prior_top_percent=arguments.prior_top_percent,
        prior_scores=arguments.prior_scores,
        prior_prefiltered_scores=arguments.prior_prefiltered_scores,
        prior_relcore_manifest=arguments.prior_relcore_manifest,
    )
    if arguments.smoke_test:
        config["train"]["log_every_steps"] = 1
        config["train"]["save_every_steps"] = 2

    visible = ",".join(str(gpu_id) for gpu_id in config["train"]["gpu_ids"])
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    paths = resolved_paths(config)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        run_preflight(config, paths)
    if arguments.preflight_only:
        return

    from .training import train

    train(config, paths, resume=arguments.resume)


if __name__ == "__main__":
    main()
