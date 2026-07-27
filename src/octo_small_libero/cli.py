from __future__ import annotations

import argparse
import os

from .config import apply_overrides, load_config, resolved_paths
from .preflight import run_preflight


def _gpu_ids(value: str) -> list[int]:
    try:
        result = [int(piece.strip()) for piece in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from error
    if not result or any(gpu_id < 0 for gpu_id in result):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
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
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu-ids", type=_gpu_ids)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    max_steps = 2 if arguments.smoke_test else arguments.max_steps
    config = apply_overrides(
        config,
        model_path=arguments.model_path,
        lerobot_path=arguments.lerobot_path,
        target_dataset=arguments.target_dataset,
        output_dir=arguments.output_dir,
        gpu_ids=arguments.gpu_ids,
        batch_size=arguments.batch_size,
        max_steps=max_steps,
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
