from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evaluation import (
    DEFAULT_CHECKPOINT,
    DEFAULT_STATISTICS,
    DEFAULT_TASK_NAME,
    EvaluationError,
    EvaluationSettings,
    SimulationInfrastructureError,
    UnrecoverableSimulationShutdownError,
    configure_evaluation_multiprocessing,
    evaluate_checkpoint,
)


SIMULATION_INFRASTRUCTURE_EXIT_CODE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an Octo-small PyTorch checkpoint in LIBERO-10"
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--base-model")
    parser.add_argument("--statistics", default=str(DEFAULT_STATISTICS))
    parser.add_argument("--output-dir", default="outputs/octo_small_libero_eval")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument(
        "--episodes",
        type=int,
        default=150,
        help=(
            "Total episodes across the fixed seeds 3471197683, 1232873419, and "
            "1448008435; must be divisible by 3 (default: 150, or 50 initial "
            "states per seed)"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument(
        "--no-auto-reduce-num-envs",
        action="store_false",
        dest="auto_reduce_num_envs",
        help=(
            "Fail immediately instead of retrying smaller worker groups when "
            "offscreen rendering cannot start"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=960)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--record-videos",
        type=int,
        default=0,
        metavar="COUNT",
        help="Record the first COUNT episodes; disabled by default",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--libero-root",
        default=os.environ.get("LIBERO_ROOT"),
        help="Official LIBERO Git checkout pinned to the required commit",
    )
    parser.add_argument("--libero-config-path")
    parser.add_argument(
        "--mujoco-gl",
        choices=("osmesa", "egl"),
        default=os.environ.get("MUJOCO_GL", "egl"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one initial state under each of the three fixed seeds",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    configure_evaluation_multiprocessing()
    parser = build_parser()
    arguments = parser.parse_args()
    os.environ["MUJOCO_GL"] = arguments.mujoco_gl
    episodes = 3 if arguments.smoke_test else arguments.episodes
    num_envs = 1 if arguments.smoke_test else arguments.num_envs
    max_steps = 8 if arguments.smoke_test else arguments.max_steps
    settings = EvaluationSettings(
        checkpoint=Path(arguments.checkpoint),
        base_model=Path(arguments.base_model) if arguments.base_model else None,
        statistics=Path(arguments.statistics),
        output_dir=Path(arguments.output_dir),
        task_name=arguments.task_name,
        episodes=episodes,
        num_envs=num_envs,
        auto_reduce_num_envs=arguments.auto_reduce_num_envs,
        max_steps=max_steps,
        device=arguments.device,
        precision=arguments.precision,
        record_videos=min(arguments.record_videos, episodes),
        video_fps=arguments.video_fps,
        libero_root=Path(arguments.libero_root) if arguments.libero_root else None,
        libero_config_path=(
            Path(arguments.libero_config_path) if arguments.libero_config_path else None
        ),
        overwrite=arguments.overwrite,
    )
    try:
        report = evaluate_checkpoint(
            settings,
            preflight_only=arguments.preflight_only,
        )
    except UnrecoverableSimulationShutdownError as error:
        print(f"simulation infrastructure error: {error}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(SIMULATION_INFRASTRUCTURE_EXIT_CODE)
    except SimulationInfrastructureError as error:
        parser.exit(
            SIMULATION_INFRASTRUCTURE_EXIT_CODE,
            f"simulation infrastructure error: {error}\n",
        )
    except EvaluationError as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
