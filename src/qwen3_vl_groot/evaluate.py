from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from octo_small_libero.evaluation import (
    DEFAULT_TASK_NAME,
    EvaluationError,
    SimulationInfrastructureError,
    UnrecoverableSimulationShutdownError,
    configure_evaluation_multiprocessing,
)

from .libero_evaluation import QwenEvaluationSettings, evaluate_qwen_checkpoint


SIMULATION_INFRASTRUCTURE_EXIT_CODE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a compact Qwen3-VL or Qwen3.5 GROOT LoRA checkpoint "
            "in LIBERO-10"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", default="outputs/qwen_libero_eval")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument(
        "--no-auto-reduce-num-envs",
        action="store_false",
        dest="auto_reduce_num_envs",
    )
    parser.add_argument("--max-steps", type=int, default=960)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--policy-batch-size", type=int, default=4)
    parser.add_argument("--record-videos", type=int, default=0, metavar="COUNT")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT"))
    parser.add_argument("--libero-config-path")
    parser.add_argument(
        "--mujoco-gl",
        choices=("osmesa", "egl"),
        default=os.environ.get("MUJOCO_GL", "egl"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
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
    settings = QwenEvaluationSettings(
        checkpoint=Path(arguments.checkpoint),
        model_path=Path(arguments.model_path) if arguments.model_path else None,
        output_dir=Path(arguments.output_dir),
        task_name=arguments.task_name,
        episodes=episodes,
        num_envs=num_envs,
        auto_reduce_num_envs=arguments.auto_reduce_num_envs,
        max_steps=max_steps,
        device=arguments.device,
        denoising_steps=arguments.denoising_steps,
        policy_batch_size=arguments.policy_batch_size,
        record_videos=min(arguments.record_videos, episodes),
        video_fps=arguments.video_fps,
        libero_root=Path(arguments.libero_root) if arguments.libero_root else None,
        libero_config_path=(
            Path(arguments.libero_config_path) if arguments.libero_config_path else None
        ),
        overwrite=arguments.overwrite,
    )
    try:
        report = evaluate_qwen_checkpoint(
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
