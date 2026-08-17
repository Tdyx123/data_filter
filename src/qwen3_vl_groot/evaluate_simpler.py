from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .simpler_evaluation import (
    OBJECT_EPISODE_IDS,
    POLICY_SEEDS,
    SimplerEvaluationError,
    SimplerEvaluationSettings,
    SimplerInfrastructureError,
    QwenSimplerPolicy,
    create_simpler_environment,
    default_simpler_root,
    evaluate_simpler_checkpoint,
    parse_sim_device,
    resolve_task_selection,
    validate_runtime_contract,
    validate_simpler_source,
)


def _write_failure(
    output_dir: Path,
    *,
    error: Exception,
    exit_code: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "failure.json"
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "route": "qwen3-vl-groot-simpler-widowx-eval",
                "exit_code": exit_code,
                "error": f"{type(error).__name__}: {error}",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen Bridge checkpoint on the four fixed SimplerEnv tasks."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Concrete Qwen step-XXXXXXXX checkpoint directory.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all' or a comma-separated subset of spoon,carrot,stack,eggplant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen_simpler_eval"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-device", type=parse_sim_device, default="cuda:0")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, choices=(1,), default=1)
    parser.add_argument("--save-videos-path", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one seed and one object episode per task for at most eight steps.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_checkpoint_and_policy(arguments: argparse.Namespace):
    try:
        from .libero_evaluation import resolve_qwen_checkpoint
        from octo_small_libero.evaluation import EvaluationError

        checkpoint = resolve_qwen_checkpoint(
            arguments.checkpoint,
            model_path=arguments.model_path,
        )
        policy = QwenSimplerPolicy.from_checkpoint(
            checkpoint,
            device=arguments.device,
        )
    except EvaluationError as error:
        raise SimplerEvaluationError(str(error)) from error
    return checkpoint, policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.overwrite:
        (arguments.output_dir / "failure.json").unlink(missing_ok=True)
    try:
        tasks = resolve_task_selection(arguments.tasks)
        if arguments.denoising_steps <= 0:
            raise SimplerEvaluationError("--denoising-steps must be positive")
        if arguments.video_fps <= 0:
            raise SimplerEvaluationError("--video-fps must be positive")

        source_versions = validate_simpler_source(default_simpler_root())
        package_versions = validate_runtime_contract(device=arguments.device)
        checkpoint, policy = _load_checkpoint_and_policy(arguments)
        settings = SimplerEvaluationSettings(
            checkpoint=arguments.checkpoint,
            output_dir=arguments.output_dir,
            tasks=tasks,
            model_path=arguments.model_path,
            device=arguments.device,
            sim_device=arguments.sim_device,
            denoising_steps=arguments.denoising_steps,
            action_horizon=arguments.action_horizon,
            policy_seeds=(0,) if arguments.smoke_test else POLICY_SEEDS,
            object_episode_ids=(0,) if arguments.smoke_test else OBJECT_EPISODE_IDS,
            max_steps=8 if arguments.smoke_test else None,
            save_videos_path=arguments.save_videos_path,
            video_fps=arguments.video_fps,
            overwrite=arguments.overwrite,
        )
        if arguments.preflight_only:
            from .simpler_evaluation import run_simpler_preflight

            report = run_simpler_preflight(
                settings,
                checkpoint=checkpoint,
                policy=policy,
                environment_factory=lambda task: create_simpler_environment(
                    task, sim_device=settings.sim_device
                ),
                source_versions=source_versions,
                package_versions=package_versions,
            )
        else:
            report = evaluate_simpler_checkpoint(
                settings,
                checkpoint=checkpoint,
                policy=policy,
                environment_factory=lambda task: create_simpler_environment(
                    task, sim_device=settings.sim_device
                ),
                source_versions={**source_versions, "package_versions": package_versions},
            )
    except SimplerInfrastructureError as error:
        _write_failure(
            arguments.output_dir,
            error=error,
            exit_code=3,
        )
        print(f"SimplerEnv infrastructure error: {error}", file=sys.stderr)
        return 3
    except SimplerEvaluationError as error:
        _write_failure(
            arguments.output_dir,
            error=error,
            exit_code=2,
        )
        print(f"SimplerEnv evaluation error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        _write_failure(
            arguments.output_dir,
            error=error,
            exit_code=1,
        )
        print(f"SimplerEnv task execution failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("status") == "completed_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
