"""Command-line entry point for Octo-small Bridge evaluation in SimplerEnv."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from simpler_bridge.evaluation import (
    OBJECT_EPISODE_IDS,
    POLICY_SEEDS,
    SimplerEvaluationError,
    SimplerInfrastructureError,
    SimplerRunSettings,
    create_simpler_environment,
    default_simpler_root,
    evaluate_simpler_policy,
    resolve_task_selection,
    run_simpler_preflight,
    validate_simpler_source,
)

from .simpler_evaluation import (
    load_octo_bridge_policy,
    validate_runtime_contract,
)


EVALUATION_ROUTE = "octo-small-bridge-simpler-widowx-eval"
PREFLIGHT_ROUTE = "octo-small-bridge-simpler-widowx-preflight"


def _write_failure(output_dir: Path, *, error: Exception, exit_code: int) -> None:
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
                "route": EVALUATION_ROUTE,
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
        description="Evaluate an Octo-small Bridge checkpoint on four fixed SimplerEnv tasks."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Concrete Octo Bridge step-XXXXXXXX checkpoint directory.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Self-contained converted Octo-small PyTorch base model.",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        required=True,
        help="Bridge LeRobot meta/stats.json used during fine-tuning.",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all' or a comma-separated subset of spoon,carrot,stack,eggplant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/octo_small_bridge_simpler_eval"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--action-horizon", type=int, choices=range(1, 9), default=8)
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


def _checkpoint_report(checkpoint: object, policy: object) -> dict[str, object]:
    report = dict(checkpoint.as_dict())
    report["statistics"] = policy.statistics.as_dict()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.overwrite:
        (arguments.output_dir / "failure.json").unlink(missing_ok=True)
    try:
        tasks = resolve_task_selection(arguments.tasks)
        if arguments.video_fps <= 0:
            raise SimplerEvaluationError("--video-fps must be positive")
        source_versions = validate_simpler_source(default_simpler_root())
        package_versions = validate_runtime_contract(device=arguments.device)
        checkpoint, policy = load_octo_bridge_policy(
            arguments.checkpoint,
            base_model=arguments.base_model,
            statistics=arguments.statistics,
            device=arguments.device,
            precision=arguments.precision,
        )
        settings = SimplerRunSettings(
            output_dir=arguments.output_dir,
            tasks=tasks,
            device=arguments.device,
            action_horizon=arguments.action_horizon,
            policy_seeds=(0,) if arguments.smoke_test else POLICY_SEEDS,
            object_episode_ids=(0,) if arguments.smoke_test else OBJECT_EPISODE_IDS,
            max_steps=8 if arguments.smoke_test else None,
            save_videos_path=arguments.save_videos_path,
            video_fps=arguments.video_fps,
            overwrite=arguments.overwrite,
        )
        checkpoint_report = _checkpoint_report(checkpoint, policy)
        if arguments.preflight_only:
            report = run_simpler_preflight(
                settings,
                checkpoint=checkpoint_report,
                policy=policy,
                environment_factory=lambda task: create_simpler_environment(task),
                source_versions=source_versions,
                package_versions=package_versions,
                route=PREFLIGHT_ROUTE,
            )
        else:
            report = evaluate_simpler_policy(
                settings,
                checkpoint=checkpoint_report,
                policy=policy,
                environment_factory=lambda task: create_simpler_environment(task),
                source_versions={
                    **source_versions,
                    "package_versions": package_versions,
                },
                route=EVALUATION_ROUTE,
                protocol_metadata=policy.protocol_metadata(),
            )
    except SimplerInfrastructureError as error:
        _write_failure(arguments.output_dir, error=error, exit_code=3)
        print(f"SimplerEnv infrastructure error: {error}", file=sys.stderr)
        return 3
    except SimplerEvaluationError as error:
        _write_failure(arguments.output_dir, error=error, exit_code=2)
        print(f"SimplerEnv evaluation error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        _write_failure(arguments.output_dir, error=error, exit_code=1)
        print(
            f"SimplerEnv task execution failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("status") == "completed_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
