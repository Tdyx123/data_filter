"""SimplerEnv client CLI for the managed StarVLA model service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from simpler_bridge.evaluation import (
    OBJECT_EPISODE_IDS,
    POLICY_SEEDS,
    SimplerEvaluationError,
    SimplerInfrastructureError,
    SimplerRunSettings,
    create_simpler_environment,
    default_simpler_root,
    evaluate_simpler_policy,
    parse_sim_device,
    resolve_task_selection,
    run_simpler_preflight,
    validate_simpler_source,
)

from .ipc import StarVLAIPCClient, StarVLAIPCError
from .simpler_evaluation import StarVLARemotePolicy


EVALUATION_ROUTE = "qwen3vl-groot-starvla-simpler-widowx-eval"
PREFLIGHT_ROUTE = "qwen3vl-groot-starvla-simpler-widowx-preflight"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the released Qwen3VL-GR00T StarVLA checkpoint in SimplerEnv."
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--auth-key-hex", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/starvla_simpler_eval"),
    )
    parser.add_argument("--action-horizon", type=int, choices=(1,), default=1)
    parser.add_argument("--sim-device", type=parse_sim_device, default="cuda:0")
    parser.add_argument("--save-videos-path", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    client: StarVLAIPCClient | None = None
    if arguments.overwrite:
        (arguments.output_dir / "failure.json").unlink(missing_ok=True)
    try:
        if arguments.video_fps <= 0:
            raise SimplerEvaluationError("--video-fps must be positive")
        try:
            authkey = bytes.fromhex(arguments.auth_key_hex)
        except ValueError as error:
            raise SimplerEvaluationError("--auth-key-hex must contain valid hex bytes") from error
        if not authkey:
            raise SimplerEvaluationError("--auth-key-hex must be non-empty")
        tasks = resolve_task_selection(arguments.tasks)
        source_versions = validate_simpler_source(default_simpler_root())
        client = StarVLAIPCClient(arguments.socket, authkey=authkey)
        policy = StarVLARemotePolicy(client)
        metadata = policy.metadata
        settings = SimplerRunSettings(
            output_dir=arguments.output_dir,
            tasks=tasks,
            device="remote-pyenv-cuda:0",
            sim_device=arguments.sim_device,
            action_horizon=arguments.action_horizon,
            policy_seeds=(0,) if arguments.smoke_test else POLICY_SEEDS,
            object_episode_ids=(0,) if arguments.smoke_test else OBJECT_EPISODE_IDS,
            max_steps=8 if arguments.smoke_test else None,
            save_videos_path=arguments.save_videos_path,
            video_fps=arguments.video_fps,
            overwrite=arguments.overwrite,
        )
        checkpoint = {
            key: metadata.get(key)
            for key in (
                "model_dir",
                "checkpoint_path",
                "base_model",
                "checkpoint_tensor_count",
                "checkpoint_parameter_bytes",
                "checkpoint_dtypes",
            )
        }
        if arguments.preflight_only:
            report = run_simpler_preflight(
                settings,
                checkpoint=checkpoint,
                policy=policy,
                environment_factory=lambda task: create_simpler_environment(
                    task, sim_device=settings.sim_device
                ),
                source_versions=source_versions,
                package_versions={"numpy": np.__version__},
                route=PREFLIGHT_ROUTE,
            )
        else:
            report = evaluate_simpler_policy(
                settings,
                checkpoint=checkpoint,
                policy=policy,
                environment_factory=lambda task: create_simpler_environment(
                    task, sim_device=settings.sim_device
                ),
                source_versions=source_versions,
                route=EVALUATION_ROUTE,
                protocol_metadata=policy.protocol_metadata(),
            )
    except SimplerInfrastructureError as error:
        _write_failure(arguments.output_dir, error=error, exit_code=3)
        print(f"SimplerEnv infrastructure error: {error}", file=sys.stderr)
        return 3
    except (SimplerEvaluationError, StarVLAIPCError) as error:
        _write_failure(arguments.output_dir, error=error, exit_code=2)
        print(f"StarVLA SimplerEnv evaluation error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        _write_failure(arguments.output_dir, error=error, exit_code=1)
        print(f"StarVLA SimplerEnv execution failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            try:
                client.shutdown()
            except Exception:
                client.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("status") == "completed_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
