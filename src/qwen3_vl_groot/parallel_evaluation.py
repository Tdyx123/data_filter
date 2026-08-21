"""Multi-replica orchestration for Qwen SimplerEnv evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from simpler_bridge.evaluation import (
    SimplerTaskSpec,
    _atomic_write_json,
    _atomic_write_jsonl,
    _summary,
    episode_inference_seed,
    parse_sim_device,
    resolve_task_selection,
)

from .ipc import QwenIPCClient


class ParallelEvaluationError(RuntimeError):
    """Raised when parallel worker state cannot produce a valid evaluation."""

    def __init__(self, message: str, *, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = int(exit_code)


@dataclass
class _Replica:
    index: int
    device: str
    socket_path: Path
    authkey: bytes
    log_path: Path
    log_handle: Any
    model_process: subprocess.Popen[Any]
    sim_process: subprocess.Popen[Any] | None = None


def parse_model_devices(value: str) -> tuple[str, ...]:
    devices = tuple(str(value).split(","))
    if (
        not devices
        or any(re.fullmatch(r"cuda:[0-9]+", device) is None for device in devices)
        or len(devices) != len(set(devices))
    ):
        raise ValueError(
            "model devices must be a non-empty comma-separated list of unique cuda:<index> values"
        )
    return devices


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sharded Qwen SimplerEnv evaluation across model replicas."
    )
    parser.add_argument("--model-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-devices", type=parse_model_devices, required=True)
    parser.add_argument("--sim-device", type=parse_sim_device, required=True)
    parser.add_argument("--denoising-steps", type=_positive_integer, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-timeout", type=_positive_integer, default=600)
    parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    return parser


def _evaluation_arguments(arguments: argparse.Namespace) -> argparse.Namespace:
    from .evaluate_simpler import build_parser as build_evaluation_parser

    forwarded = list(arguments.evaluation_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    return build_evaluation_parser().parse_args(
        [
            "--socket",
            "/tmp/qwen-parallel-planning.sock",
            "--auth-key-hex",
            "00",
            "--output-dir",
            str(arguments.output_dir),
            "--sim-device",
            arguments.sim_device,
            *forwarded,
        ]
    )


def _forwarded_arguments(arguments: argparse.Namespace) -> list[str]:
    forwarded = list(arguments.evaluation_args)
    return forwarded[1:] if forwarded[:1] == ["--"] else forwarded


def _process_group_signal(process: subprocess.Popen[Any], signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return


def _terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    _process_group_signal(process, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        _process_group_signal(process, signal.SIGKILL)
        process.wait(timeout=3)


def _wait_for_model_servers(replicas: Sequence[_Replica], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    pending = {replica.index for replica in replicas}
    while pending:
        for replica in replicas:
            if replica.index not in pending:
                continue
            status = replica.model_process.poll()
            if status is not None:
                raise ParallelEvaluationError(
                    f"model worker {replica.index} on {replica.device} exited before ready "
                    f"with status {status}; log: {replica.log_path}",
                    exit_code=status or 1,
                )
            if replica.socket_path.is_socket():
                pending.remove(replica.index)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise ParallelEvaluationError(
                f"timed out after {timeout}s waiting for model workers {sorted(pending)}"
            )
        time.sleep(0.05)


def _validate_replica_metadata(replicas: Sequence[_Replica]) -> None:
    reference: dict[str, Any] | None = None
    for replica in replicas:
        client = QwenIPCClient(replica.socket_path, authkey=replica.authkey)
        try:
            metadata = client.metadata()
        finally:
            client.close()
        if str(metadata.get("device")) != replica.device:
            raise ParallelEvaluationError(
                f"model worker {replica.index} reported device {metadata.get('device')!r}; "
                f"expected {replica.device!r}"
            )
        normalized = _without(metadata, "device")
        if reference is None:
            reference = normalized
        elif normalized != reference:
            raise ParallelEvaluationError("model replica metadata does not match")


def _shutdown_idle_replica(replica: _Replica) -> None:
    if replica.model_process.poll() is not None:
        return
    client = QwenIPCClient(replica.socket_path, authkey=replica.authkey)
    try:
        client.shutdown()
    except Exception:
        client.close()


def _wait_for_simulators(replicas: Sequence[_Replica]) -> None:
    pending = {replica.index for replica in replicas if replica.sim_process is not None}
    while pending:
        made_progress = False
        for replica in replicas:
            if replica.index not in pending:
                continue
            assert replica.sim_process is not None
            sim_status = replica.sim_process.poll()
            if sim_status is not None:
                pending.remove(replica.index)
                made_progress = True
                if sim_status != 0:
                    raise ParallelEvaluationError(
                        f"simulator worker {replica.index} exited with status {sim_status}",
                        exit_code=sim_status,
                    )
                continue
            model_status = replica.model_process.poll()
            if model_status not in {None, 0}:
                raise ParallelEvaluationError(
                    f"model worker {replica.index} exited during evaluation with status "
                    f"{model_status}; log: {replica.log_path}",
                    exit_code=model_status or 1,
                )
        if pending and not made_progress:
            time.sleep(0.05)


def _write_failure(
    output_dir: Path,
    *,
    error: BaseException,
    exit_code: int,
    replicas: Sequence[_Replica] = (),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "failure.json"
    if path.exists():
        return
    report = {
        "schema_version": 1,
        "status": "failed",
        "route": "qwen3-vl-groot-simpler-widowx-parallel-eval",
        "exit_code": int(exit_code),
        "error": f"{type(error).__name__}: {error}",
        "workers": [
            {
                "worker_index": replica.index,
                "model_device": replica.device,
                "model_log": str(replica.log_path),
            }
            for replica in replicas
        ],
    }
    _atomic_write_json(path, report)


def _check_output_targets(output_dir: Path, worker_count: int, overwrite: bool) -> None:
    protected = [
        output_dir / "results.json",
        output_dir / "episodes.jsonl",
        output_dir / "preflight.json",
        output_dir / "failure.json",
        *(
            output_dir / ".parallel" / f"worker-{index:02d}" / "results.json"
            for index in range(worker_count)
        ),
    ]
    if not overwrite and any(path.exists() for path in protected):
        raise ParallelEvaluationError("parallel evaluation output already exists; use --overwrite")


def _augment_preflight(
    output_dir: Path,
    worker_dir: Path,
    *,
    requested_devices: tuple[str, ...],
    sim_device: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    report = _load_json(worker_dir / "preflight.json")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ParallelEvaluationError("preflight runtime metadata is invalid")
    report["runtime"] = {
        **dict(runtime),
        "elapsed_seconds": float(elapsed_seconds),
        "requested_model_devices": list(requested_devices),
        "model_devices": list(requested_devices),
        "worker_count": len(requested_devices),
        "sim_device": str(sim_device),
    }
    _atomic_write_json(output_dir / "preflight.json", report)
    (output_dir / "failure.json").unlink(missing_ok=True)
    return report


def run_parallel(arguments: argparse.Namespace) -> dict[str, Any]:
    evaluation = _evaluation_arguments(arguments)
    tasks = resolve_task_selection(evaluation.tasks)
    policy_seeds = (0,) if evaluation.smoke_test else (0, 2, 4)
    object_episode_ids = (0,) if evaluation.smoke_test else tuple(range(24))
    planned_episodes = len(tasks) * len(policy_seeds) * len(object_episode_ids)
    requested_devices = tuple(arguments.model_devices)
    active_count = (
        len(requested_devices)
        if evaluation.preflight_only
        else min(len(requested_devices), planned_episodes)
    )
    active_devices = requested_devices[:active_count]
    _check_output_targets(arguments.output_dir, active_count, evaluation.overwrite)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if evaluation.overwrite:
        (arguments.output_dir / "failure.json").unlink(missing_ok=True)
    worker_root = arguments.output_dir / ".parallel"
    worker_root.mkdir(parents=True, exist_ok=True)
    ipc_root = Path(tempfile.mkdtemp(prefix="qwen-simpler-parallel."))
    replicas: list[_Replica] = []
    started = time.monotonic()
    try:
        for index, device in enumerate(active_devices):
            authkey = secrets.token_bytes(32)
            socket_path = ipc_root / f"model-{index:02d}.sock"
            log_path = arguments.output_dir / f"model-server-{index:02d}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            command = [
                str(arguments.model_python),
                "-m",
                "qwen3_vl_groot.server",
                "--socket",
                str(socket_path),
                "--auth-key-hex",
                authkey.hex(),
                "--checkpoint",
                str(arguments.checkpoint),
                "--device",
                device,
                "--denoising-steps",
                str(arguments.denoising_steps),
            ]
            if arguments.model_path is not None:
                command.extend(("--model-path", str(arguments.model_path)))
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            replicas.append(
                _Replica(
                    index=index,
                    device=device,
                    socket_path=socket_path,
                    authkey=authkey,
                    log_path=log_path,
                    log_handle=log_handle,
                    model_process=process,
                )
            )

        _wait_for_model_servers(replicas, arguments.server_timeout)
        _validate_replica_metadata(replicas)
        forwarded = _forwarded_arguments(arguments)
        simulator_replicas = replicas[:1] if evaluation.preflight_only else replicas
        shard_count = 1 if evaluation.preflight_only else len(simulator_replicas)
        for replica in simulator_replicas:
            worker_dir = worker_root / f"worker-{replica.index:02d}"
            command = [
                str(arguments.sim_python),
                "-m",
                "qwen3_vl_groot.evaluate_simpler",
                "--socket",
                str(replica.socket_path),
                "--auth-key-hex",
                replica.authkey.hex(),
                "--output-dir",
                str(worker_dir),
                "--sim-device",
                arguments.sim_device,
                "--shard-index",
                str(replica.index),
                "--shard-count",
                str(shard_count),
                "--rng-scope",
                "per_episode",
                *forwarded,
            ]
            replica.sim_process = subprocess.Popen(command, start_new_session=True)
        _wait_for_simulators(simulator_replicas)
        for replica in replicas[len(simulator_replicas) :]:
            _shutdown_idle_replica(replica)
        for replica in replicas:
            try:
                replica.model_process.wait(timeout=5)
            except subprocess.TimeoutExpired as error:
                raise ParallelEvaluationError(
                    f"model worker {replica.index} did not stop after evaluation"
                ) from error
            if replica.model_process.returncode != 0:
                raise ParallelEvaluationError(
                    f"model worker {replica.index} exited with status "
                    f"{replica.model_process.returncode}; log: {replica.log_path}",
                    exit_code=replica.model_process.returncode or 1,
                )
        elapsed = time.monotonic() - started
        if evaluation.preflight_only:
            return _augment_preflight(
                arguments.output_dir,
                worker_root / "worker-00",
                requested_devices=requested_devices,
                sim_device=arguments.sim_device,
                elapsed_seconds=elapsed,
            )
        return merge_worker_outputs(
            arguments.output_dir,
            tuple(worker_root / f"worker-{index:02d}" for index in range(active_count)),
            tasks=tasks,
            policy_seeds=policy_seeds,
            object_episode_ids=object_episode_ids,
            requested_devices=requested_devices,
            active_devices=active_devices,
            sim_device=arguments.sim_device,
            elapsed_seconds=elapsed,
        )
    except BaseException as error:
        if isinstance(error, ParallelEvaluationError):
            exit_code = error.exit_code
        elif isinstance(error, KeyboardInterrupt):
            exit_code = 130
        else:
            exit_code = 1
        _write_failure(
            arguments.output_dir,
            error=error,
            exit_code=exit_code,
            replicas=replicas,
        )
        raise
    finally:
        for replica in replicas:
            _terminate_process(replica.sim_process)
        for replica in replicas:
            _terminate_process(replica.model_process)
            replica.log_handle.close()
        shutil.rmtree(ipc_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        report = run_parallel(arguments)
    except KeyboardInterrupt as error:
        _write_failure(arguments.output_dir, error=error, exit_code=130)
        return 130
    except ParallelEvaluationError as error:
        print(f"Qwen parallel evaluation error: {error}", file=sys.stderr)
        return error.exit_code
    except Exception as error:
        print(
            f"Qwen parallel evaluation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParallelEvaluationError(f"Could not read worker result {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ParallelEvaluationError(f"Worker result must be a mapping: {path}")
    return dict(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as error:
        raise ParallelEvaluationError(f"Could not read worker episodes {path}: {error}") from error
    if any(not isinstance(value, Mapping) for value in values):
        raise ParallelEvaluationError(f"Worker episodes must be mappings: {path}")
    return [dict(value) for value in values]


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: value for key, value in mapping.items() if key not in omitted}


def merge_worker_outputs(
    output_dir: str | Path,
    worker_dirs: Sequence[str | Path],
    *,
    tasks: tuple[SimplerTaskSpec, ...],
    policy_seeds: tuple[int, ...],
    object_episode_ids: tuple[int, ...],
    requested_devices: tuple[str, ...],
    active_devices: tuple[str, ...],
    sim_device: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Validate successful shards and write one canonical evaluation report."""

    directories = tuple(Path(path) for path in worker_dirs)
    if not directories:
        raise ParallelEvaluationError("At least one worker output is required")
    if len(directories) != len(active_devices):
        raise ParallelEvaluationError("worker output count must match active model devices")

    episodes_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    base_checkpoint: dict[str, Any] | None = None
    base_route: str | None = None
    base_protocol: dict[str, Any] | None = None
    base_runtime: dict[str, Any] | None = None
    videos: list[str] = []

    for worker_index, directory in enumerate(directories):
        report = _load_json(directory / "results.json")
        if report.get("status") != "complete" or report.get("task_errors"):
            raise ParallelEvaluationError(f"worker {worker_index} did not complete successfully")
        checkpoint = report.get("checkpoint")
        protocol = report.get("protocol")
        runtime = report.get("runtime")
        if not isinstance(checkpoint, Mapping):
            raise ParallelEvaluationError(f"worker {worker_index} checkpoint is invalid")
        if not isinstance(protocol, Mapping):
            raise ParallelEvaluationError(f"worker {worker_index} protocol is invalid")
        if not isinstance(runtime, Mapping):
            raise ParallelEvaluationError(f"worker {worker_index} runtime is invalid")
        if protocol.get("shard_index") != worker_index:
            raise ParallelEvaluationError(f"worker {worker_index} shard_index is invalid")
        if protocol.get("shard_count") != len(directories):
            raise ParallelEvaluationError(f"worker {worker_index} shard_count is invalid")

        normalized_checkpoint = dict(checkpoint)
        normalized_route = str(report.get("route", ""))
        normalized_protocol = _without(protocol, "shard_index")
        normalized_runtime = _without(runtime, "device", "elapsed_seconds")
        if base_checkpoint is None:
            base_checkpoint = normalized_checkpoint
            base_route = normalized_route
            base_protocol = normalized_protocol
            base_runtime = normalized_runtime
        else:
            if normalized_checkpoint != base_checkpoint:
                raise ParallelEvaluationError("worker checkpoint metadata does not match")
            if normalized_route != base_route:
                raise ParallelEvaluationError("worker evaluation routes do not match")
            if normalized_protocol != base_protocol:
                raise ParallelEvaluationError("worker protocol metadata does not match")
            if normalized_runtime != base_runtime:
                raise ParallelEvaluationError("worker runtime metadata does not match")

        report_videos = report.get("videos", [])
        if not isinstance(report_videos, list) or any(
            not isinstance(value, str) for value in report_videos
        ):
            raise ParallelEvaluationError(f"worker {worker_index} videos are invalid")
        videos.extend(report_videos)

        for episode in _load_jsonl(directory / "episodes.jsonl"):
            try:
                key = (
                    str(episode["task"]),
                    int(episode["policy_seed"]),
                    int(episode["object_episode_id"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ParallelEvaluationError(
                    f"worker {worker_index} episode identity is invalid"
                ) from error
            if key in episodes_by_key:
                raise ParallelEvaluationError(f"duplicate episode assignment: {key}")
            expected_seed = episode_inference_seed(*key)
            if episode.get("inference_seed") != expected_seed:
                raise ParallelEvaluationError(f"episode {key} has an invalid inference_seed")
            episodes_by_key[key] = episode

    expected_keys = tuple(
        (task.key, int(policy_seed), int(object_episode_id))
        for task in tasks
        for policy_seed in policy_seeds
        for object_episode_id in object_episode_ids
    )
    expected_key_set = set(expected_keys)
    missing = [key for key in expected_keys if key not in episodes_by_key]
    unexpected = [key for key in episodes_by_key if key not in expected_key_set]
    if missing or unexpected:
        raise ParallelEvaluationError(
            f"worker episode coverage mismatch: missing={missing}, unexpected={unexpected}"
        )

    assert base_checkpoint is not None
    assert base_route is not None
    assert base_protocol is not None
    assert base_runtime is not None
    episodes = [episodes_by_key[key] for key in expected_keys]
    protocol = _without(base_protocol, "shard_count")
    protocol.update(
        {
            "parallel_workers": len(directories),
            "model_devices": list(active_devices),
            "environment_lifecycle": "one_per_task_per_worker",
        }
    )
    runtime = {
        **base_runtime,
        "device": "parallel-model-replicas",
        "elapsed_seconds": float(elapsed_seconds),
        "requested_model_devices": list(requested_devices),
        "model_devices": list(active_devices),
        "worker_count": len(directories),
        "sim_device": str(sim_device),
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "route": base_route,
        "checkpoint": base_checkpoint,
        "protocol": protocol,
        "summary": _summary(episodes, tasks),
        "task_errors": [],
        "runtime": runtime,
        "videos": sorted(set(videos)),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(target / "episodes.jsonl", episodes)
    _atomic_write_json(target / "results.json", report)
    (target / "failure.json").unlink(missing_ok=True)
    (target / "episodes.partial.jsonl").unlink(missing_ok=True)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
