#!/usr/bin/env python3
"""Train one single-GPU Octo all-tasks model per TDUS weight row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_value = str(import_root)
    if import_value not in sys.path:
        sys.path.insert(0, import_value)

from octo_small_libero.config import load_config  # noqa: E402
from tdus.convert import TDUS_COMPONENTS, convert_scores  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "octo_small_libero_1x4090.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights.jsonl"
DEFAULT_SCORES = PROJECT_ROOT / "outputs" / "tdus" / "libero90" / "chunk" / "scores.csv"
DEFAULT_GPU_IDS = (0, 1, 2, 3)
DEFAULT_PARALLEL = 8
DEFAULT_TOP_PERCENT = 10.0
DEFAULT_SAMPLE_WEIGHTS = (3.0, 1.0)
MANIFEST_VERSION = 1


class SweepError(RuntimeError):
    """Raised when a weight sweep cannot be started safely."""


@dataclass(frozen=True)
class SweepJob:
    line_number: int
    model_id: str
    weights: dict[str, float]
    source_scores_path: Path
    source_scores_sha256: str
    weighted_scores_path: Path
    weighted_scores_sha256: str
    output_dir: Path
    log_path: Path


@dataclass(frozen=True)
class SweepSettings:
    config_path: Path
    output_root: Path
    gpu_ids: tuple[int, ...]
    parallel: int
    prior_top_percent: float
    target_max_steps: int
    sample_weights: tuple[float, float] = DEFAULT_SAMPLE_WEIGHTS
    model_path: Path | None = None
    lerobot_path: Path | None = None
    max_steps_override: int | None = None
    smoke_test: bool = False
    preflight_only: bool = False
    python_executable: str = sys.executable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _top_percent(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("top percent must be numeric") from error
    if not math.isfinite(result) or not 0.0 < result <= 100.0:
        raise argparse.ArgumentTypeError("top percent must be finite and in (0, 100]")
    return result


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(piece.strip()) for piece in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be comma-separated integers"
        ) from error
    if not result or any(gpu_id < 0 for gpu_id in result):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return result


def _percent_tag(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p")


def load_weight_rows(path: str | Path) -> list[dict[str, float]]:
    """Load and validate every JSONL weight row before any training starts."""

    weights_path = Path(path).expanduser().resolve()
    try:
        lines = weights_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SweepError(f"could not read weights file {weights_path}: {error}") from error
    if not lines:
        raise SweepError(f"weights file is empty: {weights_path}")

    expected = set(TDUS_COMPONENTS)
    rows: list[dict[str, float]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise SweepError(f"weights line {line_number} is empty")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise SweepError(
                f"weights line {line_number} is not valid JSON: {error.msg}"
            ) from error
        if not isinstance(raw, dict):
            raise SweepError(f"weights line {line_number} must be a JSON object")
        actual = set(raw)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise SweepError(
                f"weights line {line_number} must contain exactly "
                f"{list(TDUS_COMPONENTS)}; missing={missing}, extra={extra}"
            )

        validated: dict[str, float] = {}
        for name in TDUS_COMPONENTS:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SweepError(
                    f"weights line {line_number} field {name!r} must be numeric"
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise SweepError(
                    f"weights line {line_number} field {name!r} "
                    "must be finite and non-negative"
                )
            validated[name] = number
        total = sum(validated.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise SweepError(
                f"weights line {line_number} must sum to 1, got {total:.12g}"
            )
        rows.append(validated)
    return rows


def prepare_jobs(
    weight_rows: Sequence[Mapping[str, float]],
    *,
    source_scores_path: str | Path,
    output_root: str | Path,
) -> list[SweepJob]:
    """Generate one deterministic weighted score CSV and job descriptor per row."""

    source = Path(source_scores_path).expanduser().resolve()
    if not source.is_file():
        raise SweepError(f"source scores CSV does not exist: {source}")
    output = Path(output_root).expanduser().resolve()
    source_sha256 = _sha256(source)
    jobs: list[SweepJob] = []
    for line_number, raw_weights in enumerate(weight_rows, start=1):
        weights = {name: float(raw_weights[name]) for name in TDUS_COMPONENTS}
        model_id = f"model-{line_number:03d}"
        weighted_scores = source.parent / f"scores_weights_{line_number:03d}.csv"
        try:
            convert_scores(
                source,
                weighted_scores,
                weights=weights,
                force=True,
            )
        except (OSError, ValueError) as error:
            raise SweepError(
                f"could not generate scores for weights line {line_number}: {error}"
            ) from error
        jobs.append(
            SweepJob(
                line_number=line_number,
                model_id=model_id,
                weights=weights,
                source_scores_path=source,
                source_scores_sha256=source_sha256,
                weighted_scores_path=weighted_scores,
                weighted_scores_sha256=_sha256(weighted_scores),
                output_dir=output / model_id,
                log_path=output / "logs" / f"{model_id}.log",
            )
        )
    if _sha256(source) != source_sha256:
        raise SweepError(f"source scores CSV changed while jobs were prepared: {source}")
    return jobs


def _job_manifest(job: SweepJob, settings: SweepSettings) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "line_number": job.line_number,
        "model_id": job.model_id,
        "weights": job.weights,
        "source_scores_path": str(job.source_scores_path),
        "source_scores_sha256": job.source_scores_sha256,
        "weighted_scores_path": str(job.weighted_scores_path),
        "weighted_scores_sha256": job.weighted_scores_sha256,
        "prior_top_percent": settings.prior_top_percent,
        "sample_weights": list(settings.sample_weights),
    }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SweepError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SweepError(f"{description} must be a JSON object: {path}")
    return value


def _validate_existing_manifests(job: SweepJob, settings: SweepSettings) -> None:
    current = _job_manifest(job, settings)
    saved = _load_json(job.output_dir / "sweep_job.json", "sweep job manifest")
    for key in (
        "line_number",
        "model_id",
        "weights",
        "source_scores_sha256",
        "weighted_scores_sha256",
        "prior_top_percent",
        "sample_weights",
    ):
        if saved.get(key) != current[key]:
            raise SweepError(
                f"{job.model_id} existing output conflicts on {key}; "
                "use a different --output-root"
            )

    dataset = _load_json(
        job.output_dir / "dataset_manifest.json",
        "training dataset manifest",
    )
    try:
        datasets = dataset["datasets"]
        prior_dataset = datasets["libero90"]
        prior_selection = prior_dataset["selection"]
        saved_prior_weight = float(prior_dataset["sample_weight"])
        saved_scores_sha256 = prior_selection["scores_sha256"]
        saved_top_percent = float(prior_selection["top_percent"])
        target_datasets = [
            value
            for name, value in datasets.items()
            if name != "libero90"
        ]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise SweepError(
            f"{job.model_id} has an invalid training dataset manifest"
        ) from error
    if saved_scores_sha256 != job.weighted_scores_sha256:
        raise SweepError(
            f"{job.model_id} existing checkpoint used different weighted scores"
        )
    if not math.isclose(
        saved_top_percent,
        settings.prior_top_percent,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SweepError(
            f"{job.model_id} existing checkpoint used a different top percent"
        )
    if len(target_datasets) != 1 or not isinstance(target_datasets[0], dict):
        raise SweepError(
            f"{job.model_id} existing checkpoint has invalid target dataset metadata"
        )
    target_dataset = target_datasets[0]
    target_selection = target_dataset.get("selection")
    if not isinstance(target_selection, dict) or target_selection.get("mode") != "all":
        raise SweepError(
            f"{job.model_id} existing checkpoint is not an all-tasks run"
        )
    try:
        saved_target_weight = float(target_dataset["sample_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise SweepError(
            f"{job.model_id} has invalid dataset sample weights"
        ) from error
    total_weight = sum(settings.sample_weights)
    expected_target_weight = settings.sample_weights[0] / total_weight
    expected_prior_weight = settings.sample_weights[1] / total_weight
    if not (
        math.isclose(
            saved_target_weight,
            expected_target_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            saved_prior_weight,
            expected_prior_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SweepError(
            f"{job.model_id} existing checkpoint used different sample weights"
        )


def _checkpoint_state(output_dir: Path) -> tuple[int, Path]:
    latest_path = output_dir / "checkpoints" / "latest.json"
    latest = _load_json(latest_path, "latest checkpoint pointer")
    try:
        step = int(latest["step"])
        checkpoint_name = str(latest["checkpoint"])
    except (KeyError, TypeError, ValueError) as error:
        raise SweepError(f"invalid latest checkpoint pointer: {latest_path}") from error
    expected_name = f"step-{step:08d}"
    if step <= 0 or checkpoint_name != expected_name:
        raise SweepError(f"invalid latest checkpoint pointer: {latest_path}")
    checkpoint = latest_path.parent / checkpoint_name
    required = (checkpoint / "model.safetensors", checkpoint / "training_state.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SweepError(
            f"latest checkpoint for {output_dir.name} is incomplete; missing={missing}"
        )
    return step, checkpoint


def existing_output_action(job: SweepJob, settings: SweepSettings) -> str:
    """Return ``new``, ``resume``, or ``skip`` for a safely reusable output."""

    if not job.output_dir.exists():
        return "new"
    try:
        has_entries = any(job.output_dir.iterdir())
    except OSError as error:
        raise SweepError(f"could not inspect output {job.output_dir}: {error}") from error
    if not has_entries:
        return "new"
    _validate_existing_manifests(job, settings)
    step, _ = _checkpoint_state(job.output_dir)
    return "skip" if step >= settings.target_max_steps else "resume"


def build_training_command(
    job: SweepJob,
    settings: SweepSettings,
    *,
    gpu_id: int,
    resume: bool,
) -> list[str]:
    command = [
        settings.python_executable,
        "-m",
        "octo_small_libero.cli",
        "--config",
        str(settings.config_path),
        "--all-tasks",
        "--sample-weights",
        format(settings.sample_weights[0], ".12g"),
        format(settings.sample_weights[1], ".12g"),
        "--gpu-ids",
        str(gpu_id),
        "--prior-top-percent",
        format(settings.prior_top_percent, ".12g"),
        "--prior-scores",
        str(job.weighted_scores_path),
        "--output-dir",
        str(job.output_dir),
    ]
    if settings.model_path is not None:
        command.extend(("--model-path", str(settings.model_path)))
    if settings.lerobot_path is not None:
        command.extend(("--lerobot-path", str(settings.lerobot_path)))
    if settings.max_steps_override is not None:
        command.extend(("--max-steps", str(settings.max_steps_override)))
    if settings.smoke_test:
        command.append("--smoke-test")
    if settings.preflight_only:
        command.append("--preflight-only")
    if resume:
        command.extend(("--resume", "latest"))
    return command


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def run_job(job: SweepJob, settings: SweepSettings, gpu_id: int) -> dict[str, Any]:
    started_at = _utc_now()
    base: dict[str, Any] = {
        "line_number": job.line_number,
        "model_id": job.model_id,
        "weights": job.weights,
        "gpu_id": gpu_id,
        "scores_path": str(job.weighted_scores_path),
        "scores_sha256": job.weighted_scores_sha256,
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "started_at": started_at,
    }
    try:
        action = (
            "new"
            if settings.preflight_only
            else existing_output_action(job, settings)
        )
        if action == "skip":
            return {
                **base,
                "status": "skipped",
                "exit_code": 0,
                "resumed": False,
                "finished_at": _utc_now(),
            }
        if not settings.preflight_only and action == "new":
            job.output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(job.output_dir / "sweep_job.json", _job_manifest(job, settings))

        command = build_training_command(
            job,
            settings,
            gpu_id=gpu_id,
            resume=action == "resume",
        )
        _append_log(
            job.log_path,
            f"\n[{started_at}] gpu={gpu_id} command={shlex.join(command)}",
        )
        environment = os.environ.copy()
        python_path = str(PROJECT_ROOT / "src")
        if environment.get("PYTHONPATH"):
            python_path = f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
        environment["PYTHONPATH"] = python_path
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        with job.log_path.open("a", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise SweepError(
                f"training command exited with status {completed.returncode}"
            )
        if not settings.preflight_only:
            final_step, _ = _checkpoint_state(job.output_dir)
            if final_step < settings.target_max_steps:
                raise SweepError(
                    f"training stopped at step {final_step}, "
                    f"expected at least {settings.target_max_steps}"
                )
        return {
            **base,
            "status": (
                "preflight_passed" if settings.preflight_only else "completed"
            ),
            "exit_code": 0,
            "resumed": action == "resume",
            "command": command,
            "finished_at": _utc_now(),
        }
    except (OSError, SweepError) as error:
        _append_log(job.log_path, f"[{_utc_now()}] error={error}")
        return {
            **base,
            "status": "failed",
            "exit_code": 1,
            "resumed": False,
            "error": str(error),
            "finished_at": _utc_now(),
        }


def run_jobs(
    jobs: Sequence[SweepJob],
    settings: SweepSettings,
) -> list[dict[str, Any]]:
    """Run fixed GPU-bound worker slots while allowing every queued job to finish."""

    worker_count = min(settings.parallel, len(jobs))
    assignments: list[list[SweepJob]] = [[] for _ in range(worker_count)]
    for index, job in enumerate(jobs):
        assignments[index % worker_count].append(job)

    results: queue.Queue[dict[str, Any]] = queue.Queue()

    def worker(slot: int) -> None:
        gpu_id = settings.gpu_ids[slot % len(settings.gpu_ids)]
        for job in assignments[slot]:
            print(
                f"[start] {job.model_id} line={job.line_number} gpu={gpu_id}",
                flush=True,
            )
            try:
                result = run_job(job, settings, gpu_id)
            except Exception as error:  # Keep other sweep rows running.
                _append_log(job.log_path, f"[{_utc_now()}] unexpected_error={error}")
                result = {
                    "line_number": job.line_number,
                    "model_id": job.model_id,
                    "weights": job.weights,
                    "gpu_id": gpu_id,
                    "scores_path": str(job.weighted_scores_path),
                    "scores_sha256": job.weighted_scores_sha256,
                    "output_dir": str(job.output_dir),
                    "log_path": str(job.log_path),
                    "status": "failed",
                    "exit_code": 1,
                    "resumed": False,
                    "error": f"unexpected error: {error}",
                    "finished_at": _utc_now(),
                }
            results.put(result)
            print(
                f"[{result['status']}] {job.model_id} "
                f"line={job.line_number} gpu={gpu_id}",
                flush=True,
            )

    threads = [
        threading.Thread(
            target=worker,
            args=(slot,),
            name=f"octo-sweep-{slot}",
        )
        for slot in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ordered = [results.get_nowait() for _ in range(len(jobs))]
    return sorted(ordered, key=lambda result: int(result["line_number"]))


def _sweep_manifest(
    jobs: Sequence[SweepJob],
    settings: SweepSettings,
    weights_path: Path,
) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "created_at": _utc_now(),
        "weights_path": str(weights_path),
        "weights_sha256": _sha256(weights_path),
        "source_scores_path": str(jobs[0].source_scores_path),
        "source_scores_sha256": jobs[0].source_scores_sha256,
        "config_path": str(settings.config_path),
        "output_root": str(settings.output_root),
        "gpu_ids": list(settings.gpu_ids),
        "parallel": settings.parallel,
        "prior_top_percent": settings.prior_top_percent,
        "sample_weights": list(settings.sample_weights),
        "target_max_steps": settings.target_max_steps,
        "jobs": [
            {
                "line_number": job.line_number,
                "model_id": job.model_id,
                "weights": job.weights,
                "weighted_scores_path": str(job.weighted_scores_path),
                "weighted_scores_sha256": job.weighted_scores_sha256,
                "output_dir": str(job.output_dir),
            }
            for job in jobs
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one single-GPU Octo all-tasks model per TDUS JSONL weight row"
        )
    )
    parser.add_argument("--weights-file", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    parser.add_argument("--gpu-ids", type=_gpu_ids, default=DEFAULT_GPU_IDS)
    parser.add_argument("--parallel", type=_positive_int, default=DEFAULT_PARALLEL)
    parser.add_argument(
        "--prior-top-percent",
        type=_top_percent,
        default=DEFAULT_TOP_PERCENT,
    )
    parser.add_argument("--max-steps", type=_positive_int)
    parser.add_argument("--model-path")
    parser.add_argument("--lerobot-path")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        weights_path = _resolve(arguments.weights_file)
        scores_path = _resolve(arguments.scores)
        config_path = _resolve(arguments.config)
        config = load_config(config_path)
        if int(config["train"]["gpu_count"]) != 1:
            raise SweepError(
                f"sweep config must use exactly one GPU per model: {config_path}"
            )
        target_max_steps = (
            2
            if arguments.smoke_test
            else (
                arguments.max_steps
                if arguments.max_steps is not None
                else int(config["train"]["max_steps"])
            )
        )
        output_root = (
            _resolve(arguments.output_root)
            if arguments.output_root is not None
            else PROJECT_ROOT
            / "outputs"
            / (
                "octo_small_libero_all_tasks_weight_sweep_"
                f"top{_percent_tag(arguments.prior_top_percent)}pct"
            )
        ).resolve()
        settings = SweepSettings(
            config_path=config_path,
            output_root=output_root,
            gpu_ids=tuple(arguments.gpu_ids),
            parallel=arguments.parallel,
            prior_top_percent=arguments.prior_top_percent,
            target_max_steps=target_max_steps,
            model_path=(
                _resolve(arguments.model_path)
                if arguments.model_path is not None
                else None
            ),
            lerobot_path=(
                _resolve(arguments.lerobot_path)
                if arguments.lerobot_path is not None
                else None
            ),
            max_steps_override=arguments.max_steps,
            smoke_test=arguments.smoke_test,
            preflight_only=arguments.preflight_only,
        )
        weight_rows = load_weight_rows(weights_path)
        jobs = prepare_jobs(
            weight_rows,
            source_scores_path=scores_path,
            output_root=output_root,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            output_root / "sweep_manifest.json",
            _sweep_manifest(jobs, settings, weights_path),
        )
        results = run_jobs(jobs, settings)
        failed = [result for result in results if result["status"] == "failed"]
        summary = {
            "version": MANIFEST_VERSION,
            "finished_at": _utc_now(),
            "total": len(results),
            "completed": sum(
                result["status"] == "completed" for result in results
            ),
            "preflight_passed": sum(
                result["status"] == "preflight_passed" for result in results
            ),
            "skipped": sum(result["status"] == "skipped" for result in results),
            "failed": len(failed),
            "failed_lines": [result["line_number"] for result in failed],
            "results": results,
        }
        _atomic_json(output_root / "sweep_summary.json", summary)
        print(
            "summary: "
            f"total={summary['total']} completed={summary['completed']} "
            f"preflight_passed={summary['preflight_passed']} "
            f"skipped={summary['skipped']} failed={summary['failed']}",
            flush=True,
        )
        return 1 if failed else 0
    except (OSError, ValueError, SweepError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
