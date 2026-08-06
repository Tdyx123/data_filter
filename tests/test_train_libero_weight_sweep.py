import csv
import json
import subprocess
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from octo_small_libero.config import load_config
from scripts import train_libero_octo_small_all_tasks_weight_sweep as sweep


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_source_scores(path: Path) -> bytes:
    path.parent.mkdir(parents=True)
    path.write_text(
        "sample_id,episode_id,start_step,end_step,length,quality,coverage,diversity,novelty,tdus\n"
        "sample-1,0,0,14,15,1.0,0.5,0.25,0.0,0.0\n"
        "sample-2,0,15,29,15,0.2,1.0,0.5,0.1,0.0\n"
        "sample-3,1,0,14,15,0.5,0.5,0.5,0.5,0.0\n",
        encoding="utf-8",
    )
    return path.read_bytes()


def _job(tmp_path: Path, line_number: int) -> sweep.SweepJob:
    model_id = f"model-{line_number:03d}"
    source = tmp_path / "chunk" / "scores.csv"
    weighted = tmp_path / "chunk" / f"scores_weights_{line_number:03d}.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.touch(exist_ok=True)
    weighted.touch(exist_ok=True)
    return sweep.SweepJob(
        line_number=line_number,
        model_id=model_id,
        weights={
            "quality": 0.4,
            "coverage": 0.3,
            "diversity": 0.2,
            "novelty": 0.1,
        },
        source_scores_path=source,
        source_scores_sha256="source-sha",
        weighted_scores_path=weighted,
        weighted_scores_sha256=f"weighted-sha-{line_number}",
        output_dir=tmp_path / "outputs" / model_id,
        log_path=tmp_path / "outputs" / "logs" / f"{model_id}.log",
    )


def _settings(tmp_path: Path, **overrides) -> sweep.SweepSettings:
    values = {
        "config_path": PROJECT_ROOT / "configs" / "octo_small_libero_1x4090.yaml",
        "output_root": tmp_path / "outputs",
        "gpu_ids": (0, 1, 2, 3),
        "parallel": 8,
        "prior_top_percent": 10.0,
        "target_max_steps": 10_000,
    }
    values.update(overrides)
    return sweep.SweepSettings(**values)


def _write_reusable_output(
    job: sweep.SweepJob,
    settings: sweep.SweepSettings,
    *,
    step: int,
) -> None:
    job.output_dir.mkdir(parents=True)
    sweep._atomic_json(
        job.output_dir / "sweep_job.json",
        sweep._job_manifest(job, settings),
    )
    sweep._atomic_json(
        job.output_dir / "dataset_manifest.json",
        {
            "datasets": {
                "libero90": {
                    "sample_weight": 0.25,
                    "selection": {
                        "source_sha256": job.weighted_scores_sha256,
                    }
                },
                "libero10_5": {
                    "sample_weight": 0.75,
                    "selection": {"mode": "all"},
                },
            }
        },
    )
    checkpoint = job.output_dir / "checkpoints" / f"step-{step:08d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "training_state.pt").write_bytes(b"training")
    sweep._atomic_json(
        job.output_dir / "checkpoints" / "latest.json",
        {"checkpoint": checkpoint.name, "step": step},
    )


def test_single_gpu_config_keeps_effective_global_batch_128():
    config = load_config(
        PROJECT_ROOT / "configs" / "octo_small_libero_1x4090.yaml"
    )

    assert config["train"]["gpu_count"] == 1
    assert config["train"]["gpu_ids"] == [0]
    assert config["train"]["micro_batch_size_per_gpu"] == 8
    assert config["train"]["gradient_accumulation_steps"] == 16
    assert config["train"]["batch_size"] == 128


def test_weight_rows_are_fully_validated_before_use(tmp_path):
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        '{"quality": 0.4, "coverage": 0.3, "diversity": 0.2, "novelty": 0.1}\n',
        encoding="utf-8",
    )
    assert sweep.load_weight_rows(valid) == [
        {
            "quality": 0.4,
            "coverage": 0.3,
            "diversity": 0.2,
            "novelty": 0.1,
        }
    ]

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        '{"quality": 0.4, "coverage": 0.3, "diversity": 0.2, '
        '"novelty": 0.1, "extra": 0}\n',
        encoding="utf-8",
    )
    with pytest.raises(sweep.SweepError, match="exactly"):
        sweep.load_weight_rows(invalid)

    invalid.write_text(
        '{"quality": true, "coverage": 0.3, "diversity": 0.2, "novelty": 0.5}\n',
        encoding="utf-8",
    )
    with pytest.raises(sweep.SweepError, match="must be numeric"):
        sweep.load_weight_rows(invalid)


def test_prepare_jobs_reweights_tdus_to_nine_decimals_without_changing_source(
    tmp_path,
):
    source = tmp_path / "tdus" / "libero90" / "chunk" / "scores.csv"
    original = _write_source_scores(source)

    jobs = sweep.prepare_jobs(
        [
            {
                "quality": 0.5,
                "coverage": 0.2,
                "diversity": 0.2,
                "novelty": 0.1,
            }
        ],
        source_scores_path=source,
        output_root=tmp_path / "models",
        prior_top_percent=50,
    )

    assert source.read_bytes() == original
    assert len(jobs) == 1
    assert jobs[0].model_id == "model-001"
    assert jobs[0].weighted_scores_path == source.parent / "scores_weights_001.csv"
    with jobs[0].weighted_scores_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ["sample-1", "sample-3"]
    assert rows[0]["tdus"].partition(".")[2]
    assert len(rows[0]["tdus"].partition(".")[2]) == 9
    assert float(rows[0]["tdus"]) == pytest.approx(0.65, abs=1e-7)


def test_training_command_uses_all_tasks_top10_and_one_physical_gpu(tmp_path):
    job = _job(tmp_path, 1)
    settings = _settings(
        tmp_path,
        target_max_steps=7_500,
        max_steps_override=7_500,
    )
    command = sweep.build_training_command(
        job,
        settings,
        gpu_id=3,
        resume=True,
    )

    assert command[:3] == [
        sweep.sys.executable,
        "-m",
        "octo_small_libero.cli",
    ]
    assert "--all-tasks" in command
    weight_index = command.index("--sample-weights")
    assert command[weight_index + 1 : weight_index + 3] == ["3", "1"]
    assert command[command.index("--gpu-ids") + 1] == "3"
    assert command[command.index("--prior-prefiltered-scores") + 1] == str(
        job.weighted_scores_path
    )
    assert "--prior-top-percent" not in command
    assert "--prior-scores" not in command
    assert command[command.index("--max-steps") + 1] == "7500"
    assert command[-2:] == ["--resume", "latest"]


def test_eight_workers_limit_each_gpu_to_two_jobs_and_continue_after_failure(
    tmp_path,
    monkeypatch,
):
    jobs = [_job(tmp_path, line_number) for line_number in range(1, 17)]
    settings = _settings(tmp_path, preflight_only=True)
    lock = threading.Lock()
    active_total = 0
    active_by_gpu: Counter[int] = Counter()
    maximum_total = 0
    maximum_by_gpu: Counter[int] = Counter()
    calls: list[tuple[list[str], int]] = []

    def fake_run(command, **kwargs):
        nonlocal active_total, maximum_total
        gpu_id = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        with lock:
            active_total += 1
            active_by_gpu[gpu_id] += 1
            maximum_total = max(maximum_total, active_total)
            maximum_by_gpu[gpu_id] = max(
                maximum_by_gpu[gpu_id],
                active_by_gpu[gpu_id],
            )
            calls.append((command, gpu_id))
        time.sleep(0.03)
        with lock:
            active_total -= 1
            active_by_gpu[gpu_id] -= 1
        output_dir = Path(command[command.index("--output-dir") + 1])
        return SimpleNamespace(returncode=7 if output_dir.name == "model-003" else 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    results = sweep.run_jobs(jobs, settings)

    assert len(calls) == 16
    assert maximum_total == 8
    assert maximum_by_gpu == Counter({0: 2, 1: 2, 2: 2, 3: 2})
    assert [result["line_number"] for result in results] == list(range(1, 17))
    assert [result["line_number"] for result in results if result["status"] == "failed"] == [
        3
    ]
    assert all("--all-tasks" in command for command, _ in calls)
    assert all("--sample-weights" in command for command, _ in calls)
    assert all("--preflight-only" in command for command, _ in calls)


def test_existing_completed_job_is_skipped_and_incomplete_job_resumes(tmp_path):
    job = _job(tmp_path, 1)
    settings = _settings(tmp_path)
    _write_reusable_output(job, settings, step=8_000)

    assert sweep.existing_output_action(job, settings) == "resume"
    assert (
        sweep.existing_output_action(
            job,
            replace(settings, target_max_steps=8_000),
        )
        == "skip"
    )


def test_existing_job_rejects_changed_weights_or_incomplete_checkpoint(tmp_path):
    job = _job(tmp_path, 1)
    settings = _settings(tmp_path)
    _write_reusable_output(job, settings, step=8_000)

    changed_job = replace(
        job,
        weights={
            "quality": 0.1,
            "coverage": 0.2,
            "diversity": 0.3,
            "novelty": 0.4,
        },
    )
    with pytest.raises(sweep.SweepError, match="conflicts on weights"):
        sweep.existing_output_action(changed_job, settings)

    checkpoint = job.output_dir / "checkpoints" / "step-00008000"
    (checkpoint / "training_state.pt").unlink()
    with pytest.raises(sweep.SweepError, match="incomplete"):
        sweep.existing_output_action(job, settings)


def test_existing_job_rejects_missing_or_changed_sample_weights(tmp_path):
    job = _job(tmp_path, 1)
    settings = _settings(tmp_path)
    _write_reusable_output(job, settings, step=8_000)

    job_manifest_path = job.output_dir / "sweep_job.json"
    job_manifest = json.loads(job_manifest_path.read_text(encoding="utf-8"))
    job_manifest.pop("sample_weights")
    sweep._atomic_json(job_manifest_path, job_manifest)
    with pytest.raises(sweep.SweepError, match="conflicts on sample_weights"):
        sweep.existing_output_action(job, settings)

    sweep._atomic_json(job_manifest_path, sweep._job_manifest(job, settings))
    dataset_manifest_path = job.output_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["datasets"]["libero10_5"]["sample_weight"] = 0.5
    dataset_manifest["datasets"]["libero90"]["sample_weight"] = 0.5
    sweep._atomic_json(dataset_manifest_path, dataset_manifest)
    with pytest.raises(sweep.SweepError, match="different sample weights"):
        sweep.existing_output_action(job, settings)


def test_existing_nonempty_output_without_checkpoint_is_not_overwritten(tmp_path):
    job = _job(tmp_path, 1)
    settings = _settings(tmp_path)
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "partial.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(sweep.SweepError, match="sweep job manifest"):
        sweep.existing_output_action(job, settings)
    assert (job.output_dir / "partial.txt").read_text(encoding="utf-8") == "keep"


def test_main_writes_failed_summary_and_returns_nonzero(tmp_path, monkeypatch):
    source = tmp_path / "tdus" / "libero90" / "chunk" / "scores.csv"
    _write_source_scores(source)
    weights = tmp_path / "weights.jsonl"
    weights.write_text(
        '{"quality": 0.4, "coverage": 0.3, "diversity": 0.2, "novelty": 0.1}\n'
        '{"quality": 0.1, "coverage": 0.2, "diversity": 0.3, "novelty": 0.4}\n',
        encoding="utf-8",
    )
    output = tmp_path / "models"

    def fake_run_jobs(jobs, settings):
        assert [job.model_id for job in jobs] == ["model-001", "model-002"]
        assert settings.parallel == 8
        assert settings.sample_weights == (3.0, 1.0)
        return [
            {"line_number": 1, "status": "completed"},
            {"line_number": 2, "status": "failed"},
        ]

    monkeypatch.setattr(sweep, "run_jobs", fake_run_jobs)
    exit_code = sweep.main(
        [
            "--weights-file",
            str(weights),
            "--scores",
            str(source),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 1
    summary = json.loads(
        (output / "sweep_summary.json").read_text(encoding="utf-8")
    )
    assert summary["failed"] == 1
    assert summary["failed_lines"] == [2]
