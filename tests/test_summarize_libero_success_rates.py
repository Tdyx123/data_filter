import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "summarize_libero_success_rates.py"


def _write_experiment(root: Path, name: str, rates: list[float]) -> Path:
    experiment = root / name
    for task_index, rate in enumerate(rates):
        task_dir = experiment / f"task-{task_index}"
        task_dir.mkdir(parents=True)
        (task_dir / "results.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "protocol": {"episodes": 150},
                    "summary": {
                        "completed_episodes": 150,
                        "success_rate": rate,
                    },
                }
            ),
            encoding="utf-8",
        )
    return experiment


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_outputs_only_mean_success_rate_in_stable_order(tmp_path: Path):
    _write_experiment(tmp_path, "beta", [0.2] * 10)
    _write_experiment(tmp_path, "alpha", [0.1, 0.3] * 5)
    _write_experiment(tmp_path, "winner", [0.4] * 10)

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout.splitlines() == [
        "winner  40.00%",
        "alpha  20.00%",
        "beta  20.00%",
    ]
    assert completed.stderr == ""


def test_failure_json_is_ignored(tmp_path: Path):
    experiment = _write_experiment(tmp_path, "recovered", [0.25] * 10)
    (experiment / "task-4" / "failure.json").write_text(
        json.dumps({"status": "failed"}),
        encoding="utf-8",
    )

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == "recovered  25.00%\n"
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing_task", "task-9/results.json"),
        ("missing_result", "task-4/results.json"),
        ("bad_json", "task-3/results.json"),
        ("failed_status", "not complete"),
        ("zero_episodes", "invalid protocol.episodes"),
        ("bool_episodes", "invalid protocol.episodes"),
        ("negative_completed", "incomplete episodes"),
        ("bool_completed", "incomplete episodes"),
        ("incomplete", "incomplete episodes"),
        ("missing_field", "valid result"),
        ("invalid_structure", "valid result"),
        ("bool_rate", "invalid summary.success_rate"),
        ("string_rate", "invalid summary.success_rate"),
        ("nan_rate", "invalid summary.success_rate"),
        ("low_rate", "invalid summary.success_rate"),
        ("high_rate", "invalid summary.success_rate"),
        ("huge_rate", "invalid summary.success_rate"),
    ],
)
def test_skips_entire_experiment_when_any_task_is_invalid(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
):
    experiment = _write_experiment(tmp_path, "invalid", [0.5] * 10)
    result_path = experiment / "task-3" / "results.json"
    value = json.loads(result_path.read_text(encoding="utf-8"))

    if mutation == "missing_task":
        (experiment / "task-9" / "results.json").unlink()
        (experiment / "task-9").rmdir()
    elif mutation == "missing_result":
        (experiment / "task-4" / "results.json").unlink()
    elif mutation == "bad_json":
        result_path.write_text("{", encoding="utf-8")
    else:
        if mutation == "failed_status":
            value["status"] = "failed"
        elif mutation == "zero_episodes":
            value["protocol"]["episodes"] = 0
            value["summary"]["completed_episodes"] = 0
        elif mutation == "bool_episodes":
            value["protocol"]["episodes"] = True
            value["summary"]["completed_episodes"] = True
        elif mutation == "negative_completed":
            value["summary"]["completed_episodes"] = -1
        elif mutation == "bool_completed":
            value["summary"]["completed_episodes"] = True
        elif mutation == "incomplete":
            value["summary"]["completed_episodes"] = 149
        elif mutation == "missing_field":
            del value["summary"]["success_rate"]
        elif mutation == "invalid_structure":
            value["summary"] = []
        elif mutation == "bool_rate":
            value["summary"]["success_rate"] = True
        elif mutation == "string_rate":
            value["summary"]["success_rate"] = "0.5"
        elif mutation == "nan_rate":
            value["summary"]["success_rate"] = math.nan
        elif mutation == "low_rate":
            value["summary"]["success_rate"] = -0.01
        elif mutation == "high_rate":
            value["summary"]["success_rate"] = 1.01
        elif mutation == "huge_rate":
            value["summary"]["success_rate"] = 10**1000
        result_path.write_text(json.dumps(value), encoding="utf-8")

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert "skip invalid:" in completed.stderr
    assert expected_reason in completed.stderr


def test_no_valid_experiment_has_empty_stdout(tmp_path: Path):
    (tmp_path / "ordinary-file").write_text("ignored", encoding="utf-8")

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_invalid_root_returns_nonzero(tmp_path: Path, root_kind: str):
    root = tmp_path / "results"
    if root_kind == "file":
        root.write_text("not a directory", encoding="utf-8")

    completed = _run(root)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "result root is not a directory" in completed.stderr
