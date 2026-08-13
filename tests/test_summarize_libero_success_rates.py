import json
import subprocess
import sys
from pathlib import Path


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
