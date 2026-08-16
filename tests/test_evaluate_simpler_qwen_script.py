import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_simpler_qwen.sh"
CHECKPOINT = "/models/qwen-run/checkpoints/step-00020000"


def _fake_python(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir(parents=True)
    executable = executable_dir / "simpler-python"
    executable.write_text(
        """#!/usr/bin/env bash
set -u
{
  printf 'PYTHONPATH=%s\037' "${PYTHONPATH:-}"
  printf 'MS2_REAL2SIM_ASSET_DIR=%s\037' "${MS2_REAL2SIM_ASSET_DIR:-}"
  for argument in "$@"; do
    printf '%s\037' "${argument}"
  done
  printf '\n'
} >> "${SIMPLER_EVAL_TEST_CALLS:?}"
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    calls_path = tmp_path / "calls"
    environment = os.environ.copy()
    environment["SIMPLER_EVAL_TEST_CALLS"] = str(calls_path)
    return executable, environment, calls_path


def _run_script(tmp_path: Path, *arguments: str):
    executable, environment, calls_path = _fake_python(tmp_path)
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--python", str(executable), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = []
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            calls.append(line.split("\x1f")[:-1])
    return completed, calls


def test_launcher_requires_checkpoint_before_starting_python(tmp_path):
    completed, calls = _run_script(tmp_path, "--tasks", "spoon")

    assert completed.returncode == 2
    assert calls == []
    assert "--checkpoint is required" in completed.stderr


def test_launcher_uses_fixed_sources_and_forwards_evaluation_options(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--checkpoint",
        CHECKPOINT,
        "--tasks=spoon,eggplant",
        "--model-path",
        "/models/Qwen3-VL-4B-Instruct",
        "--action-horizon",
        "1",
        "--save-videos-path",
        "/tmp/simpler-videos",
        "--smoke-test",
    )

    assert completed.returncode == 0
    assert len(calls) == 1
    call = calls[0]
    assert call[0] == (
        f"PYTHONPATH={PROJECT_ROOT / 'src'}:"
        f"{PROJECT_ROOT / 'third_party/SimplerEnv'}:"
        f"{PROJECT_ROOT / 'third_party/SimplerEnv/ManiSkill2_real2sim'}"
    )
    assert call[1] == (
        "MS2_REAL2SIM_ASSET_DIR="
        f"{PROJECT_ROOT / 'third_party/SimplerEnv/ManiSkill2_real2sim/data'}"
    )
    assert call[2:4] == ["-m", "qwen3_vl_groot.evaluate_simpler"]
    assert "--python" not in call
    assert call[call.index("--checkpoint") + 1] == CHECKPOINT
    assert "--tasks=spoon,eggplant" in call
    assert call[call.index("--action-horizon") + 1] == "1"
    assert call[call.index("--save-videos-path") + 1] == "/tmp/simpler-videos"
    assert "--smoke-test" in call


@pytest.mark.parametrize(
    "arguments,message",
    [
        (("--checkpoint", CHECKPOINT, "--checkpoint", CHECKPOINT), "--checkpoint"),
        (("--checkpoint", CHECKPOINT, "--python", "/bin/python3"), "--python"),
        (("--checkpoint", ""), "non-empty"),
    ],
)
def test_launcher_rejects_duplicate_or_empty_owned_options(tmp_path, arguments, message):
    completed, calls = _run_script(tmp_path, *arguments)

    assert completed.returncode == 2
    assert calls == []
    assert message in completed.stderr


def test_python_cli_parses_smoke_protocol_without_importing_simulator():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv/bin/python"),
            "-m",
            "qwen3_vl_groot.evaluate_simpler",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--preflight-only" in completed.stdout
    assert "--smoke-test" in completed.stdout
    assert "--save-videos-path" in completed.stdout
    assert "--python" not in completed.stdout


def test_python_cli_writes_failure_json_for_pre_evaluation_contract_error(
    tmp_path, monkeypatch
):
    from qwen3_vl_groot import evaluate_simpler

    def reject_source(_path):
        raise evaluate_simpler.SimplerEvaluationError("source mismatch")

    monkeypatch.setattr(evaluate_simpler, "validate_simpler_source", reject_source)
    output_dir = tmp_path / "output"

    status = evaluate_simpler.main(
        [
            "--checkpoint",
            CHECKPOINT,
            "--output-dir",
            str(output_dir),
            "--preflight-only",
        ]
    )

    assert status == 2
    failure = __import__("json").loads((output_dir / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["exit_code"] == 2
    assert failure["error"] == "SimplerEvaluationError: source mismatch"


def test_python_cli_applies_task_filter_and_smoke_protocol(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from qwen3_vl_groot import evaluate_simpler

    captured = {}
    checkpoint = SimpleNamespace()
    policy = SimpleNamespace()
    monkeypatch.setattr(
        evaluate_simpler,
        "validate_simpler_source",
        lambda path: {"simpler_env_commit": "06accaca9353"},
    )
    monkeypatch.setattr(
        evaluate_simpler,
        "validate_runtime_contract",
        lambda device: {"numpy": "1.24.4"},
    )
    monkeypatch.setattr(
        evaluate_simpler,
        "_load_checkpoint_and_policy",
        lambda arguments: (checkpoint, policy),
    )

    def evaluate(settings, **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return {"status": "complete"}

    monkeypatch.setattr(evaluate_simpler, "evaluate_simpler_checkpoint", evaluate)

    status = evaluate_simpler.main(
        [
            "--checkpoint",
            CHECKPOINT,
            "--tasks",
            "eggplant,spoon",
            "--output-dir",
            str(tmp_path / "output"),
            "--save-videos-path",
            str(tmp_path / "videos"),
            "--smoke-test",
        ]
    )

    settings = captured["settings"]
    assert status == 0
    assert [task.key for task in settings.tasks] == ["eggplant", "spoon"]
    assert settings.policy_seeds == (0,)
    assert settings.object_episode_ids == (0,)
    assert settings.max_steps == 8
    assert settings.save_videos_path == tmp_path / "videos"
    assert captured["kwargs"]["checkpoint"] is checkpoint
    assert captured["kwargs"]["policy"] is policy
