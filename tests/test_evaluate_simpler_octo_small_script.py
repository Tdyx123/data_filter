import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_simpler_octo_small.sh"
CHECKPOINT = "/models/octo-run/checkpoints/step-00020000"
BASE_MODEL = "/models/octo-small-pytorch"
STATISTICS = "/data/bridge/meta/stats.json"


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


def test_launcher_requires_all_three_model_paths_before_starting_python(tmp_path):
    completed, calls = _run_script(tmp_path, "--checkpoint", CHECKPOINT)

    assert completed.returncode == 2
    assert calls == []
    assert "--base-model is required" in completed.stderr


def test_launcher_sets_simpler_sources_and_forwards_octo_options(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--checkpoint",
        CHECKPOINT,
        "--base-model",
        BASE_MODEL,
        "--statistics",
        STATISTICS,
        "--tasks=spoon,eggplant",
        "--precision",
        "fp32",
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
    assert call[2:4] == ["-m", "octo_small_bridge.evaluate_simpler"]
    assert call[call.index("--checkpoint") + 1] == CHECKPOINT
    assert call[call.index("--base-model") + 1] == BASE_MODEL
    assert call[call.index("--statistics") + 1] == STATISTICS
    assert "--tasks=spoon,eggplant" in call
    assert "--smoke-test" in call


@pytest.mark.parametrize("option", ["--checkpoint", "--base-model", "--statistics", "--python"])
def test_launcher_rejects_duplicate_owned_options(tmp_path, option):
    values = {
        "--checkpoint": CHECKPOINT,
        "--base-model": BASE_MODEL,
        "--statistics": STATISTICS,
        "--python": "/bin/python3",
    }
    arguments = [
        "--checkpoint",
        CHECKPOINT,
        "--base-model",
        BASE_MODEL,
        "--statistics",
        STATISTICS,
        option,
        values[option],
    ]

    completed, calls = _run_script(tmp_path, *arguments)

    assert completed.returncode == 2
    assert calls == []
    assert option in completed.stderr


def test_octo_simpler_requirements_and_readme_use_the_dedicated_runtime():
    requirements = (PROJECT_ROOT / "requirements-octo-simpler-eval.txt").read_text(
        encoding="utf-8"
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for requirement in (
        "numpy==1.24.3",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "transformers==4.44.2",
        "tokenizers==0.19.1",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "sapien==2.2.2",
    ):
        assert requirement in requirements
    assert "Octo-small Bridge 的 SimplerEnv 四任务闭环评测" in readme
    assert "scripts/evaluate_simpler_octo_small.sh" in readme
    assert "/data/dwb/octo_small_bridge_v2/checkpoints/step-00020000" in readme
