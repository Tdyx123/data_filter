import subprocess
from pathlib import Path

import pytest

from tests.test_evaluate_simpler_starvla_script import (
    _make_fake_processes,
    _read_calls,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_simpler_qwen3vl_groot_rt1.sh"
FIXED_CHECKPOINT = Path(
    "/data/dwb/models/Qwen3VL-GR00T-Bridge-RT-1/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
FIXED_MODEL_DIR = Path("/data/dwb/models/Qwen3VL-GR00T-Bridge-RT-1")


def _run_fixed_launcher(tmp_path: Path, *arguments: str):
    pyenv, sim_python, environment, model_calls, sim_calls, stopped = (
        _make_fake_processes(tmp_path)
    )
    output_dir = tmp_path / "output"
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--pyenv-bin",
            str(pyenv),
            "--sim-python",
            str(sim_python),
            "--output-dir",
            str(output_dir),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    return (
        completed,
        _read_calls(model_calls),
        _read_calls(sim_calls),
        stopped,
    )


def test_fixed_launcher_uses_checkpoint_model_root_and_forwards_evaluation_arguments(
    tmp_path,
):
    forwarded = [
        "--tasks=spoon,eggplant",
        "--action-horizon",
        "1",
        "--save-videos-path",
        str(tmp_path / "videos with spaces"),
        "--preflight-only",
        "--overwrite",
    ]

    completed, model_calls, sim_calls, stopped = _run_fixed_launcher(
        tmp_path, *forwarded
    )

    assert completed.returncode == 0, completed.stderr
    assert len(model_calls) == 1
    model_call = model_calls[0]
    assert model_call[model_call.index("--model-dir") + 1] == str(FIXED_MODEL_DIR)
    assert len(sim_calls) == 1
    assert sim_calls[0][-len(forwarded) :] == forwarded
    assert stopped.read_text(encoding="utf-8") == "stopped"


def test_fixed_launcher_help_names_checkpoint_without_starting_processes(tmp_path):
    completed, model_calls, sim_calls, _stopped = _run_fixed_launcher(
        tmp_path, "--help"
    )

    assert completed.returncode == 0
    assert model_calls == []
    assert sim_calls == []
    assert str(FIXED_CHECKPOINT) in completed.stdout


@pytest.mark.parametrize(
    "override",
    [
        ("--checkpoint", "/tmp/other.pt"),
        ("--checkpoint=/tmp/other.pt",),
        ("--model-dir", "/tmp/other-model"),
        ("--model-dir=/tmp/other-model",),
    ],
)
def test_fixed_launcher_rejects_checkpoint_overrides_before_starting_processes(
    tmp_path, override
):
    completed, model_calls, sim_calls, _stopped = _run_fixed_launcher(
        tmp_path, *override
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "fixed checkpoint cannot be overridden" in completed.stderr
