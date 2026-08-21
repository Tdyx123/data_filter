import subprocess
from pathlib import Path

from tests.test_evaluate_simpler_qwen_script import (
    MODEL_PATH,
    _make_fake_processes,
    _read_calls,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_simpler_qwenvl_oft.sh"
CHECKPOINT = "/models/qwen-oft-run/checkpoints/step-00019000"


def _run(tmp_path: Path, *arguments: str):
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
            "--checkpoint",
            CHECKPOINT,
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
    return completed, _read_calls(model_calls), _read_calls(sim_calls), stopped, output_dir


def test_oft_launcher_starts_deterministic_model_and_simulator_modules(tmp_path):
    completed, model_calls, sim_calls, stopped, output_dir = _run(
        tmp_path,
        "--model-path",
        MODEL_PATH,
        "--device",
        "cuda:2",
        "--sim-device",
        "cuda:4",
        "--tasks=spoon",
        "--smoke-test",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(model_calls) == 1
    model_call = model_calls[0]
    assert model_call[2:6] == ["exec", "python", "-m", "qwen_vl_oft.server"]
    assert model_call[model_call.index("--checkpoint") + 1] == CHECKPOINT
    assert model_call[model_call.index("--model-path") + 1] == MODEL_PATH
    assert model_call[model_call.index("--device") + 1] == "cuda:2"
    assert "--denoising-steps" not in model_call
    assert len(sim_calls) == 1
    sim_call = sim_calls[0]
    assert sim_call[2:4] == ["-m", "qwen_vl_oft.evaluate_simpler"]
    assert sim_call[sim_call.index("--sim-device") + 1] == "cuda:4"
    assert "--tasks=spoon" in sim_call
    assert "--smoke-test" in sim_call
    assert stopped.read_text(encoding="utf-8") == "stopped"
    assert (output_dir / "model-server.log").is_file()


def test_oft_launcher_delegates_model_devices_to_oft_parallel_coordinator(tmp_path):
    pyenv, sim_python, environment, model_calls, sim_calls, _stopped = (
        _make_fake_processes(tmp_path)
    )
    simpler_root = PROJECT_ROOT / "third_party" / "SimplerEnv"
    maniskill_root = simpler_root / "ManiSkill2_real2sim"
    environment.update(
        {
            "QWEN_TEST_EXPECT_PARALLEL_PYTHONPATH": (
                f"{PROJECT_ROOT / 'src'}:{simpler_root}:{maniskill_root}"
            ),
            "QWEN_TEST_EXPECT_PARALLEL_ASSET_DIR": str(maniskill_root / "data"),
        }
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
            "--checkpoint",
            CHECKPOINT,
            "--model-devices",
            "cuda:0,cuda:1",
            "--sim-device",
            "cuda:4",
            "--output-dir",
            str(output_dir),
            "--tasks",
            "spoon",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    call = _read_calls(model_calls)[0]
    assert call[2:6] == [
        "exec",
        "python",
        "-m",
        "qwen_vl_oft.parallel_evaluation",
    ]
    assert call[call.index("--model-devices") + 1] == "cuda:0,cuda:1"
    assert call[call.index("--sim-device") + 1] == "cuda:4"
    assert "--denoising-steps" not in call
    assert call[call.index("--") + 1 :] == ["--tasks", "spoon"]
    assert _read_calls(sim_calls) == []


def test_oft_launcher_rejects_denoising_before_starting_processes(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--denoising-steps",
        "4",
    )

    assert completed.returncode == 2
    assert "does not support --denoising-steps" in completed.stderr
    assert model_calls == []
    assert sim_calls == []


def test_oft_launcher_help_lists_checkpoint_and_parallel_options(tmp_path):
    pyenv, sim_python, environment, model_calls, sim_calls, _stopped = (
        _make_fake_processes(tmp_path)
    )

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--pyenv-bin",
            str(pyenv),
            "--sim-python",
            str(sim_python),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )

    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--model-devices" in completed.stdout
    assert "--denoising-steps" not in completed.stdout
    assert _read_calls(model_calls) == []
    assert _read_calls(sim_calls) == []
