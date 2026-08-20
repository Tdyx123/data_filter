import os
import subprocess
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_simpler_qwen.sh"
CHECKPOINT = "/models/qwen-run/checkpoints/step-00020000"
MODEL_PATH = "/models/Qwen3-VL-4B-Instruct"


def _make_fake_processes(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    model_calls = tmp_path / "model-calls"
    sim_calls = tmp_path / "sim-calls"
    stopped = tmp_path / "model-stopped"

    model_helper = tmp_path / "fake_model_server.py"
    model_helper.write_text(
        """import os
import signal
import socket
import sys
from pathlib import Path

socket_path = sys.argv[1]
stopped_path = Path(os.environ["QWEN_TEST_STOPPED"])
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

def stop(_signum, _frame):
    server.close()
    stopped_path.write_text("stopped", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
server.bind(socket_path)
server.listen(1)
if os.environ.get("QWEN_TEST_MODEL_CRASH_AFTER_READY") == "1":
    signal.signal(signal.SIGALRM, lambda _signum, _frame: sys.exit(29))
    signal.setitimer(signal.ITIMER_REAL, 0.2)
signal.pause()
""",
        encoding="utf-8",
    )

    pyenv = bin_dir / "pyenv"
    pyenv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
{
  printf 'PYENV_VERSION=%s\037' "${PYENV_VERSION:-}"
  printf 'PYTHONPATH=%s\037' "${PYTHONPATH:-}"
  for argument in "$@"; do printf '%s\037' "${argument}"; done
  printf '\n'
} >> "${QWEN_TEST_MODEL_CALLS:?}"
if [[ "${QWEN_TEST_MODEL_CRASH:-0}" == "1" ]]; then
  exit 23
fi
if [[ "${QWEN_TEST_MODEL_NO_SOCKET:-0}" == "1" ]]; then
  trap 'exit 0' TERM INT
  while :; do sleep 1; done
fi
socket_path=""
while (($#)); do
  if [[ "$1" == "--socket" ]]; then socket_path="$2"; break; fi
  shift
done
exec python3 "${QWEN_TEST_MODEL_HELPER:?}" "${socket_path}"
""",
        encoding="utf-8",
    )
    pyenv.chmod(0o755)

    sim_python = bin_dir / "sim-python"
    sim_python.write_text(
        """#!/usr/bin/env bash
set -u
{
  printf 'PYTHONPATH=%s\037' "${PYTHONPATH:-}"
  printf 'MS2_REAL2SIM_ASSET_DIR=%s\037' "${MS2_REAL2SIM_ASSET_DIR:-}"
  for argument in "$@"; do printf '%s\037' "${argument}"; done
  printf '\n'
} >> "${QWEN_TEST_SIM_CALLS:?}"
if [[ "${QWEN_TEST_SIM_PAUSE:-0}" == "1" ]]; then
  trap 'exit 0' TERM INT
  while :; do sleep 1; done
fi
exit "${QWEN_TEST_SIM_STATUS:-0}"
""",
        encoding="utf-8",
    )
    sim_python.chmod(0o755)

    environment = os.environ.copy()
    environment.pop("PYENV_VERSION", None)
    environment.update(
        {
            "QWEN_TEST_MODEL_CALLS": str(model_calls),
            "QWEN_TEST_SIM_CALLS": str(sim_calls),
            "QWEN_TEST_MODEL_HELPER": str(model_helper),
            "QWEN_TEST_STOPPED": str(stopped),
        }
    )
    return pyenv, sim_python, environment, model_calls, sim_calls, stopped


def _read_calls(path: Path):
    if not path.exists():
        return []
    return [line.split("\x1f")[:-1] for line in path.read_text().splitlines()]


def _run(
    tmp_path: Path,
    *arguments: str,
    environment_updates=None,
    include_checkpoint=True,
):
    pyenv, sim_python, environment, model_calls, sim_calls, stopped = (
        _make_fake_processes(tmp_path)
    )
    environment.update(environment_updates or {})
    output_dir = tmp_path / "output"
    command = [
        "bash",
        str(SCRIPT),
        "--pyenv-bin",
        str(pyenv),
        "--sim-python",
        str(sim_python),
        "--output-dir",
        str(output_dir),
    ]
    if include_checkpoint:
        command.extend(["--checkpoint", CHECKPOINT])
    command.extend(arguments)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    return completed, _read_calls(model_calls), _read_calls(sim_calls), stopped, output_dir


def test_launcher_starts_current_pyenv_model_then_dedicated_simulator(tmp_path):
    completed, model_calls, sim_calls, stopped, output_dir = _run(
        tmp_path,
        "--model-path",
        MODEL_PATH,
        "--tasks=spoon,eggplant",
        "--smoke-test",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(model_calls) == 1
    model_call = model_calls[0]
    assert model_call[0] == "PYENV_VERSION="
    assert model_call[1] == f"PYTHONPATH={PROJECT_ROOT / 'src'}"
    assert model_call[2:6] == ["exec", "python", "-m", "qwen3_vl_groot.server"]
    assert model_call[model_call.index("--checkpoint") + 1] == CHECKPOINT
    assert model_call[model_call.index("--model-path") + 1] == MODEL_PATH
    assert model_call[model_call.index("--device") + 1] == "cuda:0"
    assert model_call[model_call.index("--denoising-steps") + 1] == "4"

    assert len(sim_calls) == 1
    sim_call = sim_calls[0]
    assert sim_call[0] == (
        f"PYTHONPATH={PROJECT_ROOT / 'src'}:"
        f"{PROJECT_ROOT / 'third_party/SimplerEnv'}:"
        f"{PROJECT_ROOT / 'third_party/SimplerEnv/ManiSkill2_real2sim'}"
    )
    assert sim_call[1] == (
        "MS2_REAL2SIM_ASSET_DIR="
        f"{PROJECT_ROOT / 'third_party/SimplerEnv/ManiSkill2_real2sim/data'}"
    )
    assert sim_call[2:4] == ["-m", "qwen3_vl_groot.evaluate_simpler"]
    socket_path = Path(sim_call[sim_call.index("--socket") + 1])
    assert sim_call[sim_call.index("--auth-key-hex") + 1]
    assert sim_call[sim_call.index("--output-dir") + 1] == str(output_dir)
    assert sim_call[sim_call.index("--sim-device") + 1] == "cuda:0"
    assert "--checkpoint" not in sim_call
    assert "--model-path" not in sim_call
    assert "--tasks=spoon,eggplant" in sim_call
    assert "--smoke-test" in sim_call
    assert not socket_path.exists()
    assert stopped.read_text(encoding="utf-8") == "stopped"
    assert (output_dir / "model-server.log").exists()


def test_launcher_separates_model_and_simulator_options(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--device",
        "cuda:2",
        "--denoising-steps=6",
        "--sim-device=cuda:7",
    )

    assert completed.returncode == 0, completed.stderr
    model_call = model_calls[0]
    sim_call = sim_calls[0]
    assert "--model-path" not in model_call
    assert model_call[model_call.index("--device") + 1] == "cuda:2"
    assert model_call[model_call.index("--denoising-steps") + 1] == "6"
    assert "--sim-device" not in model_call
    assert sim_call[sim_call.index("--sim-device") + 1] == "cuda:7"
    assert "--device" not in sim_call
    assert "--denoising-steps" not in sim_call


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--sim-device", "cpu"), "must match cuda:<non-negative decimal integer>"),
        (("--sim-device",), "--sim-device requires a non-empty value"),
        (
            ("--sim-device", "cuda:0", "--sim-device=cuda:1"),
            "--sim-device may only be specified once",
        ),
        (("--action-horizon", "8"), "--action-horizon must be 1"),
        (("--denoising-steps", "0"), "--denoising-steps must be a positive integer"),
        (("--server-timeout", "0"), "--server-timeout must be a positive integer"),
    ],
    ids=(
        "invalid-device",
        "missing-device",
        "duplicate-device",
        "action-horizon",
        "denoising-steps",
        "server-timeout",
    ),
)
def test_launcher_rejects_invalid_options_before_starting_processes(
    tmp_path, arguments, message
):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path, *arguments
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert message in completed.stderr


def test_launcher_rejects_removed_python_option(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path, "--python", "/bin/python3"
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "--python has been removed; use --sim-python" in completed.stderr


@pytest.mark.parametrize("option", ["--socket", "--auth-key-hex"])
def test_launcher_rejects_internal_ipc_options(tmp_path, option):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path, option, "user-controlled"
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "is managed by this launcher" in completed.stderr


def test_launcher_rejects_duplicate_model_option(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path, "--checkpoint=/models/other-checkpoint"
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "--checkpoint may only be specified once" in completed.stderr


def test_launcher_requires_checkpoint_before_starting_processes(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path, include_checkpoint=False
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "--checkpoint is required" in completed.stderr


def test_launcher_propagates_simulator_failure_and_cleans_socket(tmp_path):
    completed, _model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        environment_updates={"QWEN_TEST_SIM_STATUS": "7"},
    )

    assert completed.returncode == 7
    socket_path = Path(sim_calls[0][sim_calls[0].index("--socket") + 1])
    assert not socket_path.exists()


def test_launcher_reports_model_failure_without_starting_simulator(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--server-timeout",
        "2",
        environment_updates={"QWEN_TEST_MODEL_CRASH": "1"},
    )

    assert completed.returncode == 23
    assert len(model_calls) == 1
    assert sim_calls == []
    assert "model server exited before becoming ready" in completed.stderr


def test_launcher_reports_model_readiness_timeout(tmp_path):
    completed, _model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--server-timeout",
        "1",
        environment_updates={"QWEN_TEST_MODEL_NO_SOCKET": "1"},
    )

    assert completed.returncode == 1
    assert sim_calls == []
    assert "timed out after 1s waiting for model server" in completed.stderr


def test_launcher_stops_simulator_when_model_crashes_after_startup(tmp_path):
    completed, _model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        environment_updates={
            "QWEN_TEST_MODEL_CRASH_AFTER_READY": "1",
            "QWEN_TEST_SIM_PAUSE": "1",
        },
    )

    assert completed.returncode == 29
    assert len(sim_calls) == 1
    assert "model server failed during evaluation" in completed.stderr


def test_launcher_help_does_not_start_processes_and_documents_defaults(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(tmp_path, "--help")

    assert completed.returncode == 0
    assert model_calls == []
    assert sim_calls == []
    assert "--pyenv-bin PATH" in completed.stdout
    assert "/home/dwb/.pyenv/bin/pyenv" in completed.stdout
    assert "--sim-python PATH" in completed.stdout
    assert ".venv-octo-simpler/bin/python" in completed.stdout
    assert "--server-timeout SEC" in completed.stdout
    assert "--preflight-only" in completed.stdout


def test_launcher_term_signal_reaps_both_managed_processes(tmp_path):
    pyenv, sim_python, environment, _model_calls, sim_calls, stopped = (
        _make_fake_processes(tmp_path)
    )
    environment["QWEN_TEST_SIM_PAUSE"] = "1"
    process = subprocess.Popen(
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
            str(tmp_path / "output"),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while not sim_calls.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sim_calls.exists()

    process.terminate()
    process.communicate(timeout=10)

    assert process.returncode == 143
    assert stopped.read_text(encoding="utf-8") == "stopped"
