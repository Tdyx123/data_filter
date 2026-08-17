import os
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_simpler_starvla.sh"


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
stopped_path = Path(os.environ["STARVLA_TEST_STOPPED"])
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

def stop(_signum, _frame):
    server.close()
    stopped_path.write_text("stopped", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
server.bind(socket_path)
server.listen(1)
if os.environ.get("STARVLA_TEST_MODEL_CRASH_AFTER_READY") == "1":
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: sys.exit(29))
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
} >> "${STARVLA_TEST_MODEL_CALLS:?}"
if [[ "${STARVLA_TEST_MODEL_CRASH:-0}" == "1" ]]; then
  exit 23
fi
socket_path=""
while (($#)); do
  if [[ "$1" == "--socket" ]]; then socket_path="$2"; break; fi
  shift
done
exec python3 "${STARVLA_TEST_MODEL_HELPER:?}" "${socket_path}"
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
} >> "${STARVLA_TEST_SIM_CALLS:?}"
if [[ "${STARVLA_TEST_SIM_PAUSE:-0}" == "1" ]]; then
  trap 'exit 0' TERM INT
  while :; do sleep 1; done
fi
exit "${STARVLA_TEST_SIM_STATUS:-0}"
""",
        encoding="utf-8",
    )
    sim_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "STARVLA_TEST_MODEL_CALLS": str(model_calls),
            "STARVLA_TEST_SIM_CALLS": str(sim_calls),
            "STARVLA_TEST_MODEL_HELPER": str(model_helper),
            "STARVLA_TEST_STOPPED": str(stopped),
        }
    )
    return pyenv, sim_python, environment, model_calls, sim_calls, stopped


def _read_calls(path: Path):
    if not path.exists():
        return []
    return [line.split("\x1f")[:-1] for line in path.read_text().splitlines()]


def _run(tmp_path: Path, *arguments: str, environment_updates=None):
    pyenv, sim_python, environment, model_calls, sim_calls, stopped = (
        _make_fake_processes(tmp_path)
    )
    environment.update(environment_updates or {})
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
        output_dir,
    )


def test_launcher_starts_pyenv_model_then_simulator_and_cleans_private_socket(tmp_path):
    completed, model_calls, sim_calls, stopped, output_dir = _run(
        tmp_path,
        "--tasks=spoon,eggplant",
        "--action-horizon",
        "1",
        "--smoke-test",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(model_calls) == 1
    model_call = model_calls[0]
    assert model_call[0] == "PYENV_VERSION=miniconda3-3.12-25.11.1-1"
    assert model_call[1] == f"PYTHONPATH={PROJECT_ROOT / 'src'}"
    assert model_call[2:6] == ["exec", "python", "-m", "starvla_bridge.server"]
    assert model_call[model_call.index("--model-dir") + 1] == (
        "/data/dwb/models/Qwen3VL-GR00T-Bridge-RT-1"
    )
    assert model_call[model_call.index("--base-model") + 1] == (
        "/data/dwb/models/Qwen3-VL-4B-Instruct"
    )
    assert model_call[model_call.index("--device") + 1] == "cuda:0"

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
    assert sim_call[2:4] == ["-m", "starvla_bridge.evaluate_simpler"]
    socket_path = Path(sim_call[sim_call.index("--socket") + 1])
    assert sim_call[sim_call.index("--auth-key-hex") + 1]
    assert sim_call[sim_call.index("--output-dir") + 1] == str(output_dir)
    assert sim_call[sim_call.index("--sim-device") + 1] == "cuda:0"
    assert "--tasks=spoon,eggplant" in sim_call
    assert "--smoke-test" in sim_call
    assert not socket_path.exists()
    assert stopped.read_text(encoding="utf-8") == "stopped"
    assert (output_dir / "model-server.log").exists()


def test_launcher_keeps_model_device_separate_from_explicit_sim_device(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--device",
        "cuda:2",
        "--sim-device=cuda:7",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(model_calls) == 1
    assert len(sim_calls) == 1
    model_call = model_calls[0]
    sim_call = sim_calls[0]
    assert model_call[model_call.index("--device") + 1] == "cuda:2"
    assert "--sim-device" not in model_call
    assert sim_call.count("--sim-device") == 1
    assert sim_call[sim_call.index("--sim-device") + 1] == "cuda:7"
    assert "--device" not in sim_call


def test_launcher_rejects_invalid_sim_device_before_starting_processes(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--sim-device",
        "cpu",
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "must match cuda:<non-negative decimal integer>" in completed.stderr


def test_launcher_rejects_missing_sim_device_before_starting_processes(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--sim-device",
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "--sim-device requires a non-empty value" in completed.stderr


def test_launcher_rejects_duplicate_sim_device_before_starting_processes(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--sim-device",
        "cuda:0",
        "--sim-device=cuda:1",
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "--sim-device may only be specified once" in completed.stderr


def test_launcher_propagates_simulator_failure_and_cleans_its_socket(tmp_path):
    completed, _model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        environment_updates={"STARVLA_TEST_SIM_STATUS": "7"},
    )

    assert completed.returncode == 7
    assert len(sim_calls) == 1
    socket_path = Path(sim_calls[0][sim_calls[0].index("--socket") + 1])
    assert not socket_path.exists()


def test_launcher_reports_model_failure_without_starting_simulator(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--server-timeout",
        "2",
        environment_updates={"STARVLA_TEST_MODEL_CRASH": "1"},
    )

    assert completed.returncode != 0
    assert len(model_calls) == 1
    assert sim_calls == []
    assert "model server exited before becoming ready" in completed.stderr


def test_launcher_stops_simulator_when_model_crashes_after_startup(tmp_path):
    completed, _model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        environment_updates={
            "STARVLA_TEST_MODEL_CRASH_AFTER_READY": "1",
            "STARVLA_TEST_SIM_PAUSE": "1",
        },
    )

    assert completed.returncode == 29
    assert len(sim_calls) == 1
    assert "model server failed during evaluation" in completed.stderr


def test_launcher_help_does_not_start_either_process(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(tmp_path, "--help")

    assert completed.returncode == 0
    assert model_calls == []
    assert sim_calls == []
    assert "--pyenv-version" in completed.stdout
    assert "--preflight-only" in completed.stdout
    assert "--device DEVICE        model service CUDA device" in completed.stdout
    assert "--sim-device DEVICE    simulator renderer CUDA device" in completed.stdout


def test_launcher_rejects_legacy_action_horizon_before_loading_model(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output_dir = _run(
        tmp_path,
        "--action-horizon",
        "8",
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert "--action-horizon must be 1" in completed.stderr


def test_launcher_term_signal_reaps_both_managed_processes(tmp_path):
    pyenv, sim_python, environment, _model_calls, sim_calls, stopped = (
        _make_fake_processes(tmp_path)
    )
    environment["STARVLA_TEST_SIM_PAUSE"] = "1"
    process = subprocess.Popen(
        [
            "bash",
            str(SCRIPT),
            "--pyenv-bin",
            str(pyenv),
            "--sim-python",
            str(sim_python),
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


def test_starvla_dependency_file_and_readme_preserve_the_current_pyenv_stack():
    requirements = (PROJECT_ROOT / "requirements-starvla-pyenv.txt").read_text(
        encoding="utf-8"
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert requirements.splitlines() == ["diffusers==0.38.0"]
    assert "miniconda3-3.12-25.11.1-1" in readme
    assert "scripts/evaluate_simpler_starvla.sh" in readme
    assert "/data/dwb/models/Qwen3VL-GR00T-Bridge-RT-1" in readme
    assert "每次只执行动作块的第一个动作" in readme
    assert "共 288 回合" in readme
    assert "`--action-horizon` 的唯一合法值为 `1`" in readme
