import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_simpler_octo_small_official_pytorch.sh"


def _fake_processes(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    model_calls = tmp_path / "model-calls"
    sim_calls = tmp_path / "sim-calls"
    stopped = tmp_path / "stopped"
    helper = tmp_path / "model.py"
    helper.write_text(
        """import os
import signal
import socket
import sys
from pathlib import Path

target = sys.argv[1]
stopped = Path(os.environ["OFFICIAL_TEST_STOPPED"])
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

def stop(_signum, _frame):
    server.close()
    stopped.write_text("stopped", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
if os.environ.get("OFFICIAL_TEST_NO_SOCKET") == "1":
    signal.pause()
server.bind(target)
server.listen(1)
signal.pause()
""",
        encoding="utf-8",
    )
    pyenv = bin_dir / "pyenv"
    pyenv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
for argument in "$@"; do printf '%s\037' "${argument}"; done >> "${OFFICIAL_TEST_MODEL_CALLS:?}"
printf '\n' >> "${OFFICIAL_TEST_MODEL_CALLS:?}"
if [[ " $* " == *" octo_small_official_pytorch.checkpoint_cli "* ]]; then
  exit "${OFFICIAL_TEST_VALIDATION_STATUS:-0}"
fi
if [[ "${OFFICIAL_TEST_MODEL_CRASH:-0}" == "1" ]]; then
  exit 23
fi
socket_path=""
while (($#)); do
  if [[ "$1" == "--socket" ]]; then socket_path="$2"; break; fi
  shift
done
exec python3 "${OFFICIAL_TEST_HELPER:?}" "${socket_path}"
""",
        encoding="utf-8",
    )
    pyenv.chmod(0o755)
    sim = bin_dir / "sim-python"
    sim.write_text(
        """#!/usr/bin/env bash
set -u
for argument in "$@"; do printf '%s\037' "${argument}"; done >> "${OFFICIAL_TEST_SIM_CALLS:?}"
printf '\n' >> "${OFFICIAL_TEST_SIM_CALLS:?}"
exit "${OFFICIAL_TEST_SIM_STATUS:-0}"
""",
        encoding="utf-8",
    )
    sim.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "OFFICIAL_TEST_MODEL_CALLS": str(model_calls),
            "OFFICIAL_TEST_SIM_CALLS": str(sim_calls),
            "OFFICIAL_TEST_HELPER": str(helper),
            "OFFICIAL_TEST_STOPPED": str(stopped),
        }
    )
    return pyenv, sim, environment, model_calls, sim_calls, stopped


def _calls(path: Path):
    if not path.exists():
        return []
    return [line.split("\x1f")[:-1] for line in path.read_text().splitlines()]


def _run(tmp_path: Path, *arguments: str, updates=None, existing_outputs=()):
    pyenv, sim, environment, model_calls, sim_calls, stopped = _fake_processes(tmp_path)
    environment.update(updates or {})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "output"
    for relative_path, content in existing_outputs:
        target = output / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--pyenv-bin",
            str(pyenv),
            "--sim-python",
            str(sim),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    return completed, _calls(model_calls), _calls(sim_calls), stopped, output


def test_launcher_validates_then_forwards_separate_model_and_sim_devices(tmp_path):
    completed, model_calls, sim_calls, stopped, output = _run(
        tmp_path,
        "--device",
        "cuda:4",
        "--sim-device",
        "cuda:5",
        "--precision",
        "fp32",
        "--tasks",
        "all",
        "--action-postprocessing",
        "first_action",
        "--smoke-test",
    )

    assert completed.returncode == 0, completed.stderr
    assert "octo_small_official_pytorch.checkpoint_cli" in model_calls[0]
    server = model_calls[1]
    assert server[server.index("--device") + 1] == "cuda:4"
    assert server[server.index("--precision") + 1] == "fp32"
    assert "--sim-device" not in server
    assert len(sim_calls) == 1
    simulation = sim_calls[0]
    assert simulation[simulation.index("--sim-device") + 1] == "cuda:5"
    assert "--device" not in simulation
    assert simulation[simulation.index("--action-postprocessing") + 1] == "first_action"
    assert "--smoke-test" in simulation
    socket_path = Path(simulation[simulation.index("--socket") + 1])
    assert not socket_path.exists()
    assert stopped.read_text(encoding="utf-8") == "stopped"
    assert (output / "model-server.log").is_file()


def test_launcher_manifest_failure_prevents_model_and_simulator_start(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output = _run(
        tmp_path, updates={"OFFICIAL_TEST_VALIDATION_STATUS": "9"}
    )

    assert completed.returncode == 2
    assert len(model_calls) == 1
    assert sim_calls == []
    assert "neither a valid official base artifact nor an official fine-tune" in (
        completed.stderr
    )


def test_launcher_rejects_existing_results_before_truncating_model_log(tmp_path):
    completed, model_calls, sim_calls, _stopped, output = _run(
        tmp_path,
        existing_outputs=(("results.json", "prior result"), ("model-server.log", "prior log")),
    )

    assert completed.returncode == 2
    assert model_calls == []
    assert sim_calls == []
    assert (output / "model-server.log").read_text(encoding="utf-8") == "prior log"
    assert "use --overwrite" in completed.stderr


def test_launcher_reports_model_exit_before_starting_simulator(tmp_path):
    completed, model_calls, sim_calls, _stopped, _output = _run(
        tmp_path,
        "--server-timeout",
        "2",
        updates={"OFFICIAL_TEST_MODEL_CRASH": "1"},
    )

    assert completed.returncode == 23
    assert len(model_calls) == 2
    assert sim_calls == []
    assert "model server exited before ready" in completed.stderr


def test_launcher_readiness_timeout_cleans_child_without_starting_simulator(tmp_path):
    completed, _model_calls, sim_calls, stopped, _output = _run(
        tmp_path,
        "--server-timeout",
        "1",
        updates={"OFFICIAL_TEST_NO_SOCKET": "1"},
    )

    assert completed.returncode == 1
    assert sim_calls == []
    assert "timed out after 1s" in completed.stderr
    assert stopped.read_text(encoding="utf-8") == "stopped"


def test_launcher_propagates_simulator_failure_and_cleans_socket(tmp_path):
    completed, _model_calls, sim_calls, _stopped, _output = _run(
        tmp_path, updates={"OFFICIAL_TEST_SIM_STATUS": "7"}
    )

    assert completed.returncode == 7
    socket_path = Path(sim_calls[0][sim_calls[0].index("--socket") + 1])
    assert not socket_path.exists()
