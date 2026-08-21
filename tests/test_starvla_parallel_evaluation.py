import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def test_parallel_module_help_executes_the_coordinator_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "starvla_bridge.parallel_evaluation", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--model-devices" in completed.stdout
    assert "--model-dir" in completed.stdout
    assert "--base-model" in completed.stdout


def _make_fake_parallel_runtimes(tmp_path: Path):
    model_calls = tmp_path / "model-calls.jsonl"
    sim_calls = tmp_path / "sim-calls.jsonl"
    model_python = tmp_path / "fake-model-python"
    model_python.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

arguments = sys.argv[1:]
def option(name):
    return arguments[arguments.index(name) + 1]

socket_path = option("--socket")
authkey = bytes.fromhex(option("--auth-key-hex"))
device = option("--device")
model_dir = option("--model-dir")
base_model = option("--base-model")
descendant = None
if os.environ.get("STARVLA_PARALLEL_TEST_SPAWN_DESCENDANT") == "1":
    descendant = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
with Path(os.environ["STARVLA_PARALLEL_TEST_MODEL_CALLS"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "device": device,
        "model_dir": model_dir,
        "base_model": base_model,
        "pid": os.getpid(),
        "descendant_pid": None if descendant is None else descendant.pid,
    }) + "\n")

if os.environ.get("STARVLA_PARALLEL_TEST_NOT_READY_DEVICE") == device:
    while True:
        time.sleep(1)

metadata = {
    "protocol_version": 2,
    "model": "Qwen3VL-GR00T-Bridge-RT-1",
    "starvla_source_commit": "fake-starvla-commit",
    "model_dir": model_dir,
    "checkpoint_path": model_dir + "/checkpoints/model.pt",
    "base_model": base_model,
    "native_action_chunk_size": 16,
    "action_dim": 7,
    "available_unnorm_keys": ["oxe_bridge"],
    "checkpoint_tensor_count": 962,
    "checkpoint_parameter_bytes": 9976489486,
    "checkpoint_dtypes": ["torch.bfloat16"],
    "device": device,
    "runtime": {"python": "3.12.12", "torch": "2.10.0+cu128"},
    "startup_preflight": {"action_shape": [1, 16, 7], "finite": True},
}
if os.environ.get("STARVLA_PARALLEL_TEST_METADATA_MISMATCH_DEVICE") == device:
    metadata["checkpoint_tensor_count"] += 1
if os.environ.get("STARVLA_PARALLEL_TEST_WRONG_REPORTED_DEVICE") == device:
    metadata["device"] = "cuda:99"
listener = Listener(socket_path, family="AF_UNIX", authkey=authkey)
stopping = False
while not stopping:
    connection = listener.accept()
    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if request.get("type") == "metadata":
                connection.send({"ok": True, "data": metadata})
            elif request.get("type") == "shutdown":
                connection.send({"ok": True, "data": {"shutdown": True}})
                stopping = True
                break
            else:
                connection.send({"ok": False, "error": "unsupported fake request"})
    finally:
        connection.close()
listener.close()
Path(socket_path).unlink(missing_ok=True)
''',
        encoding="utf-8",
    )
    model_python.chmod(0o755)

    sim_python = tmp_path / "fake-sim-python"
    sim_python.write_text(
        r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

arguments = sys.argv[1:]
if arguments[:2] == ["-m", "starvla_bridge.evaluate_simpler"]:
    arguments = arguments[2:]
parser = argparse.ArgumentParser()
parser.add_argument("--socket", required=True)
parser.add_argument("--auth-key-hex", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--sim-device", required=True)
parser.add_argument("--tasks", default="all")
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--shard-count", type=int, default=1)
parser.add_argument("--rng-scope", default="per_policy_seed_stream")
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--preflight-only", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parsed, _unknown = parser.parse_known_args(arguments)

with Path(os.environ["STARVLA_PARALLEL_TEST_SIM_CALLS"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "sim_device": parsed.sim_device,
        "shard_index": parsed.shard_index,
        "shard_count": parsed.shard_count,
        "rng_scope": parsed.rng_scope,
        "pid": os.getpid(),
    }) + "\n")
if os.environ.get("STARVLA_PARALLEL_TEST_FAIL_SHARD") == str(parsed.shard_index):
    raise SystemExit(int(os.environ.get("STARVLA_PARALLEL_TEST_FAIL_STATUS", "17")))
if os.environ.get("STARVLA_PARALLEL_TEST_SIM_PAUSE") == "1":
    while True:
        time.sleep(1)

authkey = bytes.fromhex(parsed.auth_key_hex)
connection = Client(parsed.socket, family="AF_UNIX", authkey=authkey)
connection.send({"type": "metadata"})
metadata = connection.recv()["data"]
checkpoint = {
    key: metadata.get(key)
    for key in (
        "model_dir",
        "checkpoint_path",
        "base_model",
        "checkpoint_tensor_count",
        "checkpoint_parameter_bytes",
        "checkpoint_dtypes",
    )
}
parsed.output_dir.mkdir(parents=True, exist_ok=True)
if parsed.preflight_only:
    report = {
        "schema_version": 1,
        "status": "passed",
        "route": "qwen3vl-groot-starvla-simpler-widowx-preflight",
        "checkpoint": checkpoint,
        "protocol": {"native_action_chunk_size": 16},
        "environments": [],
        "model_inference": {"task": "spoon", "action_shape": [1, 16, 7]},
        "runtime": {"device": "remote-pyenv:" + metadata["device"]},
    }
    (parsed.output_dir / "preflight.json").write_text(json.dumps(report), encoding="utf-8")
else:
    task_keys = ["spoon", "carrot", "stack", "eggplant"] if parsed.tasks == "all" else parsed.tasks.split(",")
    policy_seeds = [0] if parsed.smoke_test else [0, 2, 4]
    object_episode_ids = [0] if parsed.smoke_test else list(range(24))
    episodes = []
    canonical_index = 0
    for task in task_keys:
        for policy_seed in policy_seeds:
            for object_episode_id in object_episode_ids:
                if canonical_index % parsed.shard_count == parsed.shard_index:
                    payload = "\0".join((
                        "octo-simpler-episode-v1",
                        task,
                        str(policy_seed),
                        str(object_episode_id),
                    )).encode("utf-8")
                    inference_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
                    episodes.append({
                        "task": task,
                        "instruction": "fake instruction",
                        "seed": policy_seed,
                        "policy_seed": policy_seed,
                        "object_episode_id": object_episode_id,
                        "inference_seed": inference_seed,
                        "success": True,
                        "steps": 1,
                        "termination": "success",
                        "episode_stats": {},
                    })
                canonical_index += 1
    protocol = {
        "tasks": task_keys,
        "policy_seeds": policy_seeds,
        "object_episode_ids": object_episode_ids,
        "planned_episodes": len(task_keys) * len(policy_seeds) * len(object_episode_ids),
        "action_horizon": 1,
        "execution_mode": "stepwise_first_action",
        "environment_lifecycle": "one_per_task",
        "sim_renderer_device": parsed.sim_device,
        "rng_scope": "per_episode",
        "rng_seed_derivation": "sha256-octo-simpler-episode-v1",
        "shard_index": parsed.shard_index,
        "shard_count": parsed.shard_count,
        "native_action_chunk_size": 16,
        "model": metadata["model"],
        "starvla_source_commit": metadata["starvla_source_commit"],
        "checkpoint_tensor_count": metadata["checkpoint_tensor_count"],
        "unnorm_key": "oxe_bridge",
        "uses_proprio": False,
        "terminate_episode": 0,
        "model_runtime": metadata["runtime"],
        "startup_preflight": metadata["startup_preflight"],
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "route": "qwen3vl-groot-starvla-simpler-widowx-eval",
        "checkpoint": checkpoint,
        "protocol": protocol,
        "summary": {"completed_episodes": len(episodes)},
        "task_errors": [],
        "runtime": {
            "device": "remote-pyenv:" + metadata["device"],
            "python": "3.11.9",
            "simpler_env_commit": "abc",
            "elapsed_seconds": 1.0,
        },
        "videos": [],
    }
    (parsed.output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    (parsed.output_dir / "results.json").write_text(json.dumps(report), encoding="utf-8")
connection.send({"type": "shutdown"})
connection.recv()
if os.environ.get("STARVLA_PARALLEL_TEST_PAUSE_AFTER_SHUTDOWN") == "1":
    time.sleep(0.5)
connection.close()
''',
        encoding="utf-8",
    )
    sim_python.chmod(0o755)
    environment = os.environ.copy()
    environment["STARVLA_PARALLEL_TEST_MODEL_CALLS"] = str(model_calls)
    environment["STARVLA_PARALLEL_TEST_SIM_CALLS"] = str(sim_calls)
    return model_python, sim_python, model_calls, sim_calls, environment


def _coordinator_arguments(model_python, sim_python, output_dir, *, devices="cuda:0,cuda:1"):
    return [
        "--model-python",
        str(model_python),
        "--sim-python",
        str(sim_python),
        "--model-dir",
        "/models/starvla",
        "--base-model",
        "/models/qwen",
        "--model-devices",
        devices,
        "--sim-device",
        "cuda:7",
        "--output-dir",
        str(output_dir),
        "--server-timeout",
        "3",
        "--",
        "--tasks",
        "spoon",
    ]


def _install_environment(monkeypatch, environment):
    for name, value in environment.items():
        if name.startswith("STARVLA_PARALLEL_TEST_"):
            monkeypatch.setenv(name, value)


def _assert_processes_reaped(calls_path: Path):
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    pids = [call[key] for call in calls for key in ("pid", "descendant_pid") if call[key]]
    for pid in pids:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            stat_path = Path(f"/proc/{pid}/stat")
            state = stat_path.read_text().split()[2] if stat_path.exists() else "gone"
            pytest.fail(f"process {pid} survived coordinator cleanup with state {state}")


class _FakeLifecycleProcess:
    def __init__(self, name: str, *, poll_status=None, wait_status=0):
        self.name = name
        self.poll_status = poll_status
        self.wait_status = wait_status
        self.returncode = poll_status

    def poll(self):
        return self.poll_status

    def wait(self, timeout=None):
        del timeout
        self.returncode = self.wait_status
        self.poll_status = self.wait_status
        return self.wait_status


def _prepare_fake_lifecycle_run(
    tmp_path,
    monkeypatch,
    parallel_evaluation,
    *,
    device_count=2,
):
    models = [_FakeLifecycleProcess(f"model-{index}") for index in range(device_count)]
    simulators = [
        _FakeLifecycleProcess(f"simulator-{index}") for index in range(device_count)
    ]
    pending_processes = iter([*models, *simulators])
    log_handles = []

    def launch_process(*_args, **kwargs):
        if kwargs.get("stdout") is not None:
            log_handles.append(kwargs["stdout"])
        return next(pending_processes)

    ipc_root = tmp_path / "ipc"

    def make_ipc_root(*_args, **_kwargs):
        ipc_root.mkdir()
        return str(ipc_root)

    monkeypatch.setattr(parallel_evaluation.subprocess, "Popen", launch_process)
    monkeypatch.setattr(parallel_evaluation.tempfile, "mkdtemp", make_ipc_root)
    monkeypatch.setattr(parallel_evaluation, "_wait_for_model_servers", lambda *_: None)
    monkeypatch.setattr(parallel_evaluation, "_validate_replica_metadata", lambda *_: None)
    monkeypatch.setattr(
        parallel_evaluation,
        "merge_worker_outputs",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    output_dir = tmp_path / "output"
    arguments = parallel_evaluation.build_parser().parse_args(
        _coordinator_arguments(
            sys.executable,
            sys.executable,
            output_dir,
            devices=",".join(f"cuda:{index}" for index in range(device_count)),
        )
    )
    return arguments, models, simulators, log_handles, ipc_root, output_dir


def _episode(task: str, object_episode_id: int, inference_seed: int, *, success: bool):
    return {
        "task": task,
        "instruction": f"instruction-{task}",
        "seed": 0,
        "policy_seed": 0,
        "object_episode_id": object_episode_id,
        "inference_seed": inference_seed,
        "success": success,
        "steps": 1,
        "termination": "success" if success else "max_steps",
        "episode_stats": {},
    }


def _write_worker(
    path: Path,
    *,
    shard_index: int,
    episodes: list[dict],
    checkpoint_path: str = "/models/starvla/checkpoints/model.pt",
):
    path.mkdir(parents=True)
    protocol = {
        "tasks": ["spoon", "carrot"],
        "policy_seeds": [0],
        "object_episode_ids": [0, 1],
        "planned_episodes": 4,
        "action_horizon": 1,
        "execution_mode": "stepwise_first_action",
        "environment_lifecycle": "one_per_task",
        "sim_renderer_device": "cuda:7",
        "rng_scope": "per_episode",
        "rng_seed_derivation": "sha256-octo-simpler-episode-v1",
        "shard_index": shard_index,
        "shard_count": 2,
        "native_action_chunk_size": 16,
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "route": "starvla-parallel-test",
        "checkpoint": {
            "model_dir": "/models/starvla",
            "checkpoint_path": checkpoint_path,
            "base_model": "/models/qwen",
        },
        "protocol": protocol,
        "summary": {"completed_episodes": len(episodes)},
        "task_errors": [],
        "runtime": {
            "device": f"remote-pyenv:cuda:{shard_index}",
            "python": "3.11.9",
            "simpler_env_commit": "abc",
            "elapsed_seconds": 10.0 + shard_index,
        },
        "videos": [f"video-{shard_index}.mp4"],
    }
    (path / "results.json").write_text(json.dumps(report), encoding="utf-8")
    (path / "episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )


def _worker_episodes():
    return (
        [
            _episode("spoon", 0, 8361816881672972874, success=True),
            _episode("carrot", 0, 987892709028713917, success=False),
        ],
        [
            _episode("spoon", 1, 6127971071483545067, success=True),
            _episode("carrot", 1, 1791360930171158603, success=True),
        ],
    )


def test_parse_model_devices_accepts_unique_logical_cuda_devices():
    from starvla_bridge import parallel_evaluation

    assert parallel_evaluation.parse_model_devices("cuda:0,cuda:2,cuda:7") == (
        "cuda:0",
        "cuda:2",
        "cuda:7",
    )


@pytest.mark.parametrize(
    "value",
    (
        "",
        "cuda:0,",
        "cuda:0,cuda:0",
        "cuda",
        "cpu",
        "CUDA:1",
        "cuda:-1",
        "cuda:00",
        "cuda:01",
    ),
)
def test_parse_model_devices_rejects_empty_duplicate_or_non_cuda_values(value):
    from starvla_bridge import parallel_evaluation

    with pytest.raises(ValueError, match="model devices"):
        parallel_evaluation.parse_model_devices(value)


@pytest.mark.parametrize(
    ("option", "abbreviation"),
    (
        ("--model-python", "--model-p"),
        ("--sim-python", "--sim-p"),
        ("--model-dir", "--model-di"),
        ("--base-model", "--base-m"),
        ("--model-devices", "--model-de"),
        ("--sim-device", "--sim-d"),
        ("--output-dir", "--output"),
        ("--server-timeout", "--server-t"),
    ),
)
def test_parallel_parser_rejects_abbreviated_coordinator_options(
    tmp_path, option, abbreviation
):
    from starvla_bridge import parallel_evaluation

    argv = _coordinator_arguments(
        sys.executable,
        sys.executable,
        tmp_path / "output",
        devices="cuda:0",
    )
    argv[argv.index(option)] = abbreviation

    with pytest.raises(SystemExit):
        parallel_evaluation.build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "forwarded",
    (
        ("--socket", "/tmp/override.sock"),
        ("--socket=/tmp/override.sock",),
        ("--auth-key-hex", "beef"),
        ("--auth-key-hex=beef",),
        ("--output-dir", "/tmp/override-output"),
        ("--output-dir=/tmp/override-output",),
        ("--sim-device", "cuda:6"),
        ("--sim-device=cuda:6",),
        ("--shard-index", "0"),
        ("--shard-index=0",),
        ("--shard-count", "1"),
        ("--shard-count=1",),
        ("--rng-scope", "per_episode"),
        ("--rng-scope=per_episode",),
    ),
)
def test_parallel_rejects_coordinator_managed_forwarded_arguments_before_launch(
    tmp_path, monkeypatch, forwarded
):
    from starvla_bridge import parallel_evaluation

    output_dir = tmp_path / "output"
    arguments = parallel_evaluation.build_parser().parse_args(
        [
            "--sim-python",
            sys.executable,
            "--model-dir",
            "/models/starvla",
            "--base-model",
            "/models/qwen",
            "--model-devices",
            "cuda:0",
            "--sim-device",
            "cuda:7",
            "--output-dir",
            str(output_dir),
            "--",
            *forwarded,
            "--tasks",
            "spoon",
        ]
    )

    def reject_process_launch(*_args, **_kwargs):
        pytest.fail("managed forwarded argument reached process launch")

    monkeypatch.setattr(parallel_evaluation.subprocess, "Popen", reject_process_launch)

    with pytest.raises(parallel_evaluation.ParallelEvaluationError, match="managed"):
        parallel_evaluation.run_parallel(arguments)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "forwarded",
    (
        ("--sock", "/tmp/override.sock"),
        ("--auth", "beef"),
        ("--output", "/tmp/override-output"),
        ("--sim-d", "cuda:6"),
        ("--shard-i", "0"),
        ("--shard-c", "1"),
        ("--rng", "per_episode"),
    ),
)
def test_parallel_rejects_abbreviated_managed_forwarded_arguments_before_launch(
    tmp_path, monkeypatch, forwarded
):
    from starvla_bridge import parallel_evaluation

    output_dir = tmp_path / "output"
    arguments = parallel_evaluation.build_parser().parse_args(
        [
            "--sim-python",
            sys.executable,
            "--model-dir",
            "/models/starvla",
            "--base-model",
            "/models/qwen",
            "--model-devices",
            "cuda:0",
            "--sim-device",
            "cuda:7",
            "--output-dir",
            str(output_dir),
            "--",
            *forwarded,
            "--tasks",
            "spoon",
        ]
    )

    def reject_process_launch(*_args, **_kwargs):
        pytest.fail("abbreviated managed argument reached process launch")

    monkeypatch.setattr(parallel_evaluation.subprocess, "Popen", reject_process_launch)

    with pytest.raises(SystemExit):
        parallel_evaluation.run_parallel(arguments)
    assert not output_dir.exists()


def test_parallel_coordinator_runs_starvla_replicas_and_shared_renderer_workers(
    tmp_path, monkeypatch
):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    monkeypatch.setenv("STARVLA_PARALLEL_TEST_PAUSE_AFTER_SHUTDOWN", "1")
    output_dir = tmp_path / "output"

    status = parallel_evaluation.main(
        _coordinator_arguments(
            model_python,
            sim_python,
            output_dir,
            devices="cuda:0,cuda:2",
        )
    )

    assert status == 0
    model_calls = [json.loads(line) for line in model_calls_path.read_text().splitlines()]
    sim_calls = [json.loads(line) for line in sim_calls_path.read_text().splitlines()]
    assert sorted(call["device"] for call in model_calls) == ["cuda:0", "cuda:2"]
    assert {call["model_dir"] for call in model_calls} == {"/models/starvla"}
    assert {call["base_model"] for call in model_calls} == {"/models/qwen"}
    assert sorted(call["shard_index"] for call in sim_calls) == [0, 1]
    assert {call["shard_count"] for call in sim_calls} == {2}
    assert {call["rng_scope"] for call in sim_calls} == {"per_episode"}
    assert {call["sim_device"] for call in sim_calls} == {"cuda:7"}
    report = json.loads((output_dir / "results.json").read_text())
    assert report["summary"]["completed_episodes"] == 72
    assert report["protocol"]["parallel_workers"] == 2
    assert report["protocol"]["environment_lifecycle"] == "one_per_task_per_worker"
    assert report["runtime"]["model_devices"] == ["cuda:0", "cuda:2"]
    assert len((output_dir / "episodes.jsonl").read_text().splitlines()) == 72


def test_parallel_preflight_loads_every_model_but_runs_one_simulator(tmp_path, monkeypatch):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    output_dir = tmp_path / "output"
    arguments = _coordinator_arguments(
        model_python,
        sim_python,
        output_dir,
        devices="cuda:0,cuda:1,cuda:2",
    )
    arguments.append("--preflight-only")

    status = parallel_evaluation.main(arguments)

    assert status == 0
    assert len(model_calls_path.read_text().splitlines()) == 3
    assert len(sim_calls_path.read_text().splitlines()) == 1
    report = json.loads((output_dir / "preflight.json").read_text())
    assert report["status"] == "passed"
    assert report["runtime"]["model_devices"] == ["cuda:0", "cuda:1", "cuda:2"]
    assert report["runtime"]["sim_device"] == "cuda:7"


def test_parallel_smoke_test_does_not_load_replicas_without_episode_work(
    tmp_path, monkeypatch
):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    output_dir = tmp_path / "output"
    arguments = _coordinator_arguments(
        model_python,
        sim_python,
        output_dir,
        devices="cuda:0,cuda:1,cuda:2",
    )
    arguments.append("--smoke-test")

    status = parallel_evaluation.main(arguments)

    assert status == 0
    assert len(model_calls_path.read_text().splitlines()) == 1
    assert len(sim_calls_path.read_text().splitlines()) == 1
    report = json.loads((output_dir / "results.json").read_text())
    assert report["summary"]["completed_episodes"] == 1
    assert report["runtime"]["requested_model_devices"] == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]
    assert report["runtime"]["model_devices"] == ["cuda:0"]


def test_parallel_worker_failure_stops_all_models_and_writes_failure(tmp_path, monkeypatch):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, _sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    monkeypatch.setenv("STARVLA_PARALLEL_TEST_FAIL_SHARD", "1")
    monkeypatch.setenv("STARVLA_PARALLEL_TEST_FAIL_STATUS", "17")
    monkeypatch.setenv("STARVLA_PARALLEL_TEST_SPAWN_DESCENDANT", "1")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "failure.json").write_text(
        json.dumps({"status": "failed", "exit_code": 99, "error": "stale"}),
        encoding="utf-8",
    )

    status = parallel_evaluation.main(
        [*_coordinator_arguments(model_python, sim_python, output_dir), "--overwrite"]
    )

    assert status == 17
    failure = json.loads((output_dir / "failure.json").read_text())
    assert failure["route"] == "qwen3vl-groot-starvla-simpler-widowx-parallel-eval"
    assert failure["exit_code"] == 17
    assert "simulator worker 1" in failure["error"]
    _assert_processes_reaped(model_calls_path)


def test_parallel_rejects_replica_that_reports_the_wrong_device(tmp_path, monkeypatch):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    monkeypatch.setenv("STARVLA_PARALLEL_TEST_WRONG_REPORTED_DEVICE", "cuda:1")
    output_dir = tmp_path / "output"

    status = parallel_evaluation.main(
        _coordinator_arguments(model_python, sim_python, output_dir)
    )

    assert status == 1
    assert not sim_calls_path.exists()
    failure = json.loads((output_dir / "failure.json").read_text())
    assert "reported device 'cuda:99'; expected 'cuda:1'" in failure["error"]
    _assert_processes_reaped(model_calls_path)


def test_parallel_rejects_replica_metadata_mismatch_before_simulation(tmp_path, monkeypatch):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    monkeypatch.setenv("STARVLA_PARALLEL_TEST_METADATA_MISMATCH_DEVICE", "cuda:1")
    output_dir = tmp_path / "output"

    status = parallel_evaluation.main(
        _coordinator_arguments(model_python, sim_python, output_dir)
    )

    assert status == 1
    assert not sim_calls_path.exists()
    failure = json.loads((output_dir / "failure.json").read_text())
    assert "model replica metadata does not match" in failure["error"]
    _assert_processes_reaped(model_calls_path)


def test_parallel_server_readiness_timeout_reaps_unready_model(tmp_path, monkeypatch):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    monkeypatch.setenv("STARVLA_PARALLEL_TEST_NOT_READY_DEVICE", "cuda:0")
    output_dir = tmp_path / "output"
    arguments = _coordinator_arguments(model_python, sim_python, output_dir)
    arguments[arguments.index("--server-timeout") + 1] = "1"

    status = parallel_evaluation.main(arguments)

    assert status == 1
    assert not sim_calls_path.exists()
    failure = json.loads((output_dir / "failure.json").read_text())
    assert "timed out after 1s" in failure["error"]
    _assert_processes_reaped(model_calls_path)


def test_parallel_existing_results_block_launch_without_overwrite(tmp_path, monkeypatch):
    from starvla_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, _sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    _install_environment(monkeypatch, environment)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "results.json").write_text('{"sentinel": true}', encoding="utf-8")

    status = parallel_evaluation.main(
        _coordinator_arguments(model_python, sim_python, output_dir)
    )

    assert status == 1
    assert not model_calls_path.exists()
    assert json.loads((output_dir / "results.json").read_text()) == {"sentinel": True}


def test_parallel_coordinator_term_signal_reaps_children_and_reports_130(tmp_path):
    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    environment["STARVLA_PARALLEL_TEST_SIM_PAUSE"] = "1"
    environment["STARVLA_PARALLEL_TEST_SPAWN_DESCENDANT"] = "1"
    project_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(project_root / "src")
    output_dir = tmp_path / "output"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "starvla_bridge.parallel_evaluation",
            *_coordinator_arguments(model_python, sim_python, output_dir),
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not sim_calls_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sim_calls_path.exists()

    process.terminate()
    _stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 130, stderr
    failure = json.loads((output_dir / "failure.json").read_text())
    assert failure["exit_code"] == 130
    _assert_processes_reaped(model_calls_path)


@pytest.mark.parametrize("primary_kind", ("parallel", "keyboard"))
def test_parallel_cleanup_attempts_every_resource_and_preserves_primary_error(
    tmp_path, monkeypatch, primary_kind
):
    from starvla_bridge import parallel_evaluation

    arguments, _models, _simulators, log_handles, ipc_root, output_dir = (
        _prepare_fake_lifecycle_run(tmp_path, monkeypatch, parallel_evaluation)
    )
    primary_error = (
        parallel_evaluation.ParallelEvaluationError("worker failed", exit_code=17)
        if primary_kind == "parallel"
        else KeyboardInterrupt()
    )

    def fail_during_simulator_wait(_replicas):
        raise primary_error

    cleanup_calls = []

    def fail_first_cleanup(process):
        cleanup_calls.append(process.name)
        if len(cleanup_calls) == 1:
            raise subprocess.TimeoutExpired(process.name, timeout=6)

    monkeypatch.setattr(
        parallel_evaluation, "_wait_for_simulators", fail_during_simulator_wait
    )
    monkeypatch.setattr(parallel_evaluation, "_terminate_process", fail_first_cleanup)

    try:
        parallel_evaluation.run_parallel(arguments)
    except BaseException as error:
        caught = error
    else:
        pytest.fail("primary worker error was not raised")

    assert cleanup_calls == [
        "simulator-0",
        "simulator-1",
        "model-0",
        "model-1",
    ]
    assert all(handle.closed for handle in log_handles)
    assert not ipc_root.exists()
    assert caught is primary_error
    failure = json.loads((output_dir / "failure.json").read_text())
    assert failure["exit_code"] == (17 if primary_kind == "parallel" else 130)


def test_parallel_cleanup_failure_after_success_writes_failure_and_fails(
    tmp_path, monkeypatch
):
    from starvla_bridge import parallel_evaluation

    arguments, _models, _simulators, log_handles, ipc_root, output_dir = (
        _prepare_fake_lifecycle_run(tmp_path, monkeypatch, parallel_evaluation)
    )
    monkeypatch.setattr(parallel_evaluation, "_wait_for_simulators", lambda *_: None)
    cleanup_calls = []

    def fail_final_simulator_cleanup(process):
        cleanup_calls.append(process.name)
        if process.name == "simulator-0":
            raise subprocess.TimeoutExpired(process.name, timeout=6)

    monkeypatch.setattr(
        parallel_evaluation,
        "_terminate_process",
        fail_final_simulator_cleanup,
    )

    try:
        parallel_evaluation.run_parallel(arguments)
    except BaseException as error:
        caught = error
    else:
        pytest.fail("cleanup failure was not raised")

    assert cleanup_calls == [
        "model-0",
        "model-1",
        "simulator-0",
        "simulator-1",
    ]
    assert all(handle.closed for handle in log_handles)
    assert not ipc_root.exists()
    assert isinstance(caught, parallel_evaluation.ParallelEvaluationError)
    assert "cleanup failed" in str(caught)
    failure = json.loads((output_dir / "failure.json").read_text())
    assert failure["exit_code"] == 1
    assert "TimeoutExpired" in failure["error"]


def test_completed_process_groups_are_drained_before_merge_and_not_cleaned_again(
    tmp_path, monkeypatch
):
    from starvla_bridge import parallel_evaluation

    arguments, models, simulators, _log_handles, _ipc_root, _output_dir = (
        _prepare_fake_lifecycle_run(
            tmp_path,
            monkeypatch,
            parallel_evaluation,
            device_count=1,
        )
    )
    simulators[0].poll_status = 0
    simulators[0].returncode = 0
    phase = {"value": "running"}
    cleanup_calls = []

    def record_cleanup(process):
        cleanup_calls.append((process.name, phase["value"]))

    def merge_outputs(*_args, **_kwargs):
        phase["value"] = "merged"
        return {"status": "complete"}

    monkeypatch.setattr(parallel_evaluation, "_terminate_process", record_cleanup)
    monkeypatch.setattr(parallel_evaluation, "merge_worker_outputs", merge_outputs)

    report = parallel_evaluation.run_parallel(arguments)

    assert report == {"status": "complete"}
    assert cleanup_calls == [
        ("simulator-0", "running"),
        ("model-0", "running"),
    ]


@pytest.mark.parametrize(
    ("worker_kind", "exit_code"),
    (("simulator", 17), ("model", 23)),
)
def test_worker_exit_code_survives_immediate_group_drain_failure(
    tmp_path, monkeypatch, worker_kind, exit_code
):
    from starvla_bridge import parallel_evaluation

    arguments, models, simulators, log_handles, ipc_root, output_dir = (
        _prepare_fake_lifecycle_run(
            tmp_path,
            monkeypatch,
            parallel_evaluation,
            device_count=1,
        )
    )
    phase = {"value": "waiting"}
    failed_process = simulators[0] if worker_kind == "simulator" else models[0]

    if worker_kind == "simulator":

        def observe_failure():
            phase["value"] = "observed"
            return exit_code

        simulators[0].poll = observe_failure
    else:

        def observe_failure():
            phase["value"] = "observed"
            return exit_code

        models[0].poll = observe_failure

    cleanup_calls = []

    def fail_failed_group_cleanup(process):
        cleanup_calls.append((process.name, phase["value"]))
        if process is failed_process:
            raise subprocess.TimeoutExpired(process.name, timeout=6)

    original_write_failure = parallel_evaluation._write_failure

    def record_failure(*args, **kwargs):
        phase["value"] = "failure"
        return original_write_failure(*args, **kwargs)

    monkeypatch.setattr(
        parallel_evaluation, "_terminate_process", fail_failed_group_cleanup
    )
    monkeypatch.setattr(parallel_evaluation, "_write_failure", record_failure)

    with pytest.raises(parallel_evaluation.ParallelEvaluationError) as caught:
        parallel_evaluation.run_parallel(arguments)

    assert cleanup_calls[0] == (failed_process.name, "observed")
    assert caught.value.exit_code == exit_code
    failure = json.loads((output_dir / "failure.json").read_text())
    assert failure["exit_code"] == exit_code
    assert all(handle.closed for handle in log_handles)
    assert not ipc_root.exists()


def test_process_group_cleanup_kills_term_ignoring_descendant_after_leader_exits(tmp_path):
    from starvla_bridge import parallel_evaluation

    child_pid_path = tmp_path / "child.pid"
    child_ready_path = tmp_path / "child.ready"
    leader_python = tmp_path / "leader-python"
    leader_python.write_text(
        r'''#!/usr/bin/env python3
import signal
import subprocess
import sys
import time
from pathlib import Path

child_pid_path = Path(sys.argv[1])
child_ready_path = Path(sys.argv[2])
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text('ready'); time.sleep(60)",
        str(child_ready_path),
    ]
)
deadline = time.monotonic() + 5
while not child_ready_path.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
if not child_ready_path.exists():
    raise SystemExit("descendant did not become ready")
child_pid_path.write_text(str(child.pid))
''',
        encoding="utf-8",
    )
    leader_python.chmod(0o755)
    leader = subprocess.Popen(
        [str(leader_python), str(child_pid_path), str(child_ready_path)],
        start_new_session=True,
    )
    assert leader.wait(timeout=5) == 0
    child_pid = int(child_pid_path.read_text())
    os.kill(child_pid, 0)

    try:
        parallel_evaluation._terminate_process(leader)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_merge_worker_outputs_restores_canonical_order_and_aggregates_metadata(tmp_path):
    from simpler_bridge.evaluation import SIMPLER_TASKS
    from starvla_bridge import parallel_evaluation

    worker_dirs = (tmp_path / "worker-0", tmp_path / "worker-1")
    for index, episodes in enumerate(_worker_episodes()):
        _write_worker(worker_dirs[index], shard_index=index, episodes=episodes)

    output_dir = tmp_path / "merged"
    report = parallel_evaluation.merge_worker_outputs(
        output_dir,
        worker_dirs,
        tasks=SIMPLER_TASKS[:2],
        policy_seeds=(0,),
        object_episode_ids=(0, 1),
        requested_devices=("cuda:0", "cuda:1", "cuda:2"),
        active_devices=("cuda:0", "cuda:1"),
        sim_device="cuda:7",
        elapsed_seconds=6.5,
    )

    merged = [json.loads(line) for line in (output_dir / "episodes.jsonl").read_text().splitlines()]
    assert [
        (episode["task"], episode["policy_seed"], episode["object_episode_id"])
        for episode in merged
    ] == [
        ("spoon", 0, 0),
        ("spoon", 0, 1),
        ("carrot", 0, 0),
        ("carrot", 0, 1),
    ]
    assert report["summary"]["completed_episodes"] == 4
    assert report["summary"]["successes"] == 3
    assert report["protocol"]["parallel_workers"] == 2
    assert "shard_index" not in report["protocol"]
    assert "shard_count" not in report["protocol"]
    assert report["runtime"]["requested_model_devices"] == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]
    assert report["runtime"]["model_devices"] == ["cuda:0", "cuda:1"]
    assert report["runtime"]["sim_device"] == "cuda:7"
    assert report["runtime"]["elapsed_seconds"] == 6.5


def test_merge_worker_outputs_rejects_duplicate_episode_assignments(tmp_path):
    from simpler_bridge.evaluation import SIMPLER_TASKS
    from starvla_bridge import parallel_evaluation

    first, second = _worker_episodes()
    duplicate_dirs = (tmp_path / "duplicate-0", tmp_path / "duplicate-1")
    _write_worker(duplicate_dirs[0], shard_index=0, episodes=first)
    _write_worker(duplicate_dirs[1], shard_index=1, episodes=[first[0], *second])
    with pytest.raises(parallel_evaluation.ParallelEvaluationError, match="duplicate"):
        parallel_evaluation.merge_worker_outputs(
            tmp_path / "duplicate-merged",
            duplicate_dirs,
            tasks=SIMPLER_TASKS[:2],
            policy_seeds=(0,),
            object_episode_ids=(0, 1),
            requested_devices=("cuda:0", "cuda:1"),
            active_devices=("cuda:0", "cuda:1"),
            sim_device="cuda:7",
            elapsed_seconds=1.0,
        )


def test_merge_worker_outputs_rejects_missing_episode_assignments(tmp_path):
    from simpler_bridge.evaluation import SIMPLER_TASKS
    from starvla_bridge import parallel_evaluation

    first, second = _worker_episodes()
    missing_dirs = (tmp_path / "missing-0", tmp_path / "missing-1")
    _write_worker(missing_dirs[0], shard_index=0, episodes=first)
    _write_worker(missing_dirs[1], shard_index=1, episodes=second[:-1])
    with pytest.raises(parallel_evaluation.ParallelEvaluationError, match="coverage mismatch"):
        parallel_evaluation.merge_worker_outputs(
            tmp_path / "missing-merged",
            missing_dirs,
            tasks=SIMPLER_TASKS[:2],
            policy_seeds=(0,),
            object_episode_ids=(0, 1),
            requested_devices=("cuda:0", "cuda:1"),
            active_devices=("cuda:0", "cuda:1"),
            sim_device="cuda:7",
            elapsed_seconds=1.0,
        )


def test_merge_worker_outputs_rejects_mismatched_checkpoint_metadata(tmp_path):
    from simpler_bridge.evaluation import SIMPLER_TASKS
    from starvla_bridge import parallel_evaluation

    worker_dirs = (tmp_path / "worker-0", tmp_path / "worker-1")
    first, second = _worker_episodes()
    _write_worker(worker_dirs[0], shard_index=0, episodes=first)
    _write_worker(
        worker_dirs[1],
        shard_index=1,
        episodes=second,
        checkpoint_path="/models/starvla/checkpoints/other.pt",
    )

    with pytest.raises(parallel_evaluation.ParallelEvaluationError, match="checkpoint"):
        parallel_evaluation.merge_worker_outputs(
            tmp_path / "merged",
            worker_dirs,
            tasks=SIMPLER_TASKS[:2],
            policy_seeds=(0,),
            object_episode_ids=(0, 1),
            requested_devices=("cuda:0", "cuda:1"),
            active_devices=("cuda:0", "cuda:1"),
            sim_device="cuda:7",
            elapsed_seconds=1.0,
        )
