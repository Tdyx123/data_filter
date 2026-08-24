import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_parallel_module_help_executes_the_coordinator_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "octo_small_bridge.parallel_evaluation", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--model-devices" in completed.stdout


def _make_fake_parallel_runtimes(tmp_path: Path):
    model_calls = tmp_path / "model-calls.jsonl"
    sim_calls = tmp_path / "sim-calls.jsonl"
    model_python = tmp_path / "fake-model-python"
    model_python.write_text(
        r"""#!/usr/bin/env python3
import json
import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path

arguments = sys.argv[1:]
def option(name):
    return arguments[arguments.index(name) + 1]

socket_path = option("--socket")
authkey = bytes.fromhex(option("--auth-key-hex"))
device = option("--device")
checkpoint = option("--checkpoint")
with Path(os.environ["OCTO_PARALLEL_TEST_MODEL_CALLS"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"device": device, "pid": os.getpid()}) + "\n")

metadata = {
    "model": "fake-octo",
    "native_action_chunk_size": 8,
    "action_dim": 7,
    "device": device,
    "precision": option("--precision"),
    "checkpoint": {"requested_path": checkpoint},
    "protocol": {
        "native_action_chunk_size": 8,
        "model_action_gripper": "continuous_model_prediction",
        "precision": option("--precision"),
    },
    "startup_preflight": {"action_shape": [1, 8, 7], "finite": True},
    "protocol_version": 2,
}
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
""",
        encoding="utf-8",
    )
    model_python.chmod(0o755)

    sim_python = tmp_path / "fake-sim-python"
    sim_python.write_text(
        r"""#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

arguments = sys.argv[1:]
if arguments[:2] == ["-m", "octo_small_bridge.evaluate_simpler"]:
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
parser.add_argument("--action-postprocessing", default="octo_temporal_ensemble_v1")
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--preflight-only", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parsed, _unknown = parser.parse_known_args(arguments)

with Path(os.environ["OCTO_PARALLEL_TEST_SIM_CALLS"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "sim_device": parsed.sim_device,
        "shard_index": parsed.shard_index,
        "shard_count": parsed.shard_count,
        "pid": os.getpid(),
    }) + "\n")
if os.environ.get("OCTO_PARALLEL_TEST_FAIL_SHARD") == str(parsed.shard_index):
    raise SystemExit(int(os.environ.get("OCTO_PARALLEL_TEST_FAIL_STATUS", "17")))
if os.environ.get("OCTO_PARALLEL_TEST_SIM_PAUSE") == "1":
    while True:
        time.sleep(1)

authkey = bytes.fromhex(parsed.auth_key_hex)
connection = Client(parsed.socket, family="AF_UNIX", authkey=authkey)
connection.send({"type": "metadata"})
metadata_response = connection.recv()
metadata = metadata_response["data"]
parsed.output_dir.mkdir(parents=True, exist_ok=True)
if parsed.preflight_only:
    report = {
        "schema_version": 1,
        "status": "passed",
        "route": "octo-small-bridge-simpler-widowx-preflight",
        "checkpoint": metadata["checkpoint"],
        "protocol": metadata["protocol"],
        "environments": [],
        "model_inference": {"task": "spoon", "action_shape": [1, 8, 7]},
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
        "execution_mode": parsed.action_postprocessing,
        "instruction_source": "environment",
        "episode_protocol": "octo_reference_3seed_288",
        "environment_lifecycle": "one_per_task",
        "sim_renderer_device": parsed.sim_device,
        "rng_scope": "per_episode",
        "rng_seed_derivation": "sha256-octo-simpler-episode-v1",
        "shard_index": parsed.shard_index,
        "shard_count": parsed.shard_count,
        "native_action_chunk_size": 8,
        "model_action_gripper": "continuous_model_prediction",
        "action_postprocessing": parsed.action_postprocessing,
        "temporal_ensemble_prediction_horizon": 8 if parsed.action_postprocessing == "octo_temporal_ensemble_v1" else None,
        "temporal_ensemble_temperature": 0.0 if parsed.action_postprocessing == "octo_temporal_ensemble_v1" else None,
        "gripper_binarization": "after_action_ensemble" if parsed.action_postprocessing == "octo_temporal_ensemble_v1" else "at_environment_conversion",
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "route": "octo-small-bridge-simpler-widowx-eval",
        "checkpoint": metadata["checkpoint"],
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
if os.environ.get("OCTO_PARALLEL_TEST_PAUSE_AFTER_SHUTDOWN") == "1":
    time.sleep(0.5)
connection.close()
""",
        encoding="utf-8",
    )
    sim_python.chmod(0o755)
    environment = os.environ.copy()
    environment["OCTO_PARALLEL_TEST_MODEL_CALLS"] = str(model_calls)
    environment["OCTO_PARALLEL_TEST_SIM_CALLS"] = str(sim_calls)
    return model_python, sim_python, model_calls, sim_calls, environment


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
    checkpoint: str = "/models/checkpoint",
):
    path.mkdir(parents=True)
    protocol = {
        "tasks": ["spoon", "carrot"],
        "policy_seeds": [0],
        "object_episode_ids": [0, 1],
        "planned_episodes": 4,
        "action_horizon": 1,
        "execution_mode": "octo_temporal_ensemble_v1",
        "instruction_source": "environment",
        "episode_protocol": "octo_reference_3seed_288",
        "environment_lifecycle": "one_per_task",
        "sim_renderer_device": "cuda:7",
        "rng_scope": "per_episode",
        "rng_seed_derivation": "sha256-octo-simpler-episode-v1",
        "shard_index": shard_index,
        "shard_count": 2,
        "native_action_chunk_size": 8,
        "model_action_gripper": "continuous_model_prediction",
        "action_postprocessing": "octo_temporal_ensemble_v1",
        "temporal_ensemble_prediction_horizon": 8,
        "temporal_ensemble_temperature": 0.0,
        "gripper_binarization": "after_action_ensemble",
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "route": "octo-parallel-test",
        "checkpoint": {"requested_path": checkpoint},
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
    from octo_small_bridge import parallel_evaluation

    assert parallel_evaluation.parse_model_devices("cuda:0,cuda:2,cuda:7") == (
        "cuda:0",
        "cuda:2",
        "cuda:7",
    )


@pytest.mark.parametrize(
    "value",
    ("", "cuda:0,", "cuda:0,cuda:0", "cuda", "cpu", "CUDA:1", "cuda:-1"),
)
def test_parse_model_devices_rejects_empty_duplicate_or_non_cuda_values(value):
    from octo_small_bridge import parallel_evaluation

    with pytest.raises(ValueError, match="model devices"):
        parallel_evaluation.parse_model_devices(value)


def test_parallel_coordinator_runs_model_replicas_and_shared_renderer_workers(
    tmp_path,
    monkeypatch,
):
    from octo_small_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    for name, value in environment.items():
        if name.startswith("OCTO_PARALLEL_TEST_"):
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("OCTO_PARALLEL_TEST_PAUSE_AFTER_SHUTDOWN", "1")
    output_dir = tmp_path / "output"

    status = parallel_evaluation.main(
        [
            "--model-python",
            str(model_python),
            "--sim-python",
            str(sim_python),
            "--checkpoint",
            "/models/checkpoint",
            "--base-model",
            "/models/base",
            "--model-devices",
            "cuda:0,cuda:2",
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
    )

    assert status == 0
    model_calls = [json.loads(line) for line in model_calls_path.read_text().splitlines()]
    sim_calls = [json.loads(line) for line in sim_calls_path.read_text().splitlines()]
    assert sorted(call["device"] for call in model_calls) == ["cuda:0", "cuda:2"]
    assert sorted(call["shard_index"] for call in sim_calls) == [0, 1]
    assert {call["shard_count"] for call in sim_calls} == {2}
    assert {call["sim_device"] for call in sim_calls} == {"cuda:7"}
    report = json.loads((output_dir / "results.json").read_text())
    assert report["summary"]["completed_episodes"] == 72
    assert report["protocol"]["parallel_workers"] == 2
    assert report["protocol"]["environment_lifecycle"] == "one_per_task_per_worker"
    assert report["protocol"]["execution_mode"] == "octo_temporal_ensemble_v1"
    assert report["protocol"]["instruction_source"] == "environment"
    assert report["protocol"]["episode_protocol"] == "octo_reference_3seed_288"
    assert report["protocol"]["action_postprocessing"] == "octo_temporal_ensemble_v1"
    assert report["protocol"]["temporal_ensemble_prediction_horizon"] == 8
    assert report["protocol"]["temporal_ensemble_temperature"] == 0.0
    assert report["protocol"]["gripper_binarization"] == "after_action_ensemble"
    assert report["runtime"]["model_devices"] == ["cuda:0", "cuda:2"]
    assert len((output_dir / "episodes.jsonl").read_text().splitlines()) == 72
    assert (output_dir / "model-server-00.log").exists()
    assert (output_dir / "model-server-01.log").exists()


def test_parallel_preflight_loads_every_model_but_runs_one_simulator(tmp_path, monkeypatch):
    from octo_small_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    for name, value in environment.items():
        if name.startswith("OCTO_PARALLEL_TEST_"):
            monkeypatch.setenv(name, value)
    output_dir = tmp_path / "output"

    status = parallel_evaluation.main(
        [
            "--model-python",
            str(model_python),
            "--sim-python",
            str(sim_python),
            "--checkpoint",
            "/models/checkpoint",
            "--base-model",
            "/models/base",
            "--model-devices",
            "cuda:0,cuda:1,cuda:2",
            "--sim-device",
            "cuda:7",
            "--output-dir",
            str(output_dir),
            "--server-timeout",
            "3",
            "--",
            "--tasks",
            "spoon",
            "--preflight-only",
        ]
    )

    assert status == 0
    assert len(model_calls_path.read_text().splitlines()) == 3
    assert len(sim_calls_path.read_text().splitlines()) == 1
    report = json.loads((output_dir / "preflight.json").read_text())
    assert report["status"] == "passed"
    assert report["runtime"]["model_devices"] == ["cuda:0", "cuda:1", "cuda:2"]
    assert report["runtime"]["sim_device"] == "cuda:7"


def test_parallel_smoke_test_does_not_load_replicas_without_episode_work(
    tmp_path,
    monkeypatch,
):
    from octo_small_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    for name, value in environment.items():
        if name.startswith("OCTO_PARALLEL_TEST_"):
            monkeypatch.setenv(name, value)
    output_dir = tmp_path / "output"

    status = parallel_evaluation.main(
        [
            "--model-python",
            str(model_python),
            "--sim-python",
            str(sim_python),
            "--checkpoint",
            "/models/checkpoint",
            "--base-model",
            "/models/base",
            "--model-devices",
            "cuda:0,cuda:1,cuda:2",
            "--sim-device",
            "cuda:7",
            "--output-dir",
            str(output_dir),
            "--server-timeout",
            "3",
            "--",
            "--tasks",
            "spoon",
            "--smoke-test",
        ]
    )

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


def test_parallel_worker_failure_stops_all_model_processes_and_writes_failure(
    tmp_path,
    monkeypatch,
):
    from octo_small_bridge import parallel_evaluation

    model_python, sim_python, model_calls_path, _sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    for name, value in environment.items():
        if name.startswith("OCTO_PARALLEL_TEST_"):
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("OCTO_PARALLEL_TEST_FAIL_SHARD", "1")
    monkeypatch.setenv("OCTO_PARALLEL_TEST_FAIL_STATUS", "17")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "failure.json").write_text(
        json.dumps({"status": "failed", "exit_code": 99, "error": "stale"}),
        encoding="utf-8",
    )

    status = parallel_evaluation.main(
        [
            "--model-python",
            str(model_python),
            "--sim-python",
            str(sim_python),
            "--checkpoint",
            "/models/checkpoint",
            "--base-model",
            "/models/base",
            "--model-devices",
            "cuda:0,cuda:1",
            "--sim-device",
            "cuda:7",
            "--output-dir",
            str(output_dir),
            "--server-timeout",
            "3",
            "--",
            "--tasks",
            "spoon",
            "--overwrite",
        ]
    )

    assert status == 17
    failure = json.loads((output_dir / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["exit_code"] == 17
    assert "simulator worker 1" in failure["error"]
    model_pids = [json.loads(line)["pid"] for line in model_calls_path.read_text().splitlines()]
    for pid in model_pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_parallel_coordinator_term_signal_reaps_children_and_reports_130(tmp_path):
    model_python, sim_python, model_calls_path, sim_calls_path, environment = (
        _make_fake_parallel_runtimes(tmp_path)
    )
    environment["OCTO_PARALLEL_TEST_SIM_PAUSE"] = "1"
    project_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(project_root / "src")
    output_dir = tmp_path / "output"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "octo_small_bridge.parallel_evaluation",
            "--model-python",
            str(model_python),
            "--sim-python",
            str(sim_python),
            "--checkpoint",
            "/models/checkpoint",
            "--base-model",
            "/models/base",
            "--model-devices",
            "cuda:0,cuda:1",
            "--sim-device",
            "cuda:7",
            "--output-dir",
            str(output_dir),
            "--server-timeout",
            "3",
            "--",
            "--tasks",
            "spoon",
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
    model_pids = [json.loads(line)["pid"] for line in model_calls_path.read_text().splitlines()]
    for pid in model_pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_merge_worker_outputs_validates_and_restores_canonical_episode_order(tmp_path):
    from octo_small_bridge import parallel_evaluation
    from simpler_bridge.evaluation import SIMPLER_TASKS

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
    assert report["protocol"]["environment_lifecycle"] == "one_per_task_per_worker"
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
    assert json.loads((output_dir / "results.json").read_text()) == report


def test_merge_worker_outputs_rejects_duplicate_episode_assignments(tmp_path):
    from octo_small_bridge import parallel_evaluation
    from simpler_bridge.evaluation import SIMPLER_TASKS

    worker_dirs = (tmp_path / "worker-0", tmp_path / "worker-1")
    first, second = _worker_episodes()
    _write_worker(worker_dirs[0], shard_index=0, episodes=first)
    _write_worker(worker_dirs[1], shard_index=1, episodes=[first[0], *second])

    with pytest.raises(parallel_evaluation.ParallelEvaluationError, match="duplicate"):
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


def test_merge_worker_outputs_rejects_mismatched_checkpoints(tmp_path):
    from octo_small_bridge import parallel_evaluation
    from simpler_bridge.evaluation import SIMPLER_TASKS

    worker_dirs = (tmp_path / "worker-0", tmp_path / "worker-1")
    first, second = _worker_episodes()
    _write_worker(worker_dirs[0], shard_index=0, episodes=first)
    _write_worker(
        worker_dirs[1],
        shard_index=1,
        episodes=second,
        checkpoint="/models/other-checkpoint",
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
