import json
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from octo_small_libero.evaluation import (
    ACTION_HORIZON,
    BDDL_VERSION,
    EVALUATION_SEEDS,
    GYMNASIUM_VERSION,
    LIBERO_COMMIT,
    LIBERO_10_TASK_NAMES,
    LIBERO_MULTIPROCESSING_START_METHOD,
    LIBERO_WORKER_GRACEFUL_CLOSE_TIMEOUT_SECONDS,
    LIBERO_WORKER_KILL_TIMEOUT_SECONDS,
    LIBERO_WORKER_TERMINATE_TIMEOUT_SECONDS,
    MUJOCO_COMPATIBILITY,
    MUJOCO_VERSION,
    ROBOSUITE_VERSION,
    CheckpointSpec,
    EvaluationError,
    EvaluationSettings,
    LiberoTask,
    NormalizationStatistics,
    SimulationInfrastructureError,
    SimulationStartupFailure,
    UnrecoverableSimulationShutdownError,
    _close_libero_subprocess_vector_environment,
    _find_libero_package_root,
    _install_libero_gymnasium_compatibility,
    _install_libero_subprocess_worker_bootstrap,
    _install_robosuite_mujoco_310_compatibility,
    _LiberoWorkerCloudpickleWrapper,
    _make_report,
    _run_libero_subprocess_worker_with_gymnasium,
    _validate_vector_environment_startup,
    build_model_batch,
    configure_evaluation_multiprocessing,
    evaluate_checkpoint,
    load_statistics,
    load_checkpoint_policy,
    make_vector_environment_with_backoff,
    observation_to_proprio,
    quaternion_to_axis_angle,
    resize_flipped_rgb,
    resolve_checkpoint,
    rollout_action_chunks,
    load_trusted_libero_init_states,
    validate_simulation_dependencies,
    validate_settings,
)
from octo_small_libero.evaluate import build_parser


class _RecordingOffscreenEnvironment:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def close(self):
        return None


class _RecordingVectorEnvironment:
    def __init__(self, factories):
        self.environments = [factory() for factory in factories]

    def get_env_attr(self, key):
        return [getattr(environment, key) for environment in self.environments]

    def close(self):
        for environment in self.environments:
            environment.close()


class _FakeWorkerProcess:
    def __init__(self, pid, *, terminate_exits=True, kill_exits=True):
        self.pid = pid
        self.alive = True
        self.terminate_exits = terminate_exits
        self.kill_exits = kill_exits
        self.join_timeouts = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def terminate(self):
        self.terminate_calls += 1
        if self.terminate_exits:
            self.alive = False

    def kill(self):
        self.kill_calls += 1
        if self.kill_exits:
            self.alive = False


class _FakeWorkerRemote:
    def __init__(self, process, *, ready=False, send_error=None, clock=None):
        self.process = process
        self.ready = ready
        self.send_error = send_error
        self.clock = clock
        self.sent = []
        self.poll_timeouts = []
        self.recv_calls = 0
        self.close_calls = 0

    def send(self, value):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(value)

    def poll(self, timeout):
        self.poll_timeouts.append(timeout)
        if self.clock is not None and not self.ready:
            self.clock[0] += timeout
        return self.ready

    def recv(self):
        self.recv_calls += 1
        self.process.alive = False
        return None

    def close(self):
        self.close_calls += 1


def _fake_vector_environment(*remotes_and_processes):
    workers = [
        SimpleNamespace(process=process, parent_remote=remote, is_closed=False)
        for remote, process in remotes_and_processes
    ]
    return SimpleNamespace(workers=workers, is_closed=False)


def _ignore_close_and_sigterm(remote):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    remote.send("ready")
    try:
        while True:
            command, _ = remote.recv()
            if command == "close":
                while True:
                    time.sleep(0.1)
    except EOFError:
        return


def _write_self_contained_model(root):
    (root / "text_encoder").mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "model_config.json").write_text(
        json.dumps(
            {
                "action_dim": 7,
                "action_horizon": 8,
                "proprio_dim": 8,
                "language_tokens": 16,
            }
        ),
        encoding="utf-8",
    )
    for name in ("config.json", "tokenizer_config.json", "spiece.model", "tokenizer.json"):
        (root / "text_encoder" / name).write_bytes(b"content")


def _statistics(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps(
            {
                "action": {
                    "mean": [1, 2, 3, 4, 5, 6, 0],
                    "std": [2, 2, 2, 2, 2, 2, 1],
                },
                "observation.state": {
                    "mean": [0] * 8,
                    "std": [1] * 8,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_statistics(path)


def _git_commit_all(repository):
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Octo evaluation test",
            "-c",
            "user.email=octo-eval@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )


def test_checkpoint_resolver_supports_self_contained_and_weights_only(tmp_path):
    base = tmp_path / "base"
    _write_self_contained_model(base)
    direct = resolve_checkpoint(base)
    assert direct.self_contained
    assert direct.base_model_path == base
    assert direct.weights_path == base / "model.safetensors"

    weights = tmp_path / "fine_tuned.safetensors"
    weights.write_bytes(b"fine-tuned")
    overlay = resolve_checkpoint(weights, base_model=base)
    assert not overlay.self_contained
    assert overlay.base_model_path == base
    assert overlay.weights_path == weights

    with pytest.raises(EvaluationError, match="provide --base-model"):
        resolve_checkpoint(weights)


def test_checkpoint_rejects_training_pointer_semantics(tmp_path):
    with pytest.raises(EvaluationError, match="aliases"):
        resolve_checkpoint(tmp_path / "best")
    invalid = tmp_path / "latest"
    invalid.write_text("step-00010000", encoding="utf-8")
    with pytest.raises(EvaluationError, match="aliases"):
        resolve_checkpoint(invalid)
    invalid_file = tmp_path / "checkpoint.pt"
    invalid_file.write_bytes(b"weights")
    with pytest.raises(EvaluationError, match="safetensors"):
        resolve_checkpoint(invalid_file)


def test_weights_only_checkpoint_is_loaded_strictly(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from octo_small_libero.torch_model import OctoSmallPolicy

    base = tmp_path / "base"
    _write_self_contained_model(base)
    weights = tmp_path / "fine_tuned.safetensors"
    weights.write_bytes(b"fine-tuned")
    checkpoint = resolve_checkpoint(weights, base_model=base)
    statistics = _statistics(tmp_path)
    model = SimpleNamespace(
        config=SimpleNamespace(
            action_dim=7,
            action_horizon=8,
            proprio_dim=8,
            language_tokens=16,
        ),
        to=lambda device: model,
        eval=lambda: model,
    )
    monkeypatch.setattr(
        OctoSmallPolicy,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: (model, "tokenizer")),
    )
    calls = []

    def fake_load_model(target, filename, *, strict, device):
        calls.append((target, filename, strict, device))

    monkeypatch.setattr(safetensors_torch, "load_model", fake_load_model)
    loaded = load_checkpoint_policy(
        checkpoint,
        statistics=statistics,
        device="cpu",
        precision="fp32",
    )

    assert loaded.model is model
    assert calls == [(model, str(weights), True, "cpu")]
    assert loaded.device == torch.device("cpu")


def test_statistics_and_action_mapping(tmp_path):
    statistics = _statistics(tmp_path)
    normalized = np.zeros((2, 8, 7), dtype=np.float32)
    normalized[..., :6] = 0.5
    normalized[..., 6] = np.asarray([0.0, 1.0] * 4)[None, :]
    actions = statistics.actions_to_environment(normalized)
    np.testing.assert_allclose(
        actions[..., :6],
        np.broadcast_to([2, 3, 4, 5, 6, 7], (2, 8, 6)),
    )
    np.testing.assert_allclose(
        actions[:, :, 6],
        [[1, -1] * 4, [1, -1] * 4],
    )


def test_libero_observation_preprocessing_contract():
    identity = quaternion_to_axis_angle(np.asarray([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_array_equal(identity, np.zeros(3, dtype=np.float32))
    observation = {
        "robot0_eef_pos": np.asarray([1, 2, 3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.25, 0.75], dtype=np.float32),
    }
    np.testing.assert_allclose(
        observation_to_proprio(observation),
        [1, 2, 3, 0, 0, 0, 0, 0.25],
    )

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0] = 7
    image[-1] = 19
    resized = resize_flipped_rgb(image, (4, 4))
    np.testing.assert_array_equal(resized[0], 19)
    np.testing.assert_array_equal(resized[-1], 7)


def test_model_batch_scales_and_resizes_images_and_normalizes_proprio(tmp_path):
    torch = pytest.importorskip("torch")
    statistics = _statistics(tmp_path)
    observation = {
        "agentview_image": np.zeros((8, 6, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((5, 7, 3), 255, dtype=np.uint8),
        "robot0_eef_pos": np.asarray([1, 2, 3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.25, 0.75], dtype=np.float32),
    }

    def tokenizer(instructions, **kwargs):
        assert instructions == ["put away the book"]
        assert kwargs["max_length"] == 16
        return {
            "input_ids": torch.zeros((1, 16), dtype=torch.long),
            "attention_mask": torch.ones((1, 16), dtype=torch.long),
        }

    batch = build_model_batch(
        [observation],
        instruction="put away the book",
        tokenizer=tokenizer,
        statistics=statistics,
    )
    assert batch["image_primary"].shape == (1, 1, 3, 256, 256)
    assert batch["image_wrist"].shape == (1, 1, 3, 128, 128)
    assert batch["proprio"].shape == (1, 1, 8)
    torch.testing.assert_close(batch["image_primary"], torch.full_like(batch["image_primary"], -1))
    torch.testing.assert_close(batch["image_wrist"], torch.ones_like(batch["image_wrist"]))
    torch.testing.assert_close(
        batch["proprio"][0, 0],
        torch.tensor([1, 2, 3, 0, 0, 0, 0, 0.25], dtype=torch.float32),
    )


class _FakeVectorEnvironment:
    def __init__(self, success_steps):
        self.success_steps = np.asarray(success_steps)
        self.steps = 0

    def step(self, actions):
        self.steps += 1
        observations = np.asarray(
            [{"value": self.steps} for _ in range(len(actions))],
            dtype=object,
        )
        return (
            observations,
            np.zeros(len(actions)),
            np.zeros(len(actions), dtype=bool),
            np.asarray([{} for _ in range(len(actions))], dtype=object),
        )

    def check_success(self):
        return self.steps >= self.success_steps


def test_chunk_rollout_records_first_success_and_stops_when_all_succeed():
    environment = _FakeVectorEnvironment([2, 5])
    observations = np.asarray([{"value": 0}, {"value": 0}], dtype=object)

    def predictor(current):
        assert len(current) == 2
        return np.zeros((2, ACTION_HORIZON, 7), dtype=np.float32)

    episodes, _ = rollout_action_chunks(
        environment,
        observations,
        predictor=predictor,
        episode_ids=[11, 12],
        init_state_ids=[1, 2],
        seed=1,
        max_steps=960,
    )
    assert environment.steps == 5
    assert episodes == [
        {
            "episode_id": 11,
            "init_state_id": 1,
            "seed": 1,
            "success": True,
            "steps": 2,
            "first_success_step": 2,
            "termination": "success",
        },
        {
            "episode_id": 12,
            "init_state_id": 2,
            "seed": 1,
            "success": True,
            "steps": 5,
            "first_success_step": 5,
            "termination": "success",
        },
    ]


def test_evaluation_cli_defaults_and_checkpoint_arguments():
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--checkpoint",
            "/tmp/weights.safetensors",
            "--base-model",
            "/tmp/base",
            "--statistics",
            "/tmp/stats.json",
            "--device",
            "cpu",
            "--precision",
            "fp32",
        ]
    )
    assert arguments.checkpoint == "/tmp/weights.safetensors"
    assert arguments.base_model == "/tmp/base"
    assert arguments.episodes == 150
    assert arguments.num_envs == 50
    assert arguments.auto_reduce_num_envs is True
    assert arguments.max_steps == 960
    assert arguments.preflight_only is False


def test_evaluation_protocol_uses_three_fixed_seeds_and_balanced_episodes():
    settings = EvaluationSettings(episodes=150, num_envs=50)

    assert settings.seeds == (0, 1, 2)
    validate_settings(settings)

    with pytest.raises(EvaluationError, match="divisible by 3"):
        validate_settings(EvaluationSettings(episodes=149, num_envs=50))

    with pytest.raises(EvaluationError, match="fixed evaluation seeds"):
        validate_settings(
            EvaluationSettings(episodes=150, num_envs=50, seeds=(3, 4, 5))
        )


def test_evaluate_checkpoint_repeats_initial_states_for_each_fixed_seed(
    tmp_path,
    monkeypatch,
):
    import octo_small_libero.evaluation as evaluation

    checkpoint = CheckpointSpec(
        requested_path=tmp_path / "checkpoint",
        weights_path=tmp_path / "checkpoint" / "model.safetensors",
        base_model_path=tmp_path / "checkpoint",
        self_contained=True,
        model_config_path=tmp_path / "checkpoint" / "model_config.json",
        text_encoder_path=tmp_path / "checkpoint" / "text_encoder",
        weights_sha256="a" * 64,
    )
    statistics = NormalizationStatistics(
        path=tmp_path / "stats.json",
        action_mean=np.zeros(7),
        action_std=np.ones(7),
        proprio_mean=np.zeros(8),
        proprio_std=np.ones(8),
        sha256="b" * 64,
    )
    task = LiberoTask(
        task_id=5,
        name="book-caddy",
        language="put the book in the caddy",
        bddl_file=tmp_path / "task.bddl",
        init_states_file=tmp_path / "task.init",
        init_states_sha256="c" * 64,
        init_states_git_blob="d" * 40,
        init_states=np.zeros((2, 45), dtype=np.float64),
    )
    settings = EvaluationSettings(
        checkpoint=checkpoint.requested_path,
        statistics=statistics.path,
        output_dir=tmp_path / "results",
        episodes=6,
        num_envs=2,
        max_steps=1,
        settle_steps=0,
        device="cpu",
        precision="fp32",
    )

    class FakeEnvironment:
        def __init__(self):
            self.batch_size = 0
            self.seeds = []

        def reset(self):
            return None

        def seed(self, seed):
            self.seeds.append(seed)

        def set_init_state(self, init_states):
            self.batch_size = len(init_states)
            return np.asarray(
                [{"value": index} for index in range(self.batch_size)],
                dtype=object,
            )

        def step(self, actions):
            observations = np.asarray(
                [{"value": index} for index in range(self.batch_size)],
                dtype=object,
            )
            return (
                observations,
                np.zeros(self.batch_size),
                np.zeros(self.batch_size, dtype=bool),
                np.asarray([{} for _ in range(self.batch_size)], dtype=object),
            )

        def check_success(self):
            return np.ones(self.batch_size, dtype=bool)

        def close(self):
            return None

    class FakePolicy:
        def __init__(self):
            self.generator_seeds = []

        def make_generator(self, seed):
            self.generator_seeds.append(seed)
            return seed

        def predict_action_chunk(self, observations, language, *, generator):
            assert language == task.language
            assert generator in EVALUATION_SEEDS
            return np.zeros(
                (len(observations), ACTION_HORIZON, 7),
                dtype=np.float32,
            )

    environment = FakeEnvironment()
    policy = FakePolicy()
    monkeypatch.setattr(evaluation, "validate_simulation_dependencies", lambda: {})
    monkeypatch.setattr(evaluation, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(evaluation, "load_statistics", lambda path: statistics)
    monkeypatch.setattr(
        evaluation,
        "configure_libero",
        lambda **kwargs: (tmp_path, tmp_path / "config.yaml", LIBERO_COMMIT),
    )
    monkeypatch.setattr(evaluation, "resolve_libero_task", lambda *args, **kwargs: task)

    def load_policy(*args, **kwargs):
        assert multiprocessing.get_start_method() == "spawn"
        return policy

    monkeypatch.setattr(evaluation, "load_checkpoint_policy", load_policy)
    monkeypatch.setattr(
        evaluation,
        "make_vector_environment_with_backoff",
        lambda *args, **kwargs: (environment, settings.num_envs),
    )

    report = evaluate_checkpoint(settings)
    episode_rows = [
        json.loads(line)
        for line in (settings.output_dir / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert policy.generator_seeds == [0, 1, 2]
    assert environment.seeds == [0, 1, 2]
    assert [
        (row["episode_id"], row["init_state_id"], row["seed"])
        for row in episode_rows
    ] == [
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 1),
        (4, 0, 2),
        (5, 1, 2),
    ]
    assert report["summary"]["completed_episodes"] == 6
    assert [summary["success_rate"] for summary in report["summary"]["by_seed"]] == [
        1.0,
        1.0,
        1.0,
    ]


def test_libero_source_checkout_layout_is_added_to_import_path(tmp_path, monkeypatch):
    repository = tmp_path / "LIBERO"
    package = repository / "libero" / "libero"
    for name in ("bddl_files", "init_files", "assets"):
        (package / name).mkdir(parents=True)
    monkeypatch.setattr("sys.path", list(__import__("sys").path))

    assert _find_libero_package_root(repository) == package
    assert str(repository) == __import__("sys").path[0]


def test_verified_libero_init_states_use_explicit_pickle_loading(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    import octo_small_libero.evaluation as evaluation

    repository = tmp_path / "LIBERO"
    package = repository / "libero" / "libero"
    for name in ("bddl_files", "init_files", "assets"):
        (package / name).mkdir(parents=True)
    init_file = package / "init_files" / "libero_10" / "task.init"
    init_file.parent.mkdir(parents=True)
    expected_states = np.arange(90, dtype=np.float64).reshape(2, 45)
    torch.save(expected_states, init_file)
    _git_commit_all(repository)
    monkeypatch.setattr(
        evaluation,
        "_verify_libero_commit",
        lambda root: LIBERO_COMMIT,
    )
    original_load = torch.load
    calls = []

    def recording_load(source, **kwargs):
        calls.append((source.getvalue(), kwargs))
        return original_load(source, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    states, sha256, git_blob = load_trusted_libero_init_states(package, init_file)

    np.testing.assert_array_equal(states, expected_states)
    assert len(sha256) == 64
    assert len(git_blob) == 40
    assert calls == [
        (
            init_file.read_bytes(),
            {
                "map_location": "cpu",
                "weights_only": False,
            },
        )
    ]

    init_file.write_bytes(b"modified")
    with pytest.raises(EvaluationError, match="differs from the pinned Git commit"):
        load_trusted_libero_init_states(package, init_file)

    untracked = package / "init_files" / "libero_10" / "untracked.init"
    torch.save(expected_states, untracked)
    with pytest.raises(EvaluationError, match="not tracked"):
        load_trusted_libero_init_states(package, untracked)

    outside = tmp_path / "outside.init"
    torch.save(expected_states, outside)
    with pytest.raises(EvaluationError, match="must be inside"):
        load_trusted_libero_init_states(package, outside)


def test_multiprocessing_configuration_forces_spawn_in_fresh_process():
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    program = """
import multiprocessing
from octo_small_libero.evaluation import configure_evaluation_multiprocessing
assert multiprocessing.get_start_method(allow_none=True) is None
multiprocessing.set_start_method("fork", force=True)
assert configure_evaluation_multiprocessing() == "spawn"
assert multiprocessing.get_start_method() == "spawn"
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=source_root.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


def test_worker_shutdown_timeouts_are_group_wide_and_idempotent(monkeypatch):
    import octo_small_libero.evaluation as evaluation

    assert LIBERO_WORKER_GRACEFUL_CLOSE_TIMEOUT_SECONDS == 10.0
    assert LIBERO_WORKER_TERMINATE_TIMEOUT_SECONDS == 5.0
    assert LIBERO_WORKER_KILL_TIMEOUT_SECONDS == 5.0
    clock = [100.0]
    monkeypatch.setattr(evaluation.time, "monotonic", lambda: clock[0])
    processes = [_FakeWorkerProcess(100 + index) for index in range(3)]
    remotes = [_FakeWorkerRemote(process, ready=False, clock=clock) for process in processes]
    environment = _fake_vector_environment(*zip(remotes, processes, strict=True))

    _close_libero_subprocess_vector_environment(environment)
    _close_libero_subprocess_vector_environment(environment)

    assert [remote.poll_timeouts for remote in remotes] == [[10.0], [0.0], [0.0]]
    assert all(remote.sent == [["close", None]] for remote in remotes)
    assert all(remote.close_calls == 1 for remote in remotes)
    assert all(process.terminate_calls == 1 for process in processes)
    assert all(process.kill_calls == 0 for process in processes)
    assert environment.is_closed is True
    assert all(worker.is_closed is True for worker in environment.workers)


def test_worker_shutdown_accepts_graceful_close_without_signals():
    processes = [_FakeWorkerProcess(201), _FakeWorkerProcess(202)]
    remotes = [_FakeWorkerRemote(process, ready=True) for process in processes]
    environment = _fake_vector_environment(*zip(remotes, processes, strict=True))

    _close_libero_subprocess_vector_environment(environment)

    assert all(remote.recv_calls == 1 for remote in remotes)
    assert all(process.terminate_calls == 0 for process in processes)
    assert all(process.kill_calls == 0 for process in processes)


@pytest.mark.parametrize("send_error", [BrokenPipeError(), EOFError()])
def test_worker_shutdown_handles_pipe_failure_and_escalates_to_kill(
    monkeypatch,
    send_error,
):
    import octo_small_libero.evaluation as evaluation

    monkeypatch.setattr(evaluation, "LIBERO_WORKER_GRACEFUL_CLOSE_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(evaluation, "LIBERO_WORKER_TERMINATE_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(evaluation, "LIBERO_WORKER_KILL_TIMEOUT_SECONDS", 0.0)
    process = _FakeWorkerProcess(301, terminate_exits=False, kill_exits=True)
    remote = _FakeWorkerRemote(process, send_error=send_error)
    environment = _fake_vector_environment((remote, process))

    _close_libero_subprocess_vector_environment(environment)

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert remote.close_calls == 1
    assert process.alive is False


def test_worker_shutdown_reports_workers_that_survive_kill(monkeypatch):
    import octo_small_libero.evaluation as evaluation

    monkeypatch.setattr(evaluation, "LIBERO_WORKER_GRACEFUL_CLOSE_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(evaluation, "LIBERO_WORKER_TERMINATE_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(evaluation, "LIBERO_WORKER_KILL_TIMEOUT_SECONDS", 0.0)
    process = _FakeWorkerProcess(401, terminate_exits=False, kill_exits=False)
    remote = _FakeWorkerRemote(process, ready=False)
    environment = _fake_vector_environment((remote, process))

    with pytest.raises(UnrecoverableSimulationShutdownError) as raised:
        _close_libero_subprocess_vector_environment(environment)

    assert raised.value.worker_pids == (401,)
    assert "worker_pids=[401]" in str(raised.value)
    assert remote.close_calls == 1


@pytest.mark.timeout(5)
def test_real_spawn_worker_that_ignores_close_and_sigterm_is_killed(monkeypatch):
    import octo_small_libero.evaluation as evaluation

    configure_evaluation_multiprocessing()
    context = multiprocessing.get_context("spawn")
    parent_remote, child_remote = context.Pipe()
    process = context.Process(
        target=_ignore_close_and_sigterm,
        args=(child_remote,),
        daemon=True,
    )
    process.start()
    child_remote.close()
    assert parent_remote.poll(2.0)
    assert parent_remote.recv() == "ready"
    monkeypatch.setattr(evaluation, "LIBERO_WORKER_GRACEFUL_CLOSE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(evaluation, "LIBERO_WORKER_TERMINATE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(evaluation, "LIBERO_WORKER_KILL_TIMEOUT_SECONDS", 1.0)
    environment = _fake_vector_environment((parent_remote, process))

    _close_libero_subprocess_vector_environment(environment)

    assert process.is_alive() is False
    assert process not in multiprocessing.active_children()


def test_vector_environment_uses_official_libero_robosuite_stack(
    tmp_path,
    monkeypatch,
):
    import octo_small_libero.evaluation as evaluation

    _RecordingOffscreenEnvironment.calls = []
    monkeypatch.setattr(
        evaluation,
        "_libero_environment_classes",
        lambda: (_RecordingOffscreenEnvironment, _RecordingVectorEnvironment),
    )
    task = LiberoTask(
        task_id=5,
        name=LIBERO_10_TASK_NAMES[5],
        language="put the book in the caddy",
        bddl_file=tmp_path / "libero" / "bddl_files" / "libero_10" / "task.bddl",
        init_states_file=tmp_path / "task.init",
        init_states_sha256="a" * 64,
        init_states_git_blob="b" * 40,
        init_states=np.zeros((2, 45), dtype=np.float64),
    )

    environment = evaluation.make_vector_environment(task, 2)
    try:
        assert _RecordingOffscreenEnvironment.calls == [
            {
                "bddl_file_name": str(task.bddl_file),
                "camera_heights": 128,
                "camera_widths": 128,
            },
            {
                "bddl_file_name": str(task.bddl_file),
                "camera_heights": 128,
                "camera_widths": 128,
            },
        ]
        assert all(
            item._octo_startup_failure is None
            for item in environment.environments
        )
    finally:
        environment.close()


def test_libero_gym_import_is_backed_by_gymnasium():
    gymnasium = SimpleNamespace(__version__=GYMNASIUM_VERSION)
    modules = {}

    assert _install_libero_gymnasium_compatibility(
        gymnasium_module=gymnasium,
        module_registry=modules,
    )
    assert modules == {"gym": gymnasium}
    assert not _install_libero_gymnasium_compatibility(
        gymnasium_module=gymnasium,
        module_registry=modules,
    )


def test_libero_gymnasium_alias_rejects_preimported_legacy_gym():
    gymnasium = SimpleNamespace(__version__=GYMNASIUM_VERSION)
    modules = {"gym": object()}

    with pytest.raises(EvaluationError, match="Legacy gym was imported"):
        _install_libero_gymnasium_compatibility(
            gymnasium_module=gymnasium,
            module_registry=modules,
        )


def test_libero_subprocess_worker_bootstrap_is_idempotent():
    original_worker = object()
    original_wrapper = object()
    libero_venv = SimpleNamespace(
        _worker=original_worker,
        CloudpickleWrapper=original_wrapper,
    )

    assert _install_libero_subprocess_worker_bootstrap(libero_venv)
    assert libero_venv._worker is _run_libero_subprocess_worker_with_gymnasium
    assert libero_venv.CloudpickleWrapper is _LiberoWorkerCloudpickleWrapper
    assert libero_venv._octo_small_libero_original_worker is original_worker
    assert (
        libero_venv._octo_small_libero_original_cloudpickle_wrapper
        is original_wrapper
    )
    assert not _install_libero_subprocess_worker_bootstrap(libero_venv)


@pytest.mark.timeout(10)
def test_spawn_worker_installs_gymnasium_alias_before_importing_libero(
    tmp_path,
    monkeypatch,
):
    package_files = {
        "cloudpickle.py": "from pickle import dumps, loads\n",
        "gymnasium/__init__.py": '__version__ = "1.3.0"\n',
        "libero/__init__.py": "",
        "libero/libero/__init__.py": "",
        "libero/libero/envs/__init__.py": "",
        "libero/libero/envs/venv.py": """
import gym
import gymnasium

def _worker(parent, pipe, env_fn_wrapper, observation_buffers=None):
    parent.close()
    pipe.send({
        "same_module": gym is gymnasium,
        "version": gym.__version__,
        "factory": env_fn_wrapper.data,
        "buffers": observation_buffers,
    })
    pipe.close()
""",
    }
    for relative_path, content in package_files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "cloudpickle",
        SimpleNamespace(dumps=pickle.dumps, loads=pickle.loads),
    )

    context = multiprocessing.get_context("spawn")
    parent_remote, child_remote = context.Pipe()
    process = context.Process(
        target=_run_libero_subprocess_worker_with_gymnasium,
        args=(
            parent_remote,
            child_remote,
            _LiberoWorkerCloudpickleWrapper("factory-payload"),
            "buffer-payload",
        ),
        daemon=True,
    )
    process.start()
    child_remote.close()
    try:
        assert parent_remote.poll(5.0)
        assert parent_remote.recv() == {
            "same_module": True,
            "version": GYMNASIUM_VERSION,
            "factory": "factory-payload",
            "buffers": "buffer-payload",
        }
        process.join(5.0)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join(1.0)
        parent_remote.close()


def test_mujoco_310_compatibility_accepts_native_and_robosuite_full_m_calls():
    calls = []

    class NativeData:
        pass

    class WrappedData:
        qM = property(lambda self: "legacy-matrix")

        def __init__(self):
            self._data = NativeData()

    def native_full_m(model, data, destination):
        calls.append((model, data, destination))

    mujoco = SimpleNamespace(
        __version__=MUJOCO_VERSION,
        MjData=NativeData,
        mj_fullM=native_full_m,
    )

    assert _install_robosuite_mujoco_310_compatibility(
        mujoco_module=mujoco,
        mjdata_wrapper=WrappedData,
    )
    wrapped = WrappedData()
    destination = np.empty((2, 2), dtype=np.float64)
    mujoco.mj_fullM("native-model", wrapped._data, destination)
    mujoco.mj_fullM("legacy-model", destination, wrapped.qM)

    assert calls == [
        ("native-model", wrapped._data, destination),
        ("legacy-model", wrapped._data, destination),
    ]
    assert not _install_robosuite_mujoco_310_compatibility(
        mujoco_module=mujoco,
        mjdata_wrapper=WrappedData,
    )


def test_vector_startup_failure_preserves_root_cause_and_hints():
    failure = SimulationStartupFailure(
        pid=4242,
        exception_type="ImportError",
        message="Cannot initialize a EGL device display",
        mujoco_gl="egl",
        cuda_visible_devices="0,1",
        mujoco_egl_device_id="1",
    )
    environment = SimpleNamespace(
        get_env_attr=lambda key: [None, failure],
    )

    with pytest.raises(SimulationInfrastructureError) as raised:
        _validate_vector_environment_startup(environment, num_envs=2)

    message = str(raised.value)
    assert "worker" in message
    assert "1(pid=4242)" in message
    assert "ImportError: Cannot initialize a EGL device display" in message
    assert "MUJOCO_GL='egl'" in message
    assert "CUDA_VISIBLE_DEVICES='0,1'" in message
    assert "MUJOCO_EGL_DEVICE_ID='1'" in message
    assert "--num-envs" in message


def test_vector_environment_startup_automatically_reduces_parallelism(
    monkeypatch,
    capsys,
):
    import octo_small_libero.evaluation as evaluation

    attempts = []
    expected_environment = object()

    def make_environment(task, num_envs):
        del task
        attempts.append(num_envs)
        if num_envs > 12:
            raise SimulationInfrastructureError("framebuffer unavailable")
        return expected_environment

    monkeypatch.setattr(evaluation, "make_vector_environment", make_environment)
    environment, effective_num_envs = make_vector_environment_with_backoff(
        object(),
        50,
        auto_reduce=True,
    )

    assert environment is expected_environment
    assert effective_num_envs == 12
    assert attempts == [50, 25, 12]
    warning = capsys.readouterr().err
    assert "num_envs=50" in warning
    assert "num_envs=25" in warning
    assert "num_envs=12" in warning


def test_vector_environment_startup_can_disable_parallelism_backoff(monkeypatch):
    import octo_small_libero.evaluation as evaluation

    attempts = []

    def make_environment(task, num_envs):
        del task
        attempts.append(num_envs)
        raise SimulationInfrastructureError("EGL startup failed")

    monkeypatch.setattr(evaluation, "make_vector_environment", make_environment)
    with pytest.raises(SimulationInfrastructureError, match="EGL startup failed"):
        make_vector_environment_with_backoff(object(), 50, auto_reduce=False)

    assert attempts == [50]


def test_simulation_infrastructure_error_uses_exit_code_three(monkeypatch, capsys):
    import octo_small_libero.evaluate as evaluate

    def fail_evaluation(*args, **kwargs):
        assert multiprocessing.get_start_method() == "spawn"
        raise SimulationInfrastructureError("EGL startup failed")

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", fail_evaluation)
    monkeypatch.setattr(sys, "argv", ["octo-small-libero-evaluate"])
    with pytest.raises(SystemExit) as raised:
        evaluate.main()

    assert raised.value.code == 3
    assert "simulation infrastructure error: EGL startup failed" in capsys.readouterr().err


def test_unrecoverable_shutdown_uses_hard_exit_code_three():
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    program = """
import octo_small_libero.evaluate as evaluate
from octo_small_libero.evaluation import UnrecoverableSimulationShutdownError
def fail(*args, **kwargs):
    raise UnrecoverableSimulationShutdownError(
        "worker shutdown failed (worker_pids=[4242])",
        worker_pids=[4242],
    )
evaluate.evaluate_checkpoint = fail
evaluate.main()
"""

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=source_root.parent,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 3
    assert "worker_pids=[4242]" in completed.stderr


@pytest.mark.parametrize(
    "failure",
    [
        SimulationInfrastructureError("EGL startup failed"),
        UnrecoverableSimulationShutdownError(
            "workers remained alive (worker_pids=[4242])",
            worker_pids=[4242],
        ),
    ],
)
def test_vector_startup_failure_is_recorded_in_failure_report(
    monkeypatch,
    tmp_path,
    failure,
):
    import octo_small_libero.evaluation as evaluation

    checkpoint = CheckpointSpec(
        requested_path=tmp_path / "checkpoint",
        weights_path=tmp_path / "checkpoint" / "model.safetensors",
        base_model_path=tmp_path / "checkpoint",
        self_contained=True,
        model_config_path=tmp_path / "checkpoint" / "model_config.json",
        text_encoder_path=tmp_path / "checkpoint" / "text_encoder",
        weights_sha256="a" * 64,
    )
    statistics = NormalizationStatistics(
        path=tmp_path / "stats.json",
        action_mean=np.zeros(7),
        action_std=np.ones(7),
        proprio_mean=np.zeros(8),
        proprio_std=np.ones(8),
        sha256="b" * 64,
    )
    task = LiberoTask(
        task_id=5,
        name="book-caddy",
        language="put the book in the caddy",
        bddl_file=tmp_path / "task.bddl",
        init_states_file=tmp_path / "task.init",
        init_states_sha256="c" * 64,
        init_states_git_blob="d" * 40,
        init_states=np.zeros((1, 45), dtype=np.float64),
    )
    settings = EvaluationSettings(
        checkpoint=checkpoint.requested_path,
        statistics=statistics.path,
        output_dir=tmp_path / "results",
        episodes=3,
        num_envs=1,
        device="cpu",
        precision="fp32",
    )

    monkeypatch.setattr(evaluation, "validate_simulation_dependencies", lambda: {})
    monkeypatch.setattr(evaluation, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(evaluation, "load_statistics", lambda path: statistics)
    monkeypatch.setattr(
        evaluation,
        "configure_libero",
        lambda **kwargs: (tmp_path, tmp_path / "config.yaml", LIBERO_COMMIT),
    )
    monkeypatch.setattr(evaluation, "resolve_libero_task", lambda *args, **kwargs: task)
    monkeypatch.setattr(
        evaluation,
        "load_checkpoint_policy",
        lambda *args, **kwargs: SimpleNamespace(make_generator=lambda seed: object()),
    )

    def fail_startup(*args, **kwargs):
        raise failure

    monkeypatch.setattr(evaluation, "make_vector_environment", fail_startup)
    with pytest.raises(type(failure)) as raised:
        evaluate_checkpoint(settings)
    assert str(raised.value) == str(failure)

    report = json.loads((settings.output_dir / "failure.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error"] == f"{type(failure).__name__}: {failure}"
    assert (
        report["runtime"]["libero_multiprocessing_start_method"]
        == LIBERO_MULTIPROCESSING_START_METHOD
    )


def test_simulation_dependencies_keep_core_strict_and_allow_compatible_helpers(
    monkeypatch,
):
    import octo_small_libero.evaluation as evaluation

    versions = {
        "mujoco": MUJOCO_VERSION,
        "robosuite": ROBOSUITE_VERSION,
        "bddl": BDDL_VERSION,
        "gymnasium": GYMNASIUM_VERSION,
        "cloudpickle": "3.1.2",
        "easydict": "1.13",
        "future": "1.0.0",
        "matplotlib": "3.10.8",
        "termcolor": "3.3.0",
    }
    monkeypatch.setattr(
        evaluation.importlib.metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    assert validate_simulation_dependencies() == versions

    versions["robosuite"] = "1.5.0"
    with pytest.raises(EvaluationError, match="robosuite==1.4.0"):
        validate_simulation_dependencies()

    versions["robosuite"] = ROBOSUITE_VERSION
    versions["cloudpickle"] = "1.6.0"
    with pytest.raises(
        EvaluationError,
        match=r"cloudpickle>=2\.1\.0,<4",
    ):
        validate_simulation_dependencies()


def test_missing_mujoco_dependency_has_install_hint(monkeypatch):
    import octo_small_libero.evaluation as evaluation

    monkeypatch.setattr(
        evaluation.importlib.metadata,
        "version",
        lambda distribution: (_ for _ in ()).throw(
            evaluation.importlib.metadata.PackageNotFoundError(distribution)
        ),
    )
    with pytest.raises(
        EvaluationError,
        match=rf"Missing evaluation dependency mujoco=={MUJOCO_VERSION}",
    ):
        validate_simulation_dependencies()


def test_result_schema_contains_protocol_outcome_and_runtime(tmp_path):
    checkpoint = CheckpointSpec(
        requested_path=tmp_path / "checkpoint",
        weights_path=tmp_path / "checkpoint" / "model.safetensors",
        base_model_path=tmp_path / "checkpoint",
        self_contained=True,
        model_config_path=tmp_path / "checkpoint" / "model_config.json",
        text_encoder_path=tmp_path / "checkpoint" / "text_encoder",
        weights_sha256="a" * 64,
    )
    statistics = NormalizationStatistics(
        path=tmp_path / "stats.json",
        action_mean=np.zeros(7),
        action_std=np.ones(7),
        proprio_mean=np.zeros(8),
        proprio_std=np.ones(8),
        sha256="b" * 64,
    )
    settings = EvaluationSettings(
        checkpoint=checkpoint.requested_path,
        statistics=statistics.path,
        output_dir=tmp_path / "results",
        episodes=3,
        num_envs=1,
        device="cpu",
        precision="fp32",
    )
    task = LiberoTask(
        task_id=5,
        name="book-caddy",
        language="put the book in the caddy",
        bddl_file=tmp_path / "task.bddl",
        init_states_file=tmp_path / "task.init",
        init_states_sha256="c" * 64,
        init_states_git_blob="d" * 40,
        init_states=[object()],
    )
    report = _make_report(
        status="complete",
        settings=settings,
        checkpoint=checkpoint,
        statistics=statistics,
        task=task,
        episodes=[
            {
                "episode_id": 0,
                "init_state_id": 0,
                "seed": 0,
                "success": False,
                "steps": 8,
                "first_success_step": None,
                "termination": "max_steps",
            },
            {
                "episode_id": 1,
                "init_state_id": 0,
                "seed": 1,
                "success": True,
                "steps": 4,
                "first_success_step": 4,
                "termination": "success",
            },
            {
                "episode_id": 2,
                "init_state_id": 0,
                "seed": 2,
                "success": False,
                "steps": 8,
                "first_success_step": None,
                "termination": "max_steps",
            },
        ],
        elapsed_seconds=1.25,
        videos=[],
        libero_commit=LIBERO_COMMIT,
    )

    assert report["schema_version"] == 3
    assert report["protocol"]["episodes"] == 3
    assert report["protocol"]["episodes_per_seed"] == 1
    assert report["protocol"]["seeds"] == list(EVALUATION_SEEDS)
    assert report["protocol"]["action_horizon"] == 8
    assert report["summary"]["success_rate"] == pytest.approx(1 / 3)
    assert report["summary"]["by_seed"] == [
        {
            "seed": 0,
            "completed_episodes": 1,
            "successes": 0,
            "failures": 1,
            "success_rate": 0.0,
        },
        {
            "seed": 1,
            "completed_episodes": 1,
            "successes": 1,
            "failures": 0,
            "success_rate": 1.0,
        },
        {
            "seed": 2,
            "completed_episodes": 1,
            "successes": 0,
            "failures": 1,
            "success_rate": 0.0,
        },
    ]
    assert report["runtime"]["libero_commit"] == LIBERO_COMMIT
    assert report["runtime"]["mujoco_expected_version"] == MUJOCO_VERSION
    assert report["runtime"]["simulation_compatibility"] == MUJOCO_COMPATIBILITY
    assert report["runtime"]["effective_num_envs"] is None
    assert report["runtime"]["environment_batch_sizes"] == []
    assert "mujoco" in report["runtime"]["packages"]
    assert (
        report["runtime"]["libero_multiprocessing_start_method"]
        == LIBERO_MULTIPROCESSING_START_METHOD
    )
    assert report["task"]["init_states_sha256"] == "c" * 64
    assert report["task"]["init_states_git_blob"] == "d" * 40
    json.dumps(report)
