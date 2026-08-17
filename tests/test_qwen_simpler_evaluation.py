import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _evaluation():
    return importlib.import_module("qwen3_vl_groot.simpler_evaluation")


def test_qwen_module_reexports_shared_source_commit_constants():
    evaluation = _evaluation()
    from simpler_bridge import evaluation as shared_evaluation

    assert evaluation.SIMPLER_ENV_COMMIT == shared_evaluation.SIMPLER_ENV_COMMIT
    assert (
        evaluation.MANISKILL2_REAL2SIM_COMMIT
        == shared_evaluation.MANISKILL2_REAL2SIM_COMMIT
    )


def test_task_selection_expands_the_official_widowx_protocol():
    evaluation = _evaluation()

    tasks = evaluation.resolve_task_selection("all")

    assert [task.key for task in tasks] == ["spoon", "carrot", "stack", "eggplant"]
    assert [task.env_name for task in tasks] == [
        "PutSpoonOnTableClothInScene-v0",
        "PutCarrotOnPlateInScene-v0",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "PutEggplantInBasketScene-v0",
    ]
    assert [task.instruction for task in tasks] == [
        "Put Spoon on Towel",
        "Put Carrot on Plate",
        "Stack Green Block on Yellow Block",
        "Put Eggplant in Yellow Basket",
    ]
    assert [task.max_steps for task in tasks] == [60, 60, 60, 120]
    assert tasks[-1].robot == "widowx_sink_camera_setup"
    assert tasks[-1].scene_name == "bridge_table_1_v2"
    assert tasks[0].robot_init_xy == (0.147, 0.028)
    assert tasks[-1].robot_init_xy == (0.127, 0.06)


def test_task_selection_preserves_requested_order_and_rejects_duplicates():
    evaluation = _evaluation()

    assert [task.key for task in evaluation.resolve_task_selection("eggplant,spoon")] == [
        "eggplant",
        "spoon",
    ]
    with pytest.raises(evaluation.SimplerEvaluationError, match="duplicate"):
        evaluation.resolve_task_selection("spoon,spoon")
    with pytest.raises(evaluation.SimplerEvaluationError, match="unknown"):
        evaluation.resolve_task_selection("spoon,drawer")


def test_qwen_simpler_cli_only_accepts_stepwise_action_horizon():
    from qwen3_vl_groot.evaluate_simpler import build_parser

    parser = build_parser()
    arguments = parser.parse_args(["--checkpoint", "/models/checkpoint"])

    assert arguments.action_horizon == 1
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--checkpoint", "/models/checkpoint", "--action-horizon", "8"]
        )


def test_bridge_image_preprocessing_resizes_then_center_crops():
    evaluation = _evaluation()
    image = np.empty((4, 6, 3), dtype=np.uint8)
    image[:, :2] = 10
    image[:, 2:4] = 100
    image[:, 4:] = 240

    processed = evaluation.preprocess_bridge_image(
        image,
        resize_size=6,
        crop_size=4,
        output_size=4,
    )

    assert processed.shape == (4, 4, 3)
    assert processed.dtype == np.uint8
    assert processed[:, 0].mean() < processed[:, -1].mean()
    with pytest.raises(evaluation.SimplerEvaluationError, match="HWC RGB"):
        evaluation.preprocess_bridge_image(np.zeros((4, 4)), 6, 4, 4)


class _Pose:
    def __init__(self, position, quaternion):
        self.p = np.asarray(position, dtype=np.float64)
        self.q = np.asarray(quaternion, dtype=np.float64)

    def inv(self):
        w, x, y, z = self.q
        inverse_quaternion = np.asarray([w, -x, -y, -z], dtype=np.float64)
        inverse_position = -_rotate(inverse_quaternion, self.p)
        return _Pose(inverse_position, inverse_quaternion)

    def transform(self, other):
        return _Pose(
            self.p + _rotate(self.q, other.p),
            _multiply_quaternion(self.q, other.q),
        )

    def __mul__(self, other):
        return self.transform(other)


def _multiply_quaternion(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _rotate(quaternion, vector):
    pure = np.asarray([0.0, *vector], dtype=np.float64)
    conjugate = np.asarray(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
    )
    return _multiply_quaternion(_multiply_quaternion(quaternion, pure), conjugate)[1:]


def test_environment_to_bridge_proprio_uses_tcp_pose_in_robot_base_frame():
    evaluation = _evaluation()
    half_turn_z = np.asarray([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    agent = SimpleNamespace(
        robot=SimpleNamespace(pose=_Pose([1.0, 2.0, 0.0], half_turn_z)),
        get_gripper_closedness=lambda: 0.25,
    )
    environment = SimpleNamespace(
        agent=agent,
        tcp=SimpleNamespace(pose=_Pose([1.0, 3.0, 0.5], half_turn_z)),
    )

    proprio = evaluation.environment_to_bridge_proprio(environment)

    np.testing.assert_allclose(proprio, [1.0, 0.0, 0.5, 0, 0, 0, 0, 0.75], atol=1e-6)


def test_environment_to_bridge_proprio_uses_sapien_pose_multiplication():
    evaluation = _evaluation()

    class MultiplicationOnlyPose(_Pose):
        transform = None

        def inv(self):
            pose = super().inv()
            return MultiplicationOnlyPose(pose.p, pose.q)

        def __mul__(self, other):
            pose = _Pose.transform(self, other)
            return MultiplicationOnlyPose(pose.p, pose.q)

    environment = SimpleNamespace(
        agent=SimpleNamespace(
            robot=SimpleNamespace(
                pose=MultiplicationOnlyPose([0, 0, 0], [1, 0, 0, 0])
            ),
            get_gripper_closedness=lambda: 0.0,
        ),
        tcp=SimpleNamespace(
            pose=MultiplicationOnlyPose([0.1, 0.2, 0.3], [1, 0, 0, 0])
        ),
    )

    np.testing.assert_allclose(
        evaluation.environment_to_bridge_proprio(environment),
        [0.1, 0.2, 0.3, 0, 0, 0, 0, 1],
        atol=1e-6,
    )


def test_bridge_action_conversion_uses_axis_angle_and_binary_absolute_gripper():
    evaluation = _evaluation()
    actions = np.asarray(
        [
            [0.1, -0.2, 0.3, math.pi / 2, 0, 0, 0.51],
            [0, 0, 0, 0, 0, -math.pi / 2, 0.5],
        ],
        dtype=np.float32,
    )

    converted = evaluation.bridge_actions_to_simpler(actions)

    np.testing.assert_allclose(converted[0, :6], actions[0, :6], atol=1e-6)
    np.testing.assert_allclose(converted[1, :6], actions[1, :6], atol=1e-6)
    np.testing.assert_array_equal(converted[:, 6], [1.0, -1.0])
    with pytest.raises(evaluation.SimplerEvaluationError, match="last dimension 7"):
        evaluation.bridge_actions_to_simpler(np.zeros((8, 6), dtype=np.float32))


class _FakePolicy:
    def __init__(self):
        self.calls = []

    def predict_actions(self, image, state, instruction, denoising_steps, *, generator):
        self.calls.append(
            {
                "image_shape": np.asarray(image).shape,
                "state": np.asarray(state).copy(),
                "instruction": instruction,
                "denoising_steps": denoising_steps,
                "generator": generator,
            }
        )
        actions = np.zeros((1, 8, 7), dtype=np.float32)
        actions[..., 0] = np.arange(8, dtype=np.float32)
        actions[..., 6] = 1.0
        return actions


class _FakeEnvironment:
    def __init__(self, *, success_step=None, truncate_step=None, episode_stats=None):
        self.agent = SimpleNamespace(
            robot=SimpleNamespace(pose=_Pose([0, 0, 0], [1, 0, 0, 0])),
            get_gripper_closedness=lambda: 0.0,
        )
        self.tcp = SimpleNamespace(pose=_Pose([0.1, 0.2, 0.3], [1, 0, 0, 0]))
        self.success_step = success_step
        self.truncate_step = truncate_step
        self.step_count = 0
        self.actions = []
        self.reset_options = None
        self.closed = False
        self.episode_stats = episode_stats

    def _observation(self):
        return {
            "image": {
                "3rd_view_camera": {
                    "rgb": np.full((4, 6, 3), self.step_count, dtype=np.uint8)
                }
            }
        }

    def reset(self, *, options):
        self.reset_options = options
        self.step_count = 0
        return self._observation(), {"reset": True}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.step_count += 1
        success = self.success_step == self.step_count
        truncated = self.truncate_step == self.step_count
        return (
            self._observation(),
            float(success),
            success,
            truncated,
            {
                "episode_stats": self.episode_stats
                if self.episode_stats is not None
                else {"is_src_obj_grasped": self.step_count >= 2}
            },
        )

    def close(self):
        self.close_count = getattr(self, "close_count", 0) + 1
        self.closed = True


class _CloseFailingEnvironment(_FakeEnvironment):
    def close(self):
        super().close()
        raise RuntimeError("renderer teardown failed")


def test_run_episode_replans_each_step_until_early_success():
    evaluation = _evaluation()
    environment = _FakeEnvironment(success_step=3)
    policy = _FakePolicy()

    episode, frames = evaluation.run_simpler_episode(
        task=evaluation.SIMPLER_TASKS[0],
        object_episode_id=7,
        policy_seed=2,
        policy=policy,
        environment=environment,
        generator="seeded-generator",
        data_config={"train_crop_size": 4, "output_image_size": 4},
        denoising_steps=5,
        action_horizon=1,
        max_steps=60,
        capture_video=True,
    )

    assert episode == {
        "task": "spoon",
        "instruction": "Put Spoon on Towel",
        "seed": 2,
        "policy_seed": 2,
        "object_episode_id": 7,
        "success": True,
        "steps": 3,
        "termination": "success",
        "episode_stats": {"is_src_obj_grasped": True},
    }
    assert environment.reset_options["obj_init_options"] == {"episode_id": 7}
    assert [action[0] for action in environment.actions] == [0, 0, 0]
    assert all(action[6] == 1.0 for action in environment.actions)
    assert len(policy.calls) == 3
    assert policy.calls[0]["instruction"] == "Put Spoon on Towel"
    assert policy.calls[0]["denoising_steps"] == 5
    assert policy.calls[0]["generator"] == "seeded-generator"
    assert len(frames) == 4


def test_run_episode_replans_every_step_and_stops_on_truncation():
    evaluation = _evaluation()
    environment = _FakeEnvironment(truncate_step=5)
    policy = _FakePolicy()

    episode, frames = evaluation.run_simpler_episode(
        task=evaluation.SIMPLER_TASKS[1],
        object_episode_id=0,
        policy_seed=0,
        policy=policy,
        environment=environment,
        generator=object(),
        data_config={"train_crop_size": 4, "output_image_size": 4},
        denoising_steps=4,
        action_horizon=1,
        max_steps=12,
        capture_video=False,
    )

    assert episode["success"] is False
    assert episode["steps"] == 5
    assert episode["termination"] == "truncated"
    assert len(policy.calls) == 5
    assert len(frames) == 0


def test_episode_stats_are_converted_to_json_values():
    evaluation = _evaluation()
    environment = _FakeEnvironment(
        success_step=1,
        episode_stats={
            "grasped": np.bool_(True),
            "distance": np.float32(0.25),
            "counts": np.asarray([1, 2], dtype=np.int64),
        },
    )

    episode, _ = evaluation.run_simpler_episode(
        task=evaluation.SIMPLER_TASKS[0],
        object_episode_id=0,
        policy_seed=0,
        policy=_FakePolicy(),
        environment=environment,
        generator=object(),
        data_config={"train_crop_size": 4, "output_image_size": 4},
        denoising_steps=4,
        action_horizon=1,
        max_steps=8,
        capture_video=False,
    )

    assert episode["episode_stats"] == {
        "grasped": True,
        "distance": pytest.approx(0.25),
        "counts": [1, 2],
    }
    json.dumps(episode)


class _EvaluationPolicy(_FakePolicy):
    def make_generator(self, seed):
        return f"generator-{seed}"


def test_qwen_simpler_policy_forwards_single_observations_and_seeds():
    evaluation = _evaluation()
    bridge_policy = _FakePolicy()
    seeds = []
    policy = evaluation.QwenSimplerPolicy(
        policy=bridge_policy,
        device="cpu",
        generator_factory=lambda device, seed: seeds.append((device, seed)) or f"g-{seed}",
    )

    assert policy.make_generator(4) == "g-4"
    actions = policy.predict_actions(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros(8, dtype=np.float32),
        "Put Spoon on Towel",
        denoising_steps=3,
        generator="g-4",
    )

    assert actions.shape == (1, 8, 7)
    assert seeds == [("cpu", 4)]
    assert bridge_policy.calls[0]["instruction"] == "Put Spoon on Towel"


def test_evaluate_protocol_writes_partial_and_final_reports(tmp_path):
    evaluation = _evaluation()
    environments = []

    def environment_factory(task):
        environment = _FakeEnvironment(success_step=1 if task.key == "spoon" else None)
        environments.append(environment)
        return environment

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "checkpoints" / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0], evaluation.SIMPLER_TASKS[1]),
        policy_seeds=(0, 2),
        object_episode_ids=(0, 1),
        action_horizon=1,
    )
    checkpoint = SimpleNamespace(
        requested_path=settings.checkpoint,
        config={"data": {"train_crop_size": 4, "output_image_size": 4}},
        as_dict=lambda: {"requested_path": str(settings.checkpoint)},
    )

    report = evaluation.evaluate_simpler_checkpoint(
        settings,
        checkpoint=checkpoint,
        policy=_EvaluationPolicy(),
        environment_factory=environment_factory,
        source_versions={
            "simpler_env_commit": "06accaca9353",
            "maniskill2_real2sim_commit": "ef7a4d4",
        },
    )

    assert report["status"] == "complete"
    assert report["protocol"]["planned_episodes"] == 8
    assert report["summary"] == {
        "completed_episodes": 8,
        "successes": 4,
        "failures": 4,
        "success_rate": 0.5,
        "by_task": [
            {
                "task": "spoon",
                "completed_episodes": 4,
                "successes": 4,
                "failures": 0,
                "success_rate": 1.0,
            },
            {
                "task": "carrot",
                "completed_episodes": 4,
                "successes": 0,
                "failures": 4,
                "success_rate": 0.0,
            },
        ],
        "by_policy_seed": [
            {
                "policy_seed": 0,
                "completed_episodes": 4,
                "successes": 2,
                "failures": 2,
                "success_rate": 0.5,
            },
            {
                "policy_seed": 2,
                "completed_episodes": 4,
                "successes": 2,
                "failures": 2,
                "success_rate": 0.5,
            },
        ],
    }
    episode_lines = (settings.output_dir / "episodes.jsonl").read_text().splitlines()
    assert len(episode_lines) == 8
    assert not (settings.output_dir / "episodes.partial.jsonl").exists()
    assert json.loads((settings.output_dir / "results.json").read_text()) == report
    assert all(environment.closed for environment in environments)


def test_full_single_task_protocol_runs_24_by_3_and_updates_partial(tmp_path):
    evaluation = _evaluation()
    partial_counts = []
    made_generators = []

    class PartialTrackingEnvironment(_FakeEnvironment):
        def reset(self, *, options):
            partial = settings.output_dir / "episodes.partial.jsonl"
            partial_counts.append(
                len(partial.read_text().splitlines()) if partial.exists() else 0
            )
            return super().reset(options=options)

    class TrackingPolicy(_EvaluationPolicy):
        def make_generator(self, seed):
            made_generators.append(seed)
            return super().make_generator(seed)

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0],),
        max_steps=1,
    )

    def environment_factory(task):
        del task
        return PartialTrackingEnvironment(success_step=1)

    report = evaluation.evaluate_simpler_checkpoint(
        settings,
        checkpoint=SimpleNamespace(
            config={"data": {"train_crop_size": 4, "output_image_size": 4}},
            as_dict=lambda: {},
        ),
        policy=TrackingPolicy(),
        environment_factory=environment_factory,
        source_versions={},
    )

    assert report["summary"]["completed_episodes"] == 72
    assert report["protocol"]["planned_episodes"] == 72
    assert [item["completed_episodes"] for item in report["summary"]["by_policy_seed"]] == [
        24,
        24,
        24,
    ]
    assert made_generators == [0, 2, 4]
    assert partial_counts == list(range(72))
    episodes = [
        json.loads(line)
        for line in (settings.output_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert [(item["seed"], item["object_episode_id"]) for item in episodes] == [
        (seed, episode_id) for seed in (0, 2, 4) for episode_id in range(24)
    ]


def test_video_path_enables_all_episode_recordings(tmp_path):
    evaluation = _evaluation()
    written = []

    def video_writer(path, frames, fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        written.append((path, len(frames), fps))

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "checkpoints" / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0, 1),
        save_videos_path=tmp_path / "videos",
        video_fps=7,
    )
    checkpoint = SimpleNamespace(
        config={"data": {"train_crop_size": 4, "output_image_size": 4}},
        as_dict=lambda: {},
    )

    report = evaluation.evaluate_simpler_checkpoint(
        settings,
        checkpoint=checkpoint,
        policy=_EvaluationPolicy(),
        environment_factory=lambda task: _FakeEnvironment(success_step=1),
        source_versions={},
        video_writer=video_writer,
    )

    assert report["videos"] == [
        str(tmp_path / "videos/spoon/seed-0/episode-00_success.mp4"),
        str(tmp_path / "videos/spoon/seed-0/episode-01_success.mp4"),
    ]
    assert [(path.name, frames, fps) for path, frames, fps in written] == [
        ("episode-00_success.mp4", 2, 7),
        ("episode-01_success.mp4", 2, 7),
    ]


def test_runtime_contract_requires_the_single_process_version_set():
    evaluation = _evaluation()
    versions = dict(evaluation.RUNTIME_PACKAGE_VERSIONS)

    assert evaluation.validate_runtime_contract(
        version_info=(3, 10), package_versions=versions, device="cpu"
    ) == versions
    with pytest.raises(evaluation.SimplerEvaluationError, match="Python 3.10 or 3.11"):
        evaluation.validate_runtime_contract(
            version_info=(3, 12), package_versions=versions, device="cpu"
        )
    with pytest.raises(evaluation.SimplerEvaluationError, match="numpy==1.24.4"):
        evaluation.validate_runtime_contract(
            version_info=(3, 10),
            package_versions={**versions, "numpy": "2.1.3"},
            device="cpu",
        )


def test_runtime_contract_checks_simulator_and_video_stack_versions():
    evaluation = _evaluation()
    versions = dict(evaluation.RUNTIME_PACKAGE_VERSIONS)

    assert "sapien" in versions
    assert "gymnasium" in versions
    assert "imageio" in versions
    with pytest.raises(evaluation.SimplerEvaluationError, match="sapien=="):
        evaluation.validate_runtime_contract(
            version_info=(3, 10),
            package_versions={**versions, "sapien": "3.0.0"},
            device="cpu",
        )


def _init_git_repository(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "tracked").write_text("source", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_source_validation_checks_nested_commits_and_assets(tmp_path):
    evaluation = _evaluation()
    simpler_root = tmp_path / "SimplerEnv"
    simpler_commit = _init_git_repository(simpler_root)
    maniskill_root = simpler_root / "ManiSkill2_real2sim"
    maniskill_commit = _init_git_repository(maniskill_root)
    for relative in {
        task.overlay_relative_path for task in evaluation.SIMPLER_TASKS
    }:
        target = simpler_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"asset")
    (simpler_root / "simpler_env").mkdir()
    (maniskill_root / "mani_skill2_real2sim").mkdir()

    versions = evaluation.validate_simpler_source(
        simpler_root,
        expected_simpler_commit=simpler_commit,
        expected_maniskill_commit=maniskill_commit,
    )

    assert versions == {
        "simpler_env_commit": simpler_commit,
        "maniskill2_real2sim_commit": maniskill_commit,
    }
    assert evaluation.validate_simpler_source(
        simpler_root,
        expected_simpler_commit=simpler_commit[:12],
        expected_maniskill_commit=maniskill_commit[:12],
    ) == versions
    (simpler_root / evaluation.SIMPLER_TASKS[0].overlay_relative_path).unlink()
    with pytest.raises(evaluation.SimplerEvaluationError, match="asset"):
        evaluation.validate_simpler_source(
            simpler_root,
            expected_simpler_commit=simpler_commit,
            expected_maniskill_commit=maniskill_commit,
        )


def test_source_validation_rejects_local_source_modifications(tmp_path):
    evaluation = _evaluation()
    simpler_root = tmp_path / "SimplerEnv"
    simpler_commit = _init_git_repository(simpler_root)
    maniskill_root = simpler_root / "ManiSkill2_real2sim"
    maniskill_commit = _init_git_repository(maniskill_root)
    for relative in {task.overlay_relative_path for task in evaluation.SIMPLER_TASKS}:
        target = simpler_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"asset")
    (simpler_root / "simpler_env").mkdir()
    (maniskill_root / "mani_skill2_real2sim").mkdir()
    (maniskill_root / "tracked").write_text("modified", encoding="utf-8")

    with pytest.raises(evaluation.SimplerEvaluationError, match="local modifications"):
        evaluation.validate_simpler_source(
            simpler_root,
            expected_simpler_commit=simpler_commit,
            expected_maniskill_commit=maniskill_commit,
        )


def test_environment_factory_forwards_official_visual_matching_parameters(tmp_path):
    evaluation = _evaluation()
    simpler_root = tmp_path / "SimplerEnv"
    overlay = simpler_root / evaluation.SIMPLER_TASKS[0].overlay_relative_path
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"image")
    calls = []

    def build_environment(env_name, **kwargs):
        calls.append((env_name, kwargs))
        return "environment"

    result = evaluation.create_simpler_environment(
        evaluation.SIMPLER_TASKS[0],
        simpler_root=simpler_root,
        builder=build_environment,
        sim_device="cuda:3",
    )

    assert result == "environment"
    assert calls == [
        (
            "PutSpoonOnTableClothInScene-v0",
            {
                "obs_mode": "rgbd",
                "robot": "widowx",
                "sim_freq": 500,
                "control_mode": evaluation.CONTROL_MODE,
                "control_freq": 5,
                "max_episode_steps": 60,
                "scene_name": "bridge_table_1_v1",
                "camera_cfgs": {"add_segmentation": True},
                "rgb_overlay_path": str(overlay.resolve()),
                "renderer_kwargs": {
                    "offscreen_only": True,
                    "device": "cuda:3",
                },
            },
        )
    ]

    calls.clear()
    evaluation.create_simpler_environment(
        evaluation.SIMPLER_TASKS[0],
        simpler_root=simpler_root,
        builder=build_environment,
    )
    assert calls[0][1]["renderer_kwargs"] == {
        "offscreen_only": True,
        "device": "cuda:0",
    }


def test_evaluation_protects_existing_outputs_without_overwrite(tmp_path):
    evaluation = _evaluation()
    output = tmp_path / "results"
    output.mkdir()
    (output / "results.json").write_text("{}", encoding="utf-8")
    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00000001",
        output_dir=output,
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    with pytest.raises(evaluation.SimplerEvaluationError, match="--overwrite"):
        evaluation.evaluate_simpler_checkpoint(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=lambda task: _FakeEnvironment(),
            source_versions={},
        )


@pytest.mark.parametrize("filename", ["failure.json", "episodes.partial.jsonl"])
def test_evaluation_protects_recovery_outputs_without_overwrite(tmp_path, filename):
    evaluation = _evaluation()
    output = tmp_path / "results"
    output.mkdir()
    (output / filename).write_text("preserve", encoding="utf-8")
    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00000001",
        output_dir=output,
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    with pytest.raises(evaluation.SimplerEvaluationError, match="--overwrite"):
        evaluation.evaluate_simpler_checkpoint(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=lambda task: _FakeEnvironment(success_step=1),
            source_versions={},
        )
    assert (output / filename).read_text() == "preserve"


def test_successful_overwrite_removes_stale_failure(tmp_path):
    evaluation = _evaluation()
    output = tmp_path / "results"
    output.mkdir()
    (output / "failure.json").write_text("{}", encoding="utf-8")
    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00000001",
        output_dir=output,
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
        overwrite=True,
    )

    report = evaluation.evaluate_simpler_checkpoint(
        settings,
        checkpoint=SimpleNamespace(
            config={"data": {"train_crop_size": 4, "output_image_size": 4}},
            as_dict=lambda: {},
        ),
        policy=_EvaluationPolicy(),
        environment_factory=lambda task: _FakeEnvironment(success_step=1),
        source_versions={},
    )

    assert report["status"] == "complete"
    assert not (output / "failure.json").exists()


def test_preflight_checks_every_environment_and_one_model_inference(tmp_path):
    evaluation = _evaluation()
    environments = []

    def environment_factory(task):
        environment = _FakeEnvironment()
        environments.append(environment)
        return environment

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "preflight",
        tasks=(evaluation.SIMPLER_TASKS[0], evaluation.SIMPLER_TASKS[-1]),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )
    checkpoint = SimpleNamespace(
        config={"data": {"train_crop_size": 4, "output_image_size": 4}},
        as_dict=lambda: {"requested_path": str(settings.checkpoint)},
    )
    policy = _EvaluationPolicy()

    report = evaluation.run_simpler_preflight(
        settings,
        checkpoint=checkpoint,
        policy=policy,
        environment_factory=environment_factory,
        source_versions={"simpler_env_commit": "06accaca9353"},
        package_versions={"numpy": "1.24.4"},
    )

    assert report["status"] == "passed"
    assert [item["task"] for item in report["environments"]] == ["spoon", "eggplant"]
    assert report["model_inference"]["action_shape"] == [1, 8, 7]
    assert len(policy.calls) == 1
    assert all(environment.closed for environment in environments)
    assert all(len(environment.actions) == 1 for environment in environments)
    assert all(environment.actions[0][-1] == 1.0 for environment in environments)
    assert json.loads((settings.output_dir / "preflight.json").read_text()) == report


def test_preflight_reset_failure_is_infrastructure_error(tmp_path):
    evaluation = _evaluation()

    class BrokenEnvironment(_FakeEnvironment):
        def reset(self, *, options):
            raise RuntimeError("renderer unavailable")

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "preflight",
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    with pytest.raises(evaluation.SimplerInfrastructureError, match="reset"):
        evaluation.run_simpler_preflight(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=lambda task: BrokenEnvironment(),
            source_versions={},
            package_versions={},
        )


def test_preflight_close_failure_is_infrastructure_error(tmp_path):
    evaluation = _evaluation()
    environment = _CloseFailingEnvironment(success_step=1)
    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "preflight",
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    with pytest.raises(evaluation.SimplerInfrastructureError, match="close failed") as caught:
        evaluation.run_simpler_preflight(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=lambda task: environment,
            source_versions={},
            package_versions={},
        )

    assert environment.close_count == 1
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "renderer teardown failed"
    assert not (settings.output_dir / "preflight.json").exists()


def test_preflight_protects_failure_output_without_overwrite(tmp_path):
    evaluation = _evaluation()
    output = tmp_path / "preflight"
    output.mkdir()
    (output / "failure.json").write_text("preserve", encoding="utf-8")
    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=output,
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    with pytest.raises(evaluation.SimplerEvaluationError, match="--overwrite"):
        evaluation.run_simpler_preflight(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=lambda task: _FakeEnvironment(),
            source_versions={},
            package_versions={},
        )
    assert (output / "failure.json").read_text() == "preserve"


def test_task_execution_error_is_recorded_and_later_tasks_continue(tmp_path):
    evaluation = _evaluation()
    environments = []

    class TaskFailingPolicy(_EvaluationPolicy):
        def __init__(self):
            super().__init__()
            self.calls_by_instruction = {}

        def predict_actions(
            self, image, state, instruction, denoising_steps, *, generator
        ):
            self.calls_by_instruction[instruction] = (
                self.calls_by_instruction.get(instruction, 0) + 1
            )
            if (
                instruction == "Put Spoon on Towel"
                and self.calls_by_instruction[instruction] == 2
            ):
                raise RuntimeError("policy task failed")
            return super().predict_actions(
                image,
                state,
                instruction,
                denoising_steps,
                generator=generator,
            )

    def environment_factory(task):
        environment = _FakeEnvironment(success_step=1)
        environments.append(environment)
        return environment

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0], evaluation.SIMPLER_TASKS[1]),
        policy_seeds=(0,),
        object_episode_ids=(0, 1),
        max_steps=1,
    )
    checkpoint = SimpleNamespace(
        config={"data": {"train_crop_size": 4, "output_image_size": 4}},
        as_dict=lambda: {},
    )

    report = evaluation.evaluate_simpler_checkpoint(
        settings,
        checkpoint=checkpoint,
        policy=TaskFailingPolicy(),
        environment_factory=environment_factory,
        source_versions={},
    )

    assert report["status"] == "completed_with_errors"
    assert report["summary"]["completed_episodes"] == 3
    assert report["summary"]["successes"] == 3
    assert report["task_errors"] == [
        {
            "task": "spoon",
            "policy_seed": 0,
            "object_episode_id": 1,
            "error_type": "RuntimeError",
            "error": "policy task failed",
        }
    ]
    assert json.loads((settings.output_dir / "failure.json").read_text())["status"] == (
        "completed_with_errors"
    )
    assert len(environments) == 2
    assert [environment.close_count for environment in environments] == [1, 1]


def test_simulator_reset_error_is_immediate_infrastructure_failure(tmp_path):
    evaluation = _evaluation()
    environments = []

    class BrokenEnvironment(_FakeEnvironment):
        def reset(self, *, options):
            self.reset_calls = getattr(self, "reset_calls", 0) + 1
            if self.reset_calls == 2:
                raise RuntimeError("renderer unavailable")
            return super().reset(options=options)

    def environment_factory(task):
        del task
        environment = BrokenEnvironment(success_step=1)
        environments.append(environment)
        return environment

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0], evaluation.SIMPLER_TASKS[1]),
        policy_seeds=(0,),
        object_episode_ids=(0, 1),
        max_steps=1,
    )

    with pytest.raises(evaluation.SimplerInfrastructureError, match="reset"):
        evaluation.evaluate_simpler_checkpoint(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=environment_factory,
            source_versions={},
        )
    failure = json.loads((settings.output_dir / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["exit_code"] == 3
    assert failure["summary"]["completed_episodes"] == 1
    assert len(environments) == 1
    assert environments[0].close_count == 1


def test_environment_creation_error_is_immediate_infrastructure_failure(tmp_path):
    evaluation = _evaluation()
    factory_tasks = []

    def environment_factory(task):
        factory_tasks.append(task.key)
        raise RuntimeError("renderer unavailable")

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0], evaluation.SIMPLER_TASKS[1]),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    with pytest.raises(evaluation.SimplerInfrastructureError, match="renderer"):
        evaluation.evaluate_simpler_checkpoint(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=environment_factory,
            source_versions={},
        )

    failure = json.loads((settings.output_dir / "failure.json").read_text())
    assert factory_tasks == ["spoon"]
    assert failure["status"] == "failed"
    assert failure["exit_code"] == 3
    assert failure["summary"]["completed_episodes"] == 0


def test_environment_close_error_is_immediate_infrastructure_failure(tmp_path):
    evaluation = _evaluation()
    environment = _CloseFailingEnvironment(success_step=1)
    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
        max_steps=1,
    )

    with pytest.raises(evaluation.SimplerInfrastructureError, match="close failed") as caught:
        evaluation.evaluate_simpler_checkpoint(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=lambda task: environment,
            source_versions={},
        )

    failure = json.loads((settings.output_dir / "failure.json").read_text())
    assert environment.close_count == 1
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "renderer teardown failed"
    assert failure["exit_code"] == 3
    assert failure["summary"]["completed_episodes"] == 1


def test_reset_error_remains_primary_when_environment_close_also_fails(tmp_path):
    evaluation = _evaluation()

    class ResetAndCloseFailingEnvironment(_CloseFailingEnvironment):
        def reset(self, *, options):
            del options
            raise RuntimeError("renderer unavailable")

    environment = ResetAndCloseFailingEnvironment()
    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "results",
        tasks=(evaluation.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    with pytest.raises(evaluation.SimplerInfrastructureError, match="reset failed") as caught:
        evaluation.evaluate_simpler_checkpoint(
            settings,
            checkpoint=SimpleNamespace(
                config={"data": {"train_crop_size": 4, "output_image_size": 4}},
                as_dict=lambda: {},
            ),
            policy=_EvaluationPolicy(),
            environment_factory=lambda task: environment,
            source_versions={},
        )

    failure = json.loads((settings.output_dir / "failure.json").read_text())
    assert environment.close_count == 1
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "renderer unavailable"
    assert any(
        "close failed" in note and "renderer teardown failed" in note
        for note in caught.value.__notes__
    )
    assert failure["exit_code"] == 3
    assert "reset failed" in failure["error"]


@pytest.mark.gpu
@pytest.mark.real_data
@pytest.mark.skipif(
    os.environ.get("RUN_SIMPLER_INTEGRATION") != "1",
    reason="set RUN_SIMPLER_INTEGRATION=1 in the prepared SimplerEnv GPU runtime",
)
def test_real_simpler_preflight_integration(tmp_path, monkeypatch):
    evaluation = _evaluation()
    simpler_root = evaluation.default_simpler_root()
    maniskill_root = simpler_root / "ManiSkill2_real2sim"
    monkeypatch.setenv("MS2_REAL2SIM_ASSET_DIR", str(maniskill_root / "data"))
    monkeypatch.syspath_prepend(str(maniskill_root))
    monkeypatch.syspath_prepend(str(simpler_root))
    for name in [key for key in sys.modules if key.startswith("simpler_env")]:
        sys.modules.pop(name)

    settings = evaluation.SimplerEvaluationSettings(
        checkpoint=tmp_path / "step-00020000",
        output_dir=tmp_path / "preflight",
        tasks=evaluation.SIMPLER_TASKS,
        policy_seeds=(0,),
        object_episode_ids=(0,),
        device="cuda:0",
    )
    checkpoint = SimpleNamespace(
        config={"data": {"train_crop_size": 230, "output_image_size": 256}},
        as_dict=lambda: {},
    )

    report = evaluation.run_simpler_preflight(
        settings,
        checkpoint=checkpoint,
        policy=_EvaluationPolicy(),
        environment_factory=lambda task: evaluation.create_simpler_environment(
            task, simpler_root=simpler_root
        ),
        source_versions=evaluation.validate_simpler_source(simpler_root),
        package_versions={"numpy": np.__version__},
    )

    assert report["status"] == "passed"
    assert [item["task"] for item in report["environments"]] == [
        "spoon",
        "carrot",
        "stack",
        "eggplant",
    ]
