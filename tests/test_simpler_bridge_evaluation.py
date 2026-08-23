import dataclasses
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_qwen_reexports_the_shared_simpler_protocol():
    shared = importlib.import_module("simpler_bridge.evaluation")
    qwen = importlib.import_module("qwen3_vl_groot.simpler_evaluation")

    assert qwen.SimplerEvaluationError is shared.SimplerEvaluationError
    assert qwen.SimplerInfrastructureError is shared.SimplerInfrastructureError
    assert qwen.SimplerTaskSpec is shared.SimplerTaskSpec
    assert qwen.SIMPLER_TASKS is shared.SIMPLER_TASKS
    assert qwen.resolve_task_selection is shared.resolve_task_selection
    assert qwen.environment_to_bridge_proprio is shared.environment_to_bridge_proprio
    assert qwen.bridge_actions_to_simpler is shared.bridge_actions_to_simpler
    assert qwen.image_from_simpler_observation is shared.image_from_simpler_observation
    assert qwen.validate_simpler_source is shared.validate_simpler_source
    assert qwen.create_simpler_environment is shared.create_simpler_environment
    assert qwen.default_simpler_root is shared.default_simpler_root


class _IdentityPose:
    def __init__(self, position):
        self.p = np.asarray(position, dtype=np.float64)
        self.q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def inv(self):
        return _IdentityPose(-self.p)

    def __mul__(self, other):
        return _IdentityPose(self.p + other.p)


class _Environment:
    def __init__(self):
        self.agent = SimpleNamespace(
            robot=SimpleNamespace(pose=_IdentityPose([0.0, 0.0, 0.0])),
            get_gripper_closedness=lambda: 0.0,
        )
        self.tcp = SimpleNamespace(pose=_IdentityPose([0.1, 0.2, 0.3]))
        self.actions = []

    def reset(self, *, options):
        self.options = options
        return {
            "image": {
                "3rd_view_camera": {
                    "rgb": np.zeros((8, 12, 3), dtype=np.uint8),
                }
            }
        }, {}

    def step(self, action):
        self.actions.append(np.asarray(action))
        return self.reset(options=self.options)[0], 1.0, True, False, {}

    def close(self):
        self.closed = True


class _LifecycleEnvironment(_Environment):
    def __init__(self, task_key, partial_path):
        super().__init__()
        self.task_key = task_key
        self.partial_path = partial_path
        self.reset_episode_ids = []
        self.partial_counts_before_reset = []
        self.partial_count_at_close = None
        self.close_count = 0

    def _observation(self):
        return {
            "image": {
                "3rd_view_camera": {
                    "rgb": np.zeros((8, 12, 3), dtype=np.uint8),
                }
            }
        }

    def reset(self, *, options):
        self.options = options
        self.reset_episode_ids.append(options["obj_init_options"]["episode_id"])
        self.partial_counts_before_reset.append(
            len(self.partial_path.read_text().splitlines()) if self.partial_path.exists() else 0
        )
        return self._observation(), {}

    def step(self, action):
        self.actions.append(np.asarray(action))
        return self._observation(), 1.0, True, False, {}

    def close(self):
        self.close_count += 1
        self.partial_count_at_close = (
            len(self.partial_path.read_text().splitlines()) if self.partial_path.exists() else 0
        )


class _Adapter:
    policy_name = "test-policy"

    def __init__(self):
        self.prepared = []
        self.generators = []
        self.episode_instructions = []
        self.selected_chunks = []

    def make_generator(self, seed):
        return f"generator-{seed}"

    def begin_episode(self, instruction):
        self.episode_instructions.append(instruction)

    def select_action(self, actions):
        chunk = np.asarray(actions)
        self.selected_chunks.append(chunk.copy())
        return chunk[0, 0]

    def prepare_observation(self, image, proprio, instruction):
        value = {
            "image": np.asarray(image),
            "proprio": np.asarray(proprio),
            "instruction": instruction,
        }
        self.prepared.append(value)
        return value

    def describe_observation(self, prepared):
        return {"model_image_shape": list(prepared["image"].shape)}

    def predict_actions(self, prepared, *, generator):
        self.generators.append(generator)
        actions = np.zeros((1, 8, 7), dtype=np.float32)
        actions[..., 6] = 1.0
        return actions

    def protocol_metadata(self):
        return {"policy": "test"}


def test_shared_close_preserves_primary_error_without_add_note_support():
    shared = importlib.import_module("simpler_bridge.evaluation")

    class Python310StyleError:
        def __init__(self):
            self.__notes__ = ["existing context"]

    class CloseFailingEnvironment:
        def close(self):
            raise RuntimeError("renderer teardown failed")

    primary_error = Python310StyleError()

    shared._close_simpler_environment(
        CloseFailingEnvironment(),
        task=shared.SIMPLER_TASKS[0],
        primary_error=primary_error,
    )

    assert primary_error.__notes__ == [
        "existing context",
        "SimplerEnv close failed for task spoon: RuntimeError: renderer teardown failed",
    ]


class _ThreeStepEnvironment(_Environment):
    def step(self, action):
        self.actions.append(np.asarray(action))
        success = len(self.actions) == 3
        return self.reset(options=self.options)[0], float(success), success, False, {}


class _VariableChunkAdapter(_Adapter):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size

    def predict_actions(self, prepared, *, generator):
        self.generators.append(generator)
        call_index = len(self.generators) - 1
        actions = np.full((1, self.chunk_size, 7), 99.0, dtype=np.float32)
        actions[0, 0] = np.asarray([call_index, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        return actions


@pytest.mark.parametrize("chunk_size", [1, 8, 16])
def test_shared_runner_replans_every_step_and_executes_only_chunk_head(chunk_size):
    shared = importlib.import_module("simpler_bridge.evaluation")
    environment = _ThreeStepEnvironment()
    adapter = _VariableChunkAdapter(chunk_size)

    episode, _ = shared.run_simpler_episode(
        task=shared.SIMPLER_TASKS[0],
        object_episode_id=0,
        policy_seed=2,
        policy=adapter,
        environment=environment,
        generator="generator-2",
        action_horizon=1,
        max_steps=5,
        capture_video=False,
    )

    assert episode["steps"] == 3
    assert [action[0] for action in environment.actions] == [0.0, 1.0, 2.0]
    assert adapter.generators == ["generator-2"] * 3


def test_shared_protocol_defaults_to_stepwise_first_action_and_rejects_other_horizons(
    tmp_path,
):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )

    assert settings.action_horizon == 1
    report = shared.evaluate_simpler_policy(
        settings,
        checkpoint={"requested_path": "/models/checkpoint"},
        policy=_Adapter(),
        environment_factory=lambda task: _Environment(),
        source_versions={},
        route="stepwise-test",
        protocol_metadata={"native_action_chunk_size": 8},
    )
    assert report["protocol"]["execution_mode"] == "stepwise_first_action"
    assert report["protocol"]["action_horizon"] == 1

    invalid = shared.SimplerRunSettings(
        output_dir=tmp_path / "invalid",
        tasks=(shared.SIMPLER_TASKS[0],),
        action_horizon=8,
        policy_seeds=(0,),
        object_episode_ids=(0,),
    )
    with pytest.raises(shared.SimplerEvaluationError, match="must be 1"):
        shared.evaluate_simpler_policy(
            invalid,
            checkpoint={},
            policy=_Adapter(),
            environment_factory=lambda task: _Environment(),
            source_versions={},
            route="invalid-stepwise-test",
            protocol_metadata={},
        )


def test_full_shared_protocol_plans_four_by_twenty_four_by_three_episodes(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=shared.SIMPLER_TASKS,
    )

    protocol = shared._protocol(
        settings,
        {"native_action_chunk_size": 16},
    )

    assert protocol["tasks"] == ["spoon", "carrot", "stack", "eggplant"]
    assert protocol["policy_seeds"] == [0, 2, 4]
    assert len(protocol["object_episode_ids"]) == 24
    assert protocol["planned_episodes"] == 288


def test_shared_protocol_records_policy_execution_and_instruction_source(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
        max_steps=120,
        execution_mode="adaptive_ensemble_v1",
        instruction_source="environment",
        episode_protocol="starvla_reference_24",
    )

    protocol = shared._protocol(
        settings,
        {
            "action_postprocessing": "adaptive_ensemble_v1",
            "adaptive_ensemble_horizon": 7,
            "adaptive_ensemble_alpha": 0.1,
        },
    )

    assert protocol["execution_mode"] == "adaptive_ensemble_v1"
    assert protocol["instruction_source"] == "environment"
    assert protocol["episode_protocol"] == "starvla_reference_24"
    assert protocol["max_steps_override"] == 120
    assert protocol["environment_lifecycle"] == "one_per_task"
    assert protocol["adaptive_ensemble_horizon"] == 7
    assert protocol["adaptive_ensemble_alpha"] == 0.1


def test_shared_episode_runner_uses_a_model_independent_policy_adapter():
    shared = importlib.import_module("simpler_bridge.evaluation")
    environment = _Environment()
    adapter = _Adapter()

    episode, frames = shared.run_simpler_episode(
        task=shared.SIMPLER_TASKS[0],
        object_episode_id=3,
        policy_seed=2,
        policy=adapter,
        environment=environment,
        generator="generator-2",
        action_horizon=1,
        max_steps=60,
        capture_video=True,
    )

    assert episode["success"] is True
    assert episode["steps"] == 1
    assert environment.options["obj_init_options"] == {"episode_id": 3}
    assert len(environment.actions) == 1
    assert len(frames) == 2
    assert adapter.prepared[0]["instruction"] == "Put Spoon on Towel"
    assert adapter.generators == ["generator-2"]
    assert adapter.episode_instructions == ["Put Spoon on Towel"]
    assert len(adapter.selected_chunks) == 1


def test_shared_episode_runner_executes_the_action_selected_by_the_adapter():
    shared = importlib.import_module("simpler_bridge.evaluation")
    environment = _Environment()

    class TailSelectingAdapter(_Adapter):
        def predict_actions(self, prepared, *, generator):
            actions = np.zeros((1, 8, 7), dtype=np.float32)
            actions[0, 0, 0] = 1.0
            actions[0, -1, 0] = 7.0
            actions[..., 6] = 1.0
            return actions

        def select_action(self, actions):
            return np.asarray(actions)[0, -1]

    shared.run_simpler_episode(
        task=shared.SIMPLER_TASKS[0],
        object_episode_id=0,
        policy_seed=0,
        policy=TailSelectingAdapter(),
        environment=environment,
        generator="generator-0",
        action_horizon=1,
        max_steps=1,
        capture_video=False,
    )

    assert environment.actions[0][0] == 7.0


def test_shared_episode_runner_uses_the_environment_instruction_when_requested():
    shared = importlib.import_module("simpler_bridge.evaluation")

    class EnvironmentInstruction(_Environment):
        def get_language_instruction(self):
            return "put the spoon on the towel"

    environment = EnvironmentInstruction()
    adapter = _Adapter()

    episode, _ = shared.run_simpler_episode(
        task=shared.SIMPLER_TASKS[0],
        object_episode_id=0,
        policy_seed=0,
        policy=adapter,
        environment=environment,
        generator="generator-0",
        action_horizon=1,
        max_steps=1,
        capture_video=False,
        instruction_source="environment",
    )

    assert episode["instruction"] == "put the spoon on the towel"
    assert adapter.episode_instructions == ["put the spoon on the towel"]
    assert adapter.prepared[0]["instruction"] == "put the spoon on the towel"


def test_shared_episode_runner_uses_adapter_gripper_threshold():
    shared = importlib.import_module("simpler_bridge.evaluation")
    environment = _Environment()

    class NormalizedGripperAdapter(_Adapter):
        gripper_threshold = 0.0

        def predict_actions(self, prepared, *, generator):
            actions = np.zeros((1, 8, 7), dtype=np.float32)
            actions[..., 6] = 0.25
            return actions

    shared.run_simpler_episode(
        task=shared.SIMPLER_TASKS[0],
        object_episode_id=0,
        policy_seed=0,
        policy=NormalizedGripperAdapter(),
        environment=environment,
        generator="generator-0",
        action_horizon=1,
        max_steps=60,
        capture_video=False,
    )

    assert environment.actions[0][6] == 1.0


def test_shared_evaluator_writes_model_specific_routes_and_protocol(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(2,),
        object_episode_ids=(3,),
    )

    report = shared.evaluate_simpler_policy(
        settings,
        checkpoint={"requested_path": "/models/step-00020000"},
        policy=_Adapter(),
        environment_factory=lambda task: _Environment(),
        source_versions={"simpler_env_commit": "abc"},
        route="octo-test-eval",
        protocol_metadata={"diffusion_steps": 20},
    )

    assert report["route"] == "octo-test-eval"
    assert report["checkpoint"]["requested_path"] == "/models/step-00020000"
    assert report["protocol"]["diffusion_steps"] == 20
    assert report["summary"]["completed_episodes"] == 1
    assert report["task_errors"] == []


def test_shared_evaluator_reuses_one_environment_per_task_in_episode_order(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    output_dir = tmp_path / "results"
    environments = []
    factory_tasks = []
    made_generators = []

    class TrackingAdapter(_Adapter):
        def make_generator(self, seed):
            made_generators.append(seed)
            return super().make_generator(seed)

    def environment_factory(task):
        factory_tasks.append(task.key)
        environment = _LifecycleEnvironment(
            task.key,
            output_dir / "episodes.partial.jsonl",
        )
        environments.append(environment)
        return environment

    settings = shared.SimplerRunSettings(
        output_dir=output_dir,
        tasks=(shared.SIMPLER_TASKS[0], shared.SIMPLER_TASKS[1]),
        policy_seeds=(4, 0),
        object_episode_ids=(3, 1),
        max_steps=1,
    )

    report = shared.evaluate_simpler_policy(
        settings,
        checkpoint={},
        policy=TrackingAdapter(),
        environment_factory=environment_factory,
        source_versions={},
        route="lifecycle-test",
        protocol_metadata={},
    )

    assert factory_tasks == ["spoon", "carrot"]
    assert made_generators == [4, 0, 4, 0]
    assert [environment.reset_episode_ids for environment in environments] == [
        [3, 1, 3, 1],
        [3, 1, 3, 1],
    ]
    assert [environment.close_count for environment in environments] == [1, 1]
    assert [environment.partial_count_at_close for environment in environments] == [
        4,
        8,
    ]
    assert [
        count for environment in environments for count in environment.partial_counts_before_reset
    ] == list(range(8))
    episodes = [
        json.loads(line) for line in (output_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert [
        (episode["task"], episode["policy_seed"], episode["object_episode_id"])
        for episode in episodes
    ] == [
        (task, seed, episode_id)
        for task in ("spoon", "carrot")
        for seed in (4, 0)
        for episode_id in (3, 1)
    ]
    assert report["summary"]["completed_episodes"] == 8
    assert not (output_dir / "episodes.partial.jsonl").exists()


def test_episode_inference_seed_uses_the_versioned_task_seed_episode_contract():
    shared = importlib.import_module("simpler_bridge.evaluation")

    assert shared.episode_inference_seed("spoon", 2, 3) == 1475198320439424009
    assert shared.episode_inference_seed("carrot", 2, 3) != 1475198320439424009


def test_shared_episode_plan_round_robins_the_canonical_matrix_without_gaps(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=shared.SIMPLER_TASKS[:2],
        policy_seeds=(0, 2),
        object_episode_ids=(0, 1, 2),
        shard_count=4,
        rng_scope="per_episode",
    )

    assignments = []
    for shard_index in range(4):
        shard = shared.assigned_episode_plan(dataclasses.replace(settings, shard_index=shard_index))
        assignments.extend(
            (
                episode.canonical_index,
                shard_index,
                episode.task.key,
                episode.policy_seed,
                episode.object_episode_id,
            )
            for episode in shard
        )

    assert sorted(assignments) == [
        (0, 0, "spoon", 0, 0),
        (1, 1, "spoon", 0, 1),
        (2, 2, "spoon", 0, 2),
        (3, 3, "spoon", 2, 0),
        (4, 0, "spoon", 2, 1),
        (5, 1, "spoon", 2, 2),
        (6, 2, "carrot", 0, 0),
        (7, 3, "carrot", 0, 1),
        (8, 0, "carrot", 0, 2),
        (9, 1, "carrot", 2, 0),
        (10, 2, "carrot", 2, 1),
        (11, 3, "carrot", 2, 2),
    ]


def test_delayed_shard_output_check_ignores_video_written_by_another_shard(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    videos = tmp_path / "videos"
    other_shard_video = videos / "spoon/seed-0/episode-01_success.mp4"
    other_shard_video.parent.mkdir(parents=True)
    other_shard_video.write_bytes(b"other shard")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "worker-0",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0, 1),
        save_videos_path=videos,
        shard_index=0,
        shard_count=2,
        rng_scope="per_episode",
    )

    shared._check_output_targets(settings)


def test_shared_parallel_runner_reseeds_each_episode_and_records_the_seed(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    adapter = _Adapter()
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(2,),
        object_episode_ids=(3, 4),
        max_steps=1,
        rng_scope="per_episode",
    )

    report = shared.evaluate_simpler_policy(
        settings,
        checkpoint={},
        policy=adapter,
        environment_factory=lambda task: _Environment(),
        source_versions={},
        route="parallel-rng-test",
        protocol_metadata={},
    )

    assert adapter.generators == [
        "generator-1475198320439424009",
        "generator-5734274001812672943",
    ]
    episodes = [
        json.loads(line)
        for line in (settings.output_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert [episode["inference_seed"] for episode in episodes] == [
        1475198320439424009,
        5734274001812672943,
    ]
    assert report["protocol"]["rng_scope"] == "per_episode"
    assert report["protocol"]["rng_seed_derivation"] == "sha256-octo-simpler-episode-v1"


def test_shared_runner_rejects_sharding_a_continuous_policy_seed_stream(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
        shard_count=2,
    )

    with pytest.raises(shared.SimplerEvaluationError, match="per_episode"):
        shared.evaluate_simpler_policy(
            settings,
            checkpoint={},
            policy=_Adapter(),
            environment_factory=lambda task: _Environment(),
            source_versions={},
            route="invalid-shard-rng-test",
            protocol_metadata={},
        )


def test_shared_task_execution_error_does_not_prevent_later_tasks(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    environments = []

    class TaskFailingAdapter(_Adapter):
        def predict_actions(self, prepared, *, generator):
            if prepared["instruction"] == "Put Spoon on Towel" and len(self.generators) == 1:
                raise RuntimeError("policy task failed")
            return super().predict_actions(prepared, generator=generator)

    def environment_factory(task):
        environment = _Environment()
        environments.append(environment)
        return environment

    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=shared.SIMPLER_TASKS[:2],
        policy_seeds=(0,),
        object_episode_ids=(0, 1),
        max_steps=1,
    )

    report = shared.evaluate_simpler_policy(
        settings,
        checkpoint={},
        policy=TaskFailingAdapter(),
        environment_factory=environment_factory,
        source_versions={},
        route="task-error-test",
        protocol_metadata={},
    )

    assert report["status"] == "completed_with_errors"
    assert report["summary"]["completed_episodes"] == 3
    assert report["task_errors"] == [
        {
            "task": "spoon",
            "policy_seed": 0,
            "object_episode_id": 1,
            "error_type": "RuntimeError",
            "error": "policy task failed",
        }
    ]
    assert len(environments) == 2


@pytest.mark.parametrize(
    "sim_device",
    ["cpu", "cuda", "cuda:-1", "cuda:+1", "cuda:1.0", "cuda: 1", "CUDA:1", "4"],
)
def test_shared_settings_reject_invalid_sim_renderer_devices(tmp_path, sim_device):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / sim_device.replace("/", "_"),
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
        sim_device=sim_device,
    )

    with pytest.raises(shared.SimplerEvaluationError, match="sim_device"):
        shared.evaluate_simpler_policy(
            settings,
            checkpoint={},
            policy=_Adapter(),
            environment_factory=lambda task: _Environment(),
            source_versions={},
            route="invalid-sim-device-test",
            protocol_metadata={},
        )


def test_shared_protocol_records_sim_renderer_and_task_lifecycle(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    default_settings = shared.SimplerRunSettings(output_dir=tmp_path / "default")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "results",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
        sim_device="cuda:12",
    )

    assert default_settings.sim_device == "cuda:0"
    report = shared.evaluate_simpler_policy(
        settings,
        checkpoint={},
        policy=_Adapter(),
        environment_factory=lambda task: _Environment(),
        source_versions={},
        route="sim-renderer-protocol-test",
        protocol_metadata={},
    )
    assert report["protocol"]["sim_renderer_device"] == "cuda:12"
    assert report["protocol"]["sim_renderer_offscreen_only"] is True
    assert report["protocol"]["environment_lifecycle"] == "one_per_task"


def test_shared_preflight_reports_adapter_input_metadata(tmp_path):
    shared = importlib.import_module("simpler_bridge.evaluation")
    settings = shared.SimplerRunSettings(
        output_dir=tmp_path / "preflight",
        tasks=(shared.SIMPLER_TASKS[0],),
        policy_seeds=(0,),
        object_episode_ids=(0,),
        device="cpu",
    )

    report = shared.run_simpler_preflight(
        settings,
        checkpoint={"requested_path": "/models/step-00020000"},
        policy=_Adapter(),
        environment_factory=lambda task: _Environment(),
        source_versions={"simpler_env_commit": "abc"},
        package_versions={"numpy": "1.24.3"},
        route="octo-test-preflight",
    )

    assert report["route"] == "octo-test-preflight"
    assert report["protocol"]["execution_mode"] == "stepwise_first_action"
    assert report["protocol"]["action_horizon"] == 1
    assert report["protocol"]["policy"] == "test"
    assert report["model_inference"] == {
        "task": "spoon",
        "action_shape": [1, 8, 7],
    }
    assert report["environments"][0]["model_image_shape"] == [8, 12, 3]
    assert report["runtime"]["package_versions"] == {"numpy": "1.24.3"}
