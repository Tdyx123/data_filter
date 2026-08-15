import importlib
from types import SimpleNamespace

import numpy as np


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


class _Adapter:
    policy_name = "test-policy"

    def __init__(self):
        self.prepared = []
        self.generators = []

    def make_generator(self, seed):
        return f"generator-{seed}"

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
        action_horizon=8,
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
        action_horizon=8,
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
    assert report["model_inference"] == {
        "task": "spoon",
        "action_shape": [1, 8, 7],
    }
    assert report["environments"][0]["model_image_shape"] == [8, 12, 3]
    assert report["runtime"]["package_versions"] == {"numpy": "1.24.3"}
