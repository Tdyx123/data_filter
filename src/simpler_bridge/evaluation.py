"""Model-independent SimplerEnv protocol for Bridge-trained policies."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


SIMPLER_ENV_COMMIT = "06accaca9353"
MANISKILL2_REAL2SIM_COMMIT = "ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682"
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
POLICY_SEEDS = (0, 2, 4)
OBJECT_EPISODE_IDS = tuple(range(24))


class SimplerEvaluationError(RuntimeError):
    """Raised when the local SimplerEnv evaluation contract is invalid."""


class SimplerInfrastructureError(RuntimeError):
    """Raised when SimplerEnv cannot create or run its simulation."""


@dataclass(frozen=True)
class SimplerTaskSpec:
    key: str
    env_name: str
    instruction: str
    scene_name: str
    robot: str
    robot_init_xy: tuple[float, float]
    max_steps: int
    overlay_relative_path: str


@dataclass(frozen=True)
class SimplerRunSettings:
    output_dir: Path
    tasks: tuple[SimplerTaskSpec, ...] = ()
    device: str = "cuda:0"
    action_horizon: int = 8
    policy_seeds: tuple[int, ...] = POLICY_SEEDS
    object_episode_ids: tuple[int, ...] = OBJECT_EPISODE_IDS
    max_steps: int | None = None
    save_videos_path: Path | None = None
    video_fps: int = 5
    overwrite: bool = False


class SimplerPolicyAdapter(Protocol):
    """Boundary between a model implementation and the shared episode runner."""

    policy_name: str
    gripper_threshold: float

    def make_generator(self, seed: int) -> Any: ...

    def prepare_observation(
        self,
        image: np.ndarray,
        proprio: np.ndarray,
        instruction: str,
    ) -> Any: ...

    def describe_observation(self, prepared: Any) -> Mapping[str, Any]: ...

    def predict_actions(self, prepared: Any, *, generator: Any) -> Any: ...

    def protocol_metadata(self) -> Mapping[str, Any]: ...


SIMPLER_TASKS = (
    SimplerTaskSpec(
        key="spoon",
        env_name="PutSpoonOnTableClothInScene-v0",
        instruction="Put Spoon on Towel",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        robot_init_xy=(0.147, 0.028),
        max_steps=60,
        overlay_relative_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
    ),
    SimplerTaskSpec(
        key="carrot",
        env_name="PutCarrotOnPlateInScene-v0",
        instruction="Put Carrot on Plate",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        robot_init_xy=(0.147, 0.028),
        max_steps=60,
        overlay_relative_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
    ),
    SimplerTaskSpec(
        key="stack",
        env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        instruction="Stack Green Block on Yellow Block",
        scene_name="bridge_table_1_v1",
        robot="widowx",
        robot_init_xy=(0.147, 0.028),
        max_steps=60,
        overlay_relative_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
    ),
    SimplerTaskSpec(
        key="eggplant",
        env_name="PutEggplantInBasketScene-v0",
        instruction="Put Eggplant in Yellow Basket",
        scene_name="bridge_table_1_v2",
        robot="widowx_sink_camera_setup",
        robot_init_xy=(0.127, 0.06),
        max_steps=120,
        overlay_relative_path="ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
    ),
)
_TASKS_BY_KEY = {task.key: task for task in SIMPLER_TASKS}


def resolve_task_selection(value: str) -> tuple[SimplerTaskSpec, ...]:
    selection = str(value).strip().lower()
    if selection == "all":
        return SIMPLER_TASKS
    keys = [item.strip() for item in selection.split(",")]
    if not keys or any(not key for key in keys):
        raise SimplerEvaluationError("--tasks requires 'all' or a non-empty task list")
    if len(keys) != len(set(keys)):
        raise SimplerEvaluationError("--tasks contains a duplicate task")
    unknown = [key for key in keys if key not in _TASKS_BY_KEY]
    if unknown:
        raise SimplerEvaluationError(
            f"--tasks contains unknown tasks {unknown}; expected {sorted(_TASKS_BY_KEY)}"
        )
    return tuple(_TASKS_BY_KEY[key] for key in keys)


def _quaternion_wxyz_to_matrix(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise SimplerEvaluationError(f"Expected a finite wxyz quaternion, found {quaternion}")
    norm = np.linalg.norm(quaternion)
    if norm <= np.finfo(np.float64).eps:
        raise SimplerEvaluationError("Quaternion norm must be positive")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_xyz_euler(matrix: np.ndarray) -> np.ndarray:
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-7:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(float(-matrix[1, 2]), float(matrix[1, 1]))
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def environment_to_bridge_proprio(environment: Any) -> np.ndarray:
    try:
        base_pose = environment.agent.robot.pose
        tcp_pose = environment.tcp.pose
        relative = base_pose.inv() * tcp_pose
        position = np.asarray(relative.p, dtype=np.float32)
        euler = _matrix_to_xyz_euler(_quaternion_wxyz_to_matrix(relative.q))
        closedness = float(environment.agent.get_gripper_closedness())
    except SimplerEvaluationError:
        raise
    except Exception as error:
        raise SimplerEvaluationError(f"Could not construct WidowX proprioception: {error}") from error
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise SimplerEvaluationError(f"WidowX TCP position is invalid: {position}")
    if not math.isfinite(closedness):
        raise SimplerEvaluationError("WidowX gripper closedness is not finite")
    result = np.concatenate(
        [position, euler, np.zeros(1, dtype=np.float32), [1.0 - closedness]]
    ).astype(np.float32)
    if result.shape != (8,) or not np.all(np.isfinite(result)):
        raise SimplerEvaluationError("WidowX proprioception conversion produced invalid values")
    return result


def _xyz_euler_to_matrix(value: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.asarray(value, dtype=np.float64)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def _matrix_to_rotation_vector(matrix: np.ndarray) -> np.ndarray:
    angle = math.acos(float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1.0e-7:
        return np.zeros(3, dtype=np.float32)
    if math.pi - angle < 1.0e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        norm = np.linalg.norm(axis)
        if norm <= np.finfo(np.float64).eps:
            raise SimplerEvaluationError("Could not convert a pi rotation to axis-angle")
        return (axis / norm * angle).astype(np.float32)
    axis = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return (axis * angle).astype(np.float32)


def bridge_actions_to_simpler(
    value: Any,
    *,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim < 1 or actions.shape[-1] != 7:
        raise SimplerEvaluationError(
            f"Bridge actions must have last dimension 7, found {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise SimplerEvaluationError("Bridge actions contain NaN or infinite values")
    if not math.isfinite(gripper_threshold):
        raise SimplerEvaluationError("Gripper threshold must be finite")
    result = np.empty_like(actions)
    result[..., :3] = actions[..., :3]
    flattened = actions.reshape(-1, 7)
    converted = result.reshape(-1, 7)
    for index, action in enumerate(flattened):
        converted[index, 3:6] = _matrix_to_rotation_vector(
            _xyz_euler_to_matrix(action[3:6])
        )
    result[..., 6] = np.where(
        actions[..., 6] > gripper_threshold,
        1.0,
        -1.0,
    )
    return result


def image_from_simpler_observation(observation: Mapping[str, Any]) -> np.ndarray:
    try:
        image = np.asarray(observation["image"]["3rd_view_camera"]["rgb"])
    except (KeyError, TypeError) as error:
        raise SimplerEvaluationError(
            "SimplerEnv observation is missing image.3rd_view_camera.rgb"
        ) from error
    if image.ndim != 3 or image.shape[2] != 3:
        raise SimplerEvaluationError(f"Expected an HWC RGB image, found {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise SimplerEvaluationError("SimplerEnv RGB image contains non-finite values")
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    elif np.issubdtype(image.dtype, np.integer):
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        raise SimplerEvaluationError(f"Unsupported SimplerEnv RGB dtype: {image.dtype}")
    return image.copy()


def _step_environment(
    environment: Any,
    action: np.ndarray,
) -> tuple[Any, bool, bool, dict[str, Any]]:
    try:
        result = environment.step(action)
    except Exception as error:
        raise SimplerInfrastructureError(
            f"SimplerEnv step failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(result, tuple) or len(result) not in {4, 5}:
        raise SimplerEvaluationError("SimplerEnv returned an invalid step tuple")
    if len(result) == 5:
        observation, _, terminated, truncated, info = result
    else:
        observation, _, terminated, info = result
        truncated = False
    if not isinstance(info, Mapping):
        raise SimplerEvaluationError("SimplerEnv step info must be a mapping")
    return observation, bool(terminated), bool(truncated), dict(info)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_simpler_episode(
    *,
    task: SimplerTaskSpec,
    object_episode_id: int,
    policy_seed: int,
    policy: SimplerPolicyAdapter,
    environment: Any,
    generator: Any,
    action_horizon: int,
    max_steps: int,
    capture_video: bool,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run one official WidowX episode through a model-specific adapter."""

    if not 1 <= action_horizon <= 8:
        raise SimplerEvaluationError("action_horizon must be in [1, 8]")
    if max_steps <= 0:
        raise SimplerEvaluationError("max_steps must be positive")
    try:
        observation, _ = environment.reset(
            options={
                "robot_init_options": {
                    "init_xy": np.asarray(task.robot_init_xy, dtype=np.float64),
                    "init_rot_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                },
                "obj_init_options": {"episode_id": int(object_episode_id)},
            }
        )
    except Exception as error:
        raise SimplerInfrastructureError(
            f"SimplerEnv reset failed for task {task.key}: {type(error).__name__}: {error}"
        ) from error

    frames = [image_from_simpler_observation(observation)] if capture_video else []
    steps = 0
    success = False
    truncated = False
    last_info: dict[str, Any] = {}
    while steps < max_steps and not success and not truncated:
        source_image = image_from_simpler_observation(observation)
        proprio = environment_to_bridge_proprio(environment)
        prepared = policy.prepare_observation(source_image, proprio, task.instruction)
        predicted = policy.predict_actions(prepared, generator=generator)
        if hasattr(predicted, "detach"):
            predicted = predicted.detach().float().cpu().numpy()
        actions = np.asarray(predicted, dtype=np.float32)
        if actions.shape != (1, 8, 7):
            policy_name = str(getattr(policy, "policy_name", "policy"))
            raise SimplerEvaluationError(
                f"{policy_name} returned actions with shape {actions.shape}; expected (1, 8, 7)"
            )
        simpler_actions = bridge_actions_to_simpler(
            actions[0],
            gripper_threshold=float(getattr(policy, "gripper_threshold", 0.5)),
        )
        for action in simpler_actions[: min(action_horizon, max_steps - steps)]:
            observation, success, truncated, last_info = _step_environment(
                environment, action
            )
            steps += 1
            if capture_video:
                frames.append(image_from_simpler_observation(observation))
            if success or truncated:
                break

    episode_stats = last_info.get("episode_stats", {})
    if not isinstance(episode_stats, Mapping):
        episode_stats = {}
    termination = "success" if success else "truncated" if truncated else "max_steps"
    return (
        {
            "task": task.key,
            "instruction": task.instruction,
            "seed": int(policy_seed),
            "policy_seed": int(policy_seed),
            "object_episode_id": int(object_episode_id),
            "success": bool(success),
            "steps": int(steps),
            "termination": termination,
            "episode_stats": _json_compatible(dict(episode_stats)),
        },
        frames,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(json.dumps(value, sort_keys=True) + "\n" for value in values)
    _atomic_write_text(path, content)


def _summary(
    episodes: Sequence[Mapping[str, Any]],
    tasks: Sequence[SimplerTaskSpec],
) -> dict[str, Any]:
    successes = sum(bool(episode["success"]) for episode in episodes)
    by_task = []
    for task in tasks:
        selected = [episode for episode in episodes if episode["task"] == task.key]
        task_successes = sum(bool(episode["success"]) for episode in selected)
        by_task.append(
            {
                "task": task.key,
                "completed_episodes": len(selected),
                "successes": task_successes,
                "failures": len(selected) - task_successes,
                "success_rate": task_successes / len(selected) if selected else None,
            }
        )
    return {
        "completed_episodes": len(episodes),
        "successes": successes,
        "failures": len(episodes) - successes,
        "success_rate": successes / len(episodes) if episodes else None,
        "by_task": by_task,
    }


def _validate_settings(settings: SimplerRunSettings) -> None:
    if not settings.tasks:
        raise SimplerEvaluationError("At least one SimplerEnv task is required")
    if not settings.policy_seeds or not settings.object_episode_ids:
        raise SimplerEvaluationError("policy_seeds and object_episode_ids must be non-empty")
    if not 1 <= settings.action_horizon <= 8:
        raise SimplerEvaluationError("action_horizon must be in [1, 8]")
    if settings.max_steps is not None and settings.max_steps <= 0:
        raise SimplerEvaluationError("max_steps must be positive when provided")
    if settings.video_fps <= 0:
        raise SimplerEvaluationError("video_fps must be positive")
    if len(settings.policy_seeds) != len(set(settings.policy_seeds)):
        raise SimplerEvaluationError("policy_seeds must be unique")
    if len(settings.object_episode_ids) != len(set(settings.object_episode_ids)):
        raise SimplerEvaluationError("object_episode_ids must be unique")
    if any(episode_id not in OBJECT_EPISODE_IDS for episode_id in settings.object_episode_ids):
        raise SimplerEvaluationError("object_episode_ids must be in [0, 23]")


def _check_output_targets(settings: SimplerRunSettings) -> None:
    targets = [
        settings.output_dir / "results.json",
        settings.output_dir / "episodes.jsonl",
        settings.output_dir / "episodes.partial.jsonl",
        settings.output_dir / "failure.json",
    ]
    if settings.save_videos_path is not None:
        targets.extend(
            settings.save_videos_path
            / task.key
            / f"seed-{seed}"
            / f"episode-{episode_id:02d}_{outcome}.mp4"
            for task in settings.tasks
            for seed in settings.policy_seeds
            for episode_id in settings.object_episode_ids
            for outcome in ("success", "failure")
        )
    if not settings.overwrite and any(path.exists() for path in targets):
        raise SimplerEvaluationError(
            "Evaluation output or video already exists; use --overwrite"
        )


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    try:
        import imageio.v3 as imageio
    except ImportError as error:
        raise SimplerEvaluationError(
            "imageio and imageio-ffmpeg are required when --save-videos-path is set"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.imwrite(path, np.stack(frames), fps=fps)
    except Exception as error:
        raise SimplerEvaluationError(f"Could not write evaluation video {path}: {error}") from error


def _protocol(
    settings: SimplerRunSettings,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = {
        "tasks": [task.key for task in settings.tasks],
        "policy_seeds": list(settings.policy_seeds),
        "object_episode_ids": list(settings.object_episode_ids),
        "control_frequency_hz": 5,
        "simulation_frequency_hz": 500,
        "action_horizon": settings.action_horizon,
        "max_steps_override": settings.max_steps,
    }
    overlap = sorted(set(protocol).intersection(metadata))
    if overlap:
        raise SimplerEvaluationError(
            f"Policy protocol metadata cannot replace shared fields: {overlap}"
        )
    protocol.update(dict(metadata))
    return protocol


def evaluate_simpler_policy(
    settings: SimplerRunSettings,
    *,
    checkpoint: Mapping[str, Any],
    policy: SimplerPolicyAdapter,
    environment_factory: Callable[[SimplerTaskSpec], Any],
    source_versions: Mapping[str, Any],
    route: str,
    protocol_metadata: Mapping[str, Any],
    video_writer: Callable[[Path, Sequence[np.ndarray], int], None] = _write_video,
) -> dict[str, Any]:
    """Execute the shared 3-seed/24-object protocol for one policy adapter."""

    _validate_settings(settings)
    _check_output_targets(settings)
    if not str(route).strip():
        raise SimplerEvaluationError("Evaluation route must be non-empty")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    task_errors: list[dict[str, Any]] = []
    started = time.monotonic()
    partial_path = settings.output_dir / "episodes.partial.jsonl"
    videos: list[str] = []
    try:
        for task in settings.tasks:
            task_failed = False
            for policy_seed in settings.policy_seeds:
                generator = policy.make_generator(policy_seed)
                for object_episode_id in settings.object_episode_ids:
                    try:
                        environment = environment_factory(task)
                        try:
                            episode, frames = run_simpler_episode(
                                task=task,
                                object_episode_id=object_episode_id,
                                policy_seed=policy_seed,
                                policy=policy,
                                environment=environment,
                                generator=generator,
                                action_horizon=settings.action_horizon,
                                max_steps=settings.max_steps or task.max_steps,
                                capture_video=settings.save_videos_path is not None,
                            )
                        finally:
                            environment.close()
                    except (SimplerEvaluationError, SimplerInfrastructureError):
                        raise
                    except Exception as error:
                        task_errors.append(
                            {
                                "task": task.key,
                                "policy_seed": int(policy_seed),
                                "object_episode_id": int(object_episode_id),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        task_failed = True
                        break
                    episodes.append(episode)
                    if settings.save_videos_path is not None:
                        outcome = "success" if episode["success"] else "failure"
                        video_path = (
                            settings.save_videos_path
                            / task.key
                            / f"seed-{policy_seed}"
                            / f"episode-{object_episode_id:02d}_{outcome}.mp4"
                        )
                        video_writer(video_path, frames, settings.video_fps)
                        videos.append(str(video_path))
                    _atomic_write_jsonl(partial_path, episodes)
                if task_failed:
                    break
    except Exception as error:
        if isinstance(error, SimplerInfrastructureError):
            exit_code = 3
        elif isinstance(error, SimplerEvaluationError):
            exit_code = 2
        else:
            exit_code = 1
        failure = {
            "schema_version": 1,
            "status": "failed",
            "route": route,
            "exit_code": exit_code,
            "error": f"{type(error).__name__}: {error}",
            "summary": _summary(episodes, settings.tasks),
        }
        _atomic_write_json(settings.output_dir / "failure.json", failure)
        raise

    report = {
        "schema_version": 1,
        "status": "completed_with_errors" if task_errors else "complete",
        "route": route,
        "checkpoint": dict(checkpoint),
        "protocol": _protocol(settings, protocol_metadata),
        "summary": _summary(episodes, settings.tasks),
        "task_errors": task_errors,
        "runtime": {
            "device": settings.device,
            "python": platform.python_version(),
            "elapsed_seconds": time.monotonic() - started,
            **dict(source_versions),
        },
        "videos": videos,
    }
    _atomic_write_jsonl(settings.output_dir / "episodes.jsonl", episodes)
    _atomic_write_json(settings.output_dir / "results.json", report)
    if task_errors:
        _atomic_write_json(settings.output_dir / "failure.json", report)
    else:
        (settings.output_dir / "failure.json").unlink(missing_ok=True)
    partial_path.unlink(missing_ok=True)
    return report


def _git_head(path: Path, description: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SimplerEvaluationError(
            f"{description} is not an initialized git checkout: {path}"
        ) from error
    return result.stdout.strip()


def _require_clean_git_checkout(path: Path, description: str) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SimplerEvaluationError(
            f"Could not inspect {description} checkout state: {path}"
        ) from error
    if result.stdout.strip():
        raise SimplerEvaluationError(
            f"{description} has local modifications; restore the pinned source before evaluation"
        )


def validate_simpler_source(
    simpler_root: str | Path,
    *,
    expected_simpler_commit: str = SIMPLER_ENV_COMMIT,
    expected_maniskill_commit: str = MANISKILL2_REAL2SIM_COMMIT,
) -> dict[str, str]:
    root = Path(simpler_root).expanduser().resolve()
    maniskill_root = root / "ManiSkill2_real2sim"
    required_directories = [
        root / "simpler_env",
        maniskill_root / "mani_skill2_real2sim",
    ]
    missing_directories = [str(path) for path in required_directories if not path.is_dir()]
    if missing_directories:
        raise SimplerEvaluationError(
            "SimplerEnv or its ManiSkill2_real2sim submodule is not initialized; "
            f"missing={missing_directories}"
        )
    simpler_commit = _git_head(root, "SimplerEnv")
    maniskill_commit = _git_head(maniskill_root, "ManiSkill2_real2sim")
    _require_clean_git_checkout(root, "SimplerEnv")
    _require_clean_git_checkout(maniskill_root, "ManiSkill2_real2sim")
    if not simpler_commit.startswith(expected_simpler_commit):
        raise SimplerEvaluationError(
            f"SimplerEnv must be pinned to {expected_simpler_commit}; found {simpler_commit}"
        )
    if not maniskill_commit.startswith(expected_maniskill_commit):
        raise SimplerEvaluationError(
            "ManiSkill2_real2sim must be pinned to "
            f"{expected_maniskill_commit}; found {maniskill_commit}"
        )
    missing_assets = [
        str(root / relative)
        for relative in sorted({task.overlay_relative_path for task in SIMPLER_TASKS})
        if not (root / relative).is_file()
    ]
    if missing_assets:
        raise SimplerEvaluationError(
            f"SimplerEnv visual-matching asset is missing: {missing_assets}"
        )
    return {
        "simpler_env_commit": simpler_commit,
        "maniskill2_real2sim_commit": maniskill_commit,
    }


def default_simpler_root() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "SimplerEnv"


def create_simpler_environment(
    task: SimplerTaskSpec,
    *,
    simpler_root: str | Path | None = None,
    builder: Callable[..., Any] | None = None,
) -> Any:
    root = Path(simpler_root or default_simpler_root()).expanduser().resolve()
    if builder is None:
        try:
            from simpler_env.utils.env.env_builder import build_maniskill2_env
        except ImportError as error:
            raise SimplerEvaluationError(
                "Could not import SimplerEnv; run through a scripts/evaluate_simpler_*.sh launcher"
            ) from error
        builder = build_maniskill2_env
    overlay = (root / task.overlay_relative_path).resolve()
    if not overlay.is_file():
        raise SimplerEvaluationError(f"SimplerEnv visual-matching asset is missing: {overlay}")
    try:
        return builder(
            task.env_name,
            obs_mode="rgbd",
            robot=task.robot,
            sim_freq=500,
            control_mode=CONTROL_MODE,
            control_freq=5,
            max_episode_steps=task.max_steps,
            scene_name=task.scene_name,
            camera_cfgs={"add_segmentation": True},
            rgb_overlay_path=str(overlay),
        )
    except SimplerEvaluationError:
        raise
    except Exception as error:
        raise SimplerInfrastructureError(
            f"Could not create SimplerEnv task {task.key}: {type(error).__name__}: {error}"
        ) from error


def run_simpler_preflight(
    settings: SimplerRunSettings,
    *,
    checkpoint: Mapping[str, Any],
    policy: SimplerPolicyAdapter,
    environment_factory: Callable[[SimplerTaskSpec], Any],
    source_versions: Mapping[str, Any],
    package_versions: Mapping[str, str],
    route: str,
) -> dict[str, Any]:
    """Validate every environment and perform one model inference."""

    _validate_settings(settings)
    output_path = settings.output_dir / "preflight.json"
    protected_outputs = [output_path, settings.output_dir / "failure.json"]
    if not settings.overwrite and any(path.exists() for path in protected_outputs):
        raise SimplerEvaluationError(
            "Preflight output or failure report already exists; use --overwrite"
        )
    if not str(route).strip():
        raise SimplerEvaluationError("Preflight route must be non-empty")

    environments = []
    inference: dict[str, Any] | None = None
    generator = policy.make_generator(0)
    for task in settings.tasks:
        environment = environment_factory(task)
        try:
            try:
                observation, _ = environment.reset(
                    options={
                        "robot_init_options": {
                            "init_xy": np.asarray(task.robot_init_xy, dtype=np.float64),
                            "init_rot_quat": np.asarray(
                                [0.0, 0.0, 0.0, 1.0], dtype=np.float64
                            ),
                        },
                        "obj_init_options": {"episode_id": 0},
                    }
                )
            except Exception as error:
                raise SimplerInfrastructureError(
                    f"SimplerEnv preflight reset failed for task {task.key}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            source_image = image_from_simpler_observation(observation)
            proprio = environment_to_bridge_proprio(environment)
            prepared = policy.prepare_observation(
                source_image,
                proprio,
                task.instruction,
            )
            input_metadata = dict(policy.describe_observation(prepared))
            if inference is None:
                predicted = policy.predict_actions(prepared, generator=generator)
                if hasattr(predicted, "detach"):
                    predicted = predicted.detach().float().cpu().numpy()
                predicted_actions = np.asarray(predicted, dtype=np.float32)
                if predicted_actions.shape != (1, 8, 7):
                    policy_name = str(getattr(policy, "policy_name", "policy"))
                    raise SimplerEvaluationError(
                        f"{policy_name} preflight returned actions with shape "
                        f"{predicted_actions.shape}; expected (1, 8, 7)"
                    )
                bridge_actions_to_simpler(
                    predicted_actions,
                    gripper_threshold=float(
                        getattr(policy, "gripper_threshold", 0.5)
                    ),
                )
                inference = {
                    "task": task.key,
                    "action_shape": list(predicted_actions.shape),
                }
            safe_action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            _, terminated, truncated, info = _step_environment(environment, safe_action)
            environments.append(
                {
                    "task": task.key,
                    "image_shape": list(source_image.shape),
                    **input_metadata,
                    "proprio_shape": list(proprio.shape),
                    "safe_step_terminated": terminated,
                    "safe_step_truncated": truncated,
                    "episode_stats_present": "episode_stats" in info,
                }
            )
        finally:
            environment.close()

    report = {
        "schema_version": 1,
        "status": "passed",
        "route": route,
        "checkpoint": dict(checkpoint),
        "protocol": {
            "tasks": [task.key for task in settings.tasks],
            "control_frequency_hz": 5,
            "simulation_frequency_hz": 500,
            "control_mode": CONTROL_MODE,
        },
        "environments": environments,
        "model_inference": inference,
        "runtime": {
            "device": settings.device,
            "python": platform.python_version(),
            "package_versions": dict(package_versions),
            **dict(source_versions),
        },
    }
    _atomic_write_json(output_path, report)
    (settings.output_dir / "failure.json").unlink(missing_ok=True)
    return report
