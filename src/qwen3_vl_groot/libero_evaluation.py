from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from octo_small_libero.evaluation import (
    ACTION_DIM,
    ACTION_HORIZON,
    DEFAULT_TASK_NAME,
    EVALUATION_SEEDS,
    LIBERO_COMMIT,
    MUJOCO_COMPATIBILITY,
    MUJOCO_VERSION,
    PROPRIO_DIM,
    EvaluationError,
    SimulationInfrastructureError,
    UnrecoverableSimulationShutdownError,
    _atomic_write_json,
    _atomic_write_jsonl,
    _check_output_targets,
    _frame_from_observation,
    _make_offscreen_environment,
    _multiprocessing_start_method,
    _package_versions,
    _save_videos,
    configure_evaluation_multiprocessing,
    configure_libero,
    configure_transformers_offline,
    make_vector_environment_with_backoff,
    observation_batch_to_list,
    observation_to_proprio,
    resize_flipped_rgb,
    resolve_libero_task,
    rollout_action_chunks,
    settle_vector_environment,
    validate_settings,
    validate_simulation_dependencies,
)


QWEN_COMPACT_CHECKPOINT_FORMAT = "qwen3-vl-groot-bridge-compact-v1"


@dataclass(frozen=True)
class QwenEvaluationSettings:
    checkpoint: Path
    model_path: Path | None = None
    output_dir: Path = Path("outputs/qwen_libero_eval")
    task_name: str = DEFAULT_TASK_NAME
    episodes: int = 150
    num_envs: int = 50
    auto_reduce_num_envs: bool = True
    max_steps: int = 960
    settle_steps: int = 20
    seeds: tuple[int, int, int] = EVALUATION_SEEDS
    device: str = "cuda:0"
    denoising_steps: int = 4
    policy_batch_size: int = 4
    record_videos: int = 0
    video_fps: int = 30
    libero_root: Path | None = None
    libero_config_path: Path | None = None
    overwrite: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"{description} must contain a JSON object: {path}")
    return value


def _require_integer(mapping: Mapping[str, Any], key: str, expected: int) -> None:
    try:
        actual = int(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"Qwen checkpoint is missing integer data.{key}") from error
    if actual != expected:
        raise EvaluationError(
            f"Qwen LIBERO checkpoint requires data.{key}={expected}, found {actual}"
        )


def _validate_base_model(path: Path, *, expected_family: str) -> None:
    from .modeling import ModelContractError, inspect_qwen_config

    try:
        inspect_qwen_config(path, expected_family=expected_family)
    except ModelContractError as error:
        raise EvaluationError(str(error)) from error


@dataclass(frozen=True)
class QwenCheckpointSpec:
    requested_path: Path
    weights_path: Path
    policy_config_path: Path
    normalization_path: Path
    base_model_path: Path
    config: dict[str, Any]
    global_step: int
    weights_sha256: str
    policy_config_sha256: str
    normalization_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_path": str(self.requested_path),
            "weights_path": str(self.weights_path),
            "policy_config_path": str(self.policy_config_path),
            "normalization_path": str(self.normalization_path),
            "base_model_path": str(self.base_model_path),
            "global_step": self.global_step,
            "weights_sha256": self.weights_sha256,
            "policy_config_sha256": self.policy_config_sha256,
            "normalization_sha256": self.normalization_sha256,
        }


def resolve_qwen_checkpoint(
    checkpoint: str | Path,
    *,
    model_path: str | Path | None = None,
) -> QwenCheckpointSpec:
    requested = Path(checkpoint).expanduser().resolve()
    if not requested.is_dir() or re.fullmatch(r"step-\d{8}", requested.name) is None:
        raise EvaluationError(
            "--checkpoint must name a concrete step-XXXXXXXX checkpoint directory; "
            "training roots and best/latest aliases are not supported"
        )

    policy_config_path = requested / "policy_config.json"
    normalization_path = requested / "normalization.json"
    weights_path = requested / "adapter_model.safetensors"
    missing = [
        str(path)
        for path in (policy_config_path, normalization_path, weights_path)
        if not path.is_file()
    ]
    if missing:
        raise EvaluationError(f"Qwen compact checkpoint is incomplete; missing={missing}")

    manifest = _read_json(policy_config_path, "Qwen policy config")
    if manifest.get("format") != QWEN_COMPACT_CHECKPOINT_FORMAT:
        raise EvaluationError(
            f"Unsupported Qwen inference checkpoint format: {manifest.get('format')!r}"
        )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise EvaluationError("Qwen policy config is missing its config object")
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise EvaluationError("Qwen policy config is missing its data section")
    _require_integer(data, "state_dim", PROPRIO_DIM)
    _require_integer(data, "action_dim", ACTION_DIM)
    _require_integer(data, "action_horizon", ACTION_HORIZON)
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        raise EvaluationError("Qwen policy config model section must be a mapping")
    backbone_family = str(model_config.get("backbone_family", "qwen3_vl"))
    from .config import ConfigError, validated_lora_target_modules

    try:
        validated_lora_target_modules(dict(model_config))
    except ConfigError as error:
        raise EvaluationError(str(error)) from error
    for key in ("train_crop_size", "output_image_size"):
        try:
            size = int(data[key])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"Qwen checkpoint is missing integer data.{key}") from error
        if size <= 0:
            raise EvaluationError(f"Qwen checkpoint requires positive data.{key}")

    normalization = _read_json(normalization_path, "Qwen normalization metadata")
    for key, expected in (
        ("state_q01", PROPRIO_DIM),
        ("state_q99", PROPRIO_DIM),
        ("action_q01", ACTION_DIM),
        ("action_q99", ACTION_DIM),
    ):
        try:
            value = np.asarray(normalization[key], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"Qwen normalization is missing finite {key}") from error
        if value.shape != (expected,) or not np.all(np.isfinite(value)):
            raise EvaluationError(
                f"Qwen normalization {key} must be a finite array with shape {(expected,)}"
            )

    raw_base_model = model_path if model_path is not None else manifest.get("base_model")
    if raw_base_model is None:
        raise EvaluationError("Qwen policy config has no base_model; pass --model-path")
    base_model = Path(raw_base_model).expanduser().resolve()
    if not base_model.is_dir():
        raise EvaluationError(f"Qwen base model does not exist: {base_model}")
    _validate_base_model(base_model, expected_family=backbone_family)
    try:
        global_step = int(manifest["global_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError("Qwen policy config has no valid global_step") from error
    if requested.name != f"step-{global_step:08d}":
        raise EvaluationError(
            f"Checkpoint directory {requested.name} does not match global_step={global_step}"
        )

    return QwenCheckpointSpec(
        requested_path=requested,
        weights_path=weights_path,
        policy_config_path=policy_config_path,
        normalization_path=normalization_path,
        base_model_path=base_model,
        config=config,
        global_step=global_step,
        weights_sha256=_sha256(weights_path),
        policy_config_sha256=_sha256(policy_config_path),
        normalization_sha256=_sha256(normalization_path),
    )


def build_qwen_observation_batch(
    observations: Any,
    *,
    data_config: Mapping[str, Any],
) -> tuple[list[np.ndarray], np.ndarray]:
    try:
        crop_size = int(data_config["train_crop_size"])
        output_size = int(data_config["output_image_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(
            "Qwen image preprocessing requires train_crop_size and output_image_size"
        ) from error
    if crop_size <= 0 or output_size <= 0:
        raise EvaluationError("Qwen image crop and output sizes must be positive")
    values = observation_batch_to_list(observations)
    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for observation in values:
        if "agentview_image" not in observation:
            raise EvaluationError("LIBERO observation is missing agentview_image")
        source = np.asarray(observation["agentview_image"])
        if source.ndim != 3 or source.shape[2] != 3:
            raise EvaluationError(f"Expected an HWC RGB image, found {source.shape}")
        height, width = source.shape[:2]
        if min(height, width) < crop_size:
            raise EvaluationError(
                f"LIBERO image {width}x{height} is smaller than crop {crop_size}"
            )
        flipped = resize_flipped_rgb(source, (height, width))
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        cropped = flipped[top : top + crop_size, left : left + crop_size]
        try:
            from PIL import Image
        except ImportError as error:
            raise EvaluationError("Pillow is required for Qwen image preprocessing") from error
        image = Image.fromarray(cropped)
        if image.size != (output_size, output_size):
            image = image.resize(
                (output_size, output_size),
                resample=Image.Resampling.BICUBIC,
            )
        images.append(np.asarray(image, dtype=np.uint8).copy())
        states.append(observation_to_proprio(observation))
    return images, np.stack(states).astype(np.float32, copy=False)


def qwen_actions_to_environment(value: Any) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim < 1 or actions.shape[-1] != ACTION_DIM:
        raise EvaluationError(
            f"Qwen actions must end in dimension {ACTION_DIM}, found {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise EvaluationError("Qwen checkpoint returned NaN or infinite actions")
    result = actions.copy()
    result[..., 6] = 1.0 - 2.0 * result[..., 6]
    return result


class QwenActionTransform:
    @staticmethod
    def actions_to_environment(value: Any) -> np.ndarray:
        return qwen_actions_to_environment(value)


class QwenLiberoPolicy:
    def __init__(
        self,
        *,
        checkpoint: QwenCheckpointSpec,
        policy: Any,
        device: str,
        denoising_steps: int,
        policy_batch_size: int,
    ):
        if denoising_steps <= 0:
            raise EvaluationError("denoising_steps must be positive")
        if policy_batch_size <= 0:
            raise EvaluationError("policy_batch_size must be positive")
        self.checkpoint = checkpoint
        self.policy = policy
        self.device = device
        self.denoising_steps = int(denoising_steps)
        self.policy_batch_size = int(policy_batch_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: QwenCheckpointSpec,
        *,
        device: str,
        denoising_steps: int,
        policy_batch_size: int,
    ) -> "QwenLiberoPolicy":
        from .inference import BridgePolicy

        try:
            policy = BridgePolicy.from_pretrained(
                checkpoint.requested_path,
                model_path=checkpoint.base_model_path,
                device=device,
            )
        except Exception as error:
            raise EvaluationError(
                f"Could not load Qwen compact LoRA checkpoint: {error}"
            ) from error
        return cls(
            checkpoint=checkpoint,
            policy=policy,
            device=device,
            denoising_steps=denoising_steps,
            policy_batch_size=policy_batch_size,
        )

    def make_generator(self, seed: int) -> Any:
        import torch

        return torch.Generator(device=self.device).manual_seed(int(seed))

    def predict_action_chunk(
        self,
        observations: Any,
        instruction: str,
        *,
        generator: Any,
    ) -> np.ndarray:
        values = observation_batch_to_list(observations)
        chunks: list[np.ndarray] = []
        for start in range(0, len(values), self.policy_batch_size):
            current = values[start : start + self.policy_batch_size]
            images, states = build_qwen_observation_batch(
                current,
                data_config=self.checkpoint.config["data"],
            )
            predicted = self.policy.predict_actions(
                images,
                states,
                [instruction] * len(current),
                denoising_steps=self.denoising_steps,
                generator=generator,
            )
            if hasattr(predicted, "detach"):
                predicted = predicted.detach().float().cpu().numpy()
            actions = np.asarray(predicted, dtype=np.float32)
            expected = (len(current), ACTION_HORIZON, ACTION_DIM)
            if actions.shape != expected:
                raise EvaluationError(
                    f"Qwen checkpoint returned actions with shape {actions.shape}; "
                    f"expected {expected}"
                )
            chunks.append(qwen_actions_to_environment(actions))
        return np.concatenate(chunks, axis=0)


def _datasets_path(checkpoint: QwenCheckpointSpec) -> Path:
    try:
        value = checkpoint.config["paths"]["lerobot"]
    except (KeyError, TypeError) as error:
        raise EvaluationError(
            "Qwen policy config is missing paths.lerobot for LIBERO configuration"
        ) from error
    return Path(value).expanduser().resolve()


def _validate_qwen_settings(settings: QwenEvaluationSettings) -> None:
    validate_settings(settings)
    if settings.denoising_steps <= 0:
        raise EvaluationError("denoising_steps must be positive")
    if settings.policy_batch_size <= 0:
        raise EvaluationError("policy_batch_size must be positive")


def _make_qwen_report(
    *,
    status: str,
    settings: QwenEvaluationSettings,
    checkpoint: QwenCheckpointSpec,
    task: Any,
    episodes: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    videos: Sequence[str],
    libero_commit: str,
    environment_batch_sizes: Sequence[int] = (),
    error: str | None = None,
) -> dict[str, Any]:
    successes = sum(bool(episode["success"]) for episode in episodes)
    summaries_by_seed = []
    for seed in settings.seeds:
        seeded = [episode for episode in episodes if episode.get("seed") == seed]
        seeded_successes = sum(bool(episode["success"]) for episode in seeded)
        summaries_by_seed.append(
            {
                "seed": seed,
                "completed_episodes": len(seeded),
                "successes": seeded_successes,
                "failures": len(seeded) - seeded_successes,
                "success_rate": seeded_successes / len(seeded) if seeded else None,
            }
        )
    report = {
        "schema_version": 3,
        "status": status,
        "route": "qwen3-vl-groot-libero-checkpoint-eval",
        "checkpoint": checkpoint.as_dict(),
        "task": {
            "suite": "libero_10",
            "task_id": task.task_id,
            "name": task.name,
            "language": task.language,
            "bddl_file": str(task.bddl_file),
            "init_states_file": str(task.init_states_file),
            "init_states_sha256": task.init_states_sha256,
            "init_states_git_blob": task.init_states_git_blob,
        },
        "protocol": {
            "episodes": settings.episodes,
            "episodes_per_seed": settings.episodes // len(settings.seeds),
            "seeds": list(settings.seeds),
            "num_envs": settings.num_envs,
            "auto_reduce_num_envs": settings.auto_reduce_num_envs,
            "max_steps": settings.max_steps,
            "settle_steps": settings.settle_steps,
            "action_horizon": ACTION_HORIZON,
            "denoising_steps": settings.denoising_steps,
            "policy_batch_size": settings.policy_batch_size,
        },
        "summary": {
            "completed_episodes": len(episodes),
            "successes": successes,
            "failures": len(episodes) - successes,
            "success_rate": successes / len(episodes) if episodes else None,
            "by_seed": summaries_by_seed,
        },
        "runtime": {
            "device": settings.device,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "libero_expected_commit": LIBERO_COMMIT,
            "libero_commit": libero_commit,
            "mujoco_expected_version": MUJOCO_VERSION,
            "simulation_compatibility": MUJOCO_COMPATIBILITY,
            "libero_multiprocessing_start_method": _multiprocessing_start_method(),
            "effective_num_envs": (
                max(environment_batch_sizes) if environment_batch_sizes else None
            ),
            "environment_batch_sizes": list(environment_batch_sizes),
            "elapsed_seconds": elapsed_seconds,
        },
        "videos": list(videos),
    }
    if error is not None:
        report["error"] = error
    return report


def run_qwen_preflight(
    settings: QwenEvaluationSettings,
    *,
    checkpoint: QwenCheckpointSpec,
    task: Any,
    policy: QwenLiberoPolicy,
    libero_commit: str,
) -> dict[str, Any]:
    try:
        environment = _make_offscreen_environment(
            {
                "bddl_file_name": str(task.bddl_file),
                "camera_heights": 128,
                "camera_widths": 128,
            }
        )
    except Exception as error:
        raise SimulationInfrastructureError(
            "Could not create the LIBERO offscreen environment "
            f"(num_envs=1, MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}): "
            f"{type(error).__name__}: {error}"
        ) from error
    try:
        environment.reset()
        environment.seed(settings.seeds[0])
        observation = environment.set_init_state(task.init_states[0])
        dummy = np.zeros((1, ACTION_DIM), dtype=np.float32)
        dummy[:, -1] = 1.0
        raw_dummy = qwen_actions_to_environment(dummy)[0]
        for _ in range(settings.settle_steps):
            observation = environment.step(raw_dummy)[0]
        actions = policy.predict_action_chunk(
            [observation],
            task.language,
            generator=policy.make_generator(settings.seeds[0]),
        )
    finally:
        environment.close()
    report = {
        "schema_version": 3,
        "status": "passed",
        "route": "qwen3-vl-groot-libero-checkpoint-eval",
        "checkpoint": checkpoint.as_dict(),
        "task": {
            "task_id": task.task_id,
            "name": task.name,
            "init_states": len(task.init_states),
            "init_states_file": str(task.init_states_file),
            "init_states_sha256": task.init_states_sha256,
            "init_states_git_blob": task.init_states_git_blob,
        },
        "protocol": {"seeds": list(settings.seeds)},
        "sample_action_shape": list(actions.shape),
        "runtime": {
            "device": settings.device,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "packages": _package_versions(),
            "libero_expected_commit": LIBERO_COMMIT,
            "libero_commit": libero_commit,
            "mujoco_expected_version": MUJOCO_VERSION,
            "simulation_compatibility": MUJOCO_COMPATIBILITY,
            "libero_multiprocessing_start_method": _multiprocessing_start_method(),
        },
    }
    _atomic_write_json(settings.output_dir / "preflight.json", report)
    return report


def evaluate_qwen_checkpoint(
    settings: QwenEvaluationSettings,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    configure_evaluation_multiprocessing()
    _validate_qwen_settings(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    configure_transformers_offline(settings.output_dir)
    validate_simulation_dependencies()
    checkpoint = resolve_qwen_checkpoint(
        settings.checkpoint,
        model_path=settings.model_path,
    )
    datasets_path = _datasets_path(checkpoint)
    libero_root, libero_config_file, libero_commit = configure_libero(
        output_dir=settings.output_dir,
        datasets_path=datasets_path,
        source_root=settings.libero_root,
        config_path=settings.libero_config_path,
    )
    task = resolve_libero_task(settings.task_name, libero_root=libero_root)
    episodes_per_seed = settings.episodes // len(settings.seeds)
    if episodes_per_seed > len(task.init_states):
        raise EvaluationError(
            f"Requested {episodes_per_seed} episodes per seed but task has "
            f"{len(task.init_states)} fixed initial states"
        )
    policy = QwenLiberoPolicy.from_checkpoint(
        checkpoint,
        device=settings.device,
        denoising_steps=settings.denoising_steps,
        policy_batch_size=settings.policy_batch_size,
    )
    if preflight_only:
        report = run_qwen_preflight(
            settings,
            checkpoint=checkpoint,
            task=task,
            policy=policy,
            libero_commit=libero_commit,
        )
        report["libero_config_file"] = str(libero_config_file)
        _atomic_write_json(settings.output_dir / "preflight.json", report)
        return report

    _check_output_targets(settings)
    started = time.monotonic()
    episodes: list[dict[str, Any]] = []
    video_frames: dict[int, list[np.ndarray]] = {
        episode_id: [] for episode_id in range(settings.record_videos)
    }
    videos: list[str] = []
    environment = None
    environment_batch_size = 0
    environment_batch_sizes: list[int] = []
    active_num_envs = settings.num_envs
    action_transform = QwenActionTransform()
    try:
        for seed_index, seed in enumerate(settings.seeds):
            generator = policy.make_generator(seed)
            seed_start = 0
            while seed_start < episodes_per_seed:
                batch_size = min(active_num_envs, episodes_per_seed - seed_start)
                if environment is None or environment_batch_size != batch_size:
                    if environment is not None:
                        environment.close()
                        environment = None
                        environment_batch_size = 0
                    environment, batch_size = make_vector_environment_with_backoff(
                        task,
                        batch_size,
                        auto_reduce=settings.auto_reduce_num_envs,
                    )
                    active_num_envs = min(active_num_envs, batch_size)
                    environment_batch_size = batch_size
                environment_batch_sizes.append(batch_size)
                init_state_ids = list(range(seed_start, seed_start + batch_size))
                episode_start = seed_index * episodes_per_seed + seed_start
                episode_ids = list(range(episode_start, episode_start + batch_size))
                observations = settle_vector_environment(
                    environment,
                    init_states=task.init_states[init_state_ids],
                    statistics=action_transform,
                    seed=seed + seed_start,
                    settle_steps=settings.settle_steps,
                )

                def capture_frames(current: Any, step: int) -> None:
                    del step
                    values = observation_batch_to_list(current)
                    for local_index, episode_id in enumerate(episode_ids):
                        if episode_id in video_frames:
                            video_frames[episode_id].append(
                                _frame_from_observation(values[local_index])
                            )

                group_episodes, _ = rollout_action_chunks(
                    environment,
                    observations,
                    predictor=lambda current: policy.predict_action_chunk(
                        current,
                        task.language,
                        generator=generator,
                    ),
                    episode_ids=episode_ids,
                    init_state_ids=init_state_ids,
                    seed=seed,
                    max_steps=settings.max_steps,
                    action_horizon=ACTION_HORIZON,
                    frame_callback=capture_frames if video_frames else None,
                )
                episodes.extend(group_episodes)
                _atomic_write_jsonl(
                    settings.output_dir / "episodes.partial.jsonl",
                    episodes,
                )
                seed_start += batch_size
        videos = _save_videos(settings.output_dir, video_frames, fps=settings.video_fps)
    except Exception as error:
        failure = _make_qwen_report(
            status="failed",
            settings=settings,
            checkpoint=checkpoint,
            task=task,
            episodes=episodes,
            elapsed_seconds=time.monotonic() - started,
            videos=videos,
            libero_commit=libero_commit,
            environment_batch_sizes=environment_batch_sizes,
            error=f"{type(error).__name__}: {error}",
        )
        _atomic_write_json(settings.output_dir / "failure.json", failure)
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except UnrecoverableSimulationShutdownError as error:
                failure = _make_qwen_report(
                    status="failed",
                    settings=settings,
                    checkpoint=checkpoint,
                    task=task,
                    episodes=episodes,
                    elapsed_seconds=time.monotonic() - started,
                    videos=videos,
                    libero_commit=libero_commit,
                    environment_batch_sizes=environment_batch_sizes,
                    error=f"{type(error).__name__}: {error}",
                )
                _atomic_write_json(settings.output_dir / "failure.json", failure)
                raise

    report = _make_qwen_report(
        status="complete",
        settings=settings,
        checkpoint=checkpoint,
        task=task,
        episodes=episodes,
        elapsed_seconds=time.monotonic() - started,
        videos=videos,
        libero_commit=libero_commit,
        environment_batch_sizes=environment_batch_sizes,
    )
    _atomic_write_jsonl(settings.output_dir / "episodes.jsonl", episodes)
    _atomic_write_json(settings.output_dir / "results.json", report)
    partial_path = settings.output_dir / "episodes.partial.jsonl"
    if partial_path.exists():
        partial_path.unlink()
    return report
