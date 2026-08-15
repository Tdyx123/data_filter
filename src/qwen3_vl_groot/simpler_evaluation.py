from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from simpler_bridge.evaluation import (  # noqa: F401
    CONTROL_MODE,
    OBJECT_EPISODE_IDS,
    POLICY_SEEDS,
    SIMPLER_TASKS,
    SimplerEvaluationError,
    SimplerInfrastructureError,
    SimplerTaskSpec,
    _write_video,
    bridge_actions_to_simpler,
    create_simpler_environment,
    default_simpler_root,
    environment_to_bridge_proprio,
    image_from_simpler_observation,
    resolve_task_selection,
    validate_simpler_source,
)


RUNTIME_PACKAGE_VERSIONS = {
    "numpy": "1.24.4",
    "torch": "2.9.0",
    "torchvision": "0.24.0",
    "transformers": "5.2.0",
    "peft": "0.18.0",
    "safetensors": "0.5.3",
    "Pillow": "11.1.0",
    "scipy": "1.12.0",
    "gymnasium": "0.29.1",
    "sapien": "2.2.2",
    "h5py": "3.10.0",
    "PyYAML": "6.0.2",
    "transforms3d": "0.4.2",
    "opencv-python": "4.11.0.86",
    "imageio": "2.37.0",
    "imageio-ffmpeg": "0.6.0",
    "trimesh": "4.8.3",
    "rtree": "1.4.1",
    "ruckig": "0.14.0",
}


@dataclass(frozen=True)
class SimplerEvaluationSettings:
    checkpoint: Path
    output_dir: Path = Path("outputs/qwen_simpler_eval")
    tasks: tuple[SimplerTaskSpec, ...] = ()
    model_path: Path | None = None
    device: str = "cuda:0"
    denoising_steps: int = 4
    action_horizon: int = 8
    policy_seeds: tuple[int, ...] = POLICY_SEEDS
    object_episode_ids: tuple[int, ...] = OBJECT_EPISODE_IDS
    max_steps: int | None = None
    save_videos_path: Path | None = None
    video_fps: int = 5
    overwrite: bool = False


class QwenSimplerPolicy:
    def __init__(
        self,
        *,
        policy: Any,
        device: str,
        generator_factory: Callable[[str, int], Any] | None = None,
    ):
        self.policy = policy
        self.device = str(device)
        self._generator_factory = generator_factory

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Any,
        *,
        device: str,
    ) -> "QwenSimplerPolicy":
        from .inference import BridgePolicy

        try:
            policy = BridgePolicy.from_pretrained(
                checkpoint.requested_path,
                model_path=checkpoint.base_model_path,
                device=device,
            )
        except Exception as error:
            raise SimplerEvaluationError(
                f"Could not load Qwen compact LoRA checkpoint: {error}"
            ) from error
        return cls(policy=policy, device=device)

    def make_generator(self, seed: int) -> Any:
        if self._generator_factory is not None:
            return self._generator_factory(self.device, int(seed))
        import torch

        return torch.Generator(device=self.device).manual_seed(int(seed))

    def predict_actions(
        self,
        image: Any,
        state: Any,
        instruction: str,
        denoising_steps: int,
        *,
        generator: Any,
    ) -> Any:
        return self.policy.predict_actions(
            image,
            state,
            instruction,
            denoising_steps=denoising_steps,
            generator=generator,
        )


def preprocess_bridge_image(
    value: Any,
    resize_size: int,
    crop_size: int,
    output_size: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SimplerEvaluationError(f"Expected an HWC RGB image, found {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise SimplerEvaluationError(f"Expected an integer RGB image, found {array.dtype}")
    if resize_size <= 0 or crop_size <= 0 or output_size <= 0 or crop_size > resize_size:
        raise SimplerEvaluationError("Bridge image resize/crop/output sizes are invalid")
    try:
        from PIL import Image
    except ImportError as error:
        raise SimplerEvaluationError("Pillow is required for Bridge image preprocessing") from error
    image = Image.fromarray(array.astype(np.uint8, copy=False)).resize(
        (resize_size, resize_size),
        resample=Image.Resampling.LANCZOS,
    )
    offset = (resize_size - crop_size) // 2
    image = image.crop((offset, offset, offset + crop_size, offset + crop_size))
    if output_size != crop_size:
        image = image.resize((output_size, output_size), resample=Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8).copy()


class _QwenPolicyAdapter:
    policy_name = "Qwen checkpoint"
    gripper_threshold = 0.5

    def __init__(
        self,
        *,
        policy: Any,
        crop_size: int,
        output_size: int,
        denoising_steps: int,
    ) -> None:
        self.policy = policy
        self.crop_size = int(crop_size)
        self.output_size = int(output_size)
        self.denoising_steps = int(denoising_steps)

    def make_generator(self, seed: int) -> Any:
        return self.policy.make_generator(seed)

    def prepare_observation(
        self,
        image: Any,
        proprio: Any,
        instruction: str,
    ) -> dict[str, Any]:
        return {
            "image": preprocess_bridge_image(
                image,
                resize_size=256,
                crop_size=self.crop_size,
                output_size=self.output_size,
            ),
            "proprio": np.asarray(proprio, dtype=np.float32),
            "instruction": str(instruction),
        }

    def describe_observation(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        return {"model_image_shape": list(np.asarray(prepared["image"]).shape)}

    def predict_actions(self, prepared: Mapping[str, Any], *, generator: Any) -> Any:
        return self.policy.predict_actions(
            prepared["image"],
            prepared["proprio"],
            prepared["instruction"],
            denoising_steps=self.denoising_steps,
            generator=generator,
        )

    def protocol_metadata(self) -> dict[str, int]:
        return {"denoising_steps": self.denoising_steps}


def run_simpler_episode(
    *,
    task: SimplerTaskSpec,
    object_episode_id: int,
    policy_seed: int,
    policy: Any,
    environment: Any,
    generator: Any,
    data_config: Mapping[str, Any],
    denoising_steps: int,
    action_horizon: int,
    max_steps: int,
    capture_video: bool,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    from simpler_bridge.evaluation import run_simpler_episode as run_shared_episode

    if denoising_steps <= 0:
        raise SimplerEvaluationError("denoising_steps must be positive")
    try:
        crop_size = int(data_config["train_crop_size"])
        output_size = int(data_config["output_image_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SimplerEvaluationError(
            "Bridge checkpoint requires integer train_crop_size and output_image_size"
        ) from error

    adapter = _QwenPolicyAdapter(
        policy=policy,
        crop_size=crop_size,
        output_size=output_size,
        denoising_steps=denoising_steps,
    )
    return run_shared_episode(
        task=task,
        object_episode_id=object_episode_id,
        policy_seed=policy_seed,
        policy=adapter,
        environment=environment,
        generator=generator,
        action_horizon=action_horizon,
        max_steps=max_steps,
        capture_video=capture_video,
    )


def _validate_settings(settings: SimplerEvaluationSettings) -> None:
    if not settings.tasks:
        raise SimplerEvaluationError("At least one SimplerEnv task is required")
    if not settings.policy_seeds or not settings.object_episode_ids:
        raise SimplerEvaluationError("policy_seeds and object_episode_ids must be non-empty")
    if not 1 <= settings.action_horizon <= 8:
        raise SimplerEvaluationError("action_horizon must be in [1, 8]")
    if settings.denoising_steps <= 0:
        raise SimplerEvaluationError("denoising_steps must be positive")
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


def validate_runtime_contract(
    *,
    version_info: tuple[int, int] | None = None,
    package_versions: Mapping[str, str] | None = None,
    device: str,
) -> dict[str, str]:
    current_python = version_info or (sys.version_info.major, sys.version_info.minor)
    if tuple(current_python) not in {(3, 10), (3, 11)}:
        raise SimplerEvaluationError(
            f"SimplerEnv single-process evaluation requires Python 3.10 or 3.11; "
            f"found {current_python[0]}.{current_python[1]}"
        )
    if package_versions is None:
        discovered: dict[str, str] = {}
        for name in RUNTIME_PACKAGE_VERSIONS:
            try:
                discovered[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise SimplerEvaluationError(
                    f"Required single-process package is not installed: {name}"
                ) from error
        package_versions = discovered
    normalized = {str(name): str(version) for name, version in package_versions.items()}
    for name, expected in RUNTIME_PACKAGE_VERSIONS.items():
        actual = normalized.get(name)
        if actual != expected:
            raise SimplerEvaluationError(
                f"SimplerEnv single-process evaluation requires {name}=={expected}; "
                f"found {actual or 'not installed'}"
            )
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError as error:
            raise SimplerEvaluationError("PyTorch is required for CUDA evaluation") from error
        if not torch.cuda.is_available():
            raise SimplerInfrastructureError(
                f"CUDA device {device!r} was requested but torch.cuda.is_available() is false"
            )
    return normalized


def run_simpler_preflight(
    settings: SimplerEvaluationSettings,
    *,
    checkpoint: Any,
    policy: Any,
    environment_factory: Callable[[SimplerTaskSpec], Any],
    source_versions: Mapping[str, Any],
    package_versions: Mapping[str, str],
) -> dict[str, Any]:
    from simpler_bridge.evaluation import (
        SimplerRunSettings,
        run_simpler_preflight as run_shared_preflight,
    )

    _validate_settings(settings)
    try:
        crop_size = int(checkpoint.config["data"]["train_crop_size"])
        output_size = int(checkpoint.config["data"]["output_image_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SimplerEvaluationError(
            "Bridge checkpoint requires integer train_crop_size and output_image_size"
        ) from error
    shared_settings = SimplerRunSettings(
        output_dir=settings.output_dir,
        tasks=settings.tasks,
        device=settings.device,
        action_horizon=settings.action_horizon,
        policy_seeds=settings.policy_seeds,
        object_episode_ids=settings.object_episode_ids,
        max_steps=settings.max_steps,
        save_videos_path=settings.save_videos_path,
        video_fps=settings.video_fps,
        overwrite=settings.overwrite,
    )
    adapter = _QwenPolicyAdapter(
        policy=policy,
        crop_size=crop_size,
        output_size=output_size,
        denoising_steps=settings.denoising_steps,
    )
    return run_shared_preflight(
        shared_settings,
        checkpoint=checkpoint.as_dict(),
        policy=adapter,
        environment_factory=environment_factory,
        source_versions=source_versions,
        package_versions=package_versions,
        route="qwen3-vl-groot-simpler-widowx-preflight",
    )


def evaluate_simpler_checkpoint(
    settings: SimplerEvaluationSettings,
    *,
    checkpoint: Any,
    policy: Any,
    environment_factory: Callable[[SimplerTaskSpec], Any],
    source_versions: Mapping[str, Any],
    video_writer: Callable[[Path, Sequence[np.ndarray], int], None] = _write_video,
) -> dict[str, Any]:
    from simpler_bridge.evaluation import SimplerRunSettings, evaluate_simpler_policy

    _validate_settings(settings)
    try:
        crop_size = int(checkpoint.config["data"]["train_crop_size"])
        output_size = int(checkpoint.config["data"]["output_image_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SimplerEvaluationError(
            "Bridge checkpoint requires integer train_crop_size and output_image_size"
        ) from error
    shared_settings = SimplerRunSettings(
        output_dir=settings.output_dir,
        tasks=settings.tasks,
        device=settings.device,
        action_horizon=settings.action_horizon,
        policy_seeds=settings.policy_seeds,
        object_episode_ids=settings.object_episode_ids,
        max_steps=settings.max_steps,
        save_videos_path=settings.save_videos_path,
        video_fps=settings.video_fps,
        overwrite=settings.overwrite,
    )
    adapter = _QwenPolicyAdapter(
        policy=policy,
        crop_size=crop_size,
        output_size=output_size,
        denoising_steps=settings.denoising_steps,
    )
    return evaluate_simpler_policy(
        shared_settings,
        checkpoint=checkpoint.as_dict(),
        policy=adapter,
        environment_factory=environment_factory,
        source_versions=source_versions,
        route="qwen3-vl-groot-simpler-widowx-eval",
        protocol_metadata=adapter.protocol_metadata(),
        video_writer=video_writer,
    )
