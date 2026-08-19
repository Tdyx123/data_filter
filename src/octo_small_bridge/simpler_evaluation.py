"""Octo-small Bridge adapter for the shared SimplerEnv protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from packaging.version import InvalidVersion, Version

from simpler_bridge.evaluation import SimplerEvaluationError

from .checkpoint_contract import (
    BridgeCheckpointContractError,
    validate_bridge_checkpoint,
)
from .normalization import BridgeV2NormalizationStatistics


ACTION_DIM = 7
ACTION_HORIZON = 8
PROPRIO_DIM = 8
LANGUAGE_TOKENS = 16
DIFFUSION_STEPS = 20
PRIMARY_IMAGE_SIZE = 256
OCTO_SIMPLER_RUNTIME_PACKAGE_VERSIONS = {
    "numpy": "1.24.3",
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "transformers": "4.44.2",
    "tokenizers": "0.19.1",
    "safetensors": "0.4.5",
    "sentencepiece": "0.2.0",
    "Pillow": "10.4.0",
    "setuptools": "75.8.0",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BridgeNormalizationStatistics:
    path: Path
    values: BridgeV2NormalizationStatistics
    sha256: str

    def normalize_proprio(self, value: Any) -> np.ndarray:
        try:
            return self.values.normalize_state(value)
        except ValueError as error:
            raise SimplerEvaluationError(str(error)) from error

    def actions_to_bridge(self, value: Any) -> np.ndarray:
        try:
            return self.values.denormalize_action(value)
        except ValueError as error:
            raise SimplerEvaluationError(str(error)) from error

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "contract": self.values.contract,
        }


def load_bridge_statistics(path: str | Path) -> BridgeNormalizationStatistics:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise SimplerEvaluationError(f"Bridge statistics file does not exist: {target}")
    try:
        values = BridgeV2NormalizationStatistics.load(target)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SimplerEvaluationError(
            f"Could not read Bridge V2 normalization: {target}: {error}"
        ) from error
    return BridgeNormalizationStatistics(
        path=target,
        values=values,
        sha256=_sha256(target),
    )


def validate_runtime_contract(
    *,
    version_info: tuple[int, int] | None = None,
    package_versions: Mapping[str, str] | None = None,
    device: str,
    torch_module: Any | None = None,
) -> dict[str, str]:
    current_python = version_info or (sys.version_info.major, sys.version_info.minor)
    if tuple(current_python) not in {(3, 10), (3, 11)}:
        raise SimplerEvaluationError(
            "Octo SimplerEnv evaluation requires Python 3.10 or 3.11; "
            f"found {current_python[0]}.{current_python[1]}"
        )
    if package_versions is None:
        discovered: dict[str, str] = {}
        for name in OCTO_SIMPLER_RUNTIME_PACKAGE_VERSIONS:
            try:
                discovered[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as error:
                raise SimplerEvaluationError(
                    f"Required Octo SimplerEnv package is not installed: {name}"
                ) from error
        package_versions = discovered
    normalized = {str(name): str(version) for name, version in package_versions.items()}
    for name, expected in OCTO_SIMPLER_RUNTIME_PACKAGE_VERSIONS.items():
        actual = normalized.get(name)
        matches = actual == expected
        if not matches and actual is not None and name in {"torch", "torchvision"}:
            try:
                matches = Version(actual).public == expected
            except InvalidVersion:
                matches = False
        if not matches:
            raise SimplerEvaluationError(
                f"Octo SimplerEnv evaluation requires {name}=={expected}; "
                f"found {actual or 'not installed'}"
            )
    if str(device).startswith("cuda"):
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as error:
                raise SimplerEvaluationError("PyTorch is required for CUDA evaluation") from error
        if not torch_module.cuda.is_available():
            from simpler_bridge.evaluation import SimplerInfrastructureError

            raise SimplerInfrastructureError(
                f"CUDA device {device!r} was requested but torch.cuda.is_available() is false"
            )
    return normalized


def preprocess_primary_image(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SimplerEvaluationError(f"Expected an HWC RGB image, found {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise SimplerEvaluationError(f"Expected an integer RGB image, found {array.dtype}")
    try:
        from PIL import Image
    except ImportError as error:
        raise SimplerEvaluationError("Pillow is required for Octo image preprocessing") from error
    image = Image.fromarray(array.astype(np.uint8, copy=False), mode="RGB")
    if image.size != (PRIMARY_IMAGE_SIZE, PRIMARY_IMAGE_SIZE):
        image = image.resize(
            (PRIMARY_IMAGE_SIZE, PRIMARY_IMAGE_SIZE),
            resample=Image.Resampling.LANCZOS,
        )
    channel_first = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    normalized = channel_first / np.float32(127.5) - np.float32(1.0)
    return normalized[None, None].astype(np.float32, copy=False)


class OctoBridgeSimplerPolicy:
    policy_name = "Octo-small Bridge checkpoint"
    gripper_threshold = 0.5

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        statistics: BridgeNormalizationStatistics,
        device: str,
        precision: str,
        generator_factory: Callable[[str, int], Any] | None = None,
        sampler: Callable[[Mapping[str, np.ndarray], Any], Any] | None = None,
    ) -> None:
        if precision not in {"bf16", "fp32"}:
            raise SimplerEvaluationError("precision must be bf16 or fp32")
        self.model = model
        self.tokenizer = tokenizer
        self.statistics = statistics
        self.device = str(device)
        self.precision = precision
        self._generator_factory = generator_factory
        self._sampler = sampler

    def make_generator(self, seed: int) -> Any:
        if self._generator_factory is not None:
            return self._generator_factory(self.device, int(seed))
        import torch

        return torch.Generator(device=self.device).manual_seed(int(seed))

    def prepare_observation(
        self,
        image: Any,
        proprio: Any,
        instruction: str,
    ) -> dict[str, np.ndarray]:
        proprio_array = np.asarray(proprio, dtype=np.float32)
        if proprio_array.shape != (PROPRIO_DIM,):
            raise SimplerEvaluationError(
                f"Expected raw Bridge proprio shape ({PROPRIO_DIM},), found {proprio_array.shape}"
            )
        encoded = self.tokenizer(
            [str(instruction)],
            padding="max_length",
            truncation=True,
            max_length=LANGUAGE_TOKENS,
            return_tensors="np",
        )
        try:
            input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.bool_)
        except (KeyError, TypeError, ValueError) as error:
            raise SimplerEvaluationError("Octo tokenizer returned invalid arrays") from error
        if input_ids.shape != (1, LANGUAGE_TOKENS) or attention_mask.shape != (
            1,
            LANGUAGE_TOKENS,
        ):
            raise SimplerEvaluationError(
                "Octo tokenizer must return input_ids and attention_mask with shape (1, 16)"
            )
        return {
            "image_primary": preprocess_primary_image(image),
            "proprio": self.statistics.normalize_proprio(proprio_array)[None, None],
            "language_input_ids": input_ids,
            "language_attention_mask": attention_mask,
        }

    def describe_observation(self, prepared: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {
            "model_image_shape": list(prepared["image_primary"].shape),
            "model_proprio_shape": list(prepared["proprio"].shape),
            "observation_tokenizers": ["primary"],
        }

    def _sample_actions(self, prepared: Mapping[str, np.ndarray], generator: Any) -> Any:
        if self._sampler is not None:
            return self._sampler(prepared, generator)
        import torch

        batch = {
            key: torch.from_numpy(value).to(self.device, non_blocking=True)
            for key, value in prepared.items()
        }
        use_bf16 = self.precision == "bf16"
        with torch.inference_mode():
            with torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                return self.model.sample_actions(batch, generator=generator)

    def predict_actions(self, prepared: Mapping[str, np.ndarray], *, generator: Any) -> np.ndarray:
        actions = self._sample_actions(prepared, generator)
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        normalized = np.asarray(actions, dtype=np.float32)
        expected = (1, ACTION_HORIZON, ACTION_DIM)
        if normalized.shape != expected:
            raise SimplerEvaluationError(
                f"Octo checkpoint returned actions with shape {normalized.shape}; expected {expected}"
            )
        return self.statistics.actions_to_bridge(normalized)

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "diffusion_steps": DIFFUSION_STEPS,
            "gripper_threshold": self.gripper_threshold,
            "native_action_chunk_size": ACTION_HORIZON,
            "observation_tokenizers": ["primary"],
            "precision": self.precision,
            "statistics": self.statistics.as_dict(),
        }


def _validate_model_contract(model: Any) -> None:
    config = model.config
    actual = {
        "action_dim": int(config.action_dim),
        "action_horizon": int(config.action_horizon),
        "proprio_dim": int(config.proprio_dim),
        "language_tokens": int(config.language_tokens),
        "diffusion_steps": int(config.diffusion_steps),
    }
    expected = {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "proprio_dim": PROPRIO_DIM,
        "language_tokens": LANGUAGE_TOKENS,
        "diffusion_steps": DIFFUSION_STEPS,
    }
    if actual != expected:
        raise SimplerEvaluationError(
            f"Loaded Octo Bridge model contract is {actual}; expected {expected}"
        )


def load_octo_bridge_policy(
    checkpoint: str | Path,
    *,
    base_model: str | Path,
    statistics: str | Path | None = None,
    device: str,
    precision: str,
    model_loader: Callable[..., tuple[Any, Any]] | None = None,
    weight_loader: Callable[..., Any] | None = None,
    torch_module: Any | None = None,
) -> tuple[Any, OctoBridgeSimplerPolicy]:
    """Load a weight-only Bridge checkpoint over a self-contained Octo base."""

    try:
        from octo_small_libero.evaluation import EvaluationError, resolve_checkpoint
    except ImportError as error:
        raise SimplerEvaluationError(str(error)) from error
    try:
        checkpoint_spec = resolve_checkpoint(checkpoint, base_model=base_model)
    except EvaluationError as error:
        raise SimplerEvaluationError(str(error)) from error

    try:
        checkpoint_statistics = validate_bridge_checkpoint(
            checkpoint_spec.weights_path.parent,
            expected_normalization_path=statistics,
        )
    except BridgeCheckpointContractError as error:
        raise SimplerEvaluationError(str(error)) from error
    bridge_statistics = load_bridge_statistics(
        checkpoint_statistics if statistics is None else statistics
    )
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise SimplerEvaluationError("PyTorch is required for Octo evaluation") from error
    if model_loader is None:
        from octo_small_libero.torch_model import OctoSmallPolicy

        model_loader = OctoSmallPolicy.from_pretrained
    if weight_loader is None:
        try:
            from safetensors.torch import load_model as weight_loader
        except ImportError as error:
            raise SimplerEvaluationError(
                "safetensors is required for Octo checkpoint evaluation"
            ) from error

    if precision not in {"bf16", "fp32"}:
        raise SimplerEvaluationError("precision must be bf16 or fp32")
    resolved_device = torch_module.device(device)
    device_type = str(resolved_device.type)
    if device_type == "cuda":
        if not torch_module.cuda.is_available():
            raise SimplerEvaluationError(f"CUDA device requested but CUDA is unavailable: {device}")
        if precision == "bf16" and not torch_module.cuda.is_bf16_supported():
            raise SimplerEvaluationError(f"CUDA device does not support BF16: {device}")
    elif precision == "bf16":
        raise SimplerEvaluationError("BF16 evaluation requires a CUDA device; use --precision fp32")

    try:
        model, tokenizer = model_loader(
            checkpoint_spec.base_model_path,
            device="cpu",
            observation_tokenizers=("primary",),
        )
        base_weights = (checkpoint_spec.base_model_path / "model.safetensors").resolve()
        if checkpoint_spec.weights_path.resolve() != base_weights:
            weight_loader(
                model,
                str(checkpoint_spec.weights_path),
                strict=True,
                device="cpu",
            )
        _validate_model_contract(model)
        model.to(resolved_device)
        model.eval()
    except SimplerEvaluationError:
        raise
    except Exception as error:
        raise SimplerEvaluationError(f"Could not load Octo Bridge checkpoint: {error}") from error

    return checkpoint_spec, OctoBridgeSimplerPolicy(
        model=model,
        tokenizer=tokenizer,
        statistics=bridge_statistics,
        device=str(device),
        precision=precision,
    )
