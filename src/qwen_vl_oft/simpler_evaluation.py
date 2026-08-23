from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qwen_vl_common.backbone import ModelContractError, inspect_qwen_config
from simpler_bridge.evaluation import SimplerEvaluationError, select_first_action

from .checkpointing import CHECKPOINT_FORMAT
from .config import ConfigError, validate_config


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
        raise SimplerEvaluationError(f"Could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise SimplerEvaluationError(f"{description} must contain a JSON object: {path}")
    return value


@dataclass(frozen=True)
class OFTCheckpointSpec:
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


def resolve_oft_checkpoint(
    checkpoint: str | Path,
    *,
    model_path: str | Path | None = None,
) -> OFTCheckpointSpec:
    requested = Path(checkpoint).expanduser().resolve()
    if not requested.is_dir() or re.fullmatch(r"step-\d{8}", requested.name) is None:
        raise SimplerEvaluationError(
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
        raise SimplerEvaluationError(f"Qwen-VL OFT checkpoint is incomplete; missing={missing}")

    manifest = _read_json(policy_config_path, "Qwen-VL OFT policy config")
    if manifest.get("format") != CHECKPOINT_FORMAT:
        raise SimplerEvaluationError(
            "Unsupported Qwen-VL OFT checkpoint format: "
            f"{manifest.get('format')!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise SimplerEvaluationError("Qwen-VL OFT policy config is missing its config object")
    try:
        validate_config(config)
    except ConfigError as error:
        raise SimplerEvaluationError(str(error)) from error

    data = config["data"]
    try:
        crop_size = int(data["train_crop_size"])
        output_size = int(data["output_image_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise SimplerEvaluationError(
            "Qwen-VL OFT checkpoint requires integer train_crop_size and output_image_size"
        ) from error
    if crop_size <= 0 or crop_size > 256 or output_size <= 0:
        raise SimplerEvaluationError(
            "Qwen-VL OFT checkpoint requires 0 < train_crop_size <= 256 and "
            "positive output_image_size"
        )

    normalization = _read_json(normalization_path, "Qwen-VL OFT normalization metadata")
    for key, expected in (
        ("state_q01", 8),
        ("state_q99", 8),
        ("action_q01", 7),
        ("action_q99", 7),
    ):
        try:
            value = np.asarray(normalization[key], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise SimplerEvaluationError(
                f"Qwen-VL OFT normalization is missing finite {key}"
            ) from error
        if value.shape != (expected,) or not np.all(np.isfinite(value)):
            raise SimplerEvaluationError(
                f"Qwen-VL OFT normalization {key} must be finite with shape {(expected,)}"
            )

    raw_base_model = model_path if model_path is not None else manifest.get("base_model")
    if raw_base_model is None:
        raise SimplerEvaluationError(
            "Qwen-VL OFT policy config has no base_model; pass --model-path"
        )
    base_model = Path(raw_base_model).expanduser().resolve()
    if not base_model.is_dir():
        raise SimplerEvaluationError(f"Qwen-VL OFT base model does not exist: {base_model}")
    try:
        inspect_qwen_config(base_model, expected_family="qwen3_vl")
    except ModelContractError as error:
        raise SimplerEvaluationError(str(error)) from error

    global_step = manifest.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise SimplerEvaluationError(
            "Qwen-VL OFT policy config requires a non-negative integer global_step"
        )
    if requested.name != f"step-{global_step:08d}":
        raise SimplerEvaluationError(
            f"Checkpoint directory {requested.name} does not match global_step={global_step}"
        )

    return OFTCheckpointSpec(
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


def preprocess_bridge_image(
    value: Any,
    *,
    crop_size: int,
    output_size: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SimplerEvaluationError(f"Expected an HWC RGB image, found {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise SimplerEvaluationError(f"Expected an integer RGB image, found {array.dtype}")
    if crop_size <= 0 or crop_size > 256 or output_size <= 0:
        raise SimplerEvaluationError("Bridge image crop/output sizes are invalid")
    try:
        from PIL import Image
    except ImportError as error:
        raise SimplerEvaluationError("Pillow is required for Bridge image preprocessing") from error
    image = Image.fromarray(array.astype(np.uint8, copy=False)).resize(
        (256, 256),
        resample=Image.Resampling.LANCZOS,
    )
    offset = (256 - crop_size) // 2
    image = image.crop((offset, offset, offset + crop_size, offset + crop_size))
    if output_size != crop_size:
        image = image.resize(
            (output_size, output_size),
            resample=Image.Resampling.BICUBIC,
        )
    return np.asarray(image, dtype=np.uint8).copy()


class OFTPolicyAdapter:
    policy_name = "Qwen-VL OFT checkpoint"
    gripper_threshold = 0.5

    def __init__(self, *, policy: Any, crop_size: int, output_size: int) -> None:
        self.policy = policy
        self.crop_size = int(crop_size)
        self.output_size = int(output_size)

    def make_generator(self, seed: int) -> int:
        return int(seed)

    def begin_episode(self, instruction: str) -> None:
        del instruction

    def select_action(self, actions: np.ndarray) -> np.ndarray:
        return select_first_action(actions)

    def prepare_observation(
        self,
        image: Any,
        proprio: Any,
        instruction: str,
    ) -> dict[str, Any]:
        return {
            "image": preprocess_bridge_image(
                image,
                crop_size=self.crop_size,
                output_size=self.output_size,
            ),
            "proprio": np.asarray(proprio, dtype=np.float32),
            "instruction": str(instruction),
        }

    def describe_observation(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        return {"model_image_shape": list(np.asarray(prepared["image"]).shape)}

    def predict_actions(self, prepared: Mapping[str, Any], *, generator: Any) -> Any:
        del generator
        actions = self.policy.predict_actions(
            prepared["image"],
            prepared["proprio"],
            prepared["instruction"],
        )
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "inference_strategy": "deterministic-causal-query-oft",
            "native_action_chunk_size": 8,
        }
