"""Strict metadata preflight for the supported BridgeData V2 dataset."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


STATE_AXES = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
IMAGE_KEY = "observation.images.image_0"


class BridgeDatasetError(ValueError):
    """Raised when a dataset does not satisfy the fixed BridgeData V2 contract."""


def _mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeDatasetError(f"BridgeData V2 {description} must be a mapping")
    return value


def _feature(features: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in features:
        raise BridgeDatasetError(f"BridgeData V2 is missing required feature {key!r}")
    return _mapping(features[key], description=f"feature {key!r}")


def _float_or_nan(value: object) -> float:
    if isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def validate_bridge_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """Validate and return the mounted dataset's LeRobot ``info.json`` payload."""

    root = Path(dataset_path).expanduser()
    if not root.is_dir():
        raise BridgeDatasetError(f"BridgeData V2 dataset directory does not exist: {root}")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise BridgeDatasetError(f"BridgeData V2 metadata file does not exist: {info_path}")
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeDatasetError(f"Could not read BridgeData V2 metadata {info_path}: {error}") from error
    info = _mapping(payload, description="metadata root")

    if info.get("codebase_version") != "v2.0":
        raise BridgeDatasetError("BridgeData V2 requires LeRobot v2.0 metadata")
    if info.get("robot_type") != "widowx":
        raise BridgeDatasetError("BridgeData V2 requires robot_type='widowx'")
    fps = info.get("fps")
    if isinstance(fps, bool):
        raise BridgeDatasetError("BridgeData V2 requires a 5 Hz dataset")
    fps_value = _float_or_nan(fps)
    if not math.isfinite(fps_value) or fps_value != 5.0:
        raise BridgeDatasetError("BridgeData V2 requires a 5 Hz dataset")

    features = _mapping(info.get("features"), description="features")
    state = _feature(features, "observation.state")
    state_names = state.get("names")
    motors = state_names.get("motors") if isinstance(state_names, Mapping) else None
    if state.get("dtype") != "float32" or state.get("shape") != [8]:
        raise BridgeDatasetError("BridgeData V2 observation.state must be float32 with shape [8]")
    if tuple(motors or ()) != STATE_AXES:
        raise BridgeDatasetError(
            "BridgeData V2 observation.state axis names must be " + ",".join(STATE_AXES)
        )

    action = _feature(features, "action")
    if action.get("dtype") != "float32" or action.get("shape") != [7]:
        raise BridgeDatasetError("BridgeData V2 action must be float32 with shape [7]")

    image = _feature(features, IMAGE_KEY)
    image_shape = image.get("shape")
    image_info = image.get("info")
    is_rgb_shape = (
        isinstance(image_shape, list)
        and len(image_shape) == 3
        and all(isinstance(value, int) and value > 0 for value in image_shape)
        and image_shape[-1] == 3
    )
    is_rgb_video = (
        image.get("dtype") == "video"
        and is_rgb_shape
        and image.get("names") == ["height", "width", "rgb"]
        and isinstance(image_info, Mapping)
        and image_info.get("video.channels") == 3
        and _float_or_nan(image_info.get("video.fps")) == 5.0
        and image_info.get("video.is_depth_map") is False
    )
    if not is_rgb_video:
        raise BridgeDatasetError(
            f"BridgeData V2 {IMAGE_KEY} must be a 5 Hz, three-channel RGB video"
        )
    return dict(info)
