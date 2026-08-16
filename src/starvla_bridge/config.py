"""Read-only validation for the released StarVLA Bridge checkpoint artifact."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


STARVLA_SOURCE_COMMIT = "3422b9f2387b6f682cf02802904a77b23ab13afd"


class StarVLAConfigError(RuntimeError):
    """Raised when the released StarVLA artifact violates its inference contract."""


@dataclass(frozen=True)
class BridgeActionStatistics:
    q01: np.ndarray
    q99: np.ndarray
    mask: np.ndarray

    def denormalize(self, value: Any) -> np.ndarray:
        actions = np.asarray(value, dtype=np.float32)
        if actions.ndim < 1 or actions.shape[-1] != 7:
            raise StarVLAConfigError(
                f"StarVLA actions must have last dimension 7, found {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise StarVLAConfigError("StarVLA actions contain NaN or infinite values")
        result = actions.copy()
        result[..., self.mask] = (
            (actions[..., self.mask] + np.float32(1.0))
            * np.float32(0.5)
            * (self.q99[self.mask] - self.q01[self.mask])
            + self.q01[self.mask]
        )
        return result


@dataclass(frozen=True)
class StarVLAModelSpec:
    model_dir: Path
    checkpoint_path: Path
    base_model: Path
    config: dict[str, Any]
    action_statistics: BridgeActionStatistics
    action_dim: int
    state_dim: int
    action_horizon: int
    num_inference_timesteps: int
    image_size: tuple[int, int]
    cot_prompt: str


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StarVLAConfigError(f"{name} must be a mapping")
    return value


def _action_statistics(path: Path) -> BridgeActionStatistics:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        action = _mapping(
            _mapping(payload, name="dataset statistics")["oxe_bridge"]["action"],
            name="oxe_bridge.action",
        )
        q01 = np.asarray(action["q01"], dtype=np.float32)
        q99 = np.asarray(action["q99"], dtype=np.float32)
        mask = np.asarray(action["mask"], dtype=np.bool_)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise StarVLAConfigError(f"Could not read oxe_bridge action statistics: {path}") from error
    if q01.shape != (7,) or q99.shape != (7,) or mask.shape != (7,):
        raise StarVLAConfigError("oxe_bridge action q01, q99, and mask must have shape (7,)")
    if not np.all(np.isfinite(q01)) or not np.all(np.isfinite(q99)):
        raise StarVLAConfigError("oxe_bridge action quantiles must be finite")
    if np.any(q01 > q99):
        raise StarVLAConfigError("oxe_bridge action q01 values exceed q99")
    if mask.tolist() != [True, True, True, True, True, True, False]:
        raise StarVLAConfigError(
            "oxe_bridge action mask must normalize six pose dimensions and preserve gripper"
        )
    return BridgeActionStatistics(q01=q01, q99=q99, mask=mask)


def load_model_spec(
    model_dir: str | Path,
    *,
    base_model: str | Path,
) -> StarVLAModelSpec:
    root = Path(model_dir).expanduser().resolve()
    base = Path(base_model).expanduser().resolve()
    config_path = root / "config.yaml"
    statistics_path = root / "dataset_statistics.json"
    if not root.is_dir():
        raise StarVLAConfigError(f"StarVLA model directory does not exist: {root}")
    if not base.is_dir() or not (base / "config.json").is_file():
        raise StarVLAConfigError(f"Local Qwen3-VL base model is incomplete: {base}")
    if not config_path.is_file() or not statistics_path.is_file():
        raise StarVLAConfigError(f"StarVLA config or statistics is missing under {root}")
    checkpoints = sorted((root / "checkpoints").glob("steps_*_pytorch_model.pt"))
    if len(checkpoints) != 1 or not checkpoints[0].is_file():
        raise StarVLAConfigError(
            f"Expected exactly one steps_*_pytorch_model.pt under {root / 'checkpoints'}"
        )
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise StarVLAConfigError(f"Could not read StarVLA config: {config_path}") from error
    config = copy.deepcopy(dict(_mapping(loaded, name="StarVLA config")))
    try:
        framework = _mapping(config["framework"], name="framework")
        if framework["name"] != "QwenGR00T":
            raise StarVLAConfigError("StarVLA framework.name must be QwenGR00T")
        qwenvl = dict(_mapping(framework["qwenvl"], name="framework.qwenvl"))
        action_model = _mapping(framework["action_model"], name="framework.action_model")
        vla_data = _mapping(
            _mapping(config["datasets"], name="datasets")["vla_data"],
            name="datasets.vla_data",
        )
        action_dim = int(action_model["action_dim"])
        state_dim = int(action_model["state_dim"])
        action_horizon = int(action_model["action_horizon"])
        inference_steps = int(action_model["num_inference_timesteps"])
        image_size_value = tuple(int(item) for item in vla_data["image_size"])
        cot_prompt = str(vla_data["CoT_prompt"])
        observations = list(vla_data["obs"])
    except StarVLAConfigError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise StarVLAConfigError("StarVLA config is missing its inference contract") from error
    expected = (action_dim, state_dim, action_horizon, inference_steps)
    if expected != (7, 7, 16, 4):
        raise StarVLAConfigError(
            "StarVLA action contract must be action_dim=7, state_dim=7, "
            "action_horizon=16, num_inference_timesteps=4"
        )
    if image_size_value != (224, 224) or observations != ["image_0"]:
        raise StarVLAConfigError("StarVLA must use one image_0 observation at 224x224")
    if "{instruction}" not in cot_prompt:
        raise StarVLAConfigError("StarVLA CoT_prompt must contain {instruction}")
    qwenvl["base_vlm"] = str(base)
    framework = dict(framework)
    framework["qwenvl"] = qwenvl
    config["framework"] = framework
    return StarVLAModelSpec(
        model_dir=root,
        checkpoint_path=checkpoints[0].resolve(),
        base_model=base,
        config=config,
        action_statistics=_action_statistics(statistics_path),
        action_dim=action_dim,
        state_dim=state_dim,
        action_horizon=action_horizon,
        num_inference_timesteps=inference_steps,
        image_size=image_size_value,
        cot_prompt=cot_prompt,
    )
