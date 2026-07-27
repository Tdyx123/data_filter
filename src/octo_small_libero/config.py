from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the independent Octo LIBERO configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    with target.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping: {target}")
    config["_config_path"] = str(target)
    validate_config(config)
    return config


def _boolean_mask(data: dict[str, Any], key: str, action_dim: int) -> list[bool]:
    mask = data.get(key)
    if (
        not isinstance(mask, list)
        or len(mask) != action_dim
        or not all(isinstance(value, bool) for value in mask)
    ):
        raise ConfigError(f"data.{key} must contain one bool per action dimension")
    return mask


def validate_config(config: dict[str, Any]) -> None:
    missing = {"paths", "data", "model", "train"}.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")

    paths = config["paths"]
    for name in ("model", "lerobot", "output"):
        if not isinstance(paths.get(name), str) or not paths[name].strip():
            raise ConfigError(f"paths.{name} must be a non-empty path")

    data = config["data"]
    for name in ("prior_dataset", "target_dataset", "action_key"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ConfigError(f"data.{name} must be a non-empty string")
    if data["prior_dataset"] != "libero90":
        raise ConfigError("The LIBERO prior dataset must be named libero90")
    allowed_dataset_characters = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    if any(character not in allowed_dataset_characters for character in data["target_dataset"]):
        raise ConfigError("data.target_dataset must be a lowercase LeRobot dataset directory name")
    if data["prior_dataset"] == data["target_dataset"]:
        raise ConfigError("The prior and target LeRobot dataset names must differ")

    weights = data.get("sample_weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 2
        or any(not isinstance(value, (int, float)) or value <= 0 for value in weights)
    ):
        raise ConfigError("data.sample_weights must contain two positive numbers")

    if int(data.get("state_dim", -1)) != 8:
        raise ConfigError("LIBERO proprio must have state_dim=8")
    action_dim = int(data.get("action_dim", -1))
    if action_dim != 7:
        raise ConfigError("LIBERO actions must have action_dim=7")
    if int(data.get("window_size", 0)) != 1:
        raise ConfigError("The DataMIL LIBERO configuration requires data.window_size=1")
    if int(data.get("action_horizon", 0)) != 8:
        raise ConfigError("The DataMIL LIBERO configuration requires data.action_horizon=8")

    absolute_mask = _boolean_mask(data, "absolute_action_mask", action_dim)
    normalization_mask = _boolean_mask(data, "action_normalization_mask", action_dim)
    if absolute_mask != [False] * 6 + [True]:
        raise ConfigError("Only the LIBERO gripper action may be absolute")
    if normalization_mask != [True] * 6 + [False]:
        raise ConfigError("The LIBERO gripper action must be excluded from normalization")

    image_keys = data.get("image_obs_keys")
    if image_keys != {
        "primary": "observation.images.image",
        "wrist": "observation.images.image2",
    }:
        raise ConfigError("LIBERO requires primary and wrist RGB observations")
    if data.get("state_obs_keys") != ["observation.state"]:
        raise ConfigError("LIBERO requires the converted 8-dimensional state field")
    if data.get("action_key") != "action":
        raise ConfigError("LIBERO requires the standard LeRobot action field")
    resize = data.get("resize")
    if resize != {"primary": [256, 256], "wrist": [128, 128]}:
        raise ConfigError("LIBERO resize must be primary=256 and wrist=128")

    model = config["model"]
    if int(model.get("pretrained_step", -1)) < 0:
        raise ConfigError("model.pretrained_step must be non-negative")
    if model.get("action_head") != "diffusion":
        raise ConfigError("The octo-small checkpoint requires its diffusion action head")
    if not bool(model.get("use_proprio")):
        raise ConfigError("LIBERO Octo-small requires the proprio tokenizer")
    if model.get("required_observation_tokenizers") != ["primary", "wrist"]:
        raise ConfigError("LIBERO must retain both primary and wrist tokenizers")

    train = config["train"]
    for name in (
        "gpu_count",
        "batch_size",
        "micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "max_steps",
        "episode_cache_size",
        "prefetch_factor",
    ):
        if int(train.get(name, 0)) <= 0:
            raise ConfigError(f"train.{name} must be positive")
    if int(train.get("num_workers_per_rank", -1)) < 0:
        raise ConfigError("train.num_workers_per_rank must be non-negative")
    gpu_ids = train.get("gpu_ids")
    if not isinstance(gpu_ids, list) or not all(
        isinstance(gpu_id, int) and gpu_id >= 0 for gpu_id in gpu_ids
    ):
        raise ConfigError("train.gpu_ids must be a list of non-negative integers")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ConfigError("train.gpu_ids must not contain duplicates")
    if len(gpu_ids) != int(train["gpu_count"]):
        raise ConfigError("train.gpu_ids length must equal train.gpu_count")
    if int(train["micro_batch_size_per_gpu"]) % 2:
        raise ConfigError("train.micro_batch_size_per_gpu must be even for 1:1 sampling")
    effective_batch_size = (
        int(train["micro_batch_size_per_gpu"])
        * int(train["gradient_accumulation_steps"])
        * int(train["gpu_count"])
    )
    if int(train["batch_size"]) != effective_batch_size:
        raise ConfigError(
            "train.batch_size must equal micro_batch_size_per_gpu * "
            "gradient_accumulation_steps * gpu_count"
        )
    if train.get("precision") != "bf16":
        raise ConfigError("Octo-small PyTorch training requires train.precision=bf16")

    learning_rate = train.get("learning_rate", {})
    if learning_rate.get("name") != "cosine":
        raise ConfigError("Octo training currently supports the cosine learning-rate schedule")
    if int(learning_rate.get("warmup_steps", -1)) < 0:
        raise ConfigError("train.learning_rate.warmup_steps must be non-negative")
    if int(learning_rate.get("decay_steps", 0)) < int(train["max_steps"]):
        raise ConfigError("learning-rate decay_steps must cover train.max_steps")


def apply_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    result = copy.deepcopy(config)
    mapping = {
        "model_path": ("paths", "model"),
        "lerobot_path": ("paths", "lerobot"),
        "output_dir": ("paths", "output"),
        "target_dataset": ("data", "target_dataset"),
        "gpu_ids": ("train", "gpu_ids"),
        "batch_size": ("train", "batch_size"),
        "max_steps": ("train", "max_steps"),
    }
    for name, value in overrides.items():
        if value is None or name not in mapping:
            continue
        section, key = mapping[name]
        result[section][key] = value
    if overrides.get("max_steps") is not None:
        result["train"]["learning_rate"]["decay_steps"] = max(
            int(result["train"]["learning_rate"]["decay_steps"]),
            int(overrides["max_steps"]),
        )
    validate_config(result)
    return result


def resolved_paths(config: dict[str, Any]) -> dict[str, Path]:
    config_path = Path(config["_config_path"])
    project_root = config_path.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    lerobot = resolve(config["paths"]["lerobot"])
    prior_dataset = lerobot / config["data"]["prior_dataset"]
    target_dataset = lerobot / config["data"]["target_dataset"]
    return {
        "project_root": project_root,
        "model": resolve(config["paths"]["model"]),
        "lerobot": lerobot,
        "prior_dataset": prior_dataset,
        "target_dataset": target_dataset,
        "statistics": prior_dataset / "meta" / "stats.json",
        "output": resolve(config["paths"]["output"]),
    }
