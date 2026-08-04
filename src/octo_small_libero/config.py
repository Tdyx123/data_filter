from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Sequence

import yaml

from .libero10_tasks import LIBERO_10_TASK_COUNT


class ConfigError(ValueError):
    """Raised when the independent Octo LIBERO configuration is invalid."""


def normalized_sample_weights(weights: Sequence[float]) -> tuple[float, float]:
    """Normalize target/prior sampling weights to probabilities."""
    if len(weights) != 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        for value in weights
    ):
        raise ValueError("sample weights must contain two positive finite numbers")
    total = sum(float(value) for value in weights)
    return (float(weights[0]) / total, float(weights[1]) / total)


def sample_counts_per_batch(
    weights: Sequence[float],
    batch_size: int,
) -> tuple[int, int]:
    """Resolve exact target/prior counts for one local micro-batch."""
    normalized = normalized_sample_weights(weights)
    raw_counts = tuple(batch_size * value for value in normalized)
    counts = tuple(int(round(value)) for value in raw_counts)
    if any(count <= 0 for count in counts) or any(
        not math.isclose(value, count, rel_tol=0.0, abs_tol=1.0e-9)
        for value, count in zip(raw_counts, counts, strict=True)
    ):
        raise ValueError(
            "data.sample_weights must produce positive whole-number counts for "
            "train.micro_batch_size_per_gpu"
        )
    return counts[0], counts[1]


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
    target_task_index = data.get("target_task_index")
    target_all_tasks = data.get("target_all_tasks", False)
    target_only = data.get("target_only", False)
    if not isinstance(target_all_tasks, bool):
        raise ConfigError("data.target_all_tasks must be a bool")
    if not isinstance(target_only, bool):
        raise ConfigError("data.target_only must be a bool")
    if target_task_index is not None and (
        isinstance(target_task_index, bool)
        or not isinstance(target_task_index, int)
        or not 0 <= target_task_index < LIBERO_10_TASK_COUNT
    ):
        raise ConfigError(
            f"data.target_task_index must be null or in [0, {LIBERO_10_TASK_COUNT - 1}]"
        )
    if target_task_index is not None and target_all_tasks:
        raise ConfigError(
            "data.target_task_index and data.target_all_tasks cannot both be enabled"
        )

    selection = data.get("prior_selection")
    if not isinstance(selection, dict):
        raise ConfigError("data.prior_selection must be a mapping")
    if (
        not isinstance(selection.get("scores"), str)
        or not selection["scores"].strip()
    ):
        raise ConfigError("data.prior_selection.scores must be a non-empty path")
    prefiltered = selection.get("prefiltered", False)
    if not isinstance(prefiltered, bool):
        raise ConfigError("data.prior_selection.prefiltered must be a bool")
    top_percent = selection.get("top_percent")
    if top_percent is not None:
        if (
            isinstance(top_percent, bool)
            or not isinstance(top_percent, (int, float))
            or not math.isfinite(float(top_percent))
            or not 0.0 < float(top_percent) <= 100.0
        ):
            raise ConfigError(
                "data.prior_selection.top_percent must be null or in (0, 100]"
            )
    if prefiltered and top_percent is not None:
        raise ConfigError(
            "data.prior_selection.prefiltered cannot be combined with top_percent"
        )

    weights = data.get("sample_weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in weights
        )
    ):
        raise ConfigError("data.sample_weights must contain two positive finite numbers")

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
        "log_every_steps",
        "save_every_steps",
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
    try:
        sample_counts_per_batch(weights, int(train["micro_batch_size_per_gpu"]))
    except ValueError as error:
        raise ConfigError(str(error)) from error
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
        "target_task_index": ("data", "target_task_index"),
        "target_all_tasks": ("data", "target_all_tasks"),
        "target_only": ("data", "target_only"),
        "sample_weights": ("data", "sample_weights"),
        "gpu_ids": ("train", "gpu_ids"),
        "batch_size": ("train", "batch_size"),
        "max_steps": ("train", "max_steps"),
    }
    for name, value in overrides.items():
        if value is None or name not in mapping:
            continue
        section, key = mapping[name]
        result[section][key] = value
    if overrides.get("target_all_tasks") is True:
        result["data"]["target_task_index"] = None
    elif overrides.get("target_task_index") is not None:
        result["data"]["target_all_tasks"] = False
    if overrides.get("prior_scores") is not None:
        result["data"]["prior_selection"]["scores"] = overrides["prior_scores"]
    if overrides.get("prior_prefiltered_scores") is not None:
        result["data"]["prior_selection"].update(
            {
                "scores": overrides["prior_prefiltered_scores"],
                "top_percent": None,
                "prefiltered": True,
            }
        )
    if overrides.get("prior_top_percent") is not None:
        result["data"]["prior_selection"]["top_percent"] = overrides[
            "prior_top_percent"
        ]
    if overrides.get("max_steps") is not None:
        result["train"]["learning_rate"]["decay_steps"] = max(
            int(result["train"]["learning_rate"]["decay_steps"]),
            int(overrides["max_steps"]),
        )
    validate_config(result)
    if overrides.get("output_dir") is None:
        output = Path(result["paths"]["output"])
        task_index = result["data"].get("target_task_index")
        if result["data"].get("target_all_tasks", False):
            task_suffix = "_all-tasks"
            if task_suffix not in output.name:
                output = output.with_name(output.name + task_suffix)
        elif task_index is not None:
            task_suffix = f"_task-{task_index}"
            if task_suffix not in output.name:
                output = output.with_name(output.name + task_suffix)
        if result["data"].get("target_only", False):
            target_only_suffix = "_target-only"
            if not output.name.endswith(target_only_suffix):
                output = output.with_name(output.name + target_only_suffix)
        top_percent = result["data"]["prior_selection"]["top_percent"]
        if top_percent is not None:
            tag = format(float(top_percent), ".12g").replace(".", "p")
            top_suffix = f"_top{tag}pct"
            if not output.name.endswith(top_suffix):
                output = output.with_name(output.name + top_suffix)
        result["paths"]["output"] = str(output)
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
    statistics_dataset = (
        target_dataset if config["data"].get("target_only", False) else prior_dataset
    )
    return {
        "project_root": project_root,
        "model": resolve(config["paths"]["model"]),
        "lerobot": lerobot,
        "prior_dataset": prior_dataset,
        "target_dataset": target_dataset,
        "statistics": statistics_dataset / "meta" / "stats.json",
        "prior_scores": resolve(config["data"]["prior_selection"]["scores"]),
        "output": resolve(config["paths"]["output"]),
    }
