"""Configuration loading and validation for Bridge Octo-small training."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a Bridge training configuration is invalid."""


BRIDGE_V2_NORMALIZATION_CONTRACT = "bridge_v2_q99_binary_v1"


def load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    with target.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping: {target}")
    config["_config_path"] = str(target)
    validate_config(config)
    return config


def _positive_integer(section: dict[str, Any], name: str, prefix: str) -> int:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{prefix}.{name} must be a positive integer")
    return value


def validate_config(config: dict[str, Any]) -> None:
    missing = {"paths", "data", "model", "train"}.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")
    for name in ("model", "dataset"):
        value = config["paths"].get(name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"paths.{name} must be a non-empty path")
    output = config["paths"].get("output")
    if output is not None and (not isinstance(output, str) or not output.strip()):
        raise ConfigError("paths.output must be a non-empty path when present")

    data = config["data"]
    expected_data = {
        "dataset_name": "bridge_orig_1.0.0",
        "image_obs_keys": {"primary": "observation.images.image_0"},
        "state_obs_keys": ["observation.state"],
        "action_key": "action",
        "state_dim": 8,
        "action_dim": 7,
        "window_size": 1,
        "action_horizon": 8,
        "absolute_action_mask": [False] * 6 + [True],
        "action_normalization_mask": [True] * 6 + [False],
        "resize": {"primary": [256, 256]},
        "empty_task_policy": "exclude",
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise ConfigError(f"data.{key} must equal {expected!r}")
    if data.get("normalization_contract") != BRIDGE_V2_NORMALIZATION_CONTRACT:
        raise ConfigError(
            "data.normalization_contract must equal "
            f"{BRIDGE_V2_NORMALIZATION_CONTRACT!r}"
        )
    normalization_epsilon = data.get("normalization_epsilon")
    if (
        isinstance(normalization_epsilon, bool)
        or not isinstance(normalization_epsilon, (int, float))
        or not math.isfinite(float(normalization_epsilon))
        or float(normalization_epsilon) <= 0
    ):
        raise ConfigError("data.normalization_epsilon must be a finite positive number")
    expected_counts = data.get("expected_counts")
    if not isinstance(expected_counts, dict):
        raise ConfigError("data.expected_counts must be a mapping")
    for name in (
        "source_episodes",
        "retained_episodes",
        "excluded_empty_task_episodes",
        "retained_frames",
    ):
        _positive_integer(expected_counts, name, "data.expected_counts")

    model = config["model"]
    if model.get("required_observation_tokenizers") != ["primary"]:
        raise ConfigError("Bridge Octo-small requires only the primary tokenizer")
    if model.get("action_head") != "diffusion" or not model.get("use_proprio"):
        raise ConfigError("Bridge Octo-small requires proprio and the diffusion action head")
    if model.get("finetuning_mode") != "full":
        raise ConfigError("Bridge Octo-small currently supports full fine-tuning")

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
        _positive_integer(train, name, "train")
    if int(train["gpu_count"]) != 4:
        raise ConfigError("The 4x4090 Bridge route requires train.gpu_count=4")
    workers = train.get("num_workers_per_rank")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ConfigError("train.num_workers_per_rank must be a non-negative integer")
    gpu_ids = train.get("gpu_ids")
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ConfigError("train.gpu_ids must contain unique non-negative integers")
    if len(gpu_ids) != int(train["gpu_count"]):
        raise ConfigError("train.gpu_ids length must equal train.gpu_count")
    effective_batch = (
        int(train["gpu_count"])
        * int(train["micro_batch_size_per_gpu"])
        * int(train["gradient_accumulation_steps"])
    )
    if int(train["batch_size"]) != effective_batch:
        raise ConfigError(
            "train.batch_size must equal gpu_count * micro_batch_size_per_gpu * "
            "gradient_accumulation_steps"
        )
    if train.get("precision") != "bf16":
        raise ConfigError("Bridge Octo-small training requires BF16")
    for name in ("weight_decay", "max_grad_norm"):
        value = train.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ConfigError(f"train.{name} must be a finite non-negative number")
    learning_rate = train.get("learning_rate")
    if not isinstance(learning_rate, dict) or learning_rate.get("name") != "cosine":
        raise ConfigError("train.learning_rate must use the cosine schedule")
    for name in ("init_value", "peak_value", "end_value"):
        value = learning_rate.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ConfigError(f"train.learning_rate.{name} must be non-negative")
    warmup = learning_rate.get("warmup_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ConfigError("train.learning_rate.warmup_steps must be non-negative")


def apply_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    result = copy.deepcopy(config)
    mapping = {
        "model_path": ("paths", "model"),
        "dataset_path": ("paths", "dataset"),
        "output_dir": ("paths", "output"),
        "gpu_ids": ("train", "gpu_ids"),
        "max_steps": ("train", "max_steps"),
    }
    for name, value in overrides.items():
        if value is None or name not in mapping:
            continue
        section, key = mapping[name]
        result[section][key] = value
    if overrides.get("learning_rate") is not None:
        result["train"]["learning_rate"]["peak_value"] = overrides["learning_rate"]
    if overrides.get("warmup_steps") is not None:
        result["train"]["learning_rate"]["warmup_steps"] = overrides["warmup_steps"]
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

    output = config["paths"].get("output")
    if not isinstance(output, str) or not output.strip():
        raise ConfigError("An explicit --output-dir is required")
    dataset = resolve(config["paths"]["dataset"])
    return {
        "project_root": project_root,
        "model": resolve(config["paths"]["model"]),
        "dataset": dataset,
        "output": resolve(output),
        "normalization": resolve(output) / "normalization.json",
    }
