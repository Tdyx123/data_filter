from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the run configuration violates a model/data invariant."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping: {path}")
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    mapping = {
        "model_path": ("paths", "model"),
        "dataset_path": ("paths", "dataset"),
        "output_dir": ("paths", "output"),
        "gpu_count": ("train", "gpu_count"),
        "gpu_ids": ("train", "gpu_ids"),
        "micro_batch_size": ("train", "micro_batch_size"),
        "gradient_accumulation_steps": ("train", "gradient_accumulation_steps"),
        "max_steps": ("train", "max_steps"),
        "lora_rank": ("model", "lora", "rank"),
        "lora_alpha": ("model", "lora", "alpha"),
        "lora_dropout": ("model", "lora", "dropout"),
        "dit_layers": ("model", "dit", "num_layers"),
        "dit_hidden_size": ("model", "dit", "hidden_size"),
        "deepspeed_stage": ("train", "deepspeed_stage"),
    }
    for key, value in overrides.items():
        if value is None or key not in mapping:
            continue
        target = result
        keys = mapping[key]
        for part in keys[:-1]:
            target = target[part]
        target[keys[-1]] = value

    if result["train"]["deepspeed_stage"] == 3:
        result["train"]["cpu_optimizer_offload"] = True
    validate_config(result)
    return result


def validate_config(config: dict[str, Any]) -> None:
    required = {"paths", "data", "model", "train"}
    missing = required.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")

    data = config["data"]
    model = config["model"]
    train = config["train"]
    if data["state_dim"] != 8 or data["action_dim"] != 7:
        raise ConfigError("Bridge WidowX requires state_dim=8 and action_dim=7")
    if data["action_horizon"] <= 0:
        raise ConfigError("action_horizon must be positive")
    if model["text_layers"] != 36:
        raise ConfigError(
            "This project deliberately uses every Qwen text layer; model.text_layers must be 36"
        )
    if model["context_dim"] != 2560:
        raise ConfigError("Qwen3-VL-4B context_dim must be 2560")
    if model["dit"]["hidden_size"] % model["dit"]["num_heads"]:
        raise ConfigError("DiT hidden size must be divisible by the number of heads")
    if not 0.0 <= model["state_dropout_prob"] <= 1.0:
        raise ConfigError("state_dropout_prob must be in [0, 1]")
    if train["deepspeed_stage"] not in (2, 3):
        raise ConfigError("deepspeed_stage must be 2 or 3")
    gpu_ids = train.get("gpu_ids")
    if gpu_ids is not None:
        if not isinstance(gpu_ids, list) or not all(
            isinstance(gpu_id, int) and gpu_id >= 0 for gpu_id in gpu_ids
        ):
            raise ConfigError("train.gpu_ids must be a list of non-negative integers")
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ConfigError("train.gpu_ids must not contain duplicate GPU numbers")
        if len(gpu_ids) != int(train["gpu_count"]):
            raise ConfigError(
                "train.gpu_ids length must equal train.gpu_count; "
                f"found {len(gpu_ids)} IDs for {train['gpu_count']} GPUs"
            )
    for name in (
        "gpu_count",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
    ):
        if int(train[name]) <= 0:
            raise ConfigError(f"train.{name} must be positive")


def resolved_paths(config: dict[str, Any]) -> dict[str, Path]:
    config_path = Path(config["_config_path"])
    project_root = config_path.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    return {
        "project_root": project_root,
        "model": resolve(config["paths"]["model"]),
        "dataset": resolve(config["paths"]["dataset"]),
        "output": resolve(config["paths"]["output"]),
    }


def config_digest(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resume_config_digest(config: dict[str, Any]) -> str:
    """Hash state-defining fields while allowing a run to be extended safely."""
    clean = copy.deepcopy(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    clean["paths"].pop("output", None)
    for key in ("num_workers", "prefetch_factor", "video_cache_episodes"):
        clean["data"].pop(key, None)
    for key in (
        "max_steps",
        "log_every_steps",
        "eval_every_steps",
        "save_every_steps",
        "validation_batches",
        "keep_last_checkpoints",
        "gpu_ids",
    ):
        clean["train"].pop(key, None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(target)
