from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from qwen_vl_common.contracts import (
    ConfigError as BackboneConfigError,
    backbone_contract,
    validated_lora_target_modules,
)
from qwen_vl_common.schedules import LoraUpdateSchedule


class ConfigError(ValueError):
    """Raised when an OFT run configuration violates its fixed contract."""


def load_config(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    with target.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping: {target}")
    config["_config_path"] = str(target)
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
        "compile_qwen_backbone": ("model", "torch_compile", "backbone_enabled"),
        "compile_action_head": ("model", "torch_compile", "action_head_enabled"),
        "lora_rank": ("model", "lora", "rank"),
        "lora_alpha": ("model", "lora", "alpha"),
        "lora_dropout": ("model", "lora", "dropout"),
        "deepspeed_stage": ("train", "deepspeed_stage"),
        "lora_learning_rate": ("train", "lora_learning_rate"),
        "action_head_learning_rate": ("train", "head_learning_rate"),
        "lora_freeze_steps": ("train", "lora_freeze_steps"),
    }
    for name, value in overrides.items():
        if value is None or name not in mapping:
            continue
        destination = result
        for part in mapping[name][:-1]:
            destination = destination[part]
        destination[mapping[name][-1]] = value
    if int(result["train"]["deepspeed_stage"]) == 3:
        result["train"]["cpu_optimizer_offload"] = True
    validate_config(result)
    return result


def validate_config(config: dict[str, Any]) -> None:
    missing = {"paths", "data", "model", "train"}.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")
    data = config["data"]
    model = config["model"]
    train = config["train"]
    if data.get("dataset_type", "bridge") != "bridge":
        raise ConfigError("Qwen-VL OFT supports only the Bridge dataset")
    if data.get("state_dim") != 8 or data.get("action_dim") != 7:
        raise ConfigError("Bridge OFT requires state_dim=8 and action_dim=7")
    action_horizon = data.get("action_horizon")
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or action_horizon <= 0
    ):
        raise ConfigError("data.action_horizon must be a positive integer")

    if model.get("backbone_family", "qwen3_vl") != "qwen3_vl":
        raise ConfigError("Qwen-VL OFT requires model.backbone_family=qwen3_vl")
    lora = model.get("lora")
    if not isinstance(lora, dict):
        raise ConfigError("model.lora must be a mapping")
    for name in ("rank", "alpha"):
        value = lora.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"model.lora.{name} must be a positive integer")
    dropout = lora.get("dropout")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ConfigError("model.lora.dropout must be finite and in [0, 1)")
    try:
        contract = backbone_contract(model)
        validated_lora_target_modules(model)
    except BackboneConfigError as error:
        raise ConfigError(str(error)) from error
    if model.get("text_layers") != contract["text_layers"]:
        raise ConfigError(f"Qwen3-VL-4B requires model.text_layers={contract['text_layers']}")
    if model.get("context_dim") != contract["context_dim"]:
        raise ConfigError(f"Qwen3-VL-4B requires model.context_dim={contract['context_dim']}")
    if model.get("attn_implementation") != "sdpa":
        raise ConfigError("starVLA-compatible OFT requires model.attn_implementation=sdpa")
    if model.get("state_bins") != 256:
        raise ConfigError("starVLA-compatible OFT requires model.state_bins=256")
    if model.get("action_token") != "🔍":
        raise ConfigError("starVLA-compatible OFT requires model.action_token=🔍")
    if model.get("action_head_hidden_dim") != 2 * int(model["context_dim"]):
        raise ConfigError("model.action_head_hidden_dim must equal 2 * model.context_dim")
    forbidden = {"dit", "flow", "state_dropout_prob"}.intersection(model)
    if forbidden:
        raise ConfigError(f"OFT config contains GROOT-only model fields: {sorted(forbidden)}")
    if not isinstance(model.get("gradient_checkpointing"), bool):
        raise ConfigError("model.gradient_checkpointing must be a boolean")
    compile_config = model.get("torch_compile")
    if not isinstance(compile_config, dict):
        raise ConfigError("model.torch_compile must be a mapping")
    for name in ("enabled", "backbone_enabled", "action_head_enabled", "dynamic", "fullgraph"):
        if not isinstance(compile_config.get(name), bool):
            raise ConfigError(f"model.torch_compile.{name} must be a boolean")

    if train.get("deepspeed_stage") not in (2, 3):
        raise ConfigError("train.deepspeed_stage must be 2 or 3")
    if train.get("bf16") is not True:
        raise ConfigError("Qwen-VL OFT requires train.bf16=true")
    for name in ("gpu_count", "micro_batch_size", "gradient_accumulation_steps", "max_steps"):
        if isinstance(train.get(name), bool) or int(train.get(name, 0)) <= 0:
            raise ConfigError(f"train.{name} must be positive")
    gpu_ids = train.get("gpu_ids")
    if not isinstance(gpu_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in gpu_ids
    ):
        raise ConfigError("train.gpu_ids must be a list of non-negative integers")
    if len(gpu_ids) != len(set(gpu_ids)) or len(gpu_ids) != int(train["gpu_count"]):
        raise ConfigError("train.gpu_ids must be unique and match train.gpu_count")
    for name in ("lora_learning_rate", "head_learning_rate"):
        value = train.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ConfigError(f"train.{name} must be finite and positive")
    try:
        LoraUpdateSchedule.from_train_config(train)
    except (KeyError, ValueError) as error:
        raise ConfigError(str(error)) from error


def resolved_paths(config: dict[str, Any]) -> dict[str, Path]:
    config_path = Path(config["_config_path"])
    project_root = config_path.parent.parent

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else project_root / path).resolve()

    return {
        "project_root": project_root,
        "model": resolve(config["paths"]["model"]),
        "dataset": resolve(config["paths"]["dataset"]),
        "output": resolve(config["paths"]["output"]),
    }


def config_digest(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False, allow_unicode=True)
    temporary.replace(target)
