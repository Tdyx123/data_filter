from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .schedules import LoraUpdateSchedule


class ConfigError(ValueError):
    """Raised when the run configuration violates a model/data invariant."""


BRIDGE_V2_NORMALIZATION_CONTRACT = "bridge_v2_q99_binary_v1"


def require_bridge_v2_normalization_contract(data_config: dict[str, Any]) -> str:
    contract = data_config.get("normalization_contract")
    if contract != BRIDGE_V2_NORMALIZATION_CONTRACT:
        raise ConfigError(
            "Bridge data.normalization_contract must be "
            f"{BRIDGE_V2_NORMALIZATION_CONTRACT!r}, found {contract!r}"
        )
    return BRIDGE_V2_NORMALIZATION_CONTRACT


BACKBONE_CONTRACTS: dict[str, dict[str, Any]] = {
    "qwen3_vl": {
        "display_name": "Qwen3-VL-4B",
        "architecture": "Qwen3VLForConditionalGeneration",
        "text_layers": 36,
        "context_dim": 2560,
        "layer_type_counts": {"full_attention": 36, "linear_attention": 0},
        "lora_targets": {
            "full_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
            "linear_attention": (),
            "mlp": ("gate_proj", "up_proj", "down_proj"),
        },
    },
    "qwen3_5": {
        "display_name": "Qwen3.5-0.8B",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "text_layers": 24,
        "context_dim": 1024,
        "layer_type_counts": {"full_attention": 6, "linear_attention": 18},
        "lora_targets": {
            "full_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
            "linear_attention": (
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ),
            "mlp": (),
        },
    },
}


def backbone_contract(model_config: dict[str, Any]) -> dict[str, Any]:
    family = model_config.get("backbone_family", "qwen3_vl")
    try:
        return BACKBONE_CONTRACTS[str(family)]
    except KeyError as error:
        raise ConfigError(
            f"model.backbone_family must be one of {sorted(BACKBONE_CONTRACTS)}"
        ) from error


def normalized_lora_target_modules(
    model_config: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    raw_targets = model_config["lora"]["target_modules"]
    if isinstance(raw_targets, list):
        targets = {
            "full_attention": tuple(raw_targets),
            "linear_attention": (),
            "mlp": (),
        }
    elif isinstance(raw_targets, dict):
        target_groups = ("full_attention", "linear_attention", "mlp")
        unknown = set(raw_targets).difference(target_groups)
        if unknown:
            raise ConfigError(f"Unknown model.lora.target_modules groups: {sorted(unknown)}")
        targets = {
            layer_type: tuple(raw_targets.get(layer_type, ()))
            for layer_type in target_groups
        }
    else:
        raise ConfigError("model.lora.target_modules must be a list or mapping")
    for layer_type, names in targets.items():
        if any(not isinstance(name, str) or not name for name in names):
            raise ConfigError(
                f"model.lora.target_modules.{layer_type} must contain non-empty strings"
            )
        if len(names) != len(set(names)):
            raise ConfigError(
                f"model.lora.target_modules.{layer_type} must not contain duplicates"
            )
    return targets


def validated_lora_target_modules(
    model_config: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    targets = normalized_lora_target_modules(model_config)
    contract = backbone_contract(model_config)
    if targets != contract["lora_targets"]:
        raise ConfigError(
            f"{contract['display_name']} requires LoRA targets {contract['lora_targets']}"
        )
    return targets


# Shared Qwen contracts are canonical; preserve the legacy import surface.
from qwen_vl_common.contracts import (  # noqa: E402
    BACKBONE_CONTRACTS,  # noqa: F401,F811
    ConfigError,  # noqa: F811
    backbone_contract,  # noqa: F811
    normalized_lora_target_modules,  # noqa: F401,F811
    validated_lora_target_modules,  # noqa: F811
)


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
        "lerobot_path": ("paths", "lerobot"),
        "output_dir": ("paths", "output"),
        "gpu_count": ("train", "gpu_count"),
        "gpu_ids": ("train", "gpu_ids"),
        "micro_batch_size": ("train", "micro_batch_size"),
        "gradient_accumulation_steps": ("train", "gradient_accumulation_steps"),
        "max_steps": ("train", "max_steps"),
        "compile_qwen_backbone": (
            "model",
            "torch_compile",
            "backbone_enabled",
        ),
        "compile_action_head": (
            "model",
            "torch_compile",
            "action_head_enabled",
        ),
        "episode_cache_size": ("data", "episode_cache_size"),
        "lora_rank": ("model", "lora", "rank"),
        "lora_alpha": ("model", "lora", "alpha"),
        "lora_dropout": ("model", "lora", "dropout"),
        "dit_layers": ("model", "dit", "num_layers"),
        "dit_hidden_size": ("model", "dit", "hidden_size"),
        "deepspeed_stage": ("train", "deepspeed_stage"),
        "lora_learning_rate": ("train", "lora_learning_rate"),
        "action_head_learning_rate": ("train", "head_learning_rate"),
        "lora_freeze_steps": ("train", "lora_freeze_steps"),
        "lora_cycle_steps": ("train", "lora_cycle_steps"),
        "lora_active_steps": ("train", "lora_active_steps"),
        "target_all_tasks": ("data", "target_all_tasks"),
        "target_only": ("data", "target_only"),
        "sample_weights": ("data", "sample_weights"),
    }
    for key, value in overrides.items():
        if value is None or key not in mapping:
            continue
        target = result
        keys = mapping[key]
        for part in keys[:-1]:
            target = target[part]
        target[keys[-1]] = value

    if overrides.get("prior_scores") is not None:
        result["data"]["prior_selection"]["scores"] = overrides["prior_scores"]
    if overrides.get("prior_prefiltered_scores") is not None:
        result["data"]["prior_selection"].update(
            {
                "scores": overrides["prior_prefiltered_scores"],
                "top_percent": None,
                "prefiltered": True,
                "relcore_manifest": None,
                "quality_filter_scores": None,
            }
        )
    if overrides.get("prior_top_percent") is not None:
        result["data"]["prior_selection"].update(
            {
                "top_percent": overrides["prior_top_percent"],
                "prefiltered": False,
                "relcore_manifest": None,
                "quality_filter_scores": None,
            }
        )
    if overrides.get("prior_relcore_manifest") is not None:
        result["data"]["prior_selection"].update(
            {
                "top_percent": None,
                "prefiltered": False,
                "relcore_manifest": overrides["prior_relcore_manifest"],
                "quality_filter_scores": None,
            }
        )
    if overrides.get("prior_quality_filter_scores") is not None:
        result["data"]["prior_selection"].update(
            {
                "top_percent": None,
                "prefiltered": False,
                "relcore_manifest": None,
                "quality_filter_scores": overrides["prior_quality_filter_scores"],
            }
        )

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
    dataset_type = data.get("dataset_type", "bridge")
    if dataset_type not in {"bridge", "libero"}:
        raise ConfigError("data.dataset_type must be bridge or libero")
    if data["state_dim"] != 8 or data["action_dim"] != 7:
        raise ConfigError(f"{dataset_type} requires state_dim=8 and action_dim=7")
    if data["action_horizon"] <= 0:
        raise ConfigError("action_horizon must be positive")
    if dataset_type == "bridge":
        require_bridge_v2_normalization_contract(data)
    contract = backbone_contract(model)
    expected_layers = int(contract["text_layers"])
    expected_context_dim = int(contract["context_dim"])
    if model["text_layers"] != expected_layers:
        if model.get("backbone_family", "qwen3_vl") == "qwen3_vl":
            raise ConfigError(
                "Qwen3-VL-4B uses every Qwen text layer and requires "
                f"model.text_layers={expected_layers}"
            )
        raise ConfigError(
            f"{contract['display_name']} requires model.text_layers={expected_layers}"
        )
    if model["context_dim"] != expected_context_dim:
        raise ConfigError(
            f"{contract['display_name']} requires model.context_dim={expected_context_dim}"
        )
    validated_lora_target_modules(model)
    if "context_forward" in model:
        raise ConfigError(
            "model.context_forward is no longer supported; "
            "Qwen GROOT always uses direct backbone context encoding"
        )
    if not isinstance(model["gradient_checkpointing"], bool):
        raise ConfigError("model.gradient_checkpointing must be a boolean")
    compile_config = model.get("torch_compile")
    if compile_config is not None:
        if not isinstance(compile_config, dict):
            raise ConfigError("model.torch_compile must be a mapping")
        if not isinstance(compile_config.get("enabled"), bool):
            raise ConfigError("model.torch_compile.enabled must be a boolean")
        for target_switch in ("backbone_enabled", "action_head_enabled"):
            if target_switch in compile_config and not isinstance(
                compile_config[target_switch], bool
            ):
                raise ConfigError(
                    f"model.torch_compile.{target_switch} must be a boolean"
                )
        if not isinstance(compile_config.get("backend"), str) or not compile_config["backend"]:
            raise ConfigError("model.torch_compile.backend must be a non-empty string")
        valid_compile_modes = {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }
        if compile_config.get("mode") not in valid_compile_modes:
            raise ConfigError(
                "model.torch_compile.mode must be one of "
                f"{sorted(valid_compile_modes)}"
            )
        for name in ("dynamic", "fullgraph"):
            if not isinstance(compile_config.get(name), bool):
                raise ConfigError(f"model.torch_compile.{name} must be a boolean")
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
    for name in ("lora_learning_rate", "head_learning_rate"):
        value = train.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ConfigError(f"train.{name} must be finite and positive")
    try:
        LoraUpdateSchedule.from_train_config(train)
    except (KeyError, ValueError) as error:
        raise ConfigError(str(error)) from error
    if dataset_type == "libero":
        if not isinstance(config["paths"].get("lerobot"), str):
            raise ConfigError("paths.lerobot must be set for LIBERO")
        if data.get("prior_dataset") != "libero90" or data.get("target_dataset") != "libero10_5":
            raise ConfigError("LIBERO requires prior=libero90 and target=libero10_5")
        if int(data.get("episode_cache_size", 0)) <= 0:
            raise ConfigError("data.episode_cache_size must be positive")
        weights = data.get("sample_weights")
        if not isinstance(weights, list) or len(weights) != 2:
            raise ConfigError("data.sample_weights must contain target and prior weights")
        prior = data.get("prior_selection")
        if not isinstance(prior, dict):
            raise ConfigError("data.prior_selection must be a mapping")
        top_percent = prior.get("top_percent")
        active_prior_modes = sum(
            (
                top_percent is not None,
                bool(prior.get("prefiltered", False)),
                prior.get("relcore_manifest") is not None,
                prior.get("quality_filter_scores") is not None,
            )
        )
        if active_prior_modes > 1:
            raise ConfigError("only one LIBERO prior selection mode may be configured")
        if top_percent is not None and (
            isinstance(top_percent, bool)
            or not isinstance(top_percent, (int, float))
            or not math.isfinite(float(top_percent))
            or not 0.0 < float(top_percent) <= 100.0
        ):
            raise ConfigError("data.prior_selection.top_percent must be in (0, 100]")
        from libero_lerobot.sampling import sample_counts_per_batch

        try:
            sample_counts_per_batch(
                (1.0,) if bool(data.get("target_only", False)) else weights,
                int(train["gpu_count"]) * int(train["micro_batch_size"]),
                scope="global micro-batch",
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error


def resolved_paths(config: dict[str, Any]) -> dict[str, Path]:
    config_path = Path(config["_config_path"])
    project_root = config_path.parent.parent

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate.resolve()

    output_value = config["paths"].get("output")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ConfigError("an explicit --output-dir is required")
    result = {
        "project_root": project_root,
        "model": resolve(config["paths"]["model"]),
        "output": resolve(output_value),
    }
    if config["data"].get("dataset_type", "bridge") == "bridge":
        result["dataset"] = resolve(config["paths"]["dataset"])
        return result
    lerobot = resolve(config["paths"]["lerobot"])
    result.update(
        {
            "lerobot": lerobot,
            "target_dataset": lerobot / config["data"]["target_dataset"],
            "prior_dataset": lerobot / config["data"]["prior_dataset"],
            "prior_scores": resolve(config["data"]["prior_selection"]["scores"]),
        }
    )
    relcore = config["data"]["prior_selection"].get("relcore_manifest")
    if relcore is not None:
        result["prior_relcore_manifest"] = resolve(relcore)
    quality = config["data"]["prior_selection"].get("quality_filter_scores")
    if quality is not None:
        result["prior_quality_filter_scores"] = resolve(quality)
    return result


def config_digest(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
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
