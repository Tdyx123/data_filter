from __future__ import annotations

from typing import Any


class ConfigError(ValueError):
    """Raised when a shared Qwen backbone contract is invalid."""


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
            raise ConfigError(
                f"Unknown model.lora.target_modules groups: {sorted(unknown)}"
            )
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


__all__ = [
    "BACKBONE_CONTRACTS",
    "ConfigError",
    "backbone_contract",
    "normalized_lora_target_modules",
    "validated_lora_target_modules",
]
