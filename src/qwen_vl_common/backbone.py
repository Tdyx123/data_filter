from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .contracts import (
    BACKBONE_CONTRACTS,
    ConfigError,
    backbone_contract,
    validated_lora_target_modules,
)


class ModelContractError(RuntimeError):
    """Raised when a local Qwen checkpoint violates its declared contract."""


_TEXT_LAYER_PATH = re.compile(
    r"(?:^|\.)language_model\.layers\.(\d+)\."
    r"(?:self_attn|linear_attn|mlp)\.([^.]+)$"
)
_TEXT_LORA_PARAMETER_PATH = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\."
    r"(?:self_attn|linear_attn|mlp)\.[^.]+\.lora_"
)


def _configure_qwen_gradient_checkpointing(
    backbone: nn.Module,
    *,
    enabled: bool,
) -> None:
    if enabled:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    elif hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()


def _flash_attention_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available

        return bool(is_flash_attn_2_available())
    except (ImportError, RuntimeError):
        return False


def select_attention_implementation(
    requested: str,
    *,
    backbone_family: str = "qwen3_vl",
) -> str:
    if requested == "auto":
        if backbone_family == "qwen3_5":
            return "sdpa"
        return "flash_attention_2" if _flash_attention_available() else "sdpa"
    if requested not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"Unknown attention implementation: {requested}")
    if requested == "flash_attention_2" and not _flash_attention_available():
        raise ModelContractError(
            "FlashAttention 2 was requested but is not installed; "
            "use attn_implementation=sdpa"
        )
    return requested


def _family_from_architectures(architectures: Any) -> str:
    if not isinstance(architectures, list):
        raise ModelContractError(
            f"Qwen architectures must be a list, found {architectures!r}"
        )
    matches = [
        family
        for family, contract in BACKBONE_CONTRACTS.items()
        if contract["architecture"] in architectures
    ]
    if len(matches) != 1:
        expected = sorted(
            str(contract["architecture"])
            for contract in BACKBONE_CONTRACTS.values()
        )
        raise ModelContractError(
            f"Expected exactly one supported Qwen architecture from {expected}, "
            f"found {architectures}"
        )
    return matches[0]


def _layer_types_from_qwen_config(
    config: dict[str, Any],
    *,
    family: str,
) -> tuple[str, ...]:
    text = config["text_config"]
    layer_count = int(text["num_hidden_layers"])
    if family == "qwen3_vl":
        return ("full_attention",) * layer_count
    raw_layer_types = text.get("layer_types")
    if not isinstance(raw_layer_types, list) or len(raw_layer_types) != layer_count:
        raise ModelContractError(
            f"Qwen3.5 text_config.layer_types must contain {layer_count} entries"
        )
    layer_types = tuple(str(layer_type) for layer_type in raw_layer_types)
    unknown = set(layer_types).difference({"full_attention", "linear_attention"})
    if unknown:
        raise ModelContractError(f"Unknown Qwen3.5 layer types: {sorted(unknown)}")
    return layer_types


def inspect_qwen_config(
    model_path: str | Path,
    *,
    expected_family: str | None = None,
) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve() / "config.json"
    if not path.is_file():
        raise ModelContractError(f"Missing Qwen config: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ModelContractError(f"Invalid Qwen config JSON: {path}") from error
    if not isinstance(config, dict):
        raise ModelContractError(f"Qwen config must be a JSON object: {path}")
    family = _family_from_architectures(config.get("architectures", []))
    if expected_family is not None and family != expected_family:
        raise ModelContractError(
            f"Expected backbone family {expected_family}, found {family}"
        )
    try:
        contract = backbone_contract({"backbone_family": family})
    except ConfigError as error:
        raise ModelContractError(str(error)) from error
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ModelContractError("Qwen config must contain a text_config mapping")
    expected_layers = int(contract["text_layers"])
    expected_context_dim = int(contract["context_dim"])
    try:
        actual_layers = int(text.get("num_hidden_layers", -1))
        actual_context_dim = int(text.get("hidden_size", -1))
    except (TypeError, ValueError) as error:
        raise ModelContractError(
            "Qwen text_config num_hidden_layers and hidden_size must be integers"
        ) from error
    if actual_layers != expected_layers:
        raise ModelContractError(
            f"Expected all {expected_layers} {contract['display_name']} text layers, "
            f"found {text.get('num_hidden_layers')}"
        )
    if actual_context_dim != expected_context_dim:
        raise ModelContractError(
            f"Expected {contract['display_name']} hidden_size={expected_context_dim}, "
            f"found {text.get('hidden_size')}"
        )
    layer_types = _layer_types_from_qwen_config(config, family=family)
    actual_counts = {
        layer_type: layer_types.count(layer_type)
        for layer_type in ("full_attention", "linear_attention")
    }
    if actual_counts != contract["layer_type_counts"]:
        raise ModelContractError(
            f"Expected {contract['display_name']} layer type counts "
            f"{contract['layer_type_counts']}, found {actual_counts}"
        )
    return config


def lora_target_pattern(model_config: dict[str, Any]) -> str:
    targets = validated_lora_target_modules(model_config)
    branches = []
    if targets["full_attention"]:
        names = "|".join(re.escape(name) for name in targets["full_attention"])
        branches.append(rf"self_attn\.(?:{names})")
    if targets["linear_attention"]:
        names = "|".join(re.escape(name) for name in targets["linear_attention"])
        branches.append(rf"linear_attn\.(?:{names})")
    if targets["mlp"]:
        names = "|".join(re.escape(name) for name in targets["mlp"])
        branches.append(rf"mlp\.(?:{names})")
    if not branches:
        raise ModelContractError("At least one text LoRA target module is required")
    return rf".*language_model\.layers\.\d+\.(?:{'|'.join(branches)})$"


def load_qwen_backbone(
    model_path: str | Path,
    model_config: dict[str, Any],
) -> tuple[nn.Module, Any]:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    target_pattern = lora_target_pattern(model_config)
    family = str(model_config.get("backbone_family", "qwen3_vl"))
    qwen_config = inspect_qwen_config(model_path, expected_family=family)
    attention_implementation = select_attention_implementation(
        model_config["attn_implementation"],
        backbone_family=family,
    )
    backbone = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=attention_implementation,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.config.use_cache = False
    _configure_qwen_gradient_checkpointing(
        backbone,
        enabled=bool(model_config["gradient_checkpointing"]),
    )
    lora = model_config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        target_modules=target_pattern,
    )
    backbone = get_peft_model(backbone, peft_config)
    layer_types = _layer_types_from_qwen_config(qwen_config, family=family)
    assert_full_lora_coverage(
        backbone,
        layer_types=layer_types,
        targets_by_layer_type=validated_lora_target_modules(model_config),
    )
    assert_qwen_freeze_contract(backbone)
    return backbone, processor


def lora_coverage(model: nn.Module) -> dict[int, set[str]]:
    coverage: dict[int, set[str]] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        match = _TEXT_LAYER_PATH.search(name)
        if match:
            coverage.setdefault(int(match.group(1)), set()).add(match.group(2))
    return coverage


def assert_full_lora_coverage(
    model: nn.Module,
    *,
    expected_layers: int = 36,
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    mlp_targets: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
    layer_types: Sequence[str] | None = None,
    targets_by_layer_type: dict[str, tuple[str, ...]] | None = None,
) -> None:
    if layer_types is None:
        layer_types = ("full_attention",) * expected_layers
    else:
        layer_types = tuple(layer_types)
        expected_layers = len(layer_types)
    if targets_by_layer_type is None:
        targets_by_layer_type = {
            "full_attention": tuple(targets),
            "linear_attention": (),
            "mlp": tuple(mlp_targets),
        }
    coverage = lora_coverage(model)
    missing: list[str] = []
    for layer, layer_type in enumerate(layer_types):
        if layer_type not in targets_by_layer_type:
            raise ModelContractError(f"Missing LoRA target group for {layer_type}")
        expected_targets = set(targets_by_layer_type[layer_type])
        expected_targets.update(targets_by_layer_type.get("mlp", ()))
        absent = expected_targets.difference(coverage.get(layer, set()))
        if absent:
            missing.append(f"layer {layer}: {sorted(absent)}")
    unexpected = sorted(layer for layer in coverage if layer >= expected_layers)
    if missing or unexpected:
        detail = "; ".join(missing[:8])
        if len(missing) > 8:
            detail += f"; ... ({len(missing)} layers incomplete)"
        raise ModelContractError(
            "LoRA was not installed on every configured text projection of all "
            f"{expected_layers} text layers. {detail}; unexpected layers={unexpected}"
        )


def assert_qwen_freeze_contract(model: nn.Module) -> None:
    violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ("lora_" not in name or _TEXT_LORA_PARAMETER_PATH.search(name) is None)
    ]
    if violations:
        raise ModelContractError(
            "Only text attention/MLP LoRA parameters may be trainable; violations: "
            + ", ".join(violations[:10])
        )


def resolve_compile_targets(model_config: dict[str, Any]) -> tuple[bool, bool]:
    compile_config = model_config.get("torch_compile")
    if not compile_config:
        return False, False
    legacy_enabled = bool(compile_config["enabled"])
    return (
        bool(compile_config.get("backbone_enabled", legacy_enabled)),
        bool(compile_config.get("action_head_enabled", legacy_enabled)),
    )


def compile_policy_modules(policy: Any, model_config: dict[str, Any]) -> None:
    compile_config = model_config.get("torch_compile")
    compile_backbone, compile_action_head = resolve_compile_targets(model_config)
    if not compile_config or not (compile_backbone or compile_action_head):
        return
    compile_kwargs = {
        "backend": str(compile_config["backend"]),
        "mode": str(compile_config["mode"]),
        "dynamic": bool(compile_config["dynamic"]),
        "fullgraph": bool(compile_config["fullgraph"]),
    }
    if compile_backbone:
        policy.backbone.compile(**compile_kwargs)
    if compile_action_head:
        enable_static_buckets = getattr(
            policy,
            "enable_static_action_head_context_buckets",
            None,
        )
        if not compile_kwargs["dynamic"] and enable_static_buckets is not None:
            enable_static_buckets()
        policy.action_head.compile(**compile_kwargs)


__all__ = [
    "ModelContractError",
    "assert_full_lora_coverage",
    "assert_qwen_freeze_contract",
    "compile_policy_modules",
    "inspect_qwen_config",
    "load_qwen_backbone",
    "lora_coverage",
    "lora_target_pattern",
    "resolve_compile_targets",
    "select_attention_implementation",
]
