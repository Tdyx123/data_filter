from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.autograd.profiler import record_function

from .config import (
    BACKBONE_CONTRACTS,
    BRIDGE_V2_NORMALIZATION_CONTRACT,
    ConfigError,
    backbone_contract,
    validated_lora_target_modules,
)
from .flow import (
    FlowMatchingActionHead,
    euler_denoise,
    masked_velocity_mse,
    sample_flow_batch,
)
from .normalization import QuantileStats


class ModelContractError(RuntimeError):
    """Raised when a local Qwen checkpoint violates its declared backbone contract."""


_TEXT_LAYER_PATH = re.compile(
    r"(?:^|\.)language_model\.layers\.(\d+)\."
    r"(?:self_attn|linear_attn|mlp)\.([^.]+)$"
)
_TEXT_LORA_PARAMETER_PATH = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\."
    r"(?:self_attn|linear_attn|mlp)\.[^.]+\.lora_"
)

ACTION_HEAD_CONTEXT_BUCKETS = (96, 192, 384, 512)


def _configure_qwen_gradient_checkpointing(
    backbone: nn.Module,
    *,
    enabled: bool,
) -> None:
    if enabled:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Frozen embeddings still need grad-carrying outputs so checkpointed LoRA
        # modules receive gradients.
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
        # Transformers 5.2 misroutes Qwen3.5's packed vision tokens through
        # FlashAttention varlen backward. Keep the safe PyTorch path for this
        # family while preserving Qwen3-VL's existing auto-selection behavior.
        if backbone_family == "qwen3_5":
            return "sdpa"
        return "flash_attention_2" if _flash_attention_available() else "sdpa"
    if requested not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"Unknown attention implementation: {requested}")
    if requested == "flash_attention_2" and not _flash_attention_available():
        raise ModelContractError(
            "FlashAttention 2 was requested but is not installed; use attn_implementation=sdpa"
        )
    return requested


def _family_from_architectures(architectures: Any) -> str:
    if not isinstance(architectures, list):
        raise ModelContractError(f"Qwen architectures must be a list, found {architectures!r}")
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
            f"LoRA was not installed on every configured text projection of all "
            f"{expected_layers} text layers. "
            f"{detail}; unexpected layers={unexpected}"
        )


def assert_qwen_freeze_contract(model: nn.Module) -> None:
    violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            "lora_" not in name
            or _TEXT_LORA_PARAMETER_PATH.search(name) is None
        )
    ]
    if violations:
        raise ModelContractError(
            "Only text attention/MLP LoRA parameters may be trainable; violations: "
            + ", ".join(violations[:10])
        )


def resolve_compile_targets(model_config: dict[str, Any]) -> tuple[bool, bool]:
    """Resolve target switches while preserving the legacy global flag."""
    compile_config = model_config.get("torch_compile")
    if not compile_config:
        return False, False
    legacy_enabled = bool(compile_config["enabled"])
    return (
        bool(compile_config.get("backbone_enabled", legacy_enabled)),
        bool(compile_config.get("action_head_enabled", legacy_enabled)),
    )


def compile_policy_modules(policy: Any, model_config: dict[str, Any]) -> None:
    """Compile independently selected Qwen and DiT modules in place."""
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
        if not compile_kwargs["dynamic"]:
            policy.enable_static_action_head_context_buckets()
        policy.action_head.compile(**compile_kwargs)


class Qwen3VLGrootPolicy(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        processor: Any,
        stats: QuantileStats,
        config: dict[str, Any],
    ):
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.config = config
        data = config["data"]
        model = config["model"]
        flow = model["flow"]
        self.state_dropout_prob = float(model["state_dropout_prob"])
        self.beta_alpha = float(flow["beta_alpha"])
        self.beta_beta = float(flow["beta_beta"])
        self.noise_s = float(flow["noise_s"])
        self.default_denoising_steps = int(flow["denoising_steps"])
        self.max_context_tokens = int(model["max_context_tokens"])
        self.context_dim = int(model["context_dim"])
        self.context_forward = str(model.get("context_forward", "causal_lm"))
        self.normalization_contract = data.get("normalization_contract")
        self._static_action_head_context_buckets_enabled = False

        dit = model["dit"]
        self.action_head = FlowMatchingActionHead(
            state_dim=int(data["state_dim"]),
            action_dim=int(data["action_dim"]),
            horizon=int(data["action_horizon"]),
            context_dim=int(model["context_dim"]),
            hidden_size=int(dit["hidden_size"]),
            num_layers=int(dit["num_layers"]),
            num_heads=int(dit["num_heads"]),
            mlp_ratio=int(dit["mlp_ratio"]),
            dropout=float(dit["dropout"]),
            gradient_checkpointing=bool(model["gradient_checkpointing"]),
        )
        self.register_buffer(
            "state_q01", torch.as_tensor(stats.state_q01, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "state_q99", torch.as_tensor(stats.state_q99, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "action_q01",
            torch.as_tensor(stats.action_q01, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "action_q99",
            torch.as_tensor(stats.action_q99, dtype=torch.float32),
            persistent=True,
        )
        self.normalization_epsilon = float(stats.epsilon)

    @classmethod
    def from_local_qwen(
        cls,
        *,
        model_path: str | Path,
        stats: QuantileStats,
        config: dict[str, Any],
    ) -> "Qwen3VLGrootPolicy":
        backbone, processor = load_qwen_backbone(model_path, config["model"])
        return cls(backbone=backbone, processor=processor, stats=stats, config=config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.action_head.parameters()).dtype

    @staticmethod
    def _safe_span(low: torch.Tensor, high: torch.Tensor, epsilon: float) -> torch.Tensor:
        span = high - low
        return torch.where(span.abs() < epsilon, torch.ones_like(span), span)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        low = self.state_q01.to(device=state.device, dtype=state.dtype)
        high = self.state_q99.to(device=state.device, dtype=state.dtype)
        normalized = 2.0 * (state - low) / self._safe_span(low, high, self.normalization_epsilon) - 1.0
        constant = (high - low).abs() < self.normalization_epsilon
        normalized = torch.where(constant, torch.zeros_like(normalized), normalized)
        if self.normalization_contract == BRIDGE_V2_NORMALIZATION_CONTRACT:
            continuous = normalized[..., :-1].clamp(-2.2, 2.2)
            gripper = (state[..., -1:] > 0.5).to(state.dtype)
            return torch.cat([continuous, gripper], dim=-1)
        return normalized.clamp(-1.0, 1.0)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_q01.to(device=action.device, dtype=action.dtype)
        high = self.action_q99.to(device=action.device, dtype=action.dtype)
        normalized = 2.0 * (action - low) / self._safe_span(low, high, self.normalization_epsilon) - 1.0
        constant = (high - low).abs() < self.normalization_epsilon
        normalized = torch.where(constant, torch.zeros_like(normalized), normalized)
        if self.normalization_contract == BRIDGE_V2_NORMALIZATION_CONTRACT:
            continuous = normalized[..., :-1].clamp(-2.2, 2.2)
            gripper = (action[..., -1:] > 0.5).to(action.dtype)
            return torch.cat([continuous, gripper], dim=-1)
        return normalized.clamp(-1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_q01.to(device=action.device, dtype=action.dtype)
        high = self.action_q99.to(device=action.device, dtype=action.dtype)
        result = (action + 1.0) * 0.5 * self._safe_span(
            low, high, self.normalization_epsilon
        ) + low
        constant = (high - low).abs() < self.normalization_epsilon
        result = torch.where(constant, low.expand_as(result), result)
        if self.normalization_contract == BRIDGE_V2_NORMALIZATION_CONTRACT:
            gripper = (action[..., -1:] > 0.5).to(action.dtype)
            return torch.cat([result[..., :-1], gripper], dim=-1)
        return result

    def set_lora_trainable(self, enabled: bool) -> None:
        for name, parameter in self.backbone.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(enabled)
            elif parameter.requires_grad:
                parameter.requires_grad_(False)

    def lora_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.backbone.named_parameters()
            if "lora_" in name
        ]

    def action_head_parameters(self) -> list[nn.Parameter]:
        return list(self.action_head.parameters())

    def enable_static_action_head_context_buckets(self) -> None:
        self._static_action_head_context_buckets_enabled = True

    def _bucket_action_head_context(
        self,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3:
            raise ValueError(
                "context must have shape [batch, sequence, feature], "
                f"found {tuple(context.shape)}"
            )
        if context_attention_mask.ndim != 2:
            raise ValueError(
                "context_attention_mask must have shape [batch, sequence], "
                f"found {tuple(context_attention_mask.shape)}"
            )
        if context.shape[0] != context_attention_mask.shape[0]:
            raise ValueError(
                "context and context_attention_mask batch dimensions differ: "
                f"{context.shape[0]} != {context_attention_mask.shape[0]}"
            )
        if context.shape[1] != context_attention_mask.shape[1]:
            raise ValueError(
                "context and context_attention_mask sequence dimensions differ: "
                f"{context.shape[1]} != {context_attention_mask.shape[1]}"
            )
        if context.shape[2] != self.context_dim:
            raise ValueError(
                "context feature dimension differs from model.context_dim: "
                f"{context.shape[2]} != {self.context_dim}"
            )

        context_tokens = context.shape[1]
        bucket = next(
            (
                candidate
                for candidate in ACTION_HEAD_CONTEXT_BUCKETS
                if candidate >= context_tokens
            ),
            None,
        )
        if bucket is None:
            raise ValueError(
                f"context length {context_tokens} exceeds largest supported "
                f"action-head context bucket {ACTION_HEAD_CONTEXT_BUCKETS[-1]}"
            )
        padding_tokens = bucket - context_tokens
        if padding_tokens == 0:
            return context, context_attention_mask

        context_padding = context.new_zeros(
            context.shape[0],
            padding_tokens,
            context.shape[2],
        )
        mask_padding = context_attention_mask.new_zeros(
            context_attention_mask.shape[0],
            padding_tokens,
        )
        return (
            torch.cat((context, context_padding), dim=1),
            torch.cat((context_attention_mask, mask_padding), dim=1),
        )

    def _prepare_action_head_context(
        self,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = context.to(dtype=self.compute_dtype)
        if self._static_action_head_context_buckets_enabled:
            return self._bucket_action_head_context(
                context,
                context_attention_mask,
            )
        return context, context_attention_mask

    def _prepare_context_inputs(
        self,
        images: Sequence[Any],
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        with record_function("processor"):
            if len(images) != len(instructions):
                raise ValueError("images and instructions have different batch sizes")
            prompts = []
            for instruction in instructions:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction},
                        ],
                    }
                ]
                prompts.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            inputs = self.processor(
                text=prompts,
                images=list(images),
                padding=True,
                return_tensors="pt",
            )
            return {
                key: value.to(self.device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }

    def _forward_context_model(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with record_function("qwen_backbone"):
            if self.context_forward == "causal_lm":
                outputs = self.backbone(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                return outputs.hidden_states[-1]

            if self.context_forward != "backbone":
                raise ModelContractError(
                    f"Unknown Qwen context forward mode: {self.context_forward}"
                )
            if not hasattr(self.backbone, "get_base_model"):
                raise ModelContractError(
                    "Direct Qwen context forward requires a PEFT model with get_base_model()"
                )
            conditional_generation = self.backbone.get_base_model()
            context_model = getattr(conditional_generation, "model", None)
            if not isinstance(context_model, nn.Module):
                raise ModelContractError(
                    "Direct Qwen context forward could not locate the inner multimodal model"
                )
            outputs = context_model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
            return outputs.last_hidden_state

    def encode_context(
        self,
        images: Sequence[Any],
        instructions: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._prepare_context_inputs(images, instructions)
        lora_requires_grad = self.training and any(
            parameter.requires_grad for parameter in self.lora_parameters()
        )
        gradient_context = nullcontext() if lora_requires_grad else torch.no_grad()
        with gradient_context:
            context = self._forward_context_model(inputs)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                context.shape[:2], dtype=torch.bool, device=context.device
            )
        if context.shape[1] > self.max_context_tokens:
            context = context[:, -self.max_context_tokens :]
            attention_mask = attention_mask[:, -self.max_context_tokens :]
        return context, attention_mask.bool()

    def flow_loss_from_context(
        self,
        *,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        context, context_attention_mask = self._prepare_action_head_context(
            context,
            context_attention_mask,
        )
        state = self.normalize_state(state.to(self.device, dtype=self.compute_dtype))
        actions = self.normalize_action(actions.to(self.device, dtype=self.compute_dtype))
        action_mask = action_mask.to(self.device)
        if self.training and self.state_dropout_prob > 0:
            keep = (
                torch.rand(state.shape[0], 1, device=state.device)
                >= self.state_dropout_prob
            )
            state = state * keep.to(state.dtype)
        noisy, timestep, velocity = sample_flow_batch(
            actions,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
            noise_s=self.noise_s,
        )
        with record_function("action_head"):
            prediction = self.action_head(
                noisy,
                state,
                timestep,
                context,
                context_attention_mask,
            )
        return masked_velocity_mse(prediction, velocity, action_mask)

    def forward(
        self,
        *,
        images: Sequence[Any],
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        instructions: Sequence[str],
    ) -> torch.Tensor:
        context, context_mask = self.encode_context(images, instructions)
        return self.flow_loss_from_context(
            context=context,
            context_attention_mask=context_mask,
            state=state,
            actions=actions,
            action_mask=action_mask,
        )

    @torch.no_grad()
    def predict_actions(
        self,
        image: Any | Sequence[Any],
        state: np.ndarray | torch.Tensor | Sequence[float],
        instruction: str | Sequence[str],
        denoising_steps: int | None = None,
        *,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        instructions = (
            list(instruction) if isinstance(instruction, (list, tuple)) else [instruction]
        )
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if len(images) != states.shape[0] or len(instructions) != states.shape[0]:
            raise ValueError("Image, state, and instruction batch sizes differ")

        was_training = self.training
        self.eval()
        context, context_mask = self.encode_context(images, instructions)
        context, context_mask = self._prepare_action_head_context(
            context,
            context_mask,
        )
        normalized_state = self.normalize_state(states).to(dtype=self.compute_dtype)
        normalized_actions = euler_denoise(
            self.action_head,
            state=normalized_state,
            context=context,
            context_attention_mask=context_mask,
            steps=denoising_steps or self.default_denoising_steps,
            noise_s=self.noise_s,
            initial_noise=initial_noise,
            generator=generator,
        )
        actions = self.denormalize_action(normalized_actions).float()
        if was_training:
            self.train()
        return actions

    def compact_parameter_names(self) -> list[str]:
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith("action_head.") or "lora_" in name
        ]
