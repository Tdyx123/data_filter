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

from .flow import (
    FlowMatchingActionHead,
    euler_denoise,
    masked_velocity_mse,
    sample_flow_batch,
)
from .normalization import QuantileStats


class ModelContractError(RuntimeError):
    """Raised when the local Qwen checkpoint is not the expected 36-layer model."""


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


def select_attention_implementation(requested: str) -> str:
    if requested == "auto":
        return "flash_attention_2" if _flash_attention_available() else "sdpa"
    if requested not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"Unknown attention implementation: {requested}")
    if requested == "flash_attention_2" and not _flash_attention_available():
        raise ModelContractError(
            "FlashAttention 2 was requested but is not installed; use attn_implementation=sdpa"
        )
    return requested


def inspect_qwen_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve() / "config.json"
    if not path.is_file():
        raise ModelContractError(f"Missing Qwen config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    architecture = config.get("architectures", [])
    text = config.get("text_config", {})
    if "Qwen3VLForConditionalGeneration" not in architecture:
        raise ModelContractError(f"Expected Qwen3VLForConditionalGeneration, found {architecture}")
    if int(text.get("num_hidden_layers", -1)) != 36:
        raise ModelContractError(
            f"Expected all 36 Qwen text layers, found {text.get('num_hidden_layers')}"
        )
    if int(text.get("hidden_size", -1)) != 2560:
        raise ModelContractError(f"Expected hidden_size=2560, found {text.get('hidden_size')}")
    return config


def load_qwen_backbone(
    model_path: str | Path,
    model_config: dict[str, Any],
) -> tuple[nn.Module, Any]:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    inspect_qwen_config(model_path)
    attention_implementation = select_attention_implementation(
        model_config["attn_implementation"]
    )
    backbone = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
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
        target_modules=list(lora["target_modules"]),
    )
    backbone = get_peft_model(backbone, peft_config)
    assert_full_lora_coverage(
        backbone,
        expected_layers=int(model_config["text_layers"]),
        targets=tuple(lora["target_modules"]),
    )
    assert_qwen_freeze_contract(backbone)
    return backbone, processor


def lora_coverage(model: nn.Module) -> dict[int, set[str]]:
    coverage: dict[int, set[str]] = {}
    pattern = re.compile(r"(?:language_model\.)?layers\.(\d+)\..*?\.(q_proj|k_proj|v_proj|o_proj)$")
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        match = pattern.search(name)
        if match:
            coverage.setdefault(int(match.group(1)), set()).add(match.group(2))
    return coverage


def assert_full_lora_coverage(
    model: nn.Module,
    *,
    expected_layers: int = 36,
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> None:
    coverage = lora_coverage(model)
    missing: list[str] = []
    expected_targets = set(targets)
    for layer in range(expected_layers):
        absent = expected_targets.difference(coverage.get(layer, set()))
        if absent:
            missing.append(f"layer {layer}: {sorted(absent)}")
    unexpected = sorted(layer for layer in coverage if layer >= expected_layers)
    if missing or unexpected:
        detail = "; ".join(missing[:8])
        if len(missing) > 8:
            detail += f"; ... ({len(missing)} layers incomplete)"
        raise ModelContractError(
            "LoRA was not installed on every attention projection of all 36 text layers. "
            f"{detail}; unexpected layers={unexpected}"
        )


def assert_qwen_freeze_contract(model: nn.Module) -> None:
    violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if violations:
        raise ModelContractError(
            "Original Qwen parameters must remain frozen; trainable non-LoRA parameters: "
            + ", ".join(violations[:10])
        )


def compile_policy_modules(policy: Any, model_config: dict[str, Any]) -> None:
    """Compile the Qwen/PEFT backbone and DiT action head in place for training."""
    compile_config = model_config.get("torch_compile")
    if not compile_config or not bool(compile_config["enabled"]):
        return
    compile_kwargs = {
        "backend": str(compile_config["backend"]),
        "mode": str(compile_config["mode"]),
        "dynamic": bool(compile_config["dynamic"]),
        "fullgraph": bool(compile_config["fullgraph"]),
    }
    policy.backbone.compile(**compile_kwargs)
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
        self.context_forward = str(model.get("context_forward", "causal_lm"))

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
        return torch.where(constant, torch.zeros_like(normalized), normalized).clamp(-1.0, 1.0)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_q01.to(device=action.device, dtype=action.dtype)
        high = self.action_q99.to(device=action.device, dtype=action.dtype)
        normalized = 2.0 * (action - low) / self._safe_span(low, high, self.normalization_epsilon) - 1.0
        constant = (high - low).abs() < self.normalization_epsilon
        return torch.where(constant, torch.zeros_like(normalized), normalized).clamp(-1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_q01.to(device=action.device, dtype=action.dtype)
        high = self.action_q99.to(device=action.device, dtype=action.dtype)
        result = (action + 1.0) * 0.5 * self._safe_span(
            low, high, self.normalization_epsilon
        ) + low
        constant = (high - low).abs() < self.normalization_epsilon
        return torch.where(constant, low.expand_as(result), result)

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
                context.to(dtype=self.compute_dtype),
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
        normalized_state = self.normalize_state(states).to(dtype=self.compute_dtype)
        normalized_actions = euler_denoise(
            self.action_head,
            state=normalized_state,
            context=context.to(dtype=self.compute_dtype),
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
