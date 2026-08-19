from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from qwen_vl_common.backbone import load_qwen_backbone
from qwen_vl_common.normalization import QuantileStats

from .action_head import MLPResNetActionHead, masked_l1_loss
from .prompting import (
    ACTION_TOKEN,
    build_oft_instructions,
    gather_action_queries,
    resolve_action_token_id,
)


class QwenVLOFTPolicy(nn.Module):
    """Qwen3-VL policy implementing StarVLA's causal action-query OFT path."""

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
        self.state_dim = int(data["state_dim"])
        self.action_horizon = int(data["action_horizon"])
        self.action_dim = int(data["action_dim"])
        self.context_dim = int(model["context_dim"])
        self.state_bins = int(model.get("state_bins", 256))
        self.action_token = str(model.get("action_token", ACTION_TOKEN))
        self.processor.tokenizer.padding_side = "left"
        self.action_token_id = resolve_action_token_id(
            self.processor.tokenizer,
            self.action_token,
        )
        self.action_head = MLPResNetActionHead(
            input_dim=self.context_dim,
            hidden_dim=int(model.get("action_head_hidden_dim", self.context_dim * 2)),
            action_dim=self.action_dim,
        )
        self.register_buffer("state_q01", torch.as_tensor(stats.state_q01), persistent=True)
        self.register_buffer("state_q99", torch.as_tensor(stats.state_q99), persistent=True)
        self.register_buffer("action_q01", torch.as_tensor(stats.action_q01), persistent=True)
        self.register_buffer("action_q99", torch.as_tensor(stats.action_q99), persistent=True)
        self.normalization_epsilon = float(stats.epsilon)

    @classmethod
    def from_local_qwen(
        cls,
        *,
        model_path: str | Path,
        stats: QuantileStats,
        config: dict[str, Any],
    ) -> "QwenVLOFTPolicy":
        backbone, processor = load_qwen_backbone(model_path, config["model"])
        return cls(backbone=backbone, processor=processor, stats=stats, config=config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def compute_dtype(self) -> torch.dtype:
        parameter = next(self.action_head.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    @staticmethod
    def _safe_span(low: torch.Tensor, high: torch.Tensor, epsilon: float) -> torch.Tensor:
        span = high - low
        return torch.where(span.abs() < epsilon, torch.ones_like(span), span)

    def _normalize(
        self,
        value: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        low = low.to(device=value.device, dtype=value.dtype)
        high = high.to(device=value.device, dtype=value.dtype)
        result = 2.0 * (value - low) / self._safe_span(
            low, high, self.normalization_epsilon
        ) - 1.0
        constant = (high - low).abs() < self.normalization_epsilon
        return torch.where(constant, torch.zeros_like(result), result).clamp(-1.0, 1.0)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return self._normalize(state, self.state_q01, self.state_q99)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self._normalize(action, self.action_q01, self.action_q99)

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
            parameter.requires_grad_(enabled if "lora_" in name else False)

    def lora_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.backbone.named_parameters()
            if "lora_" in name
        ]

    def action_head_parameters(self) -> list[nn.Parameter]:
        return list(self.action_head.parameters())

    def _prepare_inputs(
        self,
        images: Sequence[Any],
        normalized_state: torch.Tensor,
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        prompts = build_oft_instructions(
            instructions,
            normalized_state,
            action_horizon=self.action_horizon,
            num_bins=self.state_bins,
            action_token=self.action_token,
        )
        rendered = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            rendered.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        inputs = self.processor(
            text=rendered,
            images=list(images),
            padding=True,
            return_tensors="pt",
        )
        return {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

    def _predict_normalized(
        self,
        images: Sequence[Any],
        state: torch.Tensor,
        instructions: Sequence[str],
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state feature dimension must be {self.state_dim}; "
                f"found shape {tuple(state.shape)}"
            )
        normalized_state = self.normalize_state(state.to(self.device, dtype=torch.float32))
        inputs = self._prepare_inputs(images, normalized_state, instructions)
        lora_requires_grad = self.training and any(
            parameter.requires_grad for parameter in self.lora_parameters()
        )
        gradient_context = nullcontext() if lora_requires_grad else torch.no_grad()
        with gradient_context:
            outputs = self.backbone(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        queries = gather_action_queries(
            outputs.hidden_states[-1],
            inputs["input_ids"],
            action_token_id=self.action_token_id,
            horizon=self.action_horizon,
        )
        return self.action_head(queries.to(dtype=self.compute_dtype))

    def forward(
        self,
        *,
        images: Sequence[Any],
        state: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        instructions: Sequence[str],
    ) -> torch.Tensor:
        predictions = self._predict_normalized(images, state, instructions)
        targets = self.normalize_action(
            actions.to(self.device, dtype=predictions.dtype)
        )
        return masked_l1_loss(predictions, targets, action_mask.to(self.device))

    @torch.inference_mode()
    def predict_actions(
        self,
        image: Any | Sequence[Any],
        state: np.ndarray | torch.Tensor | Sequence[float],
        instruction: str | Sequence[str],
    ) -> torch.Tensor:
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        instructions = (
            list(instruction) if isinstance(instruction, (list, tuple)) else [instruction]
        )
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if len(images) != states.shape[0] or len(instructions) != states.shape[0]:
            raise ValueError("image, state, and instruction batch sizes differ")
        was_training = self.training
        self.eval()
        normalized = self._predict_normalized(images, states, instructions)
        actions = self.denormalize_action(normalized).float()
        if was_training:
            self.train()
        return actions

    def compact_parameter_names(self) -> list[str]:
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith("action_head.") or "lora_" in name
        ]
