# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Ported from starVLA/starVLA@3422b9f2387b6f682cf02802904a77b23ab13afd.

"""Qwen3-VL + GR00T inference model compatible with the released state dict."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .action_head import FlowMatchingActionHead
from .checkpoint import CheckpointLoadReport, load_strict_checkpoint
from .config import StarVLAConfigError, StarVLAModelSpec


def build_qwen_messages(
    image: Any,
    instruction: str,
    *,
    cot_prompt: str,
) -> list[list[dict[str, Any]]]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise StarVLAConfigError(
            f"StarVLA expects one uint8 HWC RGB image, found {array.shape} {array.dtype}"
        )
    prompt = str(cot_prompt).replace("{instruction}", str(instruction))
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    ]


def _qwen_model_from_config(base_model: Any) -> Any:
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(base_model, local_files_only=True)
    config._attn_implementation = "sdpa"
    model = Qwen3VLForConditionalGeneration(config)
    model.config.hidden_size = model.config.text_config.hidden_size
    return model


def _action_config(spec: StarVLAModelSpec, hidden_size: int) -> dict[str, Any]:
    config = dict(spec.config["framework"]["action_model"])
    diffusion = dict(config["diffusion_model_cfg"])
    diffusion["cross_attention_dim"] = int(hidden_size)
    config["diffusion_model_cfg"] = diffusion
    return config


def build_meta_policy(spec: StarVLAModelSpec) -> "StarVLAPolicy":
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        qwen_model = _qwen_model_from_config(spec.base_model)
        policy = StarVLAPolicy(
            qwen_model=qwen_model,
            action_model=FlowMatchingActionHead(
                _action_config(spec, qwen_model.config.text_config.hidden_size)
            ),
        )
    return policy


class QwenVLInterface(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model


class StarVLAPolicy(torch.nn.Module):
    def __init__(self, *, qwen_model: Any, action_model: FlowMatchingActionHead):
        super().__init__()
        self.qwen_vl_interface = QwenVLInterface(qwen_model)
        self.action_model = action_model

    @property
    def device(self) -> Any:
        return next(self.parameters()).device

    def predict_normalized_actions(
        self,
        image: np.ndarray,
        instruction: str,
        *,
        processor: Any,
        cot_prompt: str,
    ) -> np.ndarray:
        messages = build_qwen_messages(image, instruction, cot_prompt=cot_prompt)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface.model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            actions = self.action_model.predict_action(
                hidden,
                None,
                encoder_attention_mask=attention_mask,
            )
        return actions.detach().float().cpu().numpy()


@dataclass
class LoadedStarVLAPolicy:
    spec: StarVLAModelSpec
    policy: StarVLAPolicy
    processor: Any
    checkpoint_report: CheckpointLoadReport

    def predict_actions(self, image: np.ndarray, instruction: str) -> np.ndarray:
        normalized = self.policy.predict_normalized_actions(
            image,
            instruction,
            processor=self.processor,
            cot_prompt=self.spec.cot_prompt,
        )
        return self.spec.action_statistics.denormalize(normalized)


def load_starvla_policy(
    spec: StarVLAModelSpec,
    *,
    device: str = "cuda:0",
) -> LoadedStarVLAPolicy:
    from transformers import AutoProcessor

    target = torch.device(device)
    if target.type != "cuda":
        raise StarVLAConfigError("Released StarVLA checkpoint requires a CUDA device")
    if not torch.cuda.is_available():
        raise StarVLAConfigError(f"CUDA device is unavailable: {device}")
    if not torch.cuda.is_bf16_supported():
        raise StarVLAConfigError(f"CUDA device does not support BF16: {device}")
    policy = build_meta_policy(spec)
    report = load_strict_checkpoint(
        policy,
        spec.checkpoint_path,
        expected_tensor_count=962,
        expected_dtype=torch.bfloat16,
        torch_module=torch,
    )
    gc.collect()
    policy.to(target).eval()
    processor = AutoProcessor.from_pretrained(spec.base_model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    return LoadedStarVLAPolicy(
        spec=spec,
        policy=policy,
        processor=processor,
        checkpoint_report=report,
    )
