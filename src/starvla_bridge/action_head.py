# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by the StarVLA community in 2025 under the MIT License.
# Ported from starVLA/starVLA@3422b9f2387b6f682cf02802904a77b23ab13afd.

"""Inference-only StarVLA GR00T flow-matching action head."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


def _swish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=timesteps.device
        ) * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / half_dim)
        frequencies = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(value)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape (batch_size,)")
        expanded = timesteps.unsqueeze(1).expand(-1, horizon)
        action_embedding = self.layer1(actions)
        time_embedding = self.pos_encoding(expanded).to(dtype=action_embedding.dtype)
        hidden = torch.cat([action_embedding, time_embedding], dim=-1)
        return self.layer3(_swish(self.layer2(hidden)))


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, 1.0e-5, False)

    def forward(self, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(embedding)).chunk(2, dim=1)
        return self.norm(value) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dropout: float,
        cross_attention_dim: int | None,
        activation_fn: str,
        attention_bias: bool,
        upcast_attention: bool,
        norm_type: str,
        norm_elementwise_affine: bool,
        norm_eps: float,
        final_dropout: bool,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.norm1 = (
            AdaLayerNorm(dim)
            if norm_type == "ada_norm"
            else nn.LayerNorm(
                dim,
                elementwise_affine=norm_elementwise_affine,
                eps=norm_eps,
            )
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=True,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            bias=True,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (
            self.norm1(hidden_states, temb)
            if self.norm_type == "ada_norm"
            else self.norm1(hidden_states)
        )
        attention = self.attn1(
            normalized,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        if self.final_dropout is not None:
            attention = self.final_dropout(attention)
        hidden_states = attention + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1.0e-5,
        max_num_positional_embeddings: int = 512,
        final_dropout: bool = True,
        positional_embeddings: str | None = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        del max_num_positional_embeddings, positional_embeddings, kwargs
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        blocks = []
        for index in range(num_layers):
            use_self_attention = index % 2 == 1 and interleave_self_attention
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    final_dropout=final_dropout,
                    cross_attention_dim=None if use_self_attention else cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1.0e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        for index, block in enumerate(self.transformer_blocks):
            self_attention = index % 2 == 1 and self.config.interleave_self_attention
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None if self_attention else encoder_hidden_states,
                encoder_attention_mask=None if self_attention else encoder_attention_mask,
                temb=embedding,
            )
        shift, scale = self.proj_out_1(F.silu(embedding)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


class FlowMatchingActionHead(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        if str(config["action_model_type"]) != "DiT-B":
            raise ValueError("Released StarVLA checkpoint requires action_model_type=DiT-B")
        input_embedding_dim = 768
        diffusion_config = {
            "input_embedding_dim": input_embedding_dim,
            "attention_head_dim": 64,
            "num_attention_heads": 12,
            **dict(config["diffusion_model_cfg"]),
        }
        self.model = DiT(**diffusion_config)
        self.action_horizon = int(config["action_horizon"])
        self.action_dim = int(config["action_dim"])
        self.num_inference_timesteps = int(config["num_inference_timesteps"])
        self.num_timestep_buckets = int(config["num_timestep_buckets"])
        hidden_size = int(config["hidden_size"])
        state_dim = int(config["state_dim"])
        self.state_encoder = (
            MLP(state_dim, hidden_size, input_embedding_dim) if state_dim else None
        )
        self.action_encoder = ActionEncoder(self.action_dim, input_embedding_dim)
        self.action_decoder = MLP(
            int(diffusion_config["output_dim"]),
            hidden_size,
            self.action_dim,
        )
        self.future_tokens = nn.Embedding(
            int(config["num_target_vision_tokens"]),
            input_embedding_dim,
        )
        if bool(config["add_pos_embed"]):
            self.position_embedding = nn.Embedding(
                int(config["max_seq_len"]),
                input_embedding_dim,
            )
        else:
            self.position_embedding = None

    @torch.no_grad()
    def predict_action(
        self,
        vision_language_embeddings: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = vision_language_embeddings.shape[0]
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=vision_language_embeddings.dtype,
            device=vision_language_embeddings.device,
        )
        state_features = self.state_encoder(state) if state is not None else None
        step_size = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            discrete_time = int(
                step / float(self.num_inference_timesteps) * self.num_timestep_buckets
            )
            timesteps = torch.full(
                (batch_size,),
                discrete_time,
                device=actions.device,
            )
            action_features = self.action_encoder(actions, timesteps)
            if self.position_embedding is not None:
                position_ids = torch.arange(
                    action_features.shape[1],
                    dtype=torch.long,
                    device=actions.device,
                )
                action_features = action_features + self.position_embedding(
                    position_ids
                ).unsqueeze(0)
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            state_action = (
                torch.cat([state_features, future_tokens, action_features], dim=1)
                if state_features is not None
                else torch.cat([future_tokens, action_features], dim=1)
            )
            predicted = self.action_decoder(
                self.model(
                    state_action,
                    vision_language_embeddings,
                    timestep=timesteps,
                    encoder_attention_mask=encoder_attention_mask,
                )
            )
            actions = actions + step_size * predicted[:, -self.action_horizon :]
        return actions
