from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OctoOfficialConfig:
    hidden_size: int = 384
    transformer_layers: int = 12
    attention_heads: int = 6
    mlp_size: int = 1536
    transformer_dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1.0e-6
    max_horizon: int = 10
    history_horizon: int = 2
    language_tokens: int = 16
    language_features: int = 768
    vision_features: int = 512
    stem_features: tuple[int, int, int, int] = (32, 96, 192, 384)
    primary_tokens: int = 256
    wrist_tokens: int = 64
    use_proprio: bool = False
    action_dim: int = 7
    action_horizon: int = 4
    diffusion_steps: int = 20
    diffusion_time_dim: int = 32
    diffusion_blocks: int = 3
    diffusion_hidden_size: int = 256
    diffusion_dropout: float = 0.1
    max_action: float = 5.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OctoOfficialConfig":
        data = dict(value)
        if "stem_features" in data:
            data["stem_features"] = tuple(int(item) for item in data["stem_features"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dimensions = (1, 2, 3)
        weight = self.weight - self.weight.mean(dim=dimensions, keepdim=True)
        weight = weight / (weight.std(dim=dimensions, keepdim=True, unbiased=False) + 1.0e-5)
        return F.conv2d(
            inputs,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SmallStem16(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 6,
        features: tuple[int, int, int, int] = (32, 96, 192, 384),
        output_features: int = 512,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in features:
            layers.extend(
                [
                    WeightStandardizedConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(math.gcd(32, out_channels), out_channels, eps=1.0e-6),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.stem = nn.Sequential(*layers)
        self.embedding = nn.Conv2d(in_channels, output_features, kernel_size=1, stride=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.stem(inputs)).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = float(dropout)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = values.shape
        return values.view(batch, length, self.num_heads, hidden // self.num_heads).transpose(1, 2)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.query(inputs))
        key = self._heads(self.key(inputs))
        value = self._heads(self.value(inputs))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~attention_mask[:, None], torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(inputs.shape)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: OctoOfficialConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MultiHeadSelfAttention(
            config.hidden_size,
            config.attention_heads,
            config.attention_dropout,
        )
        self.attention_dropout = nn.Dropout(config.transformer_dropout)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_in = nn.Linear(config.hidden_size, config.mlp_size)
        self.mlp_out = nn.Linear(config.mlp_size, config.hidden_size)
        self.mlp_dropout = nn.Dropout(config.transformer_dropout)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        values = inputs + self.attention_dropout(
            self.attention(self.attention_norm(inputs), attention_mask)
        )
        hidden = self.mlp_in(self.mlp_norm(values))
        hidden = F.gelu(hidden, approximate="tanh")
        hidden = self.mlp_dropout(hidden)
        hidden = self.mlp_out(hidden)
        hidden = self.mlp_dropout(hidden)
        return values + hidden


class BlockTransformer(nn.Module):
    def __init__(self, config: OctoOfficialConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.transformer_layers)
        )
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        values = inputs
        for block in self.blocks:
            values = block(values, attention_mask)
        return self.encoder_norm(values)


def build_block_causal_attention_mask(
    *,
    language_mask: torch.Tensor,
    timestep_pad_mask: torch.Tensor,
    observation_tokens_per_timestep: int,
    readout_tokens_per_timestep: int = 1,
) -> torch.Tensor:
    if language_mask.ndim != 2 or timestep_pad_mask.ndim != 2:
        raise ValueError("language_mask and timestep_pad_mask must be rank two")
    if language_mask.shape[0] != timestep_pad_mask.shape[0]:
        raise ValueError("language and timestep masks must have the same batch")
    if observation_tokens_per_timestep <= 0 or readout_tokens_per_timestep <= 0:
        raise ValueError("token counts must be positive")
    batch, language_tokens = language_mask.shape
    horizon = timestep_pad_mask.shape[1]
    tokens_per_timestep = observation_tokens_per_timestep + readout_tokens_per_timestep
    total = language_tokens + horizon * tokens_per_timestep
    allowed = torch.zeros(total, total, dtype=torch.bool, device=language_mask.device)
    allowed[:language_tokens, :language_tokens] = True
    for timestep in range(horizon):
        base = language_tokens + timestep * tokens_per_timestep
        observation_slice = slice(base, base + observation_tokens_per_timestep)
        readout_slice = slice(base + observation_tokens_per_timestep, base + tokens_per_timestep)
        allowed[observation_slice, :language_tokens] = True
        allowed[readout_slice, :language_tokens] = True
        for previous in range(timestep + 1):
            previous_base = language_tokens + previous * tokens_per_timestep
            previous_observation = slice(
                previous_base,
                previous_base + observation_tokens_per_timestep,
            )
            previous_readout = slice(
                previous_base + observation_tokens_per_timestep,
                previous_base + tokens_per_timestep,
            )
            allowed[observation_slice, previous_observation] = True
            allowed[readout_slice, previous_observation] = True
            allowed[readout_slice, previous_readout] = True
    keys: list[torch.Tensor] = [language_mask.bool()]
    for timestep in range(horizon):
        valid = timestep_pad_mask[:, timestep : timestep + 1].bool()
        keys.append(valid.expand(batch, observation_tokens_per_timestep))
        keys.append(
            torch.ones(
                batch,
                readout_tokens_per_timestep,
                dtype=torch.bool,
                device=language_mask.device,
            )
        )
    key_mask = torch.cat(keys, dim=1)
    return allowed[None] & key_mask[:, None, :]


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    time = torch.linspace(0, steps, steps + 1, dtype=torch.float32) / steps
    alpha_hat = torch.cos((time + offset) / (1 + offset) * torch.pi * 0.5).square()
    alpha_hat = alpha_hat / alpha_hat[0]
    return (1 - alpha_hat[1:] / alpha_hat[:-1]).clamp(0, 0.999)


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = 2 * torch.pi * torch.matmul(inputs, self.weight.transpose(0, 1))
        return torch.cat([torch.cos(values), torch.sin(values)], dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.dropout(inputs)
        values = self.norm(values)
        values = self.linear2(F.silu(self.linear1(values)))
        return inputs + values


class DiffusionActionHead(nn.Module):
    def __init__(self, config: OctoOfficialConfig) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.output_size = self.action_dim * self.action_horizon
        self.diffusion_steps = config.diffusion_steps
        self.max_action = config.max_action
        self.time_features = FourierFeatures(config.diffusion_time_dim)
        self.time_linear1 = nn.Linear(config.diffusion_time_dim, config.diffusion_time_dim * 2)
        self.time_linear2 = nn.Linear(config.diffusion_time_dim * 2, config.diffusion_time_dim)
        self.reverse_input = nn.Linear(
            config.diffusion_time_dim + config.hidden_size + self.output_size,
            config.diffusion_hidden_size,
        )
        self.reverse_blocks = nn.ModuleList(
            ResidualMLPBlock(
                config.diffusion_hidden_size,
                config.diffusion_dropout,
                config.layer_norm_eps,
            )
            for _ in range(config.diffusion_blocks)
        )
        self.reverse_output = nn.Linear(config.diffusion_hidden_size, self.output_size)
        betas = cosine_beta_schedule(config.diffusion_steps)
        alphas = 1 - betas
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_hats", torch.cumprod(alphas, dim=0), persistent=False)

    def score(
        self,
        observation_embedding: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_features(time)
        time_embedding = self.time_linear2(F.silu(self.time_linear1(time_embedding)))
        values = torch.cat([time_embedding, observation_embedding, noisy_actions], dim=-1)
        values = self.reverse_input(values)
        for block in self.reverse_blocks:
            values = block(values)
        return self.reverse_output(F.silu(values))

    def loss(
        self,
        observation_embeddings: torch.Tensor,
        actions: torch.Tensor,
        timestep_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if observation_embeddings.ndim != 3:
            raise ValueError("observation_embeddings must have shape [B,T,H]")
        batch_size, horizon, _ = observation_embeddings.shape
        expected_action_shape = (
            batch_size,
            horizon,
            self.action_horizon,
            self.action_dim,
        )
        if tuple(actions.shape) != expected_action_shape:
            raise ValueError(
                f"action must have shape {expected_action_shape}, found {tuple(actions.shape)}"
            )
        if tuple(timestep_pad_mask.shape) != (batch_size, horizon):
            raise ValueError("timestep_pad_mask must match diffusion readouts")
        valid = timestep_pad_mask.bool()
        if not torch.any(valid):
            raise ValueError("at least one diffusion readout must be valid")

        embeddings = observation_embeddings.reshape(batch_size * horizon, -1)
        flattened_actions = actions.reshape(batch_size * horizon, self.output_size)
        diffusion_index = torch.randint(
            0,
            self.diffusion_steps,
            (batch_size * horizon,),
            device=actions.device,
        )
        noise = torch.randn_like(flattened_actions)
        alpha_hat = self.alpha_hats[diffusion_index].to(dtype=actions.dtype).unsqueeze(-1)
        noisy_actions = (
            torch.sqrt(alpha_hat) * flattened_actions
            + torch.sqrt(1 - alpha_hat) * noise
        )
        predicted_noise = self.score(
            embeddings,
            noisy_actions,
            diffusion_index.to(dtype=actions.dtype).unsqueeze(-1),
        )
        per_readout_mse = (predicted_noise - noise).square().mean(dim=-1)
        per_readout_mse = per_readout_mse.view(batch_size, horizon)
        valid_float = valid.to(dtype=per_readout_mse.dtype)
        mse = (per_readout_mse * valid_float).sum() / valid_float.sum()
        return {"loss": mse, "mse": mse.detach()}

    @torch.no_grad()
    def sample(
        self,
        observation_embedding: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        current = torch.randn(
            observation_embedding.shape[0],
            self.output_size,
            device=observation_embedding.device,
            dtype=observation_embedding.dtype,
            generator=generator,
        )
        for index in range(self.diffusion_steps - 1, -1, -1):
            time = torch.full(
                (current.shape[0], 1),
                index,
                device=current.device,
                dtype=current.dtype,
            )
            predicted_noise = self.score(observation_embedding, current, time)
            current = (
                current
                - (1 - self.alphas[index])
                / torch.sqrt(1 - self.alpha_hats[index])
                * predicted_noise
            ) / torch.sqrt(self.alphas[index])
            if index > 0:
                current = current + torch.sqrt(self.betas[index]) * torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
            current = current.clamp(-self.max_action, self.max_action)
        return current.view(-1, self.action_horizon, self.action_dim)


class OctoSmallOfficialPolicy(nn.Module):
    def __init__(self, text_encoder: Any, config: OctoOfficialConfig | None = None) -> None:
        super().__init__()
        self.config = config or OctoOfficialConfig()
        if self.config.use_proprio:
            raise ValueError("Official Octo-small policy must not use proprio")
        self.text_encoder = text_encoder
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(False)
        self.text_encoder.eval()
        self.primary_encoder = SmallStem16(
            features=self.config.stem_features,
            output_features=self.config.vision_features,
        )
        self.wrist_encoder = SmallStem16(
            features=self.config.stem_features,
            output_features=self.config.vision_features,
        )
        self.language_projection = nn.Linear(self.config.language_features, self.config.hidden_size)
        self.primary_projection = nn.Linear(self.config.vision_features, self.config.hidden_size)
        self.wrist_projection = nn.Linear(self.config.vision_features, self.config.hidden_size)
        self.language_pos_embedding = nn.Parameter(
            torch.randn(1, self.config.language_tokens, self.config.hidden_size) * 0.02
        )
        self.primary_pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.config.max_horizon,
                self.config.primary_tokens,
                self.config.hidden_size,
            )
            * 0.02
        )
        self.wrist_pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.config.max_horizon,
                self.config.wrist_tokens,
                self.config.hidden_size,
            )
            * 0.02
        )
        self.readout_pos_embedding = nn.Parameter(
            torch.randn(1, self.config.max_horizon, 1, self.config.hidden_size) * 0.02
        )
        self.transformer = BlockTransformer(self.config)
        self.action_head = DiffusionActionHead(self.config)
        self.observation_tokenizers = ("primary",)
        for parameter in self.wrist_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.wrist_projection.parameters():
            parameter.requires_grad_(False)
        self.wrist_pos_embedding.requires_grad_(False)

    def train(self, mode: bool = True) -> "OctoSmallOfficialPolicy":
        super().train(mode)
        self.text_encoder.eval()
        self.wrist_encoder.eval()
        self.wrist_projection.eval()
        return self

    @staticmethod
    def with_zero_goal(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected image tensor [B,T,3,H,W], found {tuple(images.shape)}")
        batch, horizon, _, height, width = images.shape
        current = images.reshape(batch * horizon, 3, height, width)
        return torch.cat([current, -torch.ones_like(current)], dim=1)

    def encode_readouts(self, batch: dict[str, Any]) -> torch.Tensor:
        if "proprio" in batch:
            raise ValueError("Official Octo-small does not accept proprio")
        images = batch["image_primary"]
        if images.ndim != 5:
            raise ValueError("image_primary must have shape [B,T,3,H,W]")
        batch_size, horizon = images.shape[:2]
        if not 1 <= horizon <= self.config.history_horizon:
            raise ValueError(f"history horizon must be in [1, {self.config.history_horizon}]")
        timestep_pad_mask = batch["timestep_pad_mask"]
        if tuple(timestep_pad_mask.shape) != (batch_size, horizon):
            raise ValueError("timestep_pad_mask must match image history")
        with torch.no_grad():
            language = self.text_encoder(
                input_ids=batch["language_input_ids"],
                attention_mask=batch["language_attention_mask"],
            ).last_hidden_state
        language = self.language_projection(language)
        language = language + self.language_pos_embedding[:, : language.shape[1]]

        primary = self.primary_encoder(self.with_zero_goal(images))
        if primary.shape[1] != self.config.primary_tokens:
            raise ValueError(
                f"Primary encoder produced {primary.shape[1]} tokens; "
                f"expected {self.config.primary_tokens}"
            )
        primary = primary.view(batch_size, horizon, self.config.primary_tokens, -1)
        primary = self.primary_projection(primary)
        primary = primary + self.primary_pos_embedding[:, :horizon]
        observation_groups = [primary]

        if "wrist" in self.observation_tokenizers:
            wrist_images = batch.get("image_wrist")
            if wrist_images is None:
                raise ValueError("image_wrist is required when wrist tokenizer is enabled")
            wrist = self.wrist_encoder(self.with_zero_goal(wrist_images))
            if wrist.shape[1] != self.config.wrist_tokens:
                raise ValueError(
                    f"Wrist encoder produced {wrist.shape[1]} tokens; "
                    f"expected {self.config.wrist_tokens}"
                )
            wrist = wrist.view(batch_size, horizon, self.config.wrist_tokens, -1)
            wrist = self.wrist_projection(wrist)
            wrist = wrist + self.wrist_pos_embedding[:, :horizon]
            observation_groups.append(wrist)

        observations = torch.cat(observation_groups, dim=2)
        readout = self.readout_pos_embedding[:, :horizon].expand(batch_size, -1, -1, -1)
        timestep = torch.cat([observations, readout], dim=2).flatten(1, 2)
        sequence = torch.cat([language, timestep], dim=1)
        attention_mask = build_block_causal_attention_mask(
            # The T5 attention mask applies inside T5 only. Official Octo marks
            # the whole language TokenGroup present and does not propagate
            # tokenizer padding into the block-transformer pad mask.
            language_mask=torch.ones_like(batch["language_attention_mask"], dtype=torch.bool),
            timestep_pad_mask=timestep_pad_mask,
            observation_tokens_per_timestep=observations.shape[2],
            readout_tokens_per_timestep=readout.shape[2],
        )
        encoded = self.transformer(sequence, attention_mask)
        encoded_timestep = encoded[:, language.shape[1] :]
        tokens_per_timestep = observations.shape[2] + readout.shape[2]
        encoded_timestep = encoded_timestep.view(
            batch_size,
            horizon,
            tokens_per_timestep,
            self.config.hidden_size,
        )
        return encoded_timestep[:, :, -1]

    def encode_observation(self, batch: dict[str, Any]) -> torch.Tensor:
        """Encode the last readout for official inference compatibility."""

        return self.encode_readouts(batch)[:, -1]

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "action" not in batch:
            raise ValueError("Training batch must contain action")
        readouts = self.encode_readouts(batch)
        return self.action_head.loss(
            readouts,
            batch["action"],
            batch["timestep_pad_mask"],
        )

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.action_head.sample(self.encode_observation(batch), generator=generator)
