from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OctoSmallConfig:
    hidden_size: int = 384
    transformer_layers: int = 12
    attention_heads: int = 6
    mlp_size: int = 1536
    transformer_dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1.0e-6
    max_horizon: int = 10
    language_tokens: int = 16
    language_features: int = 768
    vision_features: int = 512
    stem_features: tuple[int, int, int, int] = (32, 96, 192, 384)
    primary_tokens: int = 256
    wrist_tokens: int = 64
    proprio_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    diffusion_steps: int = 20
    diffusion_time_dim: int = 32
    diffusion_blocks: int = 3
    diffusion_hidden_size: int = 256
    diffusion_dropout: float = 0.1
    max_action: float = 5.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OctoSmallConfig":
        data = dict(value)
        if "stem_features" in data:
            data["stem_features"] = tuple(int(item) for item in data["stem_features"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        dimensions = (1, 2, 3)
        weight = weight - weight.mean(dim=dimensions, keepdim=True)
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
    """Octo SmallStem16 encoder operating on normalized NCHW image tensors."""

    def __init__(
        self,
        *,
        input_channels: int = 6,
        features: tuple[int, int, int, int] = (32, 96, 192, 384),
        output_features: int = 512,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in features:
            groups = math.gcd(32, out_channels)
            layers.extend(
                [
                    WeightStandardizedConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(groups, out_channels, eps=1.0e-6),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.stem = nn.Sequential(*layers)
        self.embedding = nn.Conv2d(in_channels, output_features, kernel_size=1, stride=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.embedding(self.stem(inputs))
        return values.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = float(dropout)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, self.num_heads, self.head_size).transpose(1, 2)

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.query(inputs))
        key = self._heads(self.key(inputs))
        value = self._heads(self.value(inputs))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        mask = attention_mask[:, None]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(inputs.shape)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: OctoSmallConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
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
    def __init__(self, config: OctoSmallConfig):
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


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    time = torch.linspace(0, steps, steps + 1, dtype=torch.float32) / steps
    alpha_hat = torch.cos((time + offset) / (1 + offset) * torch.pi * 0.5).square()
    alpha_hat = alpha_hat / alpha_hat[0]
    return (1 - alpha_hat[1:] / alpha_hat[:-1]).clamp(0, 0.999)


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = 2 * torch.pi * torch.matmul(inputs, self.weight.transpose(0, 1))
        return torch.cat([torch.cos(values), torch.sin(values)], dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, eps: float):
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
    def __init__(self, config: OctoSmallConfig):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.output_size = self.action_dim * self.action_horizon
        self.diffusion_steps = config.diffusion_steps
        self.max_action = config.max_action
        self.time_features = FourierFeatures(config.diffusion_time_dim)
        self.time_linear1 = nn.Linear(
            config.diffusion_time_dim, config.diffusion_time_dim * 2
        )
        self.time_linear2 = nn.Linear(
            config.diffusion_time_dim * 2, config.diffusion_time_dim
        )
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

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = torch.broadcast_to(mask, values.shape).to(dtype=values.dtype)
        return (values * expanded).mean() / expanded.mean().clamp_min(1.0e-5)

    def loss(
        self,
        observation_embedding: torch.Tensor,
        actions: torch.Tensor,
        action_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = actions.shape[0]
        flat_actions = actions.reshape(batch, -1).clamp(-self.max_action, self.max_action)
        flat_mask = action_pad_mask.reshape(batch, -1)
        time_index = torch.randint(
            0, self.diffusion_steps, (batch, 1), device=actions.device
        )
        noise = torch.randn_like(flat_actions)
        alpha_hat = self.alpha_hats[time_index]
        noisy_actions = alpha_hat.sqrt() * flat_actions + (1 - alpha_hat).sqrt() * noise
        predicted_noise = self.score(
            observation_embedding,
            noisy_actions,
            time_index.to(dtype=observation_embedding.dtype),
        )
        mse = self._masked_mean((predicted_noise - noise).square(), flat_mask)
        loss = mse * self.action_dim
        return loss, {"loss": loss.detach(), "mse": (mse * self.action_dim).detach()}

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
                noise = torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
                current = current + torch.sqrt(self.betas[index]) * noise
            current = current.clamp(-self.max_action, self.max_action)
        return current.view(-1, self.action_horizon, self.action_dim)


class OctoSmallPolicy(nn.Module):
    """Language-conditioned, two-camera Octo-small policy implemented in PyTorch."""

    def __init__(self, text_encoder: nn.Module, config: OctoSmallConfig | None = None):
        super().__init__()
        self.config = config or OctoSmallConfig()
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
        self.language_projection = nn.Linear(
            self.config.language_features, self.config.hidden_size
        )
        self.primary_projection = nn.Linear(
            self.config.vision_features, self.config.hidden_size
        )
        self.wrist_projection = nn.Linear(
            self.config.vision_features, self.config.hidden_size
        )
        self.proprio_projection = nn.Linear(1, self.config.hidden_size)

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
        self.proprio_pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.config.max_horizon,
                self.config.proprio_dim,
                self.config.hidden_size,
            )
            * 0.02
        )
        self.readout_pos_embedding = nn.Parameter(
            torch.randn(1, self.config.max_horizon, 1, self.config.hidden_size) * 0.02
        )
        self.transformer = BlockTransformer(self.config)
        self.action_head = DiffusionActionHead(self.config)

    def train(self, mode: bool = True) -> "OctoSmallPolicy":
        super().train(mode)
        self.text_encoder.eval()
        return self

    @staticmethod
    def _with_zero_goal(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != 1 or images.shape[2] != 3:
            raise ValueError(f"Expected image tensor [B,1,3,H,W], found {tuple(images.shape)}")
        current = images[:, 0]
        # The original tokenizer appends a zero-valued uint8 goal image and then
        # normalizes all six channels to [-1, 1], so the missing goal is -1 here.
        return torch.cat([current, -torch.ones_like(current)], dim=1)

    def _attention_mask(
        self,
        language_mask: torch.Tensor,
        *,
        observation_tokens: int,
    ) -> torch.Tensor:
        batch, language_tokens = language_mask.shape
        total = language_tokens + observation_tokens + 1
        allowed = torch.zeros(total, total, dtype=torch.bool, device=language_mask.device)
        allowed[:language_tokens, :language_tokens] = True
        observation_stop = language_tokens + observation_tokens
        allowed[language_tokens:observation_stop, :observation_stop] = True
        allowed[observation_stop:, :] = True
        key_mask = torch.cat(
            [
                language_mask.bool(),
                torch.ones(
                    batch,
                    observation_tokens + 1,
                    dtype=torch.bool,
                    device=language_mask.device,
                ),
            ],
            dim=1,
        )
        return allowed[None] & key_mask[:, None, :]

    def encode_observation(self, batch: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            language = self.text_encoder(
                input_ids=batch["language_input_ids"],
                attention_mask=batch["language_attention_mask"],
            ).last_hidden_state
        language = self.language_projection(language)
        language = language + self.language_pos_embedding[:, : language.shape[1]]

        primary = self.primary_encoder(self._with_zero_goal(batch["image_primary"]))
        wrist = self.wrist_encoder(self._with_zero_goal(batch["image_wrist"]))
        if primary.shape[1] != self.config.primary_tokens:
            raise ValueError(
                f"Primary encoder produced {primary.shape[1]} tokens; "
                f"expected {self.config.primary_tokens}"
            )
        if wrist.shape[1] != self.config.wrist_tokens:
            raise ValueError(
                f"Wrist encoder produced {wrist.shape[1]} tokens; "
                f"expected {self.config.wrist_tokens}"
            )
        primary = self.primary_projection(primary).unsqueeze(1)
        wrist = self.wrist_projection(wrist).unsqueeze(1)
        primary = primary + self.primary_pos_embedding[:, :1]
        wrist = wrist + self.wrist_pos_embedding[:, :1]

        proprio = batch["proprio"].unsqueeze(-1)
        proprio = self.proprio_projection(proprio)
        proprio = proprio + self.proprio_pos_embedding[:, :1]
        readout = self.readout_pos_embedding[:, :1].expand(
            primary.shape[0], -1, -1, -1
        )

        observation_groups = [primary, wrist, proprio]
        observation_tokens = sum(values.shape[2] for values in observation_groups)
        timestep = torch.cat([*observation_groups, readout], dim=2).flatten(1, 2)
        sequence = torch.cat([language, timestep], dim=1)
        attention_mask = self._attention_mask(
            batch["language_attention_mask"],
            observation_tokens=observation_tokens,
        )
        encoded = self.transformer(sequence, attention_mask)
        return encoded[:, -1]

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        embedding = self.encode_observation(batch)
        loss, metrics = self.action_head.loss(
            embedding,
            batch["action"],
            batch["action_pad_mask"],
        )
        return {"loss": loss, "mse": metrics["mse"]}

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.action_head.sample(
            self.encode_observation(batch),
            generator=generator,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple["OctoSmallPolicy", Any]:
        try:
            from safetensors.torch import load_model
            from transformers import AutoTokenizer, T5Config, T5EncoderModel
        except ImportError as error:
            raise RuntimeError(
                "transformers and safetensors are required to load PyTorch Octo-small"
            ) from error

        root = Path(model_path).expanduser().resolve()
        with (root / "model_config.json").open("r", encoding="utf-8") as handle:
            config = OctoSmallConfig.from_dict(json.load(handle))
        text_root = root / "text_encoder"
        text_config = T5Config.from_pretrained(text_root, local_files_only=True)
        text_encoder = T5EncoderModel(text_config)
        model = cls(text_encoder, config)
        load_model(
            model,
            str(root / "model.safetensors"),
            strict=True,
            device=str(device),
        )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(text_root, local_files_only=True)
        return model, tokenizer


def save_model_config(config: OctoSmallConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
