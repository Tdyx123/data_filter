from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def timestep_embedding(timestep: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal embedding for a scalar flow time in [0, 1]."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timestep.device)
        / max(half, 1)
    )
    angles = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class AdaLNCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        mlp_hidden = hidden_size * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(
        value: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        time_embedding: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        (
            self_shift,
            self_scale,
            self_gate,
            cross_shift,
            cross_scale,
            cross_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.modulation(time_embedding).chunk(9, dim=-1)

        query = self._modulate(self.self_norm(tokens), self_shift, self_scale)
        attended = self.self_attention(query, query, query, need_weights=False)[0]
        tokens = tokens + self_gate.unsqueeze(1) * attended

        query = self._modulate(self.cross_norm(tokens), cross_shift, cross_scale)
        attended = self.cross_attention(
            query,
            context,
            context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )[0]
        tokens = tokens + cross_gate.unsqueeze(1) * attended

        value = self._modulate(self.mlp_norm(tokens), mlp_shift, mlp_scale)
        return tokens + mlp_gate.unsqueeze(1) * self.mlp(value)


class FlowMatchingActionHead(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 8,
        action_dim: int = 7,
        horizon: int = 8,
        context_dim: int = 2560,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.2,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing

        self.state_projection = nn.Linear(state_dim, hidden_size)
        self.action_projection = nn.Linear(action_dim, hidden_size)
        self.context_projection = nn.Linear(context_dim, hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, horizon + 1, hidden_size))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNCrossAttentionBlock(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output_projection = nn.Linear(hidden_size, action_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected actions [B,{self.horizon},{self.action_dim}], "
                f"found {tuple(noisy_actions.shape)}"
            )
        time_features = timestep_embedding(timestep, self.hidden_size).to(
            dtype=self.time_mlp[0].weight.dtype
        )
        time_condition = self.time_mlp(time_features)
        state_token = self.state_projection(state).unsqueeze(1)
        action_tokens = self.action_projection(noisy_actions)
        tokens = torch.cat([state_token, action_tokens], dim=1) + self.position_embedding
        context = self.context_projection(context)
        padding_mask = None
        if context_attention_mask is not None:
            padding_mask = ~context_attention_mask.bool()

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(
                    block,
                    tokens,
                    context,
                    time_condition,
                    padding_mask,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, context, time_condition, padding_mask)

        action_tokens = self.final_norm(tokens[:, 1:])
        shift, scale = self.final_modulation(time_condition).chunk(2, dim=-1)
        action_tokens = action_tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.output_projection(action_tokens)


def sample_flow_batch(
    actions: torch.Tensor,
    *,
    beta_alpha: float = 1.5,
    beta_beta: float = 1.0,
    noise_s: float = 0.999,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(x_t, t, target_velocity)` for conditional flow matching."""
    if not 0.0 < noise_s <= 1.0:
        raise ValueError("noise_s must be in (0, 1]")
    distribution = torch.distributions.Beta(
        torch.tensor(beta_alpha, device=actions.device, dtype=torch.float32),
        torch.tensor(beta_beta, device=actions.device, dtype=torch.float32),
    )
    # torch.distributions does not expose a generator argument. Sampling on CPU with
    # a supplied generator is useful for deterministic validation.
    if generator is None:
        timestep = distribution.sample((actions.shape[0],))
        noise = torch.randn_like(actions)
    else:
        uniform = torch.rand(
            actions.shape[0], device=actions.device, generator=generator, dtype=torch.float32
        )
        # Exact inverse-CDF is unavailable for a general beta distribution. For the
        # fixed Beta(alpha, 1) recipe, x = u**(1/alpha).
        if beta_beta != 1.0:
            raise ValueError("A seeded generator currently requires beta_beta=1.0")
        timestep = uniform.pow(1.0 / beta_alpha)
        noise = torch.randn(
            actions.shape,
            device=actions.device,
            dtype=actions.dtype,
            generator=generator,
        )
    timestep = timestep.clamp(max=noise_s)
    timestep = ((noise_s - timestep) / noise_s).to(dtype=actions.dtype)
    interpolation = timestep[:, None, None]
    noisy_actions = (1.0 - interpolation) * noise + interpolation * actions
    target_velocity = actions - noise
    return noisy_actions, timestep, target_velocity


def masked_velocity_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    mask = action_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    squared_error = (prediction - target).square() * mask
    denominator = mask.sum() * prediction.shape[-1]
    return squared_error.sum() / denominator.clamp_min(1.0)


@torch.no_grad()
def euler_denoise(
    action_head: FlowMatchingActionHead,
    *,
    state: torch.Tensor,
    context: torch.Tensor,
    context_attention_mask: torch.Tensor | None,
    steps: int = 4,
    noise_s: float = 0.999,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < noise_s <= 1.0:
        raise ValueError("noise_s must be in (0, 1]")
    batch = state.shape[0]
    actions = (
        initial_noise
        if initial_noise is not None
        else torch.randn(
            batch,
            action_head.horizon,
            action_head.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
    )
    # GR00T trains with the reflected/clamped Beta schedule above but integrates
    # the learned vector field over the complete [0, 1] interval at inference.
    dt = 1.0 / steps
    for index in range(steps):
        timestep = torch.full(
            (batch,),
            index * dt,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = action_head(
            actions, state, timestep, context, context_attention_mask
        )
        actions = actions + dt * velocity
    return actions
