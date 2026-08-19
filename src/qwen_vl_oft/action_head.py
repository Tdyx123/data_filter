from __future__ import annotations

import torch
from torch import nn


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.ffn(value)


class MLPResNetActionHead(nn.Module):
    """StarVLA-style per-query continuous action regressor."""

    def __init__(self, *, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            MLPResNetBlock(hidden_dim),
            MLPResNetBlock(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, action_queries: torch.Tensor) -> torch.Tensor:
        if action_queries.ndim != 3 or action_queries.shape[-1] != self.input_dim:
            raise ValueError(
                f"action_queries must have shape [B,K,{self.input_dim}], "
                f"found {tuple(action_queries.shape)}"
            )
        batch, horizon, hidden = action_queries.shape
        actions = self.model(action_queries.reshape(batch * horizon, hidden))
        return actions.reshape(batch, horizon, self.action_dim)

    predict_action = forward


def masked_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if action_mask.shape != predictions.shape[:2]:
        raise ValueError("action_mask must have shape [batch, horizon]")
    mask = action_mask.to(device=predictions.device, dtype=predictions.dtype).unsqueeze(-1)
    denominator = mask.sum() * predictions.shape[-1]
    if denominator.item() <= 0:
        raise ValueError("action_mask must contain at least one valid action")
    return ((predictions - targets).abs() * mask).sum() / denominator
