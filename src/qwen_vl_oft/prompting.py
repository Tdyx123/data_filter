from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


ACTION_TOKEN = "🔍"


def discretize_state(state: torch.Tensor, *, num_bins: int = 256) -> torch.Tensor:
    """Map normalized state values in [-1, 1] to integer text bins."""
    if state.ndim != 2:
        raise ValueError(f"state must have shape [batch, feature], found {tuple(state.shape)}")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    scaled = torch.floor((state.float().clamp(-1.0, 1.0) + 1.0) * (num_bins / 2.0))
    return scaled.clamp(0, num_bins - 1).to(dtype=torch.int64)


def build_oft_instructions(
    instructions: Sequence[str],
    normalized_state: torch.Tensor,
    *,
    action_horizon: int,
    num_bins: int = 256,
    action_token: str = ACTION_TOKEN,
) -> list[str]:
    if len(instructions) != normalized_state.shape[0]:
        raise ValueError("instructions and state have different batch sizes")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    state_bins = discretize_state(normalized_state, num_bins=num_bins).cpu().tolist()
    queries = action_token * action_horizon
    suffix = (
        f"Please predict the next {action_horizon} robot actions: "
        f"<action>{queries}<action>."
    )
    return [
        f"{instruction} [STATE] {' '.join(str(value) for value in values)} "
        f"[ACTION] {suffix}"
        for instruction, values in zip(instructions, state_bins, strict=True)
    ]


def resolve_action_token_id(
    tokenizer: Callable[..., dict[str, Any]],
    action_token: str = ACTION_TOKEN,
) -> int:
    encoded = tokenizer(action_token, add_special_tokens=False)["input_ids"]
    if len(encoded) != 1:
        raise ValueError(
            f"OFT action query {action_token!r} must encode to exactly one token; "
            f"found {encoded}"
        )
    return int(encoded[0])


def gather_action_queries(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    action_token_id: int,
    horizon: int,
) -> torch.Tensor:
    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("hidden_states and input_ids must have shapes [B,L,H] and [B,L]")
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("hidden_states and input_ids sequence shapes differ")
    mask = input_ids.eq(action_token_id)
    counts = mask.sum(dim=1)
    if torch.any(counts < horizon):
        samples = torch.nonzero(counts < horizon, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"insufficient OFT action query tokens for samples {samples}; "
            f"expected {horizon}, counts={counts.tolist()}"
        )
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    positions = torch.where(mask, positions, torch.full_like(positions, -1))
    selected = positions.topk(horizon, dim=1).values.sort(dim=1).values
    index = selected.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    return hidden_states.gather(1, index)
