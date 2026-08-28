"""Deterministic maximum-coverage seeding for Cocore selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import CocoreObjectiveContext


@dataclass(frozen=True)
class CoverageSeed:
    selected_indices: tuple[int, ...]
    target_coverage: np.ndarray
    achieved_coverage: np.ndarray


def build_max_coverage_seed(
    context: CocoreObjectiveContext,
    *,
    budget: int,
) -> CoverageSeed:
    if budget <= 0 or budget > len(context.graph.sample_ids):
        raise ValueError("selection budget must be within candidate count")
    target = context.prototype_mass.max(axis=0)
    selected: list[int] = []
    selected_set: set[int] = set()
    for prototype, maximum in enumerate(target):
        if maximum <= 0.0:
            continue
        matches = np.flatnonzero(context.prototype_mass[:, prototype] == maximum)
        winner = min(matches.tolist(), key=lambda index: context.graph.sample_ids[index])
        if winner not in selected_set:
            selected.append(winner)
            selected_set.add(winner)
    if len(selected) > budget:
        raise ValueError(
            f"coverage seed exceeds selection budget; minimum required budget is {len(selected)}"
        )
    achieved = (
        context.prototype_mass[np.asarray(selected, dtype=np.int64)].max(axis=0)
        if selected
        else np.zeros_like(target)
    )
    return CoverageSeed(tuple(selected), target.copy(), achieved)
