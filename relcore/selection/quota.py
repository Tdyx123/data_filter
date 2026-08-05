"""Capacity-aware Hamilton allocation for hard task quotas."""

from __future__ import annotations

from collections import Counter

import numpy as np


def allocate_task_quotas(
    task_indices: np.ndarray,
    *,
    budget: int,
    minimum_per_task: int = 1,
) -> dict[int, int]:
    values = np.asarray(task_indices, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("task_indices must be a non-empty vector")
    capacities = {int(task): int(count) for task, count in Counter(values.tolist()).items()}
    tasks = sorted(capacities)
    if budget <= 0 or budget > len(values):
        raise ValueError("selection budget must be within candidate count")
    if minimum_per_task < 0:
        raise ValueError("minimum_per_task cannot be negative")
    if any(capacity < minimum_per_task for capacity in capacities.values()):
        raise ValueError("a task has fewer candidates than minimum_per_task")
    if budget < minimum_per_task * len(tasks):
        raise ValueError("selection budget cannot satisfy minimum task quotas")
    quotas = {task: minimum_per_task for task in tasks}
    remaining = budget - sum(quotas.values())
    while remaining:
        active = [task for task in tasks if quotas[task] < capacities[task]]
        if not active:
            raise ValueError("task capacities cannot satisfy selection budget")
        weight_sum = sum(capacities[task] for task in active)
        ideals = {task: remaining * capacities[task] / weight_sum for task in active}
        allocated = 0
        for task in active:
            amount = min(
                capacities[task] - quotas[task],
                int(np.floor(ideals[task])),
            )
            quotas[task] += amount
            allocated += amount
        remaining -= allocated
        if remaining == 0:
            break
        ranked = sorted(
            (task for task in active if quotas[task] < capacities[task]),
            key=lambda task: (-(ideals[task] - np.floor(ideals[task])), task),
        )
        if not ranked:
            continue
        for task in ranked:
            if remaining == 0:
                break
            quotas[task] += 1
            remaining -= 1
    if sum(quotas.values()) != budget:
        raise RuntimeError("task quota allocation does not sum to budget")
    return quotas
