from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkMetrics:
    step_seconds: float
    samples_per_second: float
    data_wait_fraction: float
    measured_steps: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_metrics(
    path: str | Path,
    *,
    warmup_steps: int,
    expected_steps: int,
    effective_batch_size: int,
) -> BenchmarkMetrics:
    if not 0 <= warmup_steps < expected_steps:
        raise ValueError("warmup_steps must be non-negative and below expected_steps")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")

    rows_by_step: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "performance/step_seconds" in row:
                rows_by_step[int(row["step"])] = row

    if expected_steps not in rows_by_step:
        raise ValueError(f"benchmark metrics do not contain completed step {expected_steps}")
    if warmup_steps and warmup_steps not in rows_by_step:
        raise ValueError(f"benchmark metrics do not contain warmup boundary {warmup_steps}")

    selected_steps = sorted(
        step for step in rows_by_step if warmup_steps < step <= expected_steps
    )
    elapsed_weighted = 0.0
    wait_weighted = 0.0
    measured_steps = 0
    previous_step = warmup_steps
    for step in selected_steps:
        span = step - previous_step
        row = rows_by_step[step]
        elapsed_weighted += float(row["performance/step_seconds"]) * span
        wait_weighted += float(row["performance/data_wait_fraction"]) * span
        measured_steps += span
        previous_step = step
    expected_measured_steps = expected_steps - warmup_steps
    if measured_steps != expected_measured_steps:
        raise ValueError(
            f"benchmark metrics cover {measured_steps} measured steps, "
            f"expected {expected_measured_steps}"
        )
    step_seconds = elapsed_weighted / measured_steps
    return BenchmarkMetrics(
        step_seconds=step_seconds,
        samples_per_second=effective_batch_size / step_seconds,
        data_wait_fraction=wait_weighted / measured_steps,
        measured_steps=measured_steps,
    )


def combine_phase_metrics(
    *,
    head_only_step_seconds: float,
    lora_active_step_seconds: float,
) -> float:
    return 0.9 * head_only_step_seconds + 0.1 * lora_active_step_seconds


def is_meaningfully_faster(
    *,
    reference_step_seconds: float,
    candidate_step_seconds: float,
    minimum_improvement: float = 0.05,
) -> bool:
    if reference_step_seconds <= 0 or candidate_step_seconds <= 0:
        raise ValueError("step durations must be positive")
    if not 0.0 < minimum_improvement < 1.0:
        raise ValueError("minimum_improvement must be in (0, 1)")
    return candidate_step_seconds <= reference_step_seconds * (1.0 - minimum_improvement)
