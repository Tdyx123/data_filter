from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class FrameIndex:
    source: int
    frame: int


def normalized_sample_weights(weights: Sequence[float]) -> tuple[float, ...]:
    if not weights or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in weights
    ):
        raise ValueError("sample weights must contain positive finite numbers")
    total = sum(float(value) for value in weights)
    return tuple(float(value) / total for value in weights)


def sample_counts_per_batch(
    weights: Sequence[float],
    batch_size: int,
    *,
    scope: str = "batch",
) -> tuple[int, ...]:
    normalized = normalized_sample_weights(weights)
    raw_counts = tuple(int(batch_size) * value for value in normalized)
    counts = tuple(int(round(value)) for value in raw_counts)
    if int(batch_size) <= 0 or any(count <= 0 for count in counts) or any(
        not math.isclose(value, count, rel_tol=0.0, abs_tol=1.0e-9)
        for value, count in zip(raw_counts, counts, strict=True)
    ):
        raise ValueError(
            f"sample weights must produce positive whole-number counts for {scope}"
        )
    return counts


def _interleaved_pattern(counts: Sequence[int]) -> tuple[int, ...]:
    remaining = [int(count) for count in counts]
    pattern: list[int] = []
    while any(remaining):
        for source, count in enumerate(remaining):
            if count > 0:
                pattern.append(source)
                remaining[source] -= 1
    return tuple(pattern)


class GloballyBalancedDistributedBatchSampler:
    """Create exact source quotas across all ranks in each micro-batch."""

    def __init__(
        self,
        source_sizes: Sequence[int],
        *,
        local_batch_size: int,
        sample_weights: Sequence[float],
        rank: int,
        world_size: int,
        seed: int,
        num_batches: int,
    ) -> None:
        self.source_sizes = tuple(int(size) for size in source_sizes)
        if not self.source_sizes or any(size <= 0 for size in self.source_sizes):
            raise ValueError("source sizes must be positive")
        if len(sample_weights) != len(self.source_sizes):
            raise ValueError("sample_weights must contain one weight per source")
        self.local_batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.num_batches = int(num_batches)
        if self.local_batch_size <= 0 or self.world_size <= 0 or self.num_batches <= 0:
            raise ValueError("batch size, world size, and num_batches must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        global_batch_size = self.local_batch_size * self.world_size
        self.source_batch_sizes = sample_counts_per_batch(
            sample_weights,
            global_batch_size,
            scope="global micro-batch",
        )
        self._source_pattern = _interleaved_pattern(self.source_batch_sizes)
        self._generators = [
            np.random.default_rng(int(seed) + source * 1_000_003)
            for source in range(len(self.source_sizes))
        ]
        self._permutations = [
            generator.permutation(size).tolist()
            for generator, size in zip(
                self._generators, self.source_sizes, strict=True
            )
        ]
        self._positions = [0 for _ in self.source_sizes]

    def _take(self, source: int, count: int) -> list[int]:
        values: list[int] = []
        while len(values) < count:
            position = self._positions[source]
            permutation = self._permutations[source]
            available = min(count - len(values), len(permutation) - position)
            values.extend(permutation[position : position + available])
            position += available
            if position == len(permutation):
                permutation = self._generators[source].permutation(
                    self.source_sizes[source]
                ).tolist()
                position = 0
                self._permutations[source] = permutation
            self._positions[source] = position
        return values

    def __iter__(self) -> Iterator[list[FrameIndex]]:
        for step in range(self.num_batches):
            frames = [
                self._take(source, count)
                for source, count in enumerate(self.source_batch_sizes)
            ]
            positions = [0 for _ in self.source_sizes]
            offset = step % len(self._source_pattern)
            pattern = self._source_pattern[offset:] + self._source_pattern[:offset]
            global_batch: list[FrameIndex] = []
            for source in pattern:
                global_batch.append(FrameIndex(source, frames[source][positions[source]]))
                positions[source] += 1
            start = self.rank * self.local_batch_size
            yield global_batch[start : start + self.local_batch_size]

    def __len__(self) -> int:
        return self.num_batches
