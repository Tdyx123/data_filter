"""Deterministic SQCN candidate and coverage-reference windows."""

from __future__ import annotations


FRAGMENT_LENGTH = 15
CANDIDATE_STRIDE = 15


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("half-up rounding requires a non-negative ratio")
    return (2 * numerator + denominator) // (2 * denominator)


def reference_sample_count(length: int) -> int:
    """Return the piecewise number of reference fragments for one episode."""

    if length < FRAGMENT_LENGTH:
        return 0
    if length < 30:
        return 1
    if length <= 90:
        return _round_half_up_ratio(length, 15)
    if length <= 180:
        return _round_half_up_ratio(length, 30) + 3
    return _round_half_up_ratio(length, 60) + 6


def reference_windows(length: int) -> list[tuple[int, int]]:
    """Place complete reference windows at both endpoints and evenly between."""

    count = reference_sample_count(length)
    if count == 0:
        return []
    if count == 1:
        return [(0, FRAGMENT_LENGTH - 1)]
    final_start = length - FRAGMENT_LENGTH
    starts = [
        _round_half_up_ratio(index * final_start, count - 1)
        for index in range(count)
    ]
    if len(starts) != len(set(starts)):
        raise ValueError(f"reference sampling produced duplicate starts for length={length}")
    return [(start, start + FRAGMENT_LENGTH - 1) for start in starts]


def candidate_windows(length: int) -> list[tuple[int, int]]:
    """Return stride-15 complete windows plus one tail-aligned window."""

    if length < FRAGMENT_LENGTH:
        return []
    final_start = length - FRAGMENT_LENGTH
    starts = list(range(0, final_start + 1, CANDIDATE_STRIDE))
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + FRAGMENT_LENGTH - 1) for start in starts]
