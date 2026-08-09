"""Selectable prototype-relation gains used by the set objective."""

from __future__ import annotations

from collections.abc import Sequence


PROTOTYPE_GAIN_METRICS = ("transition", "cooccurrence", "sequence")


def normalize_prototype_gain_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise ValueError("must be a sequence of metric names")
    values = tuple(metrics)
    if not values:
        raise ValueError("cannot be empty")
    if any(not isinstance(metric, str) for metric in values):
        raise ValueError("must contain only metric names")
    if len(set(values)) != len(values):
        raise ValueError("cannot contain duplicates")
    unknown = sorted(set(values) - set(PROTOTYPE_GAIN_METRICS))
    if unknown:
        raise ValueError(f"contains unknown metrics: {unknown}")
    selected = set(values)
    return tuple(metric for metric in PROTOTYPE_GAIN_METRICS if metric in selected)


def prototype_gain_metric_mask(metrics: Sequence[str]) -> int:
    """Encode enabled gains using canonical transition/cooccurrence/sequence bits."""
    enabled = set(normalize_prototype_gain_metrics(metrics))
    return sum(
        1 << (len(PROTOTYPE_GAIN_METRICS) - index - 1)
        for index, metric in enumerate(PROTOTYPE_GAIN_METRICS)
        if metric in enabled
    )
