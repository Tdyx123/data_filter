"""Local endpoint displacement / traveled distance in original Cartesian units."""

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

PATH_FIELDS = ("local_path_efficiency",)
PATH_CACHE_FIELDS = ("path_position_sequences", *PATH_FIELDS)


def resolve_path_config(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"delta_path"}:
        raise ValueError("local_path_efficiency requires only an explicit delta_path")
    threshold = value["delta_path"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("local_path_efficiency delta_path must be finite and positive")
    return {"delta_path": float(threshold)}


def path_contract(config: Mapping[str, Any]) -> dict[str, Any] | None:
    settings = resolve_path_config(config.get("local_path_efficiency"))
    if settings is None:
        return None
    return {
        "version": 1,
        "profile": config["prototypes"]["profile"],
        "state_key": "observation.state",
        "position_axes": [0, 1, 2],
        "input": "original_fixed_frame_positions_all_clip_samples",
        "formula": "endpoint_l2_over_sum_adjacent_l2",
        "invalid": "nan_excluded_from_per_clip_geometric_mean",
        "no_valid_metrics": "neutral_one",
        **settings,
    }


def compute_local_path_efficiency(positions: ArrayLike, *, delta_path: float) -> float:
    """Use all internal adjacent pairs; stationary clips have no defined score."""
    settings = resolve_path_config({"delta_path": delta_path})
    values = np.asarray(positions, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("local_path_efficiency requires finite [L>=2, 3] positions")
    with np.errstate(over="ignore", invalid="ignore"):
        path = float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())
        net = float(np.linalg.norm(values[-1] - values[0]))
    if not np.isfinite(path) or not np.isfinite(net):
        raise ValueError("local_path_efficiency distances must be finite")
    if path <= settings["delta_path"]:
        return float("nan")
    return float(np.clip(net / path, 0, 1))


def path_summary(
    all_rows: Sequence[Mapping[str, Any]], selected_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = {}
    for name, rows in (("all", all_rows), ("selected", selected_rows)):
        values = [
            row["local_path_efficiency"]
            for row in rows
            if row["local_path_efficiency"] is not None
            and np.isfinite(row["local_path_efficiency"])
        ]
        result[f"{name}_mean"] = float(np.mean(values)) if values else None
        result[f"{name}_valid_count"] = len(values)
        result[f"{name}_invalid_count"] = len(rows) - len(values)
    return result
