"""Pure action-distribution and visual-probability primitives for Cocore."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np


MIN_ACTION_COUNT = 40
MIN_ACTION_FREQUENCY = 0.005
MAX_VISUAL_CENTERS = 16
VISUAL_SOFTMAX_TEMPERATURE = 0.1
STATE_THRESHOLD = 0.03

_ATOMIC_ACTION_ORDER = (
    ("move", "forward"),
    ("move", "backward"),
    ("move", "right"),
    ("move", "left"),
    ("move", "up"),
    ("move", "down"),
    ("block", "tilt up"),
    ("block", "tilt down"),
    ("block", "rotate counterclockwise"),
    ("block", "rotate clockwise"),
    ("block", "open gripper"),
    ("block", "close gripper"),
)
_MOVE_TOKENS = {
    value for kind, value in _ATOMIC_ACTION_ORDER if kind == "move"
}
_BLOCK_ACTIONS = {
    value for kind, value in _ATOMIC_ACTION_ORDER if kind == "block"
}


@dataclass(frozen=True)
class ActionParentAssignment:
    action_id: int
    label: str
    probability: float


@dataclass(frozen=True)
class ActionCategory:
    action_id: int | None
    label: str
    raw_count: int
    raw_proportion: float
    retained: bool
    parents: tuple[ActionParentAssignment, ...]
    effective_mass: float | None = None
    requested_centers: int | None = None
    actual_centers: int = 0


@dataclass(frozen=True)
class LeafPrototype:
    prototype_id: int
    label: str
    action_id: int
    action_label: str
    center_id: int


@dataclass(frozen=True)
class ActionCatalog:
    total_raw_actions: int
    action_categories: tuple[ActionCategory, ...]
    leaf_prototypes: tuple[LeafPrototype, ...] = ()

    @property
    def action_labels(self) -> tuple[str, ...]:
        assigned = sorted(
            (
                category
                for category in self.action_categories
                if category.action_id is not None
            ),
            key=lambda category: int(category.action_id),
        )
        return tuple(category.label for category in assigned)

    @property
    def retained_labels(self) -> frozenset[str]:
        return frozenset(
            category.label for category in self.action_categories if category.retained
        )

    @property
    def labels(self) -> tuple[str, ...]:
        assigned = sorted(self.leaf_prototypes, key=lambda leaf: leaf.prototype_id)
        return tuple(leaf.label for leaf in assigned)

    def to_dict(self) -> dict[str, object]:
        categories: list[dict[str, object]] = []
        for category in self.action_categories:
            payload = asdict(category)
            payload["parents"] = [asdict(parent) for parent in category.parents]
            categories.append(payload)
        return {
            "method": "motion_primitives",
            "schema_version": 4,
            "strategy": "trajectory_action_subset_then_visual_softmax",
            "constants": {
                "state_threshold": STATE_THRESHOLD,
                "min_action_count": MIN_ACTION_COUNT,
                "min_action_frequency": MIN_ACTION_FREQUENCY,
                "max_visual_centers": MAX_VISUAL_CENTERS,
                "visual_softmax_temperature": VISUAL_SOFTMAX_TEMPERATURE,
                "cluster_count": (
                    "min(16, 1 + floor(log2(effective_mass)))"
                ),
            },
            "total_raw_actions": self.total_raw_actions,
            "action_categories": categories,
            "leaf_prototypes": [asdict(leaf) for leaf in self.leaf_prototypes],
        }


def _atomic_actions(label: str) -> list[tuple[str, str]]:
    if not isinstance(label, str):
        raise ValueError(f"unknown motion primitive label {label!r}")
    if label == "stop":
        return []
    actions: list[tuple[str, str]] = []
    for block in label.split(", "):
        if block.startswith("move "):
            tokens = block.removeprefix("move ").split()
            if not tokens or any(token not in _MOVE_TOKENS for token in tokens):
                raise ValueError(f"unknown motion primitive label {label!r}")
            actions.extend(("move", token) for token in tokens)
        elif block in _BLOCK_ACTIONS:
            actions.append(("block", block))
        else:
            raise ValueError(f"unknown motion primitive label {label!r}")
    return actions


def _compose_actions(actions: Sequence[tuple[str, str]]) -> str:
    included = set(actions)
    ordered = [action for action in _ATOMIC_ACTION_ORDER if action in included]
    move = [value for kind, value in ordered if kind == "move"]
    blocks = [value for kind, value in ordered if kind == "block"]
    output = (["move " + " ".join(move)] if move else []) + blocks
    return ", ".join(output) if output else "stop"


def canonical_clip_action(first_half_label: str, second_half_label: str) -> str:
    """Return the canonical atomic-action union of a clip's two halves."""

    actions = _atomic_actions(first_half_label) + _atomic_actions(second_half_label)
    return _compose_actions(actions)


def action_parent_distribution(
    label: str,
    retained_counts: Mapping[str, int],
) -> tuple[tuple[str, float], ...]:
    """Map an action to all maximum-cardinality retained atomic subsets."""

    atomic = frozenset(_atomic_actions(label))
    candidates: list[tuple[str, int, int]] = []
    for parent_label, raw_count in retained_counts.items():
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, Integral)
            or raw_count <= 0
        ):
            raise ValueError("retained action counts must be positive integers")
        parent_atomic = frozenset(_atomic_actions(parent_label))
        if parent_atomic and parent_atomic.issubset(atomic):
            candidates.append((parent_label, int(raw_count), len(parent_atomic)))
    if not candidates:
        return (("stop", 1.0),)

    maximum_cardinality = max(cardinality for _, _, cardinality in candidates)
    parents = [
        (parent_label, raw_count)
        for parent_label, raw_count, cardinality in candidates
        if cardinality == maximum_cardinality
    ]
    parents.sort(key=lambda item: (-item[1], item[0]))
    total = sum(raw_count for _, raw_count in parents)
    return tuple(
        (parent_label, raw_count / total) for parent_label, raw_count in parents
    )


def cluster_count_for_mass(effective_mass: float) -> int:
    """Return ``min(16, 1 + floor(log2(mass)))`` for positive finite mass."""

    if isinstance(effective_mass, bool) or not isinstance(effective_mass, Real):
        raise ValueError("effective action mass must be a finite positive float")
    value = float(effective_mass)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("effective action mass must be a finite positive float")
    return min(MAX_VISUAL_CENTERS, 1 + math.floor(math.log2(value)))


def visual_center_probabilities(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Return stable conditional probabilities over every visual center."""

    try:
        left = np.asarray(values, dtype=np.float64)
        right = np.asarray(centers, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "visual values and centers must be finite non-empty matrices"
        ) from error
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[0] == 0
        or right.shape[0] == 0
        or left.shape[1] == 0
        or left.shape[1] != right.shape[1]
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("visual values and centers must be finite non-empty matrices")
    if right.shape[0] == 1:
        return np.ones((left.shape[0], 1), dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        squared_distances = np.sum(
            np.square(left[:, None, :] - right[None, :, :]),
            axis=2,
        )
    if not np.all(np.isfinite(squared_distances)):
        raise ValueError("visual squared distances must be finite")
    logits = -squared_distances / VISUAL_SOFTMAX_TEMPERATURE
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities


def create_action_catalog(
    counts: Mapping[str, int],
    total_labels: int,
) -> ActionCatalog:
    """Build a deterministic schema-4 action catalog from raw action counts."""

    if isinstance(total_labels, bool) or not isinstance(total_labels, Integral):
        raise ValueError("motion primitive total_labels must be a non-negative integer")
    total = int(total_labels)
    if total < 0:
        raise ValueError("motion primitive counts must be non-negative")

    normalized: dict[str, int] = {}
    for raw_label, raw_count in counts.items():
        label = str(raw_label)
        _atomic_actions(label)
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, Integral)
            or raw_count < 0
        ):
            raise ValueError("motion primitive counts must be non-negative integers")
        normalized[label] = int(raw_count)
    if sum(normalized.values()) != total:
        raise ValueError("motion primitive counts do not match total_labels")
    normalized.setdefault("stop", 0)

    ordered = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    threshold = max(
        MIN_ACTION_COUNT,
        math.ceil(MIN_ACTION_FREQUENCY * total),
    )
    retained = {
        label for label, count in ordered if count >= threshold
    }
    retained_non_stop_counts = {
        label: count
        for label, count in ordered
        if label != "stop" and label in retained
    }
    if any(label != "stop" and count > 0 for label, count in ordered) and not (
        retained_non_stop_counts
    ):
        raise ValueError("no non-stop action meets retention threshold")

    action_labels = [
        label
        for label, _ in ordered
        if label != "stop" and label in retained
    ]
    action_labels.append("stop")
    action_ids = {label: index for index, label in enumerate(action_labels)}

    categories: list[ActionCategory] = []
    for label, raw_count in ordered:
        distribution = action_parent_distribution(label, retained_non_stop_counts)
        parents = tuple(
            ActionParentAssignment(
                action_id=action_ids[parent_label],
                label=parent_label,
                probability=probability,
            )
            for parent_label, probability in distribution
        )
        categories.append(
            ActionCategory(
                action_id=action_ids.get(label),
                label=label,
                raw_count=raw_count,
                raw_proportion=(raw_count / total if total else 0.0),
                retained=label in retained,
                parents=parents,
            )
        )
    return ActionCatalog(
        total_raw_actions=total,
        action_categories=tuple(categories),
    )
