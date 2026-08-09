"""LIBERO motion-primitive prototypes and deterministic soft assignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from trajectory_data import DatasetAdapter

from libero_motion_primitives import (
    classify_motion_primitive,
    generate_motion_primitives,
    make_libero_config,
)
from relcore.schemas import ClipRecord

from .prototypes import PrototypeData

STATE_KEY = "observation.state"
HORIZON = 8
STATE_THRESHOLD = 0.03
MIN_FREQUENCY = 0.005
DOMINANCE_RATIO = 4.0
FALLBACK_WEIGHT = 0.8
ASSIGNMENT_SLOTS = 4


@dataclass(frozen=True)
class MotionPrimitiveCategory:
    prototype_id: int | None
    label: str
    count: int
    proportion: float
    retained: bool


@dataclass(frozen=True)
class MotionPrimitiveCatalog:
    total_labels: int
    categories: tuple[MotionPrimitiveCategory, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        assigned = sorted(
            (category for category in self.categories if category.prototype_id is not None),
            key=lambda category: int(category.prototype_id),
        )
        return tuple(category.label for category in assigned)

    @property
    def retained_labels(self) -> frozenset[str]:
        return frozenset(category.label for category in self.categories if category.retained)

    @property
    def proportions(self) -> dict[str, float]:
        return {category.label: category.proportion for category in self.categories}

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "motion_primitives",
            "constants": {
                "horizon": HORIZON,
                "state_threshold": STATE_THRESHOLD,
                "min_frequency": MIN_FREQUENCY,
                "dominance_ratio": DOMINANCE_RATIO,
                "fallback_weight": FALLBACK_WEIGHT,
                "clip_anchors": [0, 7, 14],
            },
            "total_labels": self.total_labels,
            "categories": [asdict(category) for category in self.categories],
        }


def create_primitive_catalog(
    counts: Mapping[str, int],
    total_labels: int,
) -> MotionPrimitiveCatalog:
    normalized = Counter({str(label): int(count) for label, count in counts.items()})
    if total_labels < 0 or any(count < 0 for count in normalized.values()):
        raise ValueError("motion primitive counts must be non-negative")
    if sum(normalized.values()) != total_labels:
        raise ValueError("motion primitive counts do not match total_labels")
    normalized.setdefault("stop", 0)
    ordered = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    retained = {
        label
        for label, count in ordered
        if total_labels > 0 and count / total_labels > MIN_FREQUENCY
    }
    assignable = [label for label, _ in ordered if label in retained or label == "stop"]
    prototype_ids = {label: index for index, label in enumerate(assignable)}
    categories = tuple(
        MotionPrimitiveCategory(
            prototype_id=prototype_ids.get(label),
            label=label,
            count=count,
            proportion=(count / total_labels if total_labels else 0.0),
            retained=label in retained,
        )
        for label, count in ordered
    )
    return MotionPrimitiveCatalog(total_labels=total_labels, categories=categories)


_MOVE_TOKENS = {"forward", "backward", "right", "left", "up", "down"}
_BLOCK_ACTIONS = {
    "tilt up",
    "tilt down",
    "rotate counterclockwise",
    "rotate clockwise",
    "open gripper",
    "close gripper",
}


def _atomic_actions(label: str) -> list[tuple[str, str]]:
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
    move = [value for kind, value in actions if kind == "move"]
    blocks = [value for kind, value in actions if kind == "block"]
    output = (["move " + " ".join(move)] if move else []) + blocks
    return ", ".join(output) if output else "stop"


def remove_one_atomic_action(label: str) -> tuple[str, ...]:
    actions = _atomic_actions(label)
    candidates: list[str] = []
    for index in range(len(actions)):
        candidate = _compose_actions(actions[:index] + actions[index + 1 :])
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def soften_primitive_label(
    label: str,
    retained_labels: set[str] | frozenset[str],
    proportions: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if label in retained_labels:
        return ((label, 1.0),)
    candidates = [
        candidate for candidate in remove_one_atomic_action(label) if candidate in retained_labels
    ]
    candidates.sort(key=lambda candidate: (-float(proportions[candidate]), candidate))
    if not candidates:
        return (("stop", FALLBACK_WEIGHT),)
    if len(candidates) == 1:
        return ((candidates[0], FALLBACK_WEIGHT),)
    first, second = candidates[:2]
    first_proportion = float(proportions[first])
    second_proportion = float(proportions[second])
    if first_proportion / second_proportion > DOMINANCE_RATIO:
        return ((first, FALLBACK_WEIGHT),)
    total = first_proportion + second_proportion
    return ((first, first_proportion / total), (second, second_proportion / total))


def _validated_states(episode_id: int, observations: Mapping[str, np.ndarray]) -> np.ndarray:
    if STATE_KEY not in observations:
        raise ValueError(f"episode {episode_id} is missing {STATE_KEY!r}")
    states = np.asarray(observations[STATE_KEY])
    if states.ndim != 2 or states.shape[1] < 8:
        raise ValueError(f"episode {episode_id}: observation.state dimension must be at least 8")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"episode {episode_id}: observation.state must be finite")
    return states


def build_motion_primitive_prototypes(
    adapter: DatasetAdapter,
    clips: Sequence[ClipRecord],
    *,
    max_episodes: int | None = None,
    num_workers: int = 0,
) -> tuple[PrototypeData, MotionPrimitiveCatalog]:
    if STATE_KEY not in adapter.vector_observation_keys:
        raise ValueError(f"motion_primitives requires vector observation {STATE_KEY!r}")
    clips_by_episode: dict[int, list[ClipRecord]] = {}
    for clip in clips:
        clips_by_episode.setdefault(clip.episode_id, []).append(clip)
    raw_labels: dict[str, tuple[str, str]] = {}
    counts: Counter[str] = Counter()
    seen_episodes: set[int] = set()
    config = make_libero_config(
        horizon=HORIZON,
        threshold=STATE_THRESHOLD,
        tail_strategy="truncate",
    )
    for episode in adapter.iter_episodes(
        num_workers=num_workers,
        max_episodes=max_episodes,
        load_images=False,
    ):
        if episode.episode_id in seen_episodes:
            raise ValueError(f"duplicate motion primitive episode {episode.episode_id}")
        seen_episodes.add(episode.episode_id)
        states = _validated_states(episode.episode_id, episode.observations)
        if len(states) != episode.length:
            raise ValueError(f"episode {episode.episode_id}: observation.state length mismatch")
        counts.update(generate_motion_primitives(states, config))
        for clip in clips_by_episode.get(episode.episode_id, ()):
            if clip.length != 15 or clip.end_step - clip.start_step + 1 != 15:
                raise ValueError("motion_primitives requires 15-frame clips")
            if clip.start_step < 0 or clip.end_step >= len(states):
                raise ValueError(f"clip {clip.sample_id} is outside its episode")
            window = states[clip.start_step : clip.end_step + 1]
            raw_labels[clip.sample_id] = (
                classify_motion_primitive(window[0], window[7], config),
                classify_motion_primitive(window[7], window[14], config),
            )
    missing = [clip.sample_id for clip in clips if clip.sample_id not in raw_labels]
    if missing:
        raise ValueError(f"motion primitive pass did not yield clips: {missing[:3]}")

    catalog = create_primitive_catalog(counts, sum(counts.values()))
    label_to_id = {label: index for index, label in enumerate(catalog.labels)}
    indices = np.full((len(clips), ASSIGNMENT_SLOTS), -1, dtype=np.int32)
    weights = np.zeros((len(clips), ASSIGNMENT_SLOTS), dtype=np.float32)
    for clip_index, clip in enumerate(clips):
        merged: dict[str, float] = {}
        for raw_label in raw_labels[clip.sample_id]:
            for label, weight in soften_primitive_label(
                raw_label,
                catalog.retained_labels,
                catalog.proportions,
            ):
                merged[label] = max(merged.get(label, 0.0), float(weight))
        assignments = sorted(
            merged.items(),
            key=lambda item: (-item[1], -catalog.proportions[item[0]], item[0]),
        )
        for slot, (label, weight) in enumerate(assignments):
            indices[clip_index, slot] = label_to_id[label]
            weights[clip_index, slot] = weight
    return PrototypeData(None, indices, weights, catalog.labels), catalog
