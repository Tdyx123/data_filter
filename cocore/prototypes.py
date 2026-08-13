"""Cocore-owned hierarchical motion prototypes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from libero_motion_primitives import (
    classify_motion_primitive,
    make_libero_config,
)
from relcore.graph.prototypes import PrototypeData
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter


MIN_FREQUENCY = 0.005
DOMINANCE_RATIO = 4.0
FALLBACK_WEIGHT = 0.8
STATE_KEY = "observation.state"
STATE_THRESHOLD = 0.03
HALF_ACTION_SLOTS = 2
CLIP_ANCHORS = (0, 7, 14)

_MOVE_TOKENS = {"forward", "backward", "right", "left", "up", "down"}
_BLOCK_ACTIONS = {
    "tilt up",
    "tilt down",
    "rotate counterclockwise",
    "rotate clockwise",
    "open gripper",
    "close gripper",
}


@dataclass(frozen=True)
class ActionCategory:
    action_id: int | None
    label: str
    count: int
    proportion: float
    retained: bool
    fallback: bool
    bucket_size: int = 0
    requested_clusters: int | None = None
    actual_clusters: int = 0
    requested_top_m: int | None = None
    top_m: int = 0
    distance_q10: float | None = None
    distance_q90: float | None = None


@dataclass(frozen=True)
class LeafPrototype:
    prototype_id: int
    label: str
    action_id: int
    action_label: str
    cluster_id: int
    fallback: bool


@dataclass(frozen=True)
class ActionCatalog:
    total_labels: int
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
    def proportions(self) -> dict[str, float]:
        return {
            category.label: category.proportion for category in self.action_categories
        }

    @property
    def labels(self) -> tuple[str, ...]:
        assigned = sorted(self.leaf_prototypes, key=lambda leaf: leaf.prototype_id)
        return tuple(leaf.label for leaf in assigned)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "motion_primitives",
            "schema_version": 3,
            "strategy": "action_halves_then_visual_mean_kmeans",
            "constants": {
                "state_threshold": STATE_THRESHOLD,
                "min_frequency": MIN_FREQUENCY,
                "dominance_ratio": DOMINANCE_RATIO,
                "fallback_weight": FALLBACK_WEIGHT,
                "clip_anchors": list(CLIP_ANCHORS),
                "visual_half_windows": [[0, 8], [7, 15]],
                "cluster_count": "floor(3 + 2 * log2(200 * proportion))",
                "top_m": "floor(2 + 1.5 * log2(200 * proportion))",
                "distance_quantiles": [0.1, 0.9],
                "distance_weight_range": [1.0, 0.3],
            },
            "total_labels": self.total_labels,
            "action_categories": [asdict(category) for category in self.action_categories],
            "leaf_prototypes": [asdict(leaf) for leaf in self.leaf_prototypes],
        }


@dataclass(frozen=True)
class HierarchicalPrototypeResult:
    prototypes: PrototypeData
    catalog: ActionCatalog
    action_weights: np.ndarray
    distance_weights: np.ndarray
    half_action_labels: np.ndarray | None = None


def cluster_limits(proportion: float, bucket_size: int) -> tuple[int, int]:
    """Return the floored hierarchical K/M limits for one retained action."""

    value = float(proportion)
    if not math.isfinite(value) or value <= MIN_FREQUENCY:
        raise ValueError("cluster proportion must be finite and greater than 0.005")
    if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size <= 0:
        raise ValueError("cluster bucket_size must be a positive integer")
    requested_clusters, requested_top_m = requested_cluster_limits(value)
    clusters = min(bucket_size, requested_clusters)
    return clusters, min(clusters, requested_top_m)


def requested_cluster_limits(proportion: float) -> tuple[int, int]:
    value = float(proportion)
    return (
        math.floor(3.0 + 2.0 * math.log2(200.0 * value)),
        math.floor(2.0 + 1.5 * math.log2(200.0 * value)),
    )


def distance_soft_weights(
    distances: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Map one action bucket's Euclidean distances onto clipped [1, 0.3] weights."""

    values = np.asarray(distances, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("distances must be a finite non-empty matrix")
    if np.any(values < 0.0):
        raise ValueError("distances cannot be negative")
    lower, upper = np.quantile(values, (0.1, 0.9))
    lower = float(lower)
    upper = float(upper)
    if lower == upper:
        return np.ones_like(values, dtype=np.float32), lower, upper
    scaled = 1.0 - 0.7 * (values - lower) / (upper - lower)
    return np.clip(scaled, 0.3, 1.0).astype(np.float32), lower, upper


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


def _compose_actions(actions: list[tuple[str, str]]) -> str:
    move = [value for kind, value in actions if kind == "move"]
    blocks = [value for kind, value in actions if kind == "block"]
    output = (["move " + " ".join(move)] if move else []) + blocks
    return ", ".join(output) if output else "stop"


def _remove_one_atomic_action(label: str) -> tuple[str, ...]:
    actions = _atomic_actions(label)
    candidates: list[str] = []
    for index in range(len(actions)):
        candidate = _compose_actions(actions[:index] + actions[index + 1 :])
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def soften_action_label(
    label: str,
    retained_labels: set[str] | frozenset[str],
    proportions: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    if label in retained_labels:
        return ((label, 1.0),)
    candidates = [
        candidate
        for candidate in _remove_one_atomic_action(label)
        if candidate in retained_labels
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


def _fit_action_clusters(
    embeddings: np.ndarray,
    sample_weights: np.ndarray,
    *,
    clusters: int,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=clusters,
        batch_size=min(batch_size, len(embeddings)),
        max_iter=max_iter,
        random_state=seed,
        n_init=10,
    )
    model.fit(embeddings, sample_weight=sample_weights)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    masses = np.bincount(
        np.asarray(model.labels_, dtype=np.int64),
        weights=sample_weights,
        minlength=clusters,
    )
    order = sorted(
        range(clusters),
        key=lambda index: (-float(masses[index]), tuple(float(v) for v in centers[index])),
    )
    return centers[np.asarray(order, dtype=np.int64)]


def _euclidean_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Compute N-by-K distances without materializing an N-by-K-by-D tensor."""

    left = np.asarray(values, dtype=np.float32)
    right = np.asarray(centers, dtype=np.float32)
    squared = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * (left @ right.T)
    )
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared, out=squared).astype(np.float32, copy=False)


def refine_action_prototypes(
    catalog: ActionCatalog,
    action_indices: np.ndarray,
    action_weights: np.ndarray,
    visual_half_embeddings: np.ndarray,
    *,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> HierarchicalPrototypeResult:
    values = np.asarray(visual_half_embeddings, dtype=np.float32)
    first_indices = np.asarray(action_indices, dtype=np.int32)
    first_weights = np.asarray(action_weights, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[1] != 2
        or values.shape[2] == 0
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or first_indices.ndim != 3
        or first_indices.shape != first_weights.shape
        or first_indices.shape[:2] != values.shape[:2]
        or first_indices.shape[2] == 0
        or not np.all(np.isfinite(first_weights))
        or np.any(first_weights < 0.0)
    ):
        raise ValueError("hierarchical prototype inputs are invalid")
    values = values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1.0e-8)
    if batch_size <= 0 or max_iter <= 0:
        raise ValueError("hierarchical prototype KMeans limits must be positive")

    categories_by_id = {
        int(category.action_id): category
        for category in catalog.action_categories
        if category.action_id is not None
    }
    if sorted(categories_by_id) != list(range(len(categories_by_id))):
        raise ValueError("action ids must be contiguous")
    valid = first_indices >= 0
    if np.any(first_indices[valid] >= len(categories_by_id)):
        raise ValueError("first-stage action index is out of range")
    if np.any((first_indices < 0) != (first_weights <= 0.0)):
        raise ValueError("first-stage padding indices and weights do not align")

    per_clip: list[dict[int, tuple[float, float, float, int, int, int]]] = [
        {} for _ in range(len(values))
    ]
    leaves: list[LeafPrototype] = []
    centers: list[np.ndarray] = []
    updated_categories: dict[str, ActionCategory] = {}

    def retain_assignment(
        clip_index: int,
        half_index: int,
        leaf_id: int,
        first_weight: float,
        second_weight: float,
        action_id: int,
        cluster_id: int,
    ) -> None:
        combined = float(first_weight) * float(second_weight)
        candidate = (
            combined,
            float(first_weight),
            float(second_weight),
            action_id,
            cluster_id,
            half_index,
        )
        existing = per_clip[clip_index].get(leaf_id)
        if existing is None or combined > existing[0] or (
            combined == existing[0] and half_index < existing[5]
        ):
            per_clip[clip_index][leaf_id] = candidate

    for action_id in sorted(categories_by_id):
        category = categories_by_id[action_id]
        membership = first_indices == action_id
        member_map: dict[tuple[int, int], float] = {}
        for clip_index, half_index, slot in np.argwhere(membership):
            key = (int(clip_index), int(half_index))
            member_map[key] = max(
                member_map.get(key, 0.0),
                float(first_weights[clip_index, half_index, slot]),
            )
        member_positions = sorted(member_map)
        member_clips = np.asarray(
            [position[0] for position in member_positions], dtype=np.int64
        )
        member_halves = np.asarray(
            [position[1] for position in member_positions], dtype=np.int64
        )
        member_weights = np.asarray(
            [member_map[position] for position in member_positions], dtype=np.float32
        )
        member_values = values[member_clips, member_halves]

        if category.fallback:
            leaf_id = len(leaves)
            leaves.append(
                LeafPrototype(
                    prototype_id=leaf_id,
                    label="stop::fallback",
                    action_id=action_id,
                    action_label=category.label,
                    cluster_id=0,
                    fallback=True,
                )
            )
            center = (
                np.average(member_values, axis=0, weights=member_weights).astype(np.float32)
                if len(member_positions)
                else np.zeros(values.shape[2], dtype=np.float32)
            )
            centers.append(center)
            for clip_index, half_index, first_weight in zip(
                member_clips, member_halves, member_weights, strict=True
            ):
                retain_assignment(
                    int(clip_index),
                    int(half_index),
                    leaf_id,
                    float(first_weight),
                    1.0,
                    action_id,
                    0,
                )
            updated_categories[category.label] = replace(
                category,
                bucket_size=len(member_positions),
                actual_clusters=1,
                top_m=1,
            )
            continue

        requested_clusters, requested_top_m = requested_cluster_limits(category.proportion)
        if not len(member_positions):
            updated_categories[category.label] = replace(
                category,
                requested_clusters=requested_clusters,
                requested_top_m=requested_top_m,
            )
            continue
        cluster_count, top_m = cluster_limits(category.proportion, len(member_positions))
        action_centers = _fit_action_clusters(
            member_values,
            member_weights,
            clusters=cluster_count,
            batch_size=batch_size,
            max_iter=max_iter,
            seed=seed + action_id,
        )
        distances = _euclidean_distances(member_values, action_centers)
        second_weights, lower, upper = distance_soft_weights(distances)
        leaf_offset = len(leaves)
        for cluster_id, center in enumerate(action_centers):
            leaf_id = len(leaves)
            leaves.append(
                LeafPrototype(
                    prototype_id=leaf_id,
                    label=f"{category.label}::cluster_{cluster_id}",
                    action_id=action_id,
                    action_label=category.label,
                    cluster_id=cluster_id,
                    fallback=False,
                )
            )
            centers.append(center)
        best_clusters = np.argsort(-second_weights, axis=1, kind="stable")[:, :top_m]
        for row, (clip_index, half_index, first_weight) in enumerate(
            zip(member_clips, member_halves, member_weights, strict=True)
        ):
            for cluster_id in best_clusters[row]:
                cluster_id = int(cluster_id)
                second_weight = float(second_weights[row, cluster_id])
                leaf_id = leaf_offset + cluster_id
                retain_assignment(
                    int(clip_index),
                    int(half_index),
                    leaf_id,
                    float(first_weight),
                    second_weight,
                    action_id,
                    cluster_id,
                )
        updated_categories[category.label] = replace(
            category,
            bucket_size=len(member_positions),
            requested_clusters=requested_clusters,
            actual_clusters=cluster_count,
            requested_top_m=requested_top_m,
            top_m=top_m,
            distance_q10=lower,
            distance_q90=upper,
        )

    ordered_per_clip = [
        sorted(
            ((leaf_id, *assignment) for leaf_id, assignment in assignments.items()),
            key=lambda row: (-row[1], row[4], row[5]),
        )
        for assignments in per_clip
    ]
    if any(not assignments for assignments in ordered_per_clip):
        raise ValueError("every clip must retain at least one hierarchical prototype")
    width = max(len(assignments) for assignments in ordered_per_clip)
    prototype_indices = np.full((len(values), width), -1, dtype=np.int32)
    prototype_weights = np.zeros((len(values), width), dtype=np.float32)
    output_action_weights = np.zeros((len(values), width), dtype=np.float32)
    output_distance_weights = np.zeros((len(values), width), dtype=np.float32)
    for clip_index, assignments in enumerate(ordered_per_clip):
        for slot, (
            leaf_id,
            weight,
            first_weight,
            second_weight,
            _,
            _,
            _,
        ) in enumerate(assignments):
            prototype_indices[clip_index, slot] = leaf_id
            prototype_weights[clip_index, slot] = weight
            output_action_weights[clip_index, slot] = first_weight
            output_distance_weights[clip_index, slot] = second_weight

    refined_catalog = ActionCatalog(
        total_labels=catalog.total_labels,
        action_categories=tuple(
            updated_categories.get(category.label, category)
            for category in catalog.action_categories
        ),
        leaf_prototypes=tuple(leaves),
    )
    prototype_data = PrototypeData(
        centers=np.stack(centers).astype(np.float32),
        indices=prototype_indices,
        weights=prototype_weights,
        labels=refined_catalog.labels,
    )
    return HierarchicalPrototypeResult(
        prototypes=prototype_data,
        catalog=refined_catalog,
        action_weights=output_action_weights,
        distance_weights=output_distance_weights,
    )


def _validated_states(episode_id: int, observations: Mapping[str, np.ndarray]) -> np.ndarray:
    if STATE_KEY not in observations:
        raise ValueError(f"episode {episode_id} is missing {STATE_KEY!r}")
    states = np.asarray(observations[STATE_KEY])
    if states.ndim != 2 or states.shape[1] < 8:
        raise ValueError(f"episode {episode_id}: observation.state dimension must be at least 8")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"episode {episode_id}: observation.state must be finite")
    return states


def assign_half_actions(
    catalog: ActionCatalog,
    half_action_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Soften each raw half-clip action without mixing the two halves."""

    labels = np.asarray(half_action_labels)
    if labels.ndim != 2 or labels.shape[1] != 2 or labels.dtype.kind not in {"U", "S"}:
        raise ValueError("half action labels must have shape [clips, 2]")
    label_to_id = {label: index for index, label in enumerate(catalog.action_labels)}
    indices = np.full(
        (len(labels), 2, HALF_ACTION_SLOTS), -1, dtype=np.int32
    )
    weights = np.zeros(indices.shape, dtype=np.float32)
    for clip_index in range(len(labels)):
        for half_index in range(2):
            assignments = soften_action_label(
                str(labels[clip_index, half_index]),
                catalog.retained_labels,
                catalog.proportions,
            )
            if len(assignments) > HALF_ACTION_SLOTS:
                raise ValueError("half action assignments exceed the fixed slot count")
            for slot, (label, weight) in enumerate(assignments):
                indices[clip_index, half_index, slot] = label_to_id[label]
                weights[clip_index, half_index, slot] = weight
    return indices, weights


def build_hierarchical_motion_prototypes(
    adapter: DatasetAdapter,
    clips: Sequence[ClipRecord],
    visual_half_embeddings: np.ndarray,
    *,
    batch_size: int,
    max_iter: int,
    seed: int,
    max_episodes: int | None,
    num_workers: int,
) -> HierarchicalPrototypeResult:
    """Build Cocore's aligned half-action, visual-refined leaf prototypes."""

    values = np.asarray(visual_half_embeddings, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] != len(clips)
        or values.shape[1] != 2
        or values.shape[2] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("visual half embeddings must align with clips and be finite")
    if STATE_KEY not in adapter.vector_observation_keys:
        raise ValueError(f"motion_primitives requires vector observation {STATE_KEY!r}")

    clips_by_episode: dict[int, list[ClipRecord]] = {}
    for clip in clips:
        clips_by_episode.setdefault(clip.episode_id, []).append(clip)
    raw_labels: dict[str, tuple[str, str]] = {}
    seen_episodes: set[int] = set()
    primitive_config = make_libero_config(
        threshold=STATE_THRESHOLD,
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
        for clip in clips_by_episode.get(episode.episode_id, ()):
            if clip.length != 15 or clip.end_step - clip.start_step + 1 != 15:
                raise ValueError("motion_primitives requires 15-frame clips")
            if clip.start_step < 0 or clip.end_step >= len(states):
                raise ValueError(f"clip {clip.sample_id} is outside its episode")
            window = states[clip.start_step : clip.end_step + 1]
            raw_labels[clip.sample_id] = (
                classify_motion_primitive(window[0], window[7], primitive_config),
                classify_motion_primitive(window[7], window[14], primitive_config),
            )
    missing = [clip.sample_id for clip in clips if clip.sample_id not in raw_labels]
    if missing:
        raise ValueError(f"motion primitive pass did not yield clips: {missing[:3]}")

    half_action_labels = np.asarray(
        [raw_labels[clip.sample_id] for clip in clips], dtype=np.str_
    )
    counts = Counter(str(label) for label in half_action_labels.ravel())
    catalog = create_action_catalog(counts, int(half_action_labels.size))
    first_indices, first_weights = assign_half_actions(catalog, half_action_labels)
    refined = refine_action_prototypes(
        catalog,
        first_indices,
        first_weights,
        values,
        batch_size=batch_size,
        max_iter=max_iter,
        seed=seed,
    )
    return replace(refined, half_action_labels=half_action_labels)


def create_action_catalog(
    counts: Mapping[str, int],
    total_labels: int,
) -> ActionCatalog:
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
    action_ids = {label: index for index, label in enumerate(assignable)}
    return ActionCatalog(
        total_labels=total_labels,
        action_categories=tuple(
            ActionCategory(
                action_id=action_ids.get(label),
                label=label,
                count=count,
                proportion=(count / total_labels if total_labels else 0.0),
                retained=label in retained,
                fallback=label == "stop" and label not in retained,
            )
            for label, count in ordered
        ),
    )
