"""Hard action buckets and nearest visual-leaf assignments for Cocore."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
import time
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from libero_motion_primitives import classify_motion_primitive, make_libero_config
from relcore.graph.prototypes import PrototypeData
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord

from cocore.timing import TimingCallback


MIN_ACTION_COUNT = 400
MIN_ACTION_FREQUENCY = 0.005
MAX_VISUAL_CENTERS = 16
MIN_DISTANCE_WEIGHT = 0.3
STATE_THRESHOLD = 0.03
STATE_KEY = "observation.state"
TRAJECTORY_WINDOW_LENGTH = 8
CLIP_LENGTH = 15

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
_MOVE_TOKENS = {value for kind, value in _ATOMIC_ACTION_ORDER if kind == "move"}
_BLOCK_ACTIONS = {value for kind, value in _ATOMIC_ACTION_ORDER if kind == "block"}


@dataclass(frozen=True)
class ActionCategory:
    action_id: int | None
    label: str
    raw_count: int
    raw_proportion: float
    retained: bool
    training_count: int = 0
    requested_centers: int | None = None
    actual_centers: int = 0
    nearest_distance_q10: float | None = None
    nearest_distance_q90: float | None = None


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
            (category for category in self.action_categories if category.action_id is not None),
            key=lambda category: int(category.action_id),
        )
        return tuple(category.label for category in assigned)

    @property
    def retained_labels(self) -> frozenset[str]:
        return frozenset(category.label for category in self.action_categories if category.retained)

    @property
    def labels(self) -> tuple[str, ...]:
        assigned = sorted(self.leaf_prototypes, key=lambda leaf: leaf.prototype_id)
        return tuple(leaf.label for leaf in assigned)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "motion_primitives",
            "schema_version": 5,
            "strategy": "trajectory_retained_action_then_half_visual_nearest",
            "constants": {
                "state_threshold": STATE_THRESHOLD,
                "min_action_count": MIN_ACTION_COUNT,
                "min_action_frequency": MIN_ACTION_FREQUENCY,
                "max_visual_centers": MAX_VISUAL_CENTERS,
                "visual_half_windows": [[0, 8], [7, 15]],
                "cluster_count": "min(16, floor(log2(training_count)) - 2)",
                "retention_weight": "0.5 + 0.5 * retained_atomic_ratio",
                "distance_quantiles": [0.1, 0.9],
                "distance_weight_range": [1.0, MIN_DISTANCE_WEIGHT],
                "duplicate_merge": "max + 0.5 * min",
            },
            "total_raw_actions": self.total_raw_actions,
            "action_categories": [asdict(category) for category in self.action_categories],
            "leaf_prototypes": [asdict(leaf) for leaf in self.leaf_prototypes],
        }


@dataclass(frozen=True)
class HierarchicalPrototypeResult:
    prototypes: PrototypeData
    catalog: ActionCatalog
    half_action_labels: np.ndarray


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


def maximum_retained_parents(
    label: str,
    retained_counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Return every maximum-cardinality retained atomic subset in stable order."""

    atomic = frozenset(_atomic_actions(label))
    candidates: list[tuple[str, int, int]] = []
    for parent_label, raw_count in retained_counts.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral) or raw_count <= 0:
            raise ValueError("retained action counts must be positive integers")
        parent_atomic = frozenset(_atomic_actions(parent_label))
        if parent_atomic and parent_atomic.issubset(atomic):
            candidates.append((parent_label, int(raw_count), len(parent_atomic)))
    if not candidates:
        return ("stop",)
    maximum_cardinality = max(cardinality for _, _, cardinality in candidates)
    parents = [
        (parent_label, raw_count)
        for parent_label, raw_count, cardinality in candidates
        if cardinality == maximum_cardinality
    ]
    parents.sort(key=lambda item: (-item[1], item[0]))
    return tuple(parent_label for parent_label, _ in parents)


def retention_weight(raw_label: str, parent_label: str) -> float:
    """Map the retained atomic-action ratio linearly onto ``[0.5, 1]``."""

    raw = frozenset(_atomic_actions(raw_label))
    parent = frozenset(_atomic_actions(parent_label))
    if raw_label == parent_label:
        return 1.0
    if not parent:
        return 0.5
    if not raw or not parent.issubset(raw):
        raise ValueError("parent action must be an atomic subset of the raw action")
    return 0.5 + 0.5 * (len(parent) / len(raw))


def cluster_count_for_training_count(training_count: float) -> int:
    """Return ``min(16, floor(log2(training_count)) - 2)`` when positive."""

    if isinstance(training_count, bool) or not isinstance(training_count, Real):
        raise ValueError("training count must be a finite positive number")
    value = float(training_count)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("training count must be a finite positive number")
    clusters = min(MAX_VISUAL_CENTERS, math.floor(math.log2(value)) - 2)
    if clusters <= 0:
        raise ValueError(
            "training count must be finite positive and produce a positive cluster count"
        )
    return clusters


def nearest_distance_bounds(distances: np.ndarray) -> tuple[float, float]:
    """Return q10/q90 for one action bucket's nearest-center distances."""

    try:
        values = np.asarray(distances, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("nearest distances must be a finite non-empty vector") from error
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("nearest distances must be a finite non-empty non-negative vector")
    lower, upper = np.quantile(values, (0.1, 0.9))
    return float(lower), float(upper)


def distance_confidence(distance: float, lower: float, upper: float) -> float:
    """Map Euclidean distance onto clipped ``[0.3, 1]`` confidence."""

    values = (distance, lower, upper)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("distance confidence inputs must be finite non-negative numbers")
    value, low, high = (float(item) for item in values)
    if not all(math.isfinite(item) for item in (value, low, high)) or min(value, low) < 0.0:
        raise ValueError("distance confidence inputs must be finite non-negative numbers")
    if high < low:
        raise ValueError("distance confidence upper bound cannot be below lower bound")
    if high == low:
        return 1.0
    scaled = 1.0 - (1.0 - MIN_DISTANCE_WEIGHT) * (value - low) / (high - low)
    return float(np.clip(scaled, MIN_DISTANCE_WEIGHT, 1.0))


def merge_half_leaf_assignments(
    assignments: Sequence[tuple[int, float]],
) -> tuple[tuple[int, float], ...]:
    """Merge two half-clip leaves with ``max + 0.5 * min`` for duplicates."""

    if not 1 <= len(assignments) <= 2:
        raise ValueError("half-clip assignments must contain one or two leaves")
    merged: dict[int, float] = {}
    for leaf_id, raw_weight in assignments:
        if (
            isinstance(leaf_id, bool)
            or not isinstance(leaf_id, Integral)
            or int(leaf_id) < 0
            or isinstance(raw_weight, bool)
            or not isinstance(raw_weight, Real)
            or not math.isfinite(float(raw_weight))
            or float(raw_weight) <= 0.0
        ):
            raise ValueError("half-clip assignments must contain valid leaves and weights")
        key = int(leaf_id)
        weight = float(raw_weight)
        if key in merged:
            larger = max(merged[key], weight)
            smaller = min(merged[key], weight)
            merged[key] = larger + 0.5 * smaller
        else:
            merged[key] = weight
    quantized = tuple(
        (leaf_id, float(np.float32(weight))) for leaf_id, weight in merged.items()
    )
    if any(not math.isfinite(weight) for _, weight in quantized):
        raise ValueError("half-clip assignment weights must fit float32")
    return tuple(sorted(quantized, key=lambda item: (-item[1], item[0])))


def create_action_catalog(
    counts: Mapping[str, int],
    total_labels: int,
) -> ActionCatalog:
    """Build a deterministic schema-5 action catalog from raw action counts."""

    if isinstance(total_labels, bool) or not isinstance(total_labels, Integral):
        raise ValueError("motion primitive total_labels must be a non-negative integer")
    total = int(total_labels)
    if total < 0:
        raise ValueError("motion primitive counts must be non-negative")

    normalized: dict[str, int] = {}
    for raw_label, raw_count in counts.items():
        label = str(raw_label)
        _atomic_actions(label)
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral) or raw_count < 0:
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
    retained = {label for label, count in ordered if count >= threshold}
    retained_non_stop_counts = {
        label: count for label, count in ordered if label != "stop" and label in retained
    }
    if any(label != "stop" and count > 0 for label, count in ordered) and not (
        retained_non_stop_counts
    ):
        raise ValueError("no non-stop action meets retention threshold")

    action_labels = [label for label, _ in ordered if label != "stop" and label in retained]
    action_labels.append("stop")
    action_ids = {label: index for index, label in enumerate(action_labels)}

    categories: list[ActionCategory] = []
    for label, raw_count in ordered:
        training_count = raw_count if label == "stop" or label in retained else 0
        categories.append(
            ActionCategory(
                action_id=action_ids.get(label),
                label=label,
                raw_count=raw_count,
                raw_proportion=(raw_count / total if total else 0.0),
                retained=label in retained,
                training_count=training_count,
            )
        )
    return ActionCatalog(
        total_raw_actions=total,
        action_categories=tuple(categories),
    )


def _records_by_id(records: Sequence[EpisodeRecord]) -> dict[int, EpisodeRecord]:
    result = {record.episode_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("episode metadata contains duplicate episode ids")
    return result


def _validated_states(
    episode: EpisodeData,
    expected: Mapping[int, EpisodeRecord],
    seen: set[int],
    *,
    pass_name: str,
) -> np.ndarray:
    episode_id = episode.episode_id
    if episode_id in seen or episode_id not in expected:
        raise ValueError(f"unexpected or duplicate {pass_name} episode {episode_id}")
    seen.add(episode_id)
    record = expected[episode_id]
    if episode.length != record.length:
        raise ValueError(f"episode {episode_id}: state/metadata length mismatch")
    if (episode.task_index, episode.task_name) != (record.task_index, record.task_name):
        raise ValueError(f"episode {episode_id}: task metadata mismatch")
    if STATE_KEY not in episode.observations:
        raise ValueError(f"episode {episode_id} is missing {STATE_KEY!r}")
    try:
        states = np.asarray(episode.observations[STATE_KEY], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id}: observation.state must be finite") from error
    if (
        states.ndim != 2
        or states.shape[0] != record.length
        or states.shape[1] < 8
        or not np.all(np.isfinite(states))
    ):
        raise ValueError(f"episode {episode_id}: observation.state must be finite [time, dim>=8]")
    return states


def _episode_window_visuals(
    cache_root: Path,
    record: EpisodeRecord,
    *,
    embedding_dim: int,
) -> np.ndarray:
    path = cache_root / f"ep{record.episode_id:06d}.npy"
    if not path.is_file():
        raise ValueError(f"episode {record.episode_id}: missing frame embedding cache {path}")
    try:
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        values = np.asarray(cached, dtype=np.float32)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"episode {record.episode_id}: invalid frame embedding cache") from error
    if (
        values.ndim != 2
        or values.shape != (record.length, embedding_dim)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f"episode {record.episode_id}: frame cache length/dimension mismatch or non-finite values"
        )

    window_count = max(record.length - (TRAJECTORY_WINDOW_LENGTH - 1), 0)
    if window_count == 0:
        return np.empty((0, embedding_dim), dtype=np.float32)
    prefix = np.empty((record.length + 1, embedding_dim), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, axis=0, dtype=np.float64, out=prefix[1:])
    means = (prefix[TRAJECTORY_WINDOW_LENGTH:] - prefix[:-TRAJECTORY_WINDOW_LENGTH]) / float(
        TRAJECTORY_WINDOW_LENGTH
    )
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1.0e-8):
        raise ValueError(
            f"episode {record.episode_id}: window visual means must have finite positive norms"
        )
    normalized = means / norms
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"episode {record.episode_id}: invalid KMeans inputs")
    return normalized.astype(np.float32)


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(values * values, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * (values @ centers.T)
    )
    np.maximum(squared, 0.0, out=squared)
    return squared


def _euclidean_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    squared = _squared_distances(values, centers)
    return np.sqrt(squared, out=squared).astype(np.float32, copy=False)


def _initialize_action_cluster_model(
    embeddings: np.ndarray,
    *,
    clusters: int,
    batch_size: int,
    seed: int,
):
    values = np.asarray(embeddings, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.all(np.isfinite(values))
        or isinstance(clusters, bool)
        or not isinstance(clusters, Integral)
        or not 1 <= int(clusters) <= len(values)
    ):
        raise ValueError("invalid KMeans inputs")
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=int(clusters),
        batch_size=batch_size,
        max_iter=1,
        random_state=seed,
        n_init=10,
    )
    model.partial_fit(values)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    if centers.shape != (int(clusters), values.shape[1]) or not np.all(np.isfinite(centers)):
        raise ValueError("invalid KMeans outputs")
    return model


def _episode_exact_memberships(
    states: np.ndarray,
    categories_by_label: Mapping[str, ActionCategory],
    primitive_config: object,
) -> dict[int, np.ndarray]:
    local_rows: dict[int, list[int]] = {}
    for timestep in range(max(len(states) - 7, 0)):
        raw_label = classify_motion_primitive(
            states[timestep], states[timestep + 7], primitive_config
        )
        category = categories_by_label.get(raw_label)
        if category is None:
            raise ValueError(f"missing action category for {raw_label!r}")
        if category.action_id is None or category.training_count <= 0:
            continue
        local_rows.setdefault(int(category.action_id), []).append(timestep)
    return {
        action_id: np.asarray(rows, dtype=np.int64)
        for action_id, rows in local_rows.items()
    }


def build_hierarchical_motion_prototypes(
    adapter: DatasetAdapter,
    clips: Sequence[ClipRecord],
    visual_half_embeddings: np.ndarray,
    *,
    frame_cache_dir: str | Path,
    batch_size: int,
    max_iter: int,
    seed: int,
    max_episodes: int | None,
    num_workers: int,
    timing_callback: TimingCallback | None = None,
) -> HierarchicalPrototypeResult:
    """Learn exact action buckets and assign one nearest visual leaf per clip half."""

    try:
        candidate_values = np.asarray(visual_half_embeddings, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("visual half embeddings must align with clips and be finite") from error
    if (
        candidate_values.ndim != 3
        or candidate_values.shape[0] != len(clips)
        or candidate_values.shape[1] != 2
        or candidate_values.shape[2] == 0
        or not np.all(np.isfinite(candidate_values))
    ):
        raise ValueError("visual half embeddings must align with clips and be finite")
    candidate_norms = np.linalg.norm(candidate_values, axis=2)
    if len(candidate_norms) and not np.allclose(candidate_norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("visual half embeddings must be L2-normalized")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or int(batch_size) <= 0
        or isinstance(max_iter, bool)
        or not isinstance(max_iter, Integral)
        or int(max_iter) <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, Integral)
    ):
        raise ValueError("hierarchical prototype KMeans inputs are invalid")
    if STATE_KEY not in adapter.vector_observation_keys:
        raise ValueError(f"motion_primitives requires vector observation {STATE_KEY!r}")

    action_scan_started = time.perf_counter()
    records = list(adapter.episodes())
    if max_episodes is not None:
        records = records[:max_episodes]
    expected = _records_by_id(records)
    clips_by_episode: dict[int, list[tuple[int, ClipRecord]]] = {}
    sample_ids: set[str] = set()
    for clip_index, clip in enumerate(clips):
        if clip.sample_id in sample_ids:
            raise ValueError(f"duplicate motion primitive clip {clip.sample_id!r}")
        sample_ids.add(clip.sample_id)
        if clip.episode_id not in expected:
            raise ValueError(f"clip {clip.sample_id} has no indexed episode")
        record = expected[clip.episode_id]
        if (
            clip.length != CLIP_LENGTH
            or clip.end_step - clip.start_step + 1 != CLIP_LENGTH
            or clip.start_step < 0
            or clip.end_step >= record.length
        ):
            raise ValueError("motion_primitives requires aligned 15-frame clips")
        clips_by_episode.setdefault(clip.episode_id, []).append((clip_index, clip))

    primitive_config = make_libero_config(threshold=STATE_THRESHOLD)
    raw_counts: Counter[str] = Counter()
    candidate_labels: dict[int, tuple[str, str]] = {}
    seen: set[int] = set()
    for episode in adapter.iter_episodes(
        num_workers=num_workers,
        max_episodes=max_episodes,
        load_images=False,
    ):
        states = _validated_states(episode, expected, seen, pass_name="action")
        window_count = max(len(states) - (TRAJECTORY_WINDOW_LENGTH - 1), 0)
        raw_counts.update(
            classify_motion_primitive(states[t], states[t + 7], primitive_config)
            for t in range(window_count)
        )
        for clip_index, clip in clips_by_episode.get(episode.episode_id, ()):
            first = classify_motion_primitive(
                states[clip.start_step], states[clip.start_step + 7], primitive_config
            )
            second = classify_motion_primitive(
                states[clip.start_step + 7], states[clip.end_step], primitive_config
            )
            candidate_labels[clip_index] = (first, second)
    if seen != set(expected):
        raise ValueError("action pass did not yield every indexed episode exactly once")
    total_windows = sum(raw_counts.values())
    if total_windows == 0:
        raise ValueError("dataset contains no valid trajectory windows")
    if len(candidate_labels) != len(clips):
        raise ValueError("state/clip/cache length mismatch")
    catalog = create_action_catalog(raw_counts, total_windows)

    categories_by_label = {category.label: category for category in catalog.action_categories}
    categories_by_id = {
        int(category.action_id): category
        for category in catalog.action_categories
        if category.action_id is not None
    }
    if sorted(categories_by_id) != list(range(len(categories_by_id))):
        raise ValueError("action ids must be contiguous")
    requested_centers: dict[int, int] = {}
    models: dict[int, object] = {}
    initial_values: dict[int, np.ndarray] = {}
    initial_counts: dict[int, int] = {}
    updated_categories: dict[int, ActionCategory] = {}
    for action_id in sorted(categories_by_id):
        category = categories_by_id[action_id]
        training_count = int(category.training_count)
        if training_count == 0:
            continue
        clusters = cluster_count_for_training_count(training_count)
        capacity = min(
            training_count,
            max(int(batch_size), clusters),
        )
        if clusters <= 0 or clusters > capacity:
            raise ValueError("invalid KMeans inputs")
        requested_centers[action_id] = clusters
        initial_values[action_id] = np.empty(
            (capacity, candidate_values.shape[2]), dtype=np.float32
        )
        initial_counts[action_id] = 0

    if timing_callback is not None:
        timing_callback(
            "graph.prototypes.action_scan",
            time.perf_counter() - action_scan_started,
        )

    def update_action_model(
        action_id: int,
        member_values: np.ndarray,
    ) -> None:
        cursor = 0
        if action_id not in models:
            filled = initial_counts[action_id]
            capacity = len(initial_values[action_id])
            take = min(capacity - filled, len(member_values))
            if take:
                initial_values[action_id][filled : filled + take] = member_values[:take]
                filled += take
                cursor += take
                initial_counts[action_id] = filled
            if filled == capacity:
                models[action_id] = _initialize_action_cluster_model(
                    initial_values[action_id],
                    clusters=requested_centers[action_id],
                    batch_size=int(batch_size),
                    seed=int(seed) + action_id,
                )
        if action_id in models:
            model = models[action_id]
            while cursor < len(member_values):
                end = min(cursor + int(batch_size), len(member_values))
                model.partial_fit(member_values[cursor:end])
                cursor = end

    kmeans_started = time.perf_counter()
    cache_root = Path(frame_cache_dir)
    for epoch in range(int(max_iter)):
        cache_seen: set[int] = set()
        for episode in adapter.iter_episodes(
            num_workers=num_workers,
            max_episodes=max_episodes,
            load_images=False,
        ):
            states = _validated_states(
                episode,
                expected,
                cache_seen,
                pass_name=f"cache epoch {epoch + 1}",
            )
            record = expected[episode.episode_id]
            window_visuals = _episode_window_visuals(
                cache_root,
                record,
                embedding_dim=candidate_values.shape[2],
            )
            if len(window_visuals) != max(len(states) - 7, 0):
                raise ValueError(f"episode {episode.episode_id}: state/cache length mismatch")
            memberships = _episode_exact_memberships(
                states,
                categories_by_label,
                primitive_config,
            )
            for action_id in sorted(memberships):
                rows = memberships[action_id]
                update_action_model(action_id, window_visuals[rows])
        if cache_seen != set(expected):
            raise ValueError("cache pass did not yield every indexed episode exactly once")
        if set(models) != set(requested_centers):
            raise ValueError("invalid KMeans inputs")

    if timing_callback is not None:
        timing_callback("graph.prototypes.kmeans", time.perf_counter() - kmeans_started)

    center_statistics_started = time.perf_counter()
    centers: list[np.ndarray] = []
    leaves: list[LeafPrototype] = []
    leaf_ids_by_action: dict[int, np.ndarray] = {}
    centers_by_action: dict[int, np.ndarray] = {}
    assigned_masses = {
        action_id: np.zeros(requested_centers[action_id], dtype=np.int64)
        for action_id in requested_centers
    }
    nearest_distances = {
        action_id: np.empty(categories_by_id[action_id].training_count, dtype=np.float32)
        for action_id in requested_centers
    }
    distance_cursors = {action_id: 0 for action_id in requested_centers}
    ordering_seen: set[int] = set()
    for episode in adapter.iter_episodes(
        num_workers=num_workers,
        max_episodes=max_episodes,
        load_images=False,
    ):
        states = _validated_states(episode, expected, ordering_seen, pass_name="ordering")
        record = expected[episode.episode_id]
        window_visuals = _episode_window_visuals(
            cache_root,
            record,
            embedding_dim=candidate_values.shape[2],
        )
        memberships = _episode_exact_memberships(
            states,
            categories_by_label,
            primitive_config,
        )
        for action_id, rows in memberships.items():
            action_centers = np.asarray(models[action_id].cluster_centers_, dtype=np.float32)
            distances = _euclidean_distances(window_visuals[rows], action_centers)
            assigned = np.argmin(distances, axis=1)
            assigned_masses[action_id] += np.bincount(
                assigned,
                minlength=requested_centers[action_id],
            )
            closest = distances[np.arange(len(rows)), assigned]
            start = distance_cursors[action_id]
            end = start + len(closest)
            nearest_distances[action_id][start:end] = closest
            distance_cursors[action_id] = end
    if ordering_seen != set(expected):
        raise ValueError("ordering pass did not yield every indexed episode exactly once")
    if any(
        distance_cursors[action_id] != len(nearest_distances[action_id])
        for action_id in nearest_distances
    ):
        raise ValueError("exact action membership counts do not match the catalog")

    for action_id in sorted(categories_by_id):
        category = categories_by_id[action_id]
        if action_id not in models:
            continue
        unordered_centers = np.asarray(models[action_id].cluster_centers_, dtype=np.float32)
        order = sorted(
            range(len(unordered_centers)),
            key=lambda index: (
                -float(assigned_masses[action_id][index]),
                tuple(float(coordinate) for coordinate in unordered_centers[index]),
            ),
        )
        action_centers = unordered_centers[np.asarray(order, dtype=np.int64)]
        action_leaf_ids: list[int] = []
        for center_id, center in enumerate(action_centers):
            prototype_id = len(leaves)
            leaves.append(
                LeafPrototype(
                    prototype_id=prototype_id,
                    label=f"{category.label}::center_{center_id}",
                    action_id=action_id,
                    action_label=category.label,
                    center_id=center_id,
                )
            )
            centers.append(center)
            action_leaf_ids.append(prototype_id)
        leaf_ids_by_action[action_id] = np.asarray(action_leaf_ids, dtype=np.int32)
        centers_by_action[action_id] = action_centers
        lower, upper = nearest_distance_bounds(nearest_distances[action_id])
        updated_categories[action_id] = replace(
            category,
            requested_centers=requested_centers[action_id],
            actual_centers=len(action_centers),
            nearest_distance_q10=lower,
            nearest_distance_q90=upper,
        )

    refined_catalog = ActionCatalog(
        total_raw_actions=catalog.total_raw_actions,
        action_categories=tuple(
            updated_categories.get(int(category.action_id), category)
            if category.action_id is not None
            else category
            for category in catalog.action_categories
        ),
        leaf_prototypes=tuple(leaves),
    )
    if timing_callback is not None:
        timing_callback(
            "graph.prototypes.center_statistics",
            time.perf_counter() - center_statistics_started,
        )

    candidate_assignment_started = time.perf_counter()
    retained_counts = {
        category.label: category.raw_count
        for category in refined_catalog.action_categories
        if category.retained and category.label != "stop"
    }
    refined_by_label = {
        category.label: category for category in refined_catalog.action_categories
    }
    per_clip: list[tuple[tuple[int, float], ...]] = []
    for clip_index in range(len(clips)):
        half_assignments: list[tuple[int, float]] = []
        for half_index, raw_label in enumerate(candidate_labels[clip_index]):
            parent_labels = maximum_retained_parents(raw_label, retained_counts)
            nearest: tuple[float, int, str] | None = None
            for parent_label in parent_labels:
                parent_category = refined_by_label.get(parent_label)
                if parent_category is None or parent_category.action_id is None:
                    raise ValueError(f"missing parent action {parent_label!r} for clip {clip_index}")
                action_id = int(parent_category.action_id)
                if action_id not in centers_by_action or action_id not in leaf_ids_by_action:
                    raise ValueError(
                        f"missing visual centers for parent action {parent_label!r}"
                    )
                distances = _euclidean_distances(
                    candidate_values[clip_index, half_index][None, :],
                    centers_by_action[action_id],
                )[0]
                for local_index, distance in enumerate(distances):
                    leaf_id = int(leaf_ids_by_action[action_id][local_index])
                    candidate = (float(distance), leaf_id, parent_label)
                    if nearest is None or candidate[:2] < nearest[:2]:
                        nearest = candidate
            if nearest is None:
                raise ValueError(f"clip {clip_index} half {half_index} has no leaf assignment")
            distance, leaf_id, parent_label = nearest
            parent_category = refined_by_label[parent_label]
            assert parent_category.nearest_distance_q10 is not None
            assert parent_category.nearest_distance_q90 is not None
            weight = retention_weight(raw_label, parent_label) * distance_confidence(
                distance,
                parent_category.nearest_distance_q10,
                parent_category.nearest_distance_q90,
            )
            half_assignments.append((leaf_id, weight))
        per_clip.append(merge_half_leaf_assignments(tuple(half_assignments)))

    prototype_indices = np.full((len(clips), 2), -1, dtype=np.int32)
    prototype_weights = np.zeros((len(clips), 2), dtype=np.float32)
    for clip_index, assignments in enumerate(per_clip):
        for slot, (leaf_id, weight) in enumerate(assignments):
            prototype_indices[clip_index, slot] = leaf_id
            prototype_weights[clip_index, slot] = np.float32(weight)
    if len(clips) and (
        np.any((prototype_indices < 0) != (prototype_weights == 0.0))
        or np.any(prototype_weights < 0.0)
        or np.any(prototype_weights > 1.5)
        or not np.all(np.isfinite(prototype_weights))
    ):
        raise ValueError("invalid hard-nearest prototype output")

    result = HierarchicalPrototypeResult(
        prototypes=PrototypeData(
            centers=np.stack(centers).astype(np.float32),
            indices=prototype_indices,
            weights=prototype_weights,
            labels=refined_catalog.labels,
        ),
        catalog=refined_catalog,
        half_action_labels=np.asarray(
            [candidate_labels[index] for index in range(len(clips))], dtype=np.str_
        ),
    )
    if timing_callback is not None:
        timing_callback(
            "graph.prototypes.candidate_assignment",
            time.perf_counter() - candidate_assignment_started,
        )
    return result
