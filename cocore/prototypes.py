"""Hard action buckets and nearest visual-leaf assignments for Cocore."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import math
import time
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import TypeVar
import warnings

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from libero_motion_primitives import classify_motion_primitive, make_libero_config
from relcore.graph.prototypes import PrototypeData
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord

from cocore.timing import TimingCallback


_ActionResult = TypeVar("_ActionResult")


MIN_ACTION_COUNT = 400
MIN_ACTION_FREQUENCY = 0.005
MIN_VISUAL_CENTERS = 10
MAX_VISUAL_CENTERS = 30
FULL_KMEANS_MAX_TRAINING_COUNT = 65_536
FULL_KMEANS_OPENMP_THREADS = 1
MINIBATCH_KMEANS_OPENMP_THREADS = 4
MIN_DISTANCE_WEIGHT = 0.3
STATE_THRESHOLD = 0.03
STATE_KEY = "observation.state"
TRAJECTORY_WINDOW_LENGTH = 8
TRAJECTORY_WINDOW_POLICY = "full_coverage_max_gap_3_tail_rebalanced"
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
    use_stop_bucket: bool = True
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
            "schema_version": 9,
            "use_stop_bucket": self.use_stop_bucket,
            "strategy": (
                "trajectory_sampled_optional_stop_retained_action_then_cropped_pca_"
                "half_visual_hybrid_kmeans_nearest"
            ),
            "constants": {
                "state_threshold": STATE_THRESHOLD,
                "min_action_count": MIN_ACTION_COUNT,
                "min_action_frequency": MIN_ACTION_FREQUENCY,
                "max_visual_centers": MAX_VISUAL_CENTERS,
                "full_kmeans_max_training_count": FULL_KMEANS_MAX_TRAINING_COUNT,
                "full_kmeans_openmp_threads": FULL_KMEANS_OPENMP_THREADS,
                "minibatch_kmeans_openmp_threads": MINIBATCH_KMEANS_OPENMP_THREADS,
                "kmeans_n_init": 1,
                "large_bucket_parallelism": "serial",
                "trajectory_window_length": TRAJECTORY_WINDOW_LENGTH,
                "trajectory_window_policy": TRAJECTORY_WINDOW_POLICY,
                "visual_half_windows": [[0, 8], [7, 15]],
                "visual_projection": ("frame @ visual_pca.components[:, :frame_embedding_dim].T"),
                "visual_projection_centering": "none",
                "visual_projection_padding": "right_zero_to_128",
                "visual_half_encoding": "l2_normalized_mean_of_eight_projected_frames",
                "cluster_count": (
                    "min(training_count, min(30, max(10, "
                    "floor(4 * log2(training_count) - 30))))"
                ),
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
    eligible_mask: np.ndarray


@dataclass(frozen=True)
class _ActionTrainingData:
    action_id: int
    values: np.ndarray
    episode_ranges: tuple[tuple[int, int], ...]


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
    *,
    use_stop_bucket: bool = True,
) -> tuple[str, ...]:
    """Return every maximum-cardinality retained atomic subset in stable order."""

    if not isinstance(use_stop_bucket, bool):
        raise ValueError("use_stop_bucket must be a boolean")
    atomic = frozenset(_atomic_actions(label))
    candidates: list[tuple[str, int, int]] = []
    for parent_label, raw_count in retained_counts.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral) or raw_count <= 0:
            raise ValueError("retained action counts must be positive integers")
        parent_atomic = frozenset(_atomic_actions(parent_label))
        if parent_atomic and parent_atomic.issubset(atomic):
            candidates.append((parent_label, int(raw_count), len(parent_atomic)))
    if not candidates:
        return ("stop",) if use_stop_bucket else ()
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
    """Return the capped logarithmic visual-center count for one action bucket."""

    if isinstance(training_count, bool) or not isinstance(training_count, Real):
        raise ValueError("training count must be a finite positive number")
    value = float(training_count)
    if not math.isfinite(value) or value < 1.0:
        raise ValueError("training count must be a finite positive number")
    clusters = min(
        MAX_VISUAL_CENTERS,
        max(MIN_VISUAL_CENTERS, math.floor(4.0 * math.log2(value) - 30.0)),
    )
    return min(int(value), clusters)


def trajectory_window_starts(trajectory_length: int) -> tuple[int, ...]:
    """Return full-coverage eight-frame starts with gaps of at most three."""

    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, Integral):
        raise ValueError("trajectory length must be a non-negative integer")
    length = int(trajectory_length)
    if length < 0:
        raise ValueError("trajectory length must be a non-negative integer")
    last_start = length - TRAJECTORY_WINDOW_LENGTH
    if last_start < 0:
        return ()
    if last_start == 0:
        return (0,)

    full_gaps, remainder = divmod(last_start, 3)
    if remainder == 0:
        gaps = [3] * full_gaps
    elif remainder == 2:
        gaps = [3] * full_gaps + [2]
    elif last_start == 1:
        gaps = [1]
    else:
        gaps = [3] * (full_gaps - 1) + [2, 2]

    starts = [0]
    for gap in gaps:
        starts.append(starts[-1] + gap)
    return tuple(starts)


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
    quantized = tuple((leaf_id, float(np.float32(weight))) for leaf_id, weight in merged.items())
    if any(not math.isfinite(weight) for _, weight in quantized):
        raise ValueError("half-clip assignment weights must fit float32")
    return tuple(sorted(quantized, key=lambda item: (-item[1], item[0])))


def create_action_catalog(
    counts: Mapping[str, int],
    total_labels: int,
    *,
    use_stop_bucket: bool = True,
) -> ActionCatalog:
    """Build a deterministic schema-9 action catalog from raw action counts."""

    if not isinstance(use_stop_bucket, bool):
        raise ValueError("use_stop_bucket must be a boolean")
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
    if use_stop_bucket:
        action_labels.append("stop")
    action_ids = {label: index for index, label in enumerate(action_labels)}

    categories: list[ActionCategory] = []
    for label, raw_count in ordered:
        training_count = raw_count if (
            (label == "stop" and use_stop_bucket)
            or (label != "stop" and label in retained)
        ) else 0
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
        use_stop_bucket=use_stop_bucket,
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


def _project_clustering_features(
    values: np.ndarray,
    components: np.ndarray,
    *,
    output_dim: int,
) -> np.ndarray:
    """Project raw frame features with the sum block of fragment PCA components."""

    try:
        features = np.asarray(values, dtype=np.float32)
        matrix = np.asarray(components, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("clustering projection inputs must be finite numeric arrays") from error
    if features.ndim == 0 or features.shape[-1] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("clustering projection features must be finite with a positive dimension")
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("clustering PCA components must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("clustering PCA components must be finite")
    frame_dim = int(features.shape[-1])
    if matrix.shape[1] != 2 * frame_dim:
        raise ValueError("clustering PCA component width must be twice the frame dimension")
    if isinstance(output_dim, bool) or not isinstance(output_dim, Integral) or output_dim <= 0:
        raise ValueError("clustering projection output dimension must be a positive integer")
    if matrix.shape[0] > int(output_dim):
        raise ValueError("clustering PCA component count cannot exceed output dimension")

    projected = features @ matrix[:, :frame_dim].T
    if projected.shape[-1] < int(output_dim):
        padding = [(0, 0)] * projected.ndim
        padding[-1] = (0, int(output_dim) - projected.shape[-1])
        projected = np.pad(projected, padding)
    projected = np.asarray(projected, dtype=np.float32)
    if not np.all(np.isfinite(projected)):
        raise ValueError("clustering projection produced non-finite values")
    return projected


def _episode_window_visuals(
    cache_root: Path,
    record: EpisodeRecord,
    *,
    pca_components: np.ndarray,
    visual_dim: int,
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
        or values.shape[0] != record.length
        or values.shape[1] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f"episode {record.episode_id}: frame cache length/dimension mismatch or non-finite values"
        )
    projected = _project_clustering_features(
        values,
        pca_components,
        output_dim=visual_dim,
    )

    window_starts = np.asarray(trajectory_window_starts(record.length), dtype=np.int64)
    if len(window_starts) == 0:
        return np.empty((0, int(visual_dim)), dtype=np.float32)
    prefix = np.empty((record.length + 1, int(visual_dim)), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(projected, axis=0, dtype=np.float64, out=prefix[1:])
    means = (
        prefix[window_starts + TRAJECTORY_WINDOW_LENGTH] - prefix[window_starts]
    ) / float(TRAJECTORY_WINDOW_LENGTH)
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


def _episode_exact_memberships(
    states: np.ndarray,
    categories_by_label: Mapping[str, ActionCategory],
    primitive_config: object,
) -> dict[int, np.ndarray]:
    local_rows: dict[int, list[int]] = {}
    for row, timestep in enumerate(trajectory_window_starts(len(states))):
        raw_label = classify_motion_primitive(
            states[timestep],
            states[timestep + TRAJECTORY_WINDOW_LENGTH - 1],
            primitive_config,
        )
        category = categories_by_label.get(raw_label)
        if category is None:
            raise ValueError(f"missing action category for {raw_label!r}")
        if category.action_id is None or category.training_count <= 0:
            continue
        local_rows.setdefault(int(category.action_id), []).append(row)
    return {action_id: np.asarray(rows, dtype=np.int64) for action_id, rows in local_rows.items()}


def _materialize_action_training_data(
    adapter: DatasetAdapter,
    expected: Mapping[int, EpisodeRecord],
    categories_by_label: Mapping[str, ActionCategory],
    categories_by_id: Mapping[int, ActionCategory],
    primitive_config: object,
    *,
    cache_root: Path,
    pca_components: np.ndarray,
    visual_dim: int,
    max_episodes: int | None,
    num_workers: int,
) -> dict[int, _ActionTrainingData]:
    values = {
        action_id: np.empty((category.training_count, int(visual_dim)), dtype=np.float32)
        for action_id, category in categories_by_id.items()
        if category.training_count > 0
    }
    cursors = {action_id: 0 for action_id in values}
    episode_ranges: dict[int, list[tuple[int, int]]] = {
        action_id: [] for action_id in values
    }
    seen: set[int] = set()
    for episode in adapter.iter_episodes(
        num_workers=num_workers,
        max_episodes=max_episodes,
        load_images=False,
    ):
        states = _validated_states(episode, expected, seen, pass_name="training data")
        record = expected[episode.episode_id]
        window_visuals = _episode_window_visuals(
            cache_root,
            record,
            pca_components=pca_components,
            visual_dim=visual_dim,
        )
        if len(window_visuals) != len(trajectory_window_starts(len(states))):
            raise ValueError(f"episode {episode.episode_id}: state/cache length mismatch")
        memberships = _episode_exact_memberships(
            states,
            categories_by_label,
            primitive_config,
        )
        for action_id in sorted(memberships):
            rows = memberships[action_id]
            start = cursors[action_id]
            end = start + len(rows)
            if end > len(values[action_id]):
                raise ValueError("exact action membership counts do not match the catalog")
            values[action_id][start:end] = window_visuals[rows]
            episode_ranges[action_id].append((start, end))
            cursors[action_id] = end
    if seen != set(expected):
        raise ValueError("training data pass did not yield every indexed episode exactly once")
    if any(cursors[action_id] != len(action_values) for action_id, action_values in values.items()):
        raise ValueError("exact action membership counts do not match the catalog")
    return {
        action_id: _ActionTrainingData(
            action_id=action_id,
            values=action_values,
            episode_ranges=tuple(episode_ranges[action_id]),
        )
        for action_id, action_values in values.items()
    }


def _fit_action_model(
    training_data: _ActionTrainingData,
    *,
    clusters: int,
    batch_size: int,
    max_iter: int,
    tol: float,
    seed: int,
):
    values = np.asarray(training_data.values, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.all(np.isfinite(values))
        or not 1 <= int(clusters) <= len(values)
    ):
        raise ValueError("invalid KMeans inputs")
    common = {
        "n_clusters": int(clusters),
        "init": "k-means++",
        "n_init": 1,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": int(seed) + training_data.action_id,
    }
    if len(values) > FULL_KMEANS_MAX_TRAINING_COUNT:
        model = MiniBatchKMeans(
            **common,
            batch_size=int(batch_size),
            max_no_improvement=10,
        )
    else:
        model = KMeans(
            **common,
            algorithm="lloyd",
            copy_x=True,
        )
    model.fit(values)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    if centers.shape != (int(clusters), values.shape[1]) or not np.all(np.isfinite(centers)):
        raise ValueError("invalid KMeans outputs")
    return model


def _compute_action_center_statistics(
    training_data: _ActionTrainingData,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assigned_masses = np.zeros(len(centers), dtype=np.int64)
    nearest_distances = np.empty(len(training_data.values), dtype=np.float32)
    for start, end in training_data.episode_ranges:
        distances = _euclidean_distances(training_data.values[start:end], centers)
        assigned = np.argmin(distances, axis=1)
        assigned_masses += np.bincount(assigned, minlength=len(centers))
        nearest_distances[start:end] = distances[np.arange(end - start), assigned]
    return assigned_masses, nearest_distances


def _run_action_tasks(
    executor: ThreadPoolExecutor | None,
    action_ids: Sequence[int],
    operation: Callable[[int], _ActionResult],
) -> dict[int, _ActionResult]:
    if executor is None:
        return {action_id: operation(action_id) for action_id in action_ids}
    futures = {
        action_id: executor.submit(operation, action_id)
        for action_id in action_ids
    }
    try:
        return {
            action_id: futures[action_id].result()
            for action_id in action_ids
        }
    except BaseException:
        for future in futures.values():
            future.cancel()
        raise


def _fit_action_models(
    executor: ThreadPoolExecutor | None,
    training_data: Mapping[int, _ActionTrainingData],
    requested_centers: Mapping[int, int],
    *,
    batch_size: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> dict[int, object]:
    action_ids = sorted(requested_centers)
    small_action_ids = [
        action_id
        for action_id in action_ids
        if len(training_data[action_id].values) <= FULL_KMEANS_MAX_TRAINING_COUNT
    ]
    large_action_ids = [
        action_id
        for action_id in action_ids
        if len(training_data[action_id].values) > FULL_KMEANS_MAX_TRAINING_COUNT
    ]

    def fit_action(action_id: int):
        return _fit_action_model(
            training_data[action_id],
            clusters=requested_centers[action_id],
            batch_size=int(batch_size),
            max_iter=int(max_iter),
            tol=float(tol),
            seed=int(seed),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with threadpool_limits(limits=FULL_KMEANS_OPENMP_THREADS, user_api="openmp"):
            models = _run_action_tasks(executor, small_action_ids, fit_action)
        with threadpool_limits(limits=MINIBATCH_KMEANS_OPENMP_THREADS, user_api="openmp"):
            for action_id in large_action_ids:
                models[action_id] = fit_action(action_id)
    return {action_id: models[action_id] for action_id in action_ids}


def build_hierarchical_motion_prototypes(
    adapter: DatasetAdapter,
    clips: Sequence[ClipRecord],
    visual_half_embeddings: np.ndarray,
    *,
    pca_components: np.ndarray,
    visual_dim: int,
    frame_cache_dir: str | Path,
    batch_size: int,
    max_iter: int,
    seed: int,
    max_episodes: int | None,
    num_workers: int,
    tol: float = 1.0e-4,
    num_threads: int = 4,
    use_stop_bucket: bool = True,
    timing_callback: TimingCallback | None = None,
) -> HierarchicalPrototypeResult:
    """Learn exact action buckets and assign one nearest visual leaf per clip half."""

    try:
        raw_candidate_values = np.asarray(visual_half_embeddings, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("visual half embeddings must align with clips and be finite") from error
    if not isinstance(use_stop_bucket, bool):
        raise ValueError("hierarchical prototype use_stop_bucket must be a boolean")
    if (
        raw_candidate_values.ndim != 3
        or raw_candidate_values.shape[0] != len(clips)
        or raw_candidate_values.shape[1] != 2
        or raw_candidate_values.shape[2] == 0
        or not np.all(np.isfinite(raw_candidate_values))
    ):
        raise ValueError("visual half embeddings must align with clips and be finite")
    candidate_norms = np.linalg.norm(raw_candidate_values, axis=2)
    if len(candidate_norms) and not np.allclose(candidate_norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("visual half embeddings must be L2-normalized")
    candidate_values = _project_clustering_features(
        raw_candidate_values,
        pca_components,
        output_dim=visual_dim,
    )
    projected_norms = np.linalg.norm(candidate_values, axis=2, keepdims=True)
    if len(projected_norms) and (
        not np.all(np.isfinite(projected_norms)) or np.any(projected_norms <= 1.0e-8)
    ):
        raise ValueError("projected visual half embeddings must have finite positive norms")
    candidate_values = (candidate_values / np.maximum(projected_norms, 1.0e-8)).astype(np.float32)
    if (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, Integral)
        or int(num_threads) <= 0
    ):
        raise ValueError("hierarchical prototype thread count must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or int(batch_size) <= 0
        or isinstance(max_iter, bool)
        or not isinstance(max_iter, Integral)
        or int(max_iter) <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or isinstance(tol, bool)
        or not isinstance(tol, Real)
        or not math.isfinite(float(tol))
        or float(tol) <= 0.0
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
        window_starts = trajectory_window_starts(len(states))
        raw_counts.update(
            classify_motion_primitive(
                states[t],
                states[t + TRAJECTORY_WINDOW_LENGTH - 1],
                primitive_config,
            )
            for t in window_starts
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
    catalog = create_action_catalog(
        raw_counts,
        total_windows,
        use_stop_bucket=use_stop_bucket,
    )

    categories_by_label = {category.label: category for category in catalog.action_categories}
    categories_by_id = {
        int(category.action_id): category
        for category in catalog.action_categories
        if category.action_id is not None
    }
    if sorted(categories_by_id) != list(range(len(categories_by_id))):
        raise ValueError("action ids must be contiguous")
    if not categories_by_id:
        raise ValueError("no enabled action bucket remains")
    requested_centers: dict[int, int] = {}
    updated_categories: dict[int, ActionCategory] = {}
    for action_id in sorted(categories_by_id):
        category = categories_by_id[action_id]
        training_count = int(category.training_count)
        if training_count == 0:
            continue
        clusters = cluster_count_for_training_count(training_count)
        if clusters <= 0 or clusters > training_count:
            raise ValueError("invalid KMeans inputs")
        requested_centers[action_id] = clusters

    if timing_callback is not None:
        timing_callback(
            "graph.prototypes.action_scan",
            time.perf_counter() - action_scan_started,
        )

    training_data_started = time.perf_counter()
    cache_root = Path(frame_cache_dir)
    training_data = _materialize_action_training_data(
        adapter,
        expected,
        categories_by_label,
        categories_by_id,
        primitive_config,
        cache_root=cache_root,
        pca_components=pca_components,
        visual_dim=visual_dim,
        max_episodes=max_episodes,
        num_workers=num_workers,
    )
    if timing_callback is not None:
        timing_callback(
            "graph.prototypes.training_data",
            time.perf_counter() - training_data_started,
        )

    action_ids = sorted(requested_centers)
    effective_threads = min(int(num_threads), len(action_ids))
    executor = (
        ThreadPoolExecutor(
            max_workers=effective_threads,
            thread_name_prefix="cocore-prototypes",
        )
        if effective_threads > 1
        else None
    )
    try:
        kmeans_started = time.perf_counter()
        models = _fit_action_models(
            executor,
            training_data,
            requested_centers,
            batch_size=int(batch_size),
            max_iter=int(max_iter),
            tol=float(tol),
            seed=int(seed),
        )

        if timing_callback is not None:
            timing_callback("graph.prototypes.kmeans", time.perf_counter() - kmeans_started)

        center_statistics_started = time.perf_counter()

        def compute_statistics(action_id: int) -> tuple[np.ndarray, np.ndarray]:
            return _compute_action_center_statistics(
                training_data[action_id],
                np.asarray(models[action_id].cluster_centers_, dtype=np.float32),
            )

        statistics = _run_action_tasks(executor, action_ids, compute_statistics)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    training_data.clear()
    if timing_callback is not None:
        timing_callback(
            "graph.prototypes.center_statistics",
            time.perf_counter() - center_statistics_started,
        )

    centers: list[np.ndarray] = []
    leaves: list[LeafPrototype] = []
    leaf_ids_by_action: dict[int, np.ndarray] = {}
    centers_by_action: dict[int, np.ndarray] = {}
    assigned_masses = {action_id: value[0] for action_id, value in statistics.items()}
    nearest_distances = {action_id: value[1] for action_id, value in statistics.items()}

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
        use_stop_bucket=use_stop_bucket,
        leaf_prototypes=tuple(leaves),
    )
    statistics.clear()
    assigned_masses.clear()
    nearest_distances.clear()

    candidate_assignment_started = time.perf_counter()
    retained_counts = {
        category.label: category.raw_count
        for category in refined_catalog.action_categories
        if category.retained and category.label != "stop"
    }
    refined_by_label = {category.label: category for category in refined_catalog.action_categories}
    per_clip: list[tuple[tuple[int, float], ...]] = []
    for clip_index in range(len(clips)):
        half_assignments: list[tuple[int, float]] = []
        for half_index, raw_label in enumerate(candidate_labels[clip_index]):
            parent_labels = maximum_retained_parents(
                raw_label,
                retained_counts,
                use_stop_bucket=use_stop_bucket,
            )
            if not parent_labels:
                continue
            nearest: tuple[float, int, str] | None = None
            for parent_label in parent_labels:
                parent_category = refined_by_label.get(parent_label)
                if parent_category is None or parent_category.action_id is None:
                    raise ValueError(
                        f"missing parent action {parent_label!r} for clip {clip_index}"
                    )
                action_id = int(parent_category.action_id)
                if action_id not in centers_by_action or action_id not in leaf_ids_by_action:
                    raise ValueError(f"missing visual centers for parent action {parent_label!r}")
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
        per_clip.append(
            merge_half_leaf_assignments(tuple(half_assignments))
            if half_assignments
            else ()
        )

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
        eligible_mask=np.any(prototype_indices >= 0, axis=1),
    )
    if timing_callback is not None:
        timing_callback(
            "graph.prototypes.candidate_assignment",
            time.perf_counter() - candidate_assignment_started,
        )
    return result
