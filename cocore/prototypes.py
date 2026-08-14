"""Pure action-distribution and visual-probability primitives for Cocore."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from libero_motion_primitives import classify_motion_primitive, make_libero_config
from relcore.graph.prototypes import PrototypeData
from relcore.schemas import ClipRecord
from trajectory_data import DatasetAdapter, EpisodeData, EpisodeRecord


MIN_ACTION_COUNT = 40
MIN_ACTION_FREQUENCY = 0.005
MAX_VISUAL_CENTERS = 16
VISUAL_SOFTMAX_TEMPERATURE = 0.1
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
                "cluster_count": ("min(16, 1 + floor(log2(effective_mass)))"),
            },
            "total_raw_actions": self.total_raw_actions,
            "action_categories": categories,
            "leaf_prototypes": [asdict(leaf) for leaf in self.leaf_prototypes],
        }


@dataclass(frozen=True)
class HierarchicalPrototypeResult:
    prototypes: PrototypeData
    catalog: ActionCatalog
    clip_action_labels: np.ndarray


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
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral) or raw_count <= 0:
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
    return tuple((parent_label, raw_count / total) for parent_label, raw_count in parents)


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
        raise ValueError("visual values and centers must be finite non-empty matrices") from error
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


def _records_by_id(records: Sequence[EpisodeRecord]) -> dict[int, EpisodeRecord]:
    result = {record.episode_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("episode metadata contains duplicate episode ids")
    return result


def _action_bucket_statistics(
    catalog: ActionCatalog,
) -> tuple[dict[int, int], dict[int, float]]:
    action_ids = {
        int(category.action_id)
        for category in catalog.action_categories
        if category.action_id is not None
    }
    member_counts = {action_id: 0 for action_id in action_ids}
    mass_terms: dict[int, list[float]] = {action_id: [] for action_id in action_ids}
    for category in catalog.action_categories:
        for parent in category.parents:
            if parent.action_id not in action_ids:
                raise ValueError(f"missing parent assignments for action {category.label!r}")
            member_counts[parent.action_id] += category.raw_count
            mass_terms[parent.action_id].append(category.raw_count * parent.probability)
    return member_counts, {action_id: math.fsum(mass_terms[action_id]) for action_id in action_ids}


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


def _fit_action_cluster_model(
    embeddings: np.ndarray,
    sample_weights: np.ndarray,
    *,
    clusters: int,
    batch_size: int,
    max_iter: int,
    seed: int,
):
    values = np.asarray(embeddings, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or weights.ndim != 1
        or len(weights) != len(values)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or isinstance(clusters, bool)
        or not isinstance(clusters, Integral)
        or not 1 <= int(clusters) <= len(values)
    ):
        raise ValueError("invalid KMeans inputs")
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=int(clusters),
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        n_init=10,
    )
    model.fit(values, sample_weight=weights)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    if centers.shape != (int(clusters), values.shape[1]) or not np.all(np.isfinite(centers)):
        raise ValueError("invalid KMeans outputs")
    return model


def _episode_parent_memberships(
    states: np.ndarray,
    categories_by_label: Mapping[str, ActionCategory],
    action_ids: Sequence[int],
    primitive_config: object,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    local_rows: dict[int, list[int]] = {action_id: [] for action_id in action_ids}
    local_weights: dict[int, list[float]] = {action_id: [] for action_id in action_ids}
    for timestep in range(max(len(states) - 7, 0)):
        raw_label = classify_motion_primitive(
            states[timestep], states[timestep + 7], primitive_config
        )
        category = categories_by_label.get(raw_label)
        if category is None or not category.parents:
            raise ValueError(f"missing parent assignments for action {raw_label!r}")
        probability_sum = sum(parent.probability for parent in category.parents)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"non-normalized parent assignments for action {raw_label!r}")
        for parent in category.parents:
            if (
                parent.action_id not in local_rows
                or not math.isfinite(parent.probability)
                or parent.probability <= 0.0
            ):
                raise ValueError(f"missing parent assignments for action {raw_label!r}")
            local_rows[parent.action_id].append(timestep)
            local_weights[parent.action_id].append(parent.probability)
    return {
        action_id: (
            np.asarray(local_rows[action_id], dtype=np.int64),
            np.asarray(local_weights[action_id], dtype=np.float64),
        )
        for action_id in action_ids
        if local_rows[action_id]
    }


def build_hierarchical_motion_prototypes(
    adapter: DatasetAdapter,
    clips: Sequence[ClipRecord],
    visual_clip_embeddings: np.ndarray,
    *,
    frame_cache_dir: str | Path,
    batch_size: int,
    max_iter: int,
    seed: int,
    max_episodes: int | None,
    num_workers: int,
) -> HierarchicalPrototypeResult:
    """Learn weighted action/visual leaves from all stride-one trajectory windows."""

    try:
        candidate_values = np.asarray(visual_clip_embeddings, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("visual clip embeddings must align with clips and be finite") from error
    if (
        candidate_values.ndim != 2
        or candidate_values.shape[0] != len(clips)
        or candidate_values.shape[1] == 0
        or not np.all(np.isfinite(candidate_values))
    ):
        raise ValueError("visual clip embeddings must align with clips and be finite")
    candidate_norms = np.linalg.norm(candidate_values, axis=1)
    if len(candidate_norms) and not np.allclose(candidate_norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("visual clip embeddings must be L2-normalized")
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
    candidate_labels: dict[int, str] = {}
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
            candidate_labels[clip_index] = canonical_clip_action(first, second)
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
    member_counts, effective_mass = _action_bucket_statistics(catalog)

    requested_centers: dict[int, int] = {}
    models: dict[int, object] = {}
    initial_values: dict[int, np.ndarray] = {}
    initial_weights: dict[int, np.ndarray] = {}
    initial_counts: dict[int, int] = {}
    updated_categories: dict[int, ActionCategory] = {}
    for action_id in sorted(categories_by_id):
        category = categories_by_id[action_id]
        mass = float(effective_mass[action_id])
        if member_counts[action_id] == 0:
            updated_categories[action_id] = replace(category, effective_mass=0.0)
            continue
        clusters = cluster_count_for_mass(mass)
        capacity = min(
            member_counts[action_id],
            max(int(batch_size), clusters),
        )
        if clusters <= 0 or clusters > capacity:
            raise ValueError("invalid KMeans inputs")
        requested_centers[action_id] = clusters
        initial_values[action_id] = np.empty(
            (capacity, candidate_values.shape[1]), dtype=np.float32
        )
        initial_weights[action_id] = np.empty(capacity, dtype=np.float64)
        initial_counts[action_id] = 0

    def update_action_model(
        action_id: int,
        member_values: np.ndarray,
        member_weights: np.ndarray,
    ) -> None:
        cursor = 0
        if action_id not in models:
            filled = initial_counts[action_id]
            capacity = len(initial_weights[action_id])
            take = min(capacity - filled, len(member_values))
            if take:
                initial_values[action_id][filled : filled + take] = member_values[:take]
                initial_weights[action_id][filled : filled + take] = member_weights[:take]
                filled += take
                cursor += take
                initial_counts[action_id] = filled
            if filled == capacity:
                models[action_id] = _fit_action_cluster_model(
                    initial_values[action_id],
                    initial_weights[action_id],
                    clusters=requested_centers[action_id],
                    batch_size=int(batch_size),
                    max_iter=int(max_iter),
                    seed=int(seed) + action_id,
                )
        if action_id in models:
            model = models[action_id]
            while cursor < len(member_values):
                end = min(cursor + int(batch_size), len(member_values))
                model.partial_fit(
                    member_values[cursor:end],
                    sample_weight=member_weights[cursor:end],
                )
                cursor = end

    cache_root = Path(frame_cache_dir)
    cache_seen: set[int] = set()
    for episode in adapter.iter_episodes(
        num_workers=num_workers,
        max_episodes=max_episodes,
        load_images=False,
    ):
        states = _validated_states(episode, expected, cache_seen, pass_name="cache")
        record = expected[episode.episode_id]
        window_visuals = _episode_window_visuals(
            cache_root,
            record,
            embedding_dim=candidate_values.shape[1],
        )
        if len(window_visuals) != max(len(states) - 7, 0):
            raise ValueError(f"episode {episode.episode_id}: state/cache length mismatch")
        memberships = _episode_parent_memberships(
            states,
            categories_by_label,
            tuple(categories_by_id),
            primitive_config,
        )
        for action_id, (rows, weights) in memberships.items():
            update_action_model(action_id, window_visuals[rows], weights)
    if cache_seen != set(expected):
        raise ValueError("cache pass did not yield every indexed episode exactly once")
    if set(models) != set(requested_centers):
        raise ValueError("invalid KMeans inputs")

    centers: list[np.ndarray] = []
    leaves: list[LeafPrototype] = []
    leaf_ids_by_action: dict[int, np.ndarray] = {}
    centers_by_action: dict[int, np.ndarray] = {}
    assigned_masses = {
        action_id: np.zeros(requested_centers[action_id], dtype=np.float64)
        for action_id in requested_centers
    }
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
            embedding_dim=candidate_values.shape[1],
        )
        memberships = _episode_parent_memberships(
            states,
            categories_by_label,
            tuple(categories_by_id),
            primitive_config,
        )
        for action_id, (rows, weights) in memberships.items():
            action_centers = np.asarray(models[action_id].cluster_centers_, dtype=np.float32)
            assigned = np.argmin(_squared_distances(window_visuals[rows], action_centers), axis=1)
            assigned_masses[action_id] += np.bincount(
                assigned,
                weights=weights,
                minlength=requested_centers[action_id],
            )
    if ordering_seen != set(expected):
        raise ValueError("ordering pass did not yield every indexed episode exactly once")

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
        updated_categories[action_id] = replace(
            category,
            effective_mass=float(effective_mass[action_id]),
            requested_centers=requested_centers[action_id],
            actual_centers=len(action_centers),
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
    retained_counts = {
        category.label: category.raw_count
        for category in refined_catalog.action_categories
        if category.retained and category.label != "stop"
    }
    per_clip: list[list[tuple[int, float]]] = []
    for clip_index, raw_label in enumerate(candidate_labels[index] for index in range(len(clips))):
        assignments = action_parent_distribution(raw_label, retained_counts)
        if not assignments:
            raise ValueError(f"missing parent assignments for clip {clip_index}")
        merged: dict[int, float] = {}
        for parent_label, action_probability in assignments:
            parent_category = next(
                (
                    category
                    for category in refined_catalog.action_categories
                    if category.label == parent_label and category.action_id is not None
                ),
                None,
            )
            if parent_category is None:
                raise ValueError(f"missing parent assignments for clip {clip_index}")
            action_id = int(parent_category.action_id)
            if action_id not in centers_by_action or action_id not in leaf_ids_by_action:
                raise ValueError(f"missing parent assignments for clip {clip_index}")
            conditional = visual_center_probabilities(
                candidate_values[clip_index : clip_index + 1],
                centers_by_action[action_id],
            )[0]
            for leaf_id, visual_probability in zip(
                leaf_ids_by_action[action_id], conditional, strict=True
            ):
                key = int(leaf_id)
                merged[key] = merged.get(key, 0.0) + float(action_probability * visual_probability)
        ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        total = sum(weight for _, weight in ordered)
        if (
            not ordered
            or not math.isfinite(total)
            or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-10)
        ):
            raise ValueError(f"non-normalized prototype output for clip {clip_index}")
        per_clip.append([(leaf_id, weight / total) for leaf_id, weight in ordered])

    width = max((len(assignments) for assignments in per_clip), default=0)
    prototype_indices = np.full((len(clips), width), -1, dtype=np.int32)
    prototype_weights = np.zeros((len(clips), width), dtype=np.float32)
    for clip_index, assignments in enumerate(per_clip):
        for slot, (leaf_id, weight) in enumerate(assignments):
            prototype_indices[clip_index, slot] = leaf_id
            prototype_weights[clip_index, slot] = np.float32(weight)
        row_sum = float(np.sum(prototype_weights[clip_index], dtype=np.float64))
        prototype_weights[clip_index, 0] += np.float32(1.0 - row_sum)
        definitive_order = sorted(
            range(len(assignments)),
            key=lambda slot: (
                -float(prototype_weights[clip_index, slot]),
                int(prototype_indices[clip_index, slot]),
            ),
        )
        valid_indices = prototype_indices[clip_index, : len(assignments)].copy()
        valid_weights = prototype_weights[clip_index, : len(assignments)].copy()
        prototype_indices[clip_index, : len(assignments)] = valid_indices[definitive_order]
        prototype_weights[clip_index, : len(assignments)] = valid_weights[definitive_order]
    if len(clips) and (
        np.any((prototype_indices < 0) != (prototype_weights == 0.0))
        or not np.allclose(
            np.sum(prototype_weights, axis=1, dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=1.0e-7,
        )
    ):
        raise ValueError("non-normalized prototype output")

    return HierarchicalPrototypeResult(
        prototypes=PrototypeData(
            centers=np.stack(centers).astype(np.float32),
            indices=prototype_indices,
            weights=prototype_weights,
            labels=refined_catalog.labels,
        ),
        catalog=refined_catalog,
        clip_action_labels=np.asarray(
            [candidate_labels[index] for index in range(len(clips))], dtype=np.str_
        ),
    )
