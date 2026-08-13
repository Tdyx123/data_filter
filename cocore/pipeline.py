"""Independent Cocore selection artifacts over shared RelCore stages."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy import sparse

from relcore.export import write_selection_outputs
from relcore.graph import build_graph
from relcore.features.normalization import RobustNormalizer
from relcore.features.visual_encoder import (
    DummyVisualEncoder,
    FrozenClipEncoder,
    VisualEncoder,
)
from relcore.graph.prototypes import valid_prototype_assignments
from relcore.pipeline import (
    scan_stage as relcore_scan_stage,
)
from relcore.scoring import compute_reliability
from relcore.schemas import ClipRecord, EdgeTable, GraphData
from relcore.utils.io import (
    cache_is_valid,
    directory_sha256,
    publish_stage,
    stable_hash,
    write_json,
)
from relcore.utils.random import seed_everything

from cocore import __version__
from cocore.config import resolve_config, to_relcore_config
from cocore.encoding import (
    HALF_WINDOWS,
    CocoreEncodedArtifact,
    CocoreEncodedClips,
    encode_cocore_dataset,
)
from cocore.objective import CocoreObjectiveContext
from cocore.prototypes import (
    MIN_FREQUENCY,
    _euclidean_distances,
    assign_half_actions,
    build_hierarchical_motion_prototypes,
    create_action_catalog,
    distance_soft_weights,
    requested_cluster_limits,
)
from cocore.selection import (
    LazyHeapSelector,
    build_max_coverage_seed,
)


GRAPH_DIRECTORY = "graph-12-motion-primitives"
RELIABILITY_METRICS = ("support", "progress")
PROTOTYPE_SCHEMA_VERSION = 3


def _number_tag(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def selection_directory_name(relation_type: str, relation_weight: float, ratio: float) -> str:
    return (
        f"select-{relation_type}-w{_number_tag(relation_weight)}-"
        f"top{_number_tag(ratio * 100.0)}pct"
    )


def _output_root(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    return Path(
        output_dir if output_dir is not None else config["output"]["directory"]
    ).expanduser()


def _mark_shared_stage(root: Path, directory: str, stage: str) -> None:
    manifest_path = root / directory / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "producer": "cocore",
            "cocore_version": __version__,
            "cocore_stage": stage,
        }
    )
    write_json(manifest_path, payload)


def _make_visual_encoder(config: Mapping[str, Any]) -> VisualEncoder:
    name = str(config["visual"]["encoder"]).lower()
    if name == "dummy":
        return DummyVisualEncoder()
    if name == "clip":
        return FrozenClipEncoder(config["visual"])
    raise ValueError(f"unknown visual encoder {name!r}")


def _save_cocore_encoded(
    temporary: Path,
    encoded: CocoreEncodedClips,
    *,
    fingerprint: str,
    runtime_seconds: float,
) -> None:
    np.save(temporary / "embeddings.npy", encoded.embeddings)
    np.save(temporary / "raw_relations.npy", encoded.raw_relations)
    np.save(temporary / "visual_half_embeddings.npy", encoded.visual_half_embeddings)
    np.save(temporary / "state_sequences.npy", encoded.state_sequences)
    np.save(temporary / "action_sequences.npy", encoded.action_sequences)
    np.save(temporary / "visual_progress.npy", encoded.visual_progress)
    np.savez(
        temporary / "normalization.npz",
        state_median=encoded.state_normalizer.median,
        state_iqr=encoded.state_normalizer.iqr,
        action_median=encoded.action_normalizer.median,
        action_iqr=encoded.action_normalizer.iqr,
    )
    np.savez(
        temporary / "projection_matrices.npz",
        **encoded.relation_encoder.projection_matrices,
    )
    np.savez(
        temporary / "relation_pca.npz",
        mean=encoded.relation_projector.mean_,
        scale=encoded.relation_projector.scale_,
        components=encoded.relation_projector.components_,
    )
    write_json(
        temporary / "manifest.json",
        {
            "status": "complete",
            "producer": "cocore",
            "cocore_version": __version__,
            "cocore_stage": "encode",
            "fingerprint": fingerprint,
            "clips": len(encoded.clips),
            "embedding_dim": int(encoded.embeddings.shape[1]),
            "visual_half_embedding_dim": int(encoded.visual_half_embeddings.shape[2]),
            "clip_length": 15,
            "clip_stride": 15,
            "clip_anchors": [0, 7, 14],
            "visual_half_windows": [list(window) for window in HALF_WINDOWS],
            "visual_half_encoding": "mean",
            "runtime_seconds": runtime_seconds,
        },
    )


def scan_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, object, list[ClipRecord], str]:
    resolved = resolve_config(config)
    translated = to_relcore_config(resolved)
    result = relcore_scan_stage(translated, output_dir=output_dir, force=force)
    _mark_shared_stage(result[0], "scan", "scan")
    return result


def encode_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
) -> tuple[Path, object, CocoreEncodedArtifact]:
    resolved = resolve_config(config)
    seed_everything(int(resolved["seed"]))
    root, adapter, clips, scan_fingerprint = scan_stage(
        resolved, output_dir=output_dir, force=force
    )
    visual_config = dict(resolved["visual"])
    model_sha256 = (
        directory_sha256(str(visual_config["model"]))
        if visual_config["encoder"] == "clip"
        else None
    )
    fingerprint = stable_hash(
        {
            "producer": "cocore",
            "version": __version__,
            "stage": "encode",
            "adapter": adapter.fingerprint(),
            "upstream": scan_fingerprint,
            "dataset": resolved["dataset"],
            "visual": resolved["visual"],
            "normalization": resolved["normalization"],
            "relation": resolved["relation"],
            "runtime": {"max_episodes": resolved["runtime"].get("max_episodes")},
            "seed": resolved["seed"],
            "fixed_clip": {"length": 15, "stride": 15},
            "visual_half_windows": [list(window) for window in HALF_WINDOWS],
            "visual_model_sha256": model_sha256,
        }
    )
    destination = root / "encode"
    normalization = np.load(root / "scan" / "normalization.npz")
    epsilon = float(resolved["normalization"]["epsilon"])
    state_normalizer = RobustNormalizer(
        normalization["state_median"], normalization["state_iqr"], epsilon
    )
    action_normalizer = RobustNormalizer(
        normalization["action_median"], normalization["action_iqr"], epsilon
    )

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        encoder = visual_encoder or _make_visual_encoder(resolved)
        encoded = encode_cocore_dataset(
            adapter,
            encoder,
            projection_dim=int(resolved["relation"]["projection_dim"]),
            output_dim=int(resolved["relation"]["output_dim"]),
            lags=tuple(int(lag) for lag in resolved["relation"]["lags"]),
            seed=int(resolved["seed"]),
            epsilon=epsilon,
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
            max_episodes=resolved["runtime"].get("max_episodes"),
            progress_interval=100,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
        if [clip.sample_id for clip in encoded.clips] != [clip.sample_id for clip in clips]:
            raise ValueError("encode clip order does not match scan clip index")
        _save_cocore_encoded(
            temporary,
            encoded,
            fingerprint=fingerprint,
            runtime_seconds=time.perf_counter() - started,
        )

    required = (
        "embeddings.npy",
        "raw_relations.npy",
        "visual_half_embeddings.npy",
        "state_sequences.npy",
        "action_sequences.npy",
        "visual_progress.npy",
        "normalization.npz",
        "projection_matrices.npz",
        "relation_pca.npz",
    )
    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=required,
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    return (
        root,
        adapter,
        CocoreEncodedArtifact(
            clips=clips,
            embeddings=np.load(destination / "embeddings.npy"),
            visual_half_embeddings=np.load(destination / "visual_half_embeddings.npy"),
            state_sequences=np.load(destination / "state_sequences.npy"),
            action_sequences=np.load(destination / "action_sequences.npy"),
            visual_progress=np.load(destination / "visual_progress.npy"),
            fingerprint=fingerprint,
        ),
    )


def graph_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
) -> tuple[Path, object, list[ClipRecord], GraphData, str]:
    resolved = resolve_config(config)
    seed_everything(int(resolved["seed"]))
    root, adapter, encoded = encode_stage(
        resolved,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
    )
    fingerprint = stable_hash(
        {
            "producer": "cocore",
            "version": __version__,
            "stage": "graph",
            "adapter": adapter.fingerprint(),
            "upstream": encoded.fingerprint,
            "quality": resolved["quality"],
            "prototypes": resolved["prototypes"],
            "graph": resolved["graph"],
            "seed": resolved["seed"],
            "max_episodes": resolved["runtime"].get("max_episodes"),
            "reliability_metrics": list(RELIABILITY_METRICS),
            "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
        }
    )
    destination = root / GRAPH_DIRECTORY

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        quality_config = resolved["quality"]
        reliability = compute_reliability(
            encoded.embeddings,
            encoded.state_sequences,
            encoded.action_sequences,
            encoded.visual_progress,
            knn=int(quality_config["knn"]),
            gripper_progress_weight=float(quality_config["gripper_progress_weight"]),
            visual_progress_weight=float(quality_config["visual_progress_weight"]),
            noop_threshold=float(quality_config["noop_threshold"]),
            gripper_action_index=int(quality_config["gripper_action_index"]),
            min_reliability=float(quality_config["min_reliability"]),
            reliability_metrics=RELIABILITY_METRICS,
        )
        prototype_config = resolved["prototypes"]
        hierarchy = build_hierarchical_motion_prototypes(
            adapter,
            encoded.clips,
            encoded.visual_half_embeddings,
            batch_size=int(prototype_config["batch_size"]),
            max_iter=int(prototype_config["max_iter"]),
            seed=int(resolved["seed"]),
            max_episodes=resolved["runtime"].get("max_episodes"),
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
        )
        graph_config = resolved["graph"]
        graph = build_graph(
            encoded.clips,
            encoded.embeddings,
            reliability.reliability,
            hierarchy.prototypes,
            knn=int(graph_config["knn"]),
            similarity_threshold=float(graph_config["similarity_threshold"]),
            cooccurrence_max_gap=int(graph_config["cooccurrence_max_gap"]),
        )
        np.savez(
            temporary / "nodes.npz",
            task_indices=graph.task_indices,
            reliability=graph.reliability,
            prototype_indices=graph.prototype_indices,
            prototype_weights=graph.prototype_weights,
            prototype_action_weights=hierarchy.action_weights,
            prototype_distance_weights=hierarchy.distance_weights,
            support=reliability.support,
            progress=reliability.progress,
            smoothness=reliability.smoothness,
            noop_ratio=reliability.noop_ratio,
        )
        assert hierarchy.prototypes.centers is not None
        assert hierarchy.half_action_labels is not None
        np.save(temporary / "prototype_centers.npy", hierarchy.prototypes.centers)
        np.save(temporary / "half_action_labels.npy", hierarchy.half_action_labels)
        write_json(temporary / "prototype_catalog.json", hierarchy.catalog.to_dict())
        _save_edge_table(temporary / "sequence_edges.npz", graph.sequence_edges)
        _save_edge_table(temporary / "similarity_edges.npz", graph.similarity_edges)
        sparse.save_npz(temporary / "transition_matrix.npz", graph.transition_matrix)
        sparse.save_npz(temporary / "cooccurrence_matrix.npz", graph.cooccurrence_matrix)
        write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "producer": "cocore",
                "cocore_version": __version__,
                "cocore_stage": "graph",
                "fingerprint": fingerprint,
                "upstream_fingerprint": encoded.fingerprint,
                "stage_directory": GRAPH_DIRECTORY,
                "reliability_metrics": list(RELIABILITY_METRICS),
                "prototype_method": "motion_primitives",
                "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
                "nodes": len(graph.sample_ids),
                "sequence_edges": len(graph.sequence_edges.source),
                "similarity_edges": len(graph.similarity_edges.source),
                "runtime_seconds": time.perf_counter() - started,
            },
        )

    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=(
            "nodes.npz",
            "prototype_catalog.json",
            "prototype_centers.npy",
            "half_action_labels.npy",
            "sequence_edges.npz",
            "similarity_edges.npz",
            "transition_matrix.npz",
            "cooccurrence_matrix.npz",
        ),
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    nodes = np.load(destination / "nodes.npz")
    graph = GraphData(
        sample_ids=[clip.sample_id for clip in encoded.clips],
        task_indices=nodes["task_indices"],
        embeddings=encoded.embeddings,
        reliability=nodes["reliability"],
        prototype_indices=nodes["prototype_indices"],
        prototype_weights=nodes["prototype_weights"],
        sequence_edges=_load_edge_table(destination / "sequence_edges.npz", "sequence"),
        similarity_edges=_load_edge_table(destination / "similarity_edges.npz", "similarity"),
        transition_matrix=sparse.load_npz(destination / "transition_matrix.npz"),
        cooccurrence_matrix=sparse.load_npz(destination / "cooccurrence_matrix.npz"),
        prototype_labels=_prototype_labels(destination),
    )
    return root, adapter, encoded.clips, graph, fingerprint


def _selection_budget(config: Mapping[str, Any], candidate_count: int) -> int:
    configured = config["selection"].get("budget")
    budget = (
        int(configured)
        if configured is not None
        else int(np.floor(candidate_count * float(config["selection"]["ratio"]) + 0.5))
    )
    if budget <= 0 or budget > candidate_count:
        raise ValueError("selection budget must be within candidate count")
    return budget


def _load_edge_table(path: Path, edge_type: str) -> EdgeTable:
    payload = np.load(path)
    return EdgeTable(payload["source"], payload["target"], payload["weight"], edge_type)


def _save_edge_table(path: Path, table: EdgeTable) -> None:
    np.savez(path, source=table.source, target=table.target, weight=table.weight)


def _load_clips(path: Path) -> list[ClipRecord]:
    return [ClipRecord(**row) for row in pq.read_table(path).to_pylist()]


def _prototype_labels(graph_root: Path) -> tuple[str, ...]:
    payload = json.loads((graph_root / "prototype_catalog.json").read_text(encoding="utf-8"))
    assigned = sorted(
        payload["leaf_prototypes"],
        key=lambda leaf: int(leaf["prototype_id"]),
    )
    return tuple(str(leaf["label"]) for leaf in assigned)


def _leaf_prototype_metadata(graph_root: Path) -> tuple[Mapping[str, Any], ...]:
    payload = json.loads((graph_root / "prototype_catalog.json").read_text(encoding="utf-8"))
    assigned = sorted(
        payload["leaf_prototypes"],
        key=lambda leaf: int(leaf["prototype_id"]),
    )
    return tuple(assigned)


def _load_graph(root: Path) -> tuple[list[ClipRecord], GraphData, Mapping[str, np.ndarray]]:
    clips = _load_clips(root / "scan" / "clips.parquet")
    graph_root = root / GRAPH_DIRECTORY
    nodes = np.load(graph_root / "nodes.npz")
    graph = GraphData(
        sample_ids=[clip.sample_id for clip in clips],
        task_indices=nodes["task_indices"],
        embeddings=np.load(root / "encode" / "embeddings.npy"),
        reliability=nodes["reliability"],
        prototype_indices=nodes["prototype_indices"],
        prototype_weights=nodes["prototype_weights"],
        sequence_edges=_load_edge_table(graph_root / "sequence_edges.npz", "sequence"),
        similarity_edges=_load_edge_table(graph_root / "similarity_edges.npz", "similarity"),
        transition_matrix=sparse.load_npz(graph_root / "transition_matrix.npz"),
        cooccurrence_matrix=sparse.load_npz(graph_root / "cooccurrence_matrix.npz"),
        prototype_labels=_prototype_labels(graph_root),
    )
    return clips, graph, nodes


def _validate_hierarchical_graph_artifacts(root: Path) -> None:
    graph_root = root / GRAPH_DIRECTORY
    nodes = np.load(graph_root / "nodes.npz")
    required = {
        "prototype_indices",
        "prototype_weights",
        "prototype_action_weights",
        "prototype_distance_weights",
    }
    if not required <= set(nodes.files):
        raise ValueError("hierarchical prototype node arrays are missing")
    indices = nodes["prototype_indices"]
    weights = nodes["prototype_weights"]
    action_weights = nodes["prototype_action_weights"]
    distance_weights = nodes["prototype_distance_weights"]
    if not (
        indices.ndim == 2
        and indices.shape == weights.shape == action_weights.shape == distance_weights.shape
    ):
        raise ValueError("hierarchical prototype node arrays do not align")
    if not (
        np.all(np.isfinite(weights))
        and np.all(np.isfinite(action_weights))
        and np.all(np.isfinite(distance_weights))
    ):
        raise ValueError("hierarchical prototype node arrays are not finite")
    if not np.allclose(
        weights,
        action_weights * distance_weights,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError("hierarchical prototype weight product mismatch")
    payload = json.loads((graph_root / "prototype_catalog.json").read_text(encoding="utf-8"))
    if (
        payload.get("method") != "motion_primitives"
        or payload.get("schema_version") != PROTOTYPE_SCHEMA_VERSION
        or payload.get("strategy") != "action_halves_then_visual_mean_kmeans"
        or not isinstance(payload.get("action_categories"), list)
        or not isinstance(payload.get("leaf_prototypes"), list)
    ):
        raise ValueError("hierarchical prototype catalog schema is invalid")
    action_categories = payload["action_categories"]
    leaves = payload["leaf_prototypes"]
    if not leaves:
        raise ValueError("hierarchical prototype catalog has no reachable leaves")
    total_labels = payload.get("total_labels")
    if (
        isinstance(total_labels, bool)
        or not isinstance(total_labels, int)
        or total_labels < 0
    ):
        raise ValueError("hierarchical prototype total label count is invalid")
    if not all(isinstance(category, Mapping) for category in action_categories) or not all(
        isinstance(leaf, Mapping) for leaf in leaves
    ):
        raise ValueError("hierarchical prototype catalog entries are invalid")
    labels = [category.get("label") for category in action_categories]
    if (
        any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("hierarchical prototype action labels are invalid")
    counts = [category.get("count") for category in action_categories]
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts
    ) or sum(counts) != total_labels:
        raise ValueError("hierarchical prototype action counts are invalid")
    if list(zip(counts, labels, strict=True)) != sorted(
        zip(counts, labels, strict=True), key=lambda item: (-item[0], item[1])
    ):
        raise ValueError("hierarchical prototype action order is invalid")

    expected_action_id = 0
    for category in action_categories:
        count = int(category["count"])
        expected_proportion = count / total_labels if total_labels else 0.0
        proportion = category.get("proportion")
        if (
            isinstance(proportion, bool)
            or not isinstance(proportion, (int, float))
            or not np.isfinite(proportion)
            or not np.isclose(float(proportion), expected_proportion)
        ):
            raise ValueError("hierarchical prototype action proportion is invalid")
        expected_retained = expected_proportion > MIN_FREQUENCY
        expected_fallback = category["label"] == "stop" and not expected_retained
        if category.get("retained") is not expected_retained:
            raise ValueError("hierarchical prototype retention threshold is invalid")
        if category.get("fallback") is not expected_fallback:
            raise ValueError("hierarchical prototype fallback category is invalid")
        expected_id = expected_action_id if expected_retained or expected_fallback else None
        if category.get("action_id") != expected_id:
            raise ValueError("hierarchical prototype action id is invalid")
        if expected_id is not None:
            expected_action_id += 1

    if [leaf.get("prototype_id") for leaf in leaves] != list(range(len(leaves))):
        raise ValueError("hierarchical prototype ids are not contiguous")
    action_by_id = {
        int(category["action_id"]): category
        for category in action_categories
        if category.get("action_id") is not None
    }
    if sorted(action_by_id) != list(range(len(action_by_id))):
        raise ValueError("hierarchical action ids are not contiguous")
    for leaf in leaves:
        action_id = leaf.get("action_id")
        if not isinstance(action_id, int) or action_id not in action_by_id:
            raise ValueError("hierarchical prototype action id is invalid")
        action_label = str(action_by_id[action_id].get("label"))
        if leaf.get("action_label") != action_label:
            raise ValueError("hierarchical prototype action label mismatch")
        cluster_id = leaf.get("cluster_id")
        if not isinstance(cluster_id, int) or cluster_id < 0:
            raise ValueError("hierarchical prototype cluster id is invalid")
        expected_label = (
            "stop::fallback"
            if leaf.get("fallback") is True
            else f"{action_label}::cluster_{cluster_id}"
        )
        if leaf.get("label") != expected_label:
            raise ValueError("hierarchical prototype label mismatch")

    centers = np.asarray(np.load(graph_root / "prototype_centers.npy"), dtype=np.float32)
    visual_halves = np.asarray(
        np.load(root / "encode" / "visual_half_embeddings.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    half_action_labels = np.load(graph_root / "half_action_labels.npy")
    if (
        visual_halves.ndim != 3
        or visual_halves.shape[:2] != (indices.shape[0], 2)
        or visual_halves.shape[2] == 0
        or not np.all(np.isfinite(visual_halves))
        or half_action_labels.ndim != 2
        or half_action_labels.shape != (indices.shape[0], 2)
        or half_action_labels.dtype.kind not in {"U", "S"}
        or centers.ndim != 2
        or centers.shape != (len(leaves), visual_halves.shape[2])
        or not np.all(np.isfinite(centers))
    ):
        raise ValueError("hierarchical half-clip artifacts are invalid")
    observed_counts = Counter(str(label) for label in half_action_labels.ravel())
    if observed_counts != Counter(
        {str(category["label"]): int(category["count"]) for category in action_categories}
    ):
        raise ValueError("hierarchical half action counts are invalid")
    expected_catalog = create_action_catalog(observed_counts, int(half_action_labels.size))
    expected_categories = [asdict(category) for category in expected_catalog.action_categories]
    for actual, expected in zip(action_categories, expected_categories, strict=True):
        for field in ("action_id", "label", "count", "proportion", "retained", "fallback"):
            if actual.get(field) != expected[field]:
                raise ValueError("hierarchical half action catalog is invalid")
    half_indices, half_weights = assign_half_actions(expected_catalog, half_action_labels)
    normalized_visual = visual_halves / np.maximum(
        np.linalg.norm(visual_halves, axis=2, keepdims=True), 1.0e-8
    )
    padding = indices < 0
    if (
        np.any(indices[~padding] >= len(leaves))
        or np.any(indices[padding] != -1)
        or np.any(weights[padding] != 0.0)
        or np.any(action_weights[padding] != 0.0)
        or np.any(distance_weights[padding] != 0.0)
        or np.any(weights[~padding] <= 0.0)
        or np.any(action_weights[~padding] <= 0.0)
        or np.any(action_weights[~padding] > 1.0)
        or np.any(distance_weights[~padding] < 0.3)
        or np.any(distance_weights[~padding] > 1.0)
    ):
        raise ValueError("hierarchical prototype assignment arrays are invalid")
    expected_per_clip: list[dict[int, tuple[float, float, float, int, int, int]]] = [
        {} for _ in range(len(indices))
    ]

    def retain_expected(
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
        existing = expected_per_clip[clip_index].get(leaf_id)
        if existing is None or combined > existing[0] or (
            combined == existing[0] and half_index < existing[5]
        ):
            expected_per_clip[clip_index][leaf_id] = candidate

    expected_leaf_layout: list[tuple[int, int]] = []
    for action_id, category in sorted(action_by_id.items()):
        membership = half_indices == action_id
        member_map: dict[tuple[int, int], float] = {}
        for clip_index, half_index, slot in np.argwhere(membership):
            key = (int(clip_index), int(half_index))
            member_map[key] = max(
                member_map.get(key, 0.0),
                float(half_weights[clip_index, half_index, slot]),
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
        member_values = normalized_visual[member_clips, member_halves]
        bucket_size = category.get("bucket_size")
        if (
            isinstance(bucket_size, bool)
            or not isinstance(bucket_size, int)
            or bucket_size != len(member_positions)
        ):
            raise ValueError("hierarchical prototype bucket size is invalid")
        action_leaves = [leaf for leaf in leaves if leaf["action_id"] == action_id]
        action_leaves.sort(key=lambda leaf: int(leaf["cluster_id"]))

        if category["fallback"]:
            if (
                category.get("requested_clusters") is not None
                or category.get("requested_top_m") is not None
                or category.get("actual_clusters") != 1
                or category.get("top_m") != 1
                or category.get("distance_q10") is not None
                or category.get("distance_q90") is not None
                or len(action_leaves) != 1
                or action_leaves[0].get("cluster_id") != 0
                or action_leaves[0].get("fallback") is not True
            ):
                raise ValueError("hierarchical prototype fallback leaf is invalid")
            expected_leaf_layout.append((action_id, 0))
            fallback_leaf_id = int(action_leaves[0]["prototype_id"])
            expected_center = (
                np.average(member_values, axis=0, weights=member_weights).astype(np.float32)
                if len(member_positions)
                else np.zeros(visual_halves.shape[2], dtype=np.float32)
            )
            if not np.allclose(centers[fallback_leaf_id], expected_center):
                raise ValueError("hierarchical prototype fallback center is invalid")
            for clip_index, half_index, first_weight in zip(
                member_clips, member_halves, member_weights, strict=True
            ):
                retain_expected(
                    int(clip_index),
                    int(half_index),
                    fallback_leaf_id,
                    float(first_weight),
                    1.0,
                    action_id,
                    0,
                )
            continue

        requested_clusters, requested_top_m = requested_cluster_limits(
            float(category["proportion"])
        )
        if category.get("requested_clusters") != requested_clusters:
            raise ValueError("hierarchical prototype cluster formula is invalid")
        if category.get("requested_top_m") != requested_top_m:
            raise ValueError("hierarchical prototype Top-M formula is invalid")
        actual_clusters = min(len(member_positions), requested_clusters)
        top_m = min(actual_clusters, requested_top_m)
        if category.get("actual_clusters") != actual_clusters:
            raise ValueError("hierarchical prototype leaf count is invalid")
        if category.get("top_m") != top_m:
            raise ValueError("hierarchical prototype Top-M formula is invalid")
        cluster_ids = [leaf.get("cluster_id") for leaf in action_leaves]
        if (
            cluster_ids != list(range(actual_clusters))
            or any(leaf.get("fallback") is not False for leaf in action_leaves)
        ):
            raise ValueError("hierarchical prototype cluster ids or leaf count are invalid")
        expected_leaf_layout.extend((action_id, cluster_id) for cluster_id in range(actual_clusters))

        if not len(member_positions):
            if category.get("distance_q10") is not None or category.get("distance_q90") is not None:
                raise ValueError("hierarchical prototype distance quantiles are invalid")
            continue
        action_leaf_ids = np.asarray(
            [int(leaf["prototype_id"]) for leaf in action_leaves], dtype=np.int64
        )
        action_centers = centers[action_leaf_ids]
        distances = _euclidean_distances(member_values, action_centers)
        expected_distance_weights, q10, q90 = distance_soft_weights(distances)
        catalog_q10 = category.get("distance_q10")
        catalog_q90 = category.get("distance_q90")
        if (
            isinstance(catalog_q10, bool)
            or isinstance(catalog_q90, bool)
            or not isinstance(catalog_q10, (int, float))
            or not isinstance(catalog_q90, (int, float))
            or not np.isfinite(catalog_q10)
            or not np.isfinite(catalog_q90)
            or catalog_q10 < 0.0
            or catalog_q90 < catalog_q10
            or not np.isclose(float(catalog_q10), q10, rtol=1.0e-6, atol=1.0e-7)
            or not np.isclose(float(catalog_q90), q90, rtol=1.0e-6, atol=1.0e-7)
        ):
            raise ValueError("hierarchical prototype distance quantiles are invalid")
        best_clusters = np.argsort(
            -expected_distance_weights, axis=1, kind="stable"
        )[:, :top_m]
        leaf_offset = int(action_leaves[0]["prototype_id"])
        for member_index, (clip_index, half_index, first_weight) in enumerate(
            zip(member_clips, member_halves, member_weights, strict=True)
        ):
            for cluster_id in best_clusters[member_index]:
                cluster_id = int(cluster_id)
                retain_expected(
                    int(clip_index),
                    int(half_index),
                    leaf_offset + cluster_id,
                    float(first_weight),
                    float(expected_distance_weights[member_index, cluster_id]),
                    action_id,
                    cluster_id,
                )

    actual_leaf_layout = [
        (int(leaf["action_id"]), int(leaf["cluster_id"])) for leaf in leaves
    ]
    if actual_leaf_layout != expected_leaf_layout:
        raise ValueError("hierarchical prototype leaf ordering is invalid")

    for clip_index, assignments in enumerate(expected_per_clip):
        expected_rows = sorted(
            ((leaf_id, *assignment) for leaf_id, assignment in assignments.items()),
            key=lambda row: (-row[1], row[4], row[5]),
        )
        valid_slots = np.flatnonzero(indices[clip_index] >= 0)
        if len(valid_slots) != len(expected_rows):
            raise ValueError("hierarchical prototype Top-M assignments are invalid")
        for slot, expected in zip(valid_slots, expected_rows, strict=True):
            leaf_id, combined, first_weight, second_weight, _, _, _ = expected
            if (
                int(indices[clip_index, slot]) != leaf_id
                or not np.isclose(weights[clip_index, slot], combined)
                or not np.isclose(action_weights[clip_index, slot], first_weight)
                or not np.isclose(distance_weights[clip_index, slot], second_weight)
            ):
                raise ValueError("hierarchical half-clip assignments are invalid")

    for category in action_categories:
        if category.get("action_id") is not None:
            continue
        if (
            category.get("bucket_size") != 0
            or category.get("requested_clusters") is not None
            or category.get("actual_clusters") != 0
            or category.get("requested_top_m") is not None
            or category.get("top_m") != 0
            or category.get("distance_q10") is not None
            or category.get("distance_q90") is not None
        ):
            raise ValueError("hierarchical prototype unassigned action diagnostics are invalid")


def _selection_rows(
    resolved: Mapping[str, Any],
    clips: list[ClipRecord],
    graph: GraphData,
    nodes: Mapping[str, np.ndarray],
    result,
    context: CocoreObjectiveContext,
    leaf_metadata: tuple[Mapping[str, Any], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = {index: position for position, index in enumerate(result.selected_indices)}
    all_rows: list[dict[str, Any]] = []
    for index, clip in enumerate(clips):
        prototype_indices, prototype_weights = valid_prototype_assignments(
            graph.prototype_indices[index], graph.prototype_weights[index]
        )
        labels = [graph.prototype_labels[int(value)] for value in prototype_indices]
        metadata = [leaf_metadata[int(value)] for value in prototype_indices]
        action_labels = [str(value["action_label"]) for value in metadata]
        cluster_ids = [int(value["cluster_id"]) for value in metadata]
        assignment_count = len(prototype_indices)
        action_weights = nodes["prototype_action_weights"][index, :assignment_count]
        distance_weights = nodes["prototype_distance_weights"][index, :assignment_count]
        position = positions.get(index)
        row = {
            **asdict(clip),
            "selected": position is not None,
            "support": float(nodes["support"][index]),
            "progress": float(nodes["progress"][index]),
            "reliability": float(graph.reliability[index]),
            "primary_prototype": int(prototype_indices[0]),
            "prototype_indices": [int(value) for value in prototype_indices],
            "prototype_weights": [float(value) for value in prototype_weights],
            "prototype_action_labels": action_labels,
            "prototype_cluster_ids": cluster_ids,
            "prototype_action_weights": [float(value) for value in action_weights],
            "prototype_distance_weights": [float(value) for value in distance_weights],
            "primary_prototype_label": labels[0],
            "prototype_labels": labels,
            "primary_action_label": action_labels[0],
            "selection_order": position + 1 if position is not None else None,
            "selection_phase": result.selection_phases[position] if position is not None else None,
            "selection_step": result.selection_steps[position] if position is not None else None,
            "heap_refreshes": result.heap_refreshes[position] if position is not None else None,
            "selection_score_delta": (
                float(result.score_deltas[position]) if position is not None else None
            ),
        }
        all_rows.append(row)
    selected_rows: list[dict[str, Any]] = []
    for index in result.selected_indices:
        row = dict(all_rows[index])
        row["dataset_name"] = str(resolved["dataset"]["name"])
        row["dataset_path"] = str(resolved["dataset"].get("path", ""))
        row["marginal_gain"] = row["selection_score_delta"]
        selected_rows.append(row)
    return selected_rows, all_rows


def select_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
) -> Path:
    resolved = resolve_config(config)
    root = _output_root(resolved, output_dir)
    resolved["output"]["directory"] = str(root)
    root, adapter, clips, graph, graph_fingerprint = graph_stage(
        resolved,
        output_dir=root,
        force=force,
        visual_encoder=visual_encoder,
    )
    budget = _selection_budget(resolved, len(graph.sample_ids))
    relation_type = str(resolved["objective"]["relation"])
    relation_weight = float(resolved["objective"]["relation_weight"])
    ratio = float(resolved["selection"]["ratio"])
    directory = selection_directory_name(relation_type, relation_weight, ratio)
    destination = root / directory
    selection_config = resolved["selection"]
    max_refreshes = int(selection_config["max_refreshes"])
    algorithm = {"type": "lazy_max_heap", "max_refreshes": max_refreshes}
    fingerprint = stable_hash(
        {
            "producer": "cocore",
            "version": __version__,
            "stage": "select",
            "upstream": graph_fingerprint,
            "objective": resolved["objective"],
            "selection": resolved["selection"],
            "seed": resolved["seed"],
            "algorithm": algorithm,
        }
    )

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        context = CocoreObjectiveContext(
            graph,
            relation_type,
            relation_weight,
            similarity_threshold=float(resolved["graph"]["similarity_threshold"]),
        )
        coverage_seed = build_max_coverage_seed(context, budget=budget)
        selector = LazyHeapSelector(
            context,
            max_refreshes=max_refreshes,
        )
        result = selector.select(budget, initial_indices=coverage_seed.selected_indices)
        graph_nodes = np.load(root / GRAPH_DIRECTORY / "nodes.npz")
        selected_rows, all_rows = _selection_rows(
            resolved,
            clips,
            graph,
            graph_nodes,
            result,
            context,
            _leaf_prototype_metadata(root / GRAPH_DIRECTORY),
        )
        final_coverage = context.prototype_mass[
            np.asarray(result.selected_indices, dtype=np.int64)
        ].max(axis=0)
        task_counts = {
            str(task): sum(
                int(graph.task_indices[index]) == task for index in result.selected_indices
            )
            for task in sorted({int(value) for value in graph.task_indices})
        }
        scan_manifest = json.loads(
            (root / "scan" / "manifest.json").read_text(encoding="utf-8")
        )
        report = {
            "producer": "cocore",
            "number_of_episodes": int(scan_manifest["episodes"]),
            "number_of_clips": len(clips),
            "selected_clips": len(result.selected_indices),
            "selection_ratio": len(result.selected_indices) / len(clips),
            "configured_selection_ratio": ratio,
            "reliability_metrics": list(RELIABILITY_METRICS),
            "prototype_method": "motion_primitives",
            "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
            "initial_set_size": len(coverage_seed.selected_indices),
            "coverage": {
                "target": [float(value) for value in coverage_seed.target_coverage],
                "achieved": [float(value) for value in final_coverage],
            },
            "relation_type": relation_type,
            "relation_weight": relation_weight,
            "objective": {
                "relation": float(result.relation),
                "weighted_relation": relation_weight * float(result.relation),
                "redundancy": float(result.redundancy),
                "total": float(result.objective_value),
            },
            "algorithm": algorithm,
            "heap": {
                "initial_size": result.initial_heap_size,
                "total_refreshes": result.total_refreshes,
                "capped_selections": result.capped_selections,
                "max_refreshes_observed": result.max_refreshes_observed,
            },
            "task_counts": task_counts,
            "skipped_short_episodes": scan_manifest.get("skipped_short_episodes", []),
            "runtime_seconds": {"select": time.perf_counter() - started},
        }
        for stage, stage_directory in {
            "scan": "scan",
            "encode": "encode",
            "graph": GRAPH_DIRECTORY,
        }.items():
            manifest = json.loads(
                (root / stage_directory / "manifest.json").read_text(encoding="utf-8")
            )
            report["runtime_seconds"][stage] = float(manifest.get("runtime_seconds", 0.0))
        write_selection_outputs(
            temporary,
            selected_rows=selected_rows,
            all_rows=all_rows,
            report=report,
        )
        write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "producer": "cocore",
                "fingerprint": fingerprint,
                "upstream_fingerprint": graph_fingerprint,
                "stage_directory": directory,
                "selected_clips": len(result.selected_indices),
                "budget": budget,
                "selection_ratio": ratio,
                "relation_type": relation_type,
                "relation_weight": relation_weight,
                "algorithm": algorithm,
            },
        )

    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=("selected_manifest.jsonl", "all_clips.parquet", "selection_report.json"),
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    stage_directories = {
        "scan": "scan",
        "encode": "encode",
        "graph": GRAPH_DIRECTORY,
        "select": directory,
    }
    stage_fingerprints = {
        stage: json.loads(
            (root / stage_directory / "manifest.json").read_text(encoding="utf-8")
        )["fingerprint"]
        for stage, stage_directory in stage_directories.items()
    }
    write_json(
        destination / "run_manifest.json",
        {
            "status": "complete",
            "producer": "cocore",
            "fingerprint": stable_hash(
                {
                    "version": __version__,
                    "config": resolved,
                    "stage_fingerprints": stage_fingerprints,
                }
            ),
            "cocore_version": __version__,
            "reliability_metrics": list(RELIABILITY_METRICS),
            "prototype_method": "motion_primitives",
            "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
            "selection_ratio": ratio,
            "relation_type": relation_type,
            "relation_weight": relation_weight,
            "similarity_threshold": float(resolved["graph"]["similarity_threshold"]),
            "algorithm": algorithm,
            "stage_directories": stage_directories,
            "stage_fingerprints": stage_fingerprints,
        },
    )
    return destination


def run_pipeline(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
) -> Path:
    resolved = resolve_config(config)
    root = _output_root(resolved, output_dir)
    resolved["output"]["directory"] = str(root)
    result = select_stage(
        resolved,
        output_dir=root,
        force=force,
        visual_encoder=visual_encoder,
    )
    result.mkdir(parents=True, exist_ok=True)
    (result / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    write_json(
        result / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cocore_version": __version__,
        },
    )
    return result


def validate_output(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = Path(output_dir).expanduser()
    root = result.parent
    required = {
        "selected": result / "selected_manifest.jsonl",
        "all": result / "all_clips.parquet",
        "report": result / "selection_report.json",
        "run": result / "run_manifest.json",
        "select_manifest": result / "manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f"cocore output is missing files: {missing}")
    run_manifest = json.loads(required["run"].read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete" or run_manifest.get("producer") != "cocore":
        raise ValueError("cocore run manifest is not complete")
    if run_manifest.get("cocore_version") != __version__:
        raise ValueError("cocore run manifest version is incompatible")
    if run_manifest.get("prototype_schema_version") != PROTOTYPE_SCHEMA_VERSION:
        raise ValueError("cocore prototype schema version is incompatible")
    algorithm = run_manifest.get("algorithm")
    if not isinstance(algorithm, Mapping) or algorithm.get("type") != "lazy_max_heap":
        raise ValueError("cocore run manifest algorithm is invalid")
    max_refreshes = algorithm.get("max_refreshes")
    if isinstance(max_refreshes, bool) or not isinstance(max_refreshes, int):
        raise ValueError("cocore max_refreshes is invalid")
    if max_refreshes <= 0:
        raise ValueError("cocore max_refreshes is invalid")
    relation_type = str(run_manifest["relation_type"])
    if relation_type not in {"sequence", "cooccurrence"}:
        raise ValueError("cocore relation type is invalid")
    weight = float(run_manifest["relation_weight"])
    ratio = float(run_manifest["selection_ratio"])
    if result.name != selection_directory_name(relation_type, weight, ratio):
        raise ValueError("selection directory does not match relation, weight, and ratio")
    expected_directories = {
        "scan": "scan",
        "encode": "encode",
        "graph": GRAPH_DIRECTORY,
        "select": result.name,
    }
    if run_manifest.get("stage_directories") != expected_directories:
        raise ValueError("run manifest stage directories are invalid")
    stage_required = {
        "scan": ("episodes.parquet", "clips.parquet", "normalization.npz"),
        "encode": (
            "embeddings.npy",
            "raw_relations.npy",
            "visual_half_embeddings.npy",
            "state_sequences.npy",
            "action_sequences.npy",
            "visual_progress.npy",
            "normalization.npz",
            "projection_matrices.npz",
            "relation_pca.npz",
        ),
        "graph": (
            "nodes.npz",
            "prototype_catalog.json",
            "prototype_centers.npy",
            "half_action_labels.npy",
            "sequence_edges.npz",
            "similarity_edges.npz",
            "transition_matrix.npz",
            "cooccurrence_matrix.npz",
        ),
        "select": ("selected_manifest.jsonl", "all_clips.parquet", "selection_report.json"),
    }
    for stage, directory in expected_directories.items():
        path = root / directory
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing stage manifest: {stage}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not cache_is_valid(path, str(manifest.get("fingerprint", "")), stage_required[stage]):
            raise ValueError(f"invalid stage artifacts: {stage}")
        if run_manifest["stage_fingerprints"].get(stage) != manifest["fingerprint"]:
            raise ValueError(f"run/stage fingerprint mismatch: {stage}")
    _validate_hierarchical_graph_artifacts(root)
    select_manifest = json.loads(required["select_manifest"].read_text(encoding="utf-8"))
    if select_manifest.get("producer") != "cocore":
        raise ValueError("selection manifest producer is invalid")
    selected_rows = [
        json.loads(line)
        for line in required["selected"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_rows = pq.read_table(required["all"]).to_pylist()
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if select_manifest.get("algorithm") != algorithm or report.get("algorithm") != algorithm:
        raise ValueError("cocore heap algorithm metadata does not match")
    if report.get("relation_type") != relation_type:
        raise ValueError("selection report relation type mismatch")
    if report.get("prototype_schema_version") != PROTOTYPE_SCHEMA_VERSION:
        raise ValueError("selection report prototype schema version mismatch")
    if not np.isclose(float(report.get("relation_weight", np.nan)), weight):
        raise ValueError("selection report relation weight mismatch")
    if select_manifest.get("relation_type") != relation_type:
        raise ValueError("selection manifest relation type mismatch")
    if not np.isclose(float(select_manifest.get("relation_weight", np.nan)), weight):
        raise ValueError("selection manifest relation weight mismatch")
    if len({row["sample_id"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("selected manifest contains duplicate sample ids")
    if len(selected_rows) != int(select_manifest["budget"]):
        raise ValueError("selected manifest does not match budget")
    if [row["sample_id"] for row in all_rows] != sorted(row["sample_id"] for row in all_rows):
        raise ValueError("all_clips.parquet is not sorted by sample_id")
    leaf_metadata = _leaf_prototype_metadata(root / GRAPH_DIRECTORY)
    nodes = np.load(root / GRAPH_DIRECTORY / "nodes.npz")
    clip_index_by_id = {
        clip.sample_id: index
        for index, clip in enumerate(_load_clips(root / "scan" / "clips.parquet"))
    }
    if set(clip_index_by_id) != {str(row["sample_id"]) for row in all_rows}:
        raise ValueError("all-clips rows do not match the graph clip index")
    for row in all_rows:
        index = clip_index_by_id[str(row["sample_id"])]
        assigned_indices, assigned_weights = valid_prototype_assignments(
            nodes["prototype_indices"][index], nodes["prototype_weights"][index]
        )
        count = len(assigned_indices)
        expected_labels = [leaf_metadata[int(value)]["label"] for value in assigned_indices]
        expected_actions = [
            leaf_metadata[int(value)]["action_label"] for value in assigned_indices
        ]
        expected_clusters = [
            int(leaf_metadata[int(value)]["cluster_id"]) for value in assigned_indices
        ]
        if (
            row.get("prototype_indices") != [int(value) for value in assigned_indices]
            or not np.allclose(row.get("prototype_weights"), assigned_weights)
            or row.get("prototype_labels") != expected_labels
            or row.get("prototype_action_labels") != expected_actions
            or row.get("prototype_cluster_ids") != expected_clusters
            or not np.allclose(
                row.get("prototype_action_weights"),
                nodes["prototype_action_weights"][index, :count],
            )
            or not np.allclose(
                row.get("prototype_distance_weights"),
                nodes["prototype_distance_weights"][index, :count],
            )
            or row.get("primary_prototype_label") != expected_labels[0]
            or row.get("primary_action_label") != expected_actions[0]
        ):
            raise ValueError("hierarchical prototype row metadata mismatch")
    selected_from_all = sorted(
        (row for row in all_rows if row["selected"]), key=lambda row: int(row["selection_order"])
    )
    if [row["sample_id"] for row in selected_from_all] != [
        row["sample_id"] for row in selected_rows
    ]:
        raise ValueError("selected manifest and all-clips rows disagree")
    hierarchical_fields = (
        "prototype_indices",
        "prototype_weights",
        "prototype_labels",
        "prototype_action_labels",
        "prototype_cluster_ids",
        "prototype_action_weights",
        "prototype_distance_weights",
        "primary_prototype",
        "primary_prototype_label",
        "primary_action_label",
    )
    for selected_row, all_row in zip(selected_rows, selected_from_all, strict=True):
        if any(selected_row.get(field) != all_row.get(field) for field in hierarchical_fields):
            raise ValueError("selected hierarchical prototype row metadata mismatch")
    clips, graph, _ = _load_graph(root)
    del clips
    id_to_index = {sample_id: index for index, sample_id in enumerate(graph.sample_ids)}
    selected_indices = [id_to_index[row["sample_id"]] for row in selected_rows]
    context = CocoreObjectiveContext(
        graph,
        relation_type,
        weight,
        similarity_threshold=float(run_manifest["similarity_threshold"]),
    )
    initial_set_size = report.get("initial_set_size")
    if (
        isinstance(initial_set_size, bool)
        or not isinstance(initial_set_size, int)
        or not 0 < initial_set_size <= len(selected_rows)
    ):
        raise ValueError("selection report initial set size is invalid")
    expected_seed = build_max_coverage_seed(context, budget=len(selected_indices))
    if tuple(selected_indices[:initial_set_size]) != expected_seed.selected_indices:
        raise ValueError("selected coverage seed does not match the deterministic seed")

    state = context.empty_state()
    refresh_counts: list[int] = []
    for position, (index, row) in enumerate(
        zip(selected_indices, selected_rows, strict=True), start=1
    ):
        phase = row.get("selection_phase")
        step = row.get("selection_step")
        refreshes = row.get("heap_refreshes")
        if position <= initial_set_size:
            if phase != "coverage_seed" or step != 0 or refreshes != 0:
                raise ValueError("coverage seed heap metadata is invalid")
        else:
            expected_step = position - initial_set_size
            if phase != "heap" or step != expected_step:
                raise ValueError("heap selection order metadata is invalid")
            if (
                isinstance(refreshes, bool)
                or not isinstance(refreshes, int)
                or not 0 <= refreshes <= max_refreshes
            ):
                raise ValueError("heap refresh count is invalid")
            refresh_counts.append(refreshes)
        gain = context.marginal_gain(state, index)
        recorded_gain = float(row.get("selection_score_delta", np.nan))
        if not np.isfinite(recorded_gain) or not np.isclose(
            recorded_gain, gain, rtol=1.0e-7, atol=1.0e-8
        ):
            raise ValueError("selection score delta does not match the objective")
        if not np.isclose(
            float(row.get("marginal_gain", np.nan)), gain, rtol=1.0e-7, atol=1.0e-8
        ):
            raise ValueError("selected marginal gain does not match the objective")
        context.add_candidate(state, index)

    target = context.prototype_mass.max(axis=0)
    achieved = context.prototype_mass[np.asarray(selected_indices, dtype=np.int64)].max(axis=0)
    if not np.allclose(target, achieved, rtol=1.0e-7, atol=1.0e-8):
        raise ValueError("selected set does not attain maximum prototype coverage")
    recorded_coverage = report.get("coverage", {})
    if not np.allclose(recorded_coverage.get("target"), target) or not np.allclose(
        recorded_coverage.get("achieved"), achieved
    ):
        raise ValueError("selection report coverage does not match artifacts")
    objective = report.get("objective", {})
    for name, actual in {
        "relation": state.relation,
        "redundancy": state.redundancy,
        "total": state.score,
    }.items():
        if not np.isclose(float(objective.get(name, np.nan)), actual, rtol=1.0e-7, atol=1.0e-8):
            raise ValueError(f"selection report objective {name} mismatch")
    if not np.isclose(
        float(objective.get("weighted_relation", np.nan)),
        weight * state.relation,
        rtol=1.0e-7,
        atol=1.0e-8,
    ):
        raise ValueError("selection report objective weighted_relation mismatch")

    expected_heap = {
        "initial_size": (
            0 if initial_set_size == len(selected_rows) else len(graph.sample_ids) - initial_set_size
        ),
        "total_refreshes": sum(refresh_counts),
        "capped_selections": sum(value == max_refreshes for value in refresh_counts),
        "max_refreshes_observed": max(refresh_counts, default=0),
    }
    heap_report = report.get("heap")
    if not isinstance(heap_report, Mapping):
        raise ValueError("selection report heap metadata is invalid")
    if heap_report.get("initial_size") != expected_heap["initial_size"]:
        raise ValueError("selection report heap initial size mismatch")
    if heap_report.get("total_refreshes") != expected_heap["total_refreshes"]:
        raise ValueError("selection report heap total refreshes mismatch")
    if heap_report.get("capped_selections") != expected_heap["capped_selections"]:
        raise ValueError("selection report heap capped selections mismatch")
    if heap_report.get("max_refreshes_observed") != expected_heap["max_refreshes_observed"]:
        raise ValueError("selection report heap maximum refreshes mismatch")
    actual_task_counts = {
        str(task): sum(int(graph.task_indices[index]) == task for index in selected_indices)
        for task in sorted({int(value) for value in graph.task_indices})
    }
    if report.get("task_counts") != actual_task_counts:
        raise ValueError("selection report task counts mismatch")
    if config is not None:
        resolved = resolve_config(config)
        if str(resolved["objective"]["relation"]) != relation_type:
            raise ValueError("configuration relation type does not match output")
        if not np.isclose(float(resolved["objective"]["relation_weight"]), weight):
            raise ValueError("configuration relation weight does not match output")
        if not np.isclose(float(resolved["selection"]["ratio"]), ratio):
            raise ValueError("configuration selection ratio does not match output")
        if int(resolved["selection"]["max_refreshes"]) != max_refreshes:
            raise ValueError("configuration max_refreshes does not match output")
    return {"status": "valid", "selected_clips": len(selected_rows)}
