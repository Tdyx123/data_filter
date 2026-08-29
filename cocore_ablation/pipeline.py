"""Isolated LIBERO graph and selection artifacts over Cocore scan/encode caches."""

from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy import sparse

from cocore import pipeline as cocore_pipeline
from cocore.graph import build_graph
from cocore.objective import CocoreObjectiveContext
from cocore.prototypes import build_hierarchical_motion_prototypes
from cocore.random_multibranch import RandomMultiBranchSelector
from cocore.timing import emit_completed_timing
from relcore.export import write_selection_outputs
from relcore.features.visual_encoder import VisualEncoder
from relcore.scoring import compute_reliability
from relcore.schemas import GraphData
from relcore.utils.io import publish_stage, stable_hash, write_json
from relcore.utils.random import seed_everything

from . import __version__
from .config import resolve_config, to_cocore_config
from .reliability import fuse_reliability
from .selection import SeededRandomSelector, initial_selection


SCHEMA_VERSION = 1
_GRAPH_REQUIRED = (
    "nodes.npz",
    "source_clip_indices.npy",
    "prototype_catalog.json",
    "prototype_centers.npy",
    "half_action_labels.npy",
    "sequence_edges.npz",
    "similarity_edges.npz",
    "transition_matrix.npz",
    "cooccurrence_matrix.npz",
)
_SELECT_REQUIRED = (
    "selected_manifest.jsonl",
    "all_clips.parquet",
    "selection_report.json",
    "manifest.json",
)


def _number_tag(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _output_root(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    value = output_dir if output_dir is not None else config["output"]["directory"]
    return Path(value).expanduser()


def _graph_fingerprint(resolved: Mapping[str, Any], encoded_fingerprint: str) -> str:
    prototypes = resolved["prototypes"]
    return stable_hash(
        {
            "producer": "cocore_ablation",
            "schema_version": SCHEMA_VERSION,
            "stage": "graph",
            "upstream_encode": encoded_fingerprint,
            "quality": resolved["quality"],
            "graph": resolved["graph"],
            "prototype_base": {
                key: prototypes[key]
                for key in (
                    "method",
                    "profile",
                    "batch_size",
                    "max_iter",
                    "tol",
                    "use_stop_bucket",
                )
            },
            "representation": prototypes["representation"],
            "use_assignment_confidence": prototypes[
                "use_assignment_confidence"
            ],
            "reliability_metrics": resolved["reliability_metrics"],
            "seed": resolved["seed"],
            "max_episodes": resolved["runtime"].get("max_episodes"),
        }
    )


def _graph_directory(resolved: Mapping[str, Any], fingerprint: str) -> str:
    metrics = "+".join(resolved["reliability_metrics"]) or "uniform"
    prototypes = resolved["prototypes"]
    return (
        f"graph-{metrics}-{prototypes['representation']}-"
        f"conf{int(prototypes['use_assignment_confidence'])}-"
        f"stop{int(prototypes['use_stop_bucket'])}-{fingerprint[:8]}"
    )


def _selection_fingerprint(
    resolved: Mapping[str, Any], graph_fingerprint: str
) -> str:
    return stable_hash(
        {
            "producer": "cocore_ablation",
            "schema_version": SCHEMA_VERSION,
            "stage": "select",
            "upstream_graph": graph_fingerprint,
            "objective": resolved["objective"],
            "selection": resolved["selection"],
            "seed": resolved["seed"],
        }
    )


def _selection_directory(resolved: Mapping[str, Any], fingerprint: str) -> str:
    objective = resolved["objective"]
    selection = resolved["selection"]
    configured_budget = selection.get("budget")
    budget_tag = (
        f"budget{int(configured_budget)}"
        if configured_budget is not None
        else f"top{_number_tag(float(selection['ratio']) * 100.0)}pct"
    )
    return (
        f"select-{objective['relation']}-"
        f"rw{_number_tag(objective['relation_weight'])}-"
        f"dw{_number_tag(objective['redundancy_weight'])}-"
        f"cov{int(selection['use_coverage_seed'])}-"
        f"{selection['strategy']}-{budget_tag}-{fingerprint[:8]}"
    )


def _prepare_upstream(
    resolved: Mapping[str, Any],
    *,
    visual_encoder: VisualEncoder | None,
):
    upstream_config = to_cocore_config(resolved)
    upstream = Path(resolved["upstream"]["directory"]).expanduser()
    return cocore_pipeline.encode_stage(
        upstream_config,
        output_dir=upstream,
        force=False,
        visual_encoder=visual_encoder,
    )


def _catalog_payload(hierarchy, resolved: Mapping[str, Any]) -> dict[str, Any]:
    payload = hierarchy.catalog.to_dict()
    representation = str(resolved["prototypes"]["representation"])
    confidence = bool(resolved["prototypes"]["use_assignment_confidence"])
    payload.update(
        {
            "producer": "cocore_ablation",
            "schema_version": SCHEMA_VERSION,
            "representation": representation,
            "use_assignment_confidence": confidence,
            "strategy": (
                "one_leaf_per_trained_action_stable_parent"
                if representation == "action_only"
                else "cocore_action_visual_hard_nearest"
            ),
        }
    )
    constants = dict(payload["constants"])
    if representation == "action_only":
        constants.update(
            {
                "cluster_count": "one_per_trained_action",
                "visual_parent_tie_break": "unused",
                "distance_weight_range": None,
            }
        )
    constants["assignment_confidence"] = (
        "retention_times_distance"
        if confidence and representation == "action_visual"
        else "retention_only"
        if confidence
        else "uniform_per_half"
    )
    payload["constants"] = constants
    return payload


def _prototype_metadata(graph_root: Path) -> tuple[Mapping[str, Any], ...]:
    payload = json.loads((graph_root / "prototype_catalog.json").read_text())
    if (
        payload.get("producer") != "cocore_ablation"
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("leaf_prototypes"), list)
    ):
        raise ValueError("cocore_ablation prototype catalog is incompatible")
    return tuple(
        sorted(payload["leaf_prototypes"], key=lambda leaf: int(leaf["prototype_id"]))
    )


def _load_graph(
    graph_root: Path,
    encoded,
) -> tuple[list[Any], GraphData, Mapping[str, np.ndarray]]:
    source = cocore_pipeline._load_source_clip_indices(  # noqa: SLF001
        graph_root, len(encoded.clips)
    )
    clips = [encoded.clips[int(index)] for index in source]
    nodes = np.load(graph_root / "nodes.npz")
    metadata = _prototype_metadata(graph_root)
    labels = tuple(str(leaf["label"]) for leaf in metadata)
    graph = GraphData(
        sample_ids=[clip.sample_id for clip in clips],
        task_indices=nodes["task_indices"],
        embeddings=encoded.embeddings[source],
        reliability=nodes["reliability"],
        prototype_indices=nodes["prototype_indices"],
        prototype_weights=nodes["prototype_weights"],
        sequence_edges=cocore_pipeline._load_edge_table(  # noqa: SLF001
            graph_root / "sequence_edges.npz", "sequence"
        ),
        similarity_edges=cocore_pipeline._load_edge_table(  # noqa: SLF001
            graph_root / "similarity_edges.npz", "similarity"
        ),
        transition_matrix=sparse.load_npz(graph_root / "transition_matrix.npz"),
        cooccurrence_matrix=sparse.load_npz(graph_root / "cooccurrence_matrix.npz"),
        prototype_labels=labels,
    )
    if (
        graph.prototype_indices.shape != graph.prototype_weights.shape
        or graph.prototype_indices.shape[0] != len(clips)
        or graph.reliability.shape != (len(clips),)
    ):
        raise ValueError("cocore_ablation graph node arrays are invalid")
    return clips, graph, nodes


def graph_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
):
    resolved = resolve_config(config)
    seed_everything(int(resolved["seed"]))
    root = _output_root(resolved, output_dir)
    resolved["output"]["directory"] = str(root)
    upstream_root, adapter, encoded = _prepare_upstream(
        resolved, visual_encoder=visual_encoder
    )
    fingerprint = _graph_fingerprint(resolved, encoded.fingerprint)
    destination = root / _graph_directory(resolved, fingerprint)

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        quality = resolved["quality"]
        components = compute_reliability(
            encoded.embeddings,
            encoded.state_sequences,
            encoded.action_sequences,
            encoded.visual_progress,
            knn=int(quality["knn"]),
            gripper_progress_weight=float(quality["gripper_progress_weight"]),
            visual_progress_weight=float(quality["visual_progress_weight"]),
            noop_threshold=float(quality["noop_threshold"]),
            gripper_action_index=int(quality["gripper_action_index"]),
            min_reliability=float(quality["min_reliability"]),
            reliability_metrics=("support", "progress"),
        )
        reliability = fuse_reliability(
            components.support,
            components.progress,
            resolved["reliability_metrics"],
            min_reliability=float(quality["min_reliability"]),
        )
        prototype_config = resolved["prototypes"]
        visual_dim = int(resolved["encoding"]["visual_dim"])
        pca_components = cocore_pipeline._load_visual_pca_components(  # noqa: SLF001
            upstream_root / "encode", visual_dim=visual_dim
        )
        hierarchy = build_hierarchical_motion_prototypes(
            adapter,
            encoded.clips,
            encoded.visual_half_embeddings,
            pca_components=pca_components,
            visual_dim=visual_dim,
            frame_cache_dir=upstream_root / "encode" / "frame_embeddings",
            batch_size=int(prototype_config["batch_size"]),
            max_iter=int(prototype_config["max_iter"]),
            tol=float(prototype_config["tol"]),
            seed=int(resolved["seed"]),
            max_episodes=resolved["runtime"].get("max_episodes"),
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
            num_threads=int(prototype_config["num_threads"]),
            use_stop_bucket=bool(prototype_config["use_stop_bucket"]),
            profile="libero",
            representation=str(prototype_config["representation"]),
            use_assignment_confidence=bool(
                prototype_config["use_assignment_confidence"]
            ),
        )
        source = np.flatnonzero(hierarchy.eligible_mask).astype(np.int64)
        if len(source) == 0:
            raise ValueError("no eligible candidate with an action label remains")
        graph_config = resolved["graph"]
        graph = build_graph(
            encoded.clips,
            encoded.embeddings,
            reliability,
            hierarchy.prototypes,
            included_indices=source,
            knn=int(graph_config["knn"]),
            similarity_threshold=float(graph_config["similarity_threshold"]),
            cooccurrence_max_gap=int(graph_config["cooccurrence_max_gap"]),
            normalize_prototype_relations=False,
        )
        np.savez(
            temporary / "nodes.npz",
            task_indices=graph.task_indices,
            reliability=graph.reliability,
            prototype_indices=graph.prototype_indices,
            prototype_weights=graph.prototype_weights,
            support=components.support[source],
            progress=components.progress[source],
            smoothness=components.smoothness[source],
            noop_ratio=components.noop_ratio[source],
        )
        np.save(temporary / "source_clip_indices.npy", source)
        if hierarchy.prototypes.centers is None:
            raise ValueError("prototype builder produced no centers")
        np.save(temporary / "prototype_centers.npy", hierarchy.prototypes.centers)
        np.save(
            temporary / "half_action_labels.npy",
            hierarchy.half_action_labels[source],
        )
        write_json(temporary / "prototype_catalog.json", _catalog_payload(hierarchy, resolved))
        cocore_pipeline._save_edge_table(  # noqa: SLF001
            temporary / "sequence_edges.npz", graph.sequence_edges
        )
        cocore_pipeline._save_edge_table(  # noqa: SLF001
            temporary / "similarity_edges.npz", graph.similarity_edges
        )
        sparse.save_npz(temporary / "transition_matrix.npz", graph.transition_matrix)
        sparse.save_npz(temporary / "cooccurrence_matrix.npz", graph.cooccurrence_matrix)
        write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "producer": "cocore_ablation",
                "schema_version": SCHEMA_VERSION,
                "stage": "graph",
                "fingerprint": fingerprint,
                "upstream_directory": str(upstream_root),
                "upstream_fingerprint": encoded.fingerprint,
                "reliability_metrics": resolved["reliability_metrics"],
                "prototype_representation": prototype_config["representation"],
                "use_assignment_confidence": prototype_config[
                    "use_assignment_confidence"
                ],
                "use_stop_bucket": prototype_config["use_stop_bucket"],
                "nodes": len(graph.sample_ids),
                "runtime_seconds": time.perf_counter() - started,
            },
        )
        if destination.is_dir():
            for child in destination.iterdir():
                if child.is_dir() and child.name.startswith("select-"):
                    shutil.copytree(child, temporary / child.name)

    stage_started = time.perf_counter()
    built = publish_stage(
        destination,
        fingerprint=fingerprint,
        required=_GRAPH_REQUIRED,
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)) and not force,
        build=build,
    )
    if built:
        emit_completed_timing("ablation.graph", time.perf_counter() - stage_started)
    clips, graph, nodes = _load_graph(destination, encoded)
    return (
        root,
        upstream_root,
        adapter,
        encoded,
        clips,
        graph,
        nodes,
        destination,
        fingerprint,
    )


def _selection_budget(resolved: Mapping[str, Any], candidate_count: int) -> int:
    configured = resolved["selection"].get("budget")
    budget = (
        int(configured)
        if configured is not None
        else int(np.floor(candidate_count * float(resolved["selection"]["ratio"]) + 0.5))
    )
    if not 0 < budget <= candidate_count:
        raise ValueError("selection budget must be within eligible candidate count")
    return budget


def _selector(resolved: Mapping[str, Any], context: CocoreObjectiveContext):
    if resolved["selection"]["strategy"] == "random":
        return SeededRandomSelector(context, seed=int(resolved["seed"]))
    return RandomMultiBranchSelector(context, seed=int(resolved["seed"]))


def _selection_report(
    resolved: Mapping[str, Any],
    graph: GraphData,
    result,
    context: CocoreObjectiveContext,
    *,
    initial: tuple[int, ...],
    scanned_count: int,
    upstream_fingerprint: str,
) -> dict[str, Any]:
    selected = np.asarray(result.selected_indices, dtype=np.int64)
    final_coverage = context.prototype_mass[selected].max(axis=0)
    target = context.prototype_mass.max(axis=0)
    relation_weight = float(resolved["objective"]["relation_weight"])
    redundancy_weight = float(resolved["objective"]["redundancy_weight"])
    return {
        "producer": "cocore_ablation",
        "schema_version": SCHEMA_VERSION,
        "selected_clips": len(result.selected_indices),
        "eligible_clips": len(graph.sample_ids),
        "number_of_scanned_clips": scanned_count,
        "configured_selection_ratio": float(resolved["selection"]["ratio"]),
        "selection_ratio": len(result.selected_indices) / len(graph.sample_ids),
        "reliability_metrics": list(resolved["reliability_metrics"]),
        "prototype_representation": resolved["prototypes"]["representation"],
        "use_assignment_confidence": resolved["prototypes"][
            "use_assignment_confidence"
        ],
        "use_stop_bucket": resolved["prototypes"]["use_stop_bucket"],
        "relation_type": resolved["objective"]["relation"],
        "relation_weight": relation_weight,
        "redundancy_weight": redundancy_weight,
        "selection_strategy": resolved["selection"]["strategy"],
        "use_coverage_seed": resolved["selection"]["use_coverage_seed"],
        "initial_set_size": len(initial),
        "coverage": {
            "target": [float(value) for value in target],
            "achieved": [float(value) for value in final_coverage],
        },
        "objective": {
            "relation": float(result.relation),
            "weighted_relation": relation_weight * float(result.relation),
            "redundancy": float(result.redundancy),
            "weighted_redundancy": redundancy_weight * float(result.redundancy),
            "total": float(result.objective_value),
        },
        "upstream_fingerprint": upstream_fingerprint,
        "algorithm": {
            "strategy": resolved["selection"]["strategy"],
            "seed": int(resolved["seed"]),
        },
        "branch_search": {
            "rounds": result.rounds,
            "evaluated_branches": result.evaluated_branches,
            "recombinations": result.recombinations,
            "committed_clips": result.committed_clips,
            "final_active_clips": result.final_active_clips,
        },
    }


def select_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
) -> Path:
    resolved = resolve_config(config)
    (
        root,
        upstream_root,
        _adapter,
        encoded,
        clips,
        graph,
        nodes,
        graph_root,
        graph_fingerprint,
    ) = graph_stage(
        resolved,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
    )
    resolved["output"]["directory"] = str(root)
    fingerprint = _selection_fingerprint(resolved, graph_fingerprint)
    destination = graph_root / _selection_directory(resolved, fingerprint)
    budget = _selection_budget(resolved, len(graph.sample_ids))

    def build(temporary: Path) -> None:
        context = CocoreObjectiveContext(
            graph,
            str(resolved["objective"]["relation"]),
            float(resolved["objective"]["relation_weight"]),
            redundancy_weight=float(resolved["objective"]["redundancy_weight"]),
            similarity_threshold=float(resolved["graph"]["similarity_threshold"]),
        )
        initial = initial_selection(
            context,
            budget=budget,
            use_coverage_seed=bool(resolved["selection"]["use_coverage_seed"]),
        )
        result = _selector(resolved, context).select(
            budget, initial_indices=initial
        )
        half_action_labels = np.load(
            graph_root / "half_action_labels.npy", allow_pickle=False
        )
        selected_rows, all_rows = cocore_pipeline._selection_rows(  # noqa: SLF001
            resolved,
            clips,
            graph,
            nodes,
            result,
            context,
            _prototype_metadata(graph_root),
            half_action_labels,
        )
        scan_manifest = json.loads(
            (upstream_root / "scan" / "manifest.json").read_text()
        )
        report = _selection_report(
            resolved,
            graph,
            result,
            context,
            initial=initial,
            scanned_count=int(scan_manifest["clips"]),
            upstream_fingerprint=encoded.fingerprint,
        )
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
                "producer": "cocore_ablation",
                "schema_version": SCHEMA_VERSION,
                "stage": "select",
                "fingerprint": fingerprint,
                "upstream_fingerprint": graph_fingerprint,
                "selected_clips": len(result.selected_indices),
                "budget": budget,
                "selection_strategy": resolved["selection"]["strategy"],
                "use_coverage_seed": resolved["selection"]["use_coverage_seed"],
            },
        )

    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=_SELECT_REQUIRED,
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)) and not force,
        build=build,
    )
    write_json(
        destination / "run_manifest.json",
        {
            "status": "complete",
            "producer": "cocore_ablation",
            "schema_version": SCHEMA_VERSION,
            "cocore_ablation_version": __version__,
            "fingerprint": stable_hash(
                {
                    "config": resolved,
                    "graph": graph_fingerprint,
                    "select": fingerprint,
                }
            ),
            "upstream_directory": str(upstream_root),
            "upstream_fingerprint": encoded.fingerprint,
            "graph_directory": graph_root.name,
            "graph_fingerprint": graph_fingerprint,
            "select_directory": destination.name,
            "select_fingerprint": fingerprint,
        },
    )
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    write_json(
        destination / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cocore_ablation_version": __version__,
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
    return select_stage(
        config,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
    )


def _load_selected_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"selected manifest line {line_number} is invalid"
                ) from error
            if not isinstance(payload, dict) or not isinstance(
                payload.get("sample_id"), str
            ):
                raise ValueError("selected manifest row is invalid")
            rows.append(payload)
    return rows


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _edge_tables_match(actual, expected) -> bool:
    return (
        actual.edge_type == expected.edge_type
        and np.array_equal(actual.source, expected.source)
        and np.array_equal(actual.target, expected.target)
        and np.allclose(actual.weight, expected.weight, rtol=1.0e-7, atol=1.0e-8)
    )


def _sparse_matrices_match(actual, expected) -> bool:
    if actual.shape != expected.shape:
        return False
    difference = (actual.astype(np.float64) - expected.astype(np.float64)).tocsr()
    return difference.nnz == 0 or bool(
        np.allclose(difference.data, 0.0, rtol=1.0e-7, atol=1.0e-8)
    )


def _validate_graph_replay(
    *,
    graph_root: Path,
    upstream_root: Path,
    adapter: Any,
    encoded: Any,
    resolved: Mapping[str, Any],
    graph: GraphData,
    nodes: Mapping[str, np.ndarray],
) -> None:
    quality = resolved["quality"]
    components = compute_reliability(
        encoded.embeddings,
        encoded.state_sequences,
        encoded.action_sequences,
        encoded.visual_progress,
        knn=int(quality["knn"]),
        gripper_progress_weight=float(quality["gripper_progress_weight"]),
        visual_progress_weight=float(quality["visual_progress_weight"]),
        noop_threshold=float(quality["noop_threshold"]),
        gripper_action_index=int(quality["gripper_action_index"]),
        min_reliability=float(quality["min_reliability"]),
        reliability_metrics=("support", "progress"),
    )
    reliability = fuse_reliability(
        components.support,
        components.progress,
        resolved["reliability_metrics"],
        min_reliability=float(quality["min_reliability"]),
    )
    prototype_config = resolved["prototypes"]
    visual_dim = int(resolved["encoding"]["visual_dim"])
    pca_components = cocore_pipeline._load_visual_pca_components(  # noqa: SLF001
        upstream_root / "encode", visual_dim=visual_dim
    )
    hierarchy = build_hierarchical_motion_prototypes(
        adapter,
        encoded.clips,
        encoded.visual_half_embeddings,
        pca_components=pca_components,
        visual_dim=visual_dim,
        frame_cache_dir=upstream_root / "encode" / "frame_embeddings",
        batch_size=int(prototype_config["batch_size"]),
        max_iter=int(prototype_config["max_iter"]),
        tol=float(prototype_config["tol"]),
        seed=int(resolved["seed"]),
        max_episodes=resolved["runtime"].get("max_episodes"),
        num_workers=int(resolved["runtime"].get("num_workers", 0)),
        num_threads=int(prototype_config["num_threads"]),
        use_stop_bucket=bool(prototype_config["use_stop_bucket"]),
        profile="libero",
        representation=str(prototype_config["representation"]),
        use_assignment_confidence=bool(
            prototype_config["use_assignment_confidence"]
        ),
    )
    source = np.flatnonzero(hierarchy.eligible_mask).astype(np.int64)
    stored_source = cocore_pipeline._load_source_clip_indices(  # noqa: SLF001
        graph_root, len(encoded.clips)
    )
    if not np.array_equal(stored_source, source):
        raise ValueError("prototype source clips do not match graph replay")

    stored_catalog = json.loads((graph_root / "prototype_catalog.json").read_text())
    if stored_catalog != _catalog_payload(hierarchy, resolved):
        raise ValueError("prototype catalog does not match graph replay")
    replay_centers = hierarchy.prototypes.centers
    stored_centers = np.load(graph_root / "prototype_centers.npy", allow_pickle=False)
    if replay_centers is None or not np.allclose(
        stored_centers, replay_centers, rtol=1.0e-6, atol=1.0e-7
    ):
        raise ValueError("prototype centers do not match graph replay")
    stored_half_labels = np.load(
        graph_root / "half_action_labels.npy", allow_pickle=False
    )
    if not np.array_equal(stored_half_labels, hierarchy.half_action_labels[source]):
        raise ValueError("prototype half-action labels do not match graph replay")

    graph_config = resolved["graph"]
    replay_graph = build_graph(
        encoded.clips,
        encoded.embeddings,
        reliability,
        hierarchy.prototypes,
        included_indices=source,
        knn=int(graph_config["knn"]),
        similarity_threshold=float(graph_config["similarity_threshold"]),
        cooccurrence_max_gap=int(graph_config["cooccurrence_max_gap"]),
        normalize_prototype_relations=False,
    )
    expected_node_arrays = {
        "task_indices": replay_graph.task_indices,
        "reliability": replay_graph.reliability,
        "prototype_indices": replay_graph.prototype_indices,
        "prototype_weights": replay_graph.prototype_weights,
        "support": components.support[source],
        "progress": components.progress[source],
        "smoothness": components.smoothness[source],
        "noop_ratio": components.noop_ratio[source],
    }
    if set(nodes.files) != set(expected_node_arrays) or any(
        not np.allclose(nodes[name], expected, rtol=1.0e-7, atol=1.0e-8)
        for name, expected in expected_node_arrays.items()
    ):
        raise ValueError("graph node arrays do not match graph replay")
    if (
        graph.sample_ids != replay_graph.sample_ids
        or not _edge_tables_match(graph.sequence_edges, replay_graph.sequence_edges)
        or not _edge_tables_match(graph.similarity_edges, replay_graph.similarity_edges)
        or not _sparse_matrices_match(
            graph.transition_matrix, replay_graph.transition_matrix
        )
        or not _sparse_matrices_match(
            graph.cooccurrence_matrix, replay_graph.cooccurrence_matrix
        )
    ):
        raise ValueError("graph relations do not match graph replay")


def validate_output(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = Path(output_dir).expanduser()
    required = {
        name: result / name
        for name in (
            "selected_manifest.jsonl",
            "all_clips.parquet",
            "selection_report.json",
            "manifest.json",
            "run_manifest.json",
            "resolved_config.yaml",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f"cocore_ablation output is missing files: {missing}")
    stored = yaml.safe_load(required["resolved_config.yaml"].read_text())
    resolved = resolve_config(config if config is not None else stored)
    stored_resolved = resolve_config(stored)
    if stable_hash(resolved) != stable_hash(stored_resolved):
        raise ValueError("validation configuration does not match the stored run")

    run_manifest = json.loads(required["run_manifest.json"].read_text())
    select_manifest = json.loads(required["manifest.json"].read_text())
    if (
        run_manifest.get("producer") != "cocore_ablation"
        or run_manifest.get("schema_version") != SCHEMA_VERSION
        or select_manifest.get("producer") != "cocore_ablation"
        or select_manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("cocore_ablation manifest schema is incompatible")
    (
        _root,
        upstream_root,
        adapter,
        encoded,
        clips,
        graph,
        nodes,
        graph_root,
        graph_fingerprint,
    ) = graph_stage(resolved, output_dir=resolved["output"]["directory"], force=False)
    expected_select_fingerprint = _selection_fingerprint(resolved, graph_fingerprint)
    expected_run_fingerprint = stable_hash(
        {
            "config": resolved,
            "graph": graph_fingerprint,
            "select": expected_select_fingerprint,
        }
    )
    expected_result = graph_root / _selection_directory(
        resolved, expected_select_fingerprint
    )
    if expected_result.resolve() != result.resolve():
        raise ValueError("selection output path does not match its configuration")
    graph_manifest = json.loads((graph_root / "manifest.json").read_text())
    expected_graph_manifest_fields = {
        "producer": "cocore_ablation",
        "schema_version": SCHEMA_VERSION,
        "stage": "graph",
        "fingerprint": graph_fingerprint,
        "upstream_fingerprint": encoded.fingerprint,
    }
    if any(
        graph_manifest.get(key) != value
        for key, value in expected_graph_manifest_fields.items()
    ):
        raise ValueError("cocore_ablation artifact fingerprint is invalid")
    _validate_graph_replay(
        graph_root=graph_root,
        upstream_root=upstream_root,
        adapter=adapter,
        encoded=encoded,
        resolved=resolved,
        graph=graph,
        nodes=nodes,
    )

    budget = _selection_budget(resolved, len(graph.sample_ids))
    expected_select_manifest = {
        "status": "complete",
        "producer": "cocore_ablation",
        "schema_version": SCHEMA_VERSION,
        "stage": "select",
        "fingerprint": expected_select_fingerprint,
        "upstream_fingerprint": graph_fingerprint,
        "selected_clips": budget,
        "budget": budget,
        "selection_strategy": resolved["selection"]["strategy"],
        "use_coverage_seed": resolved["selection"]["use_coverage_seed"],
    }
    expected_run_manifest = {
        "status": "complete",
        "producer": "cocore_ablation",
        "schema_version": SCHEMA_VERSION,
        "cocore_ablation_version": __version__,
        "fingerprint": expected_run_fingerprint,
        "upstream_directory": str(upstream_root),
        "upstream_fingerprint": encoded.fingerprint,
        "graph_directory": graph_root.name,
        "graph_fingerprint": graph_fingerprint,
        "select_directory": expected_result.name,
        "select_fingerprint": expected_select_fingerprint,
    }
    if select_manifest != expected_select_manifest or run_manifest != expected_run_manifest:
        raise ValueError("cocore_ablation artifact fingerprint is invalid")

    context = CocoreObjectiveContext(
        graph,
        str(resolved["objective"]["relation"]),
        float(resolved["objective"]["relation_weight"]),
        redundancy_weight=float(resolved["objective"]["redundancy_weight"]),
        similarity_threshold=float(resolved["graph"]["similarity_threshold"]),
    )
    initial = initial_selection(
        context,
        budget=budget,
        use_coverage_seed=bool(resolved["selection"]["use_coverage_seed"]),
    )
    replayed = _selector(resolved, context).select(budget, initial_indices=initial)
    half_action_labels = np.load(
        graph_root / "half_action_labels.npy", allow_pickle=False
    )
    expected_selected_rows, expected_all_rows = cocore_pipeline._selection_rows(  # noqa: SLF001
        resolved,
        clips,
        graph,
        nodes,
        replayed,
        context,
        _prototype_metadata(graph_root),
        half_action_labels,
    )
    selected_rows = _load_selected_rows(required["selected_manifest.jsonl"])
    if selected_rows != _json_normalized(expected_selected_rows):
        raise ValueError(
            "selected manifest selection order or metadata does not match exact replay"
        )
    all_rows = pq.read_table(required["all_clips.parquet"]).to_pylist()
    expected_all_rows = sorted(expected_all_rows, key=lambda row: str(row["sample_id"]))
    if all_rows != _json_normalized(expected_all_rows):
        raise ValueError("all-clips artifact does not match exact selection replay")

    report = json.loads(required["selection_report.json"].read_text())
    scan_manifest = json.loads((upstream_root / "scan" / "manifest.json").read_text())
    expected_report = _selection_report(
        resolved,
        graph,
        replayed,
        context,
        initial=initial,
        scanned_count=int(scan_manifest["clips"]),
        upstream_fingerprint=encoded.fingerprint,
    )
    if report != _json_normalized(expected_report):
        raise ValueError("selection report does not match exact selection replay")
    objective = report.get("objective", {})
    expected_objective = {
        "relation": float(replayed.relation),
        "weighted_relation": float(resolved["objective"]["relation_weight"])
        * float(replayed.relation),
        "redundancy": float(replayed.redundancy),
        "weighted_redundancy": float(resolved["objective"]["redundancy_weight"])
        * float(replayed.redundancy),
        "total": float(replayed.objective_value),
    }
    if set(objective) != set(expected_objective) or any(
        not math.isclose(
            float(objective[key]), value, rel_tol=1.0e-9, abs_tol=1.0e-9
        )
        for key, value in expected_objective.items()
    ):
        raise ValueError("selection report objective does not match replay")
    if int(report.get("initial_set_size", -1)) != len(initial):
        raise ValueError("selection report initial set does not match replay")
    return {
        "status": "valid",
        "selected_clips": len(replayed.selected_indices),
        "selection_strategy": resolved["selection"]["strategy"],
    }
