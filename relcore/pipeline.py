"""Cached scan, encode, graph, and selection stages."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import sparse

from trajectory_data import DatasetAdapter, create_dataset

from relcore import __version__
from relcore.config import resolve_config
from relcore.data.index import build_clip_records
from relcore.export import write_selection_outputs
from relcore.features.encoding import (
    EncodedClips,
    encode_dataset,
    fit_numeric_normalizers,
)
from relcore.features.normalization import RobustNormalizer
from relcore.features.visual_encoder import (
    DummyVisualEncoder,
    FrozenClipEncoder,
    VisualEncoder,
)
from relcore.graph import build_graph, discover_prototypes
from relcore.graph.motion_primitives import build_motion_primitive_prototypes
from relcore.graph.prototypes import valid_prototype_assignments
from relcore.schemas import ClipRecord, EdgeTable, GraphData
from relcore.scoring import (
    RELIABILITY_METRICS,
    compute_reliability,
    normalize_reliability_metrics,
    reliability_metric_mask,
)
from relcore.selection.greedy import ExactGreedySelector, SelectionResult
from relcore.selection.multibranch import MultiBranchSelector
from relcore.selection.objective import (
    ObjectiveContext,
    ObjectiveWeights,
)
from relcore.selection.prototype_gain import (
    PROTOTYPE_GAIN_METRICS,
    normalize_prototype_gain_metrics,
    prototype_gain_metric_mask,
)
from relcore.selection.quota import allocate_task_quotas
from relcore.utils.io import (
    cache_is_valid,
    directory_sha256,
    file_sha256,
    publish_stage,
    stable_hash,
    write_json,
)
from relcore.utils.random import seed_everything


@dataclass(frozen=True)
class EncodedArtifact:
    clips: list[ClipRecord]
    embeddings: np.ndarray
    state_sequences: np.ndarray
    action_sequences: np.ndarray
    visual_progress: np.ndarray
    fingerprint: str


def _output_root(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    return Path(
        output_dir if output_dir is not None else config["output"]["directory"]
    ).expanduser()


def graph_directory_name(
    reliability_metrics: Sequence[str],
    prototype_method: str = "kmeans",
) -> str:
    name = f"graph-{reliability_metric_mask(reliability_metrics)}"
    return name if prototype_method == "kmeans" else f"{name}-motion-primitives"


def selection_directory_name(
    reliability_metrics: Sequence[str],
    ratio: float | None = None,
    prototype_method: str = "kmeans",
    *,
    prototype_gain_metrics: Sequence[str] = PROTOTYPE_GAIN_METRICS,
    quota_mode: str | None = None,
) -> str:
    prefix = (
        f"select-r{reliability_metric_mask(reliability_metrics)}"
        f"-g{prototype_gain_metric_mask(prototype_gain_metrics)}"
    )
    if prototype_method != "kmeans":
        prefix = f"{prefix}-motion-primitives"
    if ratio is not None:
        percent_tag = format(float(ratio) * 100.0, ".12g").replace(".", "p")
        prefix = f"{prefix}-top{percent_tag}pct"
    if quota_mode is not None:
        if quota_mode not in {"proportional", "none"}:
            raise ValueError("quota_mode must be proportional or none")
        prefix = f"{prefix}-quota-{quota_mode}"
    return prefix


def _tasks_hash(config: Mapping[str, Any]) -> str:
    return file_sha256(Path(str(config["dataset"].get("path", ""))) / "meta/tasks.jsonl")


def _fingerprint(
    stage: str,
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
    sections: tuple[str, ...],
    upstream: str = "",
    parameters: Mapping[str, Any] | None = None,
) -> str:
    stage_config = {
        section: (
            {"max_episodes": config["runtime"].get("max_episodes")}
            if section == "runtime"
            else config[section]
        )
        for section in sections
    }
    if stage in {"encode", "graph", "select"}:
        stage_config["seed"] = config["seed"]
    payload = {
        "version": __version__,
        "stage": stage,
        "adapter": adapter.fingerprint(),
        "tasks": _tasks_hash(config),
        "config": stage_config,
        "upstream": upstream,
    }
    if parameters:
        payload["parameters"] = dict(parameters)
    return stable_hash(payload)


def _total_fingerprint(
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
    reliability_metrics: Sequence[str],
    prototype_gain_metrics: Sequence[str],
) -> str:
    return stable_hash(
        {
            "version": __version__,
            "adapter": adapter.fingerprint(),
            "tasks": _tasks_hash(config),
            "config": config,
            "reliability_metrics": list(normalize_reliability_metrics(reliability_metrics)),
            "prototype_gain_metrics": list(
                normalize_prototype_gain_metrics(prototype_gain_metrics)
            ),
        }
    )


def _manifest_fingerprint(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["fingerprint"])


def _encode_fingerprint(
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
    scan_fingerprint: str,
) -> str:
    visual = dict(config["visual"])
    model_sha256 = (
        directory_sha256(str(visual["model"])) if visual["encoder"] == "clip" else None
    )
    return _fingerprint(
        "encode",
        adapter,
        config,
        ("dataset", "clip", "visual", "normalization", "relation", "runtime"),
        scan_fingerprint,
        parameters={"visual_model_sha256": model_sha256},
    )


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _load_clips(path: Path) -> list[ClipRecord]:
    return [ClipRecord(**row) for row in pq.read_table(path).to_pylist()]


def scan_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, DatasetAdapter, list[ClipRecord], str]:
    resolved = resolve_config(config)
    root = _output_root(resolved, output_dir)
    adapter = create_dataset(resolved["dataset"])
    if len(adapter.image_observation_keys) != 1:
        raise ValueError("relcore requires exactly one configured image observation")
    if not adapter.vector_observation_keys:
        raise ValueError("relcore requires at least one vector observation")
    max_episodes_value = resolved["runtime"].get("max_episodes")
    max_episodes = int(max_episodes_value) if max_episodes_value is not None else None
    episodes = list(adapter.episodes())
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if any(record.task_index is None or record.task_name is None for record in episodes):
        raise ValueError("relcore requires task metadata for every episode")
    clips = build_clip_records(
        episodes,
        length=int(resolved["clip"]["length"]),
        stride=int(resolved["clip"]["stride"]),
    )
    fingerprint = _fingerprint(
        "scan", adapter, resolved, ("dataset", "clip", "normalization", "runtime")
    )
    destination = root / "scan"
    skipped_short = [
        record.episode_id for record in episodes if record.length < int(resolved["clip"]["length"])
    ]

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        action_normalizer, state_normalizer = fit_numeric_normalizers(
            adapter,
            episodes,
            epsilon=float(resolved["normalization"]["epsilon"]),
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
            max_episodes=max_episodes,
        )
        _write_parquet(temporary / "episodes.parquet", [asdict(record) for record in episodes])
        _write_parquet(temporary / "clips.parquet", [asdict(clip) for clip in clips])
        np.savez(
            temporary / "normalization.npz",
            state_median=state_normalizer.median,
            state_iqr=state_normalizer.iqr,
            action_median=action_normalizer.median,
            action_iqr=action_normalizer.iqr,
        )
        write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "fingerprint": fingerprint,
                "episodes": len(episodes),
                "scanned_episodes": len(episodes),
                "clips": len(clips),
                "dataset_summary": adapter.dataset_summary(),
                "skipped_short_episodes": skipped_short,
                "skipped_short_episode_count": len(skipped_short),
                "runtime_seconds": time.perf_counter() - started,
            },
        )

    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=("episodes.parquet", "clips.parquet", "normalization.npz"),
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    return root, adapter, _load_clips(destination / "clips.parquet"), fingerprint


def _make_visual_encoder(config: Mapping[str, Any]) -> VisualEncoder:
    name = str(config["visual"]["encoder"]).lower()
    if name == "dummy":
        return DummyVisualEncoder()
    if name == "clip":
        return FrozenClipEncoder(config["visual"])
    raise ValueError(f"unknown visual encoder {name!r}")


def _save_encoded(
    temporary: Path,
    encoded: EncodedClips,
    fingerprint: str,
    runtime_seconds: float,
) -> None:
    np.save(temporary / "embeddings.npy", encoded.embeddings)
    np.save(temporary / "raw_relations.npy", encoded.raw_relations)
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
            "fingerprint": fingerprint,
            "clips": len(encoded.clips),
            "embedding_dim": int(encoded.embeddings.shape[1]),
            "runtime_seconds": runtime_seconds,
        },
    )


def encode_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
) -> tuple[Path, DatasetAdapter, EncodedArtifact]:
    resolved = resolve_config(config)
    seed_everything(int(resolved["seed"]))
    root, adapter, clips, scan_fingerprint = scan_stage(
        resolved, output_dir=output_dir, force=force
    )
    fingerprint = _encode_fingerprint(
        adapter,
        resolved,
        scan_fingerprint,
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
        encoded = encode_dataset(
            adapter,
            encoder,
            clip_length=int(resolved["clip"]["length"]),
            clip_stride=int(resolved["clip"]["stride"]),
            projection_dim=int(resolved["relation"]["projection_dim"]),
            output_dim=int(resolved["relation"]["output_dim"]),
            lags=tuple(int(lag) for lag in resolved["relation"]["lags"]),
            seed=int(resolved["seed"]),
            epsilon=float(resolved["normalization"]["epsilon"]),
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
            max_episodes=resolved["runtime"].get("max_episodes"),
            progress_interval=100,
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
        )
        if [clip.sample_id for clip in encoded.clips] != [clip.sample_id for clip in clips]:
            raise ValueError("encode clip order does not match scan clip index")
        _save_encoded(
            temporary,
            encoded,
            fingerprint,
            time.perf_counter() - started,
        )

    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=(
            "embeddings.npy",
            "raw_relations.npy",
            "state_sequences.npy",
            "action_sequences.npy",
            "visual_progress.npy",
            "normalization.npz",
            "projection_matrices.npz",
            "relation_pca.npz",
        ),
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    artifact = EncodedArtifact(
        clips=clips,
        embeddings=np.load(destination / "embeddings.npy"),
        state_sequences=np.load(destination / "state_sequences.npy"),
        action_sequences=np.load(destination / "action_sequences.npy"),
        visual_progress=np.load(destination / "visual_progress.npy"),
        fingerprint=fingerprint,
    )
    return root, adapter, artifact


def _save_edge_table(path: Path, table: EdgeTable) -> None:
    np.savez(path, source=table.source, target=table.target, weight=table.weight)


def _load_edge_table(path: Path, edge_type: str) -> EdgeTable:
    payload = np.load(path)
    return EdgeTable(payload["source"], payload["target"], payload["weight"], edge_type)


def graph_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
    reliability_metrics: Sequence[str] = RELIABILITY_METRICS,
) -> tuple[Path, DatasetAdapter, list[ClipRecord], GraphData, str]:
    resolved = resolve_config(config)
    metrics = normalize_reliability_metrics(reliability_metrics)
    metric_mask = reliability_metric_mask(metrics)
    seed_everything(int(resolved["seed"]))
    root, adapter, encoded = encode_stage(
        resolved,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
    )
    fingerprint = _fingerprint(
        "graph",
        adapter,
        resolved,
        ("quality", "prototypes", "graph"),
        encoded.fingerprint,
        parameters={"reliability_metrics": metrics},
    )
    prototype_method = str(resolved["prototypes"]["method"])
    destination = root / graph_directory_name(metrics, prototype_method)

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
            reliability_metrics=metrics,
        )
        prototype_config = resolved["prototypes"]
        if prototype_method == "kmeans":
            prototypes = discover_prototypes(
                encoded.embeddings,
                count=int(prototype_config["count"]),
                batch_size=int(prototype_config["batch_size"]),
                max_iter=int(prototype_config["max_iter"]),
                top_r=int(prototype_config["top_r"]),
                temperature=float(prototype_config["temperature"]),
                seed=int(resolved["seed"]),
            )
            primitive_catalog = None
        else:
            prototypes, primitive_catalog = build_motion_primitive_prototypes(
                adapter,
                encoded.clips,
                max_episodes=resolved["runtime"].get("max_episodes"),
                num_workers=int(resolved["runtime"].get("num_workers", 0)),
            )
        graph_config = resolved["graph"]
        graph = build_graph(
            encoded.clips,
            encoded.embeddings,
            reliability.reliability,
            prototypes,
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
            support=reliability.support,
            progress=reliability.progress,
            smoothness=reliability.smoothness,
            noop_ratio=reliability.noop_ratio,
        )
        if prototypes.centers is not None:
            np.save(temporary / "prototype_centers.npy", prototypes.centers)
        else:
            assert primitive_catalog is not None
            write_json(temporary / "prototype_catalog.json", primitive_catalog.to_dict())
        _save_edge_table(temporary / "sequence_edges.npz", graph.sequence_edges)
        _save_edge_table(temporary / "similarity_edges.npz", graph.similarity_edges)
        sparse.save_npz(temporary / "transition_matrix.npz", graph.transition_matrix)
        sparse.save_npz(temporary / "cooccurrence_matrix.npz", graph.cooccurrence_matrix)
        write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "fingerprint": fingerprint,
                "upstream_fingerprint": encoded.fingerprint,
                "stage_directory": destination.name,
                "reliability_metrics": list(metrics),
                "reliability_mask": metric_mask,
                "prototype_method": prototype_method,
                "nodes": len(graph.sample_ids),
                "sequence_edges": len(graph.sequence_edges.source),
                "similarity_edges": len(graph.similarity_edges.source),
                "runtime_seconds": time.perf_counter() - started,
            },
        )

    prototype_artifact = (
        "prototype_centers.npy" if prototype_method == "kmeans" else "prototype_catalog.json"
    )
    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=(
            "nodes.npz",
            prototype_artifact,
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
    prototype_labels: tuple[str, ...] | None = None
    if prototype_method == "motion_primitives":
        catalog_payload = json.loads(
            (destination / "prototype_catalog.json").read_text(encoding="utf-8")
        )
        assigned = sorted(
            (
                category
                for category in catalog_payload["categories"]
                if category["prototype_id"] is not None
            ),
            key=lambda category: int(category["prototype_id"]),
        )
        prototype_labels = tuple(str(category["label"]) for category in assigned)
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
        prototype_labels=prototype_labels,
    )
    return root, adapter, encoded.clips, graph, fingerprint


def _objective_weights(config: Mapping[str, Any]) -> ObjectiveWeights:
    values = config["objective"]
    return ObjectiveWeights(
        node=float(values["node_weight"]),
        transition=float(values["transition_weight"]),
        cooccurrence=float(values["cooccurrence_weight"]),
        sequence=float(values["sequence_weight"]),
        redundancy=float(values["redundancy_weight"]),
    )


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


def _select(
    config: Mapping[str, Any],
    graph: GraphData,
    prototype_gain_metrics: Sequence[str],
) -> tuple[
    SelectionResult,
    list[SelectionResult],
    ObjectiveContext,
    dict[int, int] | None,
]:
    budget = _selection_budget(config, len(graph.sample_ids))
    selection = config["selection"]
    quotas = (
        allocate_task_quotas(
            graph.task_indices,
            budget=budget,
            minimum_per_task=int(selection["minimum_per_task"]),
        )
        if selection["quota_mode"] == "proportional"
        else None
    )
    context = ObjectiveContext(
        graph,
        _objective_weights(config),
        similarity_threshold=float(config["graph"]["similarity_threshold"]),
        prototype_gain_metrics=prototype_gain_metrics,
    )
    if selection["engine"] == "exact":
        result = ExactGreedySelector(context, quotas).select(budget)
        return result, [result], context, quotas
    local = selection["local_search"]
    selector = MultiBranchSelector(
        context,
        quotas,
        branches=int(selection["branches"]),
        seed=int(config["seed"]),
        seed_candidates=int(selection["seed_candidates"]),
        seed_similarity_threshold=float(selection["seed_similarity_threshold"]),
        transition_seed_threshold=float(selection["transition_seed_threshold"]),
        pairs_per_transition=int(selection["seed_pairs_per_transition"]),
        global_candidates=int(selection["global_candidates"]),
        residual_candidates=int(selection["residual_candidates"]),
        random_candidates=int(selection["random_candidates"]),
        local_search_enabled=bool(local["enabled"]),
        local_search_selected=int(local["max_selected_candidates"]),
        local_search_unselected=int(local["max_unselected_candidates"]),
        local_search_rounds=int(local["max_rounds"]),
    )
    result = selector.select(budget)
    return result, selector.branch_results, context, quotas


def _selection_rows(
    config: Mapping[str, Any],
    clips: list[ClipRecord],
    graph: GraphData,
    result: SelectionResult,
    context: ObjectiveContext,
    reliability_metrics: Sequence[str],
    prototype_gain_metrics: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    final_state = context.state_from_indices(result.selected_indices)
    selected_position = {index: position for position, index in enumerate(result.selected_indices)}
    all_rows: list[dict[str, Any]] = []
    for index, clip in enumerate(clips):
        selected = index in selected_position
        if selected:
            without_state = context.clone_state(final_state)
            context.remove_candidate(without_state, index)
            final_remove_loss = float(final_state.objective_value - without_state.objective_value)
            final_add_gain = None
        else:
            final_remove_loss = None
            final_add_gain = float(context.marginal_gain(final_state, index))
        position = selected_position.get(index)
        prototype_indices, prototype_weights = valid_prototype_assignments(
            graph.prototype_indices[index], graph.prototype_weights[index]
        )
        row = {
            **asdict(clip),
            "selected": selected,
            "reliability": float(graph.reliability[index]),
            "primary_prototype": int(prototype_indices[0]),
            "prototype_indices": [int(value) for value in prototype_indices],
            "prototype_weights": [float(value) for value in prototype_weights],
            "selection_order": position + 1 if position is not None else None,
            "selection_marginal_gain": (
                float(result.marginal_gains[position]) if position is not None else None
            ),
            "final_add_gain": final_add_gain,
            "final_remove_loss": final_remove_loss,
            "branch_id": result.branch_id if selected else None,
        }
        if graph.prototype_labels is not None:
            labels = [graph.prototype_labels[int(value)] for value in prototype_indices]
            row["primary_prototype_label"] = labels[0]
            row["prototype_labels"] = labels
        all_rows.append(row)
    dataset_name = str(config["dataset"]["name"])
    dataset_path = str(config["dataset"].get("path", ""))
    selected_rows: list[dict[str, Any]] = []
    for position, index in enumerate(result.selected_indices):
        row = dict(all_rows[index])
        row.update({"dataset_name": dataset_name, "dataset_path": dataset_path})
        row["marginal_gain"] = row["selection_marginal_gain"]
        selected_rows.append(row)
    breakdown = context.breakdown(final_state)
    task_counts = {
        str(task): sum(int(graph.task_indices[index]) == task for index in result.selected_indices)
        for task in sorted({int(value) for value in graph.task_indices})
    }
    report = {
        "number_of_episodes": len({clip.episode_id for clip in clips}),
        "number_of_clips": len(clips),
        "selected_clips": len(result.selected_indices),
        "selection_ratio": len(result.selected_indices) / len(clips),
        "reliability_metrics": list(normalize_reliability_metrics(reliability_metrics)),
        "prototype_gain_metrics": list(
            normalize_prototype_gain_metrics(prototype_gain_metrics)
        ),
        "prototype_method": str(config["prototypes"]["method"]),
        "objective": asdict(breakdown),
        "prototype_coverage": {
            str(index): float(value) for index, value in enumerate(final_state.prototype_coverage)
        },
        "task_counts": task_counts,
    }
    return selected_rows, all_rows, report


def select_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
    selection_output_ratio: float | None = None,
    selection_output_quota_mode: str | None = None,
    reliability_metrics: Sequence[str] = RELIABILITY_METRICS,
    prototype_gain_metrics: Sequence[str] = PROTOTYPE_GAIN_METRICS,
) -> Path:
    resolved = resolve_config(config)
    metrics = normalize_reliability_metrics(reliability_metrics)
    metric_mask = reliability_metric_mask(metrics)
    gain_metrics = normalize_prototype_gain_metrics(prototype_gain_metrics)
    gain_mask = prototype_gain_metric_mask(gain_metrics)
    scoped_ratio = (
        float(selection_output_ratio) if selection_output_ratio is not None else None
    )
    if scoped_ratio is not None and scoped_ratio != float(resolved["selection"]["ratio"]):
        raise ValueError("selection_output_ratio must match selection.ratio")
    scoped_quota_mode = (
        str(selection_output_quota_mode)
        if selection_output_quota_mode is not None
        else None
    )
    if scoped_quota_mode is not None and scoped_quota_mode not in {"proportional", "none"}:
        raise ValueError("selection_output_quota_mode must be proportional or none")
    if (
        scoped_quota_mode is not None
        and scoped_quota_mode != str(resolved["selection"]["quota_mode"])
    ):
        raise ValueError("selection_output_quota_mode must match selection.quota_mode")
    seed_everything(int(resolved["seed"]))
    root, adapter, clips, graph, graph_fingerprint = graph_stage(
        resolved,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
        reliability_metrics=metrics,
    )
    resolved["output"]["directory"] = str(root)
    fingerprint = _fingerprint(
        "select",
        adapter,
        resolved,
        ("objective", "selection", "seed"),
        graph_fingerprint,
        parameters={"prototype_gain_metrics": gain_metrics},
    )
    prototype_method = str(resolved["prototypes"]["method"])
    graph_directory = graph_directory_name(metrics, prototype_method)
    selection_directory = selection_directory_name(
        metrics,
        scoped_ratio,
        prototype_method,
        prototype_gain_metrics=gain_metrics,
        quota_mode=scoped_quota_mode,
    )
    destination = root / selection_directory
    stage_directories = {
        "scan": "scan",
        "encode": "encode",
        "graph": graph_directory,
        "select": selection_directory,
    }

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        result, branch_results, context, quotas = _select(
            resolved,
            graph,
            gain_metrics,
        )
        selected_rows, all_rows, report = _selection_rows(
            resolved,
            clips,
            graph,
            result,
            context,
            metrics,
            gain_metrics,
        )
        report["quota_mode"] = str(resolved["selection"]["quota_mode"])
        report["task_quotas"] = (
            {str(task): value for task, value in quotas.items()} if quotas is not None else None
        )
        report["branches"] = [
            {
                "branch_id": branch.branch_id,
                "objective_value": branch.objective_value,
                "selected_clips": len(branch.selected_indices),
            }
            for branch in branch_results
        ]
        report["winning_branch"] = result.branch_id
        report["local_search"] = {
            "enabled": bool(resolved["selection"]["local_search"]["enabled"]),
            "accepted_swaps": len(result.local_swaps),
            "swaps": [
                {
                    "removed_sample_id": graph.sample_ids[swap.removed_index],
                    "added_sample_id": graph.sample_ids[swap.added_index],
                    "improvement": swap.improvement,
                }
                for swap in result.local_swaps
            ],
        }
        scan_manifest = json.loads((root / "scan" / "manifest.json").read_text(encoding="utf-8"))
        report["number_of_episodes"] = int(scan_manifest["episodes"])
        report["skipped_short_episodes"] = scan_manifest.get("skipped_short_episodes", [])
        report["runtime_seconds"] = {}
        for stage in ("scan", "encode", "graph"):
            stage_manifest = root / stage_directories[stage] / "manifest.json"
            report["runtime_seconds"][stage] = float(
                json.loads(stage_manifest.read_text(encoding="utf-8")).get(
                    "runtime_seconds", 0.0
                )
            )
        report["runtime_seconds"]["select"] = time.perf_counter() - started
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
                "fingerprint": fingerprint,
                "upstream_fingerprint": graph_fingerprint,
                "stage_directory": selection_directory,
                "reliability_metrics": list(metrics),
                "reliability_mask": metric_mask,
                "prototype_gain_metrics": list(gain_metrics),
                "prototype_gain_mask": gain_mask,
                "prototype_method": prototype_method,
                "selected_clips": len(result.selected_indices),
                "budget": len(result.selected_indices),
            },
        )

    publish_stage(
        destination,
        fingerprint=fingerprint,
        required=(
            "selected_manifest.jsonl",
            "all_clips.parquet",
            "selection_report.json",
        ),
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    write_json(
        destination / "run_manifest.json",
        {
            "status": "complete",
            "fingerprint": _total_fingerprint(
                adapter,
                resolved,
                metrics,
                gain_metrics,
            ),
            "reliability_metrics": list(metrics),
            "reliability_mask": metric_mask,
            "prototype_gain_metrics": list(gain_metrics),
            "prototype_gain_mask": gain_mask,
            "prototype_method": prototype_method,
            "selection_output_ratio": scoped_ratio,
            "selection_output_quota_mode": scoped_quota_mode,
            "stage_directories": stage_directories,
            "stage_fingerprints": {
                stage: _manifest_fingerprint(root / directory / "manifest.json")
                for stage, directory in stage_directories.items()
            },
        },
    )
    return destination


def run_pipeline(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
    selection_output_ratio: float | None = None,
    selection_output_quota_mode: str | None = None,
    reliability_metrics: Sequence[str] = RELIABILITY_METRICS,
    prototype_gain_metrics: Sequence[str] = PROTOTYPE_GAIN_METRICS,
) -> Path:
    resolved = resolve_config(config)
    root = _output_root(resolved, output_dir)
    resolved["output"]["directory"] = str(root)
    result = select_stage(
        resolved,
        output_dir=root,
        force=force,
        visual_encoder=visual_encoder,
        selection_output_ratio=selection_output_ratio,
        selection_output_quota_mode=selection_output_quota_mode,
        reliability_metrics=reliability_metrics,
        prototype_gain_metrics=prototype_gain_metrics,
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
            "relcore_version": __version__,
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
    required = (
        result / "selected_manifest.jsonl",
        result / "all_clips.parquet",
        result / "selection_report.json",
        result / "run_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"relcore output is missing files: {missing}")
    run_manifest = json.loads(required[3].read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise ValueError("run is not complete")
    prototype_method = str(run_manifest.get("prototype_method", "kmeans"))
    if prototype_method not in {"kmeans", "motion_primitives"}:
        raise ValueError("run manifest prototype_method is invalid")
    recorded_metrics = run_manifest.get("reliability_metrics")
    try:
        metrics = normalize_reliability_metrics(recorded_metrics)
    except ValueError as error:
        raise ValueError(f"run manifest reliability_metrics {error}") from error
    if list(metrics) != recorded_metrics:
        raise ValueError("run manifest reliability_metrics are not in canonical order")
    metric_mask = reliability_metric_mask(metrics)
    if int(run_manifest.get("reliability_mask", -1)) != metric_mask:
        raise ValueError("run manifest reliability_mask does not match reliability_metrics")
    recorded_gain_metrics = run_manifest.get("prototype_gain_metrics")
    try:
        gain_metrics = normalize_prototype_gain_metrics(recorded_gain_metrics)
    except ValueError as error:
        raise ValueError(f"run manifest prototype_gain_metrics {error}") from error
    if list(gain_metrics) != recorded_gain_metrics:
        raise ValueError(
            "run manifest prototype_gain_metrics are not in canonical order"
        )
    gain_mask = prototype_gain_metric_mask(gain_metrics)
    if int(run_manifest.get("prototype_gain_mask", -1)) != gain_mask:
        raise ValueError(
            "run manifest prototype_gain_mask does not match prototype_gain_metrics"
        )
    scoped_ratio = run_manifest.get("selection_output_ratio")
    if scoped_ratio is not None:
        try:
            scoped_ratio = float(scoped_ratio)
        except (TypeError, ValueError) as error:
            raise ValueError("run manifest selection_output_ratio is invalid") from error
    scoped_quota_mode = run_manifest.get("selection_output_quota_mode")
    if scoped_quota_mode is not None and (
        not isinstance(scoped_quota_mode, str)
        or scoped_quota_mode not in {"proportional", "none"}
    ):
        raise ValueError("run manifest selection_output_quota_mode is invalid")
    expected_directories = {
        "scan": "scan",
        "encode": "encode",
        "graph": graph_directory_name(metrics, prototype_method),
        "select": selection_directory_name(
            metrics,
            scoped_ratio,
            prototype_method,
            prototype_gain_metrics=gain_metrics,
            quota_mode=scoped_quota_mode,
        ),
    }
    stage_directories = run_manifest.get("stage_directories")
    if stage_directories != expected_directories:
        raise ValueError("run manifest stage_directories do not match metric parameters")
    if result.name != expected_directories["select"]:
        raise ValueError("selection output directory does not match metric parameters")
    stage_paths = {stage: root / directory for stage, directory in expected_directories.items()}
    stage_manifests: dict[str, dict[str, Any]] = {}
    for stage in ("scan", "encode", "graph", "select"):
        manifest_path = stage_paths[stage] / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing stage manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"stage is not complete: {stage}")
        stage_manifests[stage] = manifest
    prototype_artifact = (
        "prototype_centers.npy" if prototype_method == "kmeans" else "prototype_catalog.json"
    )
    stage_required = {
        "scan": ("episodes.parquet", "clips.parquet", "normalization.npz"),
        "encode": (
            "embeddings.npy",
            "raw_relations.npy",
            "state_sequences.npy",
            "action_sequences.npy",
            "visual_progress.npy",
            "normalization.npz",
            "projection_matrices.npz",
            "relation_pca.npz",
        ),
        "graph": (
            "nodes.npz",
            prototype_artifact,
            "sequence_edges.npz",
            "similarity_edges.npz",
            "transition_matrix.npz",
            "cooccurrence_matrix.npz",
        ),
        "select": (
            "selected_manifest.jsonl",
            "all_clips.parquet",
            "selection_report.json",
        ),
    }
    for stage, names in stage_required.items():
        if not cache_is_valid(
            stage_paths[stage],
            str(stage_manifests[stage]["fingerprint"]),
            names,
        ):
            raise ValueError(f"missing stage artifact or invalid cache index: {stage}")
        recorded = run_manifest.get("stage_fingerprints", {}).get(stage)
        if recorded != stage_manifests[stage]["fingerprint"]:
            raise ValueError(f"run/stage fingerprint mismatch: {stage}")
    expected_upstreams = {
        "graph": stage_manifests["encode"]["fingerprint"],
        "select": stage_manifests["graph"]["fingerprint"],
    }
    for stage, expected_upstream in expected_upstreams.items():
        if stage_manifests[stage].get("upstream_fingerprint") != expected_upstream:
            raise ValueError(f"{stage} manifest upstream_fingerprint mismatch")
    for stage in ("graph", "select"):
        manifest = stage_manifests[stage]
        if manifest.get("reliability_metrics") != list(metrics):
            raise ValueError(f"{stage} manifest reliability_metrics mismatch")
        if int(manifest.get("reliability_mask", -1)) != metric_mask:
            raise ValueError(f"{stage} manifest reliability_mask mismatch")
        if manifest.get("stage_directory") != expected_directories[stage]:
            raise ValueError(f"{stage} manifest stage_directory mismatch")
        if str(manifest.get("prototype_method", "kmeans")) != prototype_method:
            raise ValueError(f"{stage} manifest prototype_method mismatch")
    select_manifest = stage_manifests["select"]
    if select_manifest.get("prototype_gain_metrics") != list(gain_metrics):
        raise ValueError("select manifest prototype_gain_metrics mismatch")
    if int(select_manifest.get("prototype_gain_mask", -1)) != gain_mask:
        raise ValueError("select manifest prototype_gain_mask mismatch")
    selected_rows = [
        json.loads(line)
        for line in required[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_rows = pq.read_table(required[1]).to_pylist()
    report = json.loads(required[2].read_text(encoding="utf-8"))
    reported_metrics = report.get("reliability_metrics")
    try:
        normalized_reported_metrics = list(normalize_reliability_metrics(reported_metrics))
    except ValueError as error:
        raise ValueError(f"selection report reliability_metrics {error}") from error
    if reported_metrics != normalized_reported_metrics:
        raise ValueError("selection report reliability_metrics are not in canonical order")
    if tuple(normalized_reported_metrics) != metrics:
        raise ValueError("selection report reliability_metrics do not match the run manifest")
    reported_gain_metrics = report.get("prototype_gain_metrics")
    try:
        normalized_reported_gain_metrics = list(
            normalize_prototype_gain_metrics(reported_gain_metrics)
        )
    except ValueError as error:
        raise ValueError(
            f"selection report prototype_gain_metrics {error}"
        ) from error
    if reported_gain_metrics != normalized_reported_gain_metrics:
        raise ValueError(
            "selection report prototype_gain_metrics are not in canonical order"
        )
    if tuple(normalized_reported_gain_metrics) != gain_metrics:
        raise ValueError(
            "selection report prototype_gain_metrics do not match the run manifest"
        )
    if str(report.get("prototype_method", "kmeans")) != prototype_method:
        raise ValueError("selection report prototype_method does not match the run manifest")
    if [row["sample_id"] for row in all_rows] != sorted(row["sample_id"] for row in all_rows):
        raise ValueError("all_clips.parquet is not sorted by sample_id")
    if len({row["sample_id"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("selected manifest contains duplicate sample ids")
    if len(selected_rows) != int(report["selected_clips"]):
        raise ValueError("selected manifest count does not match report")
    if len(all_rows) != int(report["number_of_clips"]):
        raise ValueError("candidate clip count does not match report")
    if (
        len(selected_rows) != int(select_manifest.get("selected_clips", -1))
        or len(selected_rows) != int(select_manifest.get("budget", -1))
    ):
        raise ValueError("selected manifest count does not match the recorded budget")
    selected_ids = {row["sample_id"] for row in selected_rows}
    parquet_selected = {row["sample_id"] for row in all_rows if row["selected"]}
    if selected_ids != parquet_selected:
        raise ValueError("selected manifest and all_clips selection flags disagree")
    reported_counts = {str(key): int(value) for key, value in report["task_counts"].items()}
    counts: dict[str, int] = {task: 0 for task in reported_counts}
    required_fields = {
        "sample_id",
        "dataset_name",
        "dataset_path",
        "episode_id",
        "task_index",
        "task_name",
        "start_step",
        "end_step",
        "length",
        "reliability",
        "primary_prototype",
        "prototype_indices",
        "prototype_weights",
        "branch_id",
        "selection_order",
        "marginal_gain",
    }
    prototype_labels: tuple[str, ...] | None = None
    if prototype_method == "motion_primitives":
        required_fields |= {"primary_prototype_label", "prototype_labels"}
        catalog = json.loads(
            (stage_paths["graph"] / "prototype_catalog.json").read_text(encoding="utf-8")
        )
        if catalog.get("method") != "motion_primitives":
            raise ValueError("motion primitive catalog method is invalid")
        assigned = sorted(
            (
                category
                for category in catalog.get("categories", [])
                if category.get("prototype_id") is not None
            ),
            key=lambda category: int(category["prototype_id"]),
        )
        if [int(category["prototype_id"]) for category in assigned] != list(range(len(assigned))):
            raise ValueError("motion primitive catalog prototype ids are not contiguous")
        prototype_labels = tuple(str(category["label"]) for category in assigned)
    for row in selected_rows:
        missing_fields = sorted(required_fields - row.keys())
        if missing_fields:
            raise ValueError(f"selected manifest row is missing fields: {missing_fields}")
        if int(row["end_step"]) - int(row["start_step"]) + 1 != int(row["length"]):
            raise ValueError(f"invalid inclusive clip boundary: {row['sample_id']}")
        expected_id = (
            f"ep{int(row['episode_id']):06d}_fragment_"
            f"{int(row['start_step']):06d}_{int(row['end_step']):06d}"
        )
        if row["sample_id"] != expected_id:
            raise ValueError(f"non-canonical sample id: {row['sample_id']}")
        row_indices = [int(value) for value in row["prototype_indices"]]
        row_weights = [float(value) for value in row["prototype_weights"]]
        if not row_indices or len(row_indices) != len(row_weights):
            raise ValueError(f"invalid prototype assignments: {row['sample_id']}")
        if prototype_labels is not None:
            if len(row_indices) > 4 or any(
                index < 0 or index >= len(prototype_labels) for index in row_indices
            ):
                raise ValueError(f"invalid motion primitive ids: {row['sample_id']}")
            expected_labels = [prototype_labels[index] for index in row_indices]
            if row["prototype_labels"] != expected_labels:
                raise ValueError(f"motion primitive labels disagree: {row['sample_id']}")
            if row["primary_prototype_label"] != expected_labels[0]:
                raise ValueError(f"primary motion primitive label disagrees: {row['sample_id']}")
        task = str(row["task_index"])
        counts[task] = counts.get(task, 0) + 1
    if counts != reported_counts:
        raise ValueError("selected task counts do not match report")
    quota_mode = str(report.get("quota_mode", "proportional"))
    if quota_mode == "proportional":
        task_quotas = report.get("task_quotas")
        if not isinstance(task_quotas, dict):
            raise ValueError("proportional selection report has no task quotas")
        quotas = {str(key): int(value) for key, value in task_quotas.items()}
        if counts != quotas:
            raise ValueError("selected task counts do not fill the hard quotas")
    elif quota_mode == "none":
        if report.get("task_quotas") is not None:
            raise ValueError("global selection report must not contain task quotas")
        if sum(counts.values()) != len(selected_rows):
            raise ValueError("global selected task counts do not match selected clips")
    else:
        raise ValueError(f"unknown selection quota mode: {quota_mode}")
    if scoped_quota_mode is not None and quota_mode != scoped_quota_mode:
        raise ValueError("selection report quota mode does not match the run manifest")
    if config is not None:
        resolved = resolve_config(config)
        if str(resolved["prototypes"]["method"]) != prototype_method:
            raise ValueError("run prototype_method does not match the supplied config")
        if quota_mode != str(resolved["selection"]["quota_mode"]):
            raise ValueError("selection report quota mode does not match the supplied config")
        resolved["output"]["directory"] = str(root)
        adapter = create_dataset(resolved["dataset"])
        expected_scan = _fingerprint(
            "scan",
            adapter,
            resolved,
            ("dataset", "clip", "normalization", "runtime"),
        )
        expected_encode = _encode_fingerprint(
            adapter,
            resolved,
            expected_scan,
        )
        expected_graph = _fingerprint(
            "graph",
            adapter,
            resolved,
            ("quality", "prototypes", "graph"),
            expected_encode,
            parameters={"reliability_metrics": metrics},
        )
        expected_select = _fingerprint(
            "select",
            adapter,
            resolved,
            ("objective", "selection", "seed"),
            expected_graph,
            parameters={"prototype_gain_metrics": gain_metrics},
        )
        expected_stages = {
            "scan": expected_scan,
            "encode": expected_encode,
            "graph": expected_graph,
            "select": expected_select,
        }
        for stage, expected in expected_stages.items():
            actual = _manifest_fingerprint(stage_paths[stage] / "manifest.json")
            if actual != expected:
                raise ValueError(f"{stage} fingerprint does not match the supplied config")
        if run_manifest.get("fingerprint") != _total_fingerprint(
            adapter,
            resolved,
            metrics,
            gain_metrics,
        ):
            raise ValueError("run fingerprint does not match the supplied config")
        if len(selected_rows) != _selection_budget(resolved, len(all_rows)):
            raise ValueError("selected manifest count does not match the configured budget")
        episodes = {record.episode_id: record for record in adapter.episodes()}
        for row in selected_rows:
            episode_id = int(row["episode_id"])
            if episode_id not in episodes or int(row["end_step"]) >= episodes[episode_id].length:
                raise ValueError(f"selected clip cannot be located: {row['sample_id']}")
            record = episodes[episode_id]
            if (int(row["task_index"]), str(row["task_name"])) != (
                record.task_index,
                record.task_name,
            ):
                raise ValueError(f"selected clip task metadata disagrees: {row['sample_id']}")
    return {"status": "valid", "selected_clips": len(selected_rows)}
