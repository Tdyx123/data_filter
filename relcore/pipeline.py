"""Cached scan, encode, graph, and selection stages."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import sparse

from trajectory_data import DatasetAdapter, EpisodeRecord, create_dataset

from relcore import __version__
from relcore.config import resolve_config
from relcore.data.index import build_clip_records
from relcore.export import write_selection_outputs
from relcore.features.encoding import (
    EncodedClips,
    FrameCacheSummary,
    encode_dataset_from_frame_cache,
    fit_numeric_normalizers,
    populate_frame_feature_cache,
)
from relcore.features.frame_cache import FrameFeatureCache, directory_sha256
from relcore.features.normalization import RobustNormalizer
from relcore.features.visual_encoder import (
    DummyVisualEncoder,
    FrozenClipEncoder,
    VisualEncoder,
)
from relcore.graph import build_graph, discover_prototypes
from relcore.schemas import ClipRecord, EdgeTable, GraphData
from relcore.scoring import compute_reliability
from relcore.selection.greedy import ExactGreedySelector, SelectionResult
from relcore.selection.multibranch import MultiBranchSelector
from relcore.selection.objective import (
    ObjectiveContext,
    ObjectiveWeights,
)
from relcore.selection.quota import allocate_task_quotas
from relcore.utils.io import (
    cache_is_valid,
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


def selection_directory_name(ratio: float) -> str:
    percent_tag = format(float(ratio) * 100.0, ".12g").replace(".", "p")
    return f"select-top{percent_tag}pct"


def _tasks_hash(config: Mapping[str, Any]) -> str:
    return file_sha256(Path(str(config["dataset"].get("path", ""))) / "meta/tasks.jsonl")


def _fingerprint(
    stage: str,
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
    sections: tuple[str, ...],
    upstream: str = "",
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
    return stable_hash(
        {
            "version": __version__,
            "stage": stage,
            "adapter": adapter.fingerprint(),
            "tasks": _tasks_hash(config),
            "config": stage_config,
            "upstream": upstream,
        }
    )


def _total_fingerprint(adapter: DatasetAdapter, config: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "version": __version__,
            "adapter": adapter.fingerprint(),
            "tasks": _tasks_hash(config),
            "config": config,
        }
    )


def _manifest_fingerprint(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["fingerprint"])


def _configured_records(
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
) -> list[EpisodeRecord]:
    records = list(adapter.episodes())
    max_episodes = config["runtime"].get("max_episodes")
    return records[: int(max_episodes)] if max_episodes is not None else records


def _frame_cache_fingerprint(
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
) -> str:
    clip_length = int(config["clip"]["length"])
    usable = [record for record in _configured_records(adapter, config) if record.length >= clip_length]
    visual = dict(config["visual"])
    model_sha256 = (
        directory_sha256(str(visual["model"])) if visual["encoder"] == "clip" else None
    )
    return stable_hash(
        {
            "version": __version__,
            "adapter": adapter.fingerprint(),
            "tasks": _tasks_hash(config),
            "episodes": [(record.episode_id, record.length) for record in usable],
            "image_key": adapter.image_observation_keys[0],
            "visual": visual,
            "model_sha256": model_sha256,
        }
    )


def _encode_fingerprint(
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
    scan_fingerprint: str,
    frame_cache_fingerprint: str,
) -> str:
    stage_fingerprint = _fingerprint(
        "encode",
        adapter,
        config,
        ("dataset", "clip", "visual", "normalization", "relation", "runtime"),
        scan_fingerprint,
    )
    return stable_hash(
        {
            "stage_fingerprint": stage_fingerprint,
            "frame_cache_fingerprint": frame_cache_fingerprint,
        }
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
    *,
    frame_cache: FrameFeatureCache,
    frame_records: list[EpisodeRecord],
    frame_cache_fingerprint: str,
    frame_cache_summary: FrameCacheSummary,
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
    frame_files = frame_cache.publish_features(
        frame_records,
        temporary / "frame_features",
        output_dim=int(encoded.relation_encoder.projection_matrices["visual"].shape[0]),
    )
    write_json(
        temporary / "frame_features_index.json",
        {"files": frame_files, "episodes": len(frame_files)},
    )
    write_json(
        temporary / "manifest.json",
        {
            "status": "complete",
            "fingerprint": fingerprint,
            "clips": len(encoded.clips),
            "embedding_dim": int(encoded.embeddings.shape[1]),
            "encoded_episodes": len(frame_files),
            "frame_cache_fingerprint": frame_cache_fingerprint,
            "frame_cache_reused_episodes": frame_cache_summary.reused_episodes,
            "frame_cache_encoded_episodes": frame_cache_summary.encoded_episodes,
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
    frame_cache_fingerprint = _frame_cache_fingerprint(adapter, resolved)
    fingerprint = _encode_fingerprint(
        adapter,
        resolved,
        scan_fingerprint,
        frame_cache_fingerprint,
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
        records = _configured_records(adapter, resolved)
        frame_records = [
            record for record in records if record.length >= int(resolved["clip"]["length"])
        ]
        encoder = visual_encoder or _make_visual_encoder(resolved)
        frame_cache = FrameFeatureCache(
            root / ".relcore-cache" / "frame_features" / frame_cache_fingerprint,
            fingerprint=frame_cache_fingerprint,
        )
        frame_cache_summary = populate_frame_feature_cache(
            adapter,
            frame_records,
            encoder,
            frame_cache,
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
        )
        encoded = encode_dataset_from_frame_cache(
            adapter,
            records,
            frame_cache,
            clip_length=int(resolved["clip"]["length"]),
            clip_stride=int(resolved["clip"]["stride"]),
            projection_dim=int(resolved["relation"]["projection_dim"]),
            output_dim=int(resolved["relation"]["output_dim"]),
            lags=tuple(int(lag) for lag in resolved["relation"]["lags"]),
            seed=int(resolved["seed"]),
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
            action_normalizer=action_normalizer,
            state_normalizer=state_normalizer,
            visual_output_dim=int(encoder.output_dim),
        )
        if [clip.sample_id for clip in encoded.clips] != [clip.sample_id for clip in clips]:
            raise ValueError("encode clip order does not match scan clip index")
        _save_encoded(
            temporary,
            encoded,
            fingerprint,
            time.perf_counter() - started,
            frame_cache=frame_cache,
            frame_records=frame_records,
            frame_cache_fingerprint=frame_cache_fingerprint,
            frame_cache_summary=frame_cache_summary,
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
            "frame_features_index.json",
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
) -> tuple[Path, DatasetAdapter, list[ClipRecord], GraphData, str]:
    resolved = resolve_config(config)
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
    )
    destination = root / "graph"

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
        )
        prototype_config = resolved["prototypes"]
        prototypes = discover_prototypes(
            encoded.embeddings,
            count=int(prototype_config["count"]),
            batch_size=int(prototype_config["batch_size"]),
            max_iter=int(prototype_config["max_iter"]),
            top_r=int(prototype_config["top_r"]),
            temperature=float(prototype_config["temperature"]),
            seed=int(resolved["seed"]),
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
        np.save(temporary / "prototype_centers.npy", prototypes.centers)
        _save_edge_table(temporary / "sequence_edges.npz", graph.sequence_edges)
        _save_edge_table(temporary / "similarity_edges.npz", graph.similarity_edges)
        sparse.save_npz(temporary / "transition_matrix.npz", graph.transition_matrix)
        sparse.save_npz(temporary / "cooccurrence_matrix.npz", graph.cooccurrence_matrix)
        write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "fingerprint": fingerprint,
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
            "prototype_centers.npy",
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
        all_rows.append(
            {
                **asdict(clip),
                "selected": selected,
                "reliability": float(graph.reliability[index]),
                "primary_prototype": int(graph.prototype_indices[index, 0]),
                "prototype_indices": [int(value) for value in graph.prototype_indices[index]],
                "prototype_weights": [float(value) for value in graph.prototype_weights[index]],
                "selection_order": position + 1 if position is not None else None,
                "selection_marginal_gain": (
                    float(result.marginal_gains[position]) if position is not None else None
                ),
                "final_add_gain": final_add_gain,
                "final_remove_loss": final_remove_loss,
                "branch_id": result.branch_id if selected else None,
            }
        )
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
) -> Path:
    resolved = resolve_config(config)
    scoped_ratio = (
        float(selection_output_ratio) if selection_output_ratio is not None else None
    )
    if scoped_ratio is not None and scoped_ratio != float(resolved["selection"]["ratio"]):
        raise ValueError("selection_output_ratio must match selection.ratio")
    seed_everything(int(resolved["seed"]))
    root, adapter, clips, graph, graph_fingerprint = graph_stage(
        resolved,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
    )
    resolved["output"]["directory"] = str(root)
    fingerprint = _fingerprint(
        "select",
        adapter,
        resolved,
        ("objective", "selection", "seed"),
        graph_fingerprint,
    )
    destination = root / (
        selection_directory_name(scoped_ratio) if scoped_ratio is not None else "select"
    )

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        result, branch_results, context, quotas = _select(resolved, graph)
        selected_rows, all_rows, report = _selection_rows(resolved, clips, graph, result, context)
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
        report["runtime_seconds"] = {
            stage: float(
                json.loads((root / stage / "manifest.json").read_text(encoding="utf-8")).get(
                    "runtime_seconds", 0.0
                )
            )
            for stage in ("scan", "encode", "graph")
        }
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
                "selected_clips": len(result.selected_indices),
                "budget": len(result.selected_indices),
            },
        )

    built = publish_stage(
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
    if scoped_ratio is None:
        for name in ("selected_manifest.jsonl", "all_clips.parquet", "selection_report.json"):
            target = root / name
            source = destination / name
            if built or not target.is_file() or file_sha256(target) != file_sha256(source):
                shutil.copy2(destination / name, target)
        write_json(
            root / "run_manifest.json",
            {
                "status": "complete",
                "fingerprint": _total_fingerprint(adapter, resolved),
                "stage_fingerprints": {
                    stage: _manifest_fingerprint(root / stage / "manifest.json")
                    for stage in ("scan", "encode", "graph", "select")
                },
            },
        )
    return root


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
            "relcore_version": __version__,
        },
    )
    return result


def validate_output(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser()
    required = (
        root / "selected_manifest.jsonl",
        root / "all_clips.parquet",
        root / "selection_report.json",
        root / "run_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"relcore output is missing files: {missing}")
    run_manifest = json.loads(required[3].read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise ValueError("run is not complete")
    stage_manifests: dict[str, dict[str, Any]] = {}
    for stage in ("scan", "encode", "graph", "select"):
        manifest_path = root / stage / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing stage manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"stage is not complete: {stage}")
        stage_manifests[stage] = manifest
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
            "frame_features_index.json",
        ),
        "graph": (
            "nodes.npz",
            "prototype_centers.npy",
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
        if not cache_is_valid(root / stage, str(stage_manifests[stage]["fingerprint"]), names):
            raise ValueError(f"missing stage artifact or invalid cache index: {stage}")
        recorded = run_manifest.get("stage_fingerprints", {}).get(stage)
        if recorded != stage_manifests[stage]["fingerprint"]:
            raise ValueError(f"run/stage fingerprint mismatch: {stage}")
    for name in ("selected_manifest.jsonl", "all_clips.parquet", "selection_report.json"):
        if file_sha256(root / name) != file_sha256(root / "select" / name):
            raise ValueError(f"published output does not match select artifact: {name}")
    selected_rows = [
        json.loads(line)
        for line in required[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_rows = pq.read_table(required[1]).to_pylist()
    report = json.loads(required[2].read_text(encoding="utf-8"))
    if [row["sample_id"] for row in all_rows] != sorted(row["sample_id"] for row in all_rows):
        raise ValueError("all_clips.parquet is not sorted by sample_id")
    if len({row["sample_id"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("selected manifest contains duplicate sample ids")
    if len(selected_rows) != int(report["selected_clips"]):
        raise ValueError("selected manifest count does not match report")
    if len(all_rows) != int(report["number_of_clips"]):
        raise ValueError("candidate clip count does not match report")
    select_manifest = stage_manifests["select"]
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
    if config is not None:
        resolved = resolve_config(config)
        resolved["output"]["directory"] = str(root)
        adapter = create_dataset(resolved["dataset"])
        expected_scan = _fingerprint(
            "scan",
            adapter,
            resolved,
            ("dataset", "clip", "normalization", "runtime"),
        )
        expected_frame_cache = _frame_cache_fingerprint(adapter, resolved)
        expected_encode = _encode_fingerprint(
            adapter,
            resolved,
            expected_scan,
            expected_frame_cache,
        )
        expected_graph = _fingerprint(
            "graph",
            adapter,
            resolved,
            ("quality", "prototypes", "graph"),
            expected_encode,
        )
        expected_select = _fingerprint(
            "select",
            adapter,
            resolved,
            ("objective", "selection", "seed"),
            expected_graph,
        )
        expected_stages = {
            "scan": expected_scan,
            "encode": expected_encode,
            "graph": expected_graph,
            "select": expected_select,
        }
        for stage, expected in expected_stages.items():
            actual = _manifest_fingerprint(root / stage / "manifest.json")
            if actual != expected:
                raise ValueError(f"{stage} fingerprint does not match the supplied config")
        if run_manifest.get("fingerprint") != _total_fingerprint(adapter, resolved):
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
