"""Independent Cocore selection artifacts over shared RelCore stages."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq
import yaml
from scipy import sparse

from relcore.export import write_selection_outputs
from relcore.features.visual_encoder import VisualEncoder
from relcore.graph.prototypes import valid_prototype_assignments
from relcore.pipeline import (
    EncodedArtifact,
    encode_stage as relcore_encode_stage,
    graph_stage as relcore_graph_stage,
    scan_stage as relcore_scan_stage,
)
from relcore.schemas import ClipRecord, EdgeTable, GraphData
from relcore.utils.io import cache_is_valid, publish_stage, stable_hash, write_json

from cocore import __version__
from cocore.config import resolve_config, to_relcore_config
from cocore.objective import CocoreObjectiveContext
from cocore.selection import (
    LazyHeapSelector,
    build_max_coverage_seed,
)


GRAPH_DIRECTORY = "graph-12-motion-primitives"
RELIABILITY_METRICS = ("support", "progress")


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
) -> tuple[Path, object, EncodedArtifact]:
    resolved = resolve_config(config)
    translated = to_relcore_config(resolved)
    result = relcore_encode_stage(
        translated,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
    )
    _mark_shared_stage(result[0], "scan", "scan")
    _mark_shared_stage(result[0], "encode", "encode")
    return result


def graph_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: VisualEncoder | None = None,
) -> tuple[Path, object, list[ClipRecord], GraphData, str]:
    resolved = resolve_config(config)
    translated = to_relcore_config(resolved)
    result = relcore_graph_stage(
        translated,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
        reliability_metrics=RELIABILITY_METRICS,
    )
    _mark_shared_stage(result[0], "scan", "scan")
    _mark_shared_stage(result[0], "encode", "encode")
    _mark_shared_stage(result[0], GRAPH_DIRECTORY, "graph")
    return result


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


def _load_clips(path: Path) -> list[ClipRecord]:
    return [ClipRecord(**row) for row in pq.read_table(path).to_pylist()]


def _prototype_labels(graph_root: Path) -> tuple[str, ...]:
    payload = json.loads((graph_root / "prototype_catalog.json").read_text(encoding="utf-8"))
    assigned = sorted(
        (
            category
            for category in payload["categories"]
            if category["prototype_id"] is not None
        ),
        key=lambda category: int(category["prototype_id"]),
    )
    return tuple(str(category["label"]) for category in assigned)


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


def _selection_rows(
    resolved: Mapping[str, Any],
    clips: list[ClipRecord],
    graph: GraphData,
    nodes: Mapping[str, np.ndarray],
    result,
    context: CocoreObjectiveContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = {index: position for position, index in enumerate(result.selected_indices)}
    all_rows: list[dict[str, Any]] = []
    for index, clip in enumerate(clips):
        prototype_indices, prototype_weights = valid_prototype_assignments(
            graph.prototype_indices[index], graph.prototype_weights[index]
        )
        labels = [graph.prototype_labels[int(value)] for value in prototype_indices]
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
            "primary_prototype_label": labels[0],
            "prototype_labels": labels,
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
            resolved, clips, graph, graph_nodes, result, context
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
    selected_from_all = sorted(
        (row for row in all_rows if row["selected"]), key=lambda row: int(row["selection_order"])
    )
    if [row["sample_id"] for row in selected_from_all] != [
        row["sample_id"] for row in selected_rows
    ]:
        raise ValueError("selected manifest and all-clips rows disagree")
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
