"""Independent Cocore selection artifacts over shared RelCore stages."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import sparse
from trajectory_data import DatasetAdapter, EpisodeRecord, create_dataset

from relcore.export import write_selection_outputs
from relcore.features.visual_encoder import (
    DummyVisualEncoder,
    FrozenClipEncoder,
    VisualEncoder,
)
from relcore.graph.prototypes import valid_prototype_assignments
from relcore.scoring import compute_reliability
from relcore.schemas import ClipRecord, EdgeTable, GraphData
from relcore.utils.io import (
    cache_is_valid,
    directory_sha256,
    file_sha256,
    publish_stage,
    stable_hash,
    write_json,
)
from relcore.utils.random import seed_everything

from cocore import __version__
from cocore.config import resolve_config
from cocore.encoding import (
    CocoreEncodedArtifact,
    CocoreEncodedClips,
    encode_cocore_dataset,
)
from cocore.graph import SEQUENCE_ADJACENCY, build_graph
from cocore.index import CLIP_LENGTH, WINDOW_POLICY, build_clip_records
from cocore.objective import CocoreObjectiveContext
from cocore.prototypes import (
    FULL_KMEANS_MAX_TRAINING_COUNT,
    FULL_KMEANS_OPENMP_THREADS,
    MAX_VISUAL_CENTERS,
    MINIBATCH_KMEANS_OPENMP_THREADS,
    MIN_ACTION_COUNT,
    MIN_DISTANCE_WEIGHT,
    MIN_ACTION_FREQUENCY,
    STATE_THRESHOLD,
    TRAJECTORY_WINDOW_LENGTH,
    TRAJECTORY_WINDOW_POLICY,
    build_hierarchical_motion_prototypes,
    cluster_count_for_training_count,
    trajectory_window_starts,
)
from cocore.selection import (
    LazyHeapSelector,
    build_max_coverage_seed,
)
from cocore.timing import emit_completed_timing, timed_step


GRAPH_DIRECTORY = "graph-16-motion-hard-nearest-pca"
RELIABILITY_METRICS = ("support", "progress")
PROTOTYPE_SCHEMA_VERSION = 8
PROTOTYPE_STRATEGY = (
    "trajectory_sampled_retained_action_then_cropped_pca_half_visual_hybrid_kmeans_nearest"
)
PROTOTYPE_VISUAL_PROJECTION = "frame @ visual_pca.components[:, :frame_embedding_dim].T"
PROTOTYPE_VISUAL_NORMALIZATION = "l2_normalized_eight_frame_mean_after_projection"


def _number_tag(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def selection_directory_name(relation_type: str, relation_weight: float, ratio: float) -> str:
    return (
        f"select-{relation_type}-w{_number_tag(relation_weight)}-top{_number_tag(ratio * 100.0)}pct"
    )


def _output_root(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    return Path(
        output_dir if output_dir is not None else config["output"]["directory"]
    ).expanduser()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


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
    np.save(temporary / "visual_half_embeddings.npy", encoded.visual_half_embeddings)
    np.save(temporary / "state_sequences.npy", encoded.state_sequences)
    np.save(temporary / "action_sequences.npy", encoded.action_sequences)
    np.save(temporary / "visual_progress.npy", encoded.visual_progress)
    normalizers = encoded.numeric_normalizers
    state_lower = np.concatenate(
        [normalizers.observation_bounds[key][0] for key in normalizers.vector_keys]
    )
    state_upper = np.concatenate(
        [normalizers.observation_bounds[key][1] for key in normalizers.vector_keys]
    )
    offsets = [0]
    for key in normalizers.vector_keys:
        offsets.append(offsets[-1] + len(normalizers.observation_bounds[key][0]))
    np.savez(
        temporary / "numeric_normalizers.npz",
        vector_keys=np.asarray(normalizers.vector_keys),
        vector_offsets=np.asarray(offsets, dtype=np.int64),
        state_lower=state_lower,
        state_upper=state_upper,
        action_lower=normalizers.action_lower,
        action_upper=normalizers.action_upper,
        quantile_low=np.asarray(normalizers.quantile_low, dtype=np.float64),
        quantile_high=np.asarray(normalizers.quantile_high, dtype=np.float64),
        epsilon=np.asarray(normalizers.epsilon, dtype=np.float64),
    )
    projector = encoded.visual_projector
    np.savez(
        temporary / "visual_pca.npz",
        mean=projector.mean_,
        scale=projector.scale_,
        components=projector.components_,
        explained_variance_ratio=projector.explained_variance_ratio_,
    )
    index_entries = [
        {
            "episode_id": entry.episode_id,
            "path": f"frame_embeddings/{entry.filename}",
            "frames": entry.frames,
            "embedding_dim": entry.embedding_dim,
            "dtype": "float32",
            "sha256": entry.sha256,
        }
        for entry in encoded.frame_embeddings
    ]
    write_json(
        temporary / "frame_embeddings_index.json",
        {
            "version": 1,
            "dtype": "float32",
            "episodes": index_entries,
        },
    )
    candidate_count = len(encoded.clips)
    write_json(
        temporary / "manifest.json",
        {
            "status": "complete",
            "producer": "cocore",
            "cocore_version": __version__,
            "cocore_stage": "encode",
            "fingerprint": fingerprint,
            "clips": len(encoded.clips),
            "encoding": "quality_fusion",
            "visual_embedding_dim": projector.output_dim,
            "embedding_dim": int(encoded.embeddings.shape[1]),
            "visual_half_embedding_dim": int(encoded.visual_half_embeddings.shape[2]),
            "clip_length": CLIP_LENGTH,
            "window_policy": WINDOW_POLICY,
            "clip_anchors": [0, 7, 14],
            "visual_half_windows": [[0, 8], [7, 15]],
            "visual_half_encoding": "l2_normalized_eight_frame_mean",
            "counts": {
                "candidate_fragments": candidate_count,
                "pca_fit_fragments": encoded.pca_fit_fragment_count,
                "encoded_episodes": len(encoded.frame_embeddings),
                "encoded_frames": sum(entry.frames for entry in encoded.frame_embeddings),
            },
            "runtime_seconds": runtime_seconds,
        },
    )


def _load_visual_pca_components(encode_root: Path, *, visual_dim: int) -> np.ndarray:
    path = encode_root / "visual_pca.npz"
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "components" not in payload.files:
                raise ValueError("cocore visual PCA components are missing")
            components = np.asarray(payload["components"], dtype=np.float32)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("cocore visual PCA components are missing or invalid") from error
    if (
        components.ndim != 2
        or components.shape[0] == 0
        or components.shape[1] == 0
        or components.shape[0] > int(visual_dim)
        or not np.all(np.isfinite(components))
    ):
        raise ValueError("cocore visual PCA components are missing or invalid")
    return components


def _expected_frame_episodes(
    adapter: object,
    max_episodes: int | None,
) -> list[tuple[int, int]]:
    records = list(adapter.episodes())  # type: ignore[attr-defined]
    if max_episodes is not None:
        records = records[:max_episodes]
    return [(record.episode_id, record.length) for record in records]


def _validate_frame_embedding_cache(
    encode_root: Path,
    *,
    expected_episodes: list[tuple[int, int]] | None = None,
) -> None:
    index_path = encode_root / "frame_embeddings_index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cocore frame embedding index is missing or invalid") from error
    entries = payload.get("episodes")
    if payload.get("version") != 1 or payload.get("dtype") != "float32":
        raise ValueError("cocore frame embedding index schema is invalid")
    if not isinstance(entries, list) or not entries:
        raise ValueError("cocore frame embedding index has no episodes")
    actual_episodes: list[tuple[int, int]] = []
    indexed_paths: set[Path] = set()
    embedding_dim: int | None = None
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("cocore frame embedding index entry is invalid")
        episode_id = entry.get("episode_id")
        frames = entry.get("frames")
        dimension = entry.get("embedding_dim")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames <= 0
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            or entry.get("dtype") != "float32"
        ):
            raise ValueError("cocore frame embedding metadata is invalid")
        relative = Path(str(entry.get("path", "")))
        expected_relative = Path("frame_embeddings") / f"ep{episode_id:06d}.npy"
        if relative != expected_relative or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("cocore frame embedding path is invalid")
        path = encode_root / relative
        if path in indexed_paths or not path.is_file():
            raise ValueError("cocore frame embedding file is missing or duplicated")
        indexed_paths.add(path)
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError("cocore frame embedding hash does not match")
        try:
            values = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ValueError("cocore frame embedding file could not be loaded") from error
        if values.dtype != np.dtype(np.float32) or values.shape != (frames, dimension):
            raise ValueError("cocore frame embedding shape or dtype is invalid")
        for start in range(0, frames, 4096):
            if not np.all(np.isfinite(values[start : start + 4096])):
                raise ValueError("cocore frame embedding contains NaN or infinity")
        embedding_dim = dimension if embedding_dim is None else embedding_dim
        if dimension != embedding_dim:
            raise ValueError("cocore frame embedding dimension changed across episodes")
        actual_episodes.append((episode_id, frames))
    cache_root = encode_root / "frame_embeddings"
    if not cache_root.is_dir() or set(cache_root.glob("*.npy")) != indexed_paths:
        raise ValueError("cocore frame embedding file set does not match the index")
    if expected_episodes is not None and actual_episodes != expected_episodes:
        raise ValueError("cocore frame embedding episodes do not match the scan index")


def _validate_visual_half_embedding_cache(
    encode_root: Path,
    clips: list[ClipRecord],
) -> None:
    path = encode_root / "visual_half_embeddings.npy"
    try:
        visual_halves = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ValueError("cocore visual half embedding cache could not be loaded") from error
    if (
        visual_halves.dtype != np.dtype(np.float32)
        or visual_halves.ndim != 3
        or visual_halves.shape[:2] != (len(clips), 2)
        or visual_halves.shape[2] == 0
    ):
        raise ValueError("cocore visual half embedding shape or dtype is invalid")
    for start in range(0, len(visual_halves), 4096):
        block = visual_halves[start : start + 4096]
        if not np.all(np.isfinite(block)):
            raise ValueError("cocore visual half embedding contains NaN or infinity")
        norms = np.linalg.norm(block, axis=2)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 1.0e-8):
            raise ValueError("cocore visual half embedding has a non-positive norm")
        if not np.allclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("cocore visual half embeddings must be L2-normalized")

    clips_by_episode: dict[int, list[tuple[int, ClipRecord]]] = {}
    for clip_index, clip in enumerate(clips):
        if clip.length != 15 or clip.end_step - clip.start_step + 1 != 15 or clip.start_step < 0:
            raise ValueError("cocore visual half boundary is invalid")
        clips_by_episode.setdefault(clip.episode_id, []).append((clip_index, clip))

    for episode_id in sorted(clips_by_episode):
        frame_path = encode_root / "frame_embeddings" / f"ep{episode_id:06d}.npy"
        try:
            frame_values = np.load(frame_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ValueError("cocore visual half frame cache could not be loaded") from error
        if (
            frame_values.dtype != np.dtype(np.float32)
            or frame_values.ndim != 2
            or frame_values.shape[1] != visual_halves.shape[2]
        ):
            raise ValueError("cocore visual half frame cache shape or dtype is invalid")
        for clip_index, clip in clips_by_episode[episode_id]:
            if clip.end_step >= len(frame_values):
                raise ValueError("cocore visual half boundary exceeds frame cache")
            window = frame_values[clip.start_step : clip.end_step + 1]
            if window.shape != (15, visual_halves.shape[2]) or not np.all(np.isfinite(window)):
                raise ValueError("cocore visual half frame window is invalid")
            means = np.stack([window[:8].mean(axis=0), window[7:].mean(axis=0)])
            norms = np.linalg.norm(means, axis=1, keepdims=True)
            if not np.all(np.isfinite(norms)) or np.any(norms <= 1.0e-8):
                raise ValueError("cocore visual half frame mean has a non-positive norm")
            expected = (means / norms).astype(np.float32)
            if not np.allclose(
                visual_halves[clip_index],
                expected,
                rtol=1.0e-6,
                atol=1.0e-7,
            ):
                raise ValueError("visual half embeddings do not match frame cache")


def scan_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, object, list[ClipRecord], str]:
    resolved = resolve_config(config)
    root = _output_root(resolved, output_dir)
    adapter = create_dataset(resolved["dataset"])
    if len(adapter.image_observation_keys) != 1:
        raise ValueError("cocore requires exactly one configured image observation")
    if not adapter.vector_observation_keys:
        raise ValueError("cocore requires at least one vector observation")
    max_episodes_value = resolved["runtime"].get("max_episodes")
    max_episodes = int(max_episodes_value) if max_episodes_value is not None else None
    episodes = list(adapter.episodes())
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    clips = build_clip_records(episodes)
    fingerprint = stable_hash(
        {
            "producer": "cocore",
            "version": __version__,
            "stage": "scan",
            "adapter": adapter.fingerprint(),
            "dataset": resolved["dataset"],
            "runtime": {"max_episodes": max_episodes},
            "window_policy": WINDOW_POLICY,
            "clip_length": CLIP_LENGTH,
        }
    )
    destination = root / "scan"
    skipped_short = [record.episode_id for record in episodes if record.length < CLIP_LENGTH]

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        _write_parquet(temporary / "episodes.parquet", [asdict(record) for record in episodes])
        _write_parquet(temporary / "clips.parquet", [asdict(clip) for clip in clips])
        write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "producer": "cocore",
                "cocore_version": __version__,
                "cocore_stage": "scan",
                "fingerprint": fingerprint,
                "episodes": len(episodes),
                "scanned_episodes": len(episodes),
                "clips": len(clips),
                "dataset_summary": adapter.dataset_summary(),
                "skipped_short_episodes": skipped_short,
                "skipped_short_episode_count": len(skipped_short),
                "window_policy": WINDOW_POLICY,
                "clip_length": CLIP_LENGTH,
                "runtime_seconds": time.perf_counter() - started,
            },
        )

    started = time.perf_counter()
    built = publish_stage(
        destination,
        fingerprint=fingerprint,
        required=("episodes.parquet", "clips.parquet"),
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    if built:
        emit_completed_timing("scan", time.perf_counter() - started)
    return root, adapter, _load_clips(destination / "clips.parquet"), fingerprint


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
            "encoding": resolved["encoding"],
            "runtime": {"max_episodes": resolved["runtime"].get("max_episodes")},
            "seed": resolved["seed"],
            "window_policy": WINDOW_POLICY,
            "clip_length": CLIP_LENGTH,
            "visual_halves": {
                "windows": [[0, 8], [7, 15]],
                "encoding": "l2_normalized_eight_frame_mean",
            },
            "visual_model_sha256": model_sha256,
        }
    )
    destination = root / "encode"
    encoding_config = resolved["encoding"]

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        encoder = visual_encoder or _make_visual_encoder(resolved)
        encoded = encode_cocore_dataset(
            adapter,
            encoder,
            visual_dim=int(encoding_config["visual_dim"]),
            pca_fit_max_samples=encoding_config.get("pca_fit_max_samples"),
            quantile_low=float(encoding_config["quantile_low"]),
            quantile_high=float(encoding_config["quantile_high"]),
            epsilon=float(encoding_config["epsilon"]),
            seed=int(resolved["seed"]),
            frame_cache_dir=temporary / "frame_embeddings",
            num_workers=int(resolved["runtime"].get("num_workers", 0)),
            max_episodes=resolved["runtime"].get("max_episodes"),
            progress_interval=100,
            timing_callback=emit_completed_timing,
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
        "visual_half_embeddings.npy",
        "state_sequences.npy",
        "action_sequences.npy",
        "visual_progress.npy",
        "numeric_normalizers.npz",
        "visual_pca.npz",
        "frame_embeddings_index.json",
    )
    max_episodes_value = resolved["runtime"].get("max_episodes")
    max_episodes = int(max_episodes_value) if max_episodes_value is not None else None
    expected_frame_episodes = _expected_frame_episodes(adapter, max_episodes)
    resume = bool(resolved["runtime"].get("resume", True)) and not force
    aggregate_cache_valid = cache_is_valid(destination, fingerprint, required)
    frame_cache_valid = False
    if resume and aggregate_cache_valid:
        try:
            _validate_frame_embedding_cache(
                destination,
                expected_episodes=expected_frame_episodes,
            )
        except ValueError as error:
            raise FileExistsError(
                f"cocore frame embedding cache is incompatible: {destination}; pass --force"
            ) from error
        try:
            _validate_visual_half_embedding_cache(destination, clips)
        except ValueError as error:
            raise FileExistsError(
                f"cocore visual half embedding cache is incompatible: {destination}; pass --force"
            ) from error
        frame_cache_valid = True
    stage_started = time.perf_counter()
    built = publish_stage(
        destination,
        fingerprint=fingerprint,
        required=required,
        force=force,
        resume=resume and frame_cache_valid,
        build=build,
    )
    if built:
        emit_completed_timing("encode", time.perf_counter() - stage_started)
    _validate_frame_embedding_cache(
        destination,
        expected_episodes=expected_frame_episodes,
    )
    _validate_visual_half_embedding_cache(destination, clips)
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
    visual_dim = int(resolved["encoding"]["visual_dim"])
    pca_components = _load_visual_pca_components(root / "encode", visual_dim=visual_dim)
    prototype_config = resolved["prototypes"]
    prototype_fingerprint_config = {
        key: prototype_config[key] for key in ("method", "batch_size", "max_iter", "tol")
    }
    fingerprint = stable_hash(
        {
            "producer": "cocore",
            "version": __version__,
            "stage": "graph",
            "adapter": adapter.fingerprint(),
            "upstream": encoded.fingerprint,
            "quality": resolved["quality"],
            "prototypes": prototype_fingerprint_config,
            "graph": resolved["graph"],
            "seed": resolved["seed"],
            "max_episodes": resolved["runtime"].get("max_episodes"),
            "reliability_metrics": list(RELIABILITY_METRICS),
            "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
            "prototype_strategy": PROTOTYPE_STRATEGY,
            "sequence_adjacency": SEQUENCE_ADJACENCY,
        }
    )
    destination = root / GRAPH_DIRECTORY

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        quality_config = resolved["quality"]
        with timed_step("graph.reliability", emit_completed_timing):
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
        with timed_step("graph.prototypes", emit_completed_timing):
            hierarchy = build_hierarchical_motion_prototypes(
                adapter,
                encoded.clips,
                encoded.visual_half_embeddings,
                pca_components=pca_components,
                visual_dim=visual_dim,
                frame_cache_dir=root / "encode" / "frame_embeddings",
                batch_size=int(prototype_config["batch_size"]),
                max_iter=int(prototype_config["max_iter"]),
                tol=float(prototype_config["tol"]),
                seed=int(resolved["seed"]),
                max_episodes=resolved["runtime"].get("max_episodes"),
                num_workers=int(resolved["runtime"].get("num_workers", 0)),
                num_threads=int(prototype_config["num_threads"]),
                timing_callback=emit_completed_timing,
            )
        graph_config = resolved["graph"]
        with timed_step("graph.sparse_graph", emit_completed_timing):
            graph = build_graph(
                encoded.clips,
                encoded.embeddings,
                reliability.reliability,
                hierarchy.prototypes,
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
            support=reliability.support,
            progress=reliability.progress,
            smoothness=reliability.smoothness,
            noop_ratio=reliability.noop_ratio,
        )
        assert hierarchy.prototypes.centers is not None
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
                "prototype_strategy": PROTOTYPE_STRATEGY,
                "prototype_visual_dim": visual_dim,
                "prototype_visual_projection": PROTOTYPE_VISUAL_PROJECTION,
                "prototype_visual_normalization": PROTOTYPE_VISUAL_NORMALIZATION,
                "sequence_adjacency": SEQUENCE_ADJACENCY,
                "nodes": len(graph.sample_ids),
                "sequence_edges": len(graph.sequence_edges.source),
                "similarity_edges": len(graph.similarity_edges.source),
                "runtime_seconds": time.perf_counter() - started,
            },
        )

    stage_started = time.perf_counter()
    built = publish_stage(
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
    if built:
        emit_completed_timing("graph", time.perf_counter() - stage_started)
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


def _prototype_catalog(graph_root: Path) -> Mapping[str, Any]:
    payload = json.loads((graph_root / "prototype_catalog.json").read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("method") != "motion_primitives"
        or payload.get("schema_version") != PROTOTYPE_SCHEMA_VERSION
        or payload.get("strategy") != PROTOTYPE_STRATEGY
    ):
        raise ValueError("cocore prototype catalog schema is incompatible")
    return payload


def _prototype_labels(graph_root: Path) -> tuple[str, ...]:
    payload = _prototype_catalog(graph_root)
    assigned = sorted(
        payload["leaf_prototypes"],
        key=lambda leaf: int(leaf["prototype_id"]),
    )
    return tuple(str(leaf["label"]) for leaf in assigned)


def _leaf_prototype_metadata(graph_root: Path) -> tuple[Mapping[str, Any], ...]:
    payload = _prototype_catalog(graph_root)
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


def _validate_schema_eight_catalog(
    payload: Mapping[str, Any],
    *,
    expected_total_raw_actions: int,
) -> tuple[Mapping[str, Any], ...]:
    expected_constants = {
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
        "visual_projection": PROTOTYPE_VISUAL_PROJECTION,
        "visual_projection_centering": "none",
        "visual_projection_padding": "right_zero_to_128",
        "visual_half_encoding": "l2_normalized_mean_of_eight_projected_frames",
        "cluster_count": (
            "min(training_count, min(16, max(3, "
            "floor(2 * log2(training_count) - 16))))"
        ),
        "retention_weight": "0.5 + 0.5 * retained_atomic_ratio",
        "distance_quantiles": [0.1, 0.9],
        "distance_weight_range": [1.0, MIN_DISTANCE_WEIGHT],
        "duplicate_merge": "max + 0.5 * min",
    }
    if (
        payload.get("method") != "motion_primitives"
        or payload.get("schema_version") != PROTOTYPE_SCHEMA_VERSION
        or payload.get("strategy") != PROTOTYPE_STRATEGY
        or payload.get("constants") != expected_constants
        or not isinstance(payload.get("action_categories"), list)
        or not isinstance(payload.get("leaf_prototypes"), list)
    ):
        raise ValueError("hierarchical prototype catalog schema is invalid")
    total = payload.get("total_raw_actions")
    if isinstance(total, bool) or not isinstance(total, int) or total != expected_total_raw_actions:
        raise ValueError("hierarchical prototype raw action total is invalid")
    categories = payload["action_categories"]
    leaves = payload["leaf_prototypes"]
    if (
        not categories
        or not leaves
        or not all(isinstance(value, Mapping) for value in (*categories, *leaves))
    ):
        raise ValueError("hierarchical prototype catalog entries are invalid")

    labels = [category.get("label") for category in categories]
    counts = [category.get("raw_count") for category in categories]
    if (
        any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
        or labels.count("stop") != 1
    ):
        raise ValueError("hierarchical prototype action labels are invalid")
    if (
        any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts)
        or sum(counts) != total
    ):
        raise ValueError("hierarchical prototype action counts are invalid")
    if list(zip(counts, labels, strict=True)) != sorted(
        zip(counts, labels, strict=True), key=lambda item: (-item[0], item[1])
    ):
        raise ValueError("hierarchical prototype action order is invalid")

    threshold = max(MIN_ACTION_COUNT, math.ceil(MIN_ACTION_FREQUENCY * total))
    assigned_labels = [
        str(category["label"])
        for category in categories
        if category["label"] != "stop" and int(category["raw_count"]) >= threshold
    ] + ["stop"]
    action_ids = {label: index for index, label in enumerate(assigned_labels)}
    categories_by_id: dict[int, Mapping[str, Any]] = {}
    for category in categories:
        count = int(category["raw_count"])
        label = str(category["label"])
        proportion = category.get("raw_proportion")
        expected_proportion = count / total if total else 0.0
        if (
            isinstance(proportion, bool)
            or not isinstance(proportion, (int, float))
            or not math.isfinite(float(proportion))
            or not math.isclose(
                float(proportion), expected_proportion, rel_tol=0.0, abs_tol=1.0e-12
            )
        ):
            raise ValueError("hierarchical prototype action proportion is invalid")
        expected_retained = count >= threshold
        if category.get("retained") is not expected_retained:
            raise ValueError("hierarchical prototype retention threshold is invalid")
        expected_action_id = action_ids.get(label)
        action_id_value = category.get("action_id")
        if (
            action_id_value != expected_action_id
            or isinstance(action_id_value, bool)
            or (action_id_value is not None and not isinstance(action_id_value, int))
        ):
            raise ValueError("hierarchical prototype action id is invalid")
        if expected_action_id is not None:
            categories_by_id[expected_action_id] = category

        expected_training = count if label == "stop" or expected_retained else 0
        training = category.get("training_count")
        if (
            isinstance(training, bool)
            or not isinstance(training, int)
            or training != expected_training
        ):
            raise ValueError("hierarchical prototype training count is invalid")

    if sorted(categories_by_id) != list(range(len(categories_by_id))):
        raise ValueError("hierarchical action ids are not contiguous")
    for action_id, category in categories_by_id.items():
        training_count = int(category["training_count"])
        expected_requested = (
            cluster_count_for_training_count(training_count) if training_count > 0 else None
        )
        requested = category.get("requested_centers")
        if (
            requested != expected_requested
            or isinstance(requested, bool)
            or (requested is not None and not isinstance(requested, int))
        ):
            raise ValueError("hierarchical prototype requested center count is invalid")
        expected_actual = expected_requested or 0
        actual = category.get("actual_centers")
        if actual != expected_actual or isinstance(actual, bool) or not isinstance(actual, int):
            raise ValueError("hierarchical prototype leaf count is invalid")
        lower = category.get("nearest_distance_q10")
        upper = category.get("nearest_distance_q90")
        if training_count > 0:
            if (
                isinstance(lower, bool)
                or not isinstance(lower, (int, float))
                or isinstance(upper, bool)
                or not isinstance(upper, (int, float))
                or not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or float(lower) < 0.0
                or float(upper) < float(lower)
            ):
                raise ValueError("hierarchical prototype distance quantiles are invalid")
        elif lower is not None or upper is not None:
            raise ValueError("hierarchical prototype distance quantiles are invalid")
    for category in categories:
        if category.get("action_id") is None and (
            category.get("training_count") != 0
            or category.get("requested_centers") is not None
            or category.get("actual_centers") != 0
            or category.get("nearest_distance_q10") is not None
            or category.get("nearest_distance_q90") is not None
        ):
            raise ValueError("hierarchical prototype unassigned action metadata is invalid")

    if [leaf.get("prototype_id") for leaf in leaves] != list(range(len(leaves))):
        raise ValueError("hierarchical prototype ids are not contiguous")
    expected_leaf_id = 0
    for action_id, category in sorted(categories_by_id.items()):
        actual_centers = int(category["actual_centers"])
        action_leaves = [leaf for leaf in leaves if leaf.get("action_id") == action_id]
        if len(action_leaves) != actual_centers:
            raise ValueError("hierarchical prototype leaf count is invalid")
        for center_id, leaf in enumerate(action_leaves):
            action_label = str(category["label"])
            if (
                leaf.get("prototype_id") != expected_leaf_id
                or leaf.get("center_id") != center_id
                or isinstance(leaf.get("prototype_id"), bool)
                or not isinstance(leaf.get("prototype_id"), int)
                or isinstance(leaf.get("action_id"), bool)
                or not isinstance(leaf.get("action_id"), int)
                or isinstance(leaf.get("center_id"), bool)
                or not isinstance(leaf.get("center_id"), int)
                or leaf.get("action_label") != action_label
                or leaf.get("label") != f"{action_label}::center_{center_id}"
            ):
                raise ValueError("hierarchical prototype label or ordering is invalid")
            expected_leaf_id += 1
    if expected_leaf_id != len(leaves):
        raise ValueError("hierarchical prototype leaf ordering is invalid")
    return tuple(leaves)


def _numeric_values_match(actual: object, expected: np.ndarray) -> bool:
    try:
        values = np.asarray(actual, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return values.shape == expected.shape and bool(
        np.allclose(values, expected, rtol=1.0e-6, atol=1.0e-7)
    )


def _validate_hierarchical_graph_artifacts(
    root: Path,
    *,
    adapter: DatasetAdapter,
    resolved: Mapping[str, Any],
    clips: list[ClipRecord],
    expected_total_raw_actions: int,
) -> None:
    graph_root = root / GRAPH_DIRECTORY
    with np.load(graph_root / "nodes.npz") as stored:
        if not {"prototype_indices", "prototype_weights"} <= set(stored.files):
            raise ValueError("hierarchical prototype node arrays are missing")
        if {"prototype_action_weights", "prototype_distance_weights"} & set(stored.files):
            raise ValueError("schema-3 hierarchical prototype arrays are incompatible")
        indices = stored["prototype_indices"].copy()
        weights = stored["prototype_weights"].copy()
    if (
        indices.dtype != np.dtype(np.int32)
        or weights.dtype != np.dtype(np.float32)
        or indices.ndim != 2
        or indices.shape != weights.shape
        or indices.shape[0] != len(clips)
        or indices.shape[1] != 2
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("hierarchical prototype node shape or dtype is invalid")

    payload = json.loads((graph_root / "prototype_catalog.json").read_text(encoding="utf-8"))
    leaves = _validate_schema_eight_catalog(
        payload,
        expected_total_raw_actions=expected_total_raw_actions,
    )
    visual_halves = np.load(root / "encode" / "visual_half_embeddings.npy", allow_pickle=False)
    visual_dim = int(resolved["encoding"]["visual_dim"])
    pca_components = _load_visual_pca_components(root / "encode", visual_dim=visual_dim)
    centers = np.load(graph_root / "prototype_centers.npy", allow_pickle=False)
    half_action_labels = np.load(graph_root / "half_action_labels.npy", allow_pickle=False)
    if (
        visual_halves.dtype != np.dtype(np.float32)
        or visual_halves.ndim != 3
        or visual_halves.shape[:2] != (len(clips), 2)
        or visual_halves.shape[2] == 0
        or not np.all(np.isfinite(visual_halves))
        or not np.allclose(np.linalg.norm(visual_halves, axis=2), 1.0, rtol=1.0e-5, atol=1.0e-6)
    ):
        raise ValueError("visual half embeddings are invalid")
    if (
        centers.dtype != np.dtype(np.float32)
        or centers.ndim != 2
        or centers.shape != (len(leaves), visual_dim)
        or not np.all(np.isfinite(centers))
    ):
        raise ValueError("hierarchical prototype centers are invalid")
    if (
        half_action_labels.ndim != 2
        or half_action_labels.shape != (len(clips), 2)
        or half_action_labels.dtype.kind != "U"
        or any(not str(label) for label in half_action_labels.flat)
    ):
        raise ValueError("hierarchical half action labels are invalid")

    valid = indices >= 0
    counts = np.sum(valid, axis=1)
    expected_valid = np.arange(indices.shape[1])[None, :] < counts[:, None]
    if (
        np.any(counts == 0)
        or not np.array_equal(valid, expected_valid)
        or np.any(indices[~valid] != -1)
        or np.any(indices[valid] >= len(leaves))
        or np.any(weights[~valid] != 0.0)
        or np.any(weights[valid] <= 0.0)
        or np.any(weights[valid] > 1.5)
        or any(
            len(set(int(value) for value in row[:count])) != int(count)
            for row, count in zip(indices, counts, strict=True)
        )
        or any(
            list(zip(row_weights[:count], row_indices[:count], strict=True))
            != sorted(
                zip(row_weights[:count], row_indices[:count], strict=True),
                key=lambda item: (-float(item[0]), int(item[1])),
            )
            for row_indices, row_weights, count in zip(indices, weights, counts, strict=True)
        )
    ):
        raise ValueError("hierarchical prototype assignment arrays are invalid")

    prototype_config = resolved["prototypes"]
    replay = build_hierarchical_motion_prototypes(
        adapter,
        clips,
        visual_halves,
        pca_components=pca_components,
        visual_dim=visual_dim,
        frame_cache_dir=root / "encode" / "frame_embeddings",
        batch_size=int(prototype_config["batch_size"]),
        max_iter=int(prototype_config["max_iter"]),
        tol=float(prototype_config["tol"]),
        seed=int(resolved["seed"]),
        max_episodes=resolved["runtime"].get("max_episodes"),
        num_workers=int(resolved["runtime"].get("num_workers", 0)),
        num_threads=int(prototype_config["num_threads"]),
    )
    replay_centers = replay.prototypes.centers
    if replay_centers is None:
        raise ValueError("hierarchical prototype replay produced no centers")
    if payload != replay.catalog.to_dict():
        raise ValueError("hierarchical prototype catalog does not match prototype replay")
    if not np.array_equal(half_action_labels, replay.half_action_labels):
        raise ValueError("hierarchical half action labels do not match prototype replay")
    if not np.allclose(centers, replay_centers, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("hierarchical prototype centers do not match prototype replay")
    if not np.array_equal(indices, replay.prototypes.indices) or not np.allclose(
        weights,
        replay.prototypes.weights,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise ValueError("hierarchical prototype assignments do not match prototype replay")


def _selection_rows(
    resolved: Mapping[str, Any],
    clips: list[ClipRecord],
    graph: GraphData,
    nodes: Mapping[str, np.ndarray],
    result,
    context: CocoreObjectiveContext,
    leaf_metadata: tuple[Mapping[str, Any], ...],
    half_action_labels: np.ndarray,
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
        cluster_ids = [int(value["center_id"]) for value in metadata]
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
            "primary_prototype_label": labels[0],
            "prototype_labels": labels,
            "primary_action_label": action_labels[0],
            "half_action_labels": [str(value) for value in half_action_labels[index]],
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
            "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
            "prototype_strategy": PROTOTYPE_STRATEGY,
        }
    )

    def build(temporary: Path) -> None:
        started = time.perf_counter()
        with timed_step("select.context", emit_completed_timing):
            context = CocoreObjectiveContext(
                graph,
                relation_type,
                relation_weight,
                similarity_threshold=float(resolved["graph"]["similarity_threshold"]),
            )
        with timed_step("select.coverage_seed", emit_completed_timing):
            coverage_seed = build_max_coverage_seed(context, budget=budget)
        with timed_step("select.lazy_heap", emit_completed_timing):
            selector = LazyHeapSelector(
                context,
                max_refreshes=max_refreshes,
            )
            result = selector.select(budget, initial_indices=coverage_seed.selected_indices)

        export_started = time.perf_counter()
        graph_nodes = np.load(root / GRAPH_DIRECTORY / "nodes.npz")
        half_action_labels = np.load(
            root / GRAPH_DIRECTORY / "half_action_labels.npy", allow_pickle=False
        )
        selected_rows, all_rows = _selection_rows(
            resolved,
            clips,
            graph,
            graph_nodes,
            result,
            context,
            _leaf_prototype_metadata(root / GRAPH_DIRECTORY),
            half_action_labels,
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
        scan_manifest = json.loads((root / "scan" / "manifest.json").read_text(encoding="utf-8"))
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
            "prototype_strategy": PROTOTYPE_STRATEGY,
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
                "cocore_version": __version__,
                "fingerprint": fingerprint,
                "upstream_fingerprint": graph_fingerprint,
                "stage_directory": directory,
                "selected_clips": len(result.selected_indices),
                "budget": budget,
                "selection_ratio": ratio,
                "relation_type": relation_type,
                "relation_weight": relation_weight,
                "algorithm": algorithm,
                "prototype_method": "motion_primitives",
                "prototype_schema_version": PROTOTYPE_SCHEMA_VERSION,
                "prototype_strategy": PROTOTYPE_STRATEGY,
            },
        )
        emit_completed_timing("select.export", time.perf_counter() - export_started)

    stage_started = time.perf_counter()
    built = publish_stage(
        destination,
        fingerprint=fingerprint,
        required=("selected_manifest.jsonl", "all_clips.parquet", "selection_report.json"),
        force=force,
        resume=bool(resolved["runtime"].get("resume", True)),
        build=build,
    )
    if built:
        emit_completed_timing("select", time.perf_counter() - stage_started)
    stage_directories = {
        "scan": "scan",
        "encode": "encode",
        "graph": GRAPH_DIRECTORY,
        "select": directory,
    }
    stage_fingerprints = {
        stage: json.loads((root / stage_directory / "manifest.json").read_text(encoding="utf-8"))[
            "fingerprint"
        ]
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
            "prototype_strategy": PROTOTYPE_STRATEGY,
            "window_policy": WINDOW_POLICY,
            "sequence_adjacency": SEQUENCE_ADJACENCY,
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
    if run_manifest.get("prototype_strategy") != PROTOTYPE_STRATEGY:
        raise ValueError("cocore prototype strategy is incompatible")
    if run_manifest.get("window_policy") != WINDOW_POLICY:
        raise ValueError("cocore run manifest window policy is incompatible")
    if run_manifest.get("sequence_adjacency") != SEQUENCE_ADJACENCY:
        raise ValueError("cocore run manifest sequence adjacency is incompatible")
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
        "scan": ("episodes.parquet", "clips.parquet"),
        "encode": (
            "embeddings.npy",
            "visual_half_embeddings.npy",
            "state_sequences.npy",
            "action_sequences.npy",
            "visual_progress.npy",
            "numeric_normalizers.npz",
            "visual_pca.npz",
            "frame_embeddings_index.json",
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
    stage_manifests: dict[str, Mapping[str, Any]] = {}
    for stage, directory in expected_directories.items():
        path = root / directory
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing stage manifest: {stage}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_manifests[stage] = manifest
        if stage in {"scan", "encode", "graph"} and (
            manifest.get("producer") != "cocore"
            or manifest.get("cocore_version") != __version__
            or manifest.get("cocore_stage") != stage
        ):
            raise ValueError(f"stage manifest metadata is incompatible: {stage}")
        if not cache_is_valid(path, str(manifest.get("fingerprint", "")), stage_required[stage]):
            raise ValueError(f"invalid stage artifacts: {stage}")
        if run_manifest["stage_fingerprints"].get(stage) != manifest["fingerprint"]:
            raise ValueError(f"run/stage fingerprint mismatch: {stage}")
    if (
        stage_manifests["graph"].get("prototype_schema_version") != PROTOTYPE_SCHEMA_VERSION
        or stage_manifests["graph"].get("prototype_strategy") != PROTOTYPE_STRATEGY
        or stage_manifests["graph"].get("prototype_visual_dim") != 128
        or stage_manifests["graph"].get("prototype_visual_projection")
        != PROTOTYPE_VISUAL_PROJECTION
        or stage_manifests["graph"].get("prototype_visual_normalization")
        != PROTOTYPE_VISUAL_NORMALIZATION
        or stage_manifests["graph"].get("stage_directory") != GRAPH_DIRECTORY
        or stage_manifests["graph"].get("sequence_adjacency") != SEQUENCE_ADJACENCY
    ):
        raise ValueError("graph manifest prototype schema is incompatible")
    if any(
        stage_manifests[stage].get("window_policy") != WINDOW_POLICY
        or stage_manifests[stage].get("clip_length") != CLIP_LENGTH
        for stage in ("scan", "encode")
    ):
        raise ValueError("cocore stage window policy is incompatible")
    scan_clips = _load_clips(root / "scan" / "clips.parquet")
    episode_rows = pq.read_table(root / "scan" / "episodes.parquet").to_pylist()
    expected_clips = build_clip_records(
        [
            EpisodeRecord(
                episode_id=int(row["episode_id"]),
                length=int(row["length"]),
                task_index=int(row["task_index"]),
                task_name=str(row["task_name"]),
            )
            for row in episode_rows
        ]
    )
    if scan_clips != expected_clips:
        raise ValueError("scan clips do not match Cocore near-uniform candidate windows")
    _validate_frame_embedding_cache(
        root / "encode",
        expected_episodes=[(int(row["episode_id"]), int(row["length"])) for row in episode_rows],
    )
    _validate_visual_half_embedding_cache(root / "encode", scan_clips)
    if config is None:
        resolved_path = result / "resolved_config.yaml"
        if not resolved_path.is_file():
            raise ValueError("cocore validation requires configuration for prototype replay")
        stored_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        if not isinstance(stored_config, Mapping):
            raise ValueError("stored cocore configuration is invalid")
        validation_config = stored_config
    else:
        validation_config = config
    replay_resolved = resolve_config(validation_config)
    replay_resolved["output"]["directory"] = str(root)
    replay_adapter = create_dataset(replay_resolved["dataset"])
    _validate_hierarchical_graph_artifacts(
        root,
        adapter=replay_adapter,
        resolved=replay_resolved,
        clips=scan_clips,
        expected_total_raw_actions=sum(
            len(trajectory_window_starts(int(row["length"]))) for row in episode_rows
        ),
    )
    select_manifest = json.loads(required["select_manifest"].read_text(encoding="utf-8"))
    if select_manifest.get("producer") != "cocore":
        raise ValueError("selection manifest producer is invalid")
    if select_manifest.get("cocore_version") != __version__:
        raise ValueError("selection manifest Cocore version is incompatible")
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
    if (
        select_manifest.get("prototype_schema_version") != PROTOTYPE_SCHEMA_VERSION
        or select_manifest.get("prototype_strategy") != PROTOTYPE_STRATEGY
        or report.get("prototype_strategy") != PROTOTYPE_STRATEGY
    ):
        raise ValueError("selection prototype schema metadata mismatch")
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
    half_action_labels = np.load(
        root / GRAPH_DIRECTORY / "half_action_labels.npy", allow_pickle=False
    )
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
        expected_labels = [leaf_metadata[int(value)]["label"] for value in assigned_indices]
        expected_actions = [leaf_metadata[int(value)]["action_label"] for value in assigned_indices]
        expected_clusters = [
            int(leaf_metadata[int(value)]["center_id"]) for value in assigned_indices
        ]
        if (
            row.get("prototype_indices") != [int(value) for value in assigned_indices]
            or not _numeric_values_match(row.get("prototype_weights"), assigned_weights)
            or row.get("prototype_labels") != expected_labels
            or row.get("prototype_action_labels") != expected_actions
            or row.get("prototype_cluster_ids") != expected_clusters
            or "prototype_action_weights" in row
            or "prototype_distance_weights" in row
            or row.get("primary_prototype") != int(assigned_indices[0])
            or row.get("primary_prototype_label") != expected_labels[0]
            or row.get("primary_action_label") != expected_actions[0]
            or row.get("half_action_labels") != [str(value) for value in half_action_labels[index]]
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
        "primary_prototype",
        "primary_prototype_label",
        "primary_action_label",
        "half_action_labels",
    )
    for selected_row, all_row in zip(selected_rows, selected_from_all, strict=True):
        if any(
            field in selected_row
            for field in ("prototype_action_weights", "prototype_distance_weights")
        ):
            raise ValueError("selected row contains obsolete schema-3 prototype field")
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
        if not np.isclose(float(row.get("marginal_gain", np.nan)), gain, rtol=1.0e-7, atol=1.0e-8):
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
            0
            if initial_set_size == len(selected_rows)
            else len(graph.sample_ids) - initial_set_size
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
