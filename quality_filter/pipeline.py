"""Cached quality, encoding, and diversity-filtering stages."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from segment_filter_core import (
    ClipVisionEncoder,
    NumericNormalizers,
    PCAProjector,
    RawQuality,
    candidate_windows,
    fuse_fragment_features,
    raw_quality,
    reference_windows,
    score_quality,
    temporal_pool,
    visual_fragment_feature,
)
from segment_filter_core.selection import (
    PENALTY_LAMBDA,
    _AlgorithmParameters,
    select_diverse_fragments,
)
from trajectory_data import DatasetAdapter, create_dataset

from . import __version__


QUALITY_COLUMNS = (
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "action_smooth",
    "state_transition",
    "motion_efficiency",
    "quality",
)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _output_root(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    return Path(
        output_dir if output_dir is not None else config["output"]["directory"]
    ).expanduser().resolve()


def _max_episodes(config: Mapping[str, Any]) -> int | None:
    value = config["runtime"].get("max_episodes")
    return int(value) if value is not None else None


def _validate_common(config: Mapping[str, Any], adapter: DatasetAdapter) -> None:
    required = {"dataset", "clip", "encoder", "quality", "filter", "runtime", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"quality_filter config is missing sections: {missing}")
    if config["clip"] != {"length": 15, "stride": 15}:
        raise ValueError("quality_filter requires clip length=15 and stride=15")
    if len(adapter.image_observation_keys) != 1:
        raise ValueError("quality_filter requires exactly one configured image observation")
    if not adapter.vector_observation_keys:
        raise ValueError("quality_filter requires at least one vector state observation")
    encoder = config["encoder"]
    if int(encoder.get("visual_dim", -1)) != 128:
        raise ValueError("quality_filter encoder.visual_dim must be 128")
    if encoder.get("local_files_only") is not True:
        raise ValueError("quality_filter encoder.local_files_only must be true")


def _stage_cache_valid(
    destination: Path,
    fingerprint: str,
    required: tuple[str, ...],
    *,
    manifest_name: str = "manifest.json",
) -> bool:
    manifest_path = destination / manifest_name
    if not manifest_path.is_file() or any(
        not (destination / relative).is_file() for relative in required
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("status") != "complete"
        or manifest.get("fingerprint") != fingerprint
    ):
        return False
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        return False
    for relative in required:
        path = destination / relative
        key = Path(relative).stem
        if hashes.get(key) != _sha256(path):
            return False
    return True


def _publish_stage(
    destination: Path,
    *,
    fingerprint: str,
    required: tuple[str, ...],
    force: bool,
    resume: bool,
    build: Any,
    manifest_name: str = "manifest.json",
) -> None:
    if resume and _stage_cache_valid(
        destination,
        fingerprint,
        required,
        manifest_name=manifest_name,
    ):
        return
    if destination.exists() and not force:
        raise FileExistsError(
            f"quality_filter stage exists but is not a compatible cache: {destination}; "
            "pass --force"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.quality-filter-",
            dir=destination.parent,
        )
    )
    try:
        build(temporary)
        if not _stage_cache_valid(
            temporary,
            fingerprint,
            required,
            manifest_name=manifest_name,
        ):
            raise RuntimeError(f"quality_filter stage build is incomplete: {destination}")
        previous = destination.with_name(f".{destination.name}.previous")
        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            os.replace(destination, previous)
        try:
            os.replace(temporary, destination)
        except Exception:
            if previous.exists():
                os.replace(previous, destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _quality_fingerprint(config: Mapping[str, Any], adapter: DatasetAdapter) -> str:
    return _stable_hash(
        {
            "version": __version__,
            "stage": "quality",
            "adapter": adapter.fingerprint(),
            "dataset": config["dataset"],
            "clip": config["clip"],
            "quality": config["quality"],
            "max_episodes": config["runtime"].get("max_episodes"),
        }
    )


def _quality_metadata(episode_id: int, start_step: int, end_step: int) -> dict[str, Any]:
    return {
        "sample_id": f"ep{episode_id:06d}_fragment_{start_step:06d}_{end_step:06d}",
        "episode_id": int(episode_id),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "length": 15,
    }


def quality_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Compute SQCN-compatible Quality and numeric candidate features."""

    adapter = create_dataset(config["dataset"])
    _validate_common(config, adapter)
    root = _output_root(config, output_dir)
    destination = root / "quality"
    fingerprint = _quality_fingerprint(config, adapter)
    workers = int(config["runtime"].get("num_workers", 0))
    max_episodes = _max_episodes(config)
    quality_config = config["quality"]

    def build(temporary: Path) -> None:
        numeric_episodes = adapter.iter_episodes(
            num_workers=workers,
            max_episodes=max_episodes,
            load_images=False,
        )
        normalizers = NumericNormalizers.fit(
            ((episode.actions, episode.observations) for episode in numeric_episodes),
            adapter.vector_observation_keys,
            quantile_low=float(quality_config.get("quantile_low", 0.01)),
            quantile_high=float(quality_config.get("quantile_high", 0.99)),
            epsilon=float(quality_config.get("epsilon", 1.0e-8)),
        )
        metadata: list[dict[str, Any]] = []
        raw_values: list[RawQuality] = []
        state_pooled: list[np.ndarray] = []
        action_pooled: list[np.ndarray] = []
        progress: list[float] = []
        skipped_short = 0
        for episode in adapter.iter_episodes(
            num_workers=workers,
            max_episodes=max_episodes,
            load_images=False,
        ):
            windows = candidate_windows(episode.length)
            if not windows:
                skipped_short += 1
                continue
            normalized_actions = normalizers.action(episode.actions)
            normalized_state = normalizers.state(episode.observations)
            for start, end in windows:
                index = slice(start, end + 1)
                metadata.append(
                    _quality_metadata(
                        episode.episode_id,
                        int(episode.frame_indices[start]),
                        int(episode.frame_indices[end]),
                    )
                )
                raw_values.append(raw_quality(normalized_actions[index], normalized_state[index]))
                state_pooled.append(temporal_pool(normalized_state[index]))
                action_pooled.append(temporal_pool(normalized_actions[index]))
                progress.append(float(start) / float(episode.length))
        if not metadata:
            raise RuntimeError("quality_filter produced no complete candidate fragments")
        quality, details = score_quality(
            raw_values,
            quantile_low=float(quality_config.get("quantile_low", 0.01)),
            quantile_high=float(quality_config.get("quantile_high", 0.99)),
            epsilon=float(quality_config.get("epsilon", 1.0e-8)),
        )
        sample_ids = np.asarray([row["sample_id"] for row in metadata])
        order = np.lexsort((sample_ids, -quality)).astype(np.int64)
        scores_path = temporary / "scores.csv"
        with scores_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUALITY_COLUMNS)
            writer.writeheader()
            for position in order:
                index = int(position)
                writer.writerow(
                    {
                        **metadata[index],
                        "action_smooth": f"{float(details['action_smooth'][index]):.9f}",
                        "state_transition": f"{float(details['state_transition'][index]):.9f}",
                        "motion_efficiency": f"{float(details['motion_efficiency'][index]):.9f}",
                        "quality": f"{float(quality[index]):.9f}",
                    }
                )
        np.savez(
            temporary / "numeric_features.npz",
            sample_ids=sample_ids[order],
            state_pooled=np.stack(state_pooled)[order],
            action_pooled=np.stack(action_pooled)[order],
            progress=np.asarray(progress, dtype=np.float32)[order],
            raw_action_delta=np.asarray(
                [value.action_delta for value in raw_values], dtype=np.float32
            )[order],
            raw_state_transition=np.asarray(
                [value.state_transition for value in raw_values], dtype=np.float32
            )[order],
            raw_motion_efficiency=np.asarray(
                [value.motion_efficiency for value in raw_values], dtype=np.float32
            )[order],
        )
        with (temporary / "numeric_normalizers.pkl").open("wb") as handle:
            pickle.dump(normalizers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        bounds = {
            key.removesuffix("_bounds"): [float(value[0]), float(value[1])]
            for key, value in details.items()
            if key.endswith("_bounds") and isinstance(value, tuple)
        }
        _write_json(
            temporary / "manifest.json",
            {
                "version": __version__,
                "status": "complete",
                "stage": "quality",
                "fingerprint": fingerprint,
                "dataset_name": str(config["dataset"].get("name", "dataset")),
                "dataset_path": str(config["dataset"].get("path", "")),
                "quality_bounds": bounds,
                "counts": {
                    "candidate_fragments": len(metadata),
                    "skipped_short_episodes": skipped_short,
                },
                "outputs": {
                    "scores": str(root / "quality" / "scores.csv"),
                    "numeric_features": str(root / "quality" / "numeric_features.npz"),
                    "numeric_normalizers": str(
                        root / "quality" / "numeric_normalizers.pkl"
                    ),
                },
                "hashes": {
                    "scores": _sha256(scores_path),
                    "numeric_features": _sha256(temporary / "numeric_features.npz"),
                    "numeric_normalizers": _sha256(
                        temporary / "numeric_normalizers.pkl"
                    ),
                },
            },
        )

    _publish_stage(
        destination,
        fingerprint=fingerprint,
        required=(
            "scores.csv",
            "numeric_features.npz",
            "numeric_normalizers.pkl",
        ),
        force=force,
        resume=bool(config["runtime"].get("resume", True)),
        build=build,
    )
    return root


def _load_quality_rows(root: Path) -> list[dict[str, str]]:
    path = root / "quality" / "scores.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != QUALITY_COLUMNS:
            raise ValueError(f"quality_filter scores have an invalid schema: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"quality_filter scores are empty: {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids) or any(not value for value in sample_ids):
        raise ValueError("quality_filter scores require unique non-empty sample_id values")
    return rows


def _encode_fingerprint(
    config: Mapping[str, Any],
    adapter: DatasetAdapter,
    quality_manifest: Mapping[str, Any],
) -> str:
    return _stable_hash(
        {
            "version": __version__,
            "stage": "encode",
            "adapter": adapter.fingerprint(),
            "dataset": config["dataset"],
            "clip": config["clip"],
            "encoder": config["encoder"],
            "seed": config["runtime"].get("seed", 42),
            "max_episodes": config["runtime"].get("max_episodes"),
            "quality_fingerprint": quality_manifest["fingerprint"],
            "quality_hashes": quality_manifest["hashes"],
        }
    )


def _write_run_manifest(
    root: Path,
    config: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    encode_manifest: Mapping[str, Any],
) -> None:
    payload = {
        "version": __version__,
        "status": "complete",
        "fingerprint": _stable_hash(
            {
                "quality": quality_manifest["fingerprint"],
                "encode": encode_manifest["fingerprint"],
            }
        ),
        "dataset_name": str(config["dataset"].get("name", "dataset")),
        "dataset_path": str(config["dataset"].get("path", "")),
        "fragment_length": 15,
        "candidate_stride": 15,
        "visual_embedding_dim": 128,
        "embedding_dim": int(encode_manifest["embedding_dim"]),
        "config": json.loads(
            json.dumps({key: value for key, value in config.items() if key != "filter"})
        ),
        "stages": {
            "quality": {
                "manifest": str(root / "quality" / "manifest.json"),
                "fingerprint": quality_manifest["fingerprint"],
            },
            "encode": {
                "manifest": str(root / "encode" / "manifest.json"),
                "fingerprint": encode_manifest["fingerprint"],
            },
        },
        "outputs": {
            "scores": str(root / "quality" / "scores.csv"),
            "embeddings": str(root / "encode" / "embeddings.npy"),
        },
    }
    temporary = root / ".run_manifest.json.tmp"
    _write_json(temporary, payload)
    os.replace(temporary, root / "run_manifest.json")


def encode_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    visual_encoder: Any | None = None,
) -> Path:
    """Create SQCN-compatible fused embeddings without Coverage or Novelty."""

    root = quality_stage(config, output_dir=output_dir, force=force)
    adapter = create_dataset(config["dataset"])
    _validate_common(config, adapter)
    quality_manifest_path = root / "quality" / "manifest.json"
    quality_manifest = json.loads(quality_manifest_path.read_text(encoding="utf-8"))
    destination = root / "encode"
    fingerprint = _encode_fingerprint(config, adapter, quality_manifest)
    workers = int(config["runtime"].get("num_workers", 0))
    max_episodes = _max_episodes(config)

    def build(temporary: Path) -> None:
        rows = _load_quality_rows(root)
        score_sample_ids = [row["sample_id"] for row in rows]
        numeric_path = root / "quality" / "numeric_features.npz"
        with np.load(numeric_path, allow_pickle=False) as numeric:
            numeric_sample_ids = numeric["sample_ids"].astype(str).tolist()
            state_pooled = numeric["state_pooled"].astype(np.float32)
            action_pooled = numeric["action_pooled"].astype(np.float32)
            progress = numeric["progress"].astype(np.float32)
        if numeric_sample_ids != score_sample_ids:
            raise ValueError("quality scores and numeric features have misaligned sample IDs")
        vision = visual_encoder or ClipVisionEncoder(config["encoder"])
        union_visual: dict[str, np.ndarray] = {}
        reference_ids: list[str] = []
        candidate_ids_seen: set[str] = set()
        for episode in adapter.iter_episodes(
            num_workers=workers,
            max_episodes=max_episodes,
            load_images=True,
        ):
            candidates = candidate_windows(episode.length)
            references = reference_windows(episode.length)
            if not candidates:
                continue
            image_key = adapter.image_observation_keys[0]
            if image_key not in episode.observations:
                raise ValueError(
                    f"episode {episode.episode_id} is missing image {image_key!r}"
                )
            frame_features = vision.encode(episode.observations[image_key])
            if len(frame_features) != episode.length:
                raise ValueError(
                    f"episode {episode.episode_id}: encoded frames={len(frame_features)}, "
                    f"expected={episode.length}"
                )
            window_ids: dict[tuple[int, int], str] = {}
            for start, end in sorted(set(candidates) | set(references)):
                start_step = int(episode.frame_indices[start])
                end_step = int(episode.frame_indices[end])
                sample_id = _quality_metadata(
                    episode.episode_id,
                    start_step,
                    end_step,
                )["sample_id"]
                window_ids[(start, end)] = sample_id
                union_visual[sample_id] = visual_fragment_feature(
                    frame_features[start : end + 1]
                )
            candidate_ids_seen.update(window_ids[window] for window in candidates)
            reference_ids.extend(window_ids[window] for window in references)
        if candidate_ids_seen != set(score_sample_ids):
            raise ValueError("encoded candidate IDs do not match quality scores")
        union_ids = list(union_visual)
        visual_raw = np.stack([union_visual[sample_id] for sample_id in union_ids])
        projector = PCAProjector(
            output_dim=int(config["encoder"]["visual_dim"]),
            seed=int(config["runtime"].get("seed", 42)),
        )
        union_projected = projector.fit_transform(
            visual_raw,
            max_samples=config["encoder"].get("pca_fit_max_samples"),
        )
        union_index = {sample_id: index for index, sample_id in enumerate(union_ids)}
        candidate_visual = union_projected[
            np.asarray([union_index[sample_id] for sample_id in score_sample_ids])
        ]
        _, embeddings = fuse_fragment_features(
            candidate_visual,
            state_pooled,
            action_pooled,
            progress,
        )
        np.save(temporary / "embeddings.npy", embeddings)
        projector.save(temporary / "visual_pca.pkl")
        candidate_set = set(score_sample_ids)
        reference_set = set(reference_ids)
        _write_json(
            temporary / "manifest.json",
            {
                "version": __version__,
                "status": "complete",
                "stage": "encode",
                "fingerprint": fingerprint,
                "upstream": {
                    "quality_manifest": str(quality_manifest_path),
                    "quality_fingerprint": quality_manifest["fingerprint"],
                    "scores_sha256": _sha256(root / "quality" / "scores.csv"),
                    "numeric_features_sha256": _sha256(numeric_path),
                },
                "visual_embedding_dim": int(config["encoder"]["visual_dim"]),
                "embedding_dim": int(embeddings.shape[1]),
                "counts": {
                    "candidate_fragments": len(score_sample_ids),
                    "reference_fragments": len(reference_ids),
                    "overlap_fragments": len(candidate_set & reference_set),
                    "pca_union_fragments": len(union_ids),
                },
                "outputs": {
                    "embeddings": str(root / "encode" / "embeddings.npy"),
                    "visual_pca": str(root / "encode" / "visual_pca.pkl"),
                },
                "hashes": {
                    "embeddings": _sha256(temporary / "embeddings.npy"),
                    "visual_pca": _sha256(temporary / "visual_pca.pkl"),
                },
            },
        )

    _publish_stage(
        destination,
        fingerprint=fingerprint,
        required=("embeddings.npy", "visual_pca.pkl"),
        force=force,
        resume=bool(config["runtime"].get("resume", True)),
        build=build,
    )
    encode_manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    _write_run_manifest(root, config, quality_manifest, encode_manifest)
    return root


def _normalize_percent(value: Any) -> tuple[float, Decimal]:
    if isinstance(value, bool):
        raise ValueError("percent must be finite and in (0, 100]")
    try:
        percent = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("percent must be finite and in (0, 100]") from error
    if not np.isfinite(percent) or not 0.0 < percent <= 100.0:
        raise ValueError("percent must be finite and in (0, 100]")
    return percent, Decimal(str(value))


def _percent_tag(percent: float) -> str:
    return format(percent, ".12g").replace(".", "p")


def _selection_sha256(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def filter_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    percent: float | None = None,
    seed: int | None = None,
    force: bool = False,
) -> Path:
    """Select a percentage of candidates using Quality and SQCN diversity."""

    root = encode_stage(config, output_dir=output_dir, force=force)
    percent_value, percent_decimal = _normalize_percent(
        config["filter"].get("percent", 10.0) if percent is None else percent
    )
    requested_seed = config["filter"].get("seed") if seed is None else seed
    run_manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    scores_path = root / "quality" / "scores.csv"
    embeddings_path = root / "encode" / "embeddings.npy"
    rows = _load_quality_rows(root)
    scores = np.asarray([float(row["quality"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("quality scores must contain finite values in [0, 1]")
    embeddings = np.load(embeddings_path, allow_pickle=False)
    expected_dim = int(run_manifest["embedding_dim"])
    if embeddings.shape != (len(rows), expected_dim):
        raise ValueError(
            "quality_filter embeddings must align with scores and declared dimension"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("quality_filter embeddings must contain only finite values")
    target_size = int(
        (Decimal(len(rows)) * percent_decimal / Decimal(100)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    output_root = root / "filter" / f"top{_percent_tag(percent_value)}pct"
    source_hashes = {
        "run_manifest": _sha256(run_manifest_path),
        "scores": _sha256(scores_path),
        "embeddings": _sha256(embeddings_path),
    }
    fingerprint = _stable_hash(
        {
            "version": __version__,
            "stage": "filter",
            "source": source_hashes,
            "percent": percent_value,
            "seed": requested_seed,
        }
    )

    def build(temporary: Path) -> None:
        sample_ids = [row["sample_id"] for row in rows]
        result = select_diverse_fragments(
            scores,
            embeddings,
            sample_ids,
            target_size,
            seed=requested_seed,
        )
        selected_ids = [sample_ids[int(index)] for index in result.selected_indices]
        filter_columns = ("filter_rank", "adjusted_score", "knn_penalty")
        with (temporary / "scores.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=[*QUALITY_COLUMNS, *filter_columns])
            writer.writeheader()
            for rank, (index, adjusted, penalty) in enumerate(
                zip(
                    result.selected_indices,
                    result.adjusted_scores,
                    result.knn_penalties,
                    strict=True,
                ),
                start=1,
            ):
                writer.writerow(
                    {
                        **rows[int(index)],
                        "filter_rank": rank,
                        "adjusted_score": f"{float(adjusted):.9f}",
                        "knn_penalty": f"{float(penalty):.9f}",
                    }
                )
        np.save(temporary / "embeddings.npy", embeddings[result.selected_indices])
        parameters = _AlgorithmParameters()
        _write_json(
            temporary / "filter_manifest.json",
            {
                "version": __version__,
                "status": "complete",
                "stage": "filter",
                "fingerprint": fingerprint,
                "source": {
                    "input_dir": str(root),
                    "run_manifest": str(run_manifest_path),
                    "run_manifest_sha256": source_hashes["run_manifest"],
                    "scores": str(scores_path),
                    "scores_sha256": source_hashes["scores"],
                    "embeddings": str(embeddings_path),
                    "embeddings_sha256": source_hashes["embeddings"],
                },
                "algorithm": {
                    "percent": percent_value,
                    "target_size": target_size,
                    "seed": result.seed,
                    "score_column": "quality",
                    "lambda": PENALTY_LAMBDA,
                    "penalty": {
                        "policy": "mean_rbf_similarity_weighted_score_of_nearest_references",
                        "neighbor_count": parameters.neighbor_count,
                        "weight": "rbf_similarity",
                        "aggregation": (
                            "sum(similarity * score) / effective_neighbor_count"
                        ),
                    },
                    "silent": {
                        "policy": "frozen_adjusted_score_heap",
                        "initial_population": "all_non_initial_fragments",
                        "initial_candidate_fill": (
                            "top_adjusted_score_up_to_candidate_capacity"
                        ),
                        "steady_promotion": "one_after_each_selection",
                    },
                    "update_count": {
                        "unit": "reference_fragments",
                        "initial": parameters.init_select_size,
                        "promotion_minimum": (
                            "ceil(100 + log2(selected_count - 100))"
                        ),
                        "catch_up_sampling": (
                            "uniform_without_replacement_from_selected"
                        ),
                        "persisted": "internal_only",
                    },
                    "sigma": {
                        "policy": (
                            "mean_pairwise_euclidean_distance_of_raw_top_100"
                        ),
                        "top_count": min(100, len(rows)),
                        "raw": result.sigma_raw,
                        "effective": result.sigma_effective,
                    },
                    "constants": {
                        "init_select_size": parameters.init_select_size,
                        "candidate_capacity": parameters.candidate_capacity,
                        "neighbor_count": parameters.neighbor_count,
                    },
                    "ordering": [
                        "adjusted_score desc",
                        "quality desc",
                        "sample_id asc",
                    ],
                },
                "counts": {
                    "input_fragments": len(rows),
                    "selected_fragments": target_size,
                },
                "score_columns": [*QUALITY_COLUMNS, *filter_columns],
                "selection_sha256": _selection_sha256(selected_ids),
                "outputs": {
                    "scores": str(output_root / "scores.csv"),
                    "embeddings": str(output_root / "embeddings.npy"),
                    "manifest": str(output_root / "filter_manifest.json"),
                },
                "hashes": {
                    "scores": _sha256(temporary / "scores.csv"),
                    "embeddings": _sha256(temporary / "embeddings.npy"),
                },
            },
        )

    _publish_stage(
        output_root,
        fingerprint=fingerprint,
        required=("scores.csv", "embeddings.npy"),
        force=force,
        resume=bool(config["runtime"].get("resume", True)) and not force,
        build=build,
        manifest_name="filter_manifest.json",
    )
    return output_root


def run_pipeline(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    percent: float | None = None,
    seed: int | None = None,
    force: bool = False,
    visual_encoder: Any | None = None,
) -> Path:
    """Run Quality, Encode, and Filter while reusing compatible stages."""

    root = encode_stage(
        config,
        output_dir=output_dir,
        force=force,
        visual_encoder=visual_encoder,
    )
    filter_stage(
        config,
        output_dir=output_dir,
        percent=percent,
        seed=seed,
        force=force,
    )
    return root


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise ValueError(f"{label} must have status='complete': {path}")
    return value


def _require_hash(path: Path, expected: Any, label: str) -> None:
    try:
        actual = _sha256(path)
    except OSError as error:
        raise ValueError(f"could not hash {label} {path}: {error}") from error
    if actual != expected:
        raise ValueError(f"{label} hash does not match its manifest")


def _validate_filter_output(
    root: Path,
    output: Path,
    *,
    expected_percent: float | None,
    candidate_count: int,
    embedding_dim: int,
) -> tuple[str, int]:
    manifest_path = output / "filter_manifest.json"
    manifest = _read_json(manifest_path, "quality filter manifest")
    algorithm = manifest.get("algorithm")
    counts = manifest.get("counts")
    source = manifest.get("source")
    if not isinstance(algorithm, dict) or not isinstance(counts, dict):
        raise ValueError("quality filter algorithm and counts must be objects")
    if not isinstance(source, dict):
        raise ValueError("quality filter source must be an object")
    if algorithm.get("score_column") != "quality":
        raise ValueError("quality filter score_column must be quality")
    percent, percent_decimal = _normalize_percent(algorithm.get("percent"))
    if expected_percent is not None and percent != expected_percent:
        raise ValueError("quality filter percent does not match requested validation")
    target_size = int(
        (Decimal(candidate_count) * percent_decimal / Decimal(100)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    if counts != {
        "input_fragments": candidate_count,
        "selected_fragments": target_size,
    } or algorithm.get("target_size") != target_size:
        raise ValueError("quality filter counts do not match its percentage")
    source_paths = {
        "run_manifest": root / "run_manifest.json",
        "scores": root / "quality" / "scores.csv",
        "embeddings": root / "encode" / "embeddings.npy",
    }
    for key, path in source_paths.items():
        declared_path = Path(str(source.get(key, ""))).expanduser().resolve()
        if declared_path != path.resolve():
            raise ValueError(f"quality filter source {key} path does not match")
        _require_hash(path, source.get(f"{key}_sha256"), f"filter source {key}")
    scores_path = output / "scores.csv"
    embeddings_path = output / "embeddings.npy"
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError("quality filter output hashes must be an object")
    _require_hash(scores_path, hashes.get("scores"), "filtered scores")
    _require_hash(embeddings_path, hashes.get("embeddings"), "filtered embeddings")
    with scores_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = [
            *QUALITY_COLUMNS,
            "filter_rank",
            "adjusted_score",
            "knn_penalty",
        ]
        if list(reader.fieldnames or []) != expected_columns:
            raise ValueError("quality filter scores have an invalid schema")
        rows = list(reader)
    if len(rows) != target_size:
        raise ValueError("quality filter scores do not match selected count")
    identifiers: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        if int(row["filter_rank"]) != line_number:
            raise ValueError("quality filter ranks must be contiguous from one")
        identifiers.append(row["sample_id"])
        for column in (
            "action_smooth",
            "state_transition",
            "motion_efficiency",
            "quality",
            "knn_penalty",
        ):
            value = float(row[column])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"quality filter {column} must be finite in [0, 1]")
        if not np.isfinite(float(row["adjusted_score"])):
            raise ValueError("quality filter adjusted_score must be finite")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("quality filter scores contain duplicate sample IDs")
    if manifest.get("selection_sha256") != _selection_sha256(identifiers):
        raise ValueError("quality filter selection hash does not match scores")
    selected_embeddings = np.load(embeddings_path, allow_pickle=False)
    if selected_embeddings.shape != (target_size, embedding_dim):
        raise ValueError("filtered embeddings do not align with selected scores")
    return output.name, target_size


def validate_output(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    percent: float | None = None,
) -> dict[str, Any]:
    """Validate completed stage artifacts and selected filter outputs."""

    root = Path(output_dir).expanduser()
    quality_manifest = _read_json(root / "quality" / "manifest.json", "quality manifest")
    encode_manifest = _read_json(root / "encode" / "manifest.json", "encode manifest")
    run_manifest = _read_json(root / "run_manifest.json", "run manifest")
    if config is not None:
        adapter = create_dataset(config["dataset"])
        _validate_common(config, adapter)
        expected_quality = _quality_fingerprint(config, adapter)
        if quality_manifest.get("fingerprint") != expected_quality:
            raise ValueError("quality manifest fingerprint does not match config")
        expected_encode = _encode_fingerprint(config, adapter, quality_manifest)
        if encode_manifest.get("fingerprint") != expected_encode:
            raise ValueError("encode manifest fingerprint does not match config")
    quality_hashes = quality_manifest.get("hashes")
    encode_hashes = encode_manifest.get("hashes")
    if not isinstance(quality_hashes, dict) or not isinstance(encode_hashes, dict):
        raise ValueError("stage output hashes must be objects")
    for key, filename in (
        ("scores", "scores.csv"),
        ("numeric_features", "numeric_features.npz"),
        ("numeric_normalizers", "numeric_normalizers.pkl"),
    ):
        _require_hash(root / "quality" / filename, quality_hashes.get(key), key)
    for key, filename in (
        ("embeddings", "embeddings.npy"),
        ("visual_pca", "visual_pca.pkl"),
    ):
        _require_hash(root / "encode" / filename, encode_hashes.get(key), key)
    rows = _load_quality_rows(root)
    candidate_count = len(rows)
    with np.load(
        root / "quality" / "numeric_features.npz", allow_pickle=False
    ) as numeric:
        if numeric["sample_ids"].astype(str).tolist() != [
            row["sample_id"] for row in rows
        ]:
            raise ValueError("quality scores and numeric features are misaligned")
    embeddings = np.load(root / "encode" / "embeddings.npy", allow_pickle=False)
    embedding_dim = int(encode_manifest.get("embedding_dim", 0))
    if embeddings.shape != (candidate_count, embedding_dim):
        raise ValueError("quality scores and embeddings are misaligned")
    if run_manifest.get("embedding_dim") != embedding_dim:
        raise ValueError("run manifest embedding dimension does not match encode stage")
    if run_manifest.get("dataset_name") != quality_manifest.get("dataset_name"):
        raise ValueError("run and quality manifests disagree on dataset")
    filters: dict[str, int] = {}
    if percent is not None:
        percent_value, _ = _normalize_percent(percent)
        outputs = [root / "filter" / f"top{_percent_tag(percent_value)}pct"]
    else:
        filter_root = root / "filter"
        outputs = sorted(path for path in filter_root.glob("top*pct") if path.is_dir())
    for output in outputs:
        if not output.is_dir():
            raise ValueError(f"quality filter output is missing: {output}")
        name, count = _validate_filter_output(
            root,
            output,
            expected_percent=(float(percent) if percent is not None else None),
            candidate_count=candidate_count,
            embedding_dim=embedding_dim,
        )
        filters[name] = count
    return {
        "status": "valid",
        "candidate_fragments": candidate_count,
        "embedding_dim": embedding_dim,
        "filters": filters,
    }
