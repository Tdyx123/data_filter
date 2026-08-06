"""End-to-end SQCN fragment scoring pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from trajectory_data import DatasetAdapter, create_dataset

from segment_filter_core import (
    ClipVisionEncoder,
    NumericNormalizers,
    PCAProjector,
    RawQuality,
    candidate_windows,
    fuse_fragment_features as _fuse_fragment_features,
    raw_quality,
    reference_windows,
    score_quality,
    temporal_pool,
    visual_fragment_feature,
)

from .coverage import coverage_scores
from .novelty import novelty_scores
from .scoring import compute_sqcn


VERSION = "0.3.0"
SCORE_COLUMNS = (
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "novelty",
    "sqcn",
)
SEGMENT_COLUMNS = (
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
)


FragmentKey = tuple[int, int, int]


@dataclass
class _FragmentFeature:
    metadata: dict[str, Any]
    visual_raw: np.ndarray
    action_pooled: np.ndarray
    state_pooled: np.ndarray
    progress: float
    raw_quality: RawQuality


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "dataset",
        "encoder",
        "quality",
        "coverage",
        "novelty",
        "runtime",
        "output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"SQCN config is missing sections: {missing}")
    encoder = config["encoder"]
    if "embedding_dim" in encoder:
        raise ValueError(
            "SQCN encoder.embedding_dim was removed; the final dimension is derived "
            "from the fused features"
        )
    if int(encoder.get("visual_dim", -1)) != 128:
        raise ValueError("SQCN encoder.visual_dim must be 128")
    if not str(encoder.get("model", "")).strip():
        raise ValueError("SQCN encoder.model must name a local CLIP ViT")
    if encoder.get("local_files_only") is not True:
        raise ValueError("SQCN encoder.local_files_only must be true")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one SQCN YAML configuration."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("SQCN config root must be a mapping")
    _validate_config(config)
    return config


def _fingerprint(config: Mapping[str, Any], adapter: DatasetAdapter) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(
        f"sqcn-{VERSION}:{adapter.fingerprint()}:{payload}".encode("utf-8")
    ).hexdigest()


def _cache_is_valid(root: Path, fingerprint: str) -> bool:
    manifest_path = root / "run_manifest.json"
    required = (
        root / "fragment" / "scores.csv",
        root / "fragment" / "embeddings.npy",
        root / "fragment" / "features.pkl",
        root / "reference" / "segments.csv",
        root / "reference" / "embeddings.npy",
        root / "encoder_artifacts" / "visual_pca.pkl",
        root / "encoder_artifacts" / "numeric_normalizers.pkl",
    )
    if not manifest_path.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("fingerprint") == fingerprint and payload.get("status") == "complete"


def _fragment_metadata(
    episode_id: int,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    return {
        "sample_id": (
            f"ep{episode_id:06d}_fragment_{start_step:06d}_{end_step:06d}"
        ),
        "episode_id": int(episode_id),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "length": 15,
    }


def _write_metadata_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in SEGMENT_COLUMNS} for row in rows)


def _write_scores(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{float(row[key]):.9f}"
                        if key in {"quality", "coverage", "novelty", "sqcn"}
                        else row[key]
                    )
                    for key in SCORE_COLUMNS
                }
            )


def _json_bounds(
    details: Mapping[str, np.ndarray | tuple[float, float]],
) -> dict[str, list[float]]:
    return {
        key: [float(value[0]), float(value[1])]
        for key, value in details.items()
        if key.endswith("_bounds") and isinstance(value, tuple)
    }


def _build_fragment_features(
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
    normalizers: NumericNormalizers,
) -> tuple[
    dict[FragmentKey, _FragmentFeature],
    list[FragmentKey],
    list[FragmentKey],
    int,
]:
    image_key = adapter.image_observation_keys[0]
    workers = int(config["runtime"].get("num_workers", 0))
    max_episodes_value = config["runtime"].get("max_episodes")
    max_episodes = int(max_episodes_value) if max_episodes_value is not None else None
    vision = ClipVisionEncoder(config["encoder"])
    union: dict[FragmentKey, _FragmentFeature] = {}
    candidate_keys: list[FragmentKey] = []
    reference_keys: list[FragmentKey] = []
    skipped_short = 0

    for episode in adapter.iter_episodes(
        num_workers=workers,
        max_episodes=max_episodes,
        load_images=True,
    ):
        candidates = candidate_windows(episode.length)
        references = reference_windows(episode.length)
        if not candidates:
            skipped_short += 1
            continue
        if image_key not in episode.observations:
            raise ValueError(f"episode {episode.episode_id} is missing image {image_key!r}")
        frame_features = vision.encode(episode.observations[image_key])
        if len(frame_features) != episode.length:
            raise ValueError(
                f"episode {episode.episode_id}: CLIP features={len(frame_features)}, "
                f"frames={episode.length}"
            )
        normalized_actions = normalizers.action(episode.actions)
        normalized_state = normalizers.state(episode.observations)
        windows = sorted(set(candidates) | set(references))
        window_keys: dict[tuple[int, int], FragmentKey] = {}
        for start, end in windows:
            start_step = int(episode.frame_indices[start])
            end_step = int(episode.frame_indices[end])
            key = (int(episode.episode_id), start_step, end_step)
            window_keys[(start, end)] = key
            index = slice(start, end + 1)
            actions = normalized_actions[index]
            state = normalized_state[index]
            union[key] = _FragmentFeature(
                metadata=_fragment_metadata(
                    episode.episode_id,
                    start_step,
                    end_step,
                ),
                visual_raw=visual_fragment_feature(frame_features[index]),
                action_pooled=temporal_pool(actions),
                state_pooled=temporal_pool(state),
                progress=float(start) / float(episode.length),
                raw_quality=raw_quality(actions, state),
            )
        candidate_keys.extend(window_keys[window] for window in candidates)
        reference_keys.extend(window_keys[window] for window in references)

    return union, candidate_keys, reference_keys, skipped_short


def _publish(temp_root: Path, root: Path, *, force: bool) -> None:
    if root.exists():
        if not force:
            raise FileExistsError(
                f"SQCN output exists but is not a compatible cache: {root}; pass --force"
            )
        shutil.rmtree(root)
    os.replace(temp_root, root)


def run_pipeline(
    config: Mapping[str, Any],
    *,
    force: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    """Compute SQCN for every complete 15-frame candidate fragment."""

    _validate_config(config)
    adapter = create_dataset(config["dataset"])
    if len(adapter.image_observation_keys) != 1:
        raise ValueError("SQCN requires exactly one configured image observation")
    if not adapter.vector_observation_keys:
        raise ValueError("SQCN requires at least one vector state observation")

    dataset_name = str(config["dataset"].get("name", "dataset"))
    if output_dir is None:
        output_root = Path(str(config["output"]["root"])).expanduser()
        root = output_root / dataset_name
    else:
        root = Path(output_dir).expanduser()
    fingerprint = _fingerprint(config, adapter)
    if (
        not force
        and bool(config["runtime"].get("resume", True))
        and _cache_is_valid(root, fingerprint)
    ):
        return root

    workers = int(config["runtime"].get("num_workers", 0))
    max_episodes_value = config["runtime"].get("max_episodes")
    max_episodes = int(max_episodes_value) if max_episodes_value is not None else None
    quality_config = config["quality"]
    numeric_episodes = adapter.iter_episodes(
        num_workers=workers,
        max_episodes=max_episodes,
        load_images=False,
    )
    normalizers = NumericNormalizers.fit(
        (
            (episode.actions, episode.observations)
            for episode in numeric_episodes
        ),
        adapter.vector_observation_keys,
        quantile_low=float(quality_config.get("quantile_low", 0.01)),
        quantile_high=float(quality_config.get("quantile_high", 0.99)),
        epsilon=float(quality_config.get("epsilon", 1.0e-8)),
    )

    union, candidate_keys, reference_keys, skipped_short = _build_fragment_features(
        adapter,
        config,
        normalizers,
    )
    if not candidate_keys or not reference_keys:
        raise RuntimeError("SQCN produced no complete candidate/reference fragments")
    union_keys = list(union)
    union_index = {key: index for index, key in enumerate(union_keys)}
    union_features = [union[key] for key in union_keys]
    visual_raw = np.stack([feature.visual_raw for feature in union_features])
    encoder_config = config["encoder"]
    seed = int(config["runtime"].get("seed", 42))
    visual_dim = int(encoder_config["visual_dim"])
    visual_projector = PCAProjector(output_dim=visual_dim, seed=seed)
    visual_embeddings = visual_projector.fit_transform(
        visual_raw,
        max_samples=encoder_config.get("pca_fit_max_samples"),
    )
    fused_raw, union_embeddings = _fuse_fragment_features(
        visual_embeddings,
        np.stack([feature.state_pooled for feature in union_features]),
        np.stack([feature.action_pooled for feature in union_features]),
        np.asarray([feature.progress for feature in union_features], dtype=np.float32),
    )
    candidate_indices = np.asarray(
        [union_index[key] for key in candidate_keys],
        dtype=np.int64,
    )
    reference_indices = np.asarray(
        [union_index[key] for key in reference_keys],
        dtype=np.int64,
    )
    candidate_embeddings = union_embeddings[candidate_indices]
    reference_embeddings = union_embeddings[reference_indices]

    quality_values, quality_details = score_quality(
        [union[key].raw_quality for key in candidate_keys],
        quantile_low=float(quality_config.get("quantile_low", 0.01)),
        quantile_high=float(quality_config.get("quantile_high", 0.99)),
        epsilon=float(quality_config.get("epsilon", 1.0e-8)),
    )
    coverage_values, coverage_raw, coverage_bounds, sigma = coverage_scores(
        candidate_embeddings,
        reference_embeddings,
        config["coverage"],
        seed=seed,
    )
    novelty_values, novelty_raw, novelty_bounds = novelty_scores(
        candidate_embeddings,
        config["novelty"],
    )
    sqcn_values = compute_sqcn(quality_values, coverage_values, novelty_values)
    sample_ids = np.asarray([union[key].metadata["sample_id"] for key in candidate_keys])
    order = np.lexsort((sample_ids, -sqcn_values))

    root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.sqcn-", dir=root.parent)
    )
    try:
        fragment_root = temp_root / "fragment"
        reference_root = temp_root / "reference"
        artifact_root = temp_root / "encoder_artifacts"
        fragment_root.mkdir(parents=True, exist_ok=True)
        reference_root.mkdir(parents=True, exist_ok=True)
        sorted_embeddings = candidate_embeddings[order]
        np.save(fragment_root / "embeddings.npy", sorted_embeddings)
        rows: list[dict[str, Any]] = []
        for index in order:
            rows.append(
                {
                    **union[candidate_keys[int(index)]].metadata,
                    "quality": float(quality_values[index]),
                    "coverage": float(coverage_values[index]),
                    "novelty": float(novelty_values[index]),
                    "sqcn": float(sqcn_values[index]),
                }
            )
        _write_scores(fragment_root / "scores.csv", rows)
        feature_payload = {
            "version": VERSION,
            "fingerprint": fingerprint,
            "metadata": [union[candidate_keys[int(index)]].metadata for index in order],
            "fused_raw": fused_raw[candidate_indices][order],
            "raw_quality": [union[candidate_keys[int(index)]].raw_quality for index in order],
            "quality_details": {
                key: value[order] if isinstance(value, np.ndarray) else value
                for key, value in quality_details.items()
            },
            "coverage_raw": coverage_raw[order],
            "coverage_bounds": coverage_bounds,
            "novelty_raw": novelty_raw[order],
            "novelty_bounds": novelty_bounds,
        }
        with (fragment_root / "features.pkl").open("wb") as handle:
            pickle.dump(feature_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(reference_root / "embeddings.npy", reference_embeddings)
        _write_metadata_csv(
            reference_root / "segments.csv",
            [union[key].metadata for key in reference_keys],
        )
        visual_projector.save(artifact_root / "visual_pca.pkl")
        artifact_root.mkdir(parents=True, exist_ok=True)
        with (artifact_root / "numeric_normalizers.pkl").open("wb") as handle:
            pickle.dump(normalizers, handle, protocol=pickle.HIGHEST_PROTOCOL)

        candidate_set = set(candidate_keys)
        reference_set = set(reference_keys)
        manifest = {
            "version": VERSION,
            "status": "complete",
            "fingerprint": fingerprint,
            "config": json.loads(json.dumps(config)),
            "dataset_name": dataset_name,
            "dataset_path": str(config["dataset"].get("path", "")),
            "clip_model": str(encoder_config["model"]),
            "fragment_length": 15,
            "candidate_stride": 15,
            "visual_embedding_dim": visual_dim,
            "embedding_dim": int(union_embeddings.shape[1]),
            "weights": {"quality": 0.8, "coverage": 0.1, "novelty": 0.1},
            "counts": {
                "candidate_fragments": len(candidate_keys),
                "reference_fragments": len(reference_keys),
                "overlap_fragments": len(candidate_set & reference_set),
                "pca_union_fragments": len(union_keys),
                "skipped_short_episodes": skipped_short,
            },
            "quality_bounds": _json_bounds(quality_details),
            "coverage_bounds": [float(coverage_bounds[0]), float(coverage_bounds[1])],
            "novelty_bounds": [float(novelty_bounds[0]), float(novelty_bounds[1])],
            "coverage_sigma": float(sigma),
            "outputs": {
                "scores": str(root / "fragment" / "scores.csv"),
                "candidate_embeddings": str(root / "fragment" / "embeddings.npy"),
                "reference_segments": str(root / "reference" / "segments.csv"),
                "reference_embeddings": str(root / "reference" / "embeddings.npy"),
            },
        }
        (temp_root / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _publish(temp_root, root, force=force)
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise
    return root
