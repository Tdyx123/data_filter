"""End-to-end TDUS computation and command-line entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .coverage import CoverageModel
from .dataset import DatasetAdapter, create_dataset
from .diversity import sample_diversity
from .encoder import (
    NumericNormalizers,
    PCAProjector,
    TrajectoryEncoder,
    uniformly_sample_indices,
)
from .novelty import novelty_scores
from .quality import RawQuality, raw_quality, score_quality


SCORE_COLUMNS = [
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "diversity",
    "novelty",
    "tdus",
]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    _validate_config(config)
    config["_config_path"] = str(path)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "dataset",
        "segmentation",
        "encoder",
        "quality",
        "coverage",
        "diversity",
        "novelty",
        "weights",
        "runtime",
        "output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {missing}")
    weights = config["weights"]
    names = ("quality", "coverage", "diversity", "novelty")
    if any(float(weights.get(name, -1)) < 0 for name in names):
        raise ValueError("TDUS weights must be non-negative")
    if not np.isclose(sum(float(weights[name]) for name in names), 1.0):
        raise ValueError("TDUS weights must sum to 1")


def compute_tdus(
    quality: np.ndarray,
    coverage: np.ndarray,
    diversity: np.ndarray,
    novelty: np.ndarray,
    weights: Mapping[str, float],
) -> np.ndarray:
    """Aggregate four [0,1] utilities into the final [0,1] TDUS."""

    result = (
        float(weights["quality"]) * np.asarray(quality)
        + float(weights["coverage"]) * np.asarray(coverage)
        + float(weights["diversity"]) * np.asarray(diversity)
        + float(weights["novelty"]) * np.asarray(novelty)
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _run_fingerprint(config: Mapping[str, Any], adapter: DatasetAdapter) -> str:
    sanitized = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(sanitized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(
        f"tdus-0.1.0:{adapter.fingerprint()}:{payload}".encode("utf-8")
    ).hexdigest()


def _fit_pixel_fallback(
    adapter: DatasetAdapter,
    encoder: TrajectoryEncoder,
    config: Mapping[str, Any],
) -> None:
    if encoder.vision is None or encoder.vision.backend != "pixels":
        return
    encoder_config = config["encoder"]
    maximum = int(encoder_config.get("fallback_fit_frames", 10_000))
    per_segment = int(encoder_config.get("max_frames_per_segment", 8))
    image_count = max(1, len(adapter.image_observation_keys))
    max_episodes = max(2, int(np.ceil(maximum / (per_segment * image_count))))
    segment_config = config["segmentation"]
    vectors: list[np.ndarray] = []
    count = 0
    for segment in adapter.iter_segments(
        ["trajectory"],
        chunk_length=int(segment_config["chunk_length"]),
        stride=int(segment_config["stride"]),
        num_workers=0,
        max_episodes=min(max_episodes, len(adapter.episodes())),
        load_images=True,
    ):
        indices = uniformly_sample_indices(segment.length, per_segment)
        for key in adapter.image_observation_keys:
            if key not in segment.observations:
                continue
            pixels = encoder.vision._resize_pixels(  # noqa: SLF001 - shared fallback
                segment.observations[key][indices],
                int(encoder_config.get("fallback_image_size", 32)),
            )
            remaining = maximum - count
            vectors.append(pixels[:remaining])
            count += min(len(pixels), remaining)
            if count >= maximum:
                break
        if count >= maximum:
            break
    if not vectors:
        raise RuntimeError("pixel fallback could not collect any image frames")
    encoder.vision.fit_pixel_fallback(
        np.concatenate(vectors), seed=int(config["runtime"].get("seed", 42))
    )


def _write_scores(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{float(row[key]):.9f}"
                        if key in {"quality", "coverage", "diversity", "novelty", "tdus"}
                        else row[key]
                    )
                    for key in SCORE_COLUMNS
                }
            )


def _cache_is_valid(root: Path, fingerprint: str, modes: Sequence[str]) -> bool:
    manifest = root / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("fingerprint") != fingerprint:
        return False
    return all(
        all(
            (root / mode / filename).is_file()
            for filename in ("embeddings.npy", "features.pkl", "tdus_scores.csv", "scores.csv")
        )
        for mode in modes
    )


def run_pipeline(config: Mapping[str, Any], *, force: bool = False) -> dict[str, Path]:
    """Compute and persist TDUS for every configured segmentation mode."""

    adapter = create_dataset(config["dataset"])
    modes = tuple(str(mode) for mode in config["segmentation"]["modes"])
    dataset_name = str(config["dataset"].get("name", "dataset"))
    root = Path(str(config["output"]["root"])).expanduser() / dataset_name
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = _run_fingerprint(config, adapter)
    if (
        not force
        and bool(config["runtime"].get("resume", True))
        and _cache_is_valid(root, fingerprint, modes)
    ):
        return {mode: root / mode for mode in modes}

    segmentation = config["segmentation"]
    max_episodes = segmentation.get("max_episodes")
    max_episodes = int(max_episodes) if max_episodes is not None else None
    chunk_length = int(segmentation["chunk_length"])
    stride = int(segmentation["stride"])
    workers = int(config["runtime"].get("num_workers", 0))
    seed = int(config["runtime"].get("seed", 42))

    # Numeric statistics use one copy of each original frame, not overlapping chunks.
    numeric_segments = adapter.iter_segments(
        ["trajectory"],
        chunk_length=chunk_length,
        stride=stride,
        num_workers=workers,
        max_episodes=max_episodes,
        load_images=False,
    )
    normalizers = NumericNormalizers.fit(
        numeric_segments,
        adapter.vector_observation_keys,
        quantile_low=float(config["quality"].get("quantile_low", 0.01)),
        quantile_high=float(config["quality"].get("quantile_high", 0.99)),
        epsilon=float(config["quality"].get("epsilon", 1.0e-8)),
    )
    encoder = TrajectoryEncoder(
        config["encoder"],
        normalizers,
        adapter.vector_observation_keys,
        adapter.image_observation_keys,
    )
    _fit_pixel_fallback(adapter, encoder, config)

    metadata_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_features_by_mode: dict[str, list[np.ndarray]] = defaultdict(list)
    raw_quality_by_mode: dict[str, list[RawQuality]] = defaultdict(list)
    segments = adapter.iter_segments(
        modes,
        chunk_length=chunk_length,
        stride=stride,
        num_workers=workers,
        max_episodes=max_episodes,
        load_images=bool(adapter.image_observation_keys),
    )
    for segment in segments:
        feature, visual_state = encoder.encode_raw(segment)
        metadata_by_mode[segment.kind].append(segment.metadata())
        raw_features_by_mode[segment.kind].append(feature)
        raw_quality_by_mode[segment.kind].append(
            raw_quality(segment, normalizers, visual_state=visual_state)
        )
    empty = [mode for mode in modes if not raw_features_by_mode[mode]]
    if empty:
        raise RuntimeError(f"No samples were produced for modes: {empty}")

    feature_arrays = {
        mode: np.stack(raw_features_by_mode[mode]).astype(np.float32) for mode in modes
    }
    union_features = np.concatenate([feature_arrays[mode] for mode in modes])
    projector = PCAProjector(
        output_dim=int(config["encoder"].get("embedding_dim", 128)), seed=seed
    )
    projector.fit(
        union_features,
        max_samples=config["encoder"].get("pca_fit_max_samples"),
    )
    embeddings_by_mode = {
        mode: projector.transform(feature_arrays[mode]) for mode in modes
    }
    projector.save(root / "encoder_artifacts" / "trajectory_pca.pkl")
    with (root / "encoder_artifacts" / "numeric_normalizers.pkl").open("wb") as handle:
        pickle.dump(normalizers, handle)

    all_raw_quality = [
        item for mode in modes for item in raw_quality_by_mode[mode]
    ]
    all_quality, all_quality_details = score_quality(all_raw_quality, config["quality"])
    quality_by_mode: dict[str, np.ndarray] = {}
    quality_details_by_mode: dict[
        str, dict[str, np.ndarray | tuple[float, float]]
    ] = {}
    offset = 0
    for mode in modes:
        count = len(raw_quality_by_mode[mode])
        quality_by_mode[mode] = all_quality[offset : offset + count]
        quality_details_by_mode[mode] = {
            key: (
                value[offset : offset + count]
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in all_quality_details.items()
        }
        offset += count

    reference_mode = "trajectory" if "trajectory" in embeddings_by_mode else modes[0]
    coverage_model = CoverageModel.fit(
        embeddings_by_mode[reference_mode], config["coverage"], seed=seed
    )
    outputs: dict[str, Path] = {}
    for mode in modes:
        embeddings = embeddings_by_mode[mode]
        coverage_values = coverage_model.score_samples(embeddings)
        diversity_values = sample_diversity(
            embeddings, config["diversity"], seed=seed
        )
        novelty_values, novelty_raw, novelty_bounds = novelty_scores(
            embeddings, config["novelty"]
        )
        tdus_values = compute_tdus(
            quality_by_mode[mode],
            coverage_values,
            diversity_values,
            novelty_values,
            config["weights"],
        )
        order = np.lexsort(
            (
                np.asarray([row["sample_id"] for row in metadata_by_mode[mode]]),
                -tdus_values,
            )
        )
        mode_root = root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        sorted_embeddings = embeddings[order]
        np.save(mode_root / "embeddings.npy", sorted_embeddings)
        rows: list[dict[str, Any]] = []
        for index in order:
            rows.append(
                {
                    **metadata_by_mode[mode][int(index)],
                    "quality": float(quality_by_mode[mode][index]),
                    "coverage": float(coverage_values[index]),
                    "diversity": float(diversity_values[index]),
                    "novelty": float(novelty_values[index]),
                    "tdus": float(tdus_values[index]),
                }
            )
        scores_path = mode_root / "tdus_scores.csv"
        _write_scores(scores_path, rows)
        shutil.copyfile(scores_path, mode_root / "scores.csv")
        feature_payload = {
            "version": "0.1.0",
            "fingerprint": fingerprint,
            "dataset_name": dataset_name,
            "mode": mode,
            "encoder_backend": encoder.backend,
            "metadata": [metadata_by_mode[mode][int(index)] for index in order],
            "raw_features": feature_arrays[mode][order],
            "raw_quality": [raw_quality_by_mode[mode][int(index)] for index in order],
            "quality_details": quality_details_by_mode[mode],
            "novelty_raw": novelty_raw[order],
            "novelty_bounds": novelty_bounds,
            "coverage_sigma": coverage_model.sigma,
        }
        with (mode_root / "features.pkl").open("wb") as handle:
            pickle.dump(feature_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        outputs[mode] = mode_root

    manifest = {
        "version": "0.1.0",
        "fingerprint": fingerprint,
        "dataset_name": dataset_name,
        "dataset_path": str(config["dataset"]["path"]),
        "modes": list(modes),
        "encoder_backend": encoder.backend,
        "embedding_dim": int(config["encoder"].get("embedding_dim", 128)),
        "coverage_reference_mode": reference_mode,
        "coverage_sigma": coverage_model.sigma,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    with (root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute model-free trajectory TDUS")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="TDUS YAML configuration",
    )
    parser.add_argument(
        "--force", action="store_true", help="ignore compatible cached outputs"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="temporary smoke-test override; algorithm parameters remain in YAML",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.max_episodes is not None:
        config["segmentation"]["max_episodes"] = args.max_episodes
    outputs = run_pipeline(config, force=args.force)
    for mode, path in outputs.items():
        print(f"{mode}: {path}")


if __name__ == "__main__":
    main()
