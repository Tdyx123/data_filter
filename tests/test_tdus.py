from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from tdus.coverage import CoverageModel, coverage, coverage_gain, mmd_squared
from tdus.dataset import (
    DatasetAdapter,
    EpisodeRecord,
    TrajectorySegment,
    aligned_chunk_windows,
    register_dataset_adapter,
)
from tdus.diversity import sample_diversity, subset_diversity
from tdus.encoder import NumericNormalizers, PCAProjector, temporal_pool
from tdus.novelty import novelty_scores
from tdus.quality import RawQuality, raw_quality, score_quality
from tdus.selector import select_budget, select_top_k
from tdus.tdus import compute_tdus, run_pipeline


def make_segment(
    episode: int,
    state: np.ndarray,
    action: np.ndarray,
    *,
    kind: str = "trajectory",
    start: int = 0,
) -> TrajectorySegment:
    end = start + len(action) - 1
    return TrajectorySegment(
        sample_id=(
            f"ep{episode:06d}_trajectory"
            if kind == "trajectory"
            else f"ep{episode:06d}_chunk_{start:06d}_{end:06d}"
        ),
        episode_id=episode,
        start_step=start,
        end_step=end,
        timestamps=np.arange(len(action), dtype=np.float64) / 5.0,
        observations={"observation.state": np.asarray(state, dtype=np.float32)},
        actions=np.asarray(action, dtype=np.float32),
        kind=kind,
    )


def test_chunk_windows_keep_short_and_align_tail():
    assert aligned_chunk_windows(5, 8, 4) == [(0, 4)]
    assert aligned_chunk_windows(100, 32, 16) == [
        (0, 31),
        (16, 47),
        (32, 63),
        (48, 79),
        (64, 95),
        (68, 99),
    ]


def test_pooling_and_projector_have_fixed_dimension():
    sequence = np.asarray([[1, 2], [3, 6]], dtype=np.float32)
    pooled = temporal_pool(sequence)
    np.testing.assert_allclose(pooled, [2, 4, 1, 2, 3, 6])
    features = np.asarray(
        [[1, 2, 3], [2, 1, 0], [4, 3, 2], [0, 1, 4]], dtype=np.float32
    )
    embeddings = PCAProjector(output_dim=128, seed=1).fit_transform(features)
    assert embeddings.shape == (4, 128)
    assert np.all(np.isfinite(embeddings))


def test_quality_components_are_bounded_and_smooth_actions_score_higher():
    smooth = make_segment(
        0,
        np.asarray([[0, 0], [0.2, 0], [0.4, 0]], dtype=np.float32),
        np.asarray([[0, 0], [0.1, 0], [0.2, 0]], dtype=np.float32),
    )
    rough = make_segment(
        1,
        np.asarray([[0, 0], [0, 0], [1, 0]], dtype=np.float32),
        np.asarray([[0, 0], [1, 1], [0, 0]], dtype=np.float32),
    )
    normalizers = NumericNormalizers.fit(
        [smooth, rough], ["observation.state"], quantile_low=0, quantile_high=1
    )
    raw = [raw_quality(item, normalizers) for item in (smooth, rough)]
    quality, details = score_quality(
        raw,
        {
            "action_smooth_weight": 0.4,
            "state_transition_weight": 0.3,
            "motion_efficiency_weight": 0.3,
            "quantile_low": 0,
            "quantile_high": 1,
            "epsilon": 1e-8,
        },
    )
    assert np.all((quality >= 0) & (quality <= 1))
    assert details["action_smooth"][0] > details["action_smooth"][1]


def test_blockwise_mmd_matches_dense_formula_and_gain():
    reference = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    subset = np.asarray([[0.5], [1.5]], dtype=np.float32)
    sigma = 0.7
    dense_dd = np.exp(-((reference - reference.T) ** 2) / (2 * sigma**2)).mean()
    dense_ss = np.exp(-((subset - subset.T) ** 2) / (2 * sigma**2)).mean()
    dense_ds = np.exp(
        -((reference[:, None, :] - subset[None, :, :]) ** 2).sum(axis=2)
        / (2 * sigma**2)
    ).mean()
    expected = max(float(dense_dd + dense_ss - 2 * dense_ds), 0.0)
    actual = mmd_squared(reference, subset, sigma, batch_size=2, device="cpu")
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    sample = np.asarray([2.0], dtype=np.float32)
    gain = coverage_gain(
        reference, subset, sample, sigma=sigma, batch_size=2, device="cpu"
    )
    direct = coverage(
        reference,
        np.concatenate([subset, sample.reshape(1, -1)]),
        sigma=sigma,
        batch_size=2,
        device="cpu",
    ) - coverage(reference, subset, sigma=sigma, batch_size=2, device="cpu")
    np.testing.assert_allclose(gain, direct, atol=1e-6)


def test_diversity_and_novelty_identify_duplicates_and_outlier():
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]],
        dtype=np.float32,
    )
    diversity = sample_diversity(
        embeddings, {"reference_size": 4, "batch_size": 2}, seed=2
    )
    assert diversity[-1] > diversity[0]
    assert subset_diversity(embeddings[:2]) == 0.0
    novelty, raw, _ = novelty_scores(
        embeddings,
        {
            "k": 1,
            "backend": "numpy",
            "batch_size": 2,
            "quantile_low": 0,
            "quantile_high": 1,
        },
    )
    assert raw[-1] > raw[0]
    assert novelty[-1] == 1.0


def test_tdus_weights_and_selectors(tmp_path: Path):
    values = compute_tdus(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 0.0]),
        {"quality": 0.25, "coverage": 0.35, "diversity": 0.25, "novelty": 0.15},
    )
    np.testing.assert_allclose(values, [0.25, 0.35])
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "episode_id": [0, 1, 2],
            "start_step": [0, 0, 0],
            "end_step": [2, 3, 4],
            "length": [3, 4, 5],
            "quality": [0.9, 0.6, 0.4],
            "coverage": [0.5, 0.6, 0.7],
            "diversity": [0.2, 0.5, 0.8],
            "novelty": [0.2, 0.4, 0.9],
            "tdus": [0.7, 0.6, 0.8],
        }
    )
    assert select_top_k(2, frame)["sample_id"].tolist() == ["c", "a"]
    embeddings = np.asarray([[1, 0], [0.8, 0.2], [-1, 0]], dtype=np.float32)
    selected = select_budget(
        7,
        frame,
        embeddings,
        embeddings,
        coverage_config={"sigma": 1.0, "batch_size": 2, "device": "cpu"},
        weights={"quality": 0.25, "coverage": 0.35, "diversity": 0.25, "novelty": 0.15},
        output_path=tmp_path / "selection.csv",
    )
    assert selected["length"].sum() <= 7
    assert (tmp_path / "selection.csv").is_file()
    assert selected["selection_order"].tolist() == list(range(1, len(selected) + 1))
    model = CoverageModel.fit(
        embeddings, {"sigma": 1.0, "batch_size": 2, "device": "cpu"}
    )
    lookup = {sample_id: index for index, sample_id in enumerate(frame["sample_id"])}
    chosen: list[int] = []
    for _, row in selected.iterrows():
        chosen.append(lookup[row["sample_id"]])
        chosen_frame = frame.iloc[chosen]
        expected = (
            0.25
            * float((chosen_frame["length"] * chosen_frame["quality"]).sum())
            / 7
            + 0.35 * model.score_subset(embeddings[chosen])
            + 0.25 * subset_diversity(embeddings[chosen])
            + 0.15
            * float((chosen_frame["length"] * chosen_frame["novelty"]).sum())
            / 7
        )
        np.testing.assert_allclose(row["subset_tdus"], expected, atol=1e-6)


class SyntheticAdapter(DatasetAdapter):
    def __init__(self, _: Mapping[str, Any]):
        self._records = (EpisodeRecord(0, 5), EpisodeRecord(1, 6), EpisodeRecord(2, 4))
        self._segments = [
            make_segment(
                record.episode_id,
                np.stack(
                    [
                        np.linspace(0, 1 + record.episode_id, record.length),
                        np.linspace(1, 0, record.length),
                    ],
                    axis=1,
                ),
                np.stack(
                    [
                        np.linspace(0, 0.5 + record.episode_id, record.length),
                        np.linspace(0.2, 0.8, record.length),
                    ],
                    axis=1,
                ),
            )
            for record in self._records
        ]

    @property
    def vector_observation_keys(self) -> tuple[str, ...]:
        return ("observation.state",)

    @property
    def image_observation_keys(self) -> tuple[str, ...]:
        return ()

    def episodes(self) -> Sequence[EpisodeRecord]:
        return self._records

    def iter_segments(
        self,
        modes: Sequence[str],
        *,
        chunk_length: int,
        stride: int,
        num_workers: int = 0,
        max_episodes: int | None = None,
        load_images: bool = True,
    ) -> Iterator[TrajectorySegment]:
        del num_workers, load_images
        trajectories = self._segments[:max_episodes] if max_episodes else self._segments
        for trajectory in trajectories:
            if "trajectory" in modes:
                yield trajectory
            if "chunk" in modes:
                for start, end in aligned_chunk_windows(
                    trajectory.length, chunk_length, stride
                ):
                    index = slice(start, end + 1)
                    yield make_segment(
                        trajectory.episode_id,
                        trajectory.observations["observation.state"][index],
                        trajectory.actions[index],
                        kind="chunk",
                        start=start,
                    )

    def fingerprint(self) -> str:
        return "synthetic-v1"


def test_end_to_end_pipeline_with_registered_adapter(tmp_path: Path):
    register_dataset_adapter("synthetic_test", SyntheticAdapter)
    config = {
        "dataset": {
            "type": "synthetic_test",
            "name": "synthetic",
            "path": str(tmp_path),
        },
        "segmentation": {
            "modes": ["trajectory", "chunk"],
            "chunk_length": 3,
            "stride": 2,
            "max_episodes": None,
        },
        "encoder": {
            "embedding_dim": 128,
            "pca_fit_max_samples": None,
            "vision_backend": "pixels",
        },
        "quality": {
            "action_smooth_weight": 0.4,
            "state_transition_weight": 0.3,
            "motion_efficiency_weight": 0.3,
            "quantile_low": 0,
            "quantile_high": 1,
            "epsilon": 1e-8,
        },
        "coverage": {
            "sigma": "median",
            "sigma_sample_size": 100,
            "reference_max_samples": None,
            "batch_size": 2,
            "device": "cpu",
        },
        "diversity": {"reference_size": 100, "batch_size": 2},
        "novelty": {
            "k": 2,
            "backend": "numpy",
            "batch_size": 2,
            "quantile_low": 0,
            "quantile_high": 1,
        },
        "weights": {
            "quality": 0.25,
            "coverage": 0.35,
            "diversity": 0.25,
            "novelty": 0.15,
        },
        "runtime": {"seed": 42, "num_workers": 0, "resume": True},
        "output": {"root": str(tmp_path / "outputs")},
        "analysis": {"primary_mode": "chunk"},
    }
    outputs = run_pipeline(config)
    assert set(outputs) == {"trajectory", "chunk"}
    for mode, root in outputs.items():
        assert np.load(root / "embeddings.npy").shape[1] == 128
        assert (root / "features.pkl").is_file()
        with (root / "tdus_scores.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        scores = [float(row["tdus"]) for row in rows]
        assert scores == sorted(scores, reverse=True)
        assert (root / "tdus_scores.csv").read_bytes() == (root / "scores.csv").read_bytes()
    # Compatible outputs should be reused rather than rewritten.
    first_mtime = (outputs["chunk"] / "scores.csv").stat().st_mtime_ns
    run_pipeline(config)
    assert (outputs["chunk"] / "scores.csv").stat().st_mtime_ns == first_mtime
