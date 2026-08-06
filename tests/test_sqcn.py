from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import segment_filter_core.encoding as encoding
from segment_filter_core import (
    ClipVisionEncoder,
    NumericNormalizers,
    PCAProjector,
    candidate_windows,
    raw_quality,
    reference_sample_count,
    reference_windows,
    score_quality,
    temporal_pool,
    visual_fragment_feature,
)
from sqcn.cli import build_parser
from sqcn.coverage import coverage_scores, rbf_kernel
from sqcn.novelty import novelty_scores
from sqcn.scoring import compute_sqcn
from sqcn.pipeline import load_config


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (14, 0),
        (15, 1),
        (29, 1),
        (30, 2),
        (90, 6),
        (91, 6),
        (180, 9),
        (181, 9),
    ],
)
def test_reference_sample_count_obeys_piecewise_half_up_rule(
    length: int,
    expected: int,
):
    assert reference_sample_count(length) == expected


def test_reference_windows_cover_endpoints_and_evenly_round_middle_starts():
    assert reference_windows(29) == [(0, 14)]
    assert reference_windows(91) == [
        (0, 14),
        (15, 29),
        (30, 44),
        (46, 60),
        (61, 75),
        (76, 90),
    ]


def test_candidate_windows_use_stride_fifteen_and_align_the_tail():
    assert candidate_windows(14) == []
    assert candidate_windows(58) == [
        (0, 14),
        (15, 29),
        (30, 44),
        (43, 57),
    ]


def test_visual_fragment_feature_uses_sum_and_last_minus_first():
    features = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (15, 1))
    features[-1] = np.asarray([0.0, 1.0], dtype=np.float32)

    encoded = visual_fragment_feature(features)

    np.testing.assert_allclose(encoded, [14.0, 1.0, -1.0, 1.0])


def test_temporal_pool_and_pca_have_fixed_dimensions_without_implicit_l2():
    sequence = np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(
        temporal_pool(sequence),
        [2.0, 4.0, 1.0, 2.0, 3.0, 6.0],
    )
    features = np.asarray(
        [[1, 2, 3], [2, 1, 0], [4, 3, 2], [0, 1, 4]],
        dtype=np.float32,
    )

    visual = PCAProjector(output_dim=128, seed=7).fit_transform(features)

    assert visual.shape == (4, 128)

    projector = PCAProjector(output_dim=2)
    projector.mean_ = np.zeros(2, dtype=np.float32)
    projector.scale_ = np.ones(2, dtype=np.float32)
    projector.components_ = np.eye(2, dtype=np.float32)
    np.testing.assert_allclose(
        projector.transform(np.asarray([[3.0, 4.0]], dtype=np.float32)),
        [[3.0, 4.0]],
    )


def test_l2_normalize_rows_handles_regular_zero_and_non_finite_rows():
    normalize = getattr(encoding, "l2_normalize_rows")

    np.testing.assert_allclose(
        normalize(np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)),
        [[0.6, 0.8], [0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="finite"):
        normalize(np.asarray([[np.nan, 1.0]], dtype=np.float32))


def test_numeric_normalizers_scale_actions_and_join_vector_state():
    actions = [
        np.asarray([[0.0, 10.0], [10.0, 20.0]], dtype=np.float32),
        np.asarray([[5.0, 15.0]], dtype=np.float32),
    ]
    observations = [
        {
            "state.a": np.asarray([[0.0], [2.0]], dtype=np.float32),
            "state.b": np.asarray([[10.0], [20.0]], dtype=np.float32),
        },
        {
            "state.a": np.asarray([[1.0]], dtype=np.float32),
            "state.b": np.asarray([[15.0]], dtype=np.float32),
        },
    ]
    normalizers = NumericNormalizers.fit(
        zip(actions, observations, strict=True),
        ("state.a", "state.b"),
        quantile_low=0,
        quantile_high=1,
    )

    np.testing.assert_allclose(
        normalizers.action(np.asarray([[5.0, 15.0]], dtype=np.float32)),
        [[0.5, 0.5]],
    )
    np.testing.assert_allclose(
        normalizers.state(observations[0]),
        [[0.0, 0.0], [1.0, 1.0]],
    )


def test_clip_encoder_fails_instead_of_falling_back_for_unloadable_local_model():
    with pytest.raises(RuntimeError, match="CLIP ViT could not be loaded"):
        ClipVisionEncoder(
            {
                "model": "/definitely/missing/clip-vit",
                "local_files_only": True,
                "device": "cpu",
            }
        )


def test_quality_rewards_smoother_actions_with_existing_component_formula():
    smooth_actions = np.asarray([[0, 0], [0.1, 0], [0.2, 0]], dtype=np.float32)
    rough_actions = np.asarray([[0, 0], [1, 1], [0, 0]], dtype=np.float32)
    smooth_state = np.asarray([[0, 0], [0.2, 0], [0.4, 0]], dtype=np.float32)
    rough_state = np.asarray([[0, 0], [0, 0], [1, 0]], dtype=np.float32)

    values = [
        raw_quality(smooth_actions, smooth_state),
        raw_quality(rough_actions, rough_state),
    ]
    quality, details = score_quality(
        values,
        quantile_low=0,
        quantile_high=1,
    )

    assert np.all((quality >= 0) & (quality <= 1))
    assert details["action_smooth"][0] > details["action_smooth"][1]
    expected = (
        0.4 * details["action_smooth"]
        + 0.3 * details["state_transition"]
        + 0.3 * details["motion_efficiency"]
    )
    np.testing.assert_allclose(quality, expected)


def test_coverage_is_mean_rbf_affinity_then_robustly_scaled():
    reference = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    candidates = np.asarray([[0.0], [0.5], [2.0]], dtype=np.float32)
    sigma = 0.7
    expected_raw = rbf_kernel(candidates, reference, sigma).mean(axis=1)

    normalized, raw, bounds, actual_sigma = coverage_scores(
        candidates,
        reference,
        {
            "sigma": sigma,
            "batch_size": 2,
            "device": "cpu",
            "quantile_low": 0,
            "quantile_high": 1,
        },
        seed=3,
    )

    np.testing.assert_allclose(raw, expected_raw, atol=1e-6)
    assert actual_sigma == sigma
    assert bounds == pytest.approx((float(expected_raw.min()), float(expected_raw.max())))
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_novelty_excludes_self_and_identifies_an_outlier():
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]],
        dtype=np.float32,
    )

    novelty, raw, bounds = novelty_scores(
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
    assert bounds == pytest.approx((float(raw.min()), float(raw.max())))


def test_sqcn_uses_fixed_weights_and_rejects_non_finite_components():
    result = compute_sqcn(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0]),
    )
    np.testing.assert_allclose(result, [0.8, 0.2])

    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        compute_sqcn(
            np.asarray([np.nan]),
            np.asarray([0.5]),
            np.asarray([0.5]),
        )


def test_libero90_config_and_cli_keep_sqcn_dimensions_and_entrypoint():
    config = load_config(
        Path(__file__).resolve().parents[1] / "sqcn" / "config_libero90.yaml"
    )

    assert config["dataset"]["name"] == "libero90"
    assert config["encoder"]["visual_dim"] == 128
    assert "embedding_dim" not in config["encoder"]
    args = build_parser().parse_args(
        [
            "--config",
            "custom.yaml",
            "--output-dir",
            "/tmp/custom-sqcn-run",
            "--max-episodes",
            "3",
            "--force",
        ]
    )
    assert args.config == "custom.yaml"
    assert args.output_dir == "/tmp/custom-sqcn-run"
    assert args.max_episodes == 3
    assert args.force is True


@pytest.mark.real_data
def test_libero90_metadata_matches_sqcn_fragment_scale_baseline():
    episodes_path = Path(
        "/data/dwb/datasets/LIBERO_lerobot/libero90/meta/episodes.jsonl"
    )
    if not episodes_path.is_file():
        pytest.skip("LIBERO-90 metadata is not mounted")
    rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate: set[tuple[int, int, int]] = set()
    reference: set[tuple[int, int, int]] = set()
    for row in rows:
        episode_id = int(row["episode_index"])
        length = int(row["length"])
        candidate.update(
            (episode_id, start, end) for start, end in candidate_windows(length)
        )
        reference.update(
            (episode_id, start, end) for start, end in reference_windows(length)
        )

    assert len(reference) == 35_164
    assert len(candidate) == 46_705
    assert len(candidate & reference) == 10_651
    assert len(candidate | reference) == 71_218
