from __future__ import annotations

import random

import numpy as np
import pytest

from relcore import scoring
from relcore.data.index import build_clip_records, candidate_windows
from relcore.config import resolve_config
from relcore.features.normalization import RobustNormalizer
from relcore.features.relation_encoder import RelationEncoder
from relcore.utils.random import seed_everything
from trajectory_data import EpisodeRecord


def test_candidate_windows_match_sqcn_tail_alignment_for_every_small_length():
    from segment_filter_core import candidate_windows as sqcn_candidate_windows

    for length in range(301):
        assert candidate_windows(length, length=15, stride=15) == (sqcn_candidate_windows(length))


def test_clip_records_use_inclusive_sqcn_ids_and_only_link_contiguous_windows():
    records = build_clip_records(
        [EpisodeRecord(3, 58, 9, "put the bowl on the plate")],
        length=15,
        stride=15,
    )

    assert [record.sample_id for record in records] == [
        "ep000003_fragment_000000_000014",
        "ep000003_fragment_000015_000029",
        "ep000003_fragment_000030_000044",
        "ep000003_fragment_000043_000057",
    ]
    assert (records[1].previous_sample_id, records[1].next_sample_id) == (
        records[0].sample_id,
        records[2].sample_id,
    )
    assert records[-1].previous_sample_id is None
    assert records[-1].next_sample_id is None


def test_clip_records_require_task_metadata_for_hard_quotas():
    with pytest.raises(ValueError, match="task metadata"):
        build_clip_records([EpisodeRecord(0, 30)], length=15, stride=15)


def test_robust_normalizer_uses_per_dimension_median_and_iqr_without_clipping():
    normalizer = RobustNormalizer.fit(
        [
            np.asarray([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32),
            np.asarray([[4.0, 30.0]], dtype=np.float32),
        ],
        epsilon=1.0e-6,
    )

    np.testing.assert_allclose(normalizer.median, [2.0, 20.0])
    np.testing.assert_allclose(normalizer.iqr, [2.0, 10.0])
    np.testing.assert_allclose(
        normalizer.transform(np.asarray([[6.0, 40.0]], dtype=np.float32)),
        [[2.0, 2.0]],
        atol=1.0e-6,
    )


def test_relation_encoder_distinguishes_forward_and_reversed_action_sequences():
    steps = np.arange(15, dtype=np.float32)
    visual = np.stack([steps, steps**2], axis=1)
    state = np.stack([steps / 10.0, np.sin(steps)], axis=1).astype(np.float32)
    actions = np.stack([steps, np.cos(steps)], axis=1).astype(np.float32)
    encoder = RelationEncoder(projection_dim=4, lags=(0, 1, 2, 4), seed=7)

    forward = encoder.encode_raw(visual, state, actions)
    reversed_actions = encoder.encode_raw(visual, state, actions[::-1].copy())

    assert forward.ndim == 1
    assert forward.shape == reversed_actions.shape
    assert not np.allclose(forward, reversed_actions)
    np.testing.assert_allclose(
        encoder.encode_raw(visual, state, actions),
        forward,
    )


def test_config_locks_sqcn_windows_and_local_production_clip():
    with pytest.raises(ValueError, match="15-frame"):
        resolve_config({"clip": {"length": 16, "stride": 15}})
    with pytest.raises(ValueError, match="fixed local model"):
        resolve_config({"visual": {"model": "/tmp/another-clip"}})


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (["non_noop"], 1),
        (["smoothness"], 2),
        (["progress", "non_noop"], 5),
        (["support", "progress", "smoothness", "non_noop"], 15),
        (["non_noop", "progress"], 5),
    ],
)
def test_reliability_metric_mask_uses_canonical_four_bit_order(metrics, expected):
    assert scoring.reliability_metric_mask(metrics) == expected


def test_config_has_no_reliability_metric_default():
    assert "reliability_metrics" not in resolve_config({})["quality"]


def test_config_defaults_to_kmeans_prototype_method():
    assert resolve_config({})["prototypes"]["method"] == "kmeans"


@pytest.mark.parametrize("method", ["kmeans", "motion_primitives"])
def test_config_accepts_supported_prototype_methods(method: str):
    assert resolve_config({"prototypes": {"method": method}})["prototypes"]["method"] == method


def test_config_rejects_unknown_prototype_method():
    with pytest.raises(ValueError, match="prototypes.method"):
        resolve_config({"prototypes": {"method": "unknown"}})


def test_config_rejects_removed_reliability_metrics_field():
    with pytest.raises(ValueError, match="removed.*--reliability-metrics"):
        resolve_config({"quality": {"reliability_metrics": ["progress"]}})


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (["sequence"], 1),
        (["cooccurrence"], 2),
        (["transition"], 4),
        (["cooccurrence", "sequence"], 3),
        (["transition", "sequence"], 5),
        (["transition", "cooccurrence"], 6),
        (["sequence", "transition", "cooccurrence"], 7),
    ],
)
def test_prototype_gain_metric_mask_uses_canonical_three_bit_order(metrics, expected):
    from relcore.selection import prototype_gain_metric_mask

    assert prototype_gain_metric_mask(metrics) == expected


@pytest.mark.parametrize(
    "metrics",
    [[], ["transition", "transition"], ["transition", "unknown"], "transition", [1]],
)
def test_prototype_gain_metrics_reject_invalid_values(metrics):
    from relcore.selection import normalize_prototype_gain_metrics

    with pytest.raises(ValueError):
        normalize_prototype_gain_metrics(metrics)


def test_config_has_no_prototype_gain_metric_default():
    assert "prototype_gain_metrics" not in resolve_config({})["objective"]


def test_config_rejects_prototype_gain_metrics_field():
    with pytest.raises(ValueError, match="objective.prototype_gain_metrics.*--prototype-gain-metrics"):
        resolve_config({"objective": {"prototype_gain_metrics": ["transition"]}})


def test_seed_everything_resets_python_and_numpy_generators():
    seed_everything(91)
    first = (random.random(), np.random.random())
    seed_everything(91)

    assert (random.random(), np.random.random()) == first
