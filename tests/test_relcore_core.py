from __future__ import annotations

import random

import numpy as np
import pytest

from relcore.data.index import build_clip_records, candidate_windows
from relcore.config import resolve_config
from relcore.features.normalization import RobustNormalizer
from relcore.features.relation_encoder import RelationEncoder
from relcore.utils.random import seed_everything
from trajectory_data import EpisodeRecord


def test_candidate_windows_match_sqcn_tail_alignment_for_every_small_length():
    from sqcn.sampling import candidate_windows as sqcn_candidate_windows

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


def test_seed_everything_resets_python_and_numpy_generators():
    seed_everything(91)
    first = (random.random(), np.random.random())
    seed_everything(91)

    assert (random.random(), np.random.random()) == first
