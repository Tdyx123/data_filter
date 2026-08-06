from __future__ import annotations

import numpy as np

from segment_filter_core import (
    RawQuality,
    candidate_windows,
    fuse_fragment_features,
    raw_quality,
    score_quality,
)


def test_public_core_exposes_sqcn_window_quality_and_fusion_contracts() -> None:
    assert candidate_windows(31) == [(0, 14), (15, 29), (16, 30)]

    raw = raw_quality(
        np.asarray([[0.0], [1.0], [1.0]], dtype=np.float32),
        np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32),
    )
    assert raw == RawQuality(
        action_delta=0.5,
        state_transition=1.5,
        motion_efficiency=1.5,
    )
    quality, _ = score_quality([raw], quantile_low=0.0, quantile_high=1.0)
    np.testing.assert_allclose(quality, [0.4])

    fused_raw, normalized = fuse_fragment_features(
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0]], dtype=np.float32),
        np.asarray([0.25], dtype=np.float32),
    )
    expected = np.asarray(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(fused_raw, expected)
    np.testing.assert_allclose(normalized, expected / np.linalg.norm(expected))
