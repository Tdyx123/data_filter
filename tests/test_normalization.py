import numpy as np

from qwen3_vl_groot.normalization import QuantileStats


def test_quantile_normalization_round_trip_and_clipping():
    stats = QuantileStats(
        state_q01=np.array([-2.0, 0.0, 5.0], dtype=np.float32),
        state_q99=np.array([2.0, 10.0, 5.0], dtype=np.float32),
        action_q01=np.array([-1.0, 0.0], dtype=np.float32),
        action_q99=np.array([1.0, 4.0], dtype=np.float32),
    )
    state = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    normalized = stats.normalize_state(state)
    np.testing.assert_allclose(normalized, [0.5, -0.6, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(stats.denormalize_state(normalized), state, atol=1.0e-6)

    action = np.array([[0.5, 3.0]], dtype=np.float32)
    normalized_action = stats.normalize_action(action)
    np.testing.assert_allclose(
        stats.denormalize_action(normalized_action), action, atol=1.0e-6
    )
    np.testing.assert_array_equal(
        stats.normalize_action(np.array([[-10.0, 20.0]], dtype=np.float32)),
        np.array([[-1.0, 1.0]], dtype=np.float32),
    )


def test_stats_json_round_trip(tmp_path):
    stats = QuantileStats(
        state_q01=np.zeros(8, dtype=np.float32),
        state_q99=np.ones(8, dtype=np.float32),
        action_q01=np.zeros(7, dtype=np.float32),
        action_q99=np.ones(7, dtype=np.float32),
    )
    path = tmp_path / "normalization.json"
    stats.save(path, extra={"metadata_sha256": "abc"})
    loaded = QuantileStats.load(path)
    np.testing.assert_array_equal(loaded.state_q01, stats.state_q01)
    np.testing.assert_array_equal(loaded.action_q99, stats.action_q99)

