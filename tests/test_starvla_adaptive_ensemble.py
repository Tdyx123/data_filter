import math

import numpy as np


def _chunk() -> np.ndarray:
    return np.zeros((1, 16, 7), dtype=np.float32)


def test_adaptive_ensemble_aligns_overlapping_chunk_predictions() -> None:
    from starvla_bridge.action_ensemble import AdaptiveActionEnsembler

    ensembler = AdaptiveActionEnsembler(horizon=7, alpha=0.0)
    first = _chunk()
    first[0, 0, 0] = 99.0
    first[0, 1, 0] = 2.0
    second = _chunk()
    second[0, 0, 1] = 4.0

    np.testing.assert_allclose(ensembler.select_action(first), first[0, 0])
    selected = ensembler.select_action(second)

    np.testing.assert_allclose(
        selected,
        np.asarray([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rtol=0,
        atol=1.0e-7,
    )


def test_adaptive_ensemble_uses_starvla_cosine_weights() -> None:
    from starvla_bridge.action_ensemble import AdaptiveActionEnsembler

    ensembler = AdaptiveActionEnsembler(horizon=7, alpha=0.1)
    first = _chunk()
    first[0, 1, 0] = -1.0
    second = _chunk()
    second[0, 0, 0] = 2.0

    ensembler.select_action(first)
    selected = ensembler.select_action(second)
    expected = (-math.exp(-0.1) + 2.0 * math.exp(0.1)) / (
        math.exp(-0.1) + math.exp(0.1)
    )

    np.testing.assert_allclose(selected[0], expected, rtol=0, atol=1.0e-7)


def test_adaptive_ensemble_binarizes_each_chunk_gripper_before_weighting() -> None:
    from starvla_bridge.action_ensemble import AdaptiveActionEnsembler

    ensembler = AdaptiveActionEnsembler(horizon=7, alpha=0.1)
    first = _chunk()
    first[0, 0, 6] = 0.49
    first[0, 1, 6] = 0.49
    second = _chunk()
    second[0, 0, 6] = 0.51

    assert ensembler.select_action(first)[6] == 0.0
    selected = ensembler.select_action(second)

    expected = math.exp(0.1) / (1.0 + math.exp(0.1))
    np.testing.assert_allclose(selected[6], expected, rtol=0, atol=1.0e-7)


def test_adaptive_ensemble_reset_discards_prior_episode_history() -> None:
    from starvla_bridge.action_ensemble import AdaptiveActionEnsembler

    ensembler = AdaptiveActionEnsembler(horizon=7, alpha=0.1)
    first = _chunk()
    first[0, 1, 0] = 10.0
    second = _chunk()
    second[0, 0, 0] = 2.0
    ensembler.select_action(first)
    ensembler.select_action(second)

    ensembler.reset()
    new_episode = _chunk()
    new_episode[0, 0, 0] = -3.0

    np.testing.assert_allclose(ensembler.select_action(new_episode), new_episode[0, 0])


def test_adaptive_ensemble_uses_only_the_seven_most_recent_chunks() -> None:
    from starvla_bridge.action_ensemble import AdaptiveActionEnsembler

    ensembler = AdaptiveActionEnsembler(horizon=7, alpha=0.0)
    selected = None
    for chunk_id in range(8):
        chunk = _chunk()
        for time_index in range(16):
            chunk[0, time_index, 0] = 100.0 * chunk_id + time_index
        selected = ensembler.select_action(chunk)

    assert selected is not None
    expected = np.mean([106.0, 205.0, 304.0, 403.0, 502.0, 601.0, 700.0])
    np.testing.assert_allclose(selected[0], expected, rtol=0, atol=1.0e-6)
