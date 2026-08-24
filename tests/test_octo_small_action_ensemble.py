import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ENSEMBLER = (
    PROJECT_ROOT
    / "third_party"
    / "SimplerEnv"
    / "simpler_env"
    / "utils"
    / "action"
    / "action_ensemble.py"
)


def _chunk() -> np.ndarray:
    return np.zeros((1, 8, 7), dtype=np.float32)


def _load_official_ensembler():
    spec = importlib.util.spec_from_file_location(
        "pinned_simpler_action_ensemble",
        OFFICIAL_ENSEMBLER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActionEnsembler


def test_octo_temporal_ensemble_aligns_overlapping_predictions() -> None:
    from octo_small_bridge.action_ensemble import OctoTemporalActionEnsembler

    ensembler = OctoTemporalActionEnsembler(prediction_horizon=8, temperature=0.0)
    first = _chunk()
    first[0, 0, 0] = 99.0
    first[0, 1, 0] = 2.0
    second = _chunk()
    second[0, 0, 1] = 4.0

    np.testing.assert_array_equal(ensembler.select_action(first), first[0, 0])
    selected = ensembler.select_action(second)

    np.testing.assert_allclose(
        selected,
        np.asarray([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rtol=0,
        atol=1.0e-7,
    )


@pytest.mark.parametrize("temperature", (0.0, 0.2))
def test_octo_temporal_ensemble_matches_pinned_simpler_reference(
    temperature: float,
) -> None:
    from octo_small_bridge.action_ensemble import OctoTemporalActionEnsembler

    implementation = OctoTemporalActionEnsembler(
        prediction_horizon=8,
        temperature=temperature,
    )
    reference = _load_official_ensembler()(
        pred_action_horizon=8,
        action_ensemble_temp=temperature,
    )
    rng = np.random.default_rng(1234)

    for _ in range(12):
        chunk = rng.normal(size=(1, 8, 7)).astype(np.float32)
        actual = implementation.select_action(chunk)
        expected = reference.ensemble_action(chunk[0])
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_octo_temporal_ensemble_reset_discards_previous_episode() -> None:
    from octo_small_bridge.action_ensemble import OctoTemporalActionEnsembler

    ensembler = OctoTemporalActionEnsembler()
    first = _chunk()
    first[0, 1, 0] = 10.0
    ensembler.select_action(first)
    ensembler.reset()
    new_episode = _chunk()
    new_episode[0, 0, 0] = -3.0

    np.testing.assert_array_equal(
        ensembler.select_action(new_episode),
        new_episode[0, 0],
    )


def test_octo_temporal_ensemble_uses_only_eight_most_recent_chunks() -> None:
    from octo_small_bridge.action_ensemble import OctoTemporalActionEnsembler

    ensembler = OctoTemporalActionEnsembler()
    selected = None
    for chunk_id in range(9):
        chunk = _chunk()
        for time_index in range(8):
            chunk[0, time_index, 0] = 100.0 * chunk_id + time_index
        selected = ensembler.select_action(chunk)

    assert selected is not None
    expected = np.mean([107.0, 206.0, 305.0, 404.0, 503.0, 602.0, 701.0, 800.0])
    np.testing.assert_allclose(selected[0], expected, rtol=0, atol=1.0e-6)


def test_octo_temporal_ensemble_preserves_continuous_gripper_until_selection() -> None:
    from octo_small_bridge.action_ensemble import OctoTemporalActionEnsembler
    from simpler_bridge.evaluation import bridge_actions_to_simpler

    ensembler = OctoTemporalActionEnsembler()
    first = _chunk()
    first[0, 1, 6] = 0.49
    second = _chunk()
    second[0, 0, 6] = 0.9
    ensembler.select_action(first)

    selected = ensembler.select_action(second)

    np.testing.assert_allclose(selected[6], 0.695, rtol=0, atol=1.0e-7)
    assert bridge_actions_to_simpler(selected)[6] == 1.0

    binarized_before_ensemble = selected.copy()
    binarized_before_ensemble[6] = 0.5
    assert bridge_actions_to_simpler(binarized_before_ensemble)[6] == -1.0


@pytest.mark.parametrize(
    "actions",
    (
        np.zeros((8, 7), dtype=np.float32),
        np.zeros((1, 7, 7), dtype=np.float32),
        np.zeros((1, 8, 6), dtype=np.float32),
        np.full((1, 8, 7), np.nan, dtype=np.float32),
    ),
)
def test_octo_temporal_ensemble_rejects_invalid_chunks(actions: np.ndarray) -> None:
    from octo_small_bridge.action_ensemble import OctoTemporalActionEnsembler
    from simpler_bridge.evaluation import SimplerEvaluationError

    with pytest.raises(SimplerEvaluationError, match=r"finite \(1, 8, 7\)"):
        OctoTemporalActionEnsembler().select_action(actions)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"prediction_horizon": 0}, "positive integer"),
        ({"prediction_horizon": True}, "positive integer"),
        ({"temperature": np.nan}, "finite"),
    ),
)
def test_octo_temporal_ensemble_rejects_invalid_configuration(kwargs, message) -> None:
    from octo_small_bridge.action_ensemble import OctoTemporalActionEnsembler
    from simpler_bridge.evaluation import SimplerEvaluationError

    with pytest.raises(SimplerEvaluationError, match=message):
        OctoTemporalActionEnsembler(**kwargs)
