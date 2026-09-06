import numpy as np
import pytest


def score(state, timestamps=None, profile="libero", **overrides):
    from cocore.dwell import compute_dwell_ratio

    config = dict(
        position_speed_threshold=0.1,
        gripper_speed_threshold=0.1,
        angular_speed_threshold=0.1,
        gripper_mode="continuous",
    )
    config.update(overrides)
    if timestamps is None:
        timestamps = np.arange(len(state), dtype=float)
    return compute_dwell_ratio(state, timestamps, profile=profile, config=config)


def test_dwell_counts_internal_pairs_and_strict_threshold():
    state = np.zeros((101, 8))
    state[71:, 0] = np.arange(1, 31)
    assert score(state) == pytest.approx(0.70)
    assert score(np.zeros((2, 8))) == 1
    state = np.zeros((2, 8))
    state[1, 0] = 0.1
    assert score(state) == 0


def test_time_weighting_and_sampling_rate():
    state = np.zeros((3, 8))
    state[2, 0] = 1
    assert score(state, [0, 3, 4]) == pytest.approx(0.75)
    for rate in (5, 20):
        time = np.arange(rate + 1) / rate
        state = np.zeros((len(time), 8))
        state[:, 0] = time * 0.2
        assert score(state, time) == 0


@pytest.mark.parametrize("profile", ["libero", "bridge_v2"])
def test_rotation_and_wrap(profile):
    state = np.zeros((2, 8))
    state[1, 5] = 0.2
    assert score(state, profile=profile) == 0
    state[:, 5] = [np.pi - 0.01, -np.pi + 0.01]
    assert score(state, profile=profile) == 1


def test_binary_and_continuous_gripper():
    state = np.zeros((2, 8))
    state[1, 7] = 0.01
    assert score(state) == 1
    assert score(state, gripper_mode="binary") == 0
    state[:, 7] = 1
    assert score(state, gripper_mode="binary") == 1


@pytest.mark.parametrize(
    "state,time",
    [
        (np.zeros((1, 8)), [0]),
        (np.zeros((2, 8)), [0]),
        (np.zeros((2, 8)), [0, 0]),
        (np.zeros((2, 8)), [1, 0]),
        (np.full((2, 8), np.nan), [0, 1]),
        (np.zeros((2, 8)), [0, np.inf]),
    ],
)
def test_invalid_inputs(state, time):
    with pytest.raises(ValueError, match="dwell"):
        score(state, time)


def test_configuration_and_optional_fusion():
    from cocore.config import resolve_config
    from cocore.action_variation import fuse_reliability

    base = {"objective": {"relation": "sequence", "relation_weight": 1}}
    assert "non_dwell" not in resolve_config(base)["reliability_metrics"]
    with pytest.raises(ValueError, match="dwell"):
        resolve_config({**base, "reliability_metrics": ["non_dwell"]})
    config = dict(position_speed_threshold=0.1, angular_speed_threshold=0.2, gripper_mode="binary")
    resolved = resolve_config({**base, "dwell": config, "reliability_metrics": ["non_dwell"]})
    assert resolved["dwell"] == config
    ones = np.ones(2)
    result = fuse_reliability(
        ones,
        ones,
        ones,
        ones,
        ["support", "non_dwell"],
        min_reliability=0.05,
        non_dwell=np.array([0.0, 0.25]),
    )
    np.testing.assert_allclose(result, [0.05, 0.5])


@pytest.mark.parametrize(
    "key,value",
    [
        ("position_speed_threshold", 0),
        ("gripper_speed_threshold", -1),
        ("angular_speed_threshold", float("nan")),
        ("position_speed_threshold", True),
        ("angular_speed_threshold", float("inf")),
        ("gripper_mode", "auto"),
    ],
)
def test_invalid_config(key, value):
    with pytest.raises(ValueError, match="dwell"):
        score(np.zeros((2, 8)), **{key: value})


def test_combined_euler_rotation_matches_rotvec():
    from scipy.spatial.transform import Rotation

    euler = np.array([[0.3, -0.6, 2.1], [0.32, -0.64, 2.15]])
    state = np.zeros((2, 8))
    state[:, 3:6] = euler
    rotvec = state.copy()
    rotvec[:, 3:6] = Rotation.from_euler("xyz", euler).as_rotvec()
    for threshold in (0.02, 0.2):
        assert score(state, profile="bridge_v2", angular_speed_threshold=threshold) == score(
            rotvec, angular_speed_threshold=threshold
        )


def test_clip_start_has_no_preceding_pair():
    state = np.zeros((4, 8))
    state[2:, 0] = 1
    assert score(state[2:]) == 1
    assert score(state) == pytest.approx(2 / 3)
