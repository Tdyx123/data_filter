import numpy as np
import pytest

from cocore.config import resolve_config
from cocore.action_variation import fuse_reliability


def test_default_and_optional_configuration():
    base = {"objective": {"relation": "cooccurrence", "relation_weight": 1}}
    config = resolve_config(base)
    assert config["reliability_metrics"][-1] == "action_jump"
    assert config["action_jump"]["threshold_quantile"] == 0.99
    assert "action_jump" not in resolve_config({**base, "reliability_metrics": ["support"]})


def test_exact_rate_and_strict_threshold():
    from cocore.action_jump import compute_action_jump_rate

    actions = np.cumsum(np.r_[0.0, np.ones(92), np.full(8, 2.0)])[:, None]
    assert compute_action_jump_rate(actions, np.ones(1), threshold=1.0, epsilon=1e-8) == 0.08
    assert compute_action_jump_rate(actions[:93], np.ones(1), threshold=1.0, epsilon=1e-8) == 0


def test_fusion_preserves_nan_skip():
    one = np.ones(2)
    actual = fuse_reliability(
        one,
        one,
        one,
        one,
        ["action_jump", "local_path_efficiency"],
        min_reliability=0,
        action_jump=np.array([0.81, 0.64]),
        local_path_efficiency=np.array([np.nan, 0.25]),
    )
    np.testing.assert_allclose(actual, [0.81, 0.4])


def test_reference_pool_ignores_gripper_and_episode_boundaries():
    from cocore.action_jump import build_jump_arrays
    from types import SimpleNamespace

    episodes = [
        np.array([[0.0, 0.0], [1.0, 100.0], [1.0, 0.0]]),
        np.array([[100.0, 0.0], [101.0, 100.0]]),
    ]
    clips = [SimpleNamespace(episode_id=7, start_step=1, end_step=2, length=2)]
    result = build_jump_arrays(episodes, [7, 8], clips, config={}, gripper_action_index=-1)
    scale = np.std([0, 1, 1, 100, 101])
    np.testing.assert_allclose(result["action_jump_scale"], [scale])
    assert result["action_jump_pair_count"].item() == 3
    assert result["action_jump_threshold"].item() == pytest.approx(1 / scale)
    np.testing.assert_array_equal(result["action_jump_rate"], [0])


@pytest.mark.parametrize(
    "value",
    [{"epsilon": 0}, {"threshold": -1}, {"threshold": True}, {"threshold_quantile": 1}, {"x": 2}],
)
def test_invalid_config(value):
    from cocore.action_jump import resolve_jump_config

    with pytest.raises(ValueError):
        resolve_jump_config(value)


def test_scale_invariance_and_raw_outliers():
    from cocore.action_jump import build_jump_arrays
    from types import SimpleNamespace

    raw = np.zeros((101, 3))
    raw[:, 0] = np.arange(101)
    raw[-1, 1] = 10000
    raw[:, 2] = np.arange(101) % 2
    clips = [SimpleNamespace(episode_id=0, start_step=0, end_step=100, length=101)]
    a = build_jump_arrays([raw], [0], clips, config={}, gripper_action_index=-1)
    b = build_jump_arrays([raw * [100, 0.001, 999]], [0], clips, config={}, gripper_action_index=-1)
    np.testing.assert_allclose(a["action_jump_rate"], [0.01])
    np.testing.assert_allclose(a["action_jump_rate"], b["action_jump_rate"])
    np.testing.assert_allclose(a["action_jump_threshold"], b["action_jump_threshold"])


def test_constant_dimensions_and_single_frame_reference():
    from cocore.action_jump import build_jump_arrays
    from types import SimpleNamespace

    clips = [SimpleNamespace(episode_id=0, start_step=0, end_step=1, length=2)]
    result = build_jump_arrays(
        [np.ones((2, 2)), np.ones((1, 2))], [0, 1], clips, config={}, gripper_action_index=1
    )
    np.testing.assert_array_equal(result["action_jump_scale"], [0])
    np.testing.assert_array_equal(result["action_jump"], [1])
    assert result["action_jump_pair_count"] == 1
    assert result["action_jump_threshold"] == 0


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((1, 1)),
        np.array([[0], [np.nan]]),
        np.array([[0], [np.inf]]),
        np.array([[-1e308], [1e308]]),
    ],
)
def test_bad_rate_input(actions):
    from cocore.action_jump import compute_action_jump_rate

    with pytest.raises(ValueError, match="action_jump"):
        compute_action_jump_rate(actions, np.ones(1), threshold=1, epsilon=1e-8)


@pytest.mark.parametrize(
    "episodes,index",
    [
        ([np.ones((2, 1))], -1),
        ([np.ones((2, 2))], 2),
        ([np.ones((2, 2))], -3),
        ([np.ones((2, 2))], True),
        ([np.ones((2, 2)), np.ones((3, 3))], -1),
        ([np.array([[1e308, 0], [-1e308, 0]])], -1),
    ],
)
def test_bad_reference_input(episodes, index):
    from cocore.action_jump import build_jump_arrays

    with pytest.raises(ValueError, match="action_jump"):
        build_jump_arrays(
            episodes, list(range(len(episodes))), [], config={}, gripper_action_index=index
        )


@pytest.mark.parametrize(
    "score",
    [None, np.array([np.nan]), np.array([np.inf]), np.array([-1]), np.array([2]), np.ones(2)],
)
def test_fusion_rejects_missing_invalid_jump(score):
    one = np.ones(1)
    with pytest.raises(ValueError):
        fuse_reliability(one, one, one, one, ["action_jump"], min_reliability=0, action_jump=score)


@pytest.mark.parametrize("index", [True, 0.5, "-1", None])
def test_config_rejects_noninteger_gripper_index(index):
    with pytest.raises(ValueError, match="gripper_action_index"):
        resolve_config(
            {
                "objective": {"relation": "sequence", "relation_weight": 1},
                "quality": {"gripper_action_index": index},
            }
        )
