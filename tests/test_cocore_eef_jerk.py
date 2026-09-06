import numpy as np
import pytest


def test_polynomials_and_time_scaling():
    from cocore.eef_jerk import compute_eef_jerk

    t = np.arange(16, dtype=float) / 10
    for degree in (0, 1, 2):
        value, reason = compute_eef_jerk(np.column_stack((t**degree, t * 0, t * 0)), t)
        assert reason == ""
        assert value == pytest.approx(0, abs=1e-10)
    p = np.column_stack((t**3, 2 * t**3, 2 * t**3))
    assert compute_eef_jerk(p, t)[0] == pytest.approx(18)
    assert compute_eef_jerk(p, t * 3)[0] == pytest.approx(18 / 27)
    assert compute_eef_jerk(p[:4], t[:4])[0] == pytest.approx(18)


def test_norm_before_average():
    from cocore.eef_jerk import compute_eef_jerk

    p = np.zeros((5, 3))
    p[:, 0] = [0, 0, 0, 1, 2]
    assert compute_eef_jerk(p, np.arange(5))[0] == pytest.approx(1)


@pytest.mark.parametrize(
    "times,reason",
    [
        ([0, 1, 2], "too_short"),
        ([0, 1, 1, 2], "non_increasing_time"),
        ([0, 1, 2, 4], "non_uniform_time"),
    ],
)
def test_uncomputable(times, reason):
    from cocore.eef_jerk import compute_eef_jerk

    value, actual = compute_eef_jerk(np.zeros((len(times), 3)), times)
    assert np.isnan(value)
    assert actual == reason


def test_bad_inputs_and_overflow():
    from cocore.eef_jerk import compute_eef_jerk

    with pytest.raises(ValueError):
        compute_eef_jerk(np.full((4, 3), np.nan), np.arange(4))
    p = np.zeros((4, 3))
    p[3, 0] = 1e308
    value, reason = compute_eef_jerk(p, np.arange(4) * 1e-100)
    assert np.isnan(value) and reason == "overflow"


def test_normalization_and_fusion():
    from cocore.eef_jerk import normalize_eef_jerk
    from cocore.action_variation import fuse_reliability, normalize_reliability_metrics

    scores = normalize_eef_jerk(
        np.array([0.0, 1.0, 2.0, np.nan]), quantile_low=0, quantile_high=1, epsilon=1e-8
    )
    np.testing.assert_allclose(scores, [1, 0.5, 0, np.nan], equal_nan=True)
    np.testing.assert_array_equal(
        normalize_eef_jerk(np.ones(3), quantile_low=0, quantile_high=1, epsilon=1e-8), np.ones(3)
    )
    assert normalize_reliability_metrics(["eef_jerk", "support"]) == ("support", "eef_jerk")
    one = np.ones(3)
    result = fuse_reliability(
        one, one, one, one, ["support", "eef_jerk"], eef_jerk=scores[:3], min_reliability=0.05
    )
    np.testing.assert_allclose(result, [1, np.sqrt(0.5), 0.05])


def test_uniform_time_tolerance_and_all_invalid_normalization():
    from cocore.eef_jerk import compute_eef_jerk, normalize_eef_jerk

    p = np.zeros((5, 3))
    assert compute_eef_jerk(p, [0, 1, 2.00001, 3, 4])[1] == ""
    assert compute_eef_jerk(p, [0, 1, 2.001, 3, 4])[1] == "non_uniform_time"
    assert compute_eef_jerk(p, [4, 3, 2, 1, 0])[1] == "non_increasing_time"
    scores = normalize_eef_jerk(
        np.full(4, np.nan), quantile_low=0.01, quantile_high=0.99, epsilon=1e-8
    )
    assert np.isnan(scores).all()


def test_fusion_requires_finite_jerk_component():
    from cocore.action_variation import fuse_reliability

    one = np.ones(2)
    with pytest.raises(ValueError, match="eef_jerk"):
        fuse_reliability(one, one, one, one, ["eef_jerk"], min_reliability=0.05)
    with pytest.raises(ValueError, match="finite"):
        fuse_reliability(
            one, one, one, one, ["eef_jerk"], min_reliability=0.05, eef_jerk=[1, np.nan]
        )


def test_overflowed_time_denominator_is_not_zero_jerk():
    from cocore.eef_jerk import compute_eef_jerk

    p = np.zeros((4, 3))
    p[3, 0] = 1
    value, reason = compute_eef_jerk(p, np.arange(4) * 1e110)
    assert np.isnan(value) and reason == "overflow"


@pytest.mark.parametrize("field,dtype", [("eef_jerk_raw", np.float32), ("eef_jerk", np.float64)])
def test_cache_rejects_changed_score_dtype(tmp_path, field, dtype):
    import json
    from types import SimpleNamespace
    from cocore.config import resolve_config
    from cocore.eef_jerk import jerk_arrays, jerk_contract, save_jerk_cache, validate_jerk_cache

    config = resolve_config(
        {
            "objective": {"relation": "sequence", "relation_weight": 1},
            "reliability_metrics": ["eef_jerk"],
        }
    )
    arrays = jerk_arrays(
        np.zeros((1, 4, 3)),
        np.arange(4)[None, :],
        quantile_low=0.01,
        quantile_high=0.99,
        epsilon=1e-8,
    )
    arrays[field] = arrays[field].astype(dtype)
    checksums = save_jerk_cache(tmp_path, SimpleNamespace(**arrays))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"eef_jerk": jerk_contract(config), "eef_jerk_checksums": checksums})
    )
    clips = [SimpleNamespace(length=4, episode_id=0, start_step=0)]
    with pytest.raises(ValueError, match="dtype"):
        validate_jerk_cache(tmp_path, clips, config)


@pytest.mark.parametrize("package", ["cocore", "cocore_bridge_v2"])
@pytest.mark.parametrize("command", ["build-graph", "select", "run", "validate"])
def test_cli_accepts_optional_jerk(package, command):
    from importlib import import_module

    cli = import_module(f"{package}.cli")
    args = [
        command,
        "--relation",
        "sequence",
        "--relation-weight",
        "1",
        "--reliability-metrics",
        "support",
        "eef_jerk",
    ]
    if package == "cocore" and command in {"build-graph", "validate"}:
        args = [command, "--reliability-metrics", "support", "eef_jerk"]
    if command == "validate":
        args += ["--output-dir", "result"]
    parsed = cli.build_parser().parse_args(args)
    assert parsed.reliability_metrics == ["support", "eef_jerk"]
