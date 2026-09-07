import numpy as np
import pytest

from cocore.high_frequency_jitter import compute_high_frequency_jitter, resolve_hf_config

SETTINGS = dict(cutoff_hz=3.0, noise_floor_rms=1e-6, max_frequency_resolution_hz=1.0)


def signal(frequency, n=200, amplitude=1.0, fs=20.0):
    v = np.zeros((n, 3))
    v[:, 0] = amplitude * np.cos(2 * np.pi * frequency * np.arange(n) / fs)
    return np.vstack((np.zeros(3), np.cumsum(v / fs, axis=0))), np.arange(n + 1) / fs


def score(p, t, **overrides):
    return compute_high_frequency_jitter(p, t, config={**SETTINGS, **overrides})


@pytest.mark.parametrize("n", [199, 200])
def test_frequency_composition_and_scale(n):
    low = score(*signal(1.0, n))
    high = score(*signal(6.0, n))
    small = score(*signal(6.0, n, amplitude=0.01))
    assert low["high_frequency_ratio"] < 0.001
    assert high["high_frequency_ratio"] > 0.999
    assert small["high_frequency_ratio"] == pytest.approx(high["high_frequency_ratio"], abs=1e-6)
    assert small["high_frequency_rms"] == pytest.approx(high["high_frequency_rms"] * 0.01)
    assert high["high_frequency_valid"]


def test_mixed_axes_and_cutoff_boundary():
    p, t = signal(1.0)
    q, _ = signal(6.0, amplitude=0.5)
    p[:, 1] = q[:, 0]
    assert score(p, t)["high_frequency_ratio"] == pytest.approx(0.2, abs=1e-8)
    # A Hann-windowed on-bin sinusoid puts 1/6 of its energy above its peak bin.
    assert score(*signal(4.0, n=256, fs=16.0), cutoff_hz=4.0)[
        "high_frequency_ratio"
    ] == pytest.approx(1 / 6, abs=1e-8)


@pytest.mark.parametrize("velocity", [0.0, 2.0])
def test_low_change_skips_score(velocity):
    t = np.arange(16) / 20.0
    p = np.column_stack((velocity * t, t * 0, t * 0))
    result = score(p, t, max_frequency_resolution_hz=2.0)
    assert not result["high_frequency_valid"]
    assert result["high_frequency_reason"] == "low_fluctuation"
    assert np.isnan(result["low_high_frequency_jitter"])
    assert np.isfinite(result["high_frequency_ratio"])


@pytest.mark.parametrize(
    "n,overrides,reason",
    [
        (2, {}, "too_short"),
        (6, {}, "coarse_resolution"),
        (14, {"cutoff_hz": 0.1, "max_frequency_resolution_hz": 2.0}, "missing_band"),
    ],
)
def test_invalid_resolution(n, overrides, reason):
    result = score(*signal(1.0, n), **overrides)
    assert not result["high_frequency_valid"]
    assert result["high_frequency_reason"] == reason
    assert np.isnan(result["low_high_frequency_jitter"])


@pytest.mark.parametrize("kind", ["nonfinite", "decreasing", "irregular", "nyquist", "overflow"])
def test_bad_inputs_raise(kind):
    p, t = signal(1.0)
    kwargs = {}
    if kind == "nonfinite":
        p[0, 0] = np.nan
    if kind == "decreasing":
        t[2] = t[1]
    if kind == "irregular":
        t[2] += 0.001
    if kind == "nyquist":
        kwargs["cutoff_hz"] = 11.0
    if kind == "overflow":
        p[::2, 0] = 1e308
    with pytest.raises(ValueError, match="high_frequency_jitter"):
        score(p, t, **kwargs)


@pytest.mark.parametrize("key", list(SETTINGS) + ["epsilon"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True])
def test_invalid_config(key, value):
    with pytest.raises(ValueError):
        resolve_hf_config({**SETTINGS, key: value})


def test_nan_fusion_with_path():
    from cocore.action_variation import fuse_reliability

    ones = np.ones(4)
    result = fuse_reliability(
        ones,
        ones,
        ones,
        ones,
        ["local_path_efficiency", "low_high_frequency_jitter"],
        min_reliability=0.05,
        local_path_efficiency=np.array([0.25, np.nan, 0.25, np.nan]),
        low_high_frequency_jitter=np.array([np.nan, 0.81, 0.81, np.nan]),
    )
    np.testing.assert_allclose(result, [0.25, 0.81, 0.45, 1.0])


@pytest.mark.parametrize("n", [1, 2])
def test_short_signals_retain_computable_diagnostics(n):
    result = score(*signal(1.0, n))
    assert result["high_frequency_reason"] == "too_short"
    assert not result["high_frequency_valid"]
    assert np.isnan(result["low_high_frequency_jitter"])
    assert np.isfinite(result["high_frequency_ratio"])
    assert np.isfinite(result["high_frequency_rms"])
    assert np.isfinite(result["total_fluctuation_rms"])


def test_nyquist_and_constant_speed_direction_reversals():
    p, t = signal(8.0, n=128, fs=16.0)
    assert score(p, t, cutoff_hz=4.0)["high_frequency_ratio"] > 0.999
    with pytest.raises(ValueError, match="Nyquist"):
        score(p, t, cutoff_hz=8.0)


@pytest.mark.parametrize("settings", [{}, {**SETTINGS, "unknown": 1}, "invalid"])
def test_configuration_requires_exact_keys(settings):
    with pytest.raises(ValueError):
        resolve_hf_config(settings)


@pytest.mark.parametrize("values", [[np.inf], [-0.1], [1.1], [[0.5]], [0.5, 0.5]])
def test_fusion_rejects_malformed_hf(values):
    from cocore.action_variation import fuse_reliability

    ones = np.ones(1)
    with pytest.raises(ValueError, match="low_high_frequency_jitter"):
        fuse_reliability(
            ones,
            ones,
            ones,
            ones,
            ["low_high_frequency_jitter"],
            min_reliability=0.05,
            low_high_frequency_jitter=np.array(values),
        )


def test_odd_length_missing_high_band_retains_energy():
    result = score(*signal(1.0, n=5, fs=16.0), cutoff_hz=7.0, max_frequency_resolution_hz=4.0)
    assert result["high_frequency_reason"] == "missing_band"
    assert result["high_frequency_rms"] == 0.0
    assert np.isfinite(result["total_fluctuation_rms"])


def test_resolution_and_noise_threshold_boundaries():
    p, t = signal(4.0, n=16, fs=16.0)
    result = score(p, t, max_frequency_resolution_hz=1.0)
    assert result["high_frequency_valid"]
    assert result["high_frequency_resolution_hz"] == 1.0
    at_noise = score(p, t, noise_floor_rms=result["total_fluctuation_rms"])
    assert at_noise["high_frequency_reason"] == "low_fluctuation"
    t[5] += 1e-5
    assert score(p, t)["high_frequency_valid"]


def test_no_velocity_samples_have_no_diagnostics():
    result = score(np.zeros((1, 3)), np.array([0.0]))
    assert result["high_frequency_reason"] == "too_short"
    assert np.isnan(result["high_frequency_ratio"])
    assert np.isnan(result["high_frequency_resolution_hz"])
