import hashlib
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_module_available():
    assert importlib.util.find_spec("cocore.local_backtracking") is not None


@pytest.fixture
def bt():
    return importlib.import_module("cocore.local_backtracking")


def path(displacements):
    d = np.asarray(displacements, dtype=float)
    p = np.vstack((np.zeros(3), np.cumsum(d, axis=0)))
    return p, np.arange(len(p), dtype=float)


def compute(bt, displacements, **settings):
    return bt.compute_local_backtracking(
        *path(displacements), config={"epsilon_p": 0.01, **settings}
    )


@pytest.mark.parametrize(
    "second,eta,count",
    [
        ([1, 0, 0], 0.5, 0),
        ([-1, 0, 0], 0.5, 1),
        ([0, 1, 0], 0, 0),
        ([-3, 4, 0], 0.6, 0),
        ([-3, 4, 0], 0.59, 1),
    ],
)
def test_direction_and_strict_angle_boundary(bt, second, eta, count):
    row = compute(bt, [[1, 0, 0], second], eta=eta)
    assert row["local_backtracking_valid"]
    assert row["local_backtracking_valid_count"] == 1
    assert row["local_backtracking_count"] == count
    assert row["local_backtracking_rate"] == count
    assert row["low_local_backtracking"] == 1 - count
    assert row["local_backtracking_reason"] == ""


def test_twelve_reversals_over_eighty_comparisons(bt):
    steps = np.ones((81, 3))
    steps[:, 1:] = 0
    steps[:13, 0] = (-1.0) ** np.arange(13)
    row = compute(bt, steps)
    assert row["local_backtracking_count"] == 12
    assert row["local_backtracking_valid_count"] == 80
    assert row["local_backtracking_rate"] == 0.15
    assert row["low_local_backtracking"] == 0.85


@pytest.mark.parametrize("middle", [0, 0.5])
def test_small_motion_is_not_bridged_and_epsilon_is_strict(bt, middle):
    row = compute(bt, [[1, 0, 0], [middle, 0, 0], [-1, 0, 0]], epsilon_p=0.5)
    assert_invalid(row, "no_valid_comparisons")


def test_excludes_only_affected_pairs(bt):
    row = compute(bt, [[1, 0, 0], [-1, 0, 0], [0, 0, 0], [-1, 0, 0], [-1, 0, 0]])
    assert row["local_backtracking_valid_count"] == 2
    assert row["local_backtracking_count"] == 1
    assert row["local_backtracking_rate"] == 0.5


def assert_invalid(row, reason):
    assert row["local_backtracking_valid"] is False
    assert row["local_backtracking_reason"] == reason
    assert np.isnan(row["local_backtracking_rate"])
    assert np.isnan(row["low_local_backtracking"])
    assert row["local_backtracking_count"] == 0
    assert row["local_backtracking_valid_count"] == 0


@pytest.mark.parametrize("length", [0, 1, 2])
def test_too_short(bt, length):
    assert_invalid(
        bt.compute_local_backtracking(
            np.zeros((length, 3)), np.arange(length), config={"epsilon_p": 1}
        ),
        "too_short",
    )


@pytest.mark.parametrize(
    "times,reason",
    [
        ([0, 0, 1], "non_increasing_time"),
        ([2, 1, 0], "non_increasing_time"),
        ([0, 1, 2.01], "non_uniform_time"),
        ([-1e308, 0, 1e308], "overflow"),
    ],
)
def test_invalid_timestamps(bt, times, reason):
    assert_invalid(
        bt.compute_local_backtracking(np.zeros((3, 3)), times, config={"epsilon_p": 1}), reason
    )


def test_uniform_time_tolerance(bt):
    p, _ = path([[1, 0, 0], [-1, 0, 0]])
    assert bt.compute_local_backtracking(p, [0, 1, 2.0001], config={"epsilon_p": 0.1})[
        "local_backtracking_valid"
    ]
    assert_invalid(
        bt.compute_local_backtracking(p, [0, 1, 2.001], config={"epsilon_p": 0.1}),
        "non_uniform_time",
    )


@pytest.mark.parametrize(
    "p,t",
    [
        (np.zeros((3, 2)), [0, 1, 2]),
        (np.zeros(3), [0, 1, 2]),
        (np.zeros((3, 3)), [0, 1]),
        ([[0, 0, 0], [np.nan, 0, 0], [1, 0, 0]], [0, 1, 2]),
        (np.zeros((3, 3)), [0, 1, np.inf]),
    ],
)
def test_malformed_inputs_raise(bt, p, t):
    with pytest.raises(ValueError, match="local_backtracking"):
        bt.compute_local_backtracking(p, t, config={"epsilon_p": 1})


@pytest.mark.parametrize(
    "p",
    [
        [[1e308, 0, 0], [-1e308, 0, 0], [1e308, 0, 0]],
        [[0, 0, 0], [1.1e308, 1.1e308, 1.1e308], [0, 0, 0]],
    ],
)
def test_overflow_is_invalid(bt, p):
    assert_invalid(
        bt.compute_local_backtracking(p, [0, 1, 2], config={"epsilon_p": 0.1}), "overflow"
    )


@pytest.mark.parametrize("scale", [1e-170, 1e200])
def test_finite_extreme_scale_reversal_remains_valid(bt, scale):
    p = [[0, 0, 0], [scale, 0, 0], [0, 0, 0]]
    row = bt.compute_local_backtracking(p, [0, 1, 2], config={"epsilon_p": scale / 10})
    assert row["local_backtracking_valid"]
    assert row["local_backtracking_valid_count"] == 1
    assert row["local_backtracking_count"] == 1
    assert row["local_backtracking_rate"] == 1.0


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"eta": 0.5},
        True,
        [],
        {"epsilon_p": True},
        {"epsilon_p": 0},
        {"epsilon_p": -1},
        {"epsilon_p": np.inf},
        {"epsilon_p": np.nan},
        {"epsilon_p": "1"},
        {"epsilon_p": 1, "eta": True},
        {"epsilon_p": 1, "eta": -1},
        {"epsilon_p": 1, "eta": 1},
        {"epsilon_p": 1, "eta": np.nan},
        {"epsilon_p": 1, "eta": np.inf},
        {"epsilon_p": 1, "unknown": 3},
    ],
)
def test_invalid_config(bt, settings):
    with pytest.raises(ValueError, match="local_backtracking"):
        bt.resolve_backtracking_config(settings)


def test_optional_config_and_default(bt):
    assert bt.resolve_backtracking_config(None) is None
    assert bt.resolve_backtracking_config({"epsilon_p": 1}) == {"epsilon_p": 1.0, "eta": 0.5}
    with pytest.raises(ValueError):
        bt.compute_local_backtracking(np.zeros((3, 3)), [0, 1, 2], config=None)


def cache_fixture(bt, tmp_path):
    clips = [
        SimpleNamespace(sample_id=str(i), episode_id=0, start_step=i, length=3) for i in range(2)
    ]
    p = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0], [-1, 0, 0]]], dtype=float
    )
    t = np.array([[0, 1, 2], [1, 2, 3]], dtype=float)
    config = {"prototypes": {"profile": "test"}, "local_backtracking": {"epsilon_p": 0.1}}
    arrays = bt.backtracking_arrays(p, t, config=config["local_backtracking"], clips=clips)
    checksums = bt.save_backtracking_cache(tmp_path, SimpleNamespace(**arrays))
    manifest = {
        "local_backtracking": bt.backtracking_contract(config),
        "local_backtracking_checksums": checksums,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return arrays, clips, config, manifest


def test_cache_roundtrip_and_summary(bt, tmp_path):
    arrays, clips, config, _ = cache_fixture(bt, tmp_path)
    actual = bt.validate_backtracking_cache(tmp_path, clips, config)
    assert actual["local_backtracking_valid_count"].dtype == np.int64
    assert actual["local_backtracking_count"].dtype == np.int64
    assert actual["local_backtracking_reason"].dtype == np.dtype("U32")
    rows = [bt.backtracking_row(actual, i) for i in range(2)]
    assert isinstance(rows[0]["local_backtracking_count"], int)
    bt.validate_backtracking_fields(rows[0], bt.backtracking_row(arrays, 0))
    summary = bt.backtracking_summary(actual, rows, rows[:1])
    assert summary["scanned"] == {
        "valid_count": 2,
        "invalid_count": 0,
        "invalid_reasons": {},
        "rate_mean": 0.5,
        "valid_comparison_count": 2,
        "backtracking_count": 1,
    }
    assert summary["selected"]["rate_mean"] == 1.0


def test_summary_invalid_and_empty(bt):
    clips = [SimpleNamespace(sample_id="still")]
    arrays = bt.backtracking_arrays(
        np.zeros((1, 3, 3)), np.array([[0, 1, 2]]), config={"epsilon_p": 1}, clips=clips
    )
    row = bt.backtracking_row(arrays, 0)
    assert row["local_backtracking_rate"] is None
    result = bt.backtracking_summary(arrays, [row], [])
    assert result["graph"] == {
        "valid_count": 0,
        "invalid_count": 1,
        "invalid_reasons": {"no_valid_comparisons": 1},
        "rate_mean": None,
        "valid_comparison_count": 0,
        "backtracking_count": 0,
    }
    assert result["selected"]["rate_mean"] is None


@pytest.mark.parametrize("bad_count", [1.0, True, np.float64(1)])
def test_replay_rejects_noninteger_counts(bt, tmp_path, bad_count):
    arrays, _, _, _ = cache_fixture(bt, tmp_path)
    row = bt.backtracking_row(arrays, 0)
    with pytest.raises(ValueError, match="local_backtracking_count"):
        bt.validate_backtracking_fields({**row, "local_backtracking_count": bad_count}, row)


@pytest.mark.parametrize(
    "kind", ["checksum", "dtype", "replay", "overlap", "contract", "length", "missing"]
)
def test_cache_rejects_corruption(bt, tmp_path, kind):
    arrays, clips, config, manifest = cache_fixture(bt, tmp_path)
    field = "local_backtracking_count"
    if kind == "contract":
        config["local_backtracking"]["epsilon_p"] = 0.2
    elif kind == "length":
        clips[0].length = 4
    elif kind == "missing":
        (tmp_path / f"{field}.npy").unlink()
    else:
        if kind == "dtype":
            arrays[field] = arrays[field].astype(float)
        elif kind == "overlap":
            field = "local_backtracking_positions"
            arrays[field][1, 0, 0] = 2
        else:
            arrays[field][0] = 0
        file = tmp_path / f"{field}.npy"
        np.save(file, arrays[field])
        if kind != "checksum":
            manifest["local_backtracking_checksums"][field] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="local_backtracking"):
        bt.validate_backtracking_cache(tmp_path, clips, config)


def test_disabled_cache_rejects_stale_arrays(bt, tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    assert bt.validate_backtracking_cache(tmp_path, [], {}) == {}
    np.save(tmp_path / "local_backtracking_count.npy", np.zeros(0, dtype=np.int64))
    with pytest.raises(ValueError, match="unexpected"):
        bt.validate_backtracking_cache(tmp_path, [], {})


def test_replay_rejects_boolean_hidden_in_integer_list(bt, tmp_path):
    arrays, _, _, _ = cache_fixture(bt, tmp_path)
    actual = {**arrays, "local_backtracking_count": [True, 0]}
    with pytest.raises(ValueError, match="integer counts"):
        bt.validate_backtracking_fields(actual, arrays)


def test_smooth_closed_curve_is_not_local_backtracking(bt):
    angle = np.linspace(0, 2 * np.pi, 101)
    p = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(101)])
    row = bt.compute_local_backtracking(p, np.arange(101), config={"epsilon_p": 0.01})
    assert row["local_backtracking_valid_count"] == 99
    assert row["local_backtracking_rate"] == 0.0


def test_arrays_add_clip_context_to_bad_input(bt):
    with pytest.raises(ValueError, match="clip sample-A"):
        bt.backtracking_arrays(
            [np.zeros((3, 2))],
            [[0, 1, 2]],
            config={"epsilon_p": 1},
            clips=[SimpleNamespace(sample_id="sample-A")],
        )


@pytest.mark.parametrize("field", ["local_backtracking_rate", "low_local_backtracking"])
@pytest.mark.parametrize("bad_value", ["NaN", float("nan"), "0", 0, False])
def test_export_replay_requires_null_for_invalid_scores(bt, field, bad_value):
    arrays = bt.backtracking_arrays(
        np.zeros((1, 3, 3)),
        [[0, 1, 2]],
        config={"epsilon_p": 1},
        clips=[SimpleNamespace(sample_id="still")],
    )
    row = bt.backtracking_row(arrays, 0)
    with pytest.raises(ValueError, match=field):
        bt.validate_backtracking_fields({**row, field: bad_value}, row)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("local_backtracking_rate", "1.0"),
        ("local_backtracking_rate", True),
        ("low_local_backtracking", "0.0"),
        ("low_local_backtracking", False),
        ("local_backtracking_rate", None),
    ],
)
def test_export_replay_requires_real_nonboolean_scores(bt, tmp_path, field, bad_value):
    arrays, _, _, _ = cache_fixture(bt, tmp_path)
    row = bt.backtracking_row(arrays, 0)
    with pytest.raises(ValueError, match=field):
        bt.validate_backtracking_fields({**row, field: bad_value}, row)


@pytest.mark.parametrize("dtype", [str, bool, object])
def test_array_replay_rejects_nonnumeric_score_representations(bt, tmp_path, dtype):
    arrays, _, _, _ = cache_fixture(bt, tmp_path)
    field = "local_backtracking_rate"
    with pytest.raises(ValueError, match=field):
        bt.validate_backtracking_fields({**arrays, field: arrays[field].astype(dtype)}, arrays)


def test_replay_accepts_null_rows_and_nan_arrays(bt):
    arrays = bt.backtracking_arrays(
        np.zeros((1, 3, 3)),
        [[0, 1, 2]],
        config={"epsilon_p": 1},
        clips=[SimpleNamespace(sample_id="still")],
    )
    row = bt.backtracking_row(arrays, 0)
    bt.validate_backtracking_fields({**row, "local_backtracking_rate": None}, row)
    bt.validate_backtracking_fields(
        {**arrays, "local_backtracking_rate": np.array([np.nan])}, arrays
    )


def test_replay_accepts_integer_representations_of_finite_scores(bt, tmp_path):
    arrays, _, _, _ = cache_fixture(bt, tmp_path)
    row = bt.backtracking_row(arrays, 0)
    bt.validate_backtracking_fields({**row, "local_backtracking_rate": 1}, row)
    bt.validate_backtracking_fields({**arrays, "local_backtracking_rate": np.array([1, 0])}, arrays)


def test_non_axis_orthogonal_displacements_respect_strict_zero_eta(bt):
    # (-2, 5, -3) dot (-41, -50, -56) = 82 - 250 + 168 = 0.
    p = [[0, 0, 0], [-2, 5, -3], [-43, -45, -59]]
    row = bt.compute_local_backtracking(p, [0, 1, 2], config={"epsilon_p": 0.1, "eta": 0})
    assert row["local_backtracking_valid_count"] == 1
    assert row["local_backtracking_count"] == 0
    assert row["local_backtracking_rate"] == 0.0


def test_extreme_pair_fallback_does_not_change_other_orthogonal_pairs(bt):
    p = [[0, 0, 0], [-2, 5, -3], [-43, -45, -59], [0, 0, 0], [1e-170, 0, 0], [0, 0, 0]]
    row = bt.compute_local_backtracking(p, np.arange(6), config={"epsilon_p": 1e-171, "eta": 0})
    assert row["local_backtracking_valid_count"] == 4
    assert row["local_backtracking_count"] == 2
    assert row["local_backtracking_rate"] == 0.5
