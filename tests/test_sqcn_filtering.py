from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sqcn.filtering import filter_sqcn_run, select_diverse_fragments
from segment_filter_core.selection import _DiverseSelector


SCORE_COLUMNS = (
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "novelty",
    "sqcn",
)


def _write_sqcn_run(
    tmp_path: Path,
    *,
    embedding_dim: int = 128,
    quality_values: list[float] | None = None,
    sample_count: int = 5,
) -> tuple[Path, list[dict[str, object]], np.ndarray]:
    root = tmp_path / "sqcn-run"
    fragment = root / "fragment"
    fragment.mkdir(parents=True)
    if quality_values is not None:
        sample_count = len(quality_values)
    if sample_count == 5:
        sqcn_values = [0.2, 0.9, 0.6, 0.8, 0.4]
    else:
        sqcn_values = np.linspace(1.0, 0.0, sample_count, dtype=np.float64).tolist()
    qualities = quality_values if quality_values is not None else [0.5] * sample_count
    rows = [
        {
            "sample_id": f"sample-{index}",
            "episode_id": index,
            "start_step": 0,
            "end_step": 14,
            "length": 15,
            "quality": quality,
            "coverage": 0.5,
            "novelty": 0.5,
            "sqcn": score,
        }
        for index, (score, quality) in enumerate(
            zip(sqcn_values, qualities, strict=True)
        )
    ]
    with (fragment / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    embeddings = np.zeros((len(rows), embedding_dim), dtype=np.float32)
    embeddings[np.arange(len(rows)), np.arange(len(rows)) % embedding_dim] = 1.0
    np.save(fragment / "embeddings.npy", embeddings)
    (root / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "embedding_dim": embedding_dim}),
        encoding="utf-8",
    )
    return root, rows, embeddings


def test_selector_rejects_fewer_than_one_hundred_fragments():
    scores = np.linspace(1.0, 0.0, 99, dtype=np.float64)

    with pytest.raises(ValueError, match="scores must contain at least 100 fragments"):
        select_diverse_fragments(
            scores,
            np.zeros((len(scores), 2), dtype=np.float32),
            tuple(f"sample-{index:03d}" for index in range(len(scores))),
            target_size=99,
            seed=17,
        )


def test_selector_rejects_target_smaller_than_one_hundred():
    scores = np.linspace(1.0, 0.0, 100, dtype=np.float64)

    with pytest.raises(ValueError, match="target_size must be at least 100"):
        select_diverse_fragments(
            scores,
            np.zeros((len(scores), 2), dtype=np.float32),
            tuple(f"sample-{index:03d}" for index in range(len(scores))),
            target_size=99,
            seed=17,
        )


def test_target_of_exactly_one_hundred_uses_raw_score_order_without_penalties():
    scores = np.zeros(100, dtype=np.float64)
    sample_ids = tuple(f"sample-{index:03d}" for index in reversed(range(100)))

    result = select_diverse_fragments(
        scores,
        np.zeros((len(scores), 2), dtype=np.float32),
        sample_ids,
        target_size=100,
        seed=17,
    )

    assert result.selected_indices.tolist() == list(reversed(range(100)))
    np.testing.assert_array_equal(result.adjusted_scores, 0.0)
    np.testing.assert_array_equal(result.knn_penalties, 0.0)
    assert result.sigma_raw == 0.0
    assert result.sigma_effective == 1.0e-8


def test_selector_initializes_all_remaining_fragments_before_filling_candidates():
    scores = np.linspace(1.0, 0.0, 202, dtype=np.float64)
    selector = _DiverseSelector(
        scores,
        np.zeros((len(scores), 2), dtype=np.float64),
        np.asarray([f"sample-{index:03d}" for index in range(len(scores))]),
        seed=17,
        sigma_effective=1.0,
    )

    selector.select(101)

    assert selector.selected == list(range(101))
    assert selector.candidates == set(range(101, 200))
    assert selector.silent == {200, 201}
    np.testing.assert_array_equal(selector.update_counts[:100], 0)
    np.testing.assert_array_equal(selector.update_counts[100:], 100)


def test_promotion_update_count_uses_ceiling_of_logarithmic_threshold():
    selector = _DiverseSelector(
        np.linspace(1.0, 0.0, 103, dtype=np.float64),
        np.zeros((103, 2), dtype=np.float64),
        np.asarray([f"sample-{index:03d}" for index in range(103)]),
        seed=17,
        sigma_effective=1.0,
    )

    assert selector._required_update_count(101) == 100
    assert selector._required_update_count(102) == 101
    assert selector._required_update_count(103) == 102


def test_catch_up_sampling_is_uniform_without_replacement_within_a_promotion():
    selector = _DiverseSelector(
        np.linspace(1.0, 0.0, 120, dtype=np.float64),
        np.zeros((120, 2), dtype=np.float64),
        np.asarray([f"sample-{index:03d}" for index in range(120)]),
        seed=17,
        sigma_effective=1.0,
    )
    selector.selected.extend(range(100))

    references = selector._sample_catch_up_references(50)

    assert len(references) == 50
    assert len(set(references)) == 50
    assert set(references) <= set(selector.selected)


def test_selection_promotes_one_silent_fragment_after_each_candidate_selection():
    scores = np.linspace(1.0, 0.0, 205, dtype=np.float64)
    selector = _DiverseSelector(
        scores,
        np.zeros((len(scores), 2), dtype=np.float64),
        np.asarray([f"sample-{index:03d}" for index in range(len(scores))]),
        seed=17,
        sigma_effective=1.0,
    )

    selector.select(104)

    assert selector.selected == list(range(104))
    assert selector.candidates == {*range(104, 200), 200, 201, 202}
    assert selector.silent == {203, 204}
    assert selector.update_counts[200] == 102
    assert selector.update_counts[201] == 102
    assert selector.update_counts[202] == 102
    assert selector.update_counts[203] == 100
    assert selector._neighbor_indices[203].tolist() == [0, 1, 2, 3, 4]
    assert selector._neighbor_indices[204].tolist() == [0, 1, 2, 3, 4]


def test_reranking_penalizes_a_high_scoring_duplicate_of_initial_selection():
    scores = np.concatenate(
        [
            np.linspace(1.0, 0.901, 100, dtype=np.float64),
            np.asarray([0.9, 0.8], dtype=np.float64),
        ]
    )
    embeddings = np.concatenate(
        [
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (101, 1)),
            np.asarray([[-1.0, 0.0]], dtype=np.float32),
        ],
        axis=0,
    )
    sample_ids = tuple(f"sample-{index:03d}" for index in range(len(scores)))

    result = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=101,
        seed=23,
    )

    assert len(result.selected_indices) == 101
    assert result.selected_indices[-1] == 101
    assert 100 not in result.selected_indices
    assert result.adjusted_scores[-1] == pytest.approx(0.8)
    assert result.knn_penalties[-1] == pytest.approx(0.0)
    assert result.sigma_raw == 0.0
    assert result.sigma_effective == 1.0e-8


def test_selected_diagnostic_uses_knn_penalty_and_unit_lambda():
    scores = np.concatenate(
        [
            np.linspace(1.0, 0.901, 100, dtype=np.float64),
            np.asarray([0.9, 0.8], dtype=np.float64),
        ]
    )
    embeddings = np.concatenate(
        [
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (101, 1)),
            np.asarray([[-1.0, 0.0]], dtype=np.float32),
        ],
        axis=0,
    )
    sample_ids = tuple(f"sample-{index:03d}" for index in range(len(scores)))

    result = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=102,
        seed=23,
    )

    assert result.selected_indices[-2:].tolist() == [101, 100]
    assert result.knn_penalties[-1] == pytest.approx(0.998)
    assert result.adjusted_scores[-1] == pytest.approx(-0.098)


def test_knn_penalty_uses_actual_neighbor_count_and_can_decrease():
    scores = np.asarray([0.8, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    embeddings = np.asarray(
        [[0.0], [0.1], [0.01], [0.02], [0.03], [0.04]],
        dtype=np.float64,
    )
    selector = _DiverseSelector(
        scores,
        embeddings,
        np.asarray(["target", "one", "two", "three", "four", "five"]),
        seed=1,
        sigma_effective=1.0,
    )

    selector._update([0], [1])
    assert selector.penalties[0] == pytest.approx(0.9950124791926823)

    selector._update([0], [2, 3, 4, 5])
    assert selector.penalties[0] == pytest.approx(0.19900249583853646)
    assert selector.adjusted[0] == pytest.approx(0.6009975041614635)


def test_knn_penalty_retains_five_nearest_unique_references_across_updates():
    scores = np.asarray([0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
    embeddings = np.asarray(
        [[0.0], [0.1], [0.2], [0.3], [0.4], [0.05], [0.6], [0.7]],
        dtype=np.float64,
    )
    selector = _DiverseSelector(
        scores,
        embeddings,
        np.asarray([f"sample-{index}" for index in range(len(scores))]),
        seed=1,
        sigma_effective=1.0,
    )

    selector._update([0], [1, 2, 3, 4])
    selector._update([0], [4, 5, 6, 7])

    assert selector._neighbor_indices[0].tolist() == [5, 1, 2, 3, 4]
    np.testing.assert_allclose(
        selector._neighbor_similarities[0],
        [
            0.9987507809245809,
            0.9950124791926823,
            0.9801986733067553,
            0.9559974818331,
            0.9231163463866358,
        ],
    )
    assert selector.penalties[0] == pytest.approx(0.29019243122949884)


def test_knn_penalty_breaks_equal_similarity_ties_by_sample_id():
    scores = np.asarray([0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    embeddings = np.asarray([[0.0], *[[1.0]] * 6], dtype=np.float64)
    selector = _DiverseSelector(
        scores,
        embeddings,
        np.asarray(["target", "f", "e", "d", "c", "b", "a"]),
        seed=1,
        sigma_effective=1.0,
    )

    selector._update([0], [1, 2, 3, 4, 5, 6])

    assert selector.penalties[0] == pytest.approx(0.18195919791379003)


def test_knn_penalty_is_zero_when_all_rbf_similarities_underflow():
    scores = np.asarray([0.8, 1.0, 0.5], dtype=np.float64)
    embeddings = np.asarray([[0.0], [1.0], [-1.0]], dtype=np.float64)
    selector = _DiverseSelector(
        scores,
        embeddings,
        np.asarray(["target", "right", "left"]),
        seed=1,
        sigma_effective=1.0e-8,
    )

    selector._update([0], [1, 2])

    assert selector.penalties[0] == 0.0
    assert selector.adjusted[0] == pytest.approx(0.8)


def test_initial_candidate_order_breaks_raw_score_ties_by_sample_id():
    scores = np.linspace(1.0, 0.1, 206, dtype=np.float64)
    scores[101] = scores[100]
    embeddings = np.zeros((len(scores), 1), dtype=np.float64)
    sample_ids = tuple(f"sample-{index:03d}" for index in range(len(scores)))

    result = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=101,
        seed=4,
    )

    assert result.selected_indices[-1] == 100


def test_fixed_seed_replays_full_state_machine_without_duplicate_selection():
    rng = np.random.default_rng(91)
    embeddings = rng.normal(size=(620, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.linspace(1.0, 0.0, len(embeddings), dtype=np.float64)
    sample_ids = tuple(f"sample-{index:04d}" for index in range(len(scores)))

    first = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=len(scores),
        seed=2026,
    )
    second = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=len(scores),
        seed=2026,
    )

    assert len(first.selected_indices) == len(scores)
    assert set(first.selected_indices.tolist()) == set(range(len(scores)))
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    np.testing.assert_allclose(first.adjusted_scores, second.adjusted_scores)
    np.testing.assert_allclose(first.knn_penalties, second.knn_penalties)


def test_filter_run_writes_ceil_percent_aligned_artifacts_and_manifest(tmp_path: Path):
    input_root, source_rows, source_embeddings = _write_sqcn_run(tmp_path, sample_count=200)

    output_root = filter_sqcn_run(input_root, 50, seed=77)

    assert output_root == input_root / "filter" / "top50pct"
    with (output_root / "scores.csv").open(encoding="utf-8", newline="") as handle:
        filtered_rows = list(csv.DictReader(handle))
    filtered_embeddings = np.load(output_root / "embeddings.npy")
    manifest = json.loads((output_root / "filter_manifest.json").read_text(encoding="utf-8"))

    assert [row["sample_id"] for row in filtered_rows] == [
        f"sample-{index}" for index in range(100)
    ]
    assert [int(row["filter_rank"]) for row in filtered_rows] == list(range(1, 101))
    assert list(filtered_rows[0]) == [
        *SCORE_COLUMNS,
        "filter_rank",
        "adjusted_score",
        "knn_penalty",
    ]
    np.testing.assert_array_equal(filtered_embeddings, source_embeddings[:100])
    assert manifest["version"] == "0.4.0"
    assert manifest["status"] == "complete"
    assert manifest["algorithm"]["percent"] == 50.0
    assert manifest["algorithm"]["target_size"] == 100
    assert manifest["algorithm"]["seed"] == 77
    assert manifest["algorithm"]["score_column"] == "sqcn"
    assert manifest["algorithm"]["lambda"] == 1.0
    assert manifest["algorithm"]["penalty"] == {
        "policy": "mean_rbf_similarity_weighted_score_of_nearest_references",
        "neighbor_count": 5,
        "weight": "rbf_similarity",
        "aggregation": "sum(similarity * score) / effective_neighbor_count",
    }
    assert manifest["algorithm"]["silent"] == {
        "policy": "frozen_adjusted_score_heap",
        "initial_population": "all_non_initial_fragments",
        "initial_candidate_fill": "top_adjusted_score_up_to_candidate_capacity",
        "steady_promotion": "one_after_each_selection",
    }
    assert manifest["algorithm"]["update_count"] == {
        "unit": "reference_fragments",
        "initial": 100,
        "promotion_minimum": "ceil(100 + log2(selected_count - 100))",
        "catch_up_sampling": "uniform_without_replacement_from_selected",
        "persisted": "internal_only",
    }
    assert manifest["algorithm"]["constants"] == {
        "init_select_size": 100,
        "candidate_capacity": 100,
        "neighbor_count": 5,
    }
    assert manifest["counts"] == {"input_fragments": 200, "selected_fragments": 100}
    assert len(manifest["selection_sha256"]) == 64
    assert manifest["score_columns"] == [
        *SCORE_COLUMNS,
        "filter_rank",
        "adjusted_score",
        "knn_penalty",
    ]
    assert manifest["outputs"]["manifest"] == str(output_root / "filter_manifest.json")
    assert len(source_rows) == 200
    assert source_rows[0]["sample_id"] == "sample-0"
    assert source_rows[-1]["sample_id"] == "sample-199"


def test_quality_only_ranks_by_quality_and_preserves_aligned_artifacts(tmp_path: Path):
    input_root, _, source_embeddings = _write_sqcn_run(
        tmp_path,
        quality_values=np.linspace(0.0, 1.0, 200, dtype=np.float64).tolist(),
    )

    output_root = filter_sqcn_run(input_root, 50, seed=77, quality_only=True)

    with (output_root / "scores.csv").open(encoding="utf-8", newline="") as handle:
        filtered_rows = list(csv.DictReader(handle))
    filtered_embeddings = np.load(output_root / "embeddings.npy")
    manifest = json.loads((output_root / "filter_manifest.json").read_text(encoding="utf-8"))

    expected_indices = np.arange(199, 99, -1)
    assert [row["sample_id"] for row in filtered_rows] == [
        f"sample-{index}" for index in expected_indices
    ]
    np.testing.assert_allclose(
        [float(row["adjusted_score"]) for row in filtered_rows],
        np.linspace(1.0, 100 / 199, 100),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        [float(row["knn_penalty"]) for row in filtered_rows],
        0.0,
    )
    np.testing.assert_array_equal(filtered_embeddings, source_embeddings[expected_indices])
    assert manifest["algorithm"]["seed"] == 77
    assert manifest["algorithm"]["score_column"] == "quality"
    assert manifest["algorithm"]["ordering"] == [
        "adjusted_score desc",
        "quality desc",
        "sample_id asc",
    ]


def test_quality_only_uses_separate_default_output_and_respects_explicit_output(
    tmp_path: Path,
):
    default_root, _, _ = _write_sqcn_run(tmp_path / "default", sample_count=200)
    custom_root, _, _ = _write_sqcn_run(tmp_path / "custom", sample_count=200)
    chosen_output = tmp_path / "chosen-quality-output"

    generated = filter_sqcn_run(default_root, 50, seed=1, quality_only=True)
    chosen = filter_sqcn_run(
        custom_root,
        50,
        output_dir=chosen_output,
        seed=1,
        quality_only=True,
    )

    assert generated == default_root / "filter" / "quality-top50pct"
    assert chosen == chosen_output


@pytest.mark.parametrize(
    "quality",
    ["invalid", -0.1, 1.1, float("nan"), float("inf")],
)
def test_quality_only_rejects_invalid_quality_values(tmp_path: Path, quality: object):
    input_root, rows, _ = _write_sqcn_run(tmp_path)
    rows[0]["quality"] = quality
    with (input_root / "fragment" / "scores.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="quality"):
        filter_sqcn_run(input_root, 20, quality_only=True)


def test_quality_only_requires_quality_column(tmp_path: Path):
    input_root, rows, _ = _write_sqcn_run(tmp_path)
    columns = [column for column in SCORE_COLUMNS if column != "quality"]
    with (input_root / "fragment" / "scores.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=r"missing columns: \['quality'\]"):
        filter_sqcn_run(input_root, 20, quality_only=True)


def test_filter_accepts_manifest_declared_dynamic_embedding_dimension(tmp_path: Path):
    input_root, _, source_embeddings = _write_sqcn_run(
        tmp_path,
        embedding_dim=7,
        sample_count=200,
    )

    output_root = filter_sqcn_run(input_root, 50, seed=77)

    filtered_embeddings = np.load(output_root / "embeddings.npy")
    assert filtered_embeddings.shape == (100, 7)
    np.testing.assert_array_equal(filtered_embeddings, source_embeddings[:100])


def test_module_cli_runs_filter_with_explicit_seed(tmp_path: Path):
    input_root, _, _ = _write_sqcn_run(tmp_path, sample_count=200)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sqcn.filtering",
            "--input-dir",
            str(input_root),
            "--percent",
            "50",
            "--seed",
            "1234",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    expected = input_root / "filter" / "top50pct"
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"sqcn_filter_output={expected}"
    manifest = json.loads((expected / "filter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"]["seed"] == 1234


def test_module_cli_quality_only_uses_quality_mode(tmp_path: Path):
    input_root, _, _ = _write_sqcn_run(
        tmp_path,
        quality_values=np.linspace(0.0, 1.0, 200, dtype=np.float64).tolist(),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sqcn.filtering",
            "--input-dir",
            str(input_root),
            "--percent",
            "50",
            "--seed",
            "1234",
            "--quality-only",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    expected = input_root / "filter" / "quality-top50pct"
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"sqcn_filter_output={expected}"
    with (expected / "scores.csv").open(encoding="utf-8", newline="") as handle:
        filtered_rows = list(csv.DictReader(handle))
    manifest = json.loads((expected / "filter_manifest.json").read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in filtered_rows] == [
        f"sample-{index}" for index in range(199, 99, -1)
    ]
    assert manifest["algorithm"]["score_column"] == "quality"


def test_custom_output_cannot_overlap_original_sqcn_artifacts(tmp_path: Path):
    input_root, _, _ = _write_sqcn_run(tmp_path)

    with pytest.raises(ValueError, match="overlap SQCN source artifacts"):
        filter_sqcn_run(
            input_root,
            20,
            output_dir=input_root / "reference",
            force=True,
        )

    assert not (input_root / "reference").exists()


def test_selector_rejects_embeddings_without_feature_dimensions():
    with pytest.raises(ValueError, match="positive feature dimension"):
        select_diverse_fragments(
            np.asarray([0.8, 0.7]),
            np.empty((2, 0), dtype=np.float32),
            ("a", "b"),
            target_size=1,
            seed=1,
        )


def test_selector_rejects_non_numeric_embeddings_cleanly():
    with pytest.raises(ValueError, match="numeric"):
        select_diverse_fragments(
            np.asarray([0.8, 0.7]),
            np.asarray([["left"], ["right"]]),
            ("a", "b"),
            target_size=1,
            seed=1,
        )


def test_generated_seed_is_returned_and_replays_random_selection():
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(220, 4)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = np.linspace(1.0, 0.1, 220)
    sample_ids = tuple(f"item-{index:03d}" for index in range(220))

    generated = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=150,
    )
    replayed = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=150,
        seed=generated.seed,
    )

    assert 0 <= generated.seed < 2**64
    np.testing.assert_array_equal(generated.selected_indices, replayed.selected_indices)
    np.testing.assert_allclose(generated.adjusted_scores, replayed.adjusted_scores)


def test_zero_scores_and_identical_embeddings_still_select_every_item():
    scores = np.zeros(102, dtype=np.float64)
    embeddings = np.ones((102, 3), dtype=np.float32)
    sample_ids = tuple(f"zero-{index:03d}" for index in range(102))

    result = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=102,
        seed=9,
    )

    assert len(result.selected_indices) == 102
    assert len(set(result.selected_indices.tolist())) == 102
    np.testing.assert_allclose(result.adjusted_scores, 0.0)
    np.testing.assert_allclose(result.knn_penalties, 0.0)


@pytest.mark.parametrize("percent", [0, -1, 100.1, float("nan"), float("inf")])
def test_filter_rejects_invalid_percent(tmp_path: Path, percent: float):
    input_root, _, _ = _write_sqcn_run(tmp_path)

    with pytest.raises(ValueError, match=r"in \(0, 100\]"):
        filter_sqcn_run(input_root, percent)


def test_filter_rejects_percent_that_selects_fewer_than_one_hundred_fragments(
    tmp_path: Path,
):
    input_root, _, _ = _write_sqcn_run(tmp_path, sample_count=200)

    with pytest.raises(ValueError, match="target_size must be at least 100"):
        filter_sqcn_run(input_root, 49.5, seed=1)


def test_filter_requires_force_to_replace_exact_output(tmp_path: Path):
    input_root, _, _ = _write_sqcn_run(tmp_path, sample_count=200)
    output_root = filter_sqcn_run(input_root, 50, seed=1)
    (output_root / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --force"):
        filter_sqcn_run(input_root, 50, seed=2)

    replaced = filter_sqcn_run(input_root, 50, seed=2, force=True)
    manifest = json.loads((replaced / "filter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"]["seed"] == 2
    assert not (replaced / "stale.txt").exists()


def test_filter_cleans_temporary_output_after_write_failure(
    tmp_path: Path,
    monkeypatch,
):
    input_root, _, _ = _write_sqcn_run(tmp_path, sample_count=200)
    output_root = tmp_path / "chosen-output"

    def fail_write(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected write failure")

    monkeypatch.setattr("sqcn.filtering.artifacts._write_scores", fail_write)

    with pytest.raises(RuntimeError, match="injected write failure"):
        filter_sqcn_run(input_root, 50, output_dir=output_root, seed=3)

    assert not output_root.exists()
    assert not list(tmp_path.glob(".chosen-output.sqcn-filter-*"))


def test_filter_rejects_incomplete_manifest_and_misaligned_embeddings(tmp_path: Path):
    incomplete_root, _, _ = _write_sqcn_run(tmp_path / "incomplete")
    (incomplete_root / "run_manifest.json").write_text(
        json.dumps({"status": "running", "embedding_dim": 128}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status='complete'"):
        filter_sqcn_run(incomplete_root, 20)

    misaligned_root, _, embeddings = _write_sqcn_run(
        tmp_path / "misaligned",
        embedding_dim=7,
    )
    np.save(misaligned_root / "fragment" / "embeddings.npy", embeddings[:-1])
    with pytest.raises(ValueError, match=r"shape \[5, 7\]"):
        filter_sqcn_run(misaligned_root, 20)


@pytest.mark.parametrize("embedding_dim", [0, -1, True, 1.5, "7"])
def test_filter_rejects_non_positive_or_non_integer_manifest_dimension(
    tmp_path: Path,
    embedding_dim: object,
):
    input_root, _, _ = _write_sqcn_run(tmp_path)
    (input_root / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "embedding_dim": embedding_dim}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
        filter_sqcn_run(input_root, 20)
