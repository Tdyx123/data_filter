from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sqcn.filtering import filter_sqcn_run, select_diverse_fragments


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


def _write_sqcn_run(tmp_path: Path) -> tuple[Path, list[dict[str, object]], np.ndarray]:
    root = tmp_path / "sqcn-run"
    fragment = root / "fragment"
    fragment.mkdir(parents=True)
    values = [0.2, 0.9, 0.6, 0.8, 0.4]
    rows = [
        {
            "sample_id": f"sample-{index}",
            "episode_id": index,
            "start_step": 0,
            "end_step": 14,
            "length": 15,
            "quality": 0.5,
            "coverage": 0.5,
            "novelty": 0.5,
            "sqcn": score,
        }
        for index, score in enumerate(values)
    ]
    with (fragment / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    embeddings = np.zeros((len(rows), 128), dtype=np.float32)
    embeddings[np.arange(len(rows)), np.arange(len(rows))] = 1.0
    np.save(fragment / "embeddings.npy", embeddings)
    (root / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "embedding_dim": 128}),
        encoding="utf-8",
    )
    return root, rows, embeddings


def test_small_target_uses_raw_score_order_and_pairwise_sigma():
    scores = np.asarray([0.6, 0.9, 0.9, 0.1], dtype=np.float64)
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    sample_ids = ("d", "b", "a", "c")

    result = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=2,
        seed=17,
    )

    assert result.selected_indices.tolist() == [2, 1]
    np.testing.assert_allclose(result.adjusted_scores, [0.9, 0.9])
    np.testing.assert_allclose(result.max_penalties, [0.0, 0.0])
    assert result.sigma_raw == pytest.approx((4.0 + 2.0 * math.sqrt(2.0)) / 6.0)
    assert result.sigma_effective == result.sigma_raw
    assert result.seed == 17


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
    assert result.max_penalties[-1] == pytest.approx(0.0)
    assert result.sigma_raw == 0.0
    assert result.sigma_effective == 1.0e-8


def test_selected_diagnostic_uses_unit_penalty_lambda():
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
    assert result.max_penalties[-1] == pytest.approx(1.0)
    assert result.adjusted_scores[-1] == pytest.approx(-0.1)


def test_fixed_seed_replays_full_state_machine_without_duplicate_selection():
    rng = np.random.default_rng(91)
    embeddings = rng.normal(size=(520, 8)).astype(np.float32)
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
    np.testing.assert_allclose(first.max_penalties, second.max_penalties)


def test_filter_run_writes_ceil_percent_aligned_artifacts_and_manifest(tmp_path: Path):
    input_root, source_rows, source_embeddings = _write_sqcn_run(tmp_path)

    output_root = filter_sqcn_run(input_root, 26, seed=77)

    assert output_root == input_root / "filter" / "top26pct"
    with (output_root / "scores.csv").open(encoding="utf-8", newline="") as handle:
        filtered_rows = list(csv.DictReader(handle))
    filtered_embeddings = np.load(output_root / "embeddings.npy")
    manifest = json.loads((output_root / "filter_manifest.json").read_text(encoding="utf-8"))

    assert [row["sample_id"] for row in filtered_rows] == ["sample-1", "sample-3"]
    assert [int(row["filter_rank"]) for row in filtered_rows] == [1, 2]
    assert list(filtered_rows[0]) == [
        *SCORE_COLUMNS,
        "filter_rank",
        "adjusted_score",
        "max_penalty",
    ]
    np.testing.assert_array_equal(filtered_embeddings, source_embeddings[[1, 3]])
    assert manifest["status"] == "complete"
    assert manifest["algorithm"]["percent"] == 26.0
    assert manifest["algorithm"]["target_size"] == 2
    assert manifest["algorithm"]["seed"] == 77
    assert manifest["algorithm"]["lambda"] == 1.0
    assert manifest["counts"] == {"input_fragments": 5, "selected_fragments": 2}
    assert len(manifest["selection_sha256"]) == 64
    assert manifest["score_columns"] == [
        *SCORE_COLUMNS,
        "filter_rank",
        "adjusted_score",
        "max_penalty",
    ]
    assert manifest["outputs"]["manifest"] == str(output_root / "filter_manifest.json")
    assert [row["sample_id"] for row in source_rows] == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
    ]


def test_module_cli_runs_filter_with_explicit_seed(tmp_path: Path):
    input_root, _, _ = _write_sqcn_run(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sqcn.filtering",
            "--input-dir",
            str(input_root),
            "--percent",
            "40",
            "--seed",
            "1234",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    expected = input_root / "filter" / "top40pct"
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"sqcn_filter_output={expected}"
    manifest = json.loads((expected / "filter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"]["seed"] == 1234


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
    np.testing.assert_allclose(result.max_penalties, 0.0)


@pytest.mark.parametrize("percent", [0, -1, 100.1, float("nan"), float("inf")])
def test_filter_rejects_invalid_percent(tmp_path: Path, percent: float):
    input_root, _, _ = _write_sqcn_run(tmp_path)

    with pytest.raises(ValueError, match=r"in \(0, 100\]"):
        filter_sqcn_run(input_root, percent)


def test_filter_requires_force_to_replace_exact_output(tmp_path: Path):
    input_root, _, _ = _write_sqcn_run(tmp_path)
    output_root = filter_sqcn_run(input_root, 20, seed=1)
    (output_root / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --force"):
        filter_sqcn_run(input_root, 20, seed=2)

    replaced = filter_sqcn_run(input_root, 20, seed=2, force=True)
    manifest = json.loads((replaced / "filter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"]["seed"] == 2
    assert not (replaced / "stale.txt").exists()


def test_filter_cleans_temporary_output_after_write_failure(
    tmp_path: Path,
    monkeypatch,
):
    input_root, _, _ = _write_sqcn_run(tmp_path)
    output_root = tmp_path / "chosen-output"

    def fail_write(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected write failure")

    monkeypatch.setattr("sqcn.filtering.artifacts._write_scores", fail_write)

    with pytest.raises(RuntimeError, match="injected write failure"):
        filter_sqcn_run(input_root, 20, output_dir=output_root, seed=3)

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

    misaligned_root, _, embeddings = _write_sqcn_run(tmp_path / "misaligned")
    np.save(misaligned_root / "fragment" / "embeddings.npy", embeddings[:-1])
    with pytest.raises(ValueError, match=r"shape \[5, 128\]"):
        filter_sqcn_run(misaligned_root, 20)
