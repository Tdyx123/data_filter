import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from octo_small_libero.selection import PriorSelectionError, load_prior_selection


SCORE_COLUMNS = [
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "diversity",
    "novelty",
    "tdus",
]


def _metadata(tmp_path: Path):
    root = (tmp_path / "prior" / "libero90").resolve()
    root.mkdir(parents=True)
    return SimpleNamespace(
        root=root,
        episodes=(
            SimpleNamespace(episode_index=0, length=40),
            SimpleNamespace(episode_index=1, length=40),
        ),
        global_offsets={0: 0, 1: 40},
    )


def _score_row(
    episode_id: int,
    start_step: int,
    end_step: int,
    tdus: float,
):
    return {
        "sample_id": (
            f"ep{episode_id:06d}_chunk_{start_step:06d}_{end_step:06d}"
        ),
        "episode_id": episode_id,
        "start_step": start_step,
        "end_step": end_step,
        "length": end_step - start_step + 1,
        "quality": 0.5,
        "coverage": 0.5,
        "diversity": 0.5,
        "novelty": 0.5,
        "tdus": tdus,
    }


def _write_scores(
    tmp_path: Path,
    metadata,
    rows,
    *,
    fieldnames=SCORE_COLUMNS,
    dataset_path: Path | None = None,
):
    root = tmp_path / "tdus" / "libero90"
    scores = root / "chunk" / "scores.csv"
    scores.parent.mkdir(parents=True)
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "libero90",
                "dataset_path": str(dataset_path or metadata.root),
                "modes": ["trajectory", "chunk"],
            }
        ),
        encoding="utf-8",
    )
    return scores


def test_selection_uses_stable_top_percent_strict_boundaries_and_dedup(tmp_path):
    metadata = _metadata(tmp_path)
    rows = [
        _score_row(1, 8, 31, 0.8),
        _score_row(0, 16, 39, 0.8),
        _score_row(1, 0, 31, 0.8),
        _score_row(0, 0, 31, 0.9),
    ]
    scores = _write_scores(tmp_path, metadata, rows)

    selection = load_prior_selection(
        scores,
        26,
        metadata,
        action_horizon=8,
    )

    assert selection.total_chunks == 4
    assert selection.selected_chunks == 2
    assert selection.selected_sample_ids == (
        "ep000000_chunk_000000_000031",
        "ep000000_chunk_000016_000039",
    )
    assert selection.selected_episodes == 1
    assert selection.frame_indices == tuple(range(33))
    assert selection.training_starts == 33
    assert len(selection.selection_sha256) == 64


@pytest.mark.parametrize("percent", [0, -1, 100.1, float("nan"), float("inf")])
def test_selection_rejects_invalid_percent(tmp_path, percent):
    metadata = _metadata(tmp_path)
    scores = _write_scores(
        tmp_path,
        metadata,
        [_score_row(0, 0, 31, 0.5)],
    )

    with pytest.raises(PriorSelectionError, match="in \\(0, 100\\]"):
        load_prior_selection(scores, percent, metadata, action_horizon=8)


def test_selection_rejects_missing_columns_duplicate_ids_and_out_of_bounds(tmp_path):
    metadata = _metadata(tmp_path)
    row = _score_row(0, 0, 31, 0.5)
    scores = _write_scores(
        tmp_path / "missing",
        metadata,
        [row],
        fieldnames=[column for column in SCORE_COLUMNS if column != "novelty"],
    )
    with pytest.raises(PriorSelectionError, match="missing columns"):
        load_prior_selection(scores, 10, metadata, action_horizon=8)

    scores = _write_scores(tmp_path / "duplicate", metadata, [row, row])
    with pytest.raises(PriorSelectionError, match="Duplicate sample_id"):
        load_prior_selection(scores, 10, metadata, action_horizon=8)

    scores = _write_scores(
        tmp_path / "bounds",
        metadata,
        [_score_row(0, 16, 47, 0.5)],
    )
    with pytest.raises(PriorSelectionError, match="exceeds episode"):
        load_prior_selection(scores, 10, metadata, action_horizon=8)


def test_selection_rejects_tdus_dataset_source_mismatch(tmp_path):
    metadata = _metadata(tmp_path)
    scores = _write_scores(
        tmp_path,
        metadata,
        [_score_row(0, 0, 31, 0.5)],
        dataset_path=tmp_path / "different" / "libero90",
    )

    with pytest.raises(PriorSelectionError, match="computed from"):
        load_prior_selection(scores, 10, metadata, action_horizon=8)


def test_current_libero90_scores_have_expected_top_counts():
    project_root = Path(__file__).resolve().parents[1]
    scores = project_root / "outputs" / "tdus" / "libero90" / "chunk" / "scores.csv"
    prior_root = Path("/data/dwb/datasets/LIBERO_lerobot/libero90")
    if not scores.is_file() or not prior_root.is_dir():
        pytest.skip("Current LIBERO-90 scores or dataset are not mounted")

    from octo_small_libero.lerobot_v2 import LeRobotV2Metadata

    metadata = LeRobotV2Metadata(prior_root)
    top10 = load_prior_selection(scores, 10, metadata, action_horizon=8)
    top20 = load_prior_selection(scores, 20, metadata, action_horizon=8)

    assert (top10.selected_chunks, top10.training_starts) == (3_943, 85_270)
    assert (top20.selected_chunks, top20.training_starts) == (7_886, 160_739)
