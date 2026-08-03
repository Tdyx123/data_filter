import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from octo_small_libero.selection import (
    PriorSelectionError,
    load_prefiltered_sqcn_selection,
    load_prior_selection,
    resolve_prior_selection,
)


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


SQCN_SCORE_COLUMNS = [
    "sample_id",
    "episode_id",
    "start_step",
    "end_step",
    "length",
    "quality",
    "coverage",
    "novelty",
    "sqcn",
    "filter_rank",
    "adjusted_score",
    "knn_penalty",
]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_id_sha256(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sqcn_row(episode_id: int, start_step: int, end_step: int, rank: int):
    return {
        "sample_id": (
            f"ep{episode_id:06d}_fragment_{start_step:06d}_{end_step:06d}"
        ),
        "episode_id": episode_id,
        "start_step": start_step,
        "end_step": end_step,
        "length": end_step - start_step + 1,
        "quality": 0.8,
        "coverage": 0.7,
        "novelty": 0.6,
        "sqcn": 0.75,
        "filter_rank": rank,
        "adjusted_score": 0.7,
        "knn_penalty": 0.05,
    }


def _write_sqcn_selection(
    tmp_path: Path,
    metadata,
    rows,
    *,
    input_fragments: int = 30,
    percent: float = 10.0,
    dataset_path: Path | None = None,
):
    root = tmp_path / "sqcn"
    output = root / "filter" / "top10pct"
    output.mkdir(parents=True)
    scores = output / "scores.csv"
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SQCN_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    run_manifest = root / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset_name": "libero90",
                "dataset_path": str(dataset_path or metadata.root),
            }
        ),
        encoding="utf-8",
    )
    filter_manifest = output / "filter_manifest.json"
    filter_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "source": {
                    "run_manifest": str(run_manifest),
                    "run_manifest_sha256": _file_sha256(run_manifest),
                },
                "algorithm": {
                    "percent": percent,
                    "target_size": len(rows),
                    "ordering": [
                        "adjusted_score desc",
                        "sqcn desc",
                        "sample_id asc",
                    ],
                },
                "counts": {
                    "input_fragments": input_fragments,
                    "selected_fragments": len(rows),
                },
                "selection_sha256": _sample_id_sha256(rows),
                "outputs": {"scores": str(scores)},
            }
        ),
        encoding="utf-8",
    )
    return scores, filter_manifest, run_manifest


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


def test_prefiltered_sqcn_uses_every_row_with_strict_boundaries_and_dedup(tmp_path):
    metadata = _metadata(tmp_path)
    rows = [
        _sqcn_row(0, 0, 14, 1),
        _sqcn_row(0, 7, 21, 2),
        _sqcn_row(1, 10, 24, 3),
    ]
    scores, filter_manifest, run_manifest = _write_sqcn_selection(
        tmp_path, metadata, rows
    )

    selection = load_prefiltered_sqcn_selection(
        scores,
        metadata,
        action_horizon=8,
    )

    assert selection.total_chunks == 30
    assert selection.selected_chunks == 3
    assert selection.selected_sample_ids == tuple(row["sample_id"] for row in rows)
    assert selection.selected_episodes == 2
    assert selection.frame_indices == (*range(15), *range(50, 58))
    assert selection.training_starts == 23
    manifest = selection.as_manifest()
    assert manifest["mode"] == "sqcn_prefiltered"
    assert manifest["top_percent"] == 10.0
    assert manifest["filter_manifest_path"] == str(filter_manifest)
    assert manifest["run_manifest_path"] == str(run_manifest)
    assert len(manifest["filter_manifest_sha256"]) == 64
    assert len(manifest["run_manifest_sha256"]) == 64
    assert len(selection.selection_sha256) == 64


def test_prefiltered_sqcn_rejects_legacy_max_penalty_column(tmp_path):
    metadata = _metadata(tmp_path)
    scores, _, _ = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 1)],
        input_fragments=10,
    )
    scores.write_text(
        scores.read_text(encoding="utf-8").replace("knn_penalty", "max_penalty"),
        encoding="utf-8",
    )

    with pytest.raises(PriorSelectionError, match="knn_penalty"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_requires_filter_manifest(tmp_path):
    metadata = _metadata(tmp_path)
    scores, filter_manifest, _ = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 1)],
        input_fragments=10,
    )
    filter_manifest.unlink()

    with pytest.raises(PriorSelectionError, match="filter manifest"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_rejects_changed_source_manifest(tmp_path):
    metadata = _metadata(tmp_path)
    scores, _, run_manifest = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 1)],
        input_fragments=10,
    )
    run_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset_name": "libero90",
                "dataset_path": str(metadata.root),
                "changed": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PriorSelectionError, match="run manifest SHA256"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_requires_source_run_manifest(tmp_path):
    metadata = _metadata(tmp_path)
    scores, _, run_manifest = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 1)],
        input_fragments=10,
    )
    run_manifest.unlink()

    with pytest.raises(PriorSelectionError, match="run manifest"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_rejects_dataset_source_mismatch(tmp_path):
    metadata = _metadata(tmp_path)
    scores, _, _ = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 1)],
        input_fragments=10,
        dataset_path=tmp_path / "different" / "libero90",
    )

    with pytest.raises(PriorSelectionError, match="computed from"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_rejects_changed_selection_digest(tmp_path):
    metadata = _metadata(tmp_path)
    scores, filter_manifest, _ = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 1)],
        input_fragments=10,
    )
    manifest = json.loads(filter_manifest.read_text(encoding="utf-8"))
    manifest["selection_sha256"] = "0" * 64
    filter_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PriorSelectionError, match="selection_sha256"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_rejects_noncontiguous_filter_rank(tmp_path):
    metadata = _metadata(tmp_path)
    scores, _, _ = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 2)],
        input_fragments=10,
    )

    with pytest.raises(PriorSelectionError, match="filter_rank"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_rejects_duplicate_ids_and_out_of_bounds(tmp_path):
    metadata = _metadata(tmp_path)
    row = _sqcn_row(0, 0, 14, 1)
    duplicate = dict(row, filter_rank=2)
    scores, _, _ = _write_sqcn_selection(
        tmp_path / "duplicate",
        metadata,
        [row, duplicate],
        input_fragments=20,
    )
    with pytest.raises(PriorSelectionError, match="Duplicate sample_id"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)

    scores, _, _ = _write_sqcn_selection(
        tmp_path / "bounds",
        metadata,
        [_sqcn_row(0, 30, 44, 1)],
        input_fragments=10,
    )
    with pytest.raises(PriorSelectionError, match="exceeds episode"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_prefiltered_sqcn_rejects_fragments_without_complete_action_window(tmp_path):
    metadata = _metadata(tmp_path)
    scores, _, _ = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 6, 1)],
        input_fragments=10,
    )

    with pytest.raises(PriorSelectionError, match="no complete action windows"):
        load_prefiltered_sqcn_selection(scores, metadata, action_horizon=8)


def test_resolve_prior_selection_routes_prefiltered_scores(tmp_path, monkeypatch):
    metadata = _metadata(tmp_path)
    scores, _, _ = _write_sqcn_selection(
        tmp_path,
        metadata,
        [_sqcn_row(0, 0, 14, 1)],
        input_fragments=10,
    )
    monkeypatch.setattr(
        "octo_small_libero.selection.LeRobotV2Metadata",
        lambda root: metadata,
    )
    config = {
        "data": {
            "action_horizon": 8,
            "prior_selection": {
                "scores": str(scores),
                "top_percent": None,
                "prefiltered": True,
            },
        }
    }

    selection = resolve_prior_selection(
        config,
        {"prior_dataset": metadata.root, "prior_scores": scores},
    )

    assert selection is not None
    assert selection.as_manifest()["mode"] == "sqcn_prefiltered"


def test_dataset_manifest_resolves_prefiltered_prior_selection(tmp_path, monkeypatch):
    from octo_small_libero import selection as selection_module
    from octo_small_libero import training as training_module

    statistics = tmp_path / "stats.json"
    statistics.write_text("{}", encoding="utf-8")
    selected_prior = SimpleNamespace(
        training_starts=23,
        as_manifest=lambda: {"enabled": True, "mode": "sqcn_prefiltered"},
    )
    target_selection = SimpleNamespace(
        episodes=50,
        frames=500,
        as_manifest=lambda: {"enabled": True},
    )

    class FakeMetadata:
        def __init__(self, root):
            self.root = Path(root)
            self.info = {"total_episodes": 10, "total_frames": 100}

        def metadata_sha256(self):
            return "metadata-sha256"

    monkeypatch.setattr(
        selection_module,
        "resolve_prior_selection",
        lambda config, paths: selected_prior,
    )
    monkeypatch.setattr(training_module, "LeRobotV2Metadata", FakeMetadata)
    monkeypatch.setattr(
        training_module,
        "load_lerobot_statistics",
        lambda path: {"num_trajectories": 10, "num_transitions": 100},
    )
    config = {
        "data": {
            "target_dataset": "libero10_5",
            "prior_dataset": "libero90",
            "sample_weights": [3.0, 1.0],
            "prior_selection": {
                "scores": "/data/sqcn/scores.csv",
                "top_percent": None,
                "prefiltered": True,
            },
        }
    }
    paths = {
        "lerobot": tmp_path,
        "target_dataset": tmp_path / "libero10_5",
        "prior_dataset": tmp_path / "libero90",
        "statistics": statistics,
    }

    manifest = training_module.build_dataset_manifest(
        config,
        paths,
        target_selection=target_selection,
    )

    assert manifest["datasets"]["libero90"]["frames_used"] == 23
    assert manifest["datasets"]["libero90"]["selection"] == {
        "enabled": True,
        "mode": "sqcn_prefiltered",
    }


@pytest.mark.real_data
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

    assert (top10.selected_chunks, top10.training_starts) == (4_671, 36_868)
    assert (top20.selected_chunks, top20.training_starts) == (9_341, 73_746)


@pytest.mark.real_data
def test_current_sqcn_top10_scores_have_expected_training_starts():
    scores = Path("/data/dwb/libero90_sqcn/filter/top10pct/scores.csv")
    prior_root = Path("/data/dwb/datasets/LIBERO_lerobot/libero90")
    if not scores.is_file() or not prior_root.is_dir():
        pytest.skip("Current SQCN Top 10% scores or dataset are not mounted")
    with scores.open(encoding="utf-8", newline="") as handle:
        columns = set(csv.DictReader(handle).fieldnames or [])
    if "knn_penalty" not in columns:
        pytest.skip("Current SQCN Top 10% scores use the legacy max_penalty schema")

    from octo_small_libero.lerobot_v2 import LeRobotV2Metadata

    selection = load_prefiltered_sqcn_selection(
        scores,
        LeRobotV2Metadata(prior_root),
        action_horizon=8,
    )

    assert selection.selected_chunks == 4_671
    assert selection.selected_episodes == 2_923
    assert selection.training_starts == 36_263
