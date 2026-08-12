import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from octo_small_libero import selection as selection_module


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


def _load(path: Path, metadata):
    return selection_module.load_prefiltered_selection(
        path,
        metadata,
        action_horizon=8,
    )


def test_prefiltered_selection_reads_csv_key_fields_and_deduplicates_overlap(
    tmp_path,
):
    metadata = _metadata(tmp_path)
    path = tmp_path / "scores.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "start_step",
                "end_step",
                "quality",
                "filter_rank",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "episode_id": 0,
                    "start_step": 0,
                    "end_step": 14,
                    "quality": 0.9,
                    "filter_rank": 1,
                },
                {
                    "episode_id": 0,
                    "start_step": 7,
                    "end_step": 21,
                    "quality": 0.8,
                    "filter_rank": 2,
                },
                {
                    "episode_id": 1,
                    "start_step": 10,
                    "end_step": 24,
                    "quality": 0.7,
                    "filter_rank": 3,
                },
            ]
        )

    selection = _load(path, metadata)

    assert selection.input_format == "csv"
    assert selection.selected_fragments == 3
    assert selection.selected_episodes == 2
    assert selection.frame_indices == (*range(22), *range(50, 65))
    assert selection.training_starts == 37
    manifest = selection.as_manifest()
    assert manifest["mode"] == "prefiltered_fragments"
    assert manifest["source_path"] == str(path.resolve())
    assert manifest["source_sha256"] == selection.source_sha256
    assert manifest["required_fields"] == [
        "episode_id",
        "start_step",
        "end_step",
    ]
    assert "filter_manifest_path" not in manifest
    assert "run_manifest_path" not in manifest
    assert "top_percent" not in manifest
    assert manifest["boundary_policy"] == "episode_tail_repeat_last_action"


def test_prefiltered_selection_reads_jsonl_and_ignores_extra_fields(tmp_path):
    metadata = _metadata(tmp_path)
    path = tmp_path / "selected_manifest.jsonl"
    rows = [
        {
            "sample_id": "anything",
            "episode_id": 0,
            "start_step": 2,
            "end_step": 10,
            "selected": False,
        },
        {
            "episode_id": "1",
            "start_step": "4",
            "end_step": "12",
            "score": -100,
        },
    ]
    path.write_text(
        "\n" + "\n".join(json.dumps(row) for row in rows) + "\n\n",
        encoding="utf-8",
    )

    selection = _load(path, metadata)

    assert selection.input_format == "jsonl"
    assert selection.selected_fragments == 2
    assert selection.frame_indices == (*range(2, 11), *range(44, 53))


def test_prefiltered_selection_accepts_fragments_shorter_than_action_horizon(
    tmp_path,
):
    metadata = _metadata(tmp_path)
    path = tmp_path / "short.csv"
    path.write_text(
        "episode_id,start_step,end_step\n0,3,5\n",
        encoding="utf-8",
    )

    selection = _load(path, metadata)

    assert selection.frame_indices == (3, 4, 5)


def test_resolve_prior_selection_uses_only_the_unified_config_key(
    tmp_path,
    monkeypatch,
):
    metadata = _metadata(tmp_path)
    path = tmp_path / "scores.csv"
    path.write_text(
        "episode_id,start_step,end_step\n0,0,14\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(selection_module, "LeRobotV2Metadata", lambda _root: metadata)
    paths = {
        "prior_dataset": metadata.root,
        "prior_prefiltered_scores": path,
    }

    assert (
        selection_module.resolve_prior_selection(
            {
                "data": {
                    "action_horizon": 8,
                    "prior_selection": {"prefiltered_scores": None},
                }
            },
            {"prior_dataset": metadata.root},
        )
        is None
    )

    selection = selection_module.resolve_prior_selection(
        {
            "data": {
                "action_horizon": 8,
                "prior_selection": {"prefiltered_scores": str(path)},
            }
        },
        paths,
    )

    assert selection is not None
    assert selection.as_manifest()["mode"] == "prefiltered_fragments"


def test_dataset_manifest_resolves_the_unified_prefiltered_selection(
    tmp_path,
    monkeypatch,
):
    from octo_small_libero import training as training_module

    statistics = tmp_path / "stats.json"
    statistics.write_text("{}", encoding="utf-8")
    selected_prior = SimpleNamespace(
        training_starts=23,
        as_manifest=lambda: {"enabled": True, "mode": "prefiltered_fragments"},
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
            "prior_selection": {"prefiltered_scores": "/data/selected.csv"},
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
        "mode": "prefiltered_fragments",
    }


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("empty.csv", "", "empty"),
        (
            "missing.csv",
            "episode_id,start_step\n0,0\n",
            "missing required fields.*end_step",
        ),
        (
            "invalid.csv",
            "episode_id,start_step,end_step\n0,one,14\n",
            "invalid integer.*start_step.*line 2",
        ),
        (
            "unknown.csv",
            "episode_id,start_step,end_step\n9,0,14\n",
            "unknown episode_id=9.*line 2",
        ),
        (
            "bounds.csv",
            "episode_id,start_step,end_step\n0,30,40\n",
            "exceeds episode 0.*line 2",
        ),
        (
            "duplicate.csv",
            "episode_id,start_step,end_step\n0,0,14\n0,0,14\n",
            "duplicate fragment.*line 3",
        ),
        (
            "bad.jsonl",
            '{"episode_id": 0, "start_step": 0, "end_step": 14}\n{bad}\n',
            "invalid JSON.*line 2",
        ),
        (
            "array.jsonl",
            '{"episode_id": 0, "start_step": 0, "end_step": 14}\n[]\n',
            "JSON object.*line 2",
        ),
    ],
)
def test_prefiltered_selection_rejects_invalid_files(
    tmp_path,
    name,
    content,
    message,
):
    metadata = _metadata(tmp_path)
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    with pytest.raises(selection_module.PriorSelectionError, match=message):
        _load(path, metadata)
