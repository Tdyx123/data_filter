from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_random_clips_per_task.py"


def _write_metadata(
    root: Path,
    episodes: list[tuple[int, str]],
    tasks: dict[int, str],
) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "total_episodes": len(episodes),
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                ),
                "video_path": None,
                "features": {
                    "action": {"dtype": "float32", "shape": [7]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "observation.state": {"dtype": "float32", "shape": [8]},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "episode_index": episode_id,
                    "length": length,
                    "tasks": [task_name],
                }
            )
            + "\n"
            for episode_id, (length, task_name) in enumerate(episodes)
        ),
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": task_index, "task": task_name}) + "\n"
            for task_index, task_name in sorted(tasks.items())
        ),
        encoding="utf-8",
    )


def _run_cli(
    dataset: Path,
    output: Path,
    *,
    top_k_percent: str = "25",
    seed: str | None = "42",
    force: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--dataset-root",
        str(dataset),
        "--output-dir",
        str(output),
        f"--top-k-percent={top_k_percent}",
    ]
    if seed is not None:
        command.extend(("--seed", seed))
    if force:
        command.append("--force")
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_selects_ceil_percent_per_task_with_relcore_locator_fields(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "libero-mini"
    _write_metadata(
        dataset,
        episodes=[
            (31, "task zero"),
            (15, "task zero"),
            (46, "task one"),
            (14, "short task"),
        ],
        tasks={0: "task zero", 1: "task one", 2: "short task"},
    )
    output = tmp_path / "selection"

    result = _run_cli(dataset, output, seed=None)

    assert result.returncode == 0, result.stderr
    rows = _read_jsonl(output / "selected_manifest.jsonl")
    assert len(rows) == 2
    assert [row["task_index"] for row in rows] == [0, 1]
    assert [row["selection_order"] for row in rows] == [1, 2]
    assert all(
        set(row)
        == {
            "sample_id",
            "dataset_name",
            "dataset_path",
            "episode_id",
            "task_index",
            "task_name",
            "start_step",
            "end_step",
            "length",
            "previous_sample_id",
            "next_sample_id",
            "selected",
            "selection_order",
        }
        for row in rows
    )
    assert all(row["dataset_name"] == "libero-mini" for row in rows)
    assert all(row["dataset_path"] == str(dataset.resolve()) for row in rows)
    assert all(row["selected"] is True and row["length"] == 15 for row in rows)
    assert all(
        row["end_step"] - row["start_step"] + 1 == row["length"]
        for row in rows
    )
    assert all(
        row["sample_id"]
        == (
            f"ep{row['episode_id']:06d}_fragment_"
            f"{row['start_step']:06d}_{row['end_step']:06d}"
        )
        for row in rows
    )

    report = json.loads((output / "selection_report.json").read_text(encoding="utf-8"))
    assert report == {
        "clip": {"length": 15, "stride": 15},
        "dataset_name": "libero-mini",
        "dataset_path": str(dataset.resolve()),
        "number_of_clips": 8,
        "number_of_episodes": 4,
        "seed": 42,
        "selected_clips": 2,
        "selection_method": "per_task_random",
        "selection_ratio": 0.25,
        "skipped_short_episodes": [3],
        "task_candidate_counts": {"0": 4, "1": 4},
        "task_counts": {"0": 1, "1": 1},
        "task_names": {"0": "task zero", "1": "task one"},
        "top_k_percent": 25.0,
    }
    assert "Candidates: 8" in result.stdout
    assert "Selected: 2" in result.stdout
    assert str((output / "selected_manifest.jsonl").resolve()) in result.stdout
    assert str((output / "selection_report.json").resolve()) in result.stdout


def test_same_seed_is_reproducible_and_different_seed_changes_selection(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_metadata(
        dataset,
        episodes=[(151, "task zero"), (151, "task one")],
        tasks={0: "task zero", 1: "task one"},
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"

    assert _run_cli(dataset, first, top_k_percent="20", seed="123").returncode == 0
    assert _run_cli(dataset, second, top_k_percent="20", seed="123").returncode == 0
    assert _run_cli(dataset, third, top_k_percent="20", seed="124").returncode == 0

    first_rows = _read_jsonl(first / "selected_manifest.jsonl")
    second_rows = _read_jsonl(second / "selected_manifest.jsonl")
    third_rows = _read_jsonl(third / "selected_manifest.jsonl")
    assert first_rows == second_rows
    assert {row["sample_id"] for row in first_rows} != {
        row["sample_id"] for row in third_rows
    }
    assert [row["task_index"] for row in first_rows] == [0, 0, 0, 1, 1, 1]


def test_top_100_selects_every_candidate_and_training_loader_accepts_manifest(
    tmp_path: Path,
) -> None:
    from octo_small_libero.selection import load_prefiltered_selection

    dataset = tmp_path / "dataset"
    _write_metadata(
        dataset,
        episodes=[(31, "task zero"), (46, "task one")],
        tasks={0: "task zero", 1: "task one"},
    )
    output = tmp_path / "selection"

    result = _run_cli(dataset, output, top_k_percent="100")

    assert result.returncode == 0, result.stderr
    rows = _read_jsonl(output / "selected_manifest.jsonl")
    assert len(rows) == 7
    assert len({row["sample_id"] for row in rows}) == 7
    assert {row["sample_id"] for row in rows} == {
        "ep000000_fragment_000000_000014",
        "ep000000_fragment_000015_000029",
        "ep000000_fragment_000016_000030",
        "ep000001_fragment_000000_000014",
        "ep000001_fragment_000015_000029",
        "ep000001_fragment_000030_000044",
        "ep000001_fragment_000031_000045",
    }
    metadata = SimpleNamespace(
        episodes=(
            SimpleNamespace(episode_index=0, length=31),
            SimpleNamespace(episode_index=1, length=46),
        ),
        global_offsets={0: 0, 1: 31},
    )
    loaded = load_prefiltered_selection(
        output / "selected_manifest.jsonl",
        metadata,
        action_horizon=8,
    )
    assert loaded.selected_fragments == 7
    assert loaded.selected_episodes == 2


@pytest.mark.parametrize("top_k_percent", ["0", "-1", "100.1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_top_k_percent_without_outputs(
    tmp_path: Path,
    top_k_percent: str,
) -> None:
    dataset = tmp_path / "dataset"
    _write_metadata(dataset, episodes=[(31, "task")], tasks={0: "task"})
    output = tmp_path / "selection"

    result = _run_cli(dataset, output, top_k_percent=top_k_percent)

    assert result.returncode != 0
    assert "top-k percent must be finite and in (0, 100]" in result.stderr
    assert not (output / "selected_manifest.jsonl").exists()
    assert not (output / "selection_report.json").exists()


def test_cli_rejects_dataset_without_task_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_metadata(dataset, episodes=[(31, "task")], tasks={0: "task"})
    (dataset / "meta" / "tasks.jsonl").unlink()
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 31}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "selection"

    result = _run_cli(dataset, output)

    assert result.returncode == 1
    assert "task metadata" in result.stderr
    assert not output.exists()


def test_cli_rejects_dataset_without_complete_clips(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_metadata(dataset, episodes=[(14, "task")], tasks={0: "task"})
    output = tmp_path / "selection"

    result = _run_cli(dataset, output)

    assert result.returncode == 1
    assert "no complete 15-frame clips" in result.stderr
    assert not output.exists()


def test_existing_outputs_require_force_and_unrelated_files_are_preserved(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_metadata(
        dataset,
        episodes=[(151, "task zero"), (151, "task one")],
        tasks={0: "task zero", 1: "task one"},
    )
    output = tmp_path / "selection"
    first = _run_cli(dataset, output, top_k_percent="20", seed="42")
    assert first.returncode == 0, first.stderr
    unrelated = output / "keep.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    manifest_before = (output / "selected_manifest.jsonl").read_bytes()
    report_before = (output / "selection_report.json").read_bytes()

    refused = _run_cli(dataset, output, top_k_percent="20", seed="43")

    assert refused.returncode == 1
    assert "pass --force" in refused.stderr
    assert (output / "selected_manifest.jsonl").read_bytes() == manifest_before
    assert (output / "selection_report.json").read_bytes() == report_before

    replaced = _run_cli(dataset, output, top_k_percent="20", seed="43", force=True)

    assert replaced.returncode == 0, replaced.stderr
    report = json.loads((output / "selection_report.json").read_text(encoding="utf-8"))
    assert report["seed"] == 43
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not list(output.glob(".*.tmp"))
