from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "select_random_bridge_episodes.py"


def _write_dataset(root: Path, episodes: list[dict[str, object]]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "total_episodes": len(episodes),
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes),
        encoding="utf-8",
    )


def _run_cli(
    dataset: Path,
    output: Path,
    *extra_arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset),
            "--output-path",
            str(output),
            *extra_arguments,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, int]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_cli_selects_twenty_percent_of_nonempty_complete_episodes(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(
        dataset,
        [
            {"episode_index": 0, "tasks": ["pick"], "length": 3},
            {"episode_index": 1, "tasks": [" place "], "length": 5},
            *[
                {
                    "episode_index": episode_id,
                    "tasks": [f"task {episode_id}"],
                    "length": episode_id + 5,
                }
                for episode_id in range(2, 10)
            ],
            {"episode_index": 10, "tasks": [""], "length": 7},
            {"episode_index": 11, "tasks": ["   "], "length": 9},
        ],
    )
    output = tmp_path / "random.jsonl"

    result = _run_cli(dataset, output)

    assert result.returncode == 0, result.stderr
    assert _read_jsonl(output) == [
        {"episode_id": 0, "start_step": 0, "end_step": 2},
        {"episode_id": 1, "start_step": 0, "end_step": 4},
    ]
    assert "Source episodes: 12" in result.stdout
    assert "Excluded empty-task episodes: 2" in result.stdout
    assert "Selected episodes: 2" in result.stdout
    assert f"Output: {output.resolve()}" in result.stdout


def test_custom_percent_and_seed_are_reproducible(tmp_path: Path) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(
        dataset,
        [
            {
                "episode_index": episode_id,
                "tasks": [f"task {episode_id}"],
                "length": episode_id + 2,
            }
            for episode_id in range(20)
        ],
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    third = tmp_path / "third.jsonl"

    assert _run_cli(dataset, first, "--percent", "25", "--seed", "7").returncode == 0
    assert _run_cli(dataset, second, "--percent", "25", "--seed", "7").returncode == 0
    assert _run_cli(dataset, third, "--percent", "25", "--seed", "8").returncode == 0

    assert [row["episode_id"] for row in _read_jsonl(first)] == [1, 2, 4, 10, 12]
    assert _read_jsonl(second) == _read_jsonl(first)
    assert [row["episode_id"] for row in _read_jsonl(third)] == [4, 6, 7, 11, 12]


def test_percent_budget_rounds_up(tmp_path: Path) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(
        dataset,
        [
            {"episode_index": 0, "tasks": ["task zero"], "length": 3},
            {"episode_index": 1, "tasks": ["task one"], "length": 4},
            {"episode_index": 2, "tasks": ["task two"], "length": 5},
        ],
    )
    output = tmp_path / "random.jsonl"

    result = _run_cli(dataset, output, "--percent", "1")

    assert result.returncode == 0, result.stderr
    assert len(_read_jsonl(output)) == 1


@pytest.mark.parametrize("percent", ["0", "-1", "100.1", "nan", "inf", "-inf"])
def test_invalid_percent_is_rejected_without_output(
    tmp_path: Path,
    percent: str,
) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(
        dataset,
        [{"episode_index": 0, "tasks": ["task"], "length": 3}],
    )
    output = tmp_path / "random.jsonl"

    result = _run_cli(dataset, output, f"--percent={percent}")

    assert result.returncode != 0
    assert "percent must be finite and in (0, 100]" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("episodes", "message"),
    [
        (
            [
                {"episode_index": 0, "tasks": ["task zero"], "length": 3},
                {"episode_index": 0, "tasks": ["task one"], "length": 4},
            ],
            "duplicate episode_index=0",
        ),
        (
            [{"episode_index": 0, "tasks": ["task"], "length": 0}],
            "length must be a positive integer",
        ),
        (
            [{"episode_index": 0, "length": 3}],
            "tasks must be a one-item list",
        ),
        (
            [{"episode_index": 0, "tasks": ["", "task"], "length": 3}],
            "tasks must be a one-item list",
        ),
        (
            [{"episode_index": 0, "tasks": [None], "length": 3}],
            "task must be a string",
        ),
    ],
)
def test_invalid_episode_metadata_is_rejected(
    tmp_path: Path,
    episodes: list[dict[str, object]],
    message: str,
) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(dataset, episodes)
    output = tmp_path / "random.jsonl"

    result = _run_cli(dataset, output)

    assert result.returncode == 1
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_episode_count_must_match_info_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(
        dataset,
        [{"episode_index": 0, "tasks": ["task"], "length": 3}],
    )
    (dataset / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.0", "total_episodes": 2}),
        encoding="utf-8",
    )
    output = tmp_path / "random.jsonl"

    result = _run_cli(dataset, output)

    assert result.returncode == 1
    assert "episodes.jsonl has 1 entries, expected 2" in result.stderr
    assert not output.exists()


def test_dataset_without_nonempty_tasks_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(
        dataset,
        [
            {"episode_index": 0, "tasks": [""], "length": 3},
            {"episode_index": 1, "tasks": ["   "], "length": 4},
        ],
    )
    output = tmp_path / "random.jsonl"

    result = _run_cli(dataset, output)

    assert result.returncode == 1
    assert "dataset contains no nonempty-task episodes" in result.stderr
    assert not output.exists()


def test_existing_output_requires_force_and_force_replaces_it(tmp_path: Path) -> None:
    dataset = tmp_path / "bridge"
    _write_dataset(
        dataset,
        [{"episode_index": 3, "tasks": ["task"], "length": 6}],
    )
    output = tmp_path / "nested" / "random.jsonl"
    output.parent.mkdir()
    output.write_text("keep me\n", encoding="utf-8")

    refused = _run_cli(dataset, output, "--percent", "100")

    assert refused.returncode == 1
    assert "output already exists; pass --force" in refused.stderr
    assert output.read_text(encoding="utf-8") == "keep me\n"

    replaced = _run_cli(dataset, output, "--percent", "100", "--force")

    assert replaced.returncode == 0, replaced.stderr
    assert _read_jsonl(output) == [
        {"episode_id": 3, "start_step": 0, "end_step": 5}
    ]
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_output_is_accepted_by_bridge_prefiltered_loader(tmp_path: Path) -> None:
    from octo_small_bridge.selection import load_bridge_prefiltered_selection
    from trajectory_data import EpisodeRecord

    dataset = tmp_path / "bridge"
    episodes = [
        {
            "episode_index": episode_id,
            "tasks": [f"task {episode_id}"],
            "length": episode_id + 3,
        }
        for episode_id in range(4)
    ]
    _write_dataset(dataset, episodes)
    output = tmp_path / "random.jsonl"

    result = _run_cli(dataset, output, "--percent", "100")

    assert result.returncode == 0, result.stderr
    selection = load_bridge_prefiltered_selection(
        output,
        tuple(
            EpisodeRecord(
                int(row["episode_index"]),
                int(row["length"]),
                int(row["episode_index"]),
                str(row["tasks"][0]),
            )
            for row in episodes
        ),
        action_horizon=8,
    )
    assert selection.frame_positions_by_episode == {
        0: (0, 1, 2),
        1: (0, 1, 2, 3),
        2: (0, 1, 2, 3, 4),
        3: (0, 1, 2, 3, 4, 5),
    }
