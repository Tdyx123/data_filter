from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "generate_libero_motion_primitive_distribution.py"
OUTPUT_NAME = "motion_primitive_distribution.csv"


def _axis_states(length: int, *, axis: int, step: float) -> np.ndarray:
    states = np.zeros((length, 8), dtype=np.float32)
    states[:, axis] = np.arange(length, dtype=np.float32) * np.float32(step)
    return states


def _write_dataset(
    root: Path,
    episodes: list[np.ndarray],
    *,
    include_state: bool = True,
) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)

    episode_rows = []
    for episode_index, states in enumerate(episodes):
        length = len(states)
        columns = {
            "action": pa.array(
                np.zeros((length, 7), dtype=np.float32).tolist(),
                type=pa.list_(pa.float32(), list_size=7),
            ),
            "timestamp": pa.array(
                np.arange(length, dtype=np.float32) / np.float32(10.0),
                type=pa.float32(),
            ),
            "frame_index": pa.array(np.arange(length), type=pa.int64()),
            "episode_index": pa.array([episode_index] * length, type=pa.int64()),
        }
        if include_state:
            columns["observation.state"] = pa.array(
                states.tolist(),
                type=pa.list_(pa.float32(), list_size=8),
            )
        table = pa.table(columns)
        pq.write_table(table, data / f"episode_{episode_index:06d}.parquet")
        episode_rows.append({"episode_index": episode_index, "length": length})

    info = {
        "codebase_version": "v2.0",
        "total_episodes": len(episodes),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "action": {"dtype": "float32", "shape": [7]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    if include_state:
        info["features"]["observation.state"] = {"dtype": "float32", "shape": [8]}
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows),
        encoding="utf-8",
    )


def _run_cli(
    dataset: Path,
    output: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(output),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_rows(output: Path) -> list[dict[str, str]]:
    with (output / OUTPUT_NAME).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cli_writes_global_distribution_without_crossing_episode_boundaries(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(
        dataset,
        [
            _axis_states(5, axis=0, step=0.02),
            _axis_states(5, axis=1, step=-0.02),
        ],
    )
    output = tmp_path / "nested" / "output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(output),
            "--horizon",
            "3",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with (output / OUTPUT_NAME).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"primitive": "move forward", "count": "2", "proportion": "0.5"},
        {"primitive": "move right", "count": "2", "proportion": "0.5"},
    ]
    assert "Processed episodes: 2" in result.stdout
    assert "Generated labels: 4" in result.stdout
    assert f"CSV: {(output / OUTPUT_NAME).resolve()}" in result.stdout


def test_cli_writes_header_only_when_no_labels_are_generated(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, [np.zeros((3, 8), dtype=np.float32)])
    output = tmp_path / "output"

    result = _run_cli(dataset, output, "--horizon", "3")

    assert result.returncode == 0, result.stderr
    assert (output / OUTPUT_NAME).read_text(encoding="utf-8") == (
        "primitive,count,proportion\n"
    )
    assert "Generated labels: 0" in result.stdout


def test_cli_passes_tail_strategy_and_threshold_to_classifier(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, [_axis_states(3, axis=0, step=0.02)])
    output = tmp_path / "output"

    result = _run_cli(
        dataset,
        output,
        "--horizon",
        "3",
        "--threshold",
        "0.03",
        "--tail-strategy",
        "clip",
    )

    assert result.returncode == 0, result.stderr
    assert _read_rows(output) == [
        {"primitive": "stop", "count": "2", "proportion": str(2 / 3)},
        {"primitive": "move forward", "count": "1", "proportion": str(1 / 3)},
    ]


def test_cli_rejects_dataset_without_state_and_leaves_no_csv(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(
        dataset,
        [np.zeros((4, 8), dtype=np.float32)],
        include_state=False,
    )
    output = tmp_path / "output"

    result = _run_cli(dataset, output)

    assert result.returncode == 1
    assert "observation.state" in result.stderr
    assert not (output / OUTPUT_NAME).exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--horizon", "2"),
        ("--threshold", "-0.01"),
        ("--threshold", "nan"),
    ],
)
def test_cli_rejects_invalid_classifier_settings(
    tmp_path: Path,
    arguments: tuple[str, str],
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, [np.zeros((4, 8), dtype=np.float32)])
    output = tmp_path / "output"

    result = _run_cli(dataset, output, *arguments)

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert not (output / OUTPUT_NAME).exists()
