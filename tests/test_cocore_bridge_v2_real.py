from __future__ import annotations

import json
from pathlib import Path

import pytest

from cocore.index import build_clip_records
from cocore.temporal import resolve_temporal_geometry
from cocore_bridge_v2 import cli
from cocore_bridge_v2.config import DEFAULT_DATASET_PATH, build_config
from cocore_bridge_v2.preflight import validate_bridge_dataset
from trajectory_data import LeRobotDatasetAdapter


@pytest.mark.real_data
def test_mounted_bridge_metadata_parquet_and_av1_match_contract() -> None:
    if not DEFAULT_DATASET_PATH.is_dir():
        pytest.skip("BridgeData V2 dataset is not mounted")
    validate_bridge_dataset(DEFAULT_DATASET_PATH)
    config = build_config(relation="sequence", relation_weight=1.0)
    adapter = LeRobotDatasetAdapter(config["dataset"])
    records = list(adapter.episodes())
    geometry = resolve_temporal_geometry("bridge_v2")

    assert adapter.dataset_summary() == {
        "source_episodes": 53_192,
        "indexed_episodes": 38_660,
        "retained_episodes": 38_660,
        "excluded_episodes": 14_532,
        "excluded_empty_task_episodes": 14_532,
    }
    assert sum(record.length for record in records) == 1_305_714
    assert sum(record.length < geometry.clip_length for record in records) == 2
    clips = build_clip_records(records, clip_length=geometry.clip_length)
    assert len(clips) == 202_739
    assert int(len(clips) * 0.10 + 0.5) == 20_274

    episode = next(
        iter(
            adapter.iter_episode_subset(
                records[:1],
                num_workers=0,
                load_images=True,
            )
        )
    )
    assert episode.observations["observation.state"].shape == (episode.length, 8)
    assert episode.actions.shape == (episode.length, 7)
    assert episode.observations["observation.images.image_0"].shape == (
        episode.length,
        256,
        256,
        3,
    )


@pytest.mark.real_data
def test_mounted_bridge_scan_smoke_100(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if not DEFAULT_DATASET_PATH.is_dir():
        pytest.skip("BridgeData V2 dataset is not mounted")
    output = tmp_path / "bridge-scan-100"

    cli.main(
        [
            "scan",
            "--relation",
            "sequence",
            "--relation-weight",
            "1",
            "--max-episodes",
            "100",
            "--output-dir",
            str(output),
        ]
    )

    manifest = json.loads((output / "scan" / "manifest.json").read_text())
    assert manifest["scanned_episodes"] == 100
    assert manifest["clips"] > 0
    assert manifest["producer"] == "cocore"
    assert capsys.readouterr().out.startswith(f"cocore_output={output} clips=")
