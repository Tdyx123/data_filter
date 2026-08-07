from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from libero_lerobot.sampling import GloballyBalancedDistributedBatchSampler


def test_shared_metadata_reads_the_libero_lerobot_v2_contract():
    dataset = Path("/data/dwb/datasets/LIBERO_lerobot/libero10_5")
    if not dataset.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")

    from libero_lerobot.metadata import LeRobotV2Metadata

    metadata = LeRobotV2Metadata(dataset)

    assert metadata.info["robot_type"] == "libero"
    assert metadata.info["total_episodes"] == 50
    assert len(metadata.tasks) == 10


def test_octo_compatibility_error_catches_shared_metadata_failures(tmp_path):
    from libero_lerobot.metadata import LeRobotV2Metadata
    from octo_small_libero.data import LiberoDataError as OctoLiberoDataError

    with pytest.raises(OctoLiberoDataError, match="Missing LeRobot"):
        LeRobotV2Metadata(tmp_path / "missing")


def test_shared_tdus_selection_expands_only_complete_action_windows(tmp_path):
    from libero_lerobot.selection import load_prior_selection

    dataset = (tmp_path / "prior" / "libero90").resolve()
    dataset.mkdir(parents=True)
    metadata = SimpleNamespace(
        root=dataset,
        episodes=(SimpleNamespace(episode_index=0, length=20, tasks=("task",)),),
        global_offsets={0: 0},
        tasks={0: "task"},
        metadata_sha256=lambda: "metadata-sha256",
    )
    scores = tmp_path / "tdus" / "libero90" / "chunk" / "scores.csv"
    scores.parent.mkdir(parents=True)
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "ep000000_chunk_000000_000009",
                "episode_id": 0,
                "start_step": 0,
                "end_step": 9,
                "length": 10,
                "quality": 0.5,
                "coverage": 0.5,
                "diversity": 0.5,
                "novelty": 0.5,
                "tdus": 0.5,
            }
        )
    (scores.parent.parent / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "libero90",
                "dataset_path": str(dataset),
                "modes": ["trajectory", "chunk"],
            }
        ),
        encoding="utf-8",
    )

    selection = load_prior_selection(scores, 100.0, metadata, action_horizon=8)

    assert selection.frame_indices == (0, 1, 2)


def test_shared_target_selection_resolves_all_ten_tasks_and_fifty_episodes():
    dataset = Path("/data/dwb/datasets/LIBERO_lerobot/libero10_5")
    if not dataset.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from libero_lerobot.targets import resolve_target_selection

    selection = resolve_target_selection(
        dataset,
        dataset_name="libero10_5",
        task_indices=tuple(range(10)),
    )

    assert selection.selection_mode == "all"
    assert selection.task_indices == tuple(range(10))
    assert selection.episodes == 50
    assert selection.frames == 14_144


def test_octo_target_resolver_preserves_compatibility_result_type():
    dataset = Path("/data/dwb/datasets/LIBERO_lerobot/libero10_5")
    if not dataset.is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from octo_small_libero.data import (
        TargetTaskSelection,
        resolve_target_task_selection,
    )

    selection = resolve_target_task_selection(
        {
            "data": {
                "target_task_index": None,
                "target_all_tasks": True,
                "target_dataset": "libero10_5",
            }
        },
        {"target_dataset": dataset},
    )

    assert isinstance(selection, TargetTaskSelection)


def test_octo_prefiltered_prior_import_path_uses_shared_result_type():
    import libero_lerobot.prefiltered as shared_selection
    import octo_small_libero.selection as octo_selection

    assert (
        octo_selection.PrefilteredPriorSelection
        is shared_selection.PrefilteredPriorSelection
    )
    assert (
        octo_selection.load_prefiltered_selection
        is shared_selection.load_prefiltered_selection
    )


def test_global_sampler_balances_one_to_one_across_four_single_sample_ranks():
    samplers = [
        GloballyBalancedDistributedBatchSampler(
            source_sizes=(20, 20),
            local_batch_size=1,
            sample_weights=(1.0, 1.0),
            rank=rank,
            world_size=4,
            seed=42,
            num_batches=4,
        )
        for rank in range(4)
    ]
    batches_by_rank = [list(sampler) for sampler in samplers]

    for step in range(4):
        global_batch = [batches_by_rank[rank][step][0] for rank in range(4)]
        assert Counter(index.source for index in global_batch) == Counter({0: 2, 1: 2})
        assert len({(index.source, index.frame) for index in global_batch}) == 4
    assert [batches_by_rank[0][step][0].source for step in range(4)] == [0, 1, 0, 1]


def test_global_sampler_rejects_weights_without_integer_global_microbatch_quota():
    with pytest.raises(ValueError, match="global micro-batch"):
        GloballyBalancedDistributedBatchSampler(
            source_sizes=(20, 20),
            local_batch_size=1,
            sample_weights=(3.0, 1.0),
            rank=0,
            world_size=2,
            seed=42,
            num_batches=1,
        )
