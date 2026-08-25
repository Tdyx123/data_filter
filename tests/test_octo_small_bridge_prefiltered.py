from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trajectory_data import EpisodeRecord


def _records() -> tuple[EpisodeRecord, ...]:
    return (
        EpisodeRecord(0, 40, 0, "task zero"),
        EpisodeRecord(1, 40, 1, "task one"),
    )


def test_bridge_prefiltered_selection_reads_cocore_jsonl_and_deduplicates_overlap(
    tmp_path: Path,
) -> None:
    from octo_small_bridge.selection import load_bridge_prefiltered_selection

    path = tmp_path / "selected_manifest.jsonl"
    rows = [
        {
            "sample_id": "ep000000_chunk_000000_000014",
            "episode_id": 0,
            "start_step": 0,
            "end_step": 14,
            "selected": True,
            "selection_score_delta": 1.5,
        },
        {
            "sample_id": "ep000000_chunk_000007_000021",
            "episode_id": 0,
            "start_step": 7,
            "end_step": 21,
            "selected": True,
        },
        {
            "sample_id": "ep000001_chunk_000010_000024",
            "episode_id": 1,
            "start_step": 10,
            "end_step": 24,
            "selected": True,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    selection = load_bridge_prefiltered_selection(
        path,
        _records(),
        action_horizon=8,
    )

    assert selection.frame_positions_by_episode == {
        0: tuple(range(22)),
        1: tuple(range(10, 25)),
    }
    assert selection.selected_fragments == 3
    assert selection.selected_episodes == 2
    assert selection.training_starts == 37
    assert len(selection.selection_sha256) == 64
    assert selection.as_manifest() == {
        "enabled": True,
        "mode": "prefiltered_fragments",
        "input_format": "jsonl",
        "source_path": str(path.resolve()),
        "source_sha256": selection.source_sha256,
        "required_fields": ["episode_id", "start_step", "end_step"],
        "selected_fragments": 3,
        "selected_episodes": 2,
        "training_starts": 37,
        "action_horizon": 8,
        "ordering": ["source row order"],
        "boundary_policy": "episode_tail_repeat_last_action",
        "overlap_policy": "deduplicate_episode_frame_start",
        "selection_sha256": selection.selection_sha256,
    }


def test_bridge_prefiltered_selection_supports_csv_and_rejects_unknown_episode(
    tmp_path: Path,
) -> None:
    from libero_lerobot.prefiltered import PriorSelectionError
    from octo_small_bridge.selection import load_bridge_prefiltered_selection

    valid = tmp_path / "selected.csv"
    valid.write_text(
        "episode_id,start_step,end_step,score\n1,3,5,0.9\n",
        encoding="utf-8",
    )

    selection = load_bridge_prefiltered_selection(
        valid,
        _records(),
        action_horizon=8,
    )

    assert selection.input_format == "csv"
    assert selection.frame_positions_by_episode == {1: (3, 4, 5)}

    invalid = tmp_path / "unknown.jsonl"
    invalid.write_text(
        json.dumps({"episode_id": 9, "start_step": 0, "end_step": 14}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PriorSelectionError, match="unknown episode_id=9"):
        load_bridge_prefiltered_selection(
            invalid,
            _records(),
            action_horizon=8,
        )


def test_bridge_prefiltered_selection_rejects_non_fragment_jsonl(
    tmp_path: Path,
) -> None:
    from libero_lerobot.prefiltered import PriorSelectionError
    from octo_small_bridge.selection import load_bridge_prefiltered_selection

    path = tmp_path / "datamil.jsonl"
    path.write_text(
        json.dumps({"trajectory_id": 0, "num_frames": 40}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PriorSelectionError, match="episode_id.*start_step.*end_step"):
        load_bridge_prefiltered_selection(
            path,
            _records(),
            action_horizon=8,
        )


def test_bridge_sampler_uses_only_prefiltered_positions_and_remains_resumable() -> None:
    from octo_small_bridge.data import BridgeDistributedBatchSampler

    records = tuple(
        EpisodeRecord(index, 6, index, f"task {index}") for index in range(4)
    )
    selected = {
        0: (0, 2, 4),
        1: (1, 3, 5),
        2: (0, 1, 5),
        3: (2, 3, 4),
    }
    samplers = [
        BridgeDistributedBatchSampler(
            records,
            local_batch_size=2,
            rank=rank,
            world_size=2,
            seed=17,
            num_batches=3,
            frame_positions_by_episode=selected,
        )
        for rank in range(2)
    ]

    batches_by_rank = [list(sampler) for sampler in samplers]

    for batches in batches_by_rank:
        assert all(
            ref.frame_position in selected[ref.episode_id]
            for batch in batches
            for ref in batch
        )
    rank_zero_episodes = {
        ref.episode_id for batch in batches_by_rank[0] for ref in batch
    }
    rank_one_episodes = {
        ref.episode_id for batch in batches_by_rank[1] for ref in batch
    }
    assert rank_zero_episodes.isdisjoint(rank_one_episodes)

    original = BridgeDistributedBatchSampler(
        records,
        local_batch_size=2,
        rank=0,
        world_size=2,
        seed=17,
        num_batches=3,
        frame_positions_by_episode=selected,
    )
    iterator = iter(original)
    next(iterator)
    state = original.state_dict()
    expected = next(iterator)
    restored = BridgeDistributedBatchSampler(
        records,
        local_batch_size=2,
        rank=0,
        world_size=2,
        seed=17,
        num_batches=3,
        frame_positions_by_episode=selected,
    )
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected


def test_bridge_training_data_wires_prefiltered_positions_without_changing_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octo_small_bridge import data as data_module
    from octo_small_official_pytorch.checkpoint import (
        OFFICIAL_BRIDGE_ACTION_MASK,
        OFFICIAL_BRIDGE_ACTION_MEAN,
        OFFICIAL_BRIDGE_ACTION_STD,
    )

    records = tuple(
        EpisodeRecord(index, 6, index, f"task {index}") for index in range(4)
    )

    class FakeAdapter:
        vector_observation_keys = ()
        image_observation_keys = ("observation.images.image_0",)

        def __init__(self, _config) -> None:
            pass

        def episodes(self):
            return records

        def fingerprint(self) -> str:
            return "a" * 64

    monkeypatch.setattr(data_module, "LeRobotDatasetAdapter", FakeAdapter)
    statistics_path = tmp_path / "dataset_statistics.json"
    statistics_path.write_text(
        json.dumps(
            {
                "bridge_dataset": {
                    "action": {
                        "mean": list(OFFICIAL_BRIDGE_ACTION_MEAN),
                        "std": list(OFFICIAL_BRIDGE_ACTION_STD),
                        "mask": list(OFFICIAL_BRIDGE_ACTION_MASK),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    selected_path = tmp_path / "selected_manifest.jsonl"
    selected_path.write_text(
        "".join(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "start_step": 1,
                    "end_step": 4,
                }
            )
            + "\n"
            for episode_id in range(4)
        ),
        encoding="utf-8",
    )
    config = {
        "data": {
            "dataset_name": "bridge_orig_1.0.0",
            "prior_selection": {"prefiltered_scores": str(selected_path)},
            "image_obs_keys": {"primary": "observation.images.image_0"},
                "action_key": "action",
                "window_size": 2,
                "action_horizon": 4,
            "resize": {"primary": [256, 256]},
        },
        "train": {
            "micro_batch_size_per_gpu": 2,
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "episode_cache_size": 1,
            "seed": 9,
            "num_workers_per_rank": 0,
            "prefetch_factor": 2,
        },
    }
    paths = {
        "dataset": tmp_path / "bridge",
        "statistics": statistics_path,
        "prior_prefiltered_scores": selected_path,
    }

    training_data = data_module.make_training_dataset(
        config,
        paths,
        tokenizer=None,
        rank=0,
        world_size=2,
    )

    assert training_data.prior_selection is not None
    assert training_data.prior_selection.training_starts == 16
    assert training_data.statistics_path == statistics_path
    batch = next(iter(training_data.batch_sampler))
    assert all(1 <= ref.frame_position <= 4 for ref in batch)


def test_bridge_preflight_reports_and_inspects_prefiltered_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octo_small_bridge import preflight

    records = tuple(
        EpisodeRecord(index, 6, index, f"task {index}") for index in range(4)
    )

    class FakeAdapter:
        def episodes(self):
            return records

    adapter = FakeAdapter()
    selected_path = tmp_path / "selected_manifest.jsonl"
    selected_path.write_text(
        "".join(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "start_step": 1,
                    "end_step": 4,
                }
            )
            + "\n"
            for episode_id in range(4)
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}
    checkpoint = SimpleNamespace(
        statistics_path=tmp_path / "dataset_statistics.json",
        as_dict=lambda: {"checkpoint_kind": "official_parity"},
    )
    statistics = SimpleNamespace(
        as_dict=lambda: {
            "path": str(checkpoint.statistics_path),
            "normalization": "mean_std",
        }
    )
    monkeypatch.setattr(preflight, "validate_official_checkpoint", lambda _path: checkpoint)
    monkeypatch.setattr(
        preflight,
        "load_official_action_statistics",
        lambda _path: statistics,
    )
    monkeypatch.setattr(preflight, "_adapter", lambda _path: adapter)

    def inspect(_path, *, adapter, statistics, frame_positions_by_episode=None):
        observed["inspection_adapter"] = adapter
        observed["inspection_statistics"] = statistics
        observed["frame_positions"] = frame_positions_by_episode
        return {
            "source_episodes": 4,
            "retained_episodes": 4,
            "excluded_empty_task_episodes": 0,
            "retained_frames": 24,
        }

    monkeypatch.setattr(preflight, "inspect_bridge_dataset", inspect)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.10.0",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 4,
                is_bf16_supported=lambda: True,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="5.2.0"),
    )
    config = {
        "data": {
            "action_horizon": 4,
            "prior_selection": {"prefiltered_scores": str(selected_path)},
            "expected_counts": {
                "source_episodes": 4,
                "retained_episodes": 4,
                "excluded_empty_task_episodes": 0,
                "retained_frames": 24,
            },
        },
        "train": {
            "gpu_count": 4,
            "gpu_ids": [0, 1, 2, 3],
            "batch_size": 8,
            "micro_batch_size_per_gpu": 2,
            "seed": 42,
        },
    }
    paths = {
        "model": tmp_path / "model",
        "dataset": tmp_path / "dataset",
        "prior_prefiltered_scores": selected_path,
    }

    report = preflight.run_preflight(config, paths)

    assert observed["inspection_adapter"] is adapter
    assert observed["inspection_statistics"] is statistics
    assert observed["frame_positions"] == {
        episode_id: (1, 2, 3, 4) for episode_id in range(4)
    }
    assert report["selection"]["enabled"] is True
    assert report["selection"]["selected_fragments"] == 4
    assert report["selection"]["selected_episodes"] == 4
    assert report["selection"]["training_starts"] == 16


def test_bridge_selection_hash_changes_with_prefiltered_manifest(tmp_path: Path) -> None:
    from octo_small_bridge.data import _selection_sha256

    statistics = tmp_path / "dataset_statistics.json"
    statistics.write_text("{}\n", encoding="utf-8")

    class FakeAdapter:
        def fingerprint(self) -> str:
            return "d" * 64

        def episodes(self):
            return _records()

    common = {
        "seed": 42,
        "world_size": 4,
        "local_batch_size": 8,
        "gradient_accumulation_steps": 4,
    }
    adapter = FakeAdapter()

    first = _selection_sha256(
        adapter,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=4,
        sampling_contract=common,
        statistics_path=statistics,
        prior_selection_sha256="a" * 64,
    )
    second = _selection_sha256(
        adapter,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=4,
        sampling_contract=common,
        statistics_path=statistics,
        prior_selection_sha256="b" * 64,
    )
    unfiltered = _selection_sha256(
        adapter,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=4,
        sampling_contract=common,
        statistics_path=statistics,
        prior_selection_sha256=None,
    )

    assert len({first, second, unfiltered}) == 3


def test_bridge_cli_does_not_accept_single_dash_prefiltered_option() -> None:
    from octo_small_bridge.cli import parse_arguments

    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--output-dir",
                "outputs/test",
                "-prior-prefiltered-scores",
                "/data/cocore/selected_manifest.jsonl",
            ]
        )


def test_bridge_preflight_rejects_selection_too_small_for_four_ranks(
    tmp_path: Path,
) -> None:
    from octo_small_bridge.preflight import _validate_prefiltered_sampling
    from octo_small_bridge.selection import load_bridge_prefiltered_selection

    path = tmp_path / "selected_manifest.jsonl"
    path.write_text(
        json.dumps({"episode_id": 0, "start_step": 0, "end_step": 14}) + "\n",
        encoding="utf-8",
    )
    selection = load_bridge_prefiltered_selection(
        path,
        _records(),
        action_horizon=8,
    )

    with pytest.raises(ValueError, match="world_size cannot exceed"):
        _validate_prefiltered_sampling(
            {
                "train": {
                    "gpu_count": 4,
                    "micro_batch_size_per_gpu": 1,
                    "seed": 42,
                }
            },
            _records(),
            selection,
        )


def test_bridge_preflight_rejects_invalid_selection_before_dataset_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octo_small_bridge import preflight

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps({"episode_id": 99, "start_step": 0, "end_step": 14}) + "\n",
        encoding="utf-8",
    )
    adapter = SimpleNamespace(episodes=_records)
    checkpoint = SimpleNamespace(
        statistics_path=tmp_path / "dataset_statistics.json",
        as_dict=lambda: {"checkpoint_kind": "official_parity"},
    )
    monkeypatch.setattr(preflight, "validate_official_checkpoint", lambda _path: checkpoint)
    monkeypatch.setattr(preflight, "_adapter", lambda _path: adapter)
    monkeypatch.setattr(
        preflight,
        "load_official_action_statistics",
        lambda _path: SimpleNamespace(as_dict=lambda: {}),
    )
    inspection_calls: list[object] = []
    monkeypatch.setattr(
        preflight,
        "inspect_bridge_dataset",
        lambda *args, **kwargs: inspection_calls.append((args, kwargs)),
    )
    config = {
        "data": {
            "action_horizon": 4,
            "prior_selection": {"prefiltered_scores": str(invalid)},
        },
        "train": {
            "gpu_count": 4,
            "micro_batch_size_per_gpu": 1,
            "seed": 42,
        },
    }
    paths = {
        "model": tmp_path / "model",
        "dataset": tmp_path / "dataset",
        "prior_prefiltered_scores": invalid,
    }

    with pytest.raises(preflight.PreflightError, match="unknown episode_id=99"):
        preflight.run_preflight(config, paths)

    assert inspection_calls == []
