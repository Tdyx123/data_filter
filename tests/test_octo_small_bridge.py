from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
av = pytest.importorskip("av")

from trajectory_data import (  # noqa: E402
    DatasetValidationError,
    EpisodeRecord,
    LeRobotDatasetAdapter,
)
from octo_small_bridge.data import (  # noqa: E402
    BridgeDistributedBatchSampler,
    BridgeFrameDataset,
    BridgeFrameRef,
    _normalize_gripper_actions,
)
from octo_small_bridge.normalization import (  # noqa: E402
    compute_bridge_v2_statistics,
)


AV1_FIXTURE_SKIP = pytest.mark.skip(
    reason="SVT-AV1 fixture encoding is too slow for the standard regression suite"
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_vector_bridge_fixture(root: Path) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "robot_type": "widowx",
                "total_episodes": 2,
                "total_frames": 4,
                "total_tasks": 2,
                "chunks_size": 1000,
                "fps": 5,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet"
                ),
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": 0, "task": "move the block"},
            {"task_index": 1, "task": ""},
        ],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["move the block"], "length": 2},
            {"episode_index": 1, "tasks": [""], "length": 2},
        ],
    )
    for episode_index, task_index in ((0, 0), (1, 1)):
        states = np.full((2, 8), episode_index + 1, dtype=np.float32)
        actions = np.full((2, 7), episode_index + 2, dtype=np.float32)
        table = pa.table(
            {
                "observation.state": pa.array(
                    states.tolist(), type=pa.list_(pa.float32(), list_size=8)
                ),
                "action": pa.array(
                    actions.tolist(), type=pa.list_(pa.float32(), list_size=7)
                ),
                "timestamp": pa.array([0.0, 0.2], type=pa.float32()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "episode_index": pa.array(
                    [episode_index, episode_index], type=pa.int64()
                ),
                "task_index": pa.array([task_index, task_index], type=pa.int64()),
            }
        )
        pq.write_table(table, data / f"episode_{episode_index:06d}.parquet")


def _vector_adapter(root: Path) -> LeRobotDatasetAdapter:
    return LeRobotDatasetAdapter(
        {
            "path": str(root),
            "use_images": False,
            "empty_task_policy": "exclude",
            "feature_keys": {
                "action": "action",
                "timestamp": "timestamp",
                "frame_index": "frame_index",
                "episode_index": "episode_index",
                "vector_observations": ["observation.state"],
            },
        }
    )


def _write_av1(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("av1", rate=5)
    except av.error.ValueError:
        container.close()
        pytest.skip("The PyAV build has no AV1 encoder")
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_video_bridge_fixture(
    root: Path,
    *,
    gripper_values: tuple[float, float, float] = (0.0, 0.928448975, 1.0),
) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    video_root = root / "videos" / "chunk-000" / "observation.images.image_0"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    video_root.mkdir(parents=True)
    features = {
        "observation.images.image_0": {
            "dtype": "video",
            "shape": [16, 16, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.fps": 5.0,
                "video.height": 16,
                "video.width": 16,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.0",
                "robot_type": "widowx",
                "total_episodes": 2,
                "total_frames": 5,
                "total_tasks": 2,
                "total_videos": 2,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 5,
                "splits": {"train": "0:2"},
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet"
                ),
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4"
                ),
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    stats = {
        "observation.state": {
            "mean": [1.0] * 8,
            "std": [2.0] * 7 + [0.0],
            "min": [-3.0] * 8,
            "max": [5.0] * 8,
        },
        "action": {
            "mean": [1.0] * 6 + [0.5],
            "std": [2.0] * 6 + [0.5],
            "min": [-3.0] * 6 + [0.0],
            "max": [5.0] * 6 + [1.0],
        },
    }
    (meta / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": 0, "task": "move the block"},
            {"task_index": 1, "task": ""},
        ],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["move the block"], "length": 3},
            {"episode_index": 1, "tasks": [""], "length": 2},
        ],
    )
    for episode_index, episode_length, task_index in ((0, 3, 0), (1, 2, 1)):
        states = np.stack(
            [
                np.asarray(
                    [1.0 + frame * 2.0] * 7 + [1.0], dtype=np.float32
                )
                for frame in range(episode_length)
            ]
        )
        episode_gripper_values = (
            gripper_values if episode_index == 0 else (0.0, 1.0)
        )
        actions = np.stack(
            [
                np.asarray(
                    [1.0 + frame * 2.0] * 6 + [episode_gripper_values[frame]],
                    dtype=np.float32,
                )
                for frame in range(episode_length)
            ]
        )
        table = pa.table(
            {
                "observation.state": pa.array(
                    states.tolist(), type=pa.list_(pa.float32(), list_size=8)
                ),
                "action": pa.array(
                    actions.tolist(), type=pa.list_(pa.float32(), list_size=7)
                ),
                "timestamp": pa.array(
                    [frame / 5.0 for frame in range(episode_length)],
                    type=pa.float32(),
                ),
                "frame_index": pa.array(range(episode_length), type=pa.int64()),
                "episode_index": pa.array(
                    [episode_index] * episode_length, type=pa.int64()
                ),
                "task_index": pa.array([task_index] * episode_length, type=pa.int64()),
            }
        )
        pq.write_table(table, data / f"episode_{episode_index:06d}.parquet")
        frames = np.stack(
            [
                np.full((16, 16, 3), 32 + frame * 32, dtype=np.uint8)
                for frame in range(episode_length)
            ]
        )
        _write_av1(video_root / f"episode_{episode_index:06d}.mp4", frames)


def _video_adapter(root: Path) -> LeRobotDatasetAdapter:
    return LeRobotDatasetAdapter(
        {
            "path": str(root),
            "use_images": True,
            "empty_task_policy": "exclude",
            "feature_keys": {
                "action": "action",
                "timestamp": "timestamp",
                "frame_index": "frame_index",
                "episode_index": "episode_index",
                "vector_observations": ["observation.state"],
                "image_observations": ["observation.images.image_0"],
            },
        }
    )


def _video_statistics(root: Path, adapter: LeRobotDatasetAdapter):
    return compute_bridge_v2_statistics(
        adapter,
        root / "normalization.json",
        epsilon=1.0e-6,
    )


def test_lerobot_adapter_loads_one_indexed_episode_without_scanning_others(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bridge"
    _write_vector_bridge_fixture(root)
    adapter = _vector_adapter(root)

    episode = adapter.load_episode(adapter.episodes()[0], load_images=False)

    assert episode.episode_id == 0
    assert episode.task_index == 0
    assert episode.task_name == "move the block"
    assert episode.observations["observation.state"].shape == (2, 8)
    np.testing.assert_array_equal(episode.actions, np.full((2, 7), 2.0))


def test_lerobot_adapter_rejects_record_excluded_from_its_index(tmp_path: Path) -> None:
    root = tmp_path / "bridge"
    _write_vector_bridge_fixture(root)
    adapter = _vector_adapter(root)

    with pytest.raises(ValueError, match="unknown or mismatched episode record 1"):
        adapter.load_episode(
            EpisodeRecord(episode_id=1, length=2, task_index=1, task_name=""),
            load_images=False,
        )


@AV1_FIXTURE_SKIP
def test_bridge_frame_dataset_filters_empty_tasks_and_builds_octo_sample(
    tmp_path: Path,
) -> None:
    class FakeTokenizer:
        def __call__(self, text: str, **kwargs):
            assert text == "move the block"
            assert kwargs == {
                "padding": "max_length",
                "truncation": True,
                "max_length": 16,
                "return_tensors": "np",
            }
            return {
                "input_ids": np.arange(16, dtype=np.int64)[None],
                "attention_mask": np.ones((1, 16), dtype=np.int64),
            }

    root = tmp_path / "bridge"
    _write_video_bridge_fixture(root)
    adapter = _video_adapter(root)
    dataset = BridgeFrameDataset(
        adapter,
        statistics=_video_statistics(root, adapter),
        dataset_name="bridge_orig_1.0.0",
        action_horizon=5,
        primary_size=(8, 8),
        episode_cache_size=1,
        seed=17,
        train=False,
        tokenizer=FakeTokenizer(),
    )

    assert adapter.dataset_summary() == {
        "source_episodes": 2,
        "indexed_episodes": 1,
        "retained_episodes": 1,
        "excluded_episodes": 1,
        "excluded_empty_task_episodes": 1,
    }
    sample = dataset[BridgeFrameRef(epoch=0, episode_id=0, frame_position=0)]

    assert sample["image_primary"].shape == (1, 3, 8, 8)
    assert sample["image_primary"].dtype == np.float32
    assert "image_wrist" not in sample
    np.testing.assert_allclose(
        sample["proprio"],
        np.asarray([[-1.0204082] * 7 + [1.0]], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        sample["action"][:, 0],
        [-1.0204082, 0.0, 1.0204082, 1.0204082, 1.0204082],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        sample["action"][:, 6],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        sample["action_pad_mask"],
        np.asarray(
            [
                [True] * 7,
                [True] * 7,
                [True] * 7,
                [False] * 7,
                [False] * 7,
            ]
        ),
    )
    assert sample["language_instruction"] == "move the block"
    np.testing.assert_array_equal(sample["language_input_ids"], np.arange(16))
    np.testing.assert_array_equal(sample["language_attention_mask"], np.ones(16))
    assert sample["episode_index"] == 0
    assert sample["frame_index"] == 0


@AV1_FIXTURE_SKIP
def test_bridge_frame_dataset_clips_gripper_roundoff_at_unit_interval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bridge"
    _write_video_bridge_fixture(
        root,
        gripper_values=(-1.0e-5, 1.0 + 1.0e-5, 0.5),
    )
    adapter = _video_adapter(root)
    dataset = BridgeFrameDataset(
        adapter,
        statistics=_video_statistics(root, adapter),
        dataset_name="bridge_orig_1.0.0",
        action_horizon=3,
        primary_size=(8, 8),
        train=False,
    )

    sample = dataset[BridgeFrameRef(epoch=0, episode_id=0, frame_position=0)]

    np.testing.assert_allclose(sample["action"][:, 6], [0.0, 1.0, 0.0])


@pytest.mark.parametrize("invalid_gripper", [-2.0e-5, 1.0 + 2.0e-5, np.nan, np.inf])
@AV1_FIXTURE_SKIP
def test_bridge_frame_dataset_rejects_invalid_continuous_gripper_with_episode_id(
    tmp_path: Path,
    invalid_gripper: float,
) -> None:
    root = tmp_path / "bridge"
    _write_video_bridge_fixture(
        root,
        gripper_values=(invalid_gripper, 0.5, 1.0),
    )
    adapter = _video_adapter(root)
    dataset = BridgeFrameDataset(
        adapter,
        statistics=_video_statistics(root, adapter),
        dataset_name="bridge_orig_1.0.0",
        action_horizon=1,
        primary_size=(8, 8),
        train=False,
    )

    with pytest.raises(DatasetValidationError, match=r"Episode 0"):
        dataset[BridgeFrameRef(epoch=0, episode_id=0, frame_position=0)]


@pytest.mark.parametrize("invalid_gripper", [np.nan, np.inf, -np.inf])
def test_normalize_gripper_actions_rejects_non_finite_values_with_episode_id(
    invalid_gripper: float,
) -> None:
    with pytest.raises(DatasetValidationError, match=r"Episode 314.*finite"):
        _normalize_gripper_actions(
            np.asarray([invalid_gripper], dtype=np.float32),
            episode_id=314,
        )


@pytest.mark.parametrize(
    "invalid_gripper",
    [
        pytest.param(
            np.nextafter(np.float32(-1.0e-5), np.float32(-np.inf)),
            id="nearest-below-lower-tolerance",
        ),
        pytest.param(np.float32(-2.0e-5), id="clearly-below-lower-tolerance"),
        pytest.param(
            np.nextafter(
                np.float32(1.0) + np.float32(1.0e-5),
                np.float32(np.inf),
            ),
            id="nearest-above-upper-tolerance",
        ),
        pytest.param(
            np.float32(1.0) + np.float32(2.0e-5),
            id="clearly-above-upper-tolerance",
        ),
    ],
)
def test_normalize_gripper_actions_rejects_values_outside_tolerance_with_episode_id(
    invalid_gripper: np.float32,
) -> None:
    with pytest.raises(DatasetValidationError, match=r"Episode 2718.*\[-1e-5"):
        _normalize_gripper_actions(
            np.asarray([invalid_gripper], dtype=np.float32),
            episode_id=2718,
        )


def test_bridge_sampler_is_rank_disjoint_reproducible_and_resumable() -> None:
    records = tuple(
        EpisodeRecord(
            episode_id=index,
            length=5,
            task_index=index,
            task_name=f"task {index}",
        )
        for index in range(6)
    )
    rank_zero = BridgeDistributedBatchSampler(
        records,
        local_batch_size=3,
        rank=0,
        world_size=2,
        seed=23,
        num_batches=4,
    )
    rank_one = BridgeDistributedBatchSampler(
        records,
        local_batch_size=3,
        rank=1,
        world_size=2,
        seed=23,
        num_batches=4,
    )

    zero_batches = list(rank_zero)
    one_batches = list(rank_one)
    zero_episodes = {ref.episode_id for batch in zero_batches for ref in batch}
    one_episodes = {ref.episode_id for batch in one_batches for ref in batch}
    assert zero_episodes | one_episodes == set(range(6))
    assert zero_episodes.isdisjoint(one_episodes)
    assert zero_batches[0] == list(
        BridgeDistributedBatchSampler(
            records,
            local_batch_size=3,
            rank=0,
            world_size=2,
            seed=23,
            num_batches=1,
        )
    )[0]
    assert len({ref.episode_id for ref in zero_batches[0]}) == 1

    original = BridgeDistributedBatchSampler(
        records,
        local_batch_size=3,
        rank=0,
        world_size=2,
        seed=23,
        num_batches=4,
    )
    iterator = iter(original)
    next(iterator)
    state = original.state_dict()
    expected = next(iterator)
    restored = BridgeDistributedBatchSampler(
        records,
        local_batch_size=3,
        rank=0,
        world_size=2,
        seed=23,
        num_batches=4,
    )
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected


def test_bridge_sampler_reshuffles_episode_assignment_each_epoch() -> None:
    records = tuple(
        EpisodeRecord(index, 1, index, f"task {index}") for index in range(8)
    )
    samplers = [
        BridgeDistributedBatchSampler(
            records,
            local_batch_size=4,
            rank=rank,
            world_size=2,
            seed=31,
            num_batches=2,
        )
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]

    for epoch in range(2):
        rank_zero = {ref.episode_id for ref in rank_batches[0][epoch]}
        rank_one = {ref.episode_id for ref in rank_batches[1][epoch]}
        assert rank_zero.isdisjoint(rank_one)
        assert rank_zero | rank_one == set(range(8))
    assert {ref.episode_id for ref in rank_batches[0][0]} != {
        ref.episode_id for ref in rank_batches[0][1]
    }


def test_bridge_sampler_keeps_unequal_length_ranks_disjoint_across_epochs() -> None:
    records = tuple(
        EpisodeRecord(index, length, index, f"task {index}")
        for index, length in enumerate((1, 2, 3, 5, 8, 13, 21, 34))
    )
    samplers = [
        BridgeDistributedBatchSampler(
            records,
            local_batch_size=2,
            rank=rank,
            world_size=2,
            seed=7,
            num_batches=18,
        )
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]

    assert any(
        rank_batches[0][index][0].epoch != rank_batches[0][index - 1][0].epoch
        for index in range(1, len(rank_batches[0]))
    )
    for batch_index in range(18):
        rank_zero = {ref.episode_id for ref in rank_batches[0][batch_index]}
        rank_one = {ref.episode_id for ref in rank_batches[1][batch_index]}
        assert rank_zero.isdisjoint(rank_one)
        assert {ref.epoch for ref in rank_batches[0][batch_index]} == {
            ref.epoch for ref in rank_batches[1][batch_index]
        }


@AV1_FIXTURE_SKIP
def test_bridge_selection_signature_covers_gradient_accumulation(tmp_path: Path) -> None:
    from octo_small_bridge.data import _selection_sha256

    root = tmp_path / "bridge"
    _write_video_bridge_fixture(root)
    adapter = _video_adapter(root)
    normalization_path = root / "normalization.json"
    compute_bridge_v2_statistics(adapter, normalization_path)
    common = {
        "seed": 42,
        "world_size": 4,
        "local_batch_size": 8,
        "primary_size": [256, 256],
        "image_key": "observation.images.image_0",
        "state_keys": ["observation.state"],
        "action_key": "action",
    }

    baseline = _selection_sha256(
        adapter,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=8,
        normalization_path=normalization_path,
        sampling_contract={**common, "gradient_accumulation_steps": 4},
    )
    changed = _selection_sha256(
        adapter,
        dataset_name="bridge_orig_1.0.0",
        action_horizon=8,
        normalization_path=normalization_path,
        sampling_contract={**common, "gradient_accumulation_steps": 2},
    )

    assert changed != baseline


@AV1_FIXTURE_SKIP
def test_bridge_dataset_manifest_records_filtered_source(tmp_path: Path) -> None:
    from octo_small_bridge.training import build_dataset_manifest

    root = tmp_path / "bridge"
    _write_video_bridge_fixture(root)
    adapter = _video_adapter(root)
    dataset = BridgeFrameDataset(
        adapter,
        statistics=_video_statistics(root, adapter),
        dataset_name="bridge_orig_1.0.0",
        action_horizon=8,
        train=False,
    )

    manifest = build_dataset_manifest(
        {
            "data": {
                "dataset_name": "bridge_orig_1.0.0",
                "normalization_contract": "bridge_v2_q99_binary_v1",
            }
        },
        {"dataset": root, "normalization": root / "normalization.json"},
        training_data=SimpleNamespace(
            dataset=dataset,
            selection_sha256="a" * 64,
            normalization_path=root / "normalization.json",
        ),
    )

    assert manifest["dataset"] == "bridge_orig_1.0.0"
    assert manifest["source_episodes"] == 2
    assert manifest["retained_episodes"] == 1
    assert manifest["excluded_empty_task_episodes"] == 1
    assert manifest["retained_frames"] == 3
    assert len(manifest["metadata_sha256"]) == 64
    assert manifest["normalization"]["contract"] == "bridge_v2_q99_binary_v1"
    assert len(manifest["normalization"]["sha256"]) == 64
    assert manifest["selection_sha256"] == "a" * 64


def test_bridge_training_wrapper_selects_primary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import octo_small_libero.training as shared_training
    from octo_small_bridge.training import train

    recorded: dict[str, object] = {}

    def fake_train(config, paths, **kwargs):
        recorded.update(config=config, paths=paths, **kwargs)

    monkeypatch.setattr(shared_training, "train", fake_train)
    config = {"train": {"max_steps": 2}}
    paths = {"dataset": Path("/bridge")}

    train(config, paths, resume="latest")

    assert recorded["config"] is config
    assert recorded["paths"] is paths
    assert recorded["resume"] == "latest"
    assert recorded["observation_tokenizers"] == ("primary",)
    assert callable(recorded["training_data_builder"])
    assert callable(recorded["dataset_manifest_builder"])
    assert callable(recorded["checkpoint_contract_builder"])


def test_bridge_default_config_and_cli_contract(tmp_path: Path) -> None:
    from octo_small_bridge.cli import build_parser, parse_arguments
    from octo_small_bridge.config import apply_overrides, load_config, resolved_paths

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "octo_small_bridge_v2_4x4090.yaml"
    )
    assert config["paths"]["dataset"] == (
        "/data/dwb/datasets/bridge_orig_1.0.0_lerobo"
    )
    assert config["model"]["required_observation_tokenizers"] == ["primary"]
    assert config["data"]["normalization_contract"] == (
        "bridge_v2_q99_binary_v1"
    )
    assert config["data"]["normalization_epsilon"] == pytest.approx(1.0e-6)
    assert config["train"]["gpu_ids"] == [0, 1, 2, 3]
    assert config["train"]["batch_size"] == 128
    assert config["train"]["micro_batch_size_per_gpu"] == 8
    assert config["train"]["gradient_accumulation_steps"] == 4
    assert config["train"]["max_steps"] == 10_000
    assert config["train"]["learning_rate"]["warmup_steps"] == 400
    assert config["train"]["learning_rate"]["peak_value"] == pytest.approx(3.0e-4)
    assert config["data"]["action_horizon"] == 8

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    arguments = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--dataset-path",
            str(tmp_path / "bridge"),
            "--gpu-ids",
            "0,1,2,3",
            "--smoke-test",
        ]
    )
    assert arguments.output_dir == str(tmp_path / "run")
    assert arguments.gpu_ids == [0, 1, 2, 3]
    override_arguments = parse_arguments(
        [
            "--output-dir",
            str(tmp_path / "override-run"),
            "--learning-rate",
            "1e-4",
            "--warmup-steps",
            "800",
        ]
    )
    updated = apply_overrides(
        config,
        output_dir=arguments.output_dir,
        dataset_path=arguments.dataset_path,
        gpu_ids=arguments.gpu_ids,
        max_steps=2,
    )
    paths = resolved_paths(updated)
    assert paths["dataset"] == (tmp_path / "bridge").resolve()
    assert paths["output"] == (tmp_path / "run").resolve()
    assert paths["normalization"] == (
        tmp_path / "run" / "normalization.json"
    ).resolve()
    overridden = apply_overrides(
        config,
        learning_rate=override_arguments.learning_rate,
        warmup_steps=override_arguments.warmup_steps,
    )
    assert overridden["train"]["learning_rate"]["peak_value"] == pytest.approx(1.0e-4)
    assert overridden["train"]["learning_rate"]["warmup_steps"] == 800
    with pytest.raises(ValueError, match="gpu_ids length"):
        apply_overrides(config, gpu_ids=[0, 1])


@pytest.mark.parametrize(
    ("overrides", "expected_peak", "expected_warmup"),
    [
        ({"learning_rate": 2.0e-4}, 2.0e-4, 400),
        ({"warmup_steps": 0}, 3.0e-4, 0),
    ],
)
def test_bridge_learning_rate_overrides_preserve_unspecified_yaml_default(
    overrides: dict[str, float | int],
    expected_peak: float,
    expected_warmup: int,
) -> None:
    from octo_small_bridge.config import apply_overrides, load_config

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "octo_small_bridge_v2_4x4090.yaml"
    )

    updated = apply_overrides(config, **overrides)

    assert updated["train"]["learning_rate"]["peak_value"] == pytest.approx(
        expected_peak
    )
    assert updated["train"]["learning_rate"]["warmup_steps"] == expected_warmup


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--learning-rate", "0"),
        ("--learning-rate", "-1e-4"),
        ("--learning-rate", "nan"),
        ("--learning-rate", "inf"),
        ("--warmup-steps", "-1"),
    ],
)
def test_bridge_cli_rejects_invalid_learning_rate_overrides(
    option: str,
    value: str,
) -> None:
    from octo_small_bridge.cli import parse_arguments

    with pytest.raises(SystemExit):
        parse_arguments(["--output-dir", "outputs/test", option, value])


def test_bridge_cli_main_applies_learning_rate_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from octo_small_bridge import cli
    from octo_small_bridge.config import load_config

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "octo_small_bridge_v2_4x4090.yaml"
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cli,
        "resolved_paths",
        lambda _config: {"output": tmp_path / "run"},
    )
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda resolved, _paths: observed.update(config=resolved),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "octo-small-bridge",
            "--output-dir",
            str(tmp_path / "run"),
            "--learning-rate",
            "1e-4",
            "--warmup-steps",
            "0",
            "--preflight-only",
        ],
    )
    monkeypatch.setenv("RANK", "0")

    cli.main()

    resolved = observed["config"]
    assert isinstance(resolved, dict)
    assert resolved["train"]["learning_rate"]["peak_value"] == pytest.approx(1.0e-4)
    assert resolved["train"]["learning_rate"]["warmup_steps"] == 0


def test_bridge_config_rejects_missing_normalization_contract() -> None:
    from octo_small_bridge.config import load_config, validate_config

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(
        project_root / "configs" / "octo_small_bridge_v2_4x4090.yaml"
    )
    config["data"].pop("normalization_contract")

    with pytest.raises(ValueError, match="normalization_contract"):
        validate_config(config)


@AV1_FIXTURE_SKIP
def test_bridge_preflight_inspects_filtered_av1_dataset(tmp_path: Path) -> None:
    from octo_small_bridge.preflight import inspect_bridge_dataset

    root = tmp_path / "bridge"
    _write_video_bridge_fixture(root)

    adapter = _video_adapter(root)
    report = inspect_bridge_dataset(
        root,
        adapter=adapter,
        statistics=_video_statistics(root, adapter),
    )

    assert report["robot_type"] == "widowx"
    assert report["fps"] == 5
    assert report["source_episodes"] == 2
    assert report["retained_episodes"] == 1
    assert report["retained_frames"] == 3
    assert report["excluded_empty_task_episodes"] == 1
    assert report["image_key"] == "observation.images.image_0"
    assert report["sampled_episodes"] == [0]
