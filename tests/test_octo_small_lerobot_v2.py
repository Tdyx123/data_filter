import csv
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from octo_small_libero.checkpoint import load_lerobot_statistics
from octo_small_libero.data import (
    DEFAULT_TARGET_TASK,
    LiberoDataError,
    inspect_hdf5_file,
    target_demo_refs,
    task_to_dataset_name,
)


def _conversion_dependencies_or_skip():
    try:
        import datasets  # noqa: F401
        import h5py
        import pyarrow.parquet as pq
        from PIL import Image
    except Exception as error:
        pytest.skip(f"LeRobot v2 conversion dependencies are not installed: {error}")
    return h5py, pq, Image


def _h5py_or_skip():
    try:
        import h5py
    except Exception as error:
        pytest.skip(f"Compatible h5py is not installed: {error}")
    return h5py


def _write_demo_file(
    path: Path,
    *,
    demos: int = 6,
    length: int = 2,
    action_value: float | None = None,
    language_instruction: str = "pick up the book",
) -> None:
    h5py = _h5py_or_skip()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": language_instruction}
        )
        for index in range(demos):
            demo = data.create_group(f"demo_{index}")
            value = float(action_value if action_value is not None else index * 2)
            actions = np.full((length, 7), value, dtype=np.float64)
            actions[:, -1] = -1
            demo.create_dataset("actions", data=actions)
            observation = demo.create_group("obs")
            primary = np.zeros((length, 128, 128, 3), dtype=np.uint8)
            wrist = np.zeros_like(primary)
            primary[:, 0] = 7
            primary[:, -1] = 19
            wrist[:, 0] = 11
            wrist[:, -1] = 23
            observation.create_dataset("agentview_rgb", data=primary)
            observation.create_dataset("eye_in_hand_rgb", data=wrist)
            observation.create_dataset(
                "ee_pos",
                data=np.full((length, 3), index, dtype=np.float64),
            )
            observation.create_dataset(
                "ee_ori",
                data=np.full((length, 3), index + 1, dtype=np.float64),
            )
            observation.create_dataset(
                "gripper_states",
                data=np.full((length, 2), index / max(demos, 1), dtype=np.float64),
            )


def _build_synthetic_lerobot(tmp_path: Path, *, length: int = 2):
    _conversion_dependencies_or_skip()
    source = tmp_path / "raw"
    output = tmp_path / "lerobot"
    _write_demo_file(
        source / "libero_90" / "TASK_demo.hdf5",
        demos=2,
        length=length,
    )
    _write_demo_file(
        source / "libero_10" / f"{DEFAULT_TARGET_TASK}_demo.hdf5",
        demos=6,
        length=length,
        action_value=20,
    )
    from octo_small_libero.lerobot_builder import prepare_lerobot_v2

    return source, output, prepare_lerobot_v2(source, output)


def _build_synthetic_training_lerobot(tmp_path: Path, *, length: int = 2):
    source, output, report = _build_synthetic_lerobot(tmp_path, length=length)
    from octo_small_libero.libero10_builder import prepare_libero10_lerobot_v2
    from octo_small_libero.libero10_tasks import LIBERO_10_TASKS

    h5py = _h5py_or_skip()
    book_path = source / "libero_10" / f"{DEFAULT_TARGET_TASK}_demo.hdf5"
    with h5py.File(book_path, "a") as handle:
        handle["data"].attrs["problem_info"] = json.dumps(
            {"language_instruction": LIBERO_10_TASKS[5].language_instruction}
        )
    for task_index, task in enumerate(LIBERO_10_TASKS):
        if task_index == 5:
            continue
        _write_demo_file(
            source / "libero_10" / f"{task.name}_demo.hdf5",
            demos=6,
            length=length,
            action_value=float(task_index),
            language_instruction=task.language_instruction,
        )
    prepare_libero10_lerobot_v2(source, output)
    return source, output, report


def test_hdf5_metadata_and_five_demo_target_selection(tmp_path):
    path = tmp_path / "libero_10" / f"{DEFAULT_TARGET_TASK}_demo.hdf5"
    _write_demo_file(path)

    demo_ids, task_name, instruction = inspect_hdf5_file(path)
    references = target_demo_refs(tmp_path)

    assert demo_ids == [f"demo_{index}" for index in range(6)]
    assert task_name == DEFAULT_TARGET_TASK
    assert instruction == "pick up the book"
    assert len(references) == 5
    assert all(reference.language_instruction == instruction for reference in references)


def test_end_to_end_lerobot_v2_conversion_and_png_round_trip(tmp_path):
    _, output, report = _build_synthetic_lerobot(tmp_path)
    _, pq, Image = _conversion_dependencies_or_skip()
    from octo_small_libero.lerobot_v2 import (
        ACTION_KEY,
        PRIMARY_IMAGE_KEY,
        STATE_KEY,
        LeRobotV2Metadata,
        load_episode,
    )
    from octo_small_libero.preflight import _inspect_lerobot_dataset

    target_name = task_to_dataset_name(DEFAULT_TARGET_TASK)
    prior = LeRobotV2Metadata(output / "libero90")
    target = LeRobotV2Metadata(output / target_name)

    assert report["format"] == "LeRobotDataset v2.0"
    assert report["prior_trajectories"] == 2
    assert report["target_trajectories"] == 5
    assert report["prior_transitions"] == 4
    assert report["target_transitions"] == 10
    assert report["target_demo_ids"] == [
        reference.demo_id for reference in target_demo_refs(Path(report["source"]))
    ]
    assert report["source_files"]
    assert (output / "conversion_manifest.json").is_file()

    for metadata, episodes, frames in ((prior, 2, 4), (target, 5, 10)):
        assert metadata.info["codebase_version"] == "v2.0"
        assert metadata.info["robot_type"] == "libero"
        assert metadata.info["fps"] == 10
        assert metadata.info["chunks_size"] == 1000
        assert metadata.info["video_path"] is None
        assert metadata.info["total_videos"] == 0
        assert metadata.info["total_episodes"] == episodes
        assert metadata.info["total_frames"] == frames
        assert len(metadata.stats[ACTION_KEY]["mean"]) == 7
        assert len(metadata.stats[STATE_KEY]["mean"]) == 8
        assert metadata.stats[PRIMARY_IMAGE_KEY]["mean"][0][0][0] >= 0

        episode = load_episode(metadata, metadata.episodes[0])
        assert episode["state"].shape == (2, 8)
        assert episode["action"].shape == (2, 7)
        assert episode["timestamp"].tolist() == pytest.approx([0.0, 0.1])
        table = pq.read_table(metadata.episode_path(0))
        assert table.num_rows == 2
        embedded = table[PRIMARY_IMAGE_KEY].to_pylist()[0]
        assert embedded["path"] is None
        assert embedded["bytes"].startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(BytesIO(episode["image_primary"][0])) as image:
            assert image.format == "PNG"
            primary = np.asarray(image)
        with Image.open(BytesIO(episode["image_wrist"][0])) as image:
            assert image.format == "PNG"
            wrist = np.asarray(image)
        np.testing.assert_array_equal(primary[0], 19)
        np.testing.assert_array_equal(wrist[0], 23)

    inspection = _inspect_lerobot_dataset(output / "libero90")
    assert inspection["episodes"] == 2
    assert inspection["frames"] == 4
    assert inspection["embedded_png_samples_checked"] == 8
    assert inspection["metadata_sha256"] == prior.metadata_sha256()


def test_conversion_overwrite_is_scoped_and_fps_is_strict(tmp_path):
    source, output, _ = _build_synthetic_lerobot(tmp_path)
    from octo_small_libero.lerobot_builder import prepare_lerobot_v2

    unrelated = output / "keep-me.txt"
    unrelated.write_text("user data", encoding="utf-8")
    with pytest.raises(LiberoDataError, match="--overwrite"):
        prepare_lerobot_v2(source, output)
    with pytest.raises(LiberoDataError, match="fps=10"):
        prepare_lerobot_v2(source, tmp_path / "invalid-fps", fps=20)

    rebuilt = prepare_lerobot_v2(source, output, overwrite=True)
    assert rebuilt["prior_trajectories"] == 2
    assert unrelated.read_text(encoding="utf-8") == "user data"


def test_prior_lerobot_statistics_and_training_manifest(tmp_path):
    _, output, _ = _build_synthetic_training_lerobot(tmp_path)
    from octo_small_libero.config import load_config
    from octo_small_libero.training import build_dataset_manifest

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "octo_small_libero_4x4090.yaml")
    config["data"]["target_task_index"] = 5
    target_name = "libero10_5"
    paths = {
        "lerobot": output,
        "prior_dataset": output / "libero90",
        "target_dataset": output / target_name,
        "statistics": output / "libero90" / "meta" / "stats.json",
    }
    statistics = load_lerobot_statistics(paths["prior_dataset"])
    manifest = build_dataset_manifest(config, paths)

    assert statistics["num_trajectories"] == 2
    assert statistics["num_transitions"] == 4
    assert len(statistics["action"]["mean"]) == 7
    assert len(statistics["proprio"]["mean"]) == 8
    assert manifest["lerobot_root"] == str(output)
    assert manifest["datasets"]["libero90"]["episodes"] == 2
    assert manifest["datasets"][target_name]["episodes"] == 50
    assert manifest["datasets"][target_name]["episodes_used"] == 5
    assert manifest["datasets"][target_name]["selection"]["task_index"] == 5
    assert manifest["prior_statistics_sha256"]
    legacy_format = "rl" + "ds"
    assert legacy_format not in json.dumps(manifest).lower()

def test_octo_lerobot_batch_contract_normalization_and_tail_padding(tmp_path):
    torch = pytest.importorskip("torch")

    _, output, _ = _build_synthetic_training_lerobot(tmp_path, length=1)
    from octo_small_libero.config import load_config
    from octo_small_libero.data import make_training_dataset

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "octo_small_libero_4x4090.yaml")
    config["train"]["gpu_count"] = 1
    config["train"]["gpu_ids"] = [0]
    config["train"]["batch_size"] = 8
    config["train"]["micro_batch_size_per_gpu"] = 8
    config["train"]["gradient_accumulation_steps"] = 1
    config["train"]["max_steps"] = 1
    config["train"]["num_workers_per_rank"] = 0
    config["data"]["target_task_index"] = 5
    target_name = "libero10_5"
    paths = {
        "lerobot": output,
        "prior_dataset": output / "libero90",
        "target_dataset": output / target_name,
        "statistics": output / "libero90" / "meta" / "stats.json",
    }
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            del text, kwargs
            return {
                "input_ids": torch.arange(16, dtype=torch.long)[None],
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }

    training_data = make_training_dataset(config, paths, tokenizer=FakeTokenizer())
    batch = next(iter(training_data.dataloader))

    assert training_data.dataset.source_sizes == (5, 2)
    assert batch["action"].shape == (8, 8, 7)
    assert batch["image_primary"].shape == (8, 1, 3, 256, 256)
    assert batch["image_wrist"].shape == (8, 1, 3, 128, 128)
    assert batch["proprio"].shape == (8, 1, 8)
    assert batch["timestep_pad_mask"].shape == (8, 1)
    assert batch["action_pad_mask"].shape == (8, 8, 7)
    assert batch["language_input_ids"].shape == (8, 16)
    assert batch["language_attention_mask"].shape == (8, 16)
    np.testing.assert_allclose(training_data.sample_weights, [0.5, 0.5])
    torch.testing.assert_close(batch["action"][:, 1:, :6], torch.zeros(8, 7, 6))
    torch.testing.assert_close(
        batch["action"][:, 1:, 6],
        batch["action"][:, :1, 6].expand(-1, 7),
    )
    assert torch.all(batch["action_pad_mask"][:, 0])
    assert not torch.any(batch["action_pad_mask"][:, 1:])

    source_names = set(batch["dataset_name"])
    assert source_names == {"libero90", target_name}
    assert batch["dataset_name"].count("libero90") == 4
    assert batch["dataset_name"].count(target_name) == 4
    for index, name in enumerate(batch["dataset_name"]):
        if name == target_name:
            torch.testing.assert_close(
                batch["action"][index, 0, :6],
                torch.full((6,), 19.0),
            )


def test_all_tasks_training_uses_all_target_episodes_and_keeps_one_to_one_batch(
    tmp_path,
):
    torch = pytest.importorskip("torch")

    _, output, _ = _build_synthetic_training_lerobot(tmp_path, length=1)
    from octo_small_libero.config import load_config
    from octo_small_libero.data import make_training_dataset
    from octo_small_libero.training import build_dataset_manifest

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "octo_small_libero_4x4090.yaml")
    config["train"].update(
        {
            "gpu_count": 1,
            "gpu_ids": [0],
            "batch_size": 8,
            "micro_batch_size_per_gpu": 8,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "num_workers_per_rank": 0,
        }
    )
    config["data"]["target_task_index"] = None
    config["data"]["target_all_tasks"] = True
    target_name = "libero10_5"
    paths = {
        "lerobot": output,
        "prior_dataset": output / "libero90",
        "target_dataset": output / target_name,
        "statistics": output / "libero90" / "meta" / "stats.json",
    }

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            del text, kwargs
            return {
                "input_ids": torch.arange(16, dtype=torch.long)[None],
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }

    training_data = make_training_dataset(config, paths, tokenizer=FakeTokenizer())
    batch = next(iter(training_data.dataloader))

    assert training_data.dataset.source_sizes == (50, 2)
    assert training_data.target_selection.selection_mode == "all"
    assert training_data.target_selection.task_indices == tuple(range(10))
    assert training_data.target_selection.episode_indices == tuple(range(50))
    assert training_data.target_selection.episodes == 50
    assert training_data.target_selection.frames == 50
    assert batch["dataset_name"].count(target_name) == 4
    assert batch["dataset_name"].count("libero90") == 4

    manifest = build_dataset_manifest(
        config,
        paths,
        target_selection=training_data.target_selection,
    )
    target_manifest = manifest["datasets"][target_name]
    assert target_manifest["episodes"] == 50
    assert target_manifest["episodes_used"] == 50
    assert target_manifest["frames_used"] == 50
    assert target_manifest["selection"]["mode"] == "all"
    assert target_manifest["selection"]["tasks"] == 10

    config["data"]["sample_weights"] = [3.0, 1.0]
    weighted_training_data = make_training_dataset(
        config,
        paths,
        tokenizer=FakeTokenizer(),
    )
    weighted_batch = next(iter(weighted_training_data.dataloader))
    assert weighted_batch["dataset_name"].count(target_name) == 6
    assert weighted_batch["dataset_name"].count("libero90") == 2
    np.testing.assert_allclose(weighted_training_data.sample_weights, [0.75, 0.25])

    weighted_manifest = build_dataset_manifest(
        config,
        paths,
        target_selection=weighted_training_data.target_selection,
    )
    assert weighted_manifest["datasets"][target_name]["sample_weight"] == 0.75
    assert weighted_manifest["datasets"]["libero90"]["sample_weight"] == 0.25


def test_lerobot_frame_dataset_maps_selected_global_prior_frames(tmp_path):
    pytest.importorskip("torch")
    _, output, _ = _build_synthetic_lerobot(tmp_path, length=10)
    from octo_small_libero.data import LeRobotFrameDataset

    statistics = load_lerobot_statistics(output / "libero90")
    dataset = LeRobotFrameDataset(
        output / "libero90",
        dataset_name="libero90",
        statistics=statistics,
        action_horizon=8,
        frame_indices=(1, 12),
    )

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert (first["episode_index"], first["frame_index"]) == (0, 1)
    assert (second["episode_index"], second["frame_index"]) == (1, 2)
    assert first["action_pad_mask"].all()
    assert second["action_pad_mask"].all()


def test_training_dataset_keeps_target_prior_balance_with_prefiltered_subset(tmp_path):
    torch = pytest.importorskip("torch")
    _, output, _ = _build_synthetic_training_lerobot(tmp_path, length=10)
    from octo_small_libero.config import load_config
    from octo_small_libero.data import make_training_dataset
    from octo_small_libero.training import build_dataset_manifest

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "octo_small_libero_4x4090.yaml")
    config["train"].update(
        {
            "gpu_count": 1,
            "gpu_ids": [0],
            "batch_size": 8,
            "micro_batch_size_per_gpu": 8,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "num_workers_per_rank": 0,
        }
    )
    scores_root = tmp_path / "tdus" / "libero90"
    scores = scores_root / "chunk" / "scores.csv"
    scores.parent.mkdir(parents=True)
    columns = [
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
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for episode_id, tdus in ((1, 0.9),):
            writer.writerow(
                {
                    "sample_id": f"ep{episode_id:06d}_chunk_000000_000009",
                    "episode_id": episode_id,
                    "start_step": 0,
                    "end_step": 9,
                    "length": 10,
                    "quality": 0.5,
                    "coverage": 0.5,
                    "diversity": 0.5,
                    "novelty": 0.5,
                    "tdus": tdus,
                }
            )
    config["data"]["prior_selection"] = {
        "prefiltered_scores": str(scores),
    }
    config["data"]["target_task_index"] = 5
    target_name = "libero10_5"
    paths = {
        "lerobot": output,
        "prior_dataset": output / "libero90",
        "target_dataset": output / target_name,
        "statistics": output / "libero90" / "meta" / "stats.json",
        "prior_prefiltered_scores": scores,
    }

    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            del text, kwargs
            return {
                "input_ids": torch.arange(16, dtype=torch.long)[None],
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }

    training_data = make_training_dataset(config, paths, tokenizer=FakeTokenizer())
    batch = next(iter(training_data.dataloader))

    assert training_data.dataset.source_sizes == (50, 3)
    assert training_data.target_selection.task_index == 5
    assert training_data.target_selection.episode_indices == (25, 26, 27, 28, 29)
    assert training_data.prior_selection.selected_fragments == 1
    assert training_data.prior_selection.training_starts == 3
    assert batch["dataset_name"].count(target_name) == 4
    assert batch["dataset_name"].count("libero90") == 4
    for source, episode_index, frame_index in zip(
        batch["dataset_name"],
        batch["episode_index"].tolist(),
        batch["frame_index"].tolist(),
        strict=True,
    ):
        if source == "libero90":
            assert episode_index == 1
            assert 0 <= frame_index <= 2

    manifest = build_dataset_manifest(
        config,
        paths,
        prior_selection=training_data.prior_selection,
    )
    prior_manifest = manifest["datasets"]["libero90"]
    assert prior_manifest["frames"] == 20
    assert prior_manifest["frames_used"] == 3
    assert prior_manifest["selection"]["mode"] == "prefiltered_fragments"
    assert prior_manifest["selection"]["selected_fragments"] == 1
