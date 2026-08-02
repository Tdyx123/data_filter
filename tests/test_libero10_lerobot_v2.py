import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from octo_small_libero.data import LiberoDataError, select_target_demo_ids
from octo_small_libero.libero10_tasks import LIBERO_10_TASKS


def _dependencies_or_skip():
    try:
        import datasets  # noqa: F401
        import h5py
        from PIL import Image
    except Exception as error:
        pytest.skip(f"LeRobot conversion dependencies are unavailable: {error}")
    return h5py, Image


def _write_task(
    path: Path,
    *,
    task_index: int,
    language_instruction: str,
    demos: int = 6,
    reverse_creation_order: bool = False,
) -> None:
    h5py, _ = _dependencies_or_skip()
    path.parent.mkdir(parents=True, exist_ok=True)
    order = range(demos - 1, -1, -1) if reverse_creation_order else range(demos)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": language_instruction}
        )
        for demo_index in order:
            demo = data.create_group(f"demo_{demo_index}")
            actions = np.full(
                (2, 7),
                task_index * 10 + demo_index,
                dtype=np.float32,
            )
            actions[:, -1] = -1.0
            demo.create_dataset("actions", data=actions)
            observation = demo.create_group("obs")
            primary = np.zeros((2, 128, 128, 3), dtype=np.uint8)
            wrist = np.zeros_like(primary)
            primary[:, 0] = task_index + 1
            primary[:, -1] = task_index + 101
            wrist[:, 0] = task_index + 2
            wrist[:, -1] = task_index + 102
            observation.create_dataset("agentview_rgb", data=primary)
            observation.create_dataset("eye_in_hand_rgb", data=wrist)
            observation.create_dataset(
                "ee_pos",
                data=np.full((2, 3), demo_index, dtype=np.float32),
            )
            observation.create_dataset(
                "ee_ori",
                data=np.full((2, 3), demo_index + 1, dtype=np.float32),
            )
            observation.create_dataset(
                "gripper_states",
                data=np.full((2, 2), demo_index / 10, dtype=np.float32),
            )


def _write_libero10(source: Path, *, tasks: int = 10, demos: int = 6) -> None:
    for task_index, task in enumerate(LIBERO_10_TASKS[:tasks]):
        _write_task(
            source
            / "libero_10"
            / f"{task.name}_demo.hdf5",
            task_index=task_index,
            language_instruction=task.language_instruction,
            demos=demos,
            reverse_creation_order=bool(task_index % 2),
        )


def test_libero10_selection_is_five_per_task_and_deterministic(tmp_path):
    _write_libero10(tmp_path)
    from octo_small_libero.libero10_builder import (
        collect_libero10_five_demo_refs,
    )

    first_refs, first_tasks = collect_libero10_five_demo_refs(tmp_path)
    second_refs, second_tasks = collect_libero10_five_demo_refs(tmp_path)

    assert first_tasks == second_tasks
    assert [reference.demo_id for reference in first_refs] == [
        reference.demo_id for reference in second_refs
    ]
    assert len(first_tasks) == 10
    assert len(first_refs) == 50
    assert [task["task_name"] for task in first_tasks] == [
        task.name for task in LIBERO_10_TASKS
    ]
    assert first_tasks[5]["task_name"].startswith("STUDY_SCENE1_")
    for task_index, task in enumerate(first_tasks):
        start = task_index * 5
        selected = [reference.demo_id for reference in first_refs[start : start + 5]]
        assert selected == task["demo_ids"]
        assert selected == select_target_demo_ids(
            list(reversed([f"demo_{index}" for index in range(6)])),
            task["task_name"],
        )


def test_libero10_end_to_end_v2_conversion_and_scoped_overwrite(tmp_path):
    _, Image = _dependencies_or_skip()
    source = tmp_path / "raw"
    output = tmp_path / "LIBERO_lerobot"
    _write_libero10(source)

    prior_marker = output / "libero90" / "keep.txt"
    target_marker = output / "existing_target" / "keep.txt"
    root_manifest = output / "conversion_manifest.json"
    prior_marker.parent.mkdir(parents=True)
    target_marker.parent.mkdir(parents=True)
    prior_marker.write_text("prior", encoding="utf-8")
    target_marker.write_text("target", encoding="utf-8")
    root_manifest.write_text('{"keep": true}\n', encoding="utf-8")

    from octo_small_libero.lerobot_v2 import (
        ACTION_KEY,
        PRIMARY_IMAGE_KEY,
        STATE_KEY,
        LeRobotV2Metadata,
        load_episode,
    )
    from octo_small_libero.libero10_builder import (
        LIBERO10_MANIFEST_NAME,
        prepare_libero10_lerobot_v2,
    )
    from octo_small_libero.data import resolve_target_task_selection

    report = prepare_libero10_lerobot_v2(source, output)
    dataset_root = output / "libero10_5"
    metadata = LeRobotV2Metadata(dataset_root)

    assert report["format"] == "LeRobotDataset v2.0"
    assert report["dataset_name"] == "libero10_5"
    assert report["selection_method"] == "task_name_sha256"
    assert report["total_tasks"] == 10
    assert report["total_episodes"] == 50
    assert report["total_frames"] == 100
    assert len(report["tasks"]) == 10
    assert all(task["episodes"] == 5 for task in report["tasks"])
    assert all(task["frames"] == 10 for task in report["tasks"])
    assert all(len(task["demo_ids"]) == 5 for task in report["tasks"])
    assert [task["task_name"] for task in report["tasks"]] == [
        task.name for task in LIBERO_10_TASKS
    ]
    assert metadata.tasks[5] == LIBERO_10_TASKS[5].language_instruction

    assert metadata.info["codebase_version"] == "v2.0"
    assert metadata.info["robot_type"] == "libero"
    assert metadata.info["fps"] == 10
    assert metadata.info["chunks_size"] == 1000
    assert metadata.info["video_path"] is None
    assert metadata.info["total_videos"] == 0
    assert metadata.info["total_tasks"] == 10
    assert metadata.info["total_episodes"] == 50
    assert metadata.info["total_frames"] == 100
    assert len(metadata.stats[ACTION_KEY]["mean"]) == 7
    assert len(metadata.stats[STATE_KEY]["mean"]) == 8
    assert len(metadata.stats[PRIMARY_IMAGE_KEY]["mean"]) == 3
    assert len(list(dataset_root.rglob("*.parquet"))) == 50
    assert not metadata.missing_episode_files()

    expected_global_index = 0
    for episode_index, record in enumerate(metadata.episodes):
        episode = load_episode(metadata, record)
        assert episode["frame_index"].tolist() == [0, 1]
        assert episode["index"].tolist() == [
            expected_global_index,
            expected_global_index + 1,
        ]
        assert episode["task_index"].tolist() == [episode_index // 5] * 2
        expected_global_index += 2
    assert expected_global_index == 100

    selection = resolve_target_task_selection(
        {"data": {"target_dataset": "libero10_5", "target_task_index": 5}},
        {"target_dataset": dataset_root},
    )
    assert selection.task_name == LIBERO_10_TASKS[5].name
    assert selection.episode_indices == (25, 26, 27, 28, 29)
    assert selection.frames == 10

    all_selection = resolve_target_task_selection(
        {
            "data": {
                "target_dataset": "libero10_5",
                "target_task_index": None,
                "target_all_tasks": True,
            }
        },
        {"target_dataset": dataset_root},
    )
    assert all_selection.selection_mode == "all"
    assert all_selection.task_index is None
    assert all_selection.task_indices == tuple(range(10))
    assert all_selection.task_names == tuple(task.name for task in LIBERO_10_TASKS)
    assert all_selection.episode_indices == tuple(range(50))
    assert all_selection.episodes == 50
    assert all_selection.frames == 100
    assert all_selection.as_manifest()["tasks"] == 10

    first_episode = load_episode(metadata, metadata.episodes[0])
    assert first_episode["state"].shape == (2, 8)
    assert first_episode["action"].shape == (2, 7)
    np.testing.assert_array_equal(first_episode["action"][:, 6], 1.0)
    with Image.open(BytesIO(first_episode["image_primary"][0])) as image:
        primary = np.asarray(image)
    with Image.open(BytesIO(first_episode["image_wrist"][0])) as image:
        wrist = np.asarray(image)
    np.testing.assert_array_equal(primary[0], 101)
    np.testing.assert_array_equal(wrist[0], 102)

    manifest = json.loads((output / LIBERO10_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["metadata_sha256"] == metadata.metadata_sha256()
    assert manifest["source_files"] == [
        task["source_file"] for task in manifest["tasks"]
    ]
    assert prior_marker.read_text(encoding="utf-8") == "prior"
    assert target_marker.read_text(encoding="utf-8") == "target"
    assert root_manifest.read_text(encoding="utf-8") == '{"keep": true}\n'

    stale = dataset_root / "stale.txt"
    unrelated = output / "keep-me.txt"
    stale.write_text("remove", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(LiberoDataError, match="--overwrite"):
        prepare_libero10_lerobot_v2(source, output)
    rebuilt = prepare_libero10_lerobot_v2(source, output, overwrite=True)
    assert rebuilt["total_episodes"] == 50
    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert prior_marker.read_text(encoding="utf-8") == "prior"
    assert target_marker.read_text(encoding="utf-8") == "target"
    assert root_manifest.read_text(encoding="utf-8") == '{"keep": true}\n'


def test_target_selection_rejects_legacy_alphabetical_task_mapping(tmp_path):
    _dependencies_or_skip()
    source = tmp_path / "raw"
    output = tmp_path / "LIBERO_lerobot"
    _write_libero10(source)

    from octo_small_libero.data import resolve_target_task_selection
    from octo_small_libero.libero10_builder import prepare_libero10_lerobot_v2

    prepare_libero10_lerobot_v2(source, output)
    alphabetical = [
        task.language_instruction for task in sorted(LIBERO_10_TASKS, key=lambda task: task.name)
    ]
    tasks_path = output / "libero10_5" / "meta" / "tasks.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps({"task_index": index, "task": instruction}) + "\n"
            for index, instruction in enumerate(alphabetical)
        ),
        encoding="utf-8",
    )

    with pytest.raises(LiberoDataError, match="evaluation order"):
        resolve_target_task_selection(
            {"data": {"target_dataset": "libero10_5", "target_task_index": 5}},
            {"target_dataset": output / "libero10_5"},
        )


@pytest.mark.parametrize(
    ("task_count", "demo_count", "message"),
    [
        (9, 6, "exactly 10"),
        (10, 4, "5 are required"),
    ],
)
def test_libero10_rejects_incomplete_sources(
    tmp_path,
    task_count,
    demo_count,
    message,
):
    source = tmp_path / "raw"
    output = tmp_path / "output"
    _write_libero10(source, tasks=task_count, demos=demo_count)
    from octo_small_libero.libero10_builder import prepare_libero10_lerobot_v2

    with pytest.raises(LiberoDataError, match=message):
        prepare_libero10_lerobot_v2(source, output)
    assert not output.exists()


def test_libero10_rejects_unreadable_selected_trajectory_without_partial_output(
    tmp_path,
):
    h5py, _ = _dependencies_or_skip()
    source = tmp_path / "raw"
    output = tmp_path / "output"
    _write_libero10(source)
    first = sorted((source / "libero_10").glob("*.hdf5"))[0]
    with h5py.File(first, "a") as handle:
        for demo in handle["data"].values():
            del demo["obs"]["ee_pos"]

    from octo_small_libero.libero10_builder import prepare_libero10_lerobot_v2

    with pytest.raises(LiberoDataError, match="missing"):
        prepare_libero10_lerobot_v2(source, output)
    assert not output.exists()
