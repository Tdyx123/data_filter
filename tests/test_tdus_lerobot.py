import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from trajectory_data import (
    DatasetValidationError,
    LeRobotDatasetAdapter,
    aligned_chunk_windows,
)


def _png_bytes(frame: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_embedded_image_dataset(
    root: Path,
    image_values: list[dict[str, object]],
    *,
    declared_shape: tuple[int, int, int] = (4, 5, 3),
) -> None:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    length = len(image_values)
    states = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    actions = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
    image_type = pa.struct(
        [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
    )
    table = pa.table(
        {
            "observation.images.image": pa.array(image_values, type=image_type),
            "observation.state": pa.array(
                states.tolist(), type=pa.list_(pa.float32(), list_size=2)
            ),
            "action": pa.array(
                actions.tolist(), type=pa.list_(pa.float32(), list_size=3)
            ),
            "timestamp": pa.array(
                np.arange(length, dtype=np.float32) / 10.0, type=pa.float32()
            ),
            "frame_index": pa.array(np.arange(length), type=pa.int64()),
            "episode_index": pa.array([0] * length, type=pa.int64()),
        }
    )
    pq.write_table(table, data / "episode_000000.parquet")
    info = {
        "codebase_version": "v2.0",
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "observation.images.image": {
                "dtype": "image",
                "shape": list(declared_shape),
            },
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": length}) + "\n",
        encoding="utf-8",
    )


def _adapter(root: Path) -> LeRobotDatasetAdapter:
    return LeRobotDatasetAdapter(
        {
            "path": str(root),
            "use_images": True,
            "feature_keys": {
                "vector_observations": "auto",
                "image_observations": ["observation.images.image"],
            },
        }
    )


def test_lerobot_adapter_discovers_observations_without_robot_assumptions(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    info = {
        "codebase_version": "v2.0",
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [64, 64, 3]},
            "observation.proprio": {"dtype": "float32", "shape": [5]},
            "action": {"dtype": "float32", "shape": [3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 4}) + "\n", encoding="utf-8"
    )
    adapter = LeRobotDatasetAdapter(
        {
            "path": str(tmp_path),
            "use_images": True,
            "feature_keys": {
                "vector_observations": "auto",
                "image_observations": "auto",
            },
        }
    )
    assert adapter.vector_observation_keys == ("observation.proprio",)
    assert adapter.image_observation_keys == ("observation.images.front",)


def test_lerobot_adapter_decodes_embedded_png_observations(tmp_path: Path):
    values = (11, 29, 47)
    frames = [
        np.full((4, 5, 3), value, dtype=np.uint8)
        for value in values
    ]
    _write_embedded_image_dataset(
        tmp_path,
        [{"bytes": _png_bytes(frame), "path": None} for frame in frames],
    )
    adapter = _adapter(tmp_path)

    assert adapter.vector_observation_keys == ("observation.state",)
    assert adapter.image_observation_keys == ("observation.images.image",)
    assert adapter.discovered_image_keys == ("observation.images.image",)

    segments = adapter.iter_segments(
        ["trajectory", "chunk"],
        chunk_length=2,
        stride=1,
        num_workers=0,
    )
    trajectory = next(segments)
    chunk = next(segments)
    decoded = trajectory.observations["observation.images.image"]
    assert decoded.shape == (3, 4, 5, 3)
    assert decoded.dtype == np.uint8
    assert decoded[:, 0, 0, 0].tolist() == list(values)
    np.testing.assert_array_equal(
        chunk.observations["observation.images.image"],
        decoded[:2],
    )


def test_lerobot_adapter_does_not_decode_embedded_png_when_images_are_disabled(
    tmp_path: Path,
):
    _write_embedded_image_dataset(
        tmp_path,
        [{"bytes": b"not-a-png", "path": None}],
    )
    adapter = _adapter(tmp_path)

    segment = next(
        adapter.iter_segments(
            ["trajectory"],
            chunk_length=2,
            stride=1,
            num_workers=0,
            load_images=False,
        )
    )
    assert tuple(segment.observations) == ("observation.state",)
    with pytest.raises(DatasetValidationError, match="could not decode embedded image"):
        next(
            adapter.iter_segments(
                ["trajectory"],
                chunk_length=2,
                stride=1,
                num_workers=0,
                load_images=True,
            )
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_bytes", "embedded image bytes are missing"),
        ("nonempty_path", "embedded image path must be empty"),
        ("wrong_shape", "decoded shape="),
    ],
)
def test_lerobot_adapter_validates_embedded_image_payloads(
    tmp_path: Path,
    case: str,
    message: str,
):
    frame = np.full((4, 5, 3), 17, dtype=np.uint8)
    value: dict[str, object] = {"bytes": _png_bytes(frame), "path": None}
    declared_shape = (4, 5, 3)
    if case == "missing_bytes":
        value["bytes"] = None
    elif case == "nonempty_path":
        value["path"] = "frame.png"
    else:
        declared_shape = (3, 5, 3)
    _write_embedded_image_dataset(
        tmp_path,
        [value],
        declared_shape=declared_shape,
    )

    with pytest.raises(DatasetValidationError, match=message):
        next(
            _adapter(tmp_path).iter_segments(
                ["trajectory"],
                chunk_length=2,
                stride=1,
                num_workers=0,
            )
        )


def test_lerobot_adapter_keeps_video_decode_path(tmp_path: Path, monkeypatch):
    meta = tmp_path / "meta"
    data = tmp_path / "data" / "chunk-000"
    meta.mkdir()
    data.mkdir(parents=True)
    length = 2
    table = pa.table(
        {
            "observation.state": pa.array(
                [[0.0, 1.0], [1.0, 2.0]],
                type=pa.list_(pa.float32(), list_size=2),
            ),
            "action": pa.array(
                [[0.0, 0.0], [1.0, 1.0]],
                type=pa.list_(pa.float32(), list_size=2),
            ),
            "timestamp": pa.array([0.0, 0.1], type=pa.float32()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, data / "episode_000000.parquet")
    info = {
        "codebase_version": "v2.0",
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.image": {"dtype": "video", "shape": [4, 5, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": length}) + "\n",
        encoding="utf-8",
    )
    expected = np.arange(length * 4 * 5 * 3, dtype=np.uint8).reshape(
        length, 4, 5, 3
    )
    decoded_paths: list[Path] = []

    def fake_decode_video(path: Path) -> np.ndarray:
        decoded_paths.append(path)
        return expected

    monkeypatch.setattr("trajectory_data.lerobot._decode_video", fake_decode_video)
    segment = next(
        _adapter(tmp_path).iter_segments(
            ["trajectory"],
            chunk_length=2,
            stride=1,
            num_workers=0,
        )
    )
    np.testing.assert_array_equal(
        segment.observations["observation.images.image"], expected
    )
    assert decoded_paths == [
        tmp_path
        / "videos/chunk-000/observation.images.image/episode_000000.mp4"
    ]


@pytest.mark.real_data
def test_real_libero90_lerobot_smoke():
    root = Path("/data/dwb/datasets/LIBERO_lerobot/libero90")
    if not root.is_dir():
        pytest.skip("real LIBERO-90 LeRobot dataset is not mounted")
    adapter = _adapter(root)
    assert len(adapter.episodes()) == 4500
    assert adapter.vector_observation_keys == ("observation.state",)
    assert adapter.image_observation_keys == ("observation.images.image",)
    assert adapter.discovered_image_keys == (
        "observation.images.image",
        "observation.images.image2",
    )
    assert sum(
        len(aligned_chunk_windows(episode.length, 15, 15))
        for episode in adapter.episodes()
    ) == 46705
    segments = adapter.iter_segments(
        ["trajectory", "chunk"],
        chunk_length=15,
        stride=15,
        num_workers=0,
        max_episodes=1,
    )
    trajectory = next(segments)
    chunk = next(segments)
    assert trajectory.length == adapter.episodes()[0].length
    assert trajectory.observations["observation.images.image"].shape == (
        trajectory.length,
        128,
        128,
        3,
    )
    assert chunk.length == 15
