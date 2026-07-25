import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
av = pytest.importorskip("av")
pytest.importorskip("torchvision")

from qwen3_vl_groot.config import load_config  # noqa: E402
from qwen3_vl_groot.data import BridgeEpisodeDataset, BridgeMetadata  # noqa: E402
from qwen3_vl_groot.flow import (  # noqa: E402
    FlowMatchingActionHead,
    masked_velocity_mse,
    sample_flow_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_av1(path, frames):
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("av1", rate=5)
    except av.error.ValueError:
        container.close()
        pytest.skip("The PyAV build has no AV1 encoder")
    stream.width = 256
    stream.height = 256
    stream.pix_fmt = "yuv420p"
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.mark.slow
def test_synthetic_lerobot_decode_two_step_train_and_round_trip(tmp_path):
    root = tmp_path / "bridge"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    videos = root / "videos" / "chunk-000" / "observation.images.image_0"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    videos.mkdir(parents=True)

    length = 4
    rng = np.random.default_rng(11)
    states = rng.normal(size=(length, 8)).astype(np.float32)
    actions = rng.normal(size=(length, 7)).astype(np.float32)
    frames = rng.integers(0, 256, size=(length, 256, 256, 3), dtype=np.uint8)
    table = pa.table(
        {
            "observation.state": pa.array(
                states.tolist(), type=pa.list_(pa.float32(), list_size=8)
            ),
            "action": pa.array(
                actions.tolist(), type=pa.list_(pa.float32(), list_size=7)
            ),
            "task_index": pa.array([0] * length, type=pa.int64()),
        }
    )
    pq.write_table(table, data / "episode_000000.parquet")
    _write_av1(videos / "episode_000000.mp4", frames)
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": "move the block"}])
    _write_jsonl(
        meta / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["move the block"], "length": length}],
    )
    info = {
        "codebase_version": "v2.0",
        "robot_type": "widowx",
        "total_episodes": 1,
        "total_frames": length,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 5,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.image_0": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "info": {"video.codec": "av1"},
            },
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")

    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    metadata = BridgeMetadata(root, config["data"])
    dataset = BridgeEpisodeDataset(
        metadata,
        metadata.episodes,
        train=False,
        rank=0,
        world_size=1,
        seed=42,
        action_horizon=8,
        video_cache_size=1,
    )
    samples = list(dataset)
    assert len(samples) == length
    assert samples[-1]["actions"].shape == (8, 7)
    assert samples[-1]["action_mask"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    head = FlowMatchingActionHead(
        state_dim=8,
        action_dim=7,
        horizon=8,
        context_dim=16,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1.0e-3)
    state = torch.tensor(np.stack([samples[0]["state"], samples[1]["state"]]))
    target = torch.tensor(np.stack([samples[0]["actions"], samples[1]["actions"]]))
    mask = torch.tensor(np.stack([samples[0]["action_mask"], samples[1]["action_mask"]]))
    context = torch.randn(2, 5, 16)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    for _ in range(2):
        noisy, timestep, velocity = sample_flow_batch(target)
        loss = masked_velocity_mse(
            head(noisy, state, timestep, context, context_mask), velocity, mask
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)

    checkpoint = tmp_path / "head.pt"
    torch.save(head.state_dict(), checkpoint)
    restored = FlowMatchingActionHead(
        state_dim=8,
        action_dim=7,
        horizon=8,
        context_dim=16,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    for left, right in zip(head.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right)

