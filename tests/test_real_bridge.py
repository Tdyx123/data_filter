from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("av")
pytest.importorskip("torch")
pytest.importorskip("torchvision")

from qwen3_vl_groot.config import load_config  # noqa: E402
from qwen3_vl_groot.data import BridgeMetadata, validate_episode  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/data/dwb/datasets/bridge_orig_1.0.0_lerobot")


@pytest.mark.real_data
def test_first_middle_last_episode_parquet_and_av1():
    if not DATASET_ROOT.is_dir():
        pytest.skip("Bridge dataset is not mounted")
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    metadata = BridgeMetadata(DATASET_ROOT, config["data"])
    positions = [0, len(metadata.episodes) // 2, len(metadata.episodes) - 1]
    results = [validate_episode(metadata, metadata.episodes[position]) for position in positions]
    for result in results:
        assert result["decoded_frames"] == result["frames"]
        assert result["state_shape"] == [result["frames"], 8]
        assert result["action_shape"] == [result["frames"], 7]
        assert result["frame_shape"] == [256, 256, 3]

