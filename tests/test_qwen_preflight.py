from pathlib import Path

import pytest

from qwen3_vl_groot.config import apply_overrides, load_config
from qwen3_vl_groot.preflight import (
    PreflightError,
    _build_memory_probe_batch,
    _memory_probe_model_description,
    _memory_probe_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = Path("/data/dwb/datasets/LIBERO_lerobot")


class RecordingDataset:
    def __init__(self):
        self.requested = []

    def __len__(self):
        return 16

    def __getitem__(self, index):
        import numpy as np

        self.requested.append(index)
        return {
            "image": object(),
            "state": np.full(8, index, dtype=np.float32),
            "actions": np.full((8, 7), index, dtype=np.float32),
            "action_mask": np.ones(8, dtype=np.float32),
            "instruction": f"instruction {index}",
            "episode_index": index,
            "frame_index": index,
        }


def test_memory_probe_batch_uses_real_micro_batch_with_distinct_samples():
    dataset = RecordingDataset()

    batch = _build_memory_probe_batch(dataset, micro_batch_size=4)

    assert dataset.requested == [0, 1, 2, 3]
    assert len(batch["images"]) == 4
    assert tuple(batch["state"].shape) == (4, 8)
    assert batch["episode_index"].tolist() == [0, 1, 2, 3]


def test_memory_probe_rejects_reserved_memory_above_22_gib():
    gib = 2**30
    with pytest.raises(PreflightError, match="22.0 GiB"):
        _memory_probe_result(
            loss=1.0,
            peak_allocated=20 * gib,
            peak_reserved=22 * gib + 1,
            total=24 * gib,
            micro_batch_size=4,
        )


def test_memory_probe_accepts_reserved_memory_at_22_gib():
    gib = 2**30
    result = _memory_probe_result(
        loss=1.0,
        peak_allocated=20 * gib,
        peak_reserved=22 * gib,
        total=24 * gib,
        micro_batch_size=4,
    )

    assert result["micro_batch_size"] == 4
    assert result["peak_reserved_gib"] == pytest.approx(22.0)
    assert result["reserved_limit_gib"] == pytest.approx(22.0)
    assert result["reserved_headroom_gib"] == pytest.approx(0.0)


def test_qwen35_memory_probe_description_uses_declared_backbone_contract():
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_5_0_8b_groot_libero_4x4090.yaml"
    )

    assert _memory_probe_model_description(config) == (
        "24-layer Qwen3.5-0.8B + 12-layer GROOT DiT"
    )


def test_libero_target_only_preflight_does_not_access_prior(tmp_path, monkeypatch):
    if not (LIBERO_ROOT / "libero10_5").is_dir():
        pytest.skip("LIBERO-10 LeRobot data is not mounted")
    from qwen3_vl_groot import preflight

    config = apply_overrides(
        load_config(PROJECT_ROOT / "configs/qwen3_vl_4b_groot_libero_4x4090.yaml"),
        {
            "lerobot_path": str(tmp_path / "lerobot"),
            "output_dir": str(tmp_path / "output"),
            "target_only": True,
        },
    )
    (tmp_path / "lerobot").mkdir()
    (tmp_path / "lerobot" / "libero10_5").symlink_to(
        LIBERO_ROOT / "libero10_5",
        target_is_directory=True,
    )
    monkeypatch.setattr(
        preflight,
        "_inspect_qwen_config",
        lambda _path, expected_family=None: {
            "text_layers": 36,
            "backbone_family": expected_family,
        },
    )

    report = preflight.validate_paths_and_data(config, decode_samples=False)

    assert report["dataset_type"] == "libero"
    assert report["action_window_policy"] == "episode_tail_repeat_last_action"
    assert report["target"]["selection"]["episodes"] == 50
    assert report["target"]["selection"]["action_window_policy"] == (
        "episode_tail_repeat_last_action"
    )
    assert report["target"]["selection"]["frames"] == 14_144
    assert report["prior"] is None
    assert report["global_micro_batch_source_counts"] == [4]
    assert report["offline_validation"] is False
