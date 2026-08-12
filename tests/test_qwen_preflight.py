from pathlib import Path

import pytest

from qwen3_vl_groot.config import apply_overrides, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = Path("/data/dwb/datasets/LIBERO_lerobot")


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
        lambda _path: {"text_layers": 36},
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
