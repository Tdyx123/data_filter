from pathlib import Path

import pytest

from qwen3_vl_groot.config import (
    ConfigError,
    apply_overrides,
    load_config,
    resolved_paths,
    save_resolved_config,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("lora_learning_rate", 0.0),
        ("lora_learning_rate", float("nan")),
        ("head_learning_rate", -1.0),
        ("head_learning_rate", float("inf")),
    ],
)
def test_config_rejects_non_positive_or_non_finite_optimizer_learning_rates(key, value):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["train"][key] = value

    with pytest.raises(ConfigError, match=key):
        validate_config(config)


def test_libero_config_resolves_independent_lerobot_paths_and_one_to_one_defaults(tmp_path):
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )
    config = apply_overrides(
        config,
        {
            "lerobot_path": str(tmp_path / "lerobot"),
            "output_dir": str(tmp_path / "output"),
        },
    )

    paths = resolved_paths(config)

    assert config["data"]["dataset_type"] == "libero"
    assert config["data"]["target_dataset"] == "libero10_5"
    assert config["data"]["prior_dataset"] == "libero90"
    assert config["data"]["sample_weights"] == [1.0, 1.0]
    assert config["train"]["validation_enabled"] is False
    assert paths["lerobot"] == (tmp_path / "lerobot").resolve()
    assert paths["target_dataset"] == (tmp_path / "lerobot" / "libero10_5").resolve()
    assert paths["prior_dataset"] == (tmp_path / "lerobot" / "libero90").resolve()
    assert paths["output"] == (tmp_path / "output").resolve()


def test_target_only_quota_validation_does_not_apply_mixed_source_weights():
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )
    config["data"]["target_only"] = True
    config["train"]["gpu_count"] = 3
    config["train"]["gpu_ids"] = [0, 1, 2]

    validate_config(config)


def test_libero_config_rejects_conflicting_prior_selection_modes():
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )
    config["data"]["prior_selection"]["relcore_manifest"] = "/tmp/selected.jsonl"

    with pytest.raises(ConfigError, match="prior selection mode"):
        validate_config(config)


def test_libero_config_rejects_non_integer_global_source_quota():
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )
    config["train"]["gpu_count"] = 2
    config["train"]["gpu_ids"] = [0, 1]
    config["data"]["sample_weights"] = [3.0, 1.0]

    with pytest.raises(ConfigError, match="global micro-batch"):
        validate_config(config)


def test_resolved_config_persists_final_independent_learning_rates(tmp_path):
    config = apply_overrides(
        load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
        {
            "lora_learning_rate": 5e-6,
            "action_head_learning_rate": 2e-4,
        },
    )
    output = tmp_path / "run_config.yaml"

    save_resolved_config(config, output)
    reloaded = load_config(output)

    assert reloaded["train"]["lora_learning_rate"] == pytest.approx(5e-6)
    assert reloaded["train"]["head_learning_rate"] == pytest.approx(2e-4)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"lora_freeze_steps": -1}, "lora_freeze_steps"),
        ({"lora_cycle_steps": 100}, "provided together"),
        ({"lora_cycle_steps": 0, "lora_active_steps": 1}, "lora_cycle_steps"),
        ({"lora_cycle_steps": 100, "lora_active_steps": 0}, "lora_active_steps"),
        ({"lora_cycle_steps": 10, "lora_active_steps": 11}, "must not exceed"),
        ({"lora_cycle_steps": True, "lora_active_steps": 1}, "lora_cycle_steps"),
        ({"lora_cycle_steps": 100.0, "lora_active_steps": 10}, "lora_cycle_steps"),
    ],
)
def test_config_rejects_invalid_lora_schedule(updates, message):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["train"].update(updates)

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_lora_schedule_overrides_are_applied():
    config = apply_overrides(
        load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
        {
            "lora_freeze_steps": 5_000,
            "lora_cycle_steps": 100,
            "lora_active_steps": 10,
        },
    )

    assert config["train"]["lora_freeze_steps"] == 5_000
    assert config["train"]["lora_cycle_steps"] == 100
    assert config["train"]["lora_active_steps"] == 10
