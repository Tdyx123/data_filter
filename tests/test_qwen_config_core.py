from pathlib import Path

import pytest

from qwen3_vl_groot.config import (
    ConfigError,
    apply_overrides,
    load_config,
    normalized_lora_target_modules,
    resolved_paths,
    save_resolved_config,
    validate_config,
)
from libero_lerobot.selection import resolve_prior_selection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN35_CONFIG = PROJECT_ROOT / "configs" / "qwen3_5_0_8b_groot_libero_4x4090.yaml"
BRIDGE_V2_NORMALIZATION_CONTRACT = "bridge_v2_q99_binary_v1"


def test_bridge_four_gpu_training_config_uses_16_steps_and_effective_batch_64():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")

    assert config["data"]["action_horizon"] == 16
    assert (
        config["train"]["gpu_count"]
        * config["train"]["micro_batch_size"]
        * config["train"]["gradient_accumulation_steps"]
    ) == 64


@pytest.mark.parametrize("name", ["bridge_4x4090.yaml", "bridge_8x4090.yaml"])
def test_bridge_configs_require_bridge_v2_normalization_contract(name):
    config = load_config(PROJECT_ROOT / "configs" / name)

    assert (
        config["data"]["normalization_contract"]
        == BRIDGE_V2_NORMALIZATION_CONTRACT
    )


def test_bridge_config_rejects_missing_normalization_contract():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["data"].pop("normalization_contract", None)

    with pytest.raises(ConfigError, match="normalization_contract"):
        validate_config(config)


def test_libero_config_does_not_require_bridge_normalization_contract():
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )

    assert "normalization_contract" not in config["data"]
    validate_config(config)


@pytest.mark.parametrize(
    "config_path",
    [
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml",
        QWEN35_CONFIG,
    ],
)
def test_qwen_libero_configs_default_to_full_prior_dataset(config_path):
    config = load_config(config_path)

    assert config["data"]["prior_selection"] == {
        "scores": "/data/dwb/libero90_sqcn/filter/top10pct/scores.csv",
        "top_percent": None,
        "prefiltered": False,
        "relcore_manifest": None,
        "quality_filter_scores": None,
    }
    assert resolve_prior_selection(config, {}) is None


@pytest.mark.parametrize(
    "config_path",
    [
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml",
        QWEN35_CONFIG,
    ],
)
def test_qwen_libero_configs_allow_explicit_prefiltered_override(config_path):
    config = apply_overrides(
        load_config(config_path),
        {"prior_prefiltered_scores": "/data/custom/selected.csv"},
    )

    assert config["data"]["prior_selection"] == {
        "scores": "/data/custom/selected.csv",
        "top_percent": None,
        "prefiltered": True,
        "relcore_manifest": None,
        "quality_filter_scores": None,
    }


@pytest.mark.parametrize(
    "name",
    [
        "bridge_4x4090.yaml",
        "bridge_8x4090.yaml",
        "qwen3_vl_4b_groot_libero_4x4090.yaml",
    ],
)
def test_qwen3_vl_configs_enable_attention_and_mlp_lora(name):
    config = load_config(PROJECT_ROOT / "configs" / name)

    assert normalized_lora_target_modules(config["model"]) == {
        "full_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "linear_attention": (),
        "mlp": ("gate_proj", "up_proj", "down_proj"),
    }


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
    config["data"]["prior_selection"]["prefiltered"] = True
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


def test_qwen35_libero_config_keeps_groot_head_and_effective_batch_64():
    config = load_config(QWEN35_CONFIG)

    assert config["paths"]["model"] == "/data/dwb/models/Qwen3.5-0.8B"
    assert config["model"]["backbone_family"] == "qwen3_5"
    assert config["model"]["text_layers"] == 24
    assert config["model"]["context_dim"] == 1024
    assert "context_forward" not in config["model"]
    assert config["model"]["lora"]["target_modules"] == {
        "full_attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "linear_attention": [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ],
    }
    assert config["model"]["dit"] == {
        "hidden_size": 1024,
        "num_layers": 12,
        "num_heads": 16,
        "mlp_ratio": 4,
        "dropout": 0.2,
    }
    assert config["data"]["action_horizon"] == 8
    effective_batch = (
        config["train"]["gpu_count"]
        * config["train"]["micro_batch_size"]
        * config["train"]["gradient_accumulation_steps"]
    )
    assert effective_batch == 64


def test_qwen35_config_rejects_qwen3_vl_dimensions():
    config = load_config(QWEN35_CONFIG)
    config["model"]["text_layers"] = 36
    config["model"]["context_dim"] = 2560

    with pytest.raises(ConfigError, match="Qwen3.5-0.8B requires"):
        validate_config(config)


def test_qwen3_vl_legacy_flat_lora_targets_are_rejected():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["model"]["lora"]["target_modules"] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    with pytest.raises(ConfigError, match="Qwen3-VL-4B requires LoRA targets"):
        validate_config(config)


def test_qwen3_vl_rejects_missing_mlp_lora_targets():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["model"]["lora"]["target_modules"].pop("mlp", None)

    with pytest.raises(ConfigError, match="Qwen3-VL-4B requires LoRA targets"):
        validate_config(config)


def test_qwen3_vl_rejects_partial_mlp_lora_targets():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["model"]["lora"]["target_modules"]["mlp"] = ["gate_proj"]

    with pytest.raises(ConfigError, match="Qwen3-VL-4B requires LoRA targets"):
        validate_config(config)


def test_qwen35_rejects_mlp_lora_targets():
    config = load_config(QWEN35_CONFIG)
    config["model"]["lora"]["target_modules"]["mlp"] = [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    with pytest.raises(
        ConfigError,
        match="Qwen3.5-0.8B requires LoRA targets",
    ):
        validate_config(config)
