from pathlib import Path

import pytest

from qwen3_vl_groot.config import (
    ConfigError,
    load_config,
    resume_config_digest,
    validate_config,
)
from qwen3_vl_groot.modeling import inspect_qwen_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_keeps_all_qwen_layers():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    assert config["model"]["text_layers"] == 36
    assert config["model"]["context_dim"] == 2560
    assert config["model"]["lora"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_config_rejects_layer_truncation():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    config["model"]["text_layers"] = 12
    with pytest.raises(ConfigError, match="every Qwen text layer"):
        validate_config(config)


def test_resume_digest_allows_extending_max_steps_but_not_model_changes():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    original = resume_config_digest(config)
    config["train"]["max_steps"] = 20_001
    config["train"]["save_every_steps"] = 1
    assert resume_config_digest(config) == original
    config["model"]["dit"]["num_layers"] = 11
    assert resume_config_digest(config) != original


def test_local_qwen_is_36_layers():
    model_path = Path("/data/dwb/models/Qwen3-VL-4B-Instruct")
    if not model_path.is_dir():
        pytest.skip("Local Qwen checkpoint is not available")
    config = inspect_qwen_config(model_path)
    assert config["text_config"]["num_hidden_layers"] == 36
    assert config["text_config"]["hidden_size"] == 2560
