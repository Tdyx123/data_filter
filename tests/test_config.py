from pathlib import Path

import pytest

from qwen3_vl_groot.config import (
    ConfigError,
    apply_overrides,
    load_config,
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
    assert "keep_last_checkpoints" not in config["train"]


@pytest.mark.parametrize("name", ["bridge_4x4090.yaml", "bridge_8x4090.yaml"])
def test_bridge_configs_disable_checkpointing_and_enable_torch_compile(name):
    config = load_config(PROJECT_ROOT / "configs" / name)
    assert config["model"]["gradient_checkpointing"] is False
    assert config["model"]["torch_compile"] == {
        "enabled": True,
        "backend": "inductor",
        "mode": "default",
        "dynamic": True,
        "fullgraph": False,
    }


def test_four_gpu_config_preserves_effective_batch_64():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    assert config["train"]["gpu_count"] == 4
    assert config["train"]["gpu_ids"] == [0, 1, 2, 3]
    effective_batch = (
        config["train"]["gpu_count"]
        * config["train"]["micro_batch_size"]
        * config["train"]["gradient_accumulation_steps"]
    )
    assert effective_batch == 64


def test_gpu_id_override_requires_matching_count():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    updated = apply_overrides(config, {"gpu_ids": [2, 3, 6, 7]})
    assert updated["train"]["gpu_ids"] == [2, 3, 6, 7]
    with pytest.raises(ConfigError, match="length must equal"):
        apply_overrides(config, {"gpu_ids": [0, 1]})


def test_config_rejects_layer_truncation():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    config["model"]["text_layers"] = 12
    with pytest.raises(ConfigError, match="every Qwen text layer"):
        validate_config(config)


def test_config_rejects_invalid_torch_compile_mode():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    config["model"]["torch_compile"]["mode"] = "fastest"
    with pytest.raises(ConfigError, match="torch_compile.mode"):
        validate_config(config)


def test_config_rejects_unknown_context_forward_mode():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    config["model"]["context_forward"] = "hidden_states_and_logits"
    with pytest.raises(ConfigError, match="model.context_forward"):
        validate_config(config)


def test_local_qwen_is_36_layers():
    model_path = Path("/data/dwb/models/Qwen3-VL-4B-Instruct")
    if not model_path.is_dir():
        pytest.skip("Local Qwen checkpoint is not available")
    config = inspect_qwen_config(model_path)
    assert config["text_config"]["num_hidden_layers"] == 36
    assert config["text_config"]["hidden_size"] == 2560
