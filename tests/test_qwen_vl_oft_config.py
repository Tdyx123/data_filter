from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml"


def test_default_oft_config_matches_four_4090_bridge_contract():
    from qwen_vl_oft.config import load_config

    config = load_config(CONFIG)

    assert config["paths"]["model"] == "/data/dwb/models/Qwen3-VL-4B-Instruct"
    assert config["data"]["state_dim"] == 8
    assert config["data"]["action_dim"] == 7
    assert config["data"]["action_horizon"] == 16
    assert config["model"]["attn_implementation"] == "sdpa"
    assert config["model"]["state_bins"] == 256
    assert config["model"]["action_token"] == "🔍"
    assert config["model"]["action_head_hidden_dim"] == 5120
    assert "dit" not in config["model"]
    assert "flow" not in config["model"]
    assert config["train"]["gpu_ids"] == [0, 1, 2, 3]
    assert (
        config["train"]["gpu_count"]
        * config["train"]["micro_batch_size"]
        * config["train"]["gradient_accumulation_steps"]
    ) == 64
    assert config["train"]["lora_freeze_steps"] == 2000


def test_oft_config_overrides_training_and_paths_without_groot_options():
    from qwen_vl_oft.config import apply_overrides, load_config

    config = apply_overrides(
        load_config(CONFIG),
        {
            "output_dir": "outputs/test-oft",
            "gpu_ids": [2, 3],
            "gpu_count": 2,
            "gradient_accumulation_steps": 32,
            "max_steps": 10,
            "lora_rank": 8,
        },
    )

    assert config["paths"]["output"] == "outputs/test-oft"
    assert config["train"]["gpu_ids"] == [2, 3]
    assert config["train"]["max_steps"] == 10
    assert config["model"]["lora"]["rank"] == 8


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data", "state_dim"), 7, "state_dim=8"),
        (("model", "state_bins"), 1, "state_bins"),
        (("model", "action_token"), "", "action_token"),
        (("model", "action_token"), "?", "action_token"),
        (("model", "lora", "rank"), 0, "lora.rank"),
        (("model", "lora", "dropout"), 1.0, "lora.dropout"),
        (("train", "bf16"), False, "bf16"),
    ],
)
def test_oft_config_rejects_incompatible_contract(path, value, message):
    from qwen_vl_oft.config import ConfigError, load_config, validate_config

    config = load_config(CONFIG)
    destination = config
    for part in path[:-1]:
        destination = destination[part]
    destination[path[-1]] = value

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


@pytest.mark.parametrize("action_horizon", [1, 8, 16])
def test_oft_config_accepts_positive_integer_action_horizons(action_horizon):
    from qwen_vl_oft.config import load_config, validate_config

    config = load_config(CONFIG)
    config["data"]["action_horizon"] = action_horizon

    validate_config(config)


@pytest.mark.parametrize("action_horizon", [True, 0, -1, 1.5, 16.0, "16"])
def test_oft_config_rejects_non_positive_or_non_integer_action_horizons(
    action_horizon,
):
    from qwen_vl_oft.config import ConfigError, load_config, validate_config

    config = load_config(CONFIG)
    config["data"]["action_horizon"] = action_horizon

    with pytest.raises(ConfigError, match="positive integer"):
        validate_config(config)
