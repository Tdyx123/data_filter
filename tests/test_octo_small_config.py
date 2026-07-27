import ast
from pathlib import Path

import pytest

from octo_small_libero.checkpoint import (
    inspect_flax_octo_checkpoint,
)
from octo_small_libero.config import (
    ConfigError,
    apply_overrides,
    load_config,
    resolved_paths,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_octo_config_is_independent_and_points_to_local_checkpoint():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    assert config["paths"]["model"] == "/data/dwb/models/octo-small-pytorch"
    assert config["paths"]["lerobot"] == "/data/dwb/datasets/LIBERO/lerobot"
    assert "statistics" not in config["paths"]
    assert config["model"]["pretrained_step"] == 270000
    assert config["data"]["action_horizon"] == 8
    assert config["data"]["sample_weights"] == [1.0, 1.0]
    assert config["model"]["required_observation_tokenizers"] == ["primary", "wrist"]
    assert config["train"]["gpu_ids"] == [0, 1, 2, 3]
    assert config["train"]["micro_batch_size_per_gpu"] == 8
    assert config["train"]["gradient_accumulation_steps"] == 4
    assert config["train"]["precision"] == "bf16"
    assert "lora" not in config["model"]
    assert "deepspeed_stage" not in config["train"]


def test_octo_config_lerobot_cli_override_and_derived_prior_statistics(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config = apply_overrides(config, lerobot_path=str(tmp_path / "lerobot"))
    paths = resolved_paths(config)

    assert paths["lerobot"] == (tmp_path / "lerobot").resolve()
    assert paths["prior_dataset"] == paths["lerobot"] / "libero90"
    assert paths["target_dataset"] == (paths["lerobot"] / config["data"]["target_dataset"])
    assert paths["statistics"] == (paths["prior_dataset"] / "meta" / "stats.json")
    legacy_format = "rl" + "ds"
    assert legacy_format not in config["paths"]


def test_octo_cli_exposes_only_lerobot_data_override():
    from octo_small_libero.cli import build_parser

    option_strings = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert "--lerobot-path" in option_strings
    legacy_format = "rl" + "ds"
    assert f"--{legacy_format}-path" not in option_strings
    assert "--statistics-path" not in option_strings


def test_octo_config_rejects_removing_wrist_camera():
    config = load_config(PROJECT_ROOT / "configs" / "octo_small_libero_4x4090.yaml")
    config["model"]["required_observation_tokenizers"] = ["primary"]
    with pytest.raises(ConfigError, match="retain both primary and wrist"):
        validate_config(config)


def test_local_flax_octo_small_source_checkpoint_contract():
    path = Path("/data/dwb/models/octo-small")
    if not path.is_dir():
        pytest.skip("Local Octo-small checkpoint is not mounted")
    report = inspect_flax_octo_checkpoint(path, step=270000)
    assert report["transformer_layers"] == 12
    assert report["pretrained_action_horizon"] == 4
    assert {"primary", "wrist"}.issubset(report["observation_tokenizers"])


def test_octo_pytorch_runtime_has_no_legacy_framework_imports():
    runtime = PROJECT_ROOT / "src" / "octo_small_libero"
    conversion_only = {"convert_checkpoint.py"}
    banned = {"tensorflow", "dlimp", "jax", "flax", "optax", "orbax"}
    found = []
    for path in runtime.glob("*.py"):
        if path.name in conversion_only:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            for name in sorted(roots & banned):
                found.append(f"{path.name}:{node.lineno}:{name}")
    assert not found

    requirements = (
        PROJECT_ROOT / "requirements-octo-pytorch.txt"
    ).read_text(encoding="utf-8").lower()
    for package in banned:
        assert package not in requirements
