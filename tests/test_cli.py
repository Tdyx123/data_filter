import argparse
import json
import os
from pathlib import Path

import pytest

from qwen3_vl_groot.cli import (
    _resolve_config,
    build_parser,
    configure_visible_gpus,
    distributed_train,
    launch,
    parse_gpu_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_warm_start_checkpoint(tmp_path, config, *, step=7):
    source_run = tmp_path / "source-run"
    checkpoint = source_run / "checkpoints" / f"step-{step:08d}"
    checkpoint.mkdir(parents=True)
    source_config = json.loads(json.dumps(config))
    source_config.pop("_config_path", None)
    source_config["paths"]["output"] = str(source_run.resolve())
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "normalization.json").write_text("{}", encoding="utf-8")
    (checkpoint / "policy_config.json").write_text(
        json.dumps(
            {
                "format": "qwen3-vl-groot-bridge-compact-v1",
                "base_model": config["paths"]["model"],
                "global_step": step,
                "config": source_config,
                "parameter_names": [],
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_parse_gpu_ids():
    assert parse_gpu_ids("2,3, 6,7") == [2, 3, 6, 7]
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        parse_gpu_ids("0,1,1,2")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        parse_gpu_ids("0,1,-2,3")


def test_configure_visible_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    config = {"train": {"gpu_count": 4, "gpu_ids": [2, 3, 6, 7]}}
    assert configure_visible_gpus(config) == "2,3,6,7"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3,6,7"


def test_launch_cli_rejects_removed_resume_option():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "launch",
                "--config",
                "configs/bridge_4x4090.yaml",
                "--resume",
                "latest",
            ]
        )


def test_warm_start_checkpoint_option_is_available_on_launch_and_train():
    parser = build_parser()

    launch_arguments = parser.parse_args(
        [
            "launch",
            "--config",
            "configs/bridge_4x4090.yaml",
            "--warm-start-checkpoint",
            "/tmp/step-00000007",
        ]
    )
    train_arguments = parser.parse_args(
        [
            "train",
            "--config",
            "outputs/run_config.yaml",
            "--warm-start-checkpoint",
            "/tmp/step-00000007",
        ]
    )

    assert launch_arguments.warm_start_checkpoint == "/tmp/step-00000007"
    assert train_arguments.warm_start_checkpoint == "/tmp/step-00000007"


def test_launch_cli_overrides_lora_and_action_head_learning_rates_independently(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "run"),
            "--lora-learning-rate",
            "5e-6",
            "--action-head-learning-rate",
            "2e-4",
        ]
    )

    config = _resolve_config(arguments)

    assert config["train"]["lora_learning_rate"] == pytest.approx(5e-6)
    assert config["train"]["head_learning_rate"] == pytest.approx(2e-4)


def test_launch_cli_overrides_lora_update_schedule(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "run"),
            "--lora-freeze-steps",
            "5000",
            "--lora-cycle-steps",
            "100",
            "--lora-active-steps",
            "10",
        ]
    )

    config = _resolve_config(arguments)

    assert config["train"]["lora_freeze_steps"] == 5_000
    assert config["train"]["lora_cycle_steps"] == 100
    assert config["train"]["lora_active_steps"] == 10


def test_launch_cli_overrides_independent_compile_targets(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "run"),
            "--no-compile-qwen-backbone",
            "--compile-action-head",
            "--episode-cache-size",
            "16",
        ]
    )

    config = _resolve_config(arguments)

    assert "context_forward" not in config["model"]
    assert config["model"]["torch_compile"]["backbone_enabled"] is False
    assert config["model"]["torch_compile"]["action_head_enabled"] is True
    assert config["data"]["episode_cache_size"] == 16


def test_launch_cli_rejects_removed_context_forward_option(tmp_path):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "launch",
                "--config",
                str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
                "--output-dir",
                str(tmp_path / "run"),
                "--qwen-context-forward",
                "backbone",
            ]
        )


def test_launch_cli_can_disable_default_action_head_compile(tmp_path):
    parser = build_parser()
    default_arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "default-run"),
        ]
    )
    disabled_arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "disabled-run"),
            "--no-compile-action-head",
        ]
    )

    default_config = _resolve_config(default_arguments)
    disabled_config = _resolve_config(disabled_arguments)

    assert default_config["model"]["torch_compile"]["action_head_enabled"] is True
    assert disabled_config["model"]["torch_compile"]["action_head_enabled"] is False


def test_launch_persists_preflight_report_and_can_skip_memory_probe(
    tmp_path,
    monkeypatch,
):
    from qwen3_vl_groot import preflight

    parser = build_parser()
    output = tmp_path / "run"
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--output-dir",
            str(output),
            "--preflight-only",
            "--skip-memory-probe",
        ]
    )
    calls = []

    def fake_preflight(config, *, memory_probe):
        calls.append(memory_probe)
        return {"gpu": {"devices": []}}

    monkeypatch.setattr(preflight, "run_preflight", fake_preflight)

    launch(arguments)

    assert calls == [False]
    assert json.loads((output / "preflight.json").read_text()) == {
        "gpu": {"devices": []}
    }


def test_launch_propagates_warm_start_checkpoint_to_distributed_workers(
    tmp_path,
    monkeypatch,
):
    from qwen3_vl_groot import preflight

    parser = build_parser()
    output = tmp_path / "warm-start-run"
    base_arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(output),
        ]
    )
    checkpoint = _write_warm_start_checkpoint(tmp_path, _resolve_config(base_arguments))
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(output),
            "--warm-start-checkpoint",
            str(checkpoint),
        ]
    )
    commands = []

    monkeypatch.setattr(
        preflight,
        "run_preflight",
        lambda config, *, memory_probe: {"gpu": {"devices": []}},
    )
    monkeypatch.setattr(
        "qwen3_vl_groot.cli.subprocess.run",
        lambda command, *, check, env: commands.append(command),
    )

    launch(arguments)

    assert commands[0][-2:] == [
        "--warm-start-checkpoint",
        str(checkpoint.resolve()),
    ]


def test_launch_rejects_warm_start_into_source_training_output(tmp_path, monkeypatch):
    from qwen3_vl_groot import preflight

    parser = build_parser()
    seed_arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "placeholder-output"),
        ]
    )
    checkpoint = _write_warm_start_checkpoint(tmp_path, _resolve_config(seed_arguments))
    source_run = checkpoint.parents[1]
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(source_run),
            "--warm-start-checkpoint",
            str(checkpoint),
            "--preflight-only",
        ]
    )
    monkeypatch.setattr(
        preflight,
        "run_preflight",
        lambda config, *, memory_probe: {"gpu": {"devices": []}},
    )

    with pytest.raises(ValueError, match="source training output"):
        launch(arguments)


def test_launch_rejects_nonempty_warm_start_output(tmp_path):
    parser = build_parser()
    output = tmp_path / "warm-start-run"
    seed_arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(output),
        ]
    )
    checkpoint = _write_warm_start_checkpoint(tmp_path, _resolve_config(seed_arguments))
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(output),
            "--warm-start-checkpoint",
            str(checkpoint),
            "--preflight-only",
        ]
    )

    with pytest.raises(ValueError, match="must be absent or empty"):
        launch(arguments)


def test_distributed_train_forwards_warm_start_checkpoint(monkeypatch):
    from qwen3_vl_groot import training

    calls = []
    monkeypatch.setattr(
        training,
        "train",
        lambda config, *, warm_start_checkpoint: calls.append(warm_start_checkpoint),
    )
    arguments = argparse.Namespace(
        config=str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
        warm_start_checkpoint="/tmp/step-00000007",
    )

    distributed_train(arguments)

    assert calls == ["/tmp/step-00000007"]


@pytest.mark.parametrize(
    ("option", "changed_key", "changed_value", "unchanged_key", "unchanged_value"),
    [
        (
            "--lora-learning-rate",
            "lora_learning_rate",
            5e-6,
            "head_learning_rate",
            1e-4,
        ),
        (
            "--action-head-learning-rate",
            "head_learning_rate",
            2e-4,
            "lora_learning_rate",
            1e-5,
        ),
    ],
)
def test_each_learning_rate_override_keeps_the_other_yaml_default(
    tmp_path,
    option,
    changed_key,
    changed_value,
    unchanged_key,
    unchanged_value,
):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "run"),
            option,
            str(changed_value),
        ]
    )

    config = _resolve_config(arguments)

    assert config["train"][changed_key] == pytest.approx(changed_value)
    assert config["train"][unchanged_key] == pytest.approx(unchanged_value)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--lora-learning-rate", "0"),
        ("--lora-learning-rate", "nan"),
        ("--action-head-learning-rate", "-1e-4"),
        ("--action-head-learning-rate", "inf"),
    ],
)
def test_launch_cli_rejects_non_positive_or_non_finite_learning_rates(option, value):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "launch",
                "--config",
                "configs/bridge_4x4090.yaml",
                option,
                value,
            ]
        )


def test_libero_cli_applies_all_tasks_target_only_and_lerobot_overrides(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--lerobot-path",
            str(tmp_path / "lerobot"),
            "--output-dir",
            str(tmp_path / "output"),
            "--all-tasks",
            "--target-only",
        ]
    )

    config = _resolve_config(arguments)

    assert config["paths"]["lerobot"] == str((tmp_path / "lerobot").resolve())
    assert config["data"]["target_all_tasks"] is True
    assert config["data"]["target_only"] is True


def test_libero_cli_rejects_target_only_with_sample_weights(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "output"),
            "--all-tasks",
            "--target-only",
            "--sample-weights",
            "1",
            "1",
        ]
    )

    with pytest.raises(ValueError, match="target-only"):
        _resolve_config(arguments)


def test_libero_cli_rejects_target_only_with_prior_override(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "output"),
            "--target-only",
            "--prior-relcore-manifest",
            str(tmp_path / "selection.jsonl"),
        ]
    )

    with pytest.raises(ValueError, match="target-only"):
        _resolve_config(arguments)


def test_libero_cli_requires_explicit_output_directory():
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
        ]
    )

    with pytest.raises(ValueError, match="output-dir"):
        _resolve_config(arguments)
