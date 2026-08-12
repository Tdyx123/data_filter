import argparse
import json
import os
from pathlib import Path

import pytest

from qwen3_vl_groot.cli import (
    _resolve_config,
    build_parser,
    configure_visible_gpus,
    launch,
    parse_gpu_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_launch_cli_overrides_context_and_independent_compile_targets(tmp_path):
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "launch",
            "--config",
            str(PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"),
            "--output-dir",
            str(tmp_path / "run"),
            "--qwen-context-forward",
            "backbone",
            "--no-compile-qwen-backbone",
            "--compile-action-head",
            "--episode-cache-size",
            "16",
        ]
    )

    config = _resolve_config(arguments)

    assert config["model"]["context_forward"] == "backbone"
    assert config["model"]["torch_compile"]["backbone_enabled"] is False
    assert config["model"]["torch_compile"]["action_head_enabled"] is True
    assert config["data"]["episode_cache_size"] == 16


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
