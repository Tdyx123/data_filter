import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qwen_vl_common.normalization import QuantileStats
from qwen_vl_oft.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Linear(3, 2)
        self.backbone = nn.Module()
        self.backbone.register_parameter("lora_A", nn.Parameter(torch.randn(2, 2)))
        self.backbone.register_parameter("base", nn.Parameter(torch.randn(2, 2)))
        self.register_buffer("state_q01", torch.zeros(8))
        self.register_buffer("state_q99", torch.ones(8))
        self.register_buffer("action_q01", torch.zeros(7))
        self.register_buffer("action_q99", torch.ones(7))
        self.normalization_epsilon = 1.0e-6

    def compact_parameter_names(self):
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith("action_head.") or "lora_" in name
        ]


class _Engine:
    global_rank = 0

    def __init__(self, module):
        self.module = module


def test_oft_compact_checkpoint_round_trip_and_format_isolation(tmp_path):
    from qwen_vl_oft.checkpointing import (
        CheckpointError,
        inspect_compact_checkpoint,
        load_compact_weights,
        save_compact_checkpoint,
    )

    config = load_config(PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml")
    policy = _Policy()
    original = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    checkpoint = save_compact_checkpoint(
        _Engine(policy),
        tmp_path / "run",
        config=config,
        model_path="/tmp/qwen",
        global_step=7,
        validation_mae=0.25,
        is_best=True,
    )

    manifest_path = checkpoint / "policy_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "qwen-vl-oft-bridge-compact-v1"
    assert set(manifest["parameter_names"]) == {
        "action_head.bias",
        "action_head.weight",
        "backbone.lora_A",
    }
    restored_stats = QuantileStats.load(checkpoint / "normalization.json")
    np.testing.assert_array_equal(restored_stats.state_q01, np.zeros(8))
    np.testing.assert_array_equal(restored_stats.action_q99, np.ones(7))
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(10)
    load_compact_weights(policy, checkpoint)
    for name in policy.compact_parameter_names():
        torch.testing.assert_close(dict(policy.named_parameters())[name], original[name])
    torch.testing.assert_close(policy.backbone.base, original["backbone.base"] + 10)

    manifest["format"] = "qwen3-vl-groot-bridge-compact-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    current = json.loads(json.dumps(config))
    current["paths"]["output"] = str(tmp_path / "warm-start")
    try:
        inspect_compact_checkpoint(checkpoint, config=current)
    except CheckpointError as error:
        assert "Unsupported compact checkpoint format" in str(error)
    else:
        raise AssertionError("GROOT checkpoint format was accepted by OFT")


def test_oft_warm_start_accepts_only_a_new_output_and_later_step(tmp_path):
    from qwen_vl_oft.checkpointing import inspect_compact_checkpoint, save_compact_checkpoint

    config = load_config(PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml")
    source = json.loads(json.dumps(config))
    source["paths"]["output"] = str(tmp_path / "source")
    checkpoint = save_compact_checkpoint(
        _Engine(_Policy()),
        source["paths"]["output"],
        config=source,
        model_path=source["paths"]["model"],
        global_step=7,
        validation_mae=None,
    )
    resumed = json.loads(json.dumps(source))
    resumed["paths"]["output"] = str(tmp_path / "resumed")
    resumed["train"]["max_steps"] = source["train"]["max_steps"] + 100

    inspected = inspect_compact_checkpoint(checkpoint, config=resumed)

    assert inspected.global_step == 7
    assert inspected.path == checkpoint.resolve()
