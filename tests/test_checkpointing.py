import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from torch import nn  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from qwen3_vl_groot.checkpointing import (  # noqa: E402
    CheckpointError,
    inspect_compact_checkpoint,
    load_compact_weights,
    save_compact_checkpoint,
)
from qwen3_vl_groot.config import load_config  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DummyCompactPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Linear(3, 2)
        self.backbone = nn.Module()
        self.backbone.register_parameter("lora_A", nn.Parameter(torch.randn(2, 2)))
        self.backbone.register_parameter("base_weight", nn.Parameter(torch.randn(2, 2)))
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


class DummyEngine:
    def __init__(self, module):
        self.module = module
        self.global_rank = 0


def _save_inspectable_checkpoint(tmp_path, *, global_step=7, max_steps=None):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    if max_steps is not None:
        config["train"]["max_steps"] = max_steps
    checkpoint = save_compact_checkpoint(
        DummyEngine(DummyCompactPolicy()),
        tmp_path / "source-run",
        config=config,
        model_path=config["paths"]["model"],
        global_step=global_step,
        validation_mae=None,
    )
    current_config = json.loads(json.dumps(config))
    current_config["paths"]["output"] = str(tmp_path / "warm-start-run")
    return checkpoint, current_config


def test_compact_safetensors_round_trip(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    policy = DummyCompactPolicy()
    original = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
    }
    target = save_compact_checkpoint(
        DummyEngine(policy),
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=7,
        validation_mae=0.25,
    )
    assert target == tmp_path / "checkpoints" / "step-00000007"
    assert (target / "adapter_model.safetensors").is_file()
    assert (target / "normalization.json").is_file()
    assert (target / "policy_config.json").is_file()
    assert set(load_file(str(target / "adapter_model.safetensors"))) == {
        "action_head.bias",
        "action_head.weight",
        "backbone.lora_A",
    }
    assert json.loads(
        (tmp_path / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    ) == {"checkpoint": "step-00000007", "step": 7}
    checkpoint_config = json.loads(
        (target / "policy_config.json").read_text(encoding="utf-8")
    )["config"]
    assert "context_forward" not in checkpoint_config["model"]
    assert checkpoint_config["model"]["backbone_family"] == "qwen3_vl"
    assert checkpoint_config["model"]["lora"]["target_modules"] == {
        "full_attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "linear_attention": [],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
    }
    assert checkpoint_config["train"]["lora_learning_rate"] == pytest.approx(1e-5)
    assert checkpoint_config["train"]["head_learning_rate"] == pytest.approx(1e-4)

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(10)
    load_compact_weights(policy, target)
    for name in policy.compact_parameter_names():
        value = dict(policy.named_parameters())[name]
        torch.testing.assert_close(value, original[name])
    torch.testing.assert_close(
        policy.backbone.base_weight,
        original["backbone.base_weight"] + 10,
    )
    np.testing.assert_array_equal(policy.action_q99.cpu().numpy(), np.ones(7))


def test_inspect_compact_checkpoint_accepts_only_output_path_change(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)

    inspected = inspect_compact_checkpoint(checkpoint, config=current_config)

    assert inspected.path == checkpoint.resolve()
    assert inspected.global_step == 7
    assert inspected.normalization_path == checkpoint.resolve() / "normalization.json"


def test_inspect_compact_checkpoint_rejects_source_training_output(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    current_config["paths"]["output"] = str(checkpoint.parents[1])

    with pytest.raises(ValueError, match="source training output"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_missing_required_file(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    (checkpoint / "normalization.json").unlink()

    with pytest.raises(CheckpointError, match="normalization.json"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_directory_step_mismatch(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    mismatched = checkpoint.with_name("step-00000008")
    checkpoint.rename(mismatched)

    with pytest.raises(CheckpointError, match="does not match global_step 7"):
        inspect_compact_checkpoint(mismatched, config=current_config)


def test_inspect_compact_checkpoint_rejects_unsupported_format(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    manifest_path = checkpoint / "policy_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "unknown-format"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointError, match="Unsupported compact checkpoint format"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_configuration_change(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    current_config["train"]["gradient_accumulation_steps"] += 1

    with pytest.raises(CheckpointError, match="only paths.output may change"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


@pytest.mark.parametrize("mode", ["causal_lm", "backbone"])
def test_inspect_compact_checkpoint_rejects_legacy_context_forward_config(
    tmp_path,
    mode,
):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    manifest_path = checkpoint / "policy_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["model"]["context_forward"] = mode
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointError, match="only paths.output may change"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_base_model_change(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    manifest_path = checkpoint / "policy_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_model"] = str(tmp_path / "different-base-model")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointError, match="base model"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_requires_later_max_steps(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(
        tmp_path,
        global_step=7,
        max_steps=7,
    )

    with pytest.raises(CheckpointError, match="max_steps must be greater"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_invalid_global_step(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    manifest_path = checkpoint / "policy_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["global_step"] = -1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointError, match="non-negative integer"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_non_object_manifest(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    (checkpoint / "policy_config.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(CheckpointError, match="manifest must be a JSON object"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_non_mapping_config(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    manifest_path = checkpoint / "policy_config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointError, match="config must be a mapping"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_inspect_compact_checkpoint_rejects_invalid_utf8_manifest(tmp_path):
    checkpoint, current_config = _save_inspectable_checkpoint(tmp_path)
    (checkpoint / "policy_config.json").write_bytes(b"\xff\xfe")

    with pytest.raises(CheckpointError, match="Invalid compact checkpoint manifest"):
        inspect_compact_checkpoint(checkpoint, config=current_config)


def test_qwen35_compact_manifest_records_backbone_family(tmp_path):
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_5_0_8b_groot_libero_4x4090.yaml"
    )

    target = save_compact_checkpoint(
        DummyEngine(DummyCompactPolicy()),
        tmp_path,
        config=config,
        model_path="/tmp/qwen35-base-model",
        global_step=11,
        validation_mae=None,
    )

    manifest = json.loads(
        (target / "policy_config.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "qwen3-vl-groot-bridge-compact-v1"
    assert manifest["config"]["model"]["backbone_family"] == "qwen3_5"


def test_latest_and_best_markers_reuse_steps_and_retain_only_references(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_8x4090.yaml")
    engine = DummyEngine(DummyCompactPolicy())
    checkpoints = tmp_path / "checkpoints"

    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=1,
        validation_mae=0.5,
        is_best=True,
    )
    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=2,
        validation_mae=None,
    )
    assert sorted(path.name for path in checkpoints.glob("step-*")) == [
        "step-00000001",
        "step-00000002",
    ]

    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=3,
        validation_mae=None,
    )
    assert sorted(path.name for path in checkpoints.glob("step-*")) == [
        "step-00000001",
        "step-00000003",
    ]
    assert json.loads((checkpoints / "latest.json").read_text(encoding="utf-8")) == {
        "checkpoint": "step-00000003",
        "step": 3,
    }
    assert json.loads((checkpoints / "best.json").read_text(encoding="utf-8")) == {
        "checkpoint": "step-00000001",
        "step": 1,
        "validation_action_mae": 0.5,
    }

    save_compact_checkpoint(
        engine,
        tmp_path,
        config=config,
        model_path="/tmp/base-model",
        global_step=3,
        validation_mae=0.25,
        is_best=True,
    )
    assert [path.name for path in checkpoints.glob("step-*")] == ["step-00000003"]
    assert json.loads((checkpoints / "best.json").read_text(encoding="utf-8")) == {
        "checkpoint": "step-00000003",
        "step": 3,
        "validation_action_mae": 0.25,
    }
