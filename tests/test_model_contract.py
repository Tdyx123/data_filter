import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from torch import nn  # noqa: E402

from qwen3_vl_groot.modeling import (  # noqa: E402
    _configure_qwen_gradient_checkpointing,
    Qwen3VLGrootPolicy,
    assert_full_lora_coverage,
    assert_qwen_freeze_contract,
    compile_policy_modules,
    lora_coverage,
)
from qwen3_vl_groot.normalization import QuantileStats  # noqa: E402


class FakeLoraLinear(nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = FakeLoraLinear()
        self.k_proj = FakeLoraLinear()
        self.v_proj = FakeLoraLinear()
        self.o_proj = FakeLoraLinear()


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeBackbone(nn.Module):
    def __init__(self, layers=36):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([FakeLayer() for _ in range(layers)])


def test_all_36_layers_have_all_four_lora_targets():
    model = FakeBackbone()
    coverage = lora_coverage(model)
    assert set(coverage) == set(range(36))
    assert all(value == {"q_proj", "k_proj", "v_proj", "o_proj"} for value in coverage.values())
    assert_full_lora_coverage(model)
    assert_qwen_freeze_contract(model)


def test_missing_projection_is_rejected():
    model = FakeBackbone()
    model.language_model.layers[17].self_attn.o_proj = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="layer 17"):
        assert_full_lora_coverage(model)


class CompileRecorder:
    def __init__(self):
        self.calls = []

    def compile(self, **kwargs):
        self.calls.append(kwargs)


def test_compile_policy_modules_compiles_backbone_and_action_head_in_place():
    policy = type(
        "Policy",
        (),
        {"backbone": CompileRecorder(), "action_head": CompileRecorder()},
    )()
    compile_config = {
        "torch_compile": {
            "enabled": True,
            "backend": "inductor",
            "mode": "default",
            "dynamic": True,
            "fullgraph": False,
        }
    }
    compile_policy_modules(policy, compile_config)
    expected = {
        "backend": "inductor",
        "mode": "default",
        "dynamic": True,
        "fullgraph": False,
    }
    assert policy.backbone.calls == [expected]
    assert policy.action_head.calls == [expected]


def test_disabling_qwen_gradient_checkpointing_calls_disable():
    class Backbone:
        def __init__(self):
            self.disabled = False

        def gradient_checkpointing_disable(self):
            self.disabled = True

    backbone = Backbone()
    _configure_qwen_gradient_checkpointing(backbone, enabled=False)
    assert backbone.disabled


def test_policy_passes_disabled_checkpointing_to_action_head():
    config = {
        "data": {"state_dim": 8, "action_dim": 7, "action_horizon": 8},
        "model": {
            "context_dim": 16,
            "max_context_tokens": 32,
            "gradient_checkpointing": False,
            "state_dropout_prob": 0.0,
            "flow": {
                "beta_alpha": 1.5,
                "beta_beta": 1.0,
                "noise_s": 0.999,
                "denoising_steps": 4,
            },
            "dit": {
                "hidden_size": 32,
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2,
                "dropout": 0.0,
            },
        },
    }
    policy = Qwen3VLGrootPolicy(
        backbone=nn.Linear(4, 4),
        processor=None,
        stats=QuantileStats(
            state_q01=np.zeros(8),
            state_q99=np.ones(8),
            action_q01=np.zeros(7),
            action_q99=np.ones(7),
        ),
        config=config,
    )
    assert policy.action_head.gradient_checkpointing is False


def test_module_compile_preserves_state_dict_names():
    module = nn.Linear(4, 4)
    expected = set(module.state_dict())
    module.compile(backend="eager")
    assert set(module.state_dict()) == expected
