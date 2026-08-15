from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import re

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
    load_qwen_backbone,
    lora_target_pattern,
    lora_coverage,
    resolve_compile_targets,
    select_attention_implementation,
)
from qwen3_vl_groot.config import ConfigError, load_config  # noqa: E402
from qwen3_vl_groot.inference import BridgePolicy  # noqa: E402
from qwen3_vl_groot.normalization import QuantileStats  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_qwen35_auto_attention_uses_sdpa_when_flash_attention_is_installed(
    monkeypatch,
):
    import qwen3_vl_groot.modeling as modeling

    monkeypatch.setattr(modeling, "_flash_attention_available", lambda: True)

    assert (
        select_attention_implementation("auto", backbone_family="qwen3_5")
        == "sdpa"
    )
    assert (
        select_attention_implementation("auto", backbone_family="qwen3_vl")
        == "flash_attention_2"
    )


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


class FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = FakeLoraLinear()
        self.up_proj = FakeLoraLinear()
        self.down_proj = FakeLoraLinear()


class FakeLayer(nn.Module):
    def __init__(self, *, with_mlp=False):
        super().__init__()
        self.self_attn = FakeAttention()
        if with_mlp:
            self.mlp = FakeMlp()


class FakeLinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = FakeLoraLinear()
        self.in_proj_z = FakeLoraLinear()
        self.in_proj_b = FakeLoraLinear()
        self.in_proj_a = FakeLoraLinear()
        self.out_proj = FakeLoraLinear()


class FakeLinearLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = FakeLinearAttention()


class FakeBackbone(nn.Module):
    def __init__(self, layers=36, *, with_mlp=False):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(
            [FakeLayer(with_mlp=with_mlp) for _ in range(layers)]
        )


class FakeHybridBackbone(nn.Module):
    def __init__(self, layer_types):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(
            FakeLinearLayer() if layer_type == "linear_attention" else FakeLayer()
            for layer_type in layer_types
        )


class FakeContextModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.lora_context = nn.Parameter(torch.arange(hidden_size, dtype=torch.float32))
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        del kwargs
        self.calls += 1
        context = input_ids.float().unsqueeze(-1) + self.lora_context
        return SimpleNamespace(last_hidden_state=context)


class FakeConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeContextModel()
        self.lm_head_calls = 0

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        self.lm_head_calls += 1
        return SimpleNamespace(hidden_states=(outputs.last_hidden_state,))


class FakePeftBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = FakeConditionalGeneration()

    def get_base_model(self):
        return self.base

    def forward(self, **inputs):
        return self.base(**inputs)


class FakeProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def __call__(self, *, text, images, **kwargs):
        del images, kwargs
        batch_size = len(text)
        return {
            "input_ids": torch.tensor([[1, 2]]).repeat(batch_size, 1),
            "attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
        }


def _tiny_policy_config(context_forward="causal_lm"):
    return {
        "data": {"state_dim": 8, "action_dim": 7, "action_horizon": 8},
        "model": {
            "context_dim": 16,
            "max_context_tokens": 32,
            "context_forward": context_forward,
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


def _tiny_context_policy(context_forward):
    return Qwen3VLGrootPolicy(
        backbone=FakePeftBackbone(),
        processor=FakeProcessor(),
        stats=QuantileStats(
            state_q01=np.zeros(8),
            state_q99=np.ones(8),
            action_q01=np.zeros(7),
            action_q99=np.ones(7),
        ),
        config=_tiny_policy_config(context_forward),
    )


QWEN3_VL_LORA_TARGETS = {
    "full_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "linear_attention": (),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def test_all_36_layers_have_all_attention_and_mlp_lora_targets():
    model = FakeBackbone(with_mlp=True)
    coverage = lora_coverage(model)
    assert set(coverage) == set(range(36))
    assert all(
        value
        == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        for value in coverage.values()
    )
    assert_full_lora_coverage(
        model,
        targets_by_layer_type=QWEN3_VL_LORA_TARGETS,
    )
    assert_qwen_freeze_contract(model)


def test_missing_projection_is_rejected():
    model = FakeBackbone(with_mlp=True)
    model.language_model.layers[17].self_attn.o_proj = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="layer 17"):
        assert_full_lora_coverage(model)


def test_missing_mlp_projection_is_rejected():
    model = FakeBackbone(with_mlp=True)
    model.language_model.layers[17].mlp.down_proj = nn.Linear(4, 4)

    with pytest.raises(RuntimeError, match="layer 17"):
        assert_full_lora_coverage(
            model,
            targets_by_layer_type=QWEN3_VL_LORA_TARGETS,
        )


def test_text_mlp_lora_parameters_satisfy_freeze_contract():
    assert_qwen_freeze_contract(FakeBackbone(with_mlp=True))


def test_visual_tower_lora_parameters_are_rejected_by_freeze_contract():
    model = FakeBackbone()
    model.visual = nn.Module()
    model.visual.attn = FakeAttention()

    with pytest.raises(RuntimeError, match="visual"):
        assert_qwen_freeze_contract(model)


def test_qwen35_hybrid_layers_have_family_specific_lora_targets():
    layer_types = tuple(
        layer_type
        for _ in range(6)
        for layer_type in (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        )
    )
    targets = {
        "full_attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "linear_attention": (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ),
    }
    model = FakeHybridBackbone(layer_types)

    coverage = lora_coverage(model)

    assert len(coverage) == 24
    assert coverage[0] == set(targets["linear_attention"])
    assert coverage[3] == set(targets["full_attention"])
    assert_full_lora_coverage(
        model,
        layer_types=layer_types,
        targets_by_layer_type=targets,
    )


def test_qwen35_lora_target_pattern_matches_only_text_token_mixers():
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_5_0_8b_groot_libero_4x4090.yaml"
    )
    pattern = re.compile(lora_target_pattern(config["model"]))

    assert pattern.fullmatch(
        "model.language_model.layers.0.linear_attn.in_proj_qkv"
    )
    assert pattern.fullmatch(
        "base_model.model.model.language_model.layers.3.self_attn.q_proj"
    )
    assert not pattern.fullmatch("model.visual.blocks.0.attn.q_proj")
    assert not pattern.fullmatch("model.language_model.layers.0.mlp.up_proj")


def test_qwen3_vl_lora_target_pattern_matches_text_attention_and_mlp_only():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    pattern = re.compile(lora_target_pattern(config["model"]))

    assert pattern.fullmatch("model.language_model.layers.0.self_attn.q_proj")
    assert pattern.fullmatch("model.language_model.layers.35.self_attn.o_proj")
    assert pattern.fullmatch("model.language_model.layers.35.mlp.gate_proj")
    assert pattern.fullmatch("model.language_model.layers.12.mlp.up_proj")
    assert pattern.fullmatch("model.language_model.layers.7.mlp.down_proj")
    assert not pattern.fullmatch("model.visual.blocks.0.mlp.up_proj")
    assert not pattern.fullmatch("model.language_model.layers.0.input_layernorm")


def test_attention_only_lora_target_is_rejected_before_loading_the_base_model(
    monkeypatch,
):
    class FailIfCalled:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del cls, args, kwargs
            raise AssertionError("model loaders must not run for an invalid LoRA target")

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = object
    fake_peft.TaskType = SimpleNamespace(CAUSAL_LM=object())
    fake_peft.get_peft_model = lambda *args, **kwargs: None
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = FailIfCalled
    fake_transformers.AutoProcessor = FailIfCalled
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["model"]["lora"]["target_modules"].pop("mlp", None)

    with pytest.raises(ConfigError, match="Qwen3-VL-4B requires LoRA targets"):
        load_qwen_backbone("/unused/base-model", config["model"])


def test_direct_context_forward_skips_lm_head_and_preserves_lora_gradients():
    policy = _tiny_context_policy("backbone")
    policy.train()
    policy.set_lora_trainable(True)

    context, attention_mask = policy.encode_context([object()], ["pick up the cup"])
    context.sum().backward()

    conditional = policy.backbone.get_base_model()
    assert conditional.lm_head_calls == 0
    assert conditional.model.calls == 1
    assert conditional.model.lora_context.grad is not None
    assert torch.count_nonzero(conditional.model.lora_context.grad) > 0
    assert attention_mask.dtype == torch.bool


def test_direct_and_legacy_context_paths_return_identical_hidden_states():
    legacy = _tiny_context_policy("causal_lm")
    direct = _tiny_context_policy("backbone")
    direct.load_state_dict(legacy.state_dict())

    legacy_context, legacy_mask = legacy.encode_context([object()], ["pick up the cup"])
    direct_context, direct_mask = direct.encode_context([object()], ["pick up the cup"])

    assert legacy.backbone.get_base_model().lm_head_calls == 1
    assert direct.backbone.get_base_model().lm_head_calls == 0
    torch.testing.assert_close(direct_context, legacy_context)
    torch.testing.assert_close(direct_mask, legacy_mask)


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


def test_compile_policy_modules_can_compile_only_the_action_head():
    policy = type(
        "Policy",
        (),
        {"backbone": CompileRecorder(), "action_head": CompileRecorder()},
    )()
    compile_config = {
        "torch_compile": {
            "enabled": True,
            "backbone_enabled": False,
            "action_head_enabled": True,
            "backend": "inductor",
            "mode": "default",
            "dynamic": True,
            "fullgraph": False,
        }
    }

    compile_policy_modules(policy, compile_config)

    assert resolve_compile_targets(compile_config) == (False, True)
    assert policy.backbone.calls == []
    assert len(policy.action_head.calls) == 1


def test_qwen_libero_default_skips_compile_for_backbone_and_action_head():
    policy = type(
        "Policy",
        (),
        {"backbone": CompileRecorder(), "action_head": CompileRecorder()},
    )()
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )

    compile_policy_modules(policy, config["model"])

    assert policy.backbone.calls == []
    assert policy.action_head.calls == []


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


def test_bridge_policy_forwards_the_inference_generator():
    generator = torch.Generator().manual_seed(31)

    class RecordingPolicy:
        def __init__(self):
            self.generator = None

        def predict_actions(
            self,
            image,
            state,
            instruction,
            denoising_steps=None,
            *,
            generator=None,
        ):
            self.generator = generator
            return torch.zeros(1, 8, 7)

    recording = RecordingPolicy()
    policy = BridgePolicy(recording)

    actions = policy.predict_actions(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros(8, dtype=np.float32),
        "put the book away",
        generator=generator,
    )

    assert actions.shape == (1, 8, 7)
    assert recording.generator is generator


def test_qwen_policy_forwards_the_inference_generator_to_flow(monkeypatch):
    import qwen3_vl_groot.modeling as modeling

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
    policy.encode_context = lambda images, instructions: (
        torch.zeros(1, 2, 16),
        torch.ones(1, 2, dtype=torch.bool),
    )
    recorded = {}

    def fake_denoise(action_head, **kwargs):
        recorded.update(kwargs)
        return torch.zeros(1, 8, 7)

    monkeypatch.setattr(modeling, "euler_denoise", fake_denoise)
    generator = torch.Generator().manual_seed(37)

    actions = policy.predict_actions(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros(8, dtype=np.float32),
        "put the book away",
        generator=generator,
    )

    assert actions.shape == (1, 8, 7)
    assert recorded["generator"] is generator
