from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace
import re

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from torch import nn  # noqa: E402

from qwen3_vl_groot.modeling import (  # noqa: E402
    _configure_qwen_gradient_checkpointing,
    ModelContractError,
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


def _tiny_policy_config():
    return {
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


def _tiny_context_policy():
    return Qwen3VLGrootPolicy(
        backbone=FakePeftBackbone(),
        processor=FakeProcessor(),
        stats=QuantileStats(
            state_q01=np.zeros(8),
            state_q99=np.ones(8),
            action_q01=np.zeros(7),
            action_q99=np.ones(7),
        ),
        config=_tiny_policy_config(),
    )


def _tiny_bridge_v2_policy():
    config = _tiny_policy_config()
    config["data"].update(
        {
            "dataset_type": "bridge",
            "normalization_contract": "bridge_v2_q99_binary_v1",
        }
    )
    return Qwen3VLGrootPolicy(
        backbone=FakePeftBackbone(),
        processor=FakeProcessor(),
        stats=QuantileStats(
            state_q01=np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            state_q99=np.asarray([1, 1, 1, 1, 1, 1, 0, 1], dtype=np.float32),
            action_q01=np.asarray([-1, -1, -1, -1, -1, -1, 0], dtype=np.float32),
            action_q99=np.asarray([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        ),
        config=config,
    )


def test_bridge_v2_normalization_uses_q99_pose_range_and_binary_grippers():
    policy = _tiny_bridge_v2_policy()
    states = torch.tensor(
        [
            [1.6, -0.6, 0.5, 0.5, 0.5, 0.5, 99.0, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -99.0, 0.5001],
        ]
    )
    actions = torch.tensor(
        [
            [2.2, -2.2, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5001],
        ]
    )

    normalized_states = policy.normalize_state(states)
    normalized_actions = policy.normalize_action(actions)

    torch.testing.assert_close(normalized_states[0, :2], torch.tensor([2.2, -2.2]))
    torch.testing.assert_close(normalized_states[:, 6], torch.zeros(2))
    torch.testing.assert_close(normalized_states[:, 7], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(normalized_actions[0, :2], torch.tensor([2.2, -2.2]))
    torch.testing.assert_close(normalized_actions[:, 6], torch.tensor([0.0, 1.0]))


def test_bridge_v2_action_denormalization_extrapolates_pose_and_binarizes_gripper():
    policy = _tiny_bridge_v2_policy()
    normalized_actions = torch.tensor(
        [
            [2.2, -2.2, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5001],
        ]
    )

    actions = policy.denormalize_action(normalized_actions)

    torch.testing.assert_close(actions[0, :2], torch.tensor([2.2, -2.2]))
    torch.testing.assert_close(actions[:, 6], torch.tensor([0.0, 1.0]))


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


def test_backbone_context_encoding_skips_lm_head_and_preserves_lora_gradients():
    policy = _tiny_context_policy()
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


def test_backbone_context_output_requires_tensor_last_hidden_state():
    class MissingLastHiddenState(nn.Module):
        def forward(self, **inputs):
            del inputs
            return SimpleNamespace()

    policy = _tiny_context_policy()
    policy.backbone.get_base_model().model = MissingLastHiddenState()

    with pytest.raises(ModelContractError, match="last_hidden_state"):
        policy.encode_context([object()], ["pick up the cup"])


class CompileRecorder:
    def __init__(self):
        self.calls = []

    def compile(self, **kwargs):
        self.calls.append(kwargs)


class CompilePolicyRecorder:
    def __init__(self):
        self.backbone = CompileRecorder()
        self.action_head = CompileRecorder()
        self.static_context_buckets_enabled = False

    def enable_static_action_head_context_buckets(self):
        self.static_context_buckets_enabled = True


def test_compile_policy_modules_compiles_backbone_and_action_head_in_place():
    policy = CompilePolicyRecorder()
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
    policy = CompilePolicyRecorder()
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
    assert policy.static_context_buckets_enabled is False


def test_qwen_libero_default_compiles_only_the_action_head():
    policy = CompilePolicyRecorder()
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )

    compile_policy_modules(policy, config["model"])

    assert policy.backbone.calls == []
    assert policy.action_head.calls == [
        {
            "backend": "inductor",
            "mode": "default",
            "dynamic": False,
            "fullgraph": False,
        }
    ]
    assert policy.static_context_buckets_enabled is True


class RecordingActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.compile_calls = []
        self.context = None
        self.context_attention_mask = None

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)

    def forward(
        self,
        noisy_actions,
        state,
        timestep,
        context,
        context_attention_mask,
    ):
        del state, timestep
        self.context = context.detach().clone()
        self.context_attention_mask = context_attention_mask.detach().clone()
        return torch.zeros_like(noisy_actions)


def _tiny_action_head_policy(*, compile_action_head=True):
    config = _tiny_policy_config()
    config["model"]["max_context_tokens"] = 512
    config["model"]["torch_compile"] = {
        "enabled": False,
        "backbone_enabled": False,
        "action_head_enabled": compile_action_head,
        "backend": "inductor",
        "mode": "default",
        "dynamic": False,
        "fullgraph": False,
    }
    policy = Qwen3VLGrootPolicy(
        backbone=FakePeftBackbone(),
        processor=FakeProcessor(),
        stats=QuantileStats(
            state_q01=np.zeros(8),
            state_q99=np.ones(8),
            action_q01=np.zeros(7),
            action_q99=np.ones(7),
        ),
        config=config,
    )
    policy.action_head = RecordingActionHead()
    compile_policy_modules(policy, config["model"])
    return policy


@pytest.mark.parametrize(
    ("context_tokens", "expected_bucket"),
    [(74, 96), (90, 96), (97, 192), (193, 384), (385, 512)],
)
def test_static_action_head_compile_pads_context_to_bounded_bucket(
    context_tokens,
    expected_bucket,
):
    policy = _tiny_action_head_policy()
    context = torch.arange(
        context_tokens * 16,
        dtype=torch.float32,
    ).reshape(1, context_tokens, 16)
    context_mask = torch.ones(1, context_tokens, dtype=torch.bool)
    context_mask[:, 1] = False

    policy.flow_loss_from_context(
        context=context,
        context_attention_mask=context_mask,
        state=torch.zeros(1, 8),
        actions=torch.zeros(1, 8, 7),
        action_mask=torch.ones(1, 8),
    )

    recorded = policy.action_head
    assert recorded.context.shape == (1, expected_bucket, 16)
    torch.testing.assert_close(recorded.context[:, :context_tokens], context)
    assert torch.count_nonzero(recorded.context[:, context_tokens:]) == 0
    torch.testing.assert_close(
        recorded.context_attention_mask[:, :context_tokens],
        context_mask,
    )
    assert not recorded.context_attention_mask[:, context_tokens:].any()


def test_uncompiled_action_head_keeps_original_context_shape():
    policy = _tiny_action_head_policy(compile_action_head=False)
    context = torch.randn(1, 74, 16)
    context_mask = torch.ones(1, 74, dtype=torch.bool)

    policy.flow_loss_from_context(
        context=context,
        context_attention_mask=context_mask,
        state=torch.zeros(1, 8),
        actions=torch.zeros(1, 8, 7),
        action_mask=torch.ones(1, 8),
    )

    assert policy.action_head.context.shape == (1, 74, 16)
    assert policy.action_head.context_attention_mask.shape == (1, 74)
    assert policy.action_head.compile_calls == []


@pytest.mark.parametrize(
    ("context_shape", "mask_shape", "message"),
    [
        ((1, 74), (1, 74), "context must have shape"),
        ((1, 74, 16), (1, 74, 1), "context_attention_mask must have shape"),
        ((2, 74, 16), (1, 74), "batch dimensions differ"),
        ((1, 74, 16), (1, 73), "sequence dimensions differ"),
        ((1, 74, 15), (1, 74), "context feature dimension differs"),
        ((1, 513, 16), (1, 513), "exceeds largest supported"),
    ],
)
def test_static_action_head_context_buckets_validate_context_and_mask_shapes(
    context_shape,
    mask_shape,
    message,
):
    policy = _tiny_action_head_policy()

    with pytest.raises(ValueError, match=message):
        policy._bucket_action_head_context(
            torch.zeros(context_shape),
            torch.ones(mask_shape, dtype=torch.bool),
        )


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


def test_bridge_policy_rejects_legacy_checkpoint_before_loading_model(tmp_path):
    (tmp_path / "policy_config.json").write_text(
        json.dumps(
            {
                "format": "qwen3-vl-groot-bridge-compact-v1",
                "base_model": "/unused/base-model",
                "config": {
                    "data": {
                        "dataset_type": "bridge",
                        "state_dim": 8,
                        "action_dim": 7,
                        "action_horizon": 8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="normalization_contract"):
        BridgePolicy.from_pretrained(tmp_path, device="cpu")


def test_bridge_policy_rejects_removed_context_forward_checkpoint(tmp_path):
    (tmp_path / "policy_config.json").write_text(
        json.dumps(
            {
                "format": "qwen3-vl-groot-bridge-compact-v1",
                "base_model": "/unused/base-model",
                "config": {
                    "data": {
                        "dataset_type": "libero",
                        "state_dim": 8,
                        "action_dim": 7,
                        "action_horizon": 8,
                    },
                    "model": {"context_forward": "causal_lm"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context_forward"):
        BridgePolicy.from_pretrained(tmp_path, device="cpu")


def test_compact_policy_loader_keeps_libero_q99_checkpoint_compatible(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "policy_config.json").write_text(
        json.dumps(
            {
                "format": "qwen3-vl-groot-bridge-compact-v1",
                "base_model": "/unused/base-model",
                "config": {
                    "data": {
                        "dataset_type": "libero",
                        "state_dim": 8,
                        "action_dim": 7,
                        "action_horizon": 8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    QuantileStats(
        state_q01=np.zeros(8),
        state_q99=np.ones(8),
        action_q01=np.zeros(7),
        action_q99=np.ones(7),
    ).save(tmp_path / "normalization.json")
    fake_policy = nn.Linear(1, 1)
    monkeypatch.setattr(
        Qwen3VLGrootPolicy,
        "from_local_qwen",
        classmethod(lambda cls, **kwargs: fake_policy),
    )
    monkeypatch.setattr(
        "qwen3_vl_groot.inference.load_compact_weights",
        lambda policy, checkpoint: None,
    )

    loaded = BridgePolicy.from_pretrained(tmp_path, device="cpu")

    assert loaded.policy is fake_policy


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
