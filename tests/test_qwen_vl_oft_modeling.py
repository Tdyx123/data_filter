import pytest
import torch
from torch import nn

from qwen_vl_common.normalization import QuantileStats


class _FakeTokenizer:
    padding_side = "right"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [9] if text == "🔍" else [1, 2]}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.messages = None
        self.add_generation_prompt = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        return messages[0]["content"][-1]["text"]

    def __call__(self, *, text, images, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        assert len(text) == len(images)
        return {
            "input_ids": torch.tensor([[1, 2, 9, 9]]).expand(len(text), -1),
            "attention_mask": torch.ones(len(text), 4, dtype=torch.long),
        }


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.ones(()), requires_grad=False)
        self.lora_weight = nn.Parameter(torch.ones(()))

    def forward(self, *, input_ids, attention_mask, **kwargs):
        hidden = torch.arange(
            input_ids.shape[0] * input_ids.shape[1] * 4,
            device=input_ids.device,
            dtype=torch.float32,
        ).reshape(input_ids.shape[0], input_ids.shape[1], 4)
        return type("Outputs", (), {"hidden_states": [hidden * self.lora_weight]})()


class _ConstantHead(nn.Module):
    def forward(self, queries):
        return torch.tensor(
            [[[-1.0, 1.0], [0.0, 0.0]]],
            device=queries.device,
            dtype=queries.dtype,
        ).expand(queries.shape[0], -1, -1)


def _policy_config():
    return {
        "data": {"state_dim": 2, "action_dim": 2, "action_horizon": 2},
        "model": {
            "context_dim": 4,
            "state_bins": 256,
            "action_token": "🔍",
        },
    }


def _stats():
    return QuantileStats(
        state_q01=[0.0, 0.0],
        state_q99=[10.0, 20.0],
        action_q01=[10.0, 20.0],
        action_q99=[20.0, 40.0],
    )


def test_state_bins_and_prompt_match_starvla_oft_contract():
    from qwen_vl_oft.prompting import build_oft_instructions, discretize_state

    state = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])

    assert discretize_state(state, num_bins=256).tolist() == [[0, 64, 128, 192, 255]]
    assert build_oft_instructions(["Move"], state, action_horizon=2) == [
        "Move [STATE] 0 64 128 192 255 [ACTION] "
        "Please predict the next 2 robot actions: <action>🔍🔍<action>."
    ]


@pytest.mark.parametrize("ids", ([1, 2], []))
def test_action_query_token_must_encode_to_exactly_one_id(ids):
    from qwen_vl_oft.prompting import resolve_action_token_id

    def tokenizer(*args, **kwargs):
        return {"input_ids": ids}

    with pytest.raises(ValueError, match="exactly one token"):
        resolve_action_token_id(tokenizer, "🔍")


def test_gather_action_queries_uses_last_tokens_in_temporal_order():
    from qwen_vl_oft.prompting import gather_action_queries

    hidden = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    input_ids = torch.tensor(
        [
            [0, 9, 5, 9, 1, 9, 9],
            [0, 0, 4, 5, 6, 9, 9],
        ]
    )

    gathered = gather_action_queries(hidden, input_ids, action_token_id=9, horizon=2)

    assert torch.equal(gathered[0], hidden[0, [5, 6]])
    assert torch.equal(gathered[1], hidden[1, [5, 6]])


def test_gather_action_queries_rejects_missing_placeholders():
    from qwen_vl_oft.prompting import gather_action_queries

    with pytest.raises(RuntimeError, match=r"samples \[0\]"):
        gather_action_queries(
            torch.zeros(1, 3, 4),
            torch.tensor([[1, 2, 9]]),
            action_token_id=9,
            horizon=2,
        )


def test_starvla_mlp_action_head_predicts_each_query_independently():
    from qwen_vl_oft.action_head import MLPResNetActionHead, MLPResNetBlock

    head = MLPResNetActionHead(input_dim=4, hidden_dim=8, action_dim=3)

    assert sum(isinstance(module, MLPResNetBlock) for module in head.modules()) == 2
    assert head.model[1].in_features == 4
    assert head.model[1].out_features == 8
    assert head.model[-1].out_features == 3
    assert head(torch.randn(2, 5, 4)).shape == (2, 5, 3)


def test_masked_l1_ignores_padded_episode_tail_actions():
    from qwen_vl_oft.action_head import masked_l1_loss

    predictions = torch.zeros(1, 3, 2, requires_grad=True)
    targets = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    loss = masked_l1_loss(predictions, targets, mask)
    loss.backward()

    assert loss.item() == pytest.approx(1.5)
    assert torch.equal(predictions.grad[:, 2], torch.zeros(1, 2))


def test_policy_builds_generation_prompt_and_returns_denormalized_action_chunk():
    from qwen_vl_oft.modeling import QwenVLOFTPolicy

    processor = _FakeProcessor()
    policy = QwenVLOFTPolicy(
        backbone=_FakeBackbone(),
        processor=processor,
        stats=_stats(),
        config=_policy_config(),
    )
    policy.action_head = _ConstantHead()

    actions = policy.predict_actions(
        image=torch.zeros(3, 8, 8),
        state=[5.0, 20.0],
        instruction="Move",
    )

    assert processor.tokenizer.padding_side == "left"
    assert processor.add_generation_prompt is True
    assert processor.messages[0]["content"][-1]["text"].startswith(
        "Move [STATE] 128 255 [ACTION]"
    )
    assert actions.shape == (1, 2, 2)
    assert torch.equal(actions, torch.tensor([[[10.0, 40.0], [15.0, 30.0]]]))


def test_policy_only_toggles_lora_backbone_parameters():
    from qwen_vl_oft.modeling import QwenVLOFTPolicy

    policy = QwenVLOFTPolicy(
        backbone=_FakeBackbone(),
        processor=_FakeProcessor(),
        stats=_stats(),
        config=_policy_config(),
    )

    policy.set_lora_trainable(False)
    assert not policy.backbone.lora_weight.requires_grad
    assert not policy.backbone.base.requires_grad
    policy.set_lora_trainable(True)
    assert policy.backbone.lora_weight.requires_grad
    assert not policy.backbone.base.requires_grad


def test_policy_backward_reaches_only_action_head_and_enabled_lora():
    from qwen_vl_oft.modeling import QwenVLOFTPolicy

    policy = QwenVLOFTPolicy(
        backbone=_FakeBackbone(),
        processor=_FakeProcessor(),
        stats=_stats(),
        config=_policy_config(),
    )
    arguments = {
        "images": [torch.zeros(3, 8, 8)],
        "state": torch.tensor([[5.0, 10.0]]),
        "actions": torch.tensor([[[12.0, 24.0], [14.0, 28.0]]]),
        "action_mask": torch.ones(1, 2),
        "instructions": ["Move"],
    }

    policy.set_lora_trainable(False)
    policy(**arguments).backward()
    assert any(parameter.grad is not None for parameter in policy.action_head_parameters())
    assert policy.backbone.lora_weight.grad is None
    assert policy.backbone.base.grad is None

    policy.zero_grad(set_to_none=True)
    policy.set_lora_trainable(True)
    policy(**arguments).backward()
    assert any(parameter.grad is not None for parameter in policy.action_head_parameters())
    assert policy.backbone.lora_weight.grad is not None
    assert policy.backbone.base.grad is None


def test_policy_predict_actions_preserves_batch_shape():
    from qwen_vl_oft.modeling import QwenVLOFTPolicy

    policy = QwenVLOFTPolicy(
        backbone=_FakeBackbone(),
        processor=_FakeProcessor(),
        stats=_stats(),
        config=_policy_config(),
    )
    policy.action_head = _ConstantHead()

    actions = policy.predict_actions(
        image=[torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)],
        state=[[5.0, 10.0], [5.0, 10.0]],
        instruction=["Move", "Lift"],
    )

    assert actions.shape == (2, 2, 2)


def test_policy_rejects_wrong_state_feature_dimension():
    from qwen_vl_oft.modeling import QwenVLOFTPolicy

    policy = QwenVLOFTPolicy(
        backbone=_FakeBackbone(),
        processor=_FakeProcessor(),
        stats=_stats(),
        config=_policy_config(),
    )

    with pytest.raises(ValueError, match="state feature dimension"):
        policy.predict_actions(torch.zeros(3, 8, 8), [5.0], "Move")
