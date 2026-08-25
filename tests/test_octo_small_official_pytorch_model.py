from types import SimpleNamespace

import numpy as np
import pytest


def test_official_config_matches_released_octo_small_contract():
    from octo_small_official_pytorch.model import OctoOfficialConfig

    config = OctoOfficialConfig()

    assert config.history_horizon == 2
    assert config.action_horizon == 4
    assert config.action_dim == 7
    assert config.use_proprio is False
    assert config.diffusion_steps == 20


def test_block_causal_mask_matches_official_task_observation_readout_rules():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import build_block_causal_attention_mask

    mask = build_block_causal_attention_mask(
        language_mask=torch.tensor([[True, True]]),
        timestep_pad_mask=torch.tensor([[True, True]]),
        observation_tokens_per_timestep=1,
        readout_tokens_per_timestep=1,
    )

    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert mask.shape == (1, 6, 6)
    assert torch.equal(mask[0], expected)


class _TinyTextEncoder:
    def __init__(self, torch_module):
        self.torch = torch_module

    def parameters(self):
        return ()

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        values = input_ids.to(dtype=self.torch.float32).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=values.repeat(1, 1, 8))


def _tiny_config():
    from octo_small_official_pytorch.model import OctoOfficialConfig

    return OctoOfficialConfig(
        hidden_size=8,
        transformer_layers=1,
        attention_heads=2,
        mlp_size=16,
        max_horizon=2,
        language_tokens=2,
        language_features=8,
        vision_features=8,
        stem_features=(4, 4, 4, 4),
        primary_tokens=1,
        wrist_tokens=1,
        diffusion_time_dim=4,
        diffusion_blocks=1,
        diffusion_hidden_size=8,
        diffusion_dropout=0.0,
        diffusion_steps=2,
    )


def test_official_model_accepts_two_images_without_proprio_and_returns_four_actions():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    batch = {
        "image_primary": torch.zeros(1, 2, 3, 16, 16),
        "timestep_pad_mask": torch.ones(1, 2, dtype=torch.bool),
        "language_input_ids": torch.tensor([[3, 1]]),
        "language_attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    actions = model.sample_actions(batch, generator=torch.Generator().manual_seed(7))
    repeated = model.sample_actions(batch, generator=torch.Generator().manual_seed(7))

    assert actions.shape == (1, 4, 7)
    assert torch.isfinite(actions).all()
    assert torch.equal(actions, repeated)


def test_official_model_rejects_proprio_and_history_longer_than_two():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    base = {
        "image_primary": torch.zeros(1, 1, 3, 16, 16),
        "timestep_pad_mask": torch.ones(1, 1, dtype=torch.bool),
        "language_input_ids": torch.tensor([[3, 1]]),
        "language_attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="does not accept proprio"):
        model.encode_observation({**base, "proprio": torch.zeros(1, 1, 8)})

    with pytest.raises(ValueError, match=r"history horizon must be in \[1, 2\]"):
        model.encode_observation(
            {
                **base,
                "image_primary": torch.zeros(1, 3, 3, 16, 16),
                "timestep_pad_mask": torch.ones(1, 3, dtype=torch.bool),
            }
        )


def test_zero_goal_is_appended_independently_to_every_history_frame():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    images = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)], dim=0).unsqueeze(0)

    stacked = OctoSmallOfficialPolicy.with_zero_goal(images)

    assert stacked.shape == (2, 6, 8, 8)
    np.testing.assert_array_equal(stacked[0, :3].numpy(), 0.0)
    np.testing.assert_array_equal(stacked[1, :3].numpy(), 1.0)
    np.testing.assert_array_equal(stacked[:, 3:].numpy(), -1.0)


def test_t5_padding_mask_is_not_reused_as_octo_task_group_padding():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    class CaptureTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mask = None

        def forward(self, inputs, attention_mask):
            self.mask = attention_mask
            return inputs

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    capture = CaptureTransformer()
    model.transformer = capture
    batch = {
        "image_primary": torch.zeros(1, 2, 3, 16, 16),
        "timestep_pad_mask": torch.ones(1, 2, dtype=torch.bool),
        "language_input_ids": torch.tensor([[3, 0]]),
        "language_attention_mask": torch.tensor([[True, False]]),
    }

    model.encode_observation(batch)

    assert capture.mask is not None
    readout_query = 5
    assert capture.mask[0, readout_query, 0]
    assert capture.mask[0, readout_query, 1]


def test_two_frame_positions_are_applied_before_timestep_interleaving():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    class ZeroEncoder(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros(inputs.shape[0], 1, 8)

    class CaptureTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = None

        def forward(self, inputs, attention_mask):
            del attention_mask
            self.inputs = inputs
            return inputs

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    model.primary_encoder = ZeroEncoder()
    capture = CaptureTransformer()
    model.transformer = capture
    with torch.no_grad():
        model.language_projection.weight.zero_()
        model.language_projection.bias.zero_()
        model.language_pos_embedding.zero_()
        model.primary_projection.weight.copy_(torch.eye(8))
        model.primary_projection.bias.zero_()
        model.primary_pos_embedding.zero_()
        model.primary_pos_embedding[:, 0].fill_(1)
        model.primary_pos_embedding[:, 1].fill_(2)
        model.readout_pos_embedding.zero_()
    batch = {
        "image_primary": torch.zeros(1, 2, 3, 16, 16),
        "timestep_pad_mask": torch.ones(1, 2, dtype=torch.bool),
        "language_input_ids": torch.tensor([[3, 0]]),
        "language_attention_mask": torch.tensor([[True, False]]),
    }

    model.encode_observation(batch)

    assert capture.inputs is not None
    torch.testing.assert_close(capture.inputs[0, 2], torch.ones(8))
    torch.testing.assert_close(capture.inputs[0, 4], torch.full((8,), 2.0))


def _tiny_training_batch(torch):
    return {
        "image_primary": torch.zeros(2, 2, 3, 16, 16),
        "timestep_pad_mask": torch.tensor([[False, True], [True, True]]),
        "language_input_ids": torch.tensor([[3, 1], [4, 1]]),
        "language_attention_mask": torch.ones(2, 2, dtype=torch.bool),
        "action": torch.zeros(2, 2, 4, 7),
    }


def test_training_interface_encodes_every_readout_and_samples_only_last_one():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    batch = _tiny_training_batch(torch)

    readouts = model.encode_readouts(batch)
    actions = model.sample_actions(batch, generator=torch.Generator().manual_seed(9))

    assert readouts.shape == (2, 2, 8)
    assert actions.shape == (2, 4, 7)
    torch.testing.assert_close(model.encode_observation(batch), readouts[:, -1])


def test_forward_supervises_both_readouts_and_masks_first_frame_padding():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    class MaskAwareHead(torch.nn.Module):
        def loss(self, readouts, actions, timestep_pad_mask):
            assert readouts.shape == (2, 2, 8)
            assert actions.shape == (2, 2, 4, 7)
            assert torch.equal(timestep_pad_mask, torch.tensor([[False, True], [True, True]]))
            per_readout = actions.square().mean(dim=(-1, -2))
            valid = timestep_pad_mask.to(per_readout.dtype)
            mse = (per_readout * valid).sum() / valid.sum()
            return {"loss": mse, "mse": mse.detach()}

        def sample(self, readout, *, generator=None):
            del generator
            return torch.zeros(readout.shape[0], 4, 7)

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    model.action_head = MaskAwareHead()
    batch = _tiny_training_batch(torch)
    batch["action"][0, 0].fill_(10_000.0)
    batch["action"][0, 1].fill_(1.0)
    batch["action"][1, 0].fill_(2.0)
    batch["action"][1, 1].fill_(3.0)

    result = model(batch)

    assert set(result) == {"loss", "mse"}
    assert result["loss"].item() == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)


def test_real_diffusion_training_loss_is_finite_and_backpropagates():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    batch = _tiny_training_batch(torch)
    batch["action"].normal_()

    result = model(batch)
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["mse"])
    assert model.action_head.reverse_output.weight.grad is not None


def test_primary_only_finetuning_freezes_t5_and_unused_wrist_branch():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    text_encoder = torch.nn.Linear(8, 8)
    model = OctoSmallOfficialPolicy(text_encoder, _tiny_config())

    assert all(not parameter.requires_grad for parameter in model.text_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.wrist_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.wrist_projection.parameters())
    assert model.wrist_pos_embedding.requires_grad is False
    assert all(parameter.requires_grad for parameter in model.primary_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.action_head.parameters())


def test_tiny_official_model_completes_two_cpu_training_steps():
    torch = pytest.importorskip("torch")
    from octo_small_official_pytorch.model import OctoSmallOfficialPolicy

    model = OctoSmallOfficialPolicy(_TinyTextEncoder(torch), _tiny_config())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
        weight_decay=0.01,
    )
    batch = _tiny_training_batch(torch)
    batch["action"].normal_()
    before = model.action_head.reverse_output.weight.detach().clone()
    losses = []

    for step in range(2):
        torch.manual_seed(100 + step)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch)
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))

    assert all(np.isfinite(loss) for loss in losses)
    assert not torch.equal(before, model.action_head.reverse_output.weight)
