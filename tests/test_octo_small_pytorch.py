from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from octo_small_libero.data import BalancedDistributedBatchSampler  # noqa: E402
from octo_small_libero.torch_model import OctoSmallConfig, OctoSmallPolicy  # noqa: E402
from octo_small_libero.training import _load_training_state, _save_checkpoint  # noqa: E402


class TinyTextEncoder(torch.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, features)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _tiny_config():
    return OctoSmallConfig(
        hidden_size=32,
        transformer_layers=1,
        attention_heads=4,
        mlp_size=64,
        max_horizon=2,
        language_tokens=4,
        language_features=16,
        vision_features=16,
        stem_features=(8, 8, 16, 16),
        primary_tokens=4,
        wrist_tokens=1,
        diffusion_steps=4,
        diffusion_time_dim=8,
        diffusion_blocks=1,
        diffusion_hidden_size=32,
    )


def _tiny_batch(batch_size=2):
    return {
        "image_primary": torch.rand(batch_size, 1, 3, 32, 32) * 2 - 1,
        "image_wrist": torch.rand(batch_size, 1, 3, 16, 16) * 2 - 1,
        "proprio": torch.randn(batch_size, 1, 8),
        "language_input_ids": torch.randint(0, 64, (batch_size, 4)),
        "language_attention_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "action": torch.randn(batch_size, 8, 7),
        "action_pad_mask": torch.ones(batch_size, 8, 7, dtype=torch.bool),
    }


def test_tiny_pytorch_octo_two_step_cpu_smoke_and_frozen_text_encoder():
    text = TinyTextEncoder(16)
    model = OctoSmallPolicy(text, _tiny_config())
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(_tiny_batch())
        assert output["loss"].shape == ()
        assert torch.isfinite(output["loss"])
        output["loss"].backward()
        assert model.primary_encoder.stem[0].weight.grad is not None
        optimizer.step()
        losses.append(float(output["loss"].detach()))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert not model.text_encoder.training


def test_tiny_pytorch_octo_diffusion_sampling_shape():
    model = OctoSmallPolicy(TinyTextEncoder(16), _tiny_config()).eval()
    actions = model.sample_actions(
        _tiny_batch(),
        generator=torch.Generator().manual_seed(9),
    )
    assert actions.shape == (2, 8, 7)
    assert torch.all(torch.isfinite(actions))


def test_balanced_distributed_sampler_is_reproducible_disjoint_and_resumable():
    samplers = [
        BalancedDistributedBatchSampler(
            (100, 100),
            local_batch_size=8,
            rank=rank,
            world_size=4,
            seed=17,
            num_batches=3,
        )
        for rank in range(4)
    ]
    first_batches = [next(iter(sampler)) for sampler in samplers]
    for batch in first_batches:
        assert [item.source for item in batch].count(0) == 4
        assert [item.source for item in batch].count(1) == 4
    for source in range(2):
        rank_values = [
            {item.frame for item in batch if item.source == source}
            for batch in first_batches
        ]
        assert len(set.union(*rank_values)) == 16

    original = BalancedDistributedBatchSampler(
        (25, 30), local_batch_size=8, seed=5, num_batches=4
    )
    iterator = iter(original)
    next(iterator)
    state = original.state_dict()
    expected = next(iterator)
    restored = BalancedDistributedBatchSampler(
        (25, 30), local_batch_size=8, seed=5, num_batches=4
    )
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected


def test_pytorch_training_checkpoint_round_trip(tmp_path):
    model = OctoSmallPolicy(TinyTextEncoder(16), _tiny_config())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = BalancedDistributedBatchSampler(
        (20, 20), local_batch_size=4, seed=3, num_batches=3
    )
    next(iter(sampler))
    checkpoint = _save_checkpoint(
        output=tmp_path,
        step=7,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config={"train": {"seed": 3}},
    )
    expected = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        next(model.parameters()).add_(1)
    step = _load_training_state(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        device="cpu",
    )
    assert step == 7
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name])
