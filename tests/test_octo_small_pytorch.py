import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from octo_small_libero.data import (  # noqa: E402
    BalancedDistributedBatchSampler,
    training_selection_sha256,
)
from octo_small_libero.torch_model import OctoSmallConfig, OctoSmallPolicy  # noqa: E402
from octo_small_libero.training import (  # noqa: E402
    _learning_rate_lambda,
    _load_training_state,
    _resolve_resume,
    _save_checkpoint,
    _should_save_checkpoint,
)


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


def _step_checkpoint_directories(checkpoints):
    return sorted(
        path.name
        for path in checkpoints.iterdir()
        if (
            path.is_dir()
            and path.name.startswith("step-")
            and len(path.name) == len("step-00000000")
            and path.name.removeprefix("step-").isdigit()
        )
    )


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


def test_weighted_distributed_sampler_uses_three_to_one_disjoint_batches():
    samplers = [
        BalancedDistributedBatchSampler(
            (100, 100),
            local_batch_size=8,
            sample_weights=(3.0, 1.0),
            rank=rank,
            world_size=4,
            seed=17,
            num_batches=1,
        )
        for rank in range(4)
    ]
    batches = [next(iter(sampler)) for sampler in samplers]
    for batch in batches:
        assert [item.source for item in batch].count(0) == 6
        assert [item.source for item in batch].count(1) == 2
    for source, expected in ((0, 24), (1, 8)):
        rank_values = [
            {item.frame for item in batch if item.source == source}
            for batch in batches
        ]
        assert len(set.union(*rank_values)) == expected


def test_single_source_distributed_sampler_is_disjoint_and_resumable():
    samplers = [
        BalancedDistributedBatchSampler(
            (100,),
            local_batch_size=8,
            sample_weights=(1.0,),
            rank=rank,
            world_size=4,
            seed=17,
            num_batches=2,
        )
        for rank in range(4)
    ]
    batches = [next(iter(sampler)) for sampler in samplers]
    assert all({item.source for item in batch} == {0} for batch in batches)
    rank_frames = [{item.frame for item in batch} for batch in batches]
    assert len(set.union(*rank_frames)) == 32

    original = BalancedDistributedBatchSampler(
        (25,),
        local_batch_size=8,
        sample_weights=(1.0,),
        seed=5,
        num_batches=4,
    )
    iterator = iter(original)
    next(iterator)
    state = original.state_dict()
    expected = next(iterator)
    restored = BalancedDistributedBatchSampler(
        (25,),
        local_batch_size=8,
        sample_weights=(1.0,),
        seed=5,
        num_batches=4,
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
        mean_train_loss=1.25,
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
    assert _resolve_resume(tmp_path, "latest") == checkpoint
    step = _load_training_state(
        _resolve_resume(tmp_path, "latest"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        device="cpu",
    )
    assert step == 7
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name])


def test_checkpoint_resume_realigns_scheduler_to_new_max_steps(tmp_path):
    model = OctoSmallPolicy(TinyTextEncoder(16), _tiny_config())
    peak_lr = 3.0e-4
    resume_step = 2_700
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=peak_lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_lambda(
            step,
            warmup_steps=400,
            max_steps=10_000,
            end_ratio=0.0,
        ),
    )
    old_learning_rate = peak_lr * _learning_rate_lambda(
        resume_step,
        warmup_steps=400,
        max_steps=10_000,
        end_ratio=0.0,
    )
    optimizer.param_groups[0]["lr"] = old_learning_rate
    scheduler.last_epoch = resume_step
    scheduler._last_lr = [old_learning_rate]
    sampler = BalancedDistributedBatchSampler(
        (20, 20), local_batch_size=4, seed=3, num_batches=3
    )
    checkpoint = _save_checkpoint(
        output=tmp_path,
        step=resume_step,
        mean_train_loss=1.0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config={"train": {"seed": 3, "max_steps": 10_000}},
    )

    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=peak_lr,
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer,
        lambda step: _learning_rate_lambda(
            step,
            warmup_steps=400,
            max_steps=5_000,
            end_ratio=0.0,
        ),
    )
    resumed_sampler = BalancedDistributedBatchSampler(
        (20, 20), local_batch_size=4, seed=3, num_batches=3
    )

    restored_step = _load_training_state(
        checkpoint,
        model=model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        sampler=resumed_sampler,
        device="cpu",
    )

    expected_learning_rate = peak_lr * _learning_rate_lambda(
        resume_step,
        warmup_steps=400,
        max_steps=5_000,
        end_ratio=0.0,
    )
    assert restored_step == resume_step
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_learning_rate)
    assert resumed_scheduler.get_last_lr() == pytest.approx([expected_learning_rate])


def test_checkpoint_resume_rejects_changed_target_or_prior_selection(tmp_path):
    model = OctoSmallPolicy(TinyTextEncoder(16), _tiny_config())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = BalancedDistributedBatchSampler(
        (20, 20), local_batch_size=4, seed=3, num_batches=3
    )
    checkpoint = _save_checkpoint(
        output=tmp_path,
        step=1,
        mean_train_loss=1.0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config={"train": {"seed": 3}},
        selection_signature="task-5-top10-selection",
    )

    with pytest.raises(RuntimeError, match="data selection differs"):
        _load_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            device="cpu",
            selection_signature="task-6-top10-selection",
        )


def test_training_selection_signature_covers_target_and_prior():
    task_5 = SimpleNamespace(selection_sha256="task-5")
    task_6 = SimpleNamespace(selection_sha256="task-6")
    top_10 = SimpleNamespace(selection_sha256="top-10")
    top_20 = SimpleNamespace(selection_sha256="top-20")

    baseline = training_selection_sha256(task_5, top_10)
    assert training_selection_sha256(task_6, top_10) != baseline
    assert training_selection_sha256(task_5, top_20) != baseline
    assert training_selection_sha256(task_5, None) != baseline
    assert training_selection_sha256(task_5, top_10, [1.0, 1.0]) == baseline
    assert training_selection_sha256(task_5, top_10, [3.0, 1.0]) != baseline


def test_checkpoint_retention_keeps_only_latest_and_best(tmp_path):
    model = OctoSmallPolicy(TinyTextEncoder(16), _tiny_config())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = BalancedDistributedBatchSampler(
        (20, 20), local_batch_size=4, seed=3, num_batches=3
    )
    config = {"train": {"seed": 3}}
    checkpoints = tmp_path / "checkpoints"
    unrelated_directory = checkpoints / "step-manual"
    temporary_directory = checkpoints / ".step-99999999.tmp"
    unrelated_directory.mkdir(parents=True)
    temporary_directory.mkdir()

    for step, mean_train_loss in ((1_000, 3.0), (2_000, 1.0), (3_000, 2.0)):
        _save_checkpoint(
            output=tmp_path,
            step=step,
            mean_train_loss=mean_train_loss,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            config=config,
        )

    step_directories = _step_checkpoint_directories(checkpoints)
    assert step_directories == ["step-00002000", "step-00003000"]
    assert unrelated_directory.is_dir()
    assert temporary_directory.is_dir()
    assert not (tmp_path / "best").exists()
    assert not (checkpoints / "best").exists()

    latest = json.loads((checkpoints / "latest.json").read_text(encoding="utf-8"))
    best = json.loads((checkpoints / "best.json").read_text(encoding="utf-8"))
    assert latest == {"checkpoint": "step-00003000", "step": 3_000}
    assert best == {
        "checkpoint": "step-00002000",
        "step": 2_000,
        "mean_train_loss": 1.0,
    }

    _save_checkpoint(
        output=tmp_path,
        step=4_000,
        mean_train_loss=1.0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
    )
    best_after_tie = json.loads(
        (checkpoints / "best.json").read_text(encoding="utf-8")
    )
    assert best_after_tie["step"] == 2_000
    assert _step_checkpoint_directories(checkpoints) == [
        "step-00002000",
        "step-00004000",
    ]

    _save_checkpoint(
        output=tmp_path,
        step=4_500,
        mean_train_loss=0.5,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        config=config,
    )
    assert _step_checkpoint_directories(checkpoints) == ["step-00004500"]
    assert json.loads(
        (checkpoints / "best.json").read_text(encoding="utf-8")
    )["step"] == 4_500


def test_checkpoint_schedule_always_includes_final_step():
    assert _should_save_checkpoint(step=1_000, max_steps=4_500, save_every=1_000)
    assert not _should_save_checkpoint(step=4_499, max_steps=4_500, save_every=1_000)
    assert _should_save_checkpoint(step=4_500, max_steps=4_500, save_every=1_000)
