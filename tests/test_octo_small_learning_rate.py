from __future__ import annotations

import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from octo_small_libero import training


def test_cosine_learning_rate_uses_max_steps_as_decay_end():
    values = {
        step: training._learning_rate_lambda(
            step,
            warmup_steps=400,
            max_steps=5_000,
            end_ratio=0.0,
        )
        for step in (0, 399, 400, 2_700, 5_000, 5_500)
    }

    assert values[0] == pytest.approx(1.0 / 400.0)
    assert values[399] == pytest.approx(1.0)
    assert values[400] == pytest.approx(1.0)
    assert values[2_700] == pytest.approx(0.5)
    assert values[5_000] == pytest.approx(0.0)
    assert values[5_500] == pytest.approx(0.0)


def test_short_smoke_schedule_remains_warmup_only():
    values = [
        training._learning_rate_lambda(
            step,
            warmup_steps=400,
            max_steps=2,
            end_ratio=0.0,
        )
        for step in (0, 1)
    ]

    assert values == pytest.approx([1.0 / 400.0, 2.0 / 400.0])


def test_resume_realigns_learning_rate_to_absolute_step_and_new_max_steps(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    peak_lr = 3.0e-4
    resume_step = 2_700
    old_factor = training._learning_rate_lambda(
        resume_step,
        warmup_steps=400,
        max_steps=10_000,
        end_ratio=0.0,
    )
    expected_factor = training._learning_rate_lambda(
        resume_step,
        warmup_steps=400,
        max_steps=5_000,
        end_ratio=0.0,
    )
    state = {
        "step": resume_step,
        "optimizer": {"lr": peak_lr * old_factor},
        "scheduler": {
            "last_epoch": resume_step,
            "_last_lr": [peak_lr * old_factor],
        },
        "sampler": {},
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": object(),
        "cuda_rng": None,
        "selection_signature": None,
    }

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": peak_lr}]

        def load_state_dict(self, optimizer_state):
            self.param_groups[0]["lr"] = optimizer_state["lr"]

    class FakeScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.base_lrs = [peak_lr]
            self.lr_lambdas = [
                lambda step: training._learning_rate_lambda(
                    step,
                    warmup_steps=400,
                    max_steps=5_000,
                    end_ratio=0.0,
                )
            ]
            self.last_epoch = 0
            self._last_lr = [peak_lr]

        def load_state_dict(self, scheduler_state):
            self.last_epoch = scheduler_state["last_epoch"]
            self._last_lr = list(scheduler_state["_last_lr"])

        def get_last_lr(self):
            return list(self._last_lr)

    class FakeSampler:
        def load_state_dict(self, _state):
            pass

    torch_module = ModuleType("torch")
    torch_module.load = lambda *_args, **_kwargs: state
    torch_module.set_rng_state = lambda _state: None
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    safetensors_module = ModuleType("safetensors")
    safetensors_torch_module = ModuleType("safetensors.torch")
    safetensors_torch_module.load_model = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_module)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch_module)

    optimizer = FakeOptimizer()
    scheduler = FakeScheduler(optimizer)
    restored_step = training._load_training_state(
        checkpoint,
        model=object(),
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=FakeSampler(),
        device="cpu",
    )

    assert restored_step == resume_step
    assert optimizer.param_groups[0]["lr"] == pytest.approx(peak_lr * expected_factor)
    assert scheduler.get_last_lr() == pytest.approx([peak_lr * expected_factor])
