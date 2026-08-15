import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from qwen3_vl_groot.config import load_config  # noqa: E402
from qwen3_vl_groot.training import (  # noqa: E402
    PerformanceWindow,
    RankZeroLogger,
    _lora_step_metrics,
    _runtime_metadata,
    _runtime_versions,
    _should_save_checkpoint,
    _should_save_final_checkpoint,
    build_optimizer_and_scheduler,
)
from qwen3_vl_groot.schedules import LoraUpdateSchedule  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Linear(4, 4)
        self.lora_a = nn.Parameter(torch.randn(4, 2))
        self.lora_b = nn.Parameter(torch.randn(2, 4))

    def action_head_parameters(self):
        return list(self.action_head.parameters())

    def lora_parameters(self):
        return [self.lora_a, self.lora_b]


def test_rank_zero_logger_mirrors_persisted_payload_to_stdout(tmp_path, capsys):
    logger = RankZeroLogger(tmp_path, enabled=True)
    try:
        logger.log({"step": 10, "train/loss": 1.25})
    finally:
        logger.close()

    stdout_payload = json.loads(capsys.readouterr().out)
    persisted_payload = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert stdout_payload == persisted_payload
    assert stdout_payload["step"] == 10
    assert stdout_payload["train/loss"] == pytest.approx(1.25)


def test_rank_zero_logger_status_messages_obey_enabled_flag(tmp_path, capsys):
    enabled_logger = RankZeroLogger(tmp_path / "enabled", enabled=True)
    disabled_logger = RankZeroLogger(tmp_path / "disabled", enabled=False)
    try:
        enabled_logger.status("Starting first training batch.")
        disabled_logger.status("This must remain hidden.")
    finally:
        enabled_logger.close()
        disabled_logger.close()

    assert capsys.readouterr().out == "Starting first training batch.\n"


def test_optimizer_groups_are_nonempty_and_trainable_before_zero_init():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    policy = TinyPolicy()
    optimizer, scheduler = build_optimizer_and_scheduler(policy, config)
    assert len(optimizer.param_groups) == 2
    assert all(
        any(parameter.requires_grad for parameter in group["params"])
        for group in optimizer.param_groups
    )
    assert optimizer.param_groups[0]["group_name"] == "action_head"
    assert optimizer.param_groups[1]["group_name"] == "qwen_lora"
    # LambdaLR initializes the delayed LoRA group at zero learning rate.
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert scheduler is not None


def test_optimizer_groups_receive_independent_configured_learning_rates():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["train"]["head_learning_rate"] = 2e-4
    config["train"]["lora_learning_rate"] = 5e-6

    optimizer, scheduler = build_optimizer_and_scheduler(TinyPolicy(), config)

    assert scheduler.base_lrs == pytest.approx([2e-4, 5e-6])
    assert optimizer.param_groups[0]["group_name"] == "action_head"
    assert optimizer.param_groups[1]["group_name"] == "qwen_lora"


def test_lora_scheduler_uses_only_active_update_clock():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["train"].update(
        {
            "lora_freeze_steps": 5,
            "lora_cycle_steps": 10,
            "lora_active_steps": 2,
            "lora_warmup_steps": 4,
            "max_steps": 25,
        }
    )
    optimizer, scheduler = build_optimizer_and_scheduler(TinyPolicy(), config)
    base_lr = float(config["train"]["lora_learning_rate"])

    used_lrs = {}
    for step in range(1, 26):
        used_lrs[step] = optimizer.param_groups[1]["lr"]
        optimizer.step()
        scheduler.step()

    assert {step for step, learning_rate in used_lrs.items() if learning_rate > 0} == {
        14,
        15,
        24,
        25,
    }
    assert [used_lrs[step] / base_lr for step in (14, 15, 24, 25)] == pytest.approx(
        [0.25, 0.5, 0.75, 1.0]
    )


@pytest.mark.parametrize(
    ("step", "enabled", "updates"),
    [
        (5_090, 0, 0),
        (5_091, 1, 1),
        (5_100, 1, 10),
        (5_101, 0, 10),
    ],
)
def test_lora_step_metrics_describe_completed_optimizer_step(step, enabled, updates):
    schedule = LoraUpdateSchedule(
        freeze_steps=5_000,
        cycle_steps=100,
        active_steps=10,
    )

    metrics = _lora_step_metrics(schedule, optimizer_step=step, learning_rate=5e-6)

    assert metrics == {
        "train/lora_lr": 5e-6,
        "train/lora_enabled": enabled,
        "train/lora_updates": updates,
    }


def test_checkpoint_schedule_saves_improvements_and_unscheduled_final_step():
    assert _should_save_checkpoint(step=1_000, save_every=1_000, improved=False)
    assert _should_save_checkpoint(step=750, save_every=1_000, improved=True)
    assert not _should_save_checkpoint(step=750, save_every=1_000, improved=False)
    assert _should_save_final_checkpoint(step=4_500, last_checkpoint_step=4_000)
    assert not _should_save_final_checkpoint(step=4_000, last_checkpoint_step=4_000)


def test_performance_window_reports_effective_batch_throughput_and_data_wait():
    window = PerformanceWindow(
        effective_batch_size=64,
        start_step=20,
        started_at=100.0,
    )
    window.add_data_wait(4.0)

    metrics = window.metrics(optimizer_step=22, now=120.0)

    assert metrics == pytest.approx(
        {
            "performance/step_seconds": 10.0,
            "performance/samples_per_second": 6.4,
            "performance/data_wait_fraction": 0.2,
        }
    )


def test_performance_window_reset_starts_a_fresh_interval():
    window = PerformanceWindow(
        effective_batch_size=64,
        start_step=0,
        started_at=10.0,
    )
    window.add_data_wait(2.0)
    window.reset(start_step=5, started_at=30.0)
    window.add_data_wait(1.0)

    metrics = window.metrics(optimizer_step=7, now=40.0)

    assert metrics["performance/step_seconds"] == pytest.approx(5.0)
    assert metrics["performance/samples_per_second"] == pytest.approx(12.8)
    assert metrics["performance/data_wait_fraction"] == pytest.approx(0.1)


def test_performance_window_excludes_evaluation_and_checkpoint_overhead():
    window = PerformanceWindow(
        effective_batch_size=64,
        start_step=10,
        started_at=100.0,
    )
    window.add_data_wait(2.0)
    window.exclude_elapsed(30.0)

    metrics = window.metrics(optimizer_step=12, now=150.0)

    assert metrics["performance/step_seconds"] == pytest.approx(10.0)
    assert metrics["performance/data_wait_fraction"] == pytest.approx(0.1)


def test_runtime_versions_records_optional_packages_without_failing(monkeypatch):
    versions = {
        "torch": "2.9.0",
        "transformers": "4.57.3",
        "peft": "0.17.1",
        "deepspeed": "0.17.6",
        "flash-attn": "2.7.4.post1",
    }

    def fake_version(name):
        if name == "flash-attn":
            raise ModuleNotFoundError(name)
        return versions[name]

    monkeypatch.setattr("qwen3_vl_groot.training.package_version", fake_version)

    result = _runtime_versions()

    assert result["torch"] == "2.9.0"
    assert result["transformers"] == "4.57.3"
    assert result["peft"] == "0.17.1"
    assert result["deepspeed"] == "0.17.6"
    assert result["flash_attn"] is None


def test_runtime_metadata_records_resolved_attention_and_compile_targets(monkeypatch):
    config = load_config(
        PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
    )
    config["model"]["attn_implementation"] = "eager"
    config["model"]["context_forward"] = "backbone"
    monkeypatch.setattr(
        "qwen3_vl_groot.training._runtime_versions",
        lambda: {"torch": "test"},
    )

    metadata = _runtime_metadata(config, world_size=4)

    assert metadata["packages"] == {"torch": "test"}
    assert metadata["attention"] == {"requested": "eager", "resolved": "eager"}
    assert metadata["context_forward"] == "backbone"
    assert metadata["torch_compile"]["backbone_enabled"] is False
    assert metadata["torch_compile"]["action_head_enabled"] is True
    assert metadata["torch_compile"]["dynamic"] is False
    assert metadata["torch_compile"]["action_head_context_buckets"] == [
        96,
        192,
        384,
        512,
    ]
    assert metadata["effective_batch_size"] == 64
