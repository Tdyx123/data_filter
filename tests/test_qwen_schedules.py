import pytest

from qwen3_vl_groot.schedules import LoraUpdateSchedule


def test_cyclic_lora_schedule_uses_last_ten_steps_of_each_post_freeze_cycle():
    schedule = LoraUpdateSchedule(
        freeze_steps=5_000,
        cycle_steps=100,
        active_steps=10,
    )

    expected = {
        0: False,
        5_000: False,
        5_001: False,
        5_090: False,
        5_091: True,
        5_100: True,
        5_101: False,
        5_190: False,
        5_191: True,
    }
    assert {step: schedule.is_active(step) for step in expected} == expected


def test_cyclic_lora_schedule_counts_full_and_partial_active_windows():
    schedule = LoraUpdateSchedule(
        freeze_steps=5_000,
        cycle_steps=100,
        active_steps=10,
    )

    assert schedule.active_steps_through(5_000) == 0
    assert schedule.active_steps_through(5_090) == 0
    assert schedule.active_steps_through(5_091) == 1
    assert schedule.active_steps_through(5_100) == 10
    assert schedule.active_steps_through(5_190) == 10
    assert schedule.active_steps_through(5_195) == 15
    assert schedule.active_steps_through(5_200) == 20


def test_schedule_without_cycle_stays_active_after_freeze():
    schedule = LoraUpdateSchedule(freeze_steps=2_000)

    assert not schedule.is_active(2_000)
    assert schedule.is_active(2_001)
    assert schedule.is_active(20_000)
    assert schedule.active_steps_through(2_000) == 0
    assert schedule.active_steps_through(2_003) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"freeze_steps": -1}, "lora_freeze_steps"),
        ({"freeze_steps": True}, "lora_freeze_steps"),
        ({"freeze_steps": 5_000, "cycle_steps": 100}, "provided together"),
        ({"freeze_steps": 5_000, "active_steps": 10}, "provided together"),
        (
            {"freeze_steps": 5_000, "cycle_steps": 0, "active_steps": 1},
            "lora_cycle_steps",
        ),
        (
            {"freeze_steps": 5_000, "cycle_steps": 100, "active_steps": 0},
            "lora_active_steps",
        ),
        (
            {"freeze_steps": 5_000, "cycle_steps": 10, "active_steps": 11},
            "must not exceed",
        ),
        (
            {"freeze_steps": 5_000, "cycle_steps": True, "active_steps": 1},
            "lora_cycle_steps",
        ),
        (
            {"freeze_steps": 5_000, "cycle_steps": 100.0, "active_steps": 10},
            "lora_cycle_steps",
        ),
    ],
)
def test_lora_schedule_rejects_invalid_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LoraUpdateSchedule(**kwargs)
