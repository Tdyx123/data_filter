from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"train.{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class LoraUpdateSchedule:
    freeze_steps: int
    cycle_steps: int | None = None
    active_steps: int | None = None

    @classmethod
    def from_train_config(cls, train: Mapping[str, Any]) -> LoraUpdateSchedule:
        return cls(
            freeze_steps=train["lora_freeze_steps"],
            cycle_steps=train.get("lora_cycle_steps"),
            active_steps=train.get("lora_active_steps"),
        )

    def __post_init__(self) -> None:
        _require_int("lora_freeze_steps", self.freeze_steps, minimum=0)
        if (self.cycle_steps is None) != (self.active_steps is None):
            raise ValueError(
                "train.lora_cycle_steps and train.lora_active_steps must be provided together"
            )
        if self.cycle_steps is None:
            return

        cycle_steps = _require_int("lora_cycle_steps", self.cycle_steps, minimum=1)
        active_steps = _require_int("lora_active_steps", self.active_steps, minimum=1)
        if active_steps > cycle_steps:
            raise ValueError(
                "train.lora_active_steps must not exceed train.lora_cycle_steps"
            )

    def is_active(self, optimizer_step: int) -> bool:
        if optimizer_step <= self.freeze_steps:
            return False
        if self.cycle_steps is None:
            return True

        assert self.active_steps is not None
        position = (optimizer_step - self.freeze_steps - 1) % self.cycle_steps
        return position >= self.cycle_steps - self.active_steps

    def active_steps_through(self, optimizer_step: int) -> int:
        post_freeze = max(optimizer_step - self.freeze_steps, 0)
        if self.cycle_steps is None:
            return post_freeze

        assert self.active_steps is not None
        full_cycles, tail = divmod(post_freeze, self.cycle_steps)
        tail_active = max(tail - (self.cycle_steps - self.active_steps), 0)
        return full_cycles * self.active_steps + tail_active
