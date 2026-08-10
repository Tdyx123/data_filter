# Qwen LIBERO 周期性 LoRA 训练实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个 4×4090 LIBERO 全任务训练脚本，使前 5000 个 optimizer step 只训练动作头，之后每 100 步仅最后 10 步训练 LoRA 与动作头。

**Architecture:** 在独立的纯 Python 调度模块中定义 1-based LoRA 更新规则和累计有效更新计数，训练循环与 LambdaLR 共用该规则，避免边界不一致。新 shell 脚本包装现有 LIBERO 入口并注入默认调度参数；现有未配置周期字段的入口继续在冻结结束后持续训练 LoRA。

**Tech Stack:** Python 3.12、PyTorch 2.9、DeepSpeed 0.17.6、argparse、Bash、pytest 8.3。

## 全局约束

- 直接在当前 `main` 分支修改，不创建 worktree 或新分支。
- Qwen3-VL-4B 原始参数始终冻结；“训练 Qwen”仅指训练覆盖 36 层注意力投影的 LoRA。
- 调度按 optimizer step 而非 gradient-accumulation micro-step 计数。
- 第 1–5000 步仅动作头；5001–5090 仅动作头；5091–5100 LoRA 与动作头；此后每 100 步重复。
- LoRA warmup 和余弦衰减只按实际 LoRA 更新步推进；动作头调度保持现状。
- 不启动真实多卡训练，只运行单元测试、shell 语法检查和静态检查。

---

### Task 1: LoRA 周期调度模型与配置校验

**Files:**
- Create: `src/qwen3_vl_groot/schedules.py`
- Modify: `src/qwen3_vl_groot/config.py`
- Test: `tests/test_qwen_schedules.py`
- Test: `tests/test_qwen_config_core.py`

**Interfaces:**
- Produces: `LoraUpdateSchedule.from_train_config(train: Mapping[str, Any]) -> LoraUpdateSchedule`
- Produces: `LoraUpdateSchedule.is_active(optimizer_step: int) -> bool`
- Produces: `LoraUpdateSchedule.active_steps_through(optimizer_step: int) -> int`
- Configuration keys: `train.lora_freeze_steps`, optional paired `train.lora_cycle_steps` and `train.lora_active_steps`

- [ ] **Step 1: 写入调度边界失败测试**

```python
def test_cyclic_lora_schedule_uses_last_ten_steps_of_each_post_freeze_cycle():
    schedule = LoraUpdateSchedule(freeze_steps=5_000, cycle_steps=100, active_steps=10)
    expected = {
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
    assert schedule.active_steps_through(5_090) == 0
    assert schedule.active_steps_through(5_100) == 10
    assert schedule.active_steps_through(5_195) == 15
```

- [ ] **Step 2: 运行测试并确认因模块尚不存在而失败**

Run: `pytest -q tests/test_qwen_schedules.py`

Expected: FAIL，错误指向 `qwen3_vl_groot.schedules` 或 `LoraUpdateSchedule` 不存在。

- [ ] **Step 3: 实现不可变调度值对象**

```python
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
    def from_train_config(cls, train: Mapping[str, Any]) -> "LoraUpdateSchedule":
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
            raise ValueError("train.lora_active_steps must not exceed train.lora_cycle_steps")

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
```

实现规则：非正 step 视为尚无更新；无周期字段时 `step > freeze_steps` 均激活；有周期字段时用 `(step - freeze_steps - 1) % cycle_steps` 判断后 `active_steps` 个位置；累计值按完整周期加尾周期激活区间计算。

- [ ] **Step 4: 增加配置失败测试**

```python
@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"lora_freeze_steps": -1}, "lora_freeze_steps"),
        ({"lora_cycle_steps": 100}, "provided together"),
        ({"lora_cycle_steps": 0, "lora_active_steps": 1}, "lora_cycle_steps"),
        ({"lora_cycle_steps": 100, "lora_active_steps": 0}, "lora_active_steps"),
        ({"lora_cycle_steps": 10, "lora_active_steps": 11}, "must not exceed"),
        ({"lora_cycle_steps": True, "lora_active_steps": 1}, "lora_cycle_steps"),
        ({"lora_cycle_steps": 100.0, "lora_active_steps": 10}, "lora_cycle_steps"),
    ],
)
def test_config_rejects_invalid_lora_schedule(updates, message):
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    config["train"].update(updates)
    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_existing_config_keeps_continuous_lora_schedule():
    config = load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml")
    schedule = LoraUpdateSchedule.from_train_config(config["train"])
    assert schedule.cycle_steps is None
    assert not schedule.is_active(config["train"]["lora_freeze_steps"])
    assert schedule.is_active(config["train"]["lora_freeze_steps"] + 1)
```

- [ ] **Step 5: 在 `validate_config` 复用调度对象校验**

```python
try:
    LoraUpdateSchedule.from_train_config(train)
except (KeyError, ValueError) as error:
    raise ConfigError(str(error)) from error
```

- [ ] **Step 6: 运行调度与配置测试**

Run: `pytest -q tests/test_qwen_schedules.py tests/test_qwen_config_core.py`

Expected: PASS。

- [ ] **Step 7: 提交调度模型**

```bash
git add src/qwen3_vl_groot/schedules.py src/qwen3_vl_groot/config.py tests/test_qwen_schedules.py tests/test_qwen_config_core.py
git commit -m "feat: add cyclic LoRA update schedule"
```

### Task 2: 优化器调度和训练循环接入

**Files:**
- Modify: `src/qwen3_vl_groot/training.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `LoraUpdateSchedule.from_train_config(train)`、`schedule.is_active(step)`、`schedule.active_steps_through(step)`
- Preserves: `build_optimizer_and_scheduler(policy, config)` 的返回类型与两个 parameter group 顺序
- Log fields: `train/lora_enabled`、`train/lora_updates`、`train/lora_lr`

- [ ] **Step 1: 写入 active-step 学习率失败测试**

```python
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

    assert {step for step, lr in used_lrs.items() if lr > 0} == {14, 15, 24, 25}
    assert [used_lrs[step] / base_lr for step in (14, 15, 24, 25)] == pytest.approx(
        [0.25, 0.5, 0.75, 1.0]
    )
```

- [ ] **Step 2: 运行目标测试并确认旧 scheduler 在冻结后连续给出非零 LR**

Run: `pytest -q tests/test_training.py -k "active_update_clock"`

Expected: FAIL，非零 LoRA LR 步集合不是 `{14, 15, 24, 25}`。

- [ ] **Step 3: 让 LambdaLR 使用累计有效 LoRA 更新数**

```python
lora_schedule_config = LoraUpdateSchedule.from_train_config(train)
maximum_lora_updates = lora_schedule_config.active_steps_through(maximum)

def lora_schedule(completed_steps: int) -> float:
    upcoming_step = completed_steps + 1
    if not lora_schedule_config.is_active(upcoming_step):
        return 0.0
    active_before = lora_schedule_config.active_steps_through(completed_steps)
    return _cosine_after_warmup(active_before, lora_warmup, maximum_lora_updates)
```

- [ ] **Step 4: 运行 scheduler 测试并确认通过**

Run: `pytest -q tests/test_training.py -k "optimizer or active_update_clock"`

Expected: PASS，包括原有连续 LoRA 调度测试。

- [ ] **Step 5: 写入训练状态/日志语义的失败测试**

```python
def test_lora_log_values_describe_the_completed_optimizer_step():
    schedule = LoraUpdateSchedule(freeze_steps=5_000, cycle_steps=100, active_steps=10)
    assert (schedule.is_active(5_090), schedule.active_steps_through(5_090)) == (False, 0)
    assert (schedule.is_active(5_091), schedule.active_steps_through(5_091)) == (True, 1)
    assert (schedule.is_active(5_100), schedule.active_steps_through(5_100)) == (True, 10)
    assert (schedule.is_active(5_101), schedule.active_steps_through(5_101)) == (False, 10)
```

- [ ] **Step 6: 在训练循环按下一 optimizer step 切换 LoRA**

```python
upcoming_step = global_step + 1
lora_enabled_for_update = lora_update_schedule.is_active(upcoming_step)
if lora_enabled_for_update != any(
    parameter.requires_grad for parameter in policy.lora_parameters()
):
    policy.set_lora_trainable(lora_enabled_for_update)

lora_lr_used = float(optimizer.param_groups[1]["lr"])
loss = engine(images=batch["images"], state=batch["state"], actions=batch["actions"],
              action_mask=batch["action_mask"], instructions=batch["instructions"])
engine.backward(loss)
engine.step()

metrics.update(
    {
        "train/lora_lr": lora_lr_used,
        "train/lora_enabled": int(lora_update_schedule.is_active(global_step)),
        "train/lora_updates": lora_update_schedule.active_steps_through(global_step),
    }
)
```

- [ ] **Step 7: 运行训练测试**

Run: `pytest -q tests/test_training.py tests/test_model_contract.py`

Expected: PASS。

- [ ] **Step 8: 提交训练循环接入**

```bash
git add src/qwen3_vl_groot/training.py tests/test_training.py
git commit -m "feat: apply cyclic LoRA schedule during training"
```

### Task 3: CLI 覆盖项和新训练脚本

**Files:**
- Create: `scripts/train_libero_qwen3_vl_4b_groot_cyclic_lora_all_tasks_4x4090.sh`
- Modify: `src/qwen3_vl_groot/cli.py`
- Modify: `src/qwen3_vl_groot/config.py`
- Modify: `tests/test_train_qwen_libero_script.py`
- Test: `tests/test_qwen_config_core.py`

**Interfaces:**
- CLI flags: `--lora-freeze-steps INT`、`--lora-cycle-steps INT`、`--lora-active-steps INT`
- New script defaults: `5000`、`100`、`10`

- [ ] **Step 1: 写入 CLI 与 shell 入口失败测试**

```python
def test_lora_schedule_overrides_are_applied():
    config = apply_overrides(
        load_config(PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"),
        {"lora_freeze_steps": 5_000, "lora_cycle_steps": 100, "lora_active_steps": 10},
    )
    assert config["train"]["lora_freeze_steps"] == 5_000
    assert config["train"]["lora_cycle_steps"] == 100
    assert config["train"]["lora_active_steps"] == 10


def test_cyclic_libero_script_injects_schedule_and_preserves_defaults(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)
    subprocess.run(
        ["bash", str(CYCLIC_LIBERO_SCRIPT), "--output-dir", "outputs/cyclic", "--preflight-only"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--lora-freeze-steps") + 1] == "5000"
    assert arguments[arguments.index("--lora-cycle-steps") + 1] == "100"
    assert arguments[arguments.index("--lora-active-steps") + 1] == "10"
    assert "--all-tasks" in arguments
    assert arguments[arguments.index("--sample-weights") + 1 : arguments.index("--sample-weights") + 3] == ["1", "1"]
    assert arguments[arguments.index("--prior-prefiltered-scores") + 1] == SQCN_SCORES
```

- [ ] **Step 2: 运行测试并确认新脚本/CLI 参数尚不存在**

Run: `pytest -q tests/test_train_qwen_libero_script.py tests/test_qwen_config_core.py -k "cyclic or schedule_override"`

Expected: FAIL，原因是脚本不存在或 override mapping 缺少字段。

- [ ] **Step 3: 增加 CLI 参数和 override mapping**

```python
# config.py / apply_overrides.mapping
"lora_freeze_steps": ("train", "lora_freeze_steps"),
"lora_cycle_steps": ("train", "lora_cycle_steps"),
"lora_active_steps": ("train", "lora_active_steps"),

# cli.py / _add_override_arguments
parser.add_argument("--lora-freeze-steps", type=int)
parser.add_argument("--lora-cycle-steps", type=int)
parser.add_argument("--lora-active-steps", type=int)
```

并将三个 destination 名称加入 `_overrides` 的 `keys` 元组，统一交由配置校验报告非法组合。

- [ ] **Step 4: 创建可执行包装脚本**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh" \
  --lora-freeze-steps 5000 \
  --lora-cycle-steps 100 \
  --lora-active-steps 10 \
  "$@"
```

将脚本设置为可执行；默认参数放在 `"$@"` 前面，使调用者可显式覆盖。

- [ ] **Step 5: 运行脚本与配置测试**

Run: `pytest -q tests/test_train_qwen_libero_script.py tests/test_qwen_config_core.py`

Expected: PASS。

- [ ] **Step 6: 运行 shell 语法检查**

Run: `bash -n scripts/train_libero_qwen3_vl_4b_groot_cyclic_lora_all_tasks_4x4090.sh scripts/train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh`

Expected: exit 0 且无输出。

- [ ] **Step 7: 提交入口脚本**

```bash
git add scripts/train_libero_qwen3_vl_4b_groot_cyclic_lora_all_tasks_4x4090.sh src/qwen3_vl_groot/cli.py src/qwen3_vl_groot/config.py tests/test_train_qwen_libero_script.py tests/test_qwen_config_core.py
git commit -m "feat: add cyclic LoRA LIBERO launcher"
```

### Task 4: 使用文档与完整回归验证

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: 新脚本命令、精确 step 边界、LoRA-only 含义、CLI 覆盖项和日志字段

- [ ] **Step 1: 更新 README 的 Qwen LIBERO 训练章节**

添加新脚本示例，并明确前 5000 步、首个 100-step 周期的 5001–5090/5091–5100 边界；说明基础 Qwen 参数不解冻，LoRA warmup 只统计有效更新。

- [ ] **Step 2: 运行相关测试集**

Run: `pytest -q tests/test_qwen_schedules.py tests/test_training.py tests/test_qwen_config_core.py tests/test_train_qwen_libero_script.py tests/test_cli.py tests/test_model_contract.py`

Expected: PASS。

- [ ] **Step 3: 运行全量测试与静态检查**

Run: `pytest -q`

Run: `python -m compileall -q src/qwen3_vl_groot`

Run: `git diff --check`

Expected: 三条命令均 exit 0；pytest 无失败，静态检查无输出。

- [ ] **Step 4: 检查最终差异和分支**

Run: `git branch --show-current && git status --short && git diff --stat HEAD~3..HEAD`

Expected: 当前分支为 `main`；只包含本功能相关变更。

- [ ] **Step 5: 提交文档与最终修整**

```bash
git add README.md
git commit -m "docs: explain cyclic LoRA LIBERO training"
```
