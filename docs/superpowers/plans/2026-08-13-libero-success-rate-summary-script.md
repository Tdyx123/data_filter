# LIBERO 平均成功率汇总脚本实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个只读取 `results.json` 的命令行脚本，输出完整完成 LIBERO-10 十个任务的实验文件夹及其等权平均成功率。

**Architecture:** 新脚本使用标准库将“单任务解析”“单实验十任务校验”“根目录扫描”和“命令行输出”分开。无效实验通过自定义异常携带简洁原因，由根目录扫描层跳过并写入标准错误；根目录级错误由命令行入口返回非零状态。

**Tech Stack:** Python 3.10+ 标准库、pytest、subprocess

## Global Constraints

- 默认根目录严格为 `/data/dwb/octo_small_libero/`，同时接受一个可选位置参数覆盖根目录。
- 只扫描根目录的直属子文件夹；每个有效实验必须完整包含 `task-0` 至 `task-9`。
- 只读取各任务的 `results.json`，完全忽略 `failure.json` 和 `episodes.jsonl`。
- 每个结果必须满足：`status == "complete"`、`protocol.episodes` 为正整数、`summary.completed_episodes` 为非负整数且等于总回合数、`summary.success_rate` 为 `[0, 1]` 内有限数值；布尔值不得作为整数或成功率。
- 实验成功率只取十个 `summary.success_rate` 的等权算术平均，不做 episode 加权，不读取 `summary.by_seed`。
- 标准输出每行只含文件夹名和两位小数百分比，按平均成功率降序、名称升序稳定排序；跳过原因只写标准错误。
- 不修改结果目录，不增加第三方运行时依赖。

---

### Task 1: 完整实验的解析、平均与稳定输出

**Files:**
- Create: `tests/test_summarize_libero_success_rates.py`
- Create: `scripts/summarize_libero_success_rates.py`

**Interfaces:**
- Consumes: 根目录路径；每个 `task-N/results.json` 的 `status`、`protocol.episodes`、`summary.completed_episodes` 和 `summary.success_rate`。
- Produces: `parse_task_success_rate(path: Path) -> float`、`summarize_experiment(path: Path) -> float`、`collect_success_rates(root: Path) -> list[tuple[str, float]]`、`main(argv: Sequence[str] | None = None) -> int`。

- [ ] **Step 1: 写入完整实验、排序和忽略 `failure.json` 的失败测试**

```python
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "summarize_libero_success_rates.py"


def _write_experiment(root: Path, name: str, rates: list[float]) -> Path:
    experiment = root / name
    for task_index, rate in enumerate(rates):
        task_dir = experiment / f"task-{task_index}"
        task_dir.mkdir(parents=True)
        (task_dir / "results.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "protocol": {"episodes": 150},
                    "summary": {
                        "completed_episodes": 150,
                        "success_rate": rate,
                    },
                }
            ),
            encoding="utf-8",
        )
    return experiment


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_outputs_only_mean_success_rate_in_stable_order(tmp_path):
    _write_experiment(tmp_path, "beta", [0.2] * 10)
    _write_experiment(tmp_path, "alpha", [0.1, 0.3] * 5)
    _write_experiment(tmp_path, "winner", [0.4] * 10)

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout.splitlines() == [
        "winner  40.00%",
        "alpha  20.00%",
        "beta  20.00%",
    ]
    assert completed.stderr == ""


def test_failure_json_is_ignored(tmp_path):
    experiment = _write_experiment(tmp_path, "recovered", [0.25] * 10)
    (experiment / "task-4" / "failure.json").write_text(
        json.dumps({"status": "failed"}),
        encoding="utf-8",
    )

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == "recovered  25.00%\n"
    assert completed.stderr == ""
```

- [ ] **Step 2: 运行测试并确认因脚本缺失而失败**

Run: `pytest -q tests/test_summarize_libero_success_rates.py`

Expected: FAIL；子进程无法打开 `scripts/summarize_libero_success_rates.py`，而不是测试夹具或断言错误。

- [ ] **Step 3: 写入使完整实验测试通过的最小实现**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_ROOT = Path("/data/dwb/octo_small_libero/")
TASK_COUNT = 10


class InvalidExperiment(ValueError):
    """Raised when one experiment does not contain ten valid completed results."""


def parse_task_success_rate(path: Path) -> float:
    value = json.loads(path.read_text(encoding="utf-8"))
    return float(value["summary"]["success_rate"])


def summarize_experiment(path: Path) -> float:
    rates = [
        parse_task_success_rate(path / f"task-{index}" / "results.json")
        for index in range(TASK_COUNT)
    ]
    return math.fsum(rates) / TASK_COUNT


def collect_success_rates(root: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for experiment in sorted(root.iterdir(), key=lambda item: item.name):
        if not experiment.is_dir():
            continue
        try:
            mean = summarize_experiment(experiment)
        except InvalidExperiment as error:
            print(f"skip {experiment.name}: {error}", file=sys.stderr)
            continue
        rows.append((experiment.name, mean))
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print mean LIBERO-10 task success rates for complete experiments."
    )
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = build_parser().parse_args(argv).root
    if not root.is_dir():
        print(f"error: result root is not a directory: {root}", file=sys.stderr)
        return 1
    try:
        rows = collect_success_rates(root)
    except OSError as error:
        print(f"error: cannot read result root {root}: {error}", file=sys.stderr)
        return 1
    for name, rate in rows:
        print(f"{name}  {rate:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试并确认完整实验行为通过**

Run: `pytest -q tests/test_summarize_libero_success_rates.py`

Expected: `2 passed`。

- [ ] **Step 5: 提交完整实验的首个可用版本**

```bash
git add scripts/summarize_libero_success_rates.py tests/test_summarize_libero_success_rates.py
git commit -m "feat: summarize complete LIBERO success rates"
```

### Task 2: 无效实验和根目录错误的严格校验

**Files:**
- Modify: `tests/test_summarize_libero_success_rates.py`
- Modify: `scripts/summarize_libero_success_rates.py`

**Interfaces:**
- Consumes: Task 1 的 `_write_experiment()`、`_run()` 夹具以及脚本四个公共函数。
- Produces: 对缺任务、缺结果、状态未完成、回合数不完整、损坏/无效 JSON 和根目录错误的稳定跳过或失败行为。

- [ ] **Step 1: 添加无效实验与根目录错误的参数化失败测试**

```python
import math

import pytest


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing_task", "task-9/results.json"),
        ("missing_result", "task-4/results.json"),
        ("bad_json", "task-3/results.json"),
        ("failed_status", "not complete"),
        ("incomplete", "incomplete episodes"),
        ("missing_field", "valid result"),
        ("bool_episodes", "invalid protocol.episodes"),
        ("bool_rate", "invalid summary.success_rate"),
        ("nan_rate", "invalid summary.success_rate"),
        ("high_rate", "invalid summary.success_rate"),
    ],
)
def test_skips_entire_experiment_when_any_task_is_invalid(
    tmp_path, mutation, expected_reason
):
    experiment = _write_experiment(tmp_path, "invalid", [0.5] * 10)
    result_path = experiment / "task-3" / "results.json"
    value = json.loads(result_path.read_text(encoding="utf-8"))

    if mutation == "missing_task":
        (experiment / "task-9" / "results.json").unlink()
        (experiment / "task-9").rmdir()
    elif mutation == "missing_result":
        (experiment / "task-4" / "results.json").unlink()
    elif mutation == "bad_json":
        result_path.write_text("{", encoding="utf-8")
    else:
        if mutation == "failed_status":
            value["status"] = "failed"
        elif mutation == "incomplete":
            value["summary"]["completed_episodes"] = 149
        elif mutation == "missing_field":
            del value["summary"]["success_rate"]
        elif mutation == "bool_episodes":
            value["protocol"]["episodes"] = True
            value["summary"]["completed_episodes"] = True
        elif mutation == "bool_rate":
            value["summary"]["success_rate"] = True
        elif mutation == "nan_rate":
            value["summary"]["success_rate"] = math.nan
        elif mutation == "high_rate":
            value["summary"]["success_rate"] = 1.01
        result_path.write_text(json.dumps(value), encoding="utf-8")

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert "skip invalid:" in completed.stderr
    assert expected_reason in completed.stderr


def test_no_valid_experiment_has_empty_stdout(tmp_path):
    (tmp_path / "ordinary-file").write_text("ignored", encoding="utf-8")

    completed = _run(tmp_path)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_invalid_root_returns_nonzero(tmp_path, root_kind):
    root = tmp_path / "results"
    if root_kind == "file":
        root.write_text("not a directory", encoding="utf-8")

    completed = _run(root)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "result root is not a directory" in completed.stderr
```

- [ ] **Step 2: 运行新增测试并确认边界行为失败**

Run: `pytest -q tests/test_summarize_libero_success_rates.py`

Expected: FAIL；缺文件或损坏 JSON 会产生未捕获异常，失败状态、未完成回合、布尔值或越界成功率会被错误输出。确认失败来自缺少严格校验，不是夹具错误。

- [ ] **Step 3: 按失败结果补齐最小校验，不改变输出口径**

将 `parse_task_success_rate()` 替换为以下严格解析；保留 `collect_success_rates()` 的逐实验跳过逻辑与 `main()` 的根目录非零退出逻辑：

```python
def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_task_success_rate(path: Path) -> float:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        status = value["status"]
        episodes = value["protocol"]["episodes"]
        completed = value["summary"]["completed_episodes"]
        rate = value["summary"]["success_rate"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise InvalidExperiment(f"cannot read a valid result at {path}") from error
    if status != "complete":
        raise InvalidExperiment(f"result is not complete at {path}")
    if not _is_int(episodes) or episodes <= 0:
        raise InvalidExperiment(f"invalid protocol.episodes at {path}")
    if not _is_int(completed) or completed < 0 or completed != episodes:
        raise InvalidExperiment(f"incomplete episodes at {path}")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise InvalidExperiment(f"invalid summary.success_rate at {path}")
    try:
        result = float(rate)
    except (OverflowError, ValueError) as error:
        raise InvalidExperiment(f"invalid summary.success_rate at {path}") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise InvalidExperiment(f"invalid summary.success_rate at {path}")
    return result
```

- [ ] **Step 4: 运行目标测试并确认全部通过**

Run: `pytest -q tests/test_summarize_libero_success_rates.py`

Expected: `15 passed`，无 warning 或 traceback。

- [ ] **Step 5: 对真实目录执行只读验收**

Run: `python3 scripts/summarize_libero_success_rates.py /data/dwb/octo_small_libero/`

Expected: 返回码为 `0`；标准输出只包含通过十任务完整性校验的实验名称与两位小数平均成功率，并按成功率降序；已知不完整目录（例如仅一个结果的目录）只出现在标准错误的跳过诊断中。

- [ ] **Step 6: 运行相关回归测试**

Run: `pytest -q tests/test_summarize_libero_success_rates.py tests/test_octo_small_evaluation.py tests/test_evaluate_libero_script.py`

Expected: 全部 PASS，且无新的 warning 或 traceback。

- [ ] **Step 7: 提交严格校验与测试**

```bash
git add scripts/summarize_libero_success_rates.py tests/test_summarize_libero_success_rates.py
git commit -m "test: cover invalid LIBERO evaluation folders"
```
