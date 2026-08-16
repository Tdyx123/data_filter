# Cocore 分步骤耗时日志实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Cocore 顶层阶段和关键子步骤增加成功完成耗时日志，同时保持缓存、artifact 和 stdout 契约不变。

**Architecture:** 新增独立的 `cocore.timing` 辅助模块，用可选回调把计时与业务计算解耦。Cocore pipeline 负责注入 stderr 输出器和判断阶段是否实际重建；Encode 与动作—视觉原型函数只报告自身内部步骤。Scan 通过 RelCore 的可选构建完成回调报告且不影响其他调用者。

**Tech Stack:** Python 3、`time.perf_counter()`、pytest、现有 Cocore/RelCore 原子缓存流水线。

## Global Constraints

- 日志固定为 `cocore_timing step=<step> seconds=<六位小数> status=completed`，写入 `stderr` 并 flush。
- 只报告成功完成的步骤；缓存命中阶段完全静默。
- 不增加配置或 CLI 参数，不改变 artifact schema、fingerprint、manifest 字段与 stdout。
- Cocore Bridge V2 通过现有 Cocore pipeline 自动继承行为。
- 保留工作树中与本任务无关的现有修改，不暂存或提交它们。

---

### Task 1: 统一计时接口与 Scan 构建完成回调

**Files:**
- Create: `cocore/timing.py`
- Create: `tests/test_cocore_timing.py`
- Modify: `relcore/pipeline.py:205-279`
- Modify: `tests/test_relcore_pipeline.py:339-350`
- Modify: `cocore/pipeline.py:349-359`

**Interfaces:**
- Produces: `TimingCallback = Callable[[str, float], None]`
- Produces: `emit_completed_timing(step: str, elapsed_seconds: float) -> None`
- Produces: `timed_step(step: str, callback: TimingCallback | None) -> ContextManager[None]`
- Extends: `relcore.pipeline.scan_stage(..., on_built: Callable[[], None] | None = None)`

- [ ] **Step 1: 为计时格式、成功/失败语义写失败测试**

```python
def test_emit_completed_timing_writes_stable_stderr(capsys):
    emit_completed_timing("graph.prototypes", 1.25)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "cocore_timing step=graph.prototypes seconds=1.250000 status=completed\n"
    )

def test_timed_step_only_reports_success():
    events = []
    with timed_step("encode.pca_fusion", lambda step, seconds: events.append((step, seconds))):
        pass
    assert events[0][0] == "encode.pca_fusion"
    assert events[0][1] >= 0.0
    with pytest.raises(RuntimeError):
        with timed_step("failed", lambda step, seconds: events.append((step, seconds))):
            raise RuntimeError("boom")
    assert [step for step, _ in events] == ["encode.pca_fusion"]
```

- [ ] **Step 2: 运行计时测试确认 RED**

Run: `pytest -q tests/test_cocore_timing.py`
Expected: FAIL，因为 `cocore.timing` 尚不存在。

- [ ] **Step 3: 实现最小计时模块**

```python
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import sys
import time

TimingCallback = Callable[[str, float], None]

def emit_completed_timing(step: str, elapsed_seconds: float) -> None:
    print(
        f"cocore_timing step={step} seconds={float(elapsed_seconds):.6f} status=completed",
        file=sys.stderr,
        flush=True,
    )

@contextmanager
def timed_step(step: str, callback: TimingCallback | None) -> Iterator[None]:
    if callback is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    except Exception:
        raise
    else:
        callback(step, time.perf_counter() - started)
```

- [ ] **Step 4: 为 RelCore Scan 的 `on_built` 写失败测试**

在现有 synthetic adapter/config 上调用两次 `scan_stage(config, on_built=...)`，断言第一次回调一次、第二次缓存命中不再回调；原四元组返回值保持不变。

- [ ] **Step 5: 实现 Scan 回调并接入 Cocore**

将 RelCore 中的发布调用改为：

```python
built = publish_stage(...)
if built and on_built is not None:
    on_built()
```

Cocore wrapper 在调用前记录 `perf_counter()`，传入闭包，仅在 RelCore 新建并发布 scan 后调用 `emit_completed_timing("scan", ...)`。

- [ ] **Step 6: 运行聚焦测试确认 GREEN**

Run: `pytest -q tests/test_cocore_timing.py tests/test_relcore_pipeline.py -k 'timing or scan'`
Expected: PASS。

- [ ] **Step 7: 提交基础设施**

```bash
git add cocore/timing.py tests/test_cocore_timing.py relcore/pipeline.py tests/test_relcore_pipeline.py cocore/pipeline.py
git commit -m "Add Cocore stage timing infrastructure"
```

### Task 2: Encode 子步骤与顶层阶段计时

**Files:**
- Modify: `cocore/encoding.py:393-585`
- Modify: `cocore/pipeline.py:362-487`
- Modify: `tests/test_cocore_encoding.py`
- Modify: `tests/test_cocore_pipeline.py:148-298`

**Interfaces:**
- Consumes: `TimingCallback`, `timed_step`, `emit_completed_timing`
- Extends: `encode_cocore_dataset(..., timing_callback: TimingCallback | None = None)`
- Produces steps: `encode.numeric_normalization`, `encode.visual_cache`, `encode.pca_fusion`, `encode`

- [ ] **Step 1: 写 Encode 步骤顺序与失败语义测试**

向 `encode_cocore_dataset` 传入收集事件的回调，断言正常路径步骤严格为：

```python
[
    "encode.numeric_normalization",
    "encode.visual_cache",
    "encode.pca_fusion",
]
```

对中断视觉编码器断言仅已完成的 `encode.numeric_normalization` 被报告，`encode.visual_cache` 与 `encode.pca_fusion` 不报告。

- [ ] **Step 2: 运行 Encode 测试确认 RED**

Run: `pytest -q tests/test_cocore_encoding.py -k timing`
Expected: FAIL，因为函数尚不接受 `timing_callback`。

- [ ] **Step 3: 用显式成功边界包裹三个 Encode 子步骤**

在数值 episode 收集与 normalizer fit、逐帧视觉编码/cache/半段特征、PCA 与最终融合三段分别使用 `timed_step`。默认回调为 `None`，现有直接调用保持静默。

- [ ] **Step 4: 接入 pipeline 顶层计时和缓存判断**

Pipeline 调用 Encode 时传入 `emit_completed_timing`；对 `publish_stage` 保存布尔结果：

```python
stage_started = time.perf_counter()
built = publish_stage(...)
if built:
    emit_completed_timing("encode", time.perf_counter() - stage_started)
```

顶层计时从 Encode 发布调用前开始，包含 artifact 写入和原子发布，不包含上游 scan。

- [ ] **Step 5: 测试新建、失败和缓存命中**

首次 `encode_stage` 断言 stderr 含三个子步骤和 `encode`；中断编码断言无 `encode.visual_cache`、`encode.pca_fusion`、`encode`；同配置第二次运行断言 stderr 不含 `step=encode`。

- [ ] **Step 6: 运行 Encode 测试确认 GREEN**

Run: `pytest -q tests/test_cocore_timing.py tests/test_cocore_encoding.py tests/test_cocore_pipeline.py -k 'timing or encode'`
Expected: PASS。

- [ ] **Step 7: 提交 Encode 计时**

```bash
git add cocore/encoding.py cocore/pipeline.py tests/test_cocore_encoding.py tests/test_cocore_pipeline.py
git commit -m "Report Cocore encode step timings"
```

### Task 3: 动作—视觉原型与 Graph 计时

**Files:**
- Modify: `cocore/prototypes.py:481-843`
- Modify: `cocore/pipeline.py:490-634`
- Modify: `tests/test_cocore_prototypes.py`
- Modify: `tests/test_cocore_pipeline.py`

**Interfaces:**
- Consumes: `TimingCallback`, `timed_step`, `emit_completed_timing`
- Extends: `build_hierarchical_motion_prototypes(..., timing_callback: TimingCallback | None = None)`
- Produces prototype steps: `graph.prototypes.action_scan`, `graph.prototypes.kmeans`, `graph.prototypes.center_statistics`, `graph.prototypes.candidate_assignment`
- Produces graph steps: `graph.reliability`, `graph.prototypes`, `graph.sparse_graph`, `graph`

- [ ] **Step 1: 写原型步骤聚合测试**

使用现有 `_TrajectoryPrototypeAdapter` 和 frame cache，传入事件回调，断言四个原型步骤按顺序各出现一次；`max_iter` 大于 1 时 `graph.prototypes.kmeans` 仍只出现一次。

- [ ] **Step 2: 运行原型测试确认 RED**

Run: `pytest -q tests/test_cocore_prototypes.py -k timing`
Expected: FAIL，因为原型构建函数尚不接受回调。

- [ ] **Step 3: 包裹原型的四个成功边界**

- `action_scan`：episode action 分类、原始计数、候选半段标签、catalog 与模型容量初始化。
- `kmeans`：完整 `for epoch in range(max_iter)` 循环，汇总为一条日志。
- `center_statistics`：ordering pass、最近中心距离、中心稳定排序和距离分位。
- `candidate_assignment`：父动作回退、候选半段最近叶分配和最终数组构造。

异常只阻止当前及外层未完成事件，不改变原异常。

- [ ] **Step 4: 写 Graph 聚合步骤失败测试并实现**

在 Graph build 中分别用 `timed_step` 包裹 `compute_reliability`、整个原型构建调用和 `build_graph`；向原型构建注入同一个输出回调。保存 `publish_stage` 返回值，仅新建成功后报告顶层 `graph`。

- [ ] **Step 5: 运行 Graph/原型测试确认 GREEN**

Run: `pytest -q tests/test_cocore_prototypes.py tests/test_cocore_pipeline.py -k 'timing or graph or prototype'`
Expected: PASS。

- [ ] **Step 6: 提交 Graph 计时**

```bash
git add cocore/prototypes.py cocore/pipeline.py tests/test_cocore_prototypes.py tests/test_cocore_pipeline.py
git commit -m "Report Cocore graph and prototype timings"
```

### Task 4: Select 计时、全链路缓存与 CLI 兼容性

**Files:**
- Modify: `cocore/pipeline.py:1068-1224`
- Modify: `tests/test_cocore_pipeline.py`
- Modify: `tests/test_cocore_cli.py`
- Modify: `tests/test_cocore_bridge_v2.py`

**Interfaces:**
- Consumes: `timed_step`, `emit_completed_timing`
- Produces steps: `select.context`, `select.coverage_seed`, `select.lazy_heap`, `select.export`, `select`

- [ ] **Step 1: 写 Select 步骤和全链路事件测试**

首次 synthetic `run_pipeline` 解析 stderr，断言以下步骤各出现一次且 `seconds` 是有限非负六位小数：

```python
EXPECTED_STEPS = {
    "scan", "encode.numeric_normalization", "encode.visual_cache",
    "encode.pca_fusion", "encode", "graph.reliability",
    "graph.prototypes.action_scan", "graph.prototypes.kmeans",
    "graph.prototypes.center_statistics",
    "graph.prototypes.candidate_assignment", "graph.prototypes",
    "graph.sparse_graph", "graph", "select.context",
    "select.coverage_seed", "select.lazy_heap", "select.export", "select",
}
```

- [ ] **Step 2: 运行全链路测试确认 RED**

Run: `pytest -q tests/test_cocore_pipeline.py -k timing`
Expected: FAIL，缺少 Select 事件。

- [ ] **Step 3: 实现 Select 四个子步骤与顶层计时**

- `select.context` 只包裹 `CocoreObjectiveContext` 构造。
- `select.coverage_seed` 包裹 `build_max_coverage_seed`。
- `select.lazy_heap` 包裹 selector 构造和 `selector.select(...)`。
- `select.export` 从节点/标签读取开始，覆盖 rows/report 构造及 selection artifact/manifest 写入。
- 保存 `publish_stage` 布尔返回值，仅新建并原子发布成功后报告 `select`。

- [ ] **Step 4: 验证缓存命中完全静默**

清空第一次运行的 capsys 后，以相同配置和失败视觉编码器再次执行 `run_pipeline`，断言 stderr 不含 `cocore_timing`，且返回同一 selection 目录。

- [ ] **Step 5: 验证 stdout 与 Bridge V2 兼容**

Cocore CLI 和 Bridge V2 CLI 的既有 stdout 精确断言保持不变；新增断言确认 timing 只出现在 stderr。Bridge 不增加任何复制的计时代码。

- [ ] **Step 6: 运行完整目标测试集**

Run: `pytest -q tests/test_cocore_timing.py tests/test_cocore_encoding.py tests/test_cocore_prototypes.py tests/test_cocore_core.py tests/test_cocore_pipeline.py tests/test_cocore_cli.py tests/test_cocore_bridge_v2.py tests/test_relcore_pipeline.py`
Expected: PASS。

- [ ] **Step 7: 检查格式与差异**

Run: `git diff --check`
Expected: 无输出且退出码为 0。确认 `git status --short` 中与本任务有关的文件之外，原有 README/Octo 修改仍未被暂存或改变。

- [ ] **Step 8: 提交 Select 与集成测试**

```bash
git add cocore/pipeline.py tests/test_cocore_pipeline.py tests/test_cocore_cli.py tests/test_cocore_bridge_v2.py
git commit -m "Complete Cocore step timing reports"
```

- [ ] **Step 9: 完成前验证**

使用 `superpowers:verification-before-completion` 重跑完整目标测试集并检查实际输出，再声明任务完成。
