# Quality Filter 惩罚系数与更新计数实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quality Filter 使用 `lambda=0.5`，所有筛选流程使用平方根更新阈值，同时保持 SQCN 的 `lambda=1.0`。

**Architecture:** 共享选择器参数化惩罚系数并统一计算 `ceil(100 + sqrt(selected_count - 100))`。Quality Filter 显式传入 `0.5`，并把系数与更新策略写入缓存指纹、manifest 和校验逻辑；SQCN 使用默认系数并同步更新 manifest。

**Tech Stack:** Python 3、NumPy、pytest、JSON/YAML 产物。

## Global Constraints

- 仅 Quality Filter 使用 `lambda=0.5`；SQCN 与 SQCN quality-only 保持 `1.0`。
- 所有筛选流程使用平方根更新阈值。
- 不新增 YAML/CLI 配置，不修改 KNN、RBF、候选集或 silent 晋升顺序。
- Quality Filter 包版本保持 `0.1.0`；旧缓存通过指纹变化失效。

---

### Task 1: 参数化共享选择器并统一更新阈值

**Files:**
- Modify: `segment_filter_core/selection.py`
- Modify: `sqcn/filtering/artifacts.py`
- Test: `tests/test_sqcn_filtering.py`

**Interfaces:**
- Produces: `select_diverse_fragments(..., *, seed: int | None = None, penalty_lambda: float = 1.0) -> SelectionResult`。
- Produces: `PROMOTION_MINIMUM_POLICY = "ceil(100 + sqrt(selected_count - 100))"`，供 manifest 复用。

- [x] 增加失败测试，覆盖显式 `0.5`、非法系数、平方根边界值、状态机更新计数和 SQCN manifest。
- [x] 确认测试因缺少参数、旧公式和旧 manifest 文案而失败。
- [x] 验证 `penalty_lambda` 为有限非负实数并传入 `_DiverseSelector`。
- [x] 使用实例系数计算 adjusted score，并用 `math.sqrt` 计算晋升阈值。
- [x] SQCN manifest 复用统一策略字符串，同时保留 `lambda=1.0`。
- [x] 运行 `pytest -q tests/test_sqcn_filtering.py`，确认完整测试通过。

### Task 2: Quality Filter 固定使用 0.5

**Files:**
- Modify: `quality_filter/pipeline.py`
- Test: `tests/test_quality_filter_pipeline.py`

**Interfaces:**
- Produces: `QUALITY_FILTER_PENALTY_LAMBDA = 0.5`。
- Consumes: `select_diverse_fragments(..., penalty_lambda=QUALITY_FILTER_PENALTY_LAMBDA)`。

- [x] 使用 200 个候选选择 102 个的 fixture 写入失败测试，并确保 fixture 产生非零惩罚。
- [x] 在筛选调用、manifest 和输出校验中使用 `0.5`。
- [x] 将惩罚系数与 `PROMOTION_MINIMUM_POLICY` 加入 Quality Filter 筛选指纹。
- [x] 校验器拒绝旧系数与旧更新策略，旧算法指纹不再命中缓存。
- [x] 跨流程测试仅保留 Quality 与 embedding 一致性。
- [x] 运行 `pytest -q tests/test_quality_filter_pipeline.py tests/test_quality_filter_cli.py`，确认完整测试通过。

### Task 3: 文档与最终验证

**Files:**
- Modify: `quality_filter/README.md`
- Modify: `sqcn/README.md`
- Modify: `docs/superpowers/specs/2026-08-06-quality-filter-penalty-coefficient-design.md`
- Modify: `docs/superpowers/plans/2026-08-06-quality-filter-penalty-coefficient.md`

- [x] README 与设计文档记录 `lambda=0.5`、SQCN 的 `lambda=1.0` 以及平方根更新阈值。
- [x] 运行 `pytest -q tests/test_sqcn_filtering.py tests/test_quality_filter_pipeline.py tests/test_quality_filter_cli.py tests/test_segment_filter_core.py`。
- [x] 检查 `git diff --check`、最终差异和工作区状态。
- [x] 提交文档并使用 verification-before-completion 重新运行最终验证。
