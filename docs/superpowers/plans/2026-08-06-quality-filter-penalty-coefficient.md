# Quality Filter 惩罚系数 0.5 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让独立 Quality Filter 按 `adjusted_score = quality - 0.5 * knn_penalty` 重排，同时保持 SQCN 与 SQCN quality-only 的系数为 `1.0`。

**Architecture:** 共享选择器新增仅限关键字的 `penalty_lambda` 参数，默认使用现有 `PENALTY_LAMBDA = 1.0`。Quality Filter 通过专用常量显式传入 `0.5`，并把该值写入缓存指纹、输出清单和校验逻辑。

**Tech Stack:** Python 3、NumPy、pytest、YAML/JSON 产物清单。

## Global Constraints

- 仅独立 `quality_filter` 使用 `0.5`；SQCN 和 SQCN quality-only 保持 `1.0`。
- 不新增 YAML 或 CLI 配置项，不修改 KNN/RBF/候选集/silent 晋升算法。
- `penalty_lambda` 只接受有限且非负的实数。
- Quality Filter 包版本保持 `0.1.0`；缓存通过指纹中的系数失效。
- 保留工作区中现有 RelCore 未提交改动，不暂存、不修改。

---

### Task 1: 参数化共享多样性选择器

**Files:**
- Modify: `segment_filter_core/selection.py`
- Test: `tests/test_sqcn_filtering.py`

**Interfaces:**
- Consumes: 现有 `PENALTY_LAMBDA = 1.0` 与 `select_diverse_fragments(..., *, seed=None)`。
- Produces: `select_diverse_fragments(..., *, seed: int | None = None, penalty_lambda: float = PENALTY_LAMBDA) -> SelectionResult`；省略新参数时行为不变。

- [ ] **Step 1: 写入失败测试**

在已有 unit-lambda 用例旁新增显式缩放测试，并覆盖非法参数：

```python
def test_penalty_lambda_scales_adjusted_score():
    scores = np.concatenate(
        [
            np.linspace(1.0, 0.901, 100, dtype=np.float64),
            np.asarray([0.9, 0.8], dtype=np.float64),
        ]
    )
    embeddings = np.concatenate(
        [
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (101, 1)),
            np.asarray([[-1.0, 0.0]], dtype=np.float32),
        ],
        axis=0,
    )
    sample_ids = tuple(f"sample-{index:03d}" for index in range(len(scores)))

    result = select_diverse_fragments(
        scores,
        embeddings,
        sample_ids,
        target_size=102,
        seed=23,
        penalty_lambda=0.5,
    )

    assert result.selected_indices[-2:].tolist() == [101, 100]
    assert result.knn_penalties[-1] == pytest.approx(0.998)
    assert result.adjusted_scores[-1] == pytest.approx(0.401)


@pytest.mark.parametrize("penalty_lambda", [-0.1, float("nan"), float("inf"), True, "0.5"])
def test_penalty_lambda_rejects_invalid_values(penalty_lambda: object):
    scores = np.linspace(1.0, 0.0, 100, dtype=np.float64)
    embeddings = np.eye(100, dtype=np.float32)
    sample_ids = tuple(f"sample-{index:03d}" for index in range(len(scores)))

    with pytest.raises(ValueError, match="penalty_lambda"):
        select_diverse_fragments(
            scores,
            embeddings,
            sample_ids,
            target_size=100,
            penalty_lambda=penalty_lambda,
        )
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run: `pytest -q tests/test_sqcn_filtering.py -k 'penalty_lambda or unit_lambda'`

Expected: 新用例因 `select_diverse_fragments()` 不接受 `penalty_lambda` 而失败；现有 unit-lambda 用例通过。

- [ ] **Step 3: 写入最小实现**

在 `_DiverseSelector` 中保存系数并替代硬编码全局常量：

```python
class _DiverseSelector:
    def __init__(
        self,
        scores: np.ndarray,
        embeddings: np.ndarray,
        sample_ids: np.ndarray,
        *,
        seed: int,
        sigma_effective: float,
        penalty_lambda: float = PENALTY_LAMBDA,
        raw_order: np.ndarray | None = None,
        parameters: _AlgorithmParameters = _AlgorithmParameters(),
    ):
        self.penalty_lambda = penalty_lambda

    def _update_batch(self, target: np.ndarray, reference: np.ndarray) -> None:
        # 保留现有惩罚计算。
        self.adjusted[target] = (
            self.scores[target] - self.penalty_lambda * self.penalties[target]
        )
```

在公共函数入口验证并传递参数：

```python
def select_diverse_fragments(
    scores: np.ndarray,
    embeddings: np.ndarray,
    sample_ids: Sequence[str],
    target_size: int,
    *,
    seed: int | None = None,
    penalty_lambda: float = PENALTY_LAMBDA,
) -> SelectionResult:
    if isinstance(penalty_lambda, (bool, np.bool_)) or not isinstance(
        penalty_lambda,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("penalty_lambda must be a finite non-negative real number")
    penalty_lambda_value = float(penalty_lambda)
    if not np.isfinite(penalty_lambda_value) or penalty_lambda_value < 0.0:
        raise ValueError("penalty_lambda must be a finite non-negative real number")
    # 创建 _DiverseSelector 时传入 penalty_lambda=penalty_lambda_value。
```

- [ ] **Step 4: 运行共享选择器测试并确认通过**

Run: `pytest -q tests/test_sqcn_filtering.py`

Expected: 全部通过，现有默认系数 `1.0` 的排序和清单断言保持不变。

- [ ] **Step 5: 提交共享选择器变更**

```bash
git add segment_filter_core/selection.py tests/test_sqcn_filtering.py
git commit -m "feat: parameterize diversity penalty coefficient"
```

---

### Task 2: Quality Filter 固定使用 0.5

**Files:**
- Modify: `quality_filter/pipeline.py`
- Modify: `quality_filter/README.md`
- Test: `tests/test_quality_filter_pipeline.py`

**Interfaces:**
- Consumes: Task 1 的 `select_diverse_fragments(..., penalty_lambda=...)`。
- Produces: `QUALITY_FILTER_PENALTY_LAMBDA = 0.5`，供筛选、指纹、清单和验证共同使用。

- [ ] **Step 1: 写入失败的 Quality Filter 行为测试**

把 `test_filter_stage_uses_quality_and_writes_ranked_aligned_artifacts` 的筛选比例改为 `51.0`，使 200 个候选中选择 102 个并实际触发惩罚；新增以下断言：

```python
assert output == tmp_path / "quality-filter-output" / "filter" / "top51pct"
assert len(rows) == 102
assert embeddings.shape == (102, 141)
assert manifest["algorithm"]["target_size"] == 102
assert manifest["algorithm"]["lambda"] == 0.5
penalties = [float(row["knn_penalty"]) for row in rows]
assert any(penalty > 0.0 for penalty in penalties)
for row in rows:
    assert float(row["adjusted_score"]) == pytest.approx(
        float(row["quality"]) - 0.5 * float(row["knn_penalty"]),
        abs=2.0e-9,
    )
```

把上游一致性用例重命名为 `test_quality_filter_matches_sqcn_quality_only_quality_and_embeddings`，移除 SQCN quality-only 筛选调用及最终选择列表相等断言，同时删除不再使用的 `filter_sqcn_run` 导入。

- [ ] **Step 2: 运行测试并确认按预期失败**

Run: `pytest -q tests/test_quality_filter_pipeline.py -k 'ranked_aligned or matches_sqcn_quality_only'`

Expected: 清单仍记录 `1.0`，且产生惩罚的行仍按系数 `1.0` 计算，导致新断言失败。

- [ ] **Step 3: 写入 Quality Filter 最小实现与文档**

在 `quality_filter/pipeline.py` 中停止导入共享的 `PENALTY_LAMBDA`，并定义：

```python
QUALITY_FILTER_PENALTY_LAMBDA = 0.5
```

筛选调用显式传参：

```python
result = select_diverse_fragments(
    scores,
    embeddings,
    sample_ids,
    target_size,
    seed=requested_seed,
    penalty_lambda=QUALITY_FILTER_PENALTY_LAMBDA,
)
```

筛选指纹增加 `"penalty_lambda": QUALITY_FILTER_PENALTY_LAMBDA`，清单的 `algorithm.lambda` 使用该常量；`_validate_filter_output` 在检查 `score_column` 后增加：

```python
if algorithm.get("lambda") != QUALITY_FILTER_PENALTY_LAMBDA:
    raise ValueError("quality filter penalty lambda must be 0.5")
```

将 `quality_filter/README.md` 的固定算法说明从 ``lambda=1`` 更新为 ``lambda=0.5``。

- [ ] **Step 4: 运行 Quality Filter 测试并确认通过**

Run: `pytest -q tests/test_quality_filter_pipeline.py tests/test_quality_filter_cli.py`

Expected: 全部通过；输出清单为 `0.5`，实际调整分数遵循新公式，上游 Quality 与 embedding 仍与 SQCN quality-only 一致。

- [ ] **Step 5: 运行完整相关回归测试**

Run: `pytest -q tests/test_sqcn_filtering.py tests/test_quality_filter_pipeline.py tests/test_quality_filter_cli.py tests/test_segment_filter_core.py`

Expected: 全部通过，SQCN 默认系数及其清单仍为 `1.0`。

- [ ] **Step 6: 提交 Quality Filter 变更和实施计划**

```bash
git add quality_filter/pipeline.py quality_filter/README.md tests/test_quality_filter_pipeline.py docs/superpowers/plans/2026-08-06-quality-filter-penalty-coefficient.md
git commit -m "feat: use half penalty in quality filter"
```
