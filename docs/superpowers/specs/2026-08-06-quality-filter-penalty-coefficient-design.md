# Quality Filter 惩罚系数与更新计数设计

## 目标

独立 `quality_filter` 使用固定惩罚系数 `0.5`：

```text
adjusted_score = quality - 0.5 * knn_penalty
```

SQCN 及 SQCN quality-only 继续使用系数 `1.0`。所有筛选流程的 silent 晋升更新
阈值统一为：

```text
ceil(100 + sqrt(selected_count - 100))
```

不新增 YAML 或 CLI 配置项。

## 设计

- `segment_filter_core.selection.select_diverse_fragments` 增加仅限关键字的
  `penalty_lambda` 参数，默认值为 `1.0`；仅接受有限且非负的实数。
- 共享选择器保存该系数并用于 adjusted score，同时把更新阈值的增长项从 `log2`
  统一改为 `sqrt`。
- `quality_filter.pipeline.filter_stage` 使用模块级常量 `0.5` 显式调用共享选择器；
  筛选指纹同时包含惩罚系数和更新阈值策略。
- Quality Filter 与 SQCN 的 manifest 都记录统一的平方根阈值；Quality Filter 输出
  校验额外要求 `algorithm.lambda=0.5` 以及正确的阈值策略。
- 旧 Quality Filter 筛选缓存不会命中新指纹；同名目录按现有行为需要 `--force`
  重新生成。SQCN 新运行自动使用平方根阈值，既有产物不原地迁移。

## 测试与文档

- 共享选择器覆盖默认 `1.0`、显式 `0.5`、非法系数、平方根边界值及完整状态机计数。
- Quality Filter 使用能产生非零 RBF 惩罚的真实流水线 fixture，验证输出逐行满足
  `adjusted_score = quality - 0.5 * knn_penalty`。
- 验证 Quality Filter manifest、输出校验和旧缓存失效；验证 SQCN manifest 保持
  `lambda=1.0` 并记录平方根阈值。
- 跨流程测试只要求 Quality 分数和 embedding 一致，不要求不同惩罚系数下的最终
  选择完全一致。

## 非目标

- 不改变 KNN、RBF、候选集或 silent 晋升顺序。
- 不改变 SQCN 或 SQCN quality-only 的惩罚系数。
- 不把惩罚系数或更新阈值暴露为配置。
- 不修改包版本号。
