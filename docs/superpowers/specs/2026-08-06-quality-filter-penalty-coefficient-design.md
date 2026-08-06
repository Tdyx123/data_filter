# Quality Filter 惩罚系数 0.5 设计

## 目标

独立 `quality_filter` 在多样性重排中使用固定惩罚系数 `0.5`：

```text
adjusted_score = quality - 0.5 * knn_penalty
```

SQCN 及 SQCN quality-only 模式继续使用现有系数 `1.0`，不新增 YAML 配置项。

## 设计

- 为 `segment_filter_core.selection.select_diverse_fragments` 增加仅限关键字的
  `penalty_lambda` 参数，默认值保持 `1.0`；选择器实例保存该值并用于调整分数。
- 参数必须是有限且非负的实数；非法值在开始选择前抛出 `ValueError`。
- `quality_filter.pipeline.filter_stage` 使用模块级常量 `0.5` 显式调用共享选择器，
  并在筛选指纹和 `filter_manifest.json` 的 `algorithm.lambda` 中记录同一值。
- Quality Filter 输出校验要求清单中的 `algorithm.lambda` 等于 `0.5`，避免旧算法
  产物被误认为当前产物。系数加入指纹后，已有系数为 `1.0` 的筛选缓存不会命中；
  按现有覆盖策略，用户需使用 `--force` 重新生成同名筛选目录。
- SQCN 调用不传新参数，继续走默认值 `1.0`，其清单和文档保持不变。

## 测试与文档

- 在共享选择器测试中覆盖默认 `1.0`、显式 `0.5` 的调整分数，以及负数、NaN、
  无穷大等非法参数。
- 在 Quality Filter 流程测试中验证清单系数为 `0.5`，并对实际产生惩罚的选中项
  验证 `adjusted_score = quality - 0.5 * knn_penalty`。
- 调整原先要求 Quality Filter 与 SQCN quality-only 最终选择完全一致的测试：只保留
  Quality 分数和 embedding 的上游一致性，因为两者的惩罚系数现在有意不同。
- 更新 Quality Filter README，将固定系数从 `lambda=1` 改为 `lambda=0.5`。

## 非目标

- 不改变 KNN、RBF、候选集或 silent 晋升算法。
- 不改变 SQCN 或 SQCN quality-only 的惩罚系数。
- 不把惩罚系数暴露为命令行或 YAML 配置。
- 不修改 Quality Filter 包版本号；筛选指纹直接包含系数并负责缓存失效。
