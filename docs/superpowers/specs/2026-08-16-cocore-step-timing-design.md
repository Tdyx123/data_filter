# Cocore 分步骤耗时日志设计

## 目标

为 Cocore 的主要流水线阶段及关键子步骤增加完成耗时日志，让长时间运行能够直接看出时间消耗位置。日志只写入 `stderr`，不得改变 Cocore 与 Cocore Bridge V2 现有的 `stdout` 输出契约。

## 日志契约

每个成功完成的步骤输出一行稳定的键值日志：

```text
cocore_timing step=<step> seconds=<浮点秒数> status=completed
```

- `seconds` 使用 `time.perf_counter()` 测量并固定输出六位小数。
- 输出到 `stderr` 并立即 flush。
- 仅在步骤正常返回后输出；抛出异常的步骤及其未完成的外层步骤不输出完成事件，已经完成的前置步骤保留日志。
- 缓存命中的阶段不输出计时日志；使用 `--force` 实际重建时正常输出。
- 不增加配置项或 CLI 参数，不改变 artifact schema、fingerprint 或既有 manifest 字段。

## 步骤边界

顶层阶段使用以下名称：

- `scan`
- `encode`
- `graph`
- `select`

关键子步骤使用以下名称：

- Encode：`encode.numeric_normalization`、`encode.visual_cache`、`encode.pca_fusion`
- Graph：`graph.reliability`、`graph.prototypes`、`graph.sparse_graph`
- 动作—视觉原型：`graph.prototypes.action_scan`、`graph.prototypes.kmeans`、`graph.prototypes.center_statistics`、`graph.prototypes.candidate_assignment`
- Select：`select.context`、`select.coverage_seed`、`select.lazy_heap`、`select.export`

`graph.prototypes.kmeans` 汇总全部 epoch，不逐 epoch 输出。顶层阶段总耗时包含 artifact 写入和原子发布；子步骤只覆盖对应计算。调用 `run` 时，新建的各层阶段按实际完成顺序各输出一次，缓存阶段完全静默。

## 实现设计

新增一个 Cocore 内部计时模块，提供：

- `TimingCallback`：接收 `(step, elapsed_seconds)` 的回调类型。
- stderr 完成事件输出函数，集中负责格式化和 flush。
- 只在代码块成功结束时调用回调的计时辅助接口。

`encode_cocore_dataset` 与 `build_hierarchical_motion_prototypes` 增加可选计时回调；默认 `None`，保持直接调用与单元测试静默。Cocore pipeline 调用它们时注入统一 stderr 输出函数，并在现有计算边界报告子步骤。

`encode_stage`、`graph_stage` 和 `select_stage` 使用 `publish_stage()` 的布尔返回值判断是否实际构建；只有返回 `True` 才输出顶层耗时。Scan 复用 RelCore 实现，因此给 `relcore.pipeline.scan_stage` 增加一个默认关闭的“新构建完成”回调；Cocore 通过该回调输出 `scan`，其他 RelCore 调用者行为不变。Cocore Bridge V2 继续委托 Cocore pipeline，无需复制计时代码即可获得相同行为。

## 错误与兼容性

- 计时输出不得吞掉、包装或改变原异常。
- 回调默认关闭，新增参数均为可选关键字参数，现有调用保持兼容。
- 日志不参与缓存 fingerprint，启用计时不会导致缓存失效。
- `stdout` 继续只输出原有 `cocore_output=...` 或 JSON 结果；Bridge V2 同样保持不变。

## 测试与验收

- 单元测试验证日志格式、六位小数、stderr、flush，以及异常路径不输出未完成步骤。
- Synthetic pipeline 首次运行验证所有顶层步骤和关键子步骤各有一条完成日志，秒数有限且非负。
- 同一配置再次运行验证缓存命中的阶段没有计时日志。
- 注入视觉编码失败，验证失败的 Encode 子步骤和 `encode` 顶层不输出完成事件，异常仍原样传播。
- 验证 Cocore 与 Bridge V2 既有 stdout 精确匹配测试继续通过。
- 运行 Cocore timing/core/prototype/encoding/pipeline/CLI 及 Bridge V2 相关测试，确认计时改动不影响 artifact、选择结果和缓存行为。
