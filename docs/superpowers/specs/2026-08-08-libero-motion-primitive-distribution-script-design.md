# LIBERO 运动原语分布脚本设计

## 目标

新增一个可直接运行的 Python 脚本，读取完整的 LeRobot v2 数据集，按 episode
生成 LIBERO ECoT 运动原语标签，并把整个数据集的类别数量和占比汇总到指定输出
目录中的 CSV 文件。

## 命令行接口

脚本路径为 `scripts/generate_libero_motion_primitive_distribution.py`，提供以下参数：

- `--dataset-root PATH`：必填，LeRobot v2 数据集根目录。
- `--output-dir PATH`：必填，CSV 输出目录；目录不存在时自动创建。
- `--horizon INT`：可选，默认 `4`，有效范围 `3` 到 `8`。
- `--threshold FLOAT`：可选，默认 `0.03`，必须为有限非负数。
- `--tail-strategy {truncate,clip,pad_last}`：可选，默认 `truncate`。

固定输出文件名为 `motion_primitive_distribution.csv`。命令成功后在终端打印数据集
episode 数、生成的标签总数和 CSV 绝对路径；失败时输出明确错误并返回非零状态。

## 数据处理

脚本复用 `trajectory_data.LeRobotDatasetAdapter`，配置
`use_images=False`，并将向量观测限定为 `observation.state`。适配器负责验证 LeRobot
v2 元数据、episode 记录和 Parquet 数据；脚本不读取或解码图像。

每个 episode 独立调用 `generate_motion_primitives()`，使用
`make_libero_config()` 创建的 LIBERO 坐标映射。标签不会跨 episode 边界生成。
所有 episode 的标签使用计数器汇总，再调用现有
`compute_primitive_statistics()` 计算数量和占比。占比分母是配置的尾部策略实际生成的
全部标签数。

CSV 使用 UTF-8 和稳定表头 `primitive,count,proportion`，按数量降序、类别名称升序
排列。占比写为 Python 浮点数的稳定十进制表示；空数据集或没有生成任何标签时仅写
表头，并报告标签总数为零。文件通过同目录临时文件和原子替换写入，避免失败时留下
半成品。

## 测试与文档

新增脚本级测试，使用临时的最小 LeRobot v2 数据集验证：

- 多个 episode 只在各自边界内分类，并正确汇总 `primitive/count/proportion`。
- CSV 排序、表头、输出目录创建和终端摘要稳定。
- `horizon`、`threshold` 和 `tail-strategy` 参数传入核心配置。
- 元数据、状态字段或 Parquet 数据无效时返回非零状态且不产生半成品 CSV。
- 没有可生成标签的短轨迹得到仅含表头的 CSV。

更新 `libero_motion_primitives/README.md`，加入完整命令示例和 CSV 格式说明。实现保持
现有 `libero_motion_primitives` 公共 API 不变，不引入新依赖。

## 实施约束

- 直接在当前 `main` 分支修改，不创建 worktree 或功能分支。
- 按 TDD 顺序先提交会因脚本缺失而失败的测试，再实现最小代码并运行回归测试。
- 不加载图像，不跨 episode 计算，不修改现有分类规则。
