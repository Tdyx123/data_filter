# RelCore：LIBERO 层次化关系数据筛选

RelCore 从 LeRobot v2 LIBERO episode 中选择固定预算的 15 帧训练片段。数据读取方式
与 SQCN 一致：第一遍只读取 action/state 拟合全池统计，第二遍逐 episode 读取图像，
每帧只执行一次冻结 CLIP 编码。它不加载或训练任何 VLA/策略模型。

## 安装与输入

```bash
pip install -r relcore/requirements.txt
```

默认输入为 `/data/dwb/datasets/LIBERO_lerobot/libero90`，使用
`observation.images.image`、`observation.state` 和 `action`。每个 episode 必须在
`meta/episodes.jsonl`、`meta/tasks.jsonl` 与 Parquet `task_index` 中具有一致的单任务
映射。

## 运行

```bash
python -m relcore scan --config relcore/config_libero90.yaml
python -m relcore encode --config relcore/config_libero90.yaml
python -m relcore build-graph --config relcore/config_libero90.yaml
python -m relcore select --config relcore/config_libero90.yaml
python -m relcore run --config relcore/config_libero90.yaml
python -m relcore validate --output-dir outputs/relcore/libero90 \
  --config relcore/config_libero90.yaml
```

各阶段用数据、相关配置和上游 artifact 指纹恢复；已有不兼容阶段必须显式传
`--force`。兼容的上游阶段即使在 `--force` 下也会复用，因此只修改预算或分支参数
不会重新运行 CLIP。快速 CPU 检查使用：

```bash
python -m relcore run --config relcore/config_debug.yaml --force
```

## 数据与输出

窗口固定长度 15、stride 15，并与 SQCN 一样追加末尾对齐窗口；不足 15 帧的 episode
跳过。只有边界严格相邻的窗口才建立时序边，重叠的尾部窗口不会伪造时序关系。

最终输出位于配置的 `output.directory`：

- `selected_manifest.jsonl`：按选择顺序记录 LeRobot `episode_id` 与 inclusive
  `start_step/end_step`；
- `all_clips.parquet`：全部候选、可靠性、原型、选择状态和最终边际；
- `selection_report.json`：目标分解、任务额度和分支结果；
- `run_manifest.json`：数据、任务元数据、完整配置和阶段指纹；
- `scan/ encode/ graph/ select/`：可检查、可恢复的阶段 artifact，其中逐 episode
  CLIP 帧特征保存在 `encode/frame_features/`。

`selected_manifest.jsonl` 使用 `sample_id`、`episode_id`、`task_index`、`task_name`、
inclusive `start_step/end_step` 定位原始 LeRobot 数据。`all_clips.parquet` 区分插入
边际、最终增加收益与最终移除损失。

RelCore 不修改原始 LeRobot 数据，也不输出 HDF5 路径或伪造完整成功轨迹。
