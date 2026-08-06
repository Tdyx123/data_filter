# RelCore：LeRobot 层次化关系数据筛选

RelCore 从 LeRobot v2 episode 中选择固定预算的 15 帧训练片段。数据读取方式
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

`select` 和 `run` 可通过 `--selection-ratio` 在调用时设置筛选比例，取值范围为
`(0, 1]`。例如筛选 20% 的候选片段：

```bash
python -m relcore run --config relcore/config_libero90.yaml \
  --selection-ratio 0.20 --force
```

命令行比例会覆盖配置中的 `selection.ratio`，并忽略固定的 `selection.budget`；未传
该参数时仍完全使用配置文件中的比例或固定预算。已有输出与新比例不兼容时需要传
`--force`。

各阶段用数据、相关配置和上游 artifact 指纹恢复；已有不兼容阶段必须显式传
`--force`。兼容的上游阶段即使在 `--force` 下也会复用，因此只修改预算或分支参数
不会重新运行 CLIP。快速 CPU 检查使用：

```bash
python -m relcore run --config relcore/config_debug.yaml --force
```

## BridgeData V2

`config_bridge.yaml` 直接读取
`/data/dwb/datasets/bridge_orig_1.0.0_lerobo/`，只使用
`observation.images.image_0`、`observation.state` 和 `action`。数据集中任务名为空的
14,532 条 episode 会在 `max_episodes` 生效前排除；其视频、state 和 action 不参与
归一化、CLIP 或候选片段构建。其余任务保留原始 `task_index` 和 `task_name`，但 Top
10% 筛选使用 `quota_mode: none`，不施加逐任务硬配额。

scan 不需要 GPU，可先核对输入规模：

```bash
python -m relcore scan --config relcore/config_bridge.yaml
```

预期保留 38,660 条 episode、1,305,714 帧；其中 537 条短于 15 帧，最终生成
106,625 个候选片段。正式编码固定使用单张 CUDA GPU；通过
`CUDA_VISIBLE_DEVICES` 选择物理卡：

```bash
CUDA_VISIBLE_DEVICES=0 python -m relcore encode --config relcore/config_bridge.yaml
CUDA_VISIBLE_DEVICES=0 python -m relcore run --config relcore/config_bridge.yaml
python -m relcore validate \
  --output-dir outputs/relcore/bridge_orig_1.0.0_top10pct \
  --config relcore/config_bridge.yaml
```

完整运行前可在独立目录做约 100 条 episode 的中断恢复 smoke test；中断后重跑同一
命令，日志中的 `reused` 应显示已完成的缓存前缀：

```bash
CUDA_VISIBLE_DEVICES=0 python -m relcore encode \
  --config relcore/config_bridge.yaml --max-episodes 100 \
  --output-dir outputs/relcore/bridge_smoke100
```

可参与 CLIP 的 38,123 条 episode 共 1,299,118 帧；完整运行应选择 10,663 个片段。
CLIP 帧特征逐 episode 原子写入
`OUTPUT/.relcore-cache/frame_features/<fingerprint>/`。中断后直接重跑相同命令即可从
第一条缺失或损坏的 episode 继续，已验证缓存不会重新运行 CLIP。缓存默认保留，
以便 relation/PCA 失败或参数调整时复用；完整输出通过 `validate` 后，如确定不再重建
encode，可手工删除该输出目录下的 `.relcore-cache`。已发布结果的验证不依赖隐藏缓存。

## 原生线程池

RelCore 会通过共享的 `trajectory_data` 启动保护，在导入 NumPy/SciPy 前把
OpenBLAS、OpenMP、MKL 和 NumExpr 统一限制为每进程 1 个线程。这可避免高核数
服务器上的 scikit-learn/FAISS 图构建与数据 worker 形成嵌套并行。该限制与
`runtime.num_workers` 相互独立，因此不需要为了规避 OpenBLAS 崩溃而关闭数据并行。

如需调高原生计算并行度，使用范围为 1–64 的共享环境变量：

```bash
TRAJECTORY_DATA_NUM_THREADS=4 python -m relcore select \
  --config relcore/config_libero90.yaml \
  --selection-ratio 0.30 \
  --output-dir /data/dwb/libero_filter/relcore_top30pct
```

在尚未包含该启动保护的旧版本上，可用下面的等价命令临时规避；四个变量必须在
启动 Python 前设置：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 python -m relcore select \
  --config relcore/config_libero90.yaml \
  --selection-ratio 0.30 \
  --output-dir /data/dwb/libero_filter/relcore_top30pct
```

## 数据与输出

窗口固定长度 15、stride 15，并与 SQCN 一样追加末尾对齐窗口；不足 15 帧的 episode
跳过。只有边界严格相邻的窗口才建立时序边，重叠的尾部窗口不会伪造时序关系。

最终输出位于配置的 `output.directory`：

- `selected_manifest.jsonl`：按选择顺序记录 LeRobot `episode_id` 与 inclusive
  `start_step/end_step`；
- `all_clips.parquet`：全部候选、可靠性、原型、选择状态和最终边际；
- `selection_report.json`：目标分解、实际任务计数、可选任务额度和分支结果；
- `run_manifest.json`：数据、任务元数据、完整配置和阶段指纹；
- `scan/ encode/ graph/ select/`：可检查、可恢复的阶段 artifact，其中逐 episode
  CLIP 帧特征保存在 `encode/frame_features/`。

`selected_manifest.jsonl` 使用 `sample_id`、`episode_id`、`task_index`、`task_name`、
inclusive `start_step/end_step` 定位原始 LeRobot 数据。`all_clips.parquet` 区分插入
边际、最终增加收益与最终移除损失。

RelCore 不修改原始 LeRobot 数据，也不输出 HDF5 路径或伪造完整成功轨迹。
