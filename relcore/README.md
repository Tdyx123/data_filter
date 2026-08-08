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
python -m relcore validate --output-dir outputs/relcore/libero90/select-15 \
  --config relcore/config_libero90.yaml
```

默认启用全部四项可靠性指标，因此 graph 和最终结果分别位于 `graph-15/` 与
`select-15/`。`build-graph`、`select` 和 `run` 可通过逗号分隔参数选择指标：

```bash
python -m relcore run --config relcore/config_libero90.yaml \
  --reliability-metrics progress,non_noop
```

上述结果使用掩码 5，位于 `graph-5/` 和 `select-5/`，并与默认结果共享同一份
`scan/` 和聚合后的 `encode/` 阶段产物。

`select` 和 `run` 可通过 `--selection-ratio` 在调用时设置筛选比例，取值范围为
`(0, 1]`。命令行比例会覆盖配置中的 `selection.ratio`，并忽略固定的
`selection.budget`；未传该参数时仍完全使用配置文件中的比例或固定预算。

显式设置比例会在指标掩码后追加比例，因此不同指标和筛选比例无需 `--force` 即可并存。
例如默认四项指标下：

```bash
python -m relcore select --config relcore/config_libero90.yaml \
  --selection-ratio 0.10
python -m relcore select --config relcore/config_libero90.yaml \
  --selection-ratio 0.20
```

上述结果分别位于 `outputs/relcore/libero90/select-15-top10pct/` 和
`outputs/relcore/libero90/select-15-top20pct/`；12.5% 使用
`select-15-top12p5pct/`。
比例目录包含 `manifest.json`、`selected_manifest.jsonl`、`all_clips.parquet` 和
`selection_report.json`，同时保存 `run_manifest.json`。命令打印的
`relcore_output=` 直接指向本次选择目录。同一指标掩码和比例目录的其他配置发生
不兼容变化时仍需传 `--force`。

`run --selection-ratio` 使用相同的隔离布局。例如筛选 20% 的候选片段：

```bash
python -m relcore run --config relcore/config_libero90.yaml \
  --selection-ratio 0.20 --force
```

`validate` 接受具体的 `select-<mask>/` 或 `select-<mask>-topXpct/` 目录，并从其中的
manifest 定位共享输出根目录下的 scan、encode 和匹配的 graph。

## 可靠性指标

可靠性指标是独立的 CLI/Python API 参数，不属于 YAML quality 配置。未传
`--reliability-metrics` 时默认启用全部四项：

- `support`：关系嵌入的 KNN 邻域支持度；
- `progress`：末端状态、夹爪状态和视觉变化组成的进展分数；
- `smoothness`：根据动作 jerk 得到的平滑度；
- `non_noop`：有效动作比例，即 `1 - noop_ratio`。

列表必须非空、不能重复且只能包含上述名称；书写顺序不会影响结果，解析后按上述固定
顺序保存。每项沿用固定指数：`support`、`progress`、`non_noop` 为 `0.5`，
`smoothness` 为 `0.25`。禁用某项时直接省略对应因子，不重新归一化剩余指数，因此
启用项较少时合成可靠性可能整体升高。例如只使用任务进展和有效动作：

```bash
python -m relcore run --config relcore/config_libero90.yaml \
  --reliability-metrics progress,non_noop
```

固定位序为 `[support, progress, smoothness, non_noop]`，对应二进制权重
`[8, 4, 2, 1]`。例如仅 `non_noop` 的后缀为 1，`progress,non_noop` 为 5，全部启用为
15。旧配置若仍包含 `quality.reliability_metrics` 会明确报错并提示迁移到 CLI 参数。

四个原始分量始终计算并保存在 `graph-<mask>/nodes.npz`，但只有参数子集合成的单一
`reliability` 用于建图、目标函数、种子和候选选择。实际生效列表写入
`selection_report.json`。不同指标集合写入不同 graph/select 目录，无需 `--force`；
scan 和聚合后的 encode 阶段仍可复用。

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
  --output-dir outputs/relcore/bridge_orig_1.0.0_top10pct/select-15 \
  --config relcore/config_bridge.yaml
```

完整运行前可在独立目录做约 100 条 episode 的编码 smoke test；日志中的
`relcore_encode completed=... remaining=...` 会显示当前进度：

```bash
CUDA_VISIBLE_DEVICES=0 python -m relcore encode \
  --config relcore/config_bridge.yaml --max-episodes 100 \
  --output-dir outputs/relcore/bridge_smoke100
```

可参与 CLIP 的 38,123 条 episode 共 1,299,118 帧；完整运行应选择 10,663 个片段。
CLIP 帧特征只在处理当前 episode 时驻留内存，生成关系特征后立即释放，不会写入
`.relcore-cache` 或 `encode/frame_features/`。encode 阶段仍以聚合产物原子发布；如果
编码中断，临时产物会被清理，重跑时会重新编码全部可用 episode。

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
  --output-dir /data/dwb/libero_filter/relcore
```

该命令的筛选结果位于 `/data/dwb/libero_filter/relcore/select-15-top30pct/`。

在尚未包含该启动保护的旧版本上，可用下面的等价命令临时规避；四个变量必须在
启动 Python 前设置：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 python -m relcore select \
  --config relcore/config_libero90.yaml \
  --selection-ratio 0.30 \
  --output-dir /data/dwb/libero_filter/relcore
```

## 数据与输出

窗口固定长度 15、stride 15，并与 SQCN 一样追加末尾对齐窗口；不足 15 帧的 episode
跳过。只有边界严格相邻的窗口才建立时序边，重叠的尾部窗口不会伪造时序关系。

共享输出根目录始终保留 `scan/`、`encode/` 和一个或多个 `graph-<mask>/`。最终结果
位于 `select-<mask>/`；显式传入 `--selection-ratio` 时位于
`select-<mask>-topXpct/`。每个选择目录包含：

- `selected_manifest.jsonl`：按选择顺序记录 LeRobot `episode_id` 与 inclusive
  `start_step/end_step`；
- `all_clips.parquet`：全部候选、可靠性、原型、选择状态和最终边际；
- `selection_report.json`：目标分解、实际任务计数、可选任务额度和分支结果；
- `run_manifest.json`：可靠性指标、掩码、阶段目录和阶段指纹；
- `resolved_config.yaml`、`environment.json`：`run` 写入的配置与运行环境。

`encode/` 只保存聚合后的 embeddings、关系特征、状态/动作序列、视觉进展、归一化与
投影参数，不保存逐帧 CLIP 特征。输出根目录不再发布 `selected_manifest.jsonl` 等
“最后一次运行”副本，因此不同指标组合不会互相覆盖。

`selected_manifest.jsonl` 使用 `sample_id`、`episode_id`、`task_index`、`task_name`、
inclusive `start_step/end_step` 定位原始 LeRobot 数据。`all_clips.parquet` 区分插入
边际、最终增加收益与最终移除损失。

RelCore 不修改原始 LeRobot 数据，也不输出 HDF5 路径或伪造完整成功轨迹。
