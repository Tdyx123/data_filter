# Cocore BridgeV2

`cocore_bridge_v2` 是仓库内的 BridgeData V2 专用 Cocore 命令包。它不复制
Cocore 的编码、运动原语、关系目标或选择算法，而是固定 Bridge 数据契约后
调用现有 `cocore.pipeline`。输出仍是 Cocore artifact，可直接交给现有训练入口和
`cocore` 校验器消费。

## 固定数据契约

- 默认数据路径：`/data/dwb/datasets/bridge_orig_1.0.0_lerobot`；
- LeRobot `v2.0`、WidowX、5 Hz；
- 只读取 `observation.images.image_0`、8 维 `observation.state` 和 7 维 `action`；
- 排除任务名为空的 episode，再应用 `--max-episodes`；
- 固定使用 7 帧近似均匀候选和 Cocore schema 10 两级动作原型；原型学习在完整轨迹
  上使用首尾覆盖、起点间隔最大为 2 的四帧窗口，只用精确保留动作训练硬视觉桶；
- 固定 `prototypes.profile: bridge_v2`，CLI 不提供 profile 覆盖；四帧窗口始终使用
  `state[t] → state[t+3]`；
- xyz 严格阈值为 `0.03 m`，roll/pitch 为 `0.18 rad`，yaw 为 `0.24 rad`，gripper
  为 `0.20`；等于阈值不激活动作；
- roll 与 yaw 使用 `[-π, π)` 最短角差。原子顺序固定为平移、
  `roll positive/negative`、pitch tilt、yaw rotate、gripper，再合成为单一复合标签；
- 设完整轨迹窗口数为 `W`，非 stop 动作桶保留条件固定为
  `count >= max(400, ceil(0.005 * W))`；
- 令 `M_a` 为单动作桶的训练窗口数，视觉中心数固定为
  `min(M_a, min(30, max(10, floor(4 * log2(M_a) - 30))))`；保留的非 stop 动作使用
  10～30 个中心；
- 默认 `prototypes.use_stop_bucket: true`，保留 Cocore 的 stop 桶与回退行为；graph
  相关命令可用 `--no-use-stop-bucket` 关闭；
- 可靠性默认使用 `sqrt(support * progress)`；graph 相关命令可用
  `--reliability-metrics support` 切换为 `sqrt(support)`；两种结果均应用
  `quality.min_reliability` 截断；
- 不超过 65,536 个训练窗口的动作桶使用完整 KMeans，以
  `prototypes.num_threads: 4` 并行且每个模型使用 1 个 OpenMP 线程；超过阈值的桶使用
  MiniBatchKMeans，按动作 ID 串行且每个模型使用 4 个 OpenMP 线程；
- `prototypes.num_threads` 与只控制 episode 读取进程的 `runtime.num_workers: 4` 相互
  独立，改变线程数不改变 graph 指纹；
- 候选按 `[0..3]`、`[3..6]` 独立选择最近叶原型，权重为保留比例置信度与桶内距离
  置信度的乘积；同叶合并，最终绝对权重不归一；
- 固定使用随机多分支全局选择，不施加逐任务配额。

Bridge 为 5 Hz，因此 7 帧片段按帧数计约 1.4 秒，运动原语的
`state[t] → state[t+3]` 状态差跨度为 0.6 秒。本适配包不重采样轨迹帧，也不改变 Cocore
的帧级语义；常规起点每隔 2 帧（0.4 秒），尾差为奇数时最后一个间隔取 1 帧，以保持
首尾完整覆盖。该间隔只控制动作原型训练窗口的起点。

安装依赖：

```bash
pip install -r cocore_bridge_v2/requirements.txt
```

## 命令

所有子命令都必须显式指定关系类型和非负有限权重，不存在隐藏的默认目标：

```bash
python -m cocore_bridge_v2 scan \
  --relation sequence --relation-weight 1.0

python -m cocore_bridge_v2 encode \
  --relation sequence --relation-weight 1.0

python -m cocore_bridge_v2 build-graph \
  --relation sequence --relation-weight 1.0

python -m cocore_bridge_v2 select \
  --relation sequence --relation-weight 1.0 \
  --selection-ratio 0.10

python -m cocore_bridge_v2 run \
  --relation sequence --relation-weight 1.0 \
  --selection-ratio 0.10 \
  --reliability-metrics support \
  --output-dir outputs/cocore_bridge_v2/bridge-support-only \
  --no-use-stop-bucket \
  --force
```

关系可选择 `sequence` 或 `cooccurrence`。选择算法固定为 `random_multibranch`；
`select`、`run` 和 `validate` 的 `--selection-ratio` 默认是 `0.10`。执行阶段还支持：

- `--dataset-path PATH`：覆盖默认挂载点，但目标必须满足同一 Bridge schema；
- `--output-dir PATH`：覆盖默认输出根目录；
- `--max-episodes N`：在排除空任务后只处理前 N 条有效 episode；
- `--force`：按 Cocore 的缓存规则重建不兼容阶段。
- `--no-use-stop-bucket`：仅用于 `build-graph`、`select`、`run` 和 `validate`，关闭 stop
  桶并排除双半段均无非 stop 标签的候选；未传时保持默认启用。
- `--reliability-metrics support|support,progress`：仅用于 `build-graph`、`select`、
  `run` 和 `validate`；默认 `support,progress`，只用 support 时必须显式传 `support`。

本包不接受任意 YAML `--config`，以防绕过固定相机、空任务策略或运动原语契约。

## 输出与验证

默认输出根目录是：

```text
outputs/cocore_bridge_v2/bridge_orig_1.0.0
```

其中包含 `scan/`、`encode/`、`graph-18-motion-hard-nearest-pca/` 和
`select-<relation>-w<weight>-top<ratio>pct-random-multibranch/`。选择目录继续提供
`selected_manifest.jsonl`、`all_clips.parquet`、`selection_report.json`、
`manifest.json` 和 `run_manifest.json`；encode 目录提供 Quality 融合
`embeddings.npy`、`visual_pca.npz`、`numeric_normalizers.npz`、按 episode 分片的
`frame_embeddings/`、对应索引以及候选双半段均值
`visual_half_embeddings.npy`。逐帧缓存覆盖所有已索引 episode，包括短 episode。
graph 目录提供 schema 10 的 `prototype_catalog.json`、128 维
`prototype_centers.npy`、节点到 scan/encode 行号的 `source_clip_indices.npy` 和内部
校验用 `half_action_labels.npy`。聚类复用 Encode 的
PCA components 前半列进行逐帧纯矩阵投影，不使用 mean/scale。选择输出包含最终
原型标签、动作标签、绝对置信度和 `half_action_labels`，不包含旧的 action/distance
分解权重。catalog 与 manifest 记录 `bridge_v2` profile、分轴阈值、roll 标签、环绕轴
和 `0.5%/400` 保留策略，并记录起点最大间隔 2 的窗口策略。select manifest 与报告
使用 selection schema 1；manifest 的生产者仍为 `cocore`。Cocore 版本为 0.17.0，
Bridge 包版本为 0.11.0。graph/select/run manifest、报告和解析配置均记录实际可靠性指标。

视觉中心训练只物化一次保留窗口投影；小桶并行执行完整 KMeans，大桶串行执行
MiniBatchKMeans。Bridge V2 完整生产数据的基础额外内存约为 257 MiB（按参考精确保留
非 stop 与原始 stop 窗口估算，每窗口 `128 × 4` 字节），不使用 memmap 或磁盘 fallback。

Cocore schema 9 artifact 不迁移且 validator 会拒绝。Cocore 0.17.0/schema 10 与 Bridge
0.11.0 新增可靠性指标契约；版本校验保持严格，旧 artifact 必须使用 `--force` 重建。
同一输出根切换 `support` 与 `support,progress` 也会使 graph 指纹不兼容；需要保留两组
实验时应使用不同的 `--output-dir`。

验证时必须重复传入生成该选择结果时使用的目标、比例、数据集路径以及
`--max-episodes`（若生成时设置）。省略 `--max-episodes` 表示按完整有效数据集重放；
若生成阶段只处理前 N 条有效 episode，验证也必须传入相同的 N。执行类命令会先做
Bridge schema preflight；validator 会从该路径重放源 episode 的 state/trajectory，
并核对 artifact 中的逐帧视觉缓存，因此源数据仍必须可访问：

```bash
python -m cocore_bridge_v2 validate \
  --output-dir \
    outputs/cocore_bridge_v2/bridge_orig_1.0.0/select-sequence-w1-top10pct-random-multibranch \
  --dataset-path /data/dwb/datasets/bridge_orig_1.0.0_lerobot \
  --relation sequence --relation-weight 1.0 \
  --selection-ratio 0.10 \
  --max-episodes 100 \
  --reliability-metrics support \
  --no-use-stop-bucket
```

只有生成结果时传入了 `--no-use-stop-bucket` 或 `--reliability-metrics support`，验证时才
重复对应参数。升级旧版本，或切换 stop/可靠性指标并复用同一输出根目录时，应使用
`--force` 重建全部不兼容阶段。

## 运行基线

正式数据预期包含 53,192 条源 episode。排除 14,532 条空任务 episode 后保留
38,660 条、1,305,714 帧；其中 2 条短于 7 帧。最终产生 202,739 个候选片段，
Top 10% 预算为 20,274。

只读动作诊断不加载图像、不创建缓存：

```bash
.venv/bin/python scripts/analyze_cocore_bridge_action_thresholds.py \
  --dataset-path /data/dwb/datasets/bridge_orig_1.0.0_lerobot \
  --verify-reference
```

它输出七轴绝对差分位与激活率、复合类别数、stop/回退率、精确覆盖、原子保留质量、
动作桶数和预计视觉叶原型数。完整训练集参考契约为 622,782 个四帧窗口、保留门槛
3,114、1,181 个实际复合标签、25 个保留的非 stop 桶和约 594 个叶原型；非 stop 精确
覆盖约 60.62%，父类回退约 14.22%，无父类回退不超过 1.23%，原始 stop 约 23.94%。
原子动作保留质量至少 68.79%，原子 occurrence 保留质量至少 84.67%。

间隔 2 参考契约下的完整保留桶与预计叶原型分布如下；`count` 即硬训练窗口数：

| 动作桶 | count | 叶原型 |
|---|---:|---:|
| stop | 149,124 | 30 |
| close gripper | 56,944 | 30 |
| move down | 41,712 | 30 |
| open gripper | 36,257 | 30 |
| move up | 35,990 | 30 |
| move left | 34,649 | 30 |
| move right | 30,456 | 29 |
| move forward | 17,442 | 26 |
| move backward | 13,775 | 24 |
| move forward right | 10,581 | 23 |
| move backward left | 9,833 | 23 |
| move left down | 8,925 | 22 |
| move right down | 8,919 | 22 |
| move up, open gripper | 8,396 | 22 |
| move forward left | 8,299 | 22 |
| move right up | 8,052 | 21 |
| move left up | 7,921 | 21 |
| move backward right | 7,116 | 21 |
| move forward down | 6,279 | 20 |
| move right, rotate clockwise | 4,440 | 18 |
| move left, rotate counterclockwise | 4,289 | 18 |
| move backward up | 4,031 | 17 |
| move backward down | 3,498 | 17 |
| rotate counterclockwise | 3,300 | 16 |
| move forward right down | 3,233 | 16 |
| move forward up | 3,186 | 16 |
| **合计（26 桶）** | **526,647** | **594** |

无需 GPU 的 100 episode scan 冒烟：

```bash
python -m cocore_bridge_v2 scan \
  --relation sequence --relation-weight 1.0 \
  --max-episodes 100 \
  --output-dir outputs/cocore_bridge_v2/bridge-smoke-100
```

`encode` 和完整 `run` 会解码 `image_0` AV1 视频并使用单路 CLIP 特征。每个有效
episode（包括不足 7 帧的短 episode）的完整逐帧特征会写入 encode 缓存，并经 128 维视觉 PCA 与 state/action
时序池化特征融合。生产配置固定从 `/data/dwb/models/clip-vit-base-patch32` 本地加载
模型，要求 CUDA；不会访问网络，也不会读取 `image_1`、`image_2` 或 `image_3`。
