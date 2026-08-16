# Cocore：运动原语关系筛选

Cocore 从 LeRobot v2 LIBERO episode 中选择固定预算的 15 帧片段。它复用 RelCore
的片段索引和稀疏图能力，但独立管理 Quality 风格编码流水线、两级动作原型、配置、
缓存与输出产物；运行时代码不依赖 `quality_filter` 或 `segment_filter_core`。

编码先按全数据 1%/99% 分位将 action 和向量 observation 缩放到 `[0,1]`。每个
episode 只执行一次视觉模型前向；候选与 reference 片段的视觉特征
`[sum(v0..v14), v14-v0]` 在去重并集上拟合 128 维 PCA，再与 state/action 的
`mean/std/max` 及 `start/episode_length` 拼接并做行 L2 归一化。该 embedding 用于
可靠性 support 和相似图。原型分配另将候选拆成共享第 7 帧的 `[0..7]`、`[7..14]`
两个半段，分别计算 8 帧 CLIP 均值并 L2 归一化。

动作—视觉原型保留两级解耦：一级表达动作桶，二级表达桶内视觉中心。学习原型时，
对每条完整轨迹生成所有 stride=1 的 8 帧窗口 `[t,t+7]`，用
`state[t]→state[t+7]` 分类动作，并用窗口内 8 个逐帧 CLIP embedding 的归一化均值
学习视觉中心。设全部窗口数为 `W`，动作保留条件为：

```text
count >= max(400, ceil(0.005 * W))
```

非 `stop` 桶只训练原始标签本身达到门槛的硬窗口；低频窗口不进入任何训练桶。`stop`
只训练原始 `stop` 窗口。令 `M_a` 为桶内硬样本数，中心数固定为：

```text
K_a = min(16, floor(log2(M_a)) - 2)
```

MiniBatchKMeans 不使用样本权重；中心按硬归属数降序、坐标字典序稳定编号。每桶还记录
训练窗口到最近中心的欧氏距离 q10/q90。

构造 15 帧候选标签时，分别分类 `state[0]→state[7]` 与
`state[7]→state[14]`，两个半段各自独立打一个叶标签。若原始动作未保留，先找原子数
最多的保留子集；多个父动作并列时，在它们的所有中心中选欧氏距离最近的叶原型，距离
相同取较小叶 ID。没有非空父集时回退 `stop`。

```text
w_r = 0.5 + 0.5 * |A_parent| / |A_raw|
w_d = q10 处 1.0、q90 处 0.3，中间线性插值并截断到 [0.3, 1.0]
w_half = w_r * w_d
```

精确标签（包括原始 `stop`）的保留比例取 1；非 `stop` 回退 `stop` 的比例取 0。两半
命中同一叶时合并为 `max(w1,w2)+0.5*min(w1,w2)`；同动作不同中心不合并。最终固定
最多两个槽位，按权重降序、叶 ID 破平局；权重不归一、不截断，范围可到 1.5。

可靠性固定为 `sqrt(support * progress)`。初始集合为每个可达运动原语选择
`reliability * assignment` 最大的片段并取并集，使所有原型 coverage 达到全池最大值。
其余预算使用确定性的惰性最大堆近似优化：

```text
c_p(S) = max_{i in S}(reliability_i * assignment_{i,p})
cooccurrence(S) = sum_{p,q} C_{p,q} * sqrt(c_p(S) * c_q(S))

n_{p,q}(S) = sum_{(i,j) in E_sequence, i,j in S}
               reliability_i * reliability_j * assignment_{i,p} * assignment_{j,q}
sequence(S) = sum_{p,q} T_{p,q} * log1p(n_{p,q}(S))
              / (log1p(n_{p,q}(V)) + epsilon)

score(S) = relation_weight * relation(S) - redundancy(S)
```

`objective.relation` 必须显式选择 `cooccurrence` 或 `sequence`。`C` 是 RelCore 的
运动原语共现矩阵；`T` 是运动原语 transition 矩阵；`E_sequence` 是有向的真实相邻片段
边；`V` 表示全池。两项指标与 RelCore 使用同一个实现，冗余也沿用 RelCore 的可靠性
加权相似边定义。由于覆盖种子已经使每个 `c_p(S)` 达到全池最大值，`cooccurrence`
模式在初始化后不再产生正向关系增益，剩余候选主要按冗余惩罚竞争。

初始化后，所有剩余片段按相对初始集合的边际增益建成最大堆。每选一个片段后，堆顶的
旧增益按当前集合惰性重算；如果已更新条目成为堆顶就立即选择，否则每轮最多重算
`selection.max_refreshes` 个条目（默认 100），达到上限时选择本轮已更新条目中增益
最大的片段。初始化集合不计入堆选择步数；在准备选择第 8、16、32、……个堆候选前，
算法会按当前集合重算全部未选片段的边际增益并重建堆，再选择新的堆顶。全量重建次数
不计入 `heap_refreshes` 等惰性刷新统计。增益相同时按 `sample_id` 稳定排序。

堆阶段不施加任务配额，所有剩余片段全局竞争。由于 sequence 项可能使边际增益随集合增长，
旧堆值不一定是严格上界，因此这是有界近似算法，不保证与全量贪心或旧束搜索结果一致。

## 运行

```bash
pip install -r cocore/requirements.txt

python -m cocore scan --config cocore/config_libero90.yaml
python -m cocore encode --config cocore/config_libero90.yaml
python -m cocore build-graph --config cocore/config_libero90.yaml
python -m cocore select --config cocore/config_libero90.yaml
python -m cocore run --config cocore/config_libero90.yaml
```

配置必须显式声明关系类型与权重：

```yaml
encoding:
  visual_dim: 128
  pca_fit_max_samples: null
  quantile_low: 0.01
  quantile_high: 0.99
  epsilon: 1.0e-8

objective:
  relation: cooccurrence  # 或 sequence
  relation_weight: 1.0

prototypes:
  method: motion_primitives
  batch_size: 4096
  max_iter: 100
```

片段长度与步长是 Cocore 固定算法的一部分，均为 15；配置中不接受 `clip` section。
Quality 风格编码取代了旧关系编码，因此不再接受顶层 `relation` 或 `normalization`；
`encoding.visual_dim` 固定为 128，`pca_fit_max_samples` 可限制 PCA 拟合样本数。
动作门槛、中心数公式、16 个中心上限、距离分位和权重公式都是 Cocore 固定算法，
不可配置；`prototypes` 只接受 `method`、`batch_size` 和 `max_iter`，并明确拒绝旧
`count`、`top_r` 或 `temperature`。

可在运行时覆盖选择比例、关系类型与关系权重：

```bash
python -m cocore run --config cocore/config_libero90.yaml \
  --selection-ratio 0.20 \
  --relation sequence \
  --relation-weight 1.5
```

上述选择写入 `outputs/cocore/libero90/select-sequence-w1p5-top20pct/`。
`--cooccurrence-weight` 已移除；Cocore 也不接受
RelCore 的 `--reliability-metrics`、`--prototype-method` 或
`--prototype-gain-metrics` 参数。

快速 CPU 检查：

```bash
python -m cocore run --config cocore/config_debug.yaml --force
```

Cocore 0.9.0 使用 prototype schema 5，并按 episode 持久化完整逐帧 CLIP 特征。
schema 4 的 encode、graph 和 selection 缓存不迁移、也不会按 schema 5 读取；升级后
必须通过 `--force` 重新构建 graph 和 selection，或使用新的输出目录。

## 输出与校验

输出根目录包含 `scan/`、`encode/`、`graph-14-motion-hard-nearest/` 和一个或多个
`select-<关系>-w<权重>-top<比例>pct/`。选择目录包含：

- graph 目录中的 `prototype_catalog.json` 与 `prototype_centers.npy`：动作类别、
  原始计数、训练计数、视觉中心数量、距离 q10/q90、固定常量以及按叶 ID 对齐的中心；
- encode 目录中的 `embeddings.npy`、`visual_pca.npz` 和
  `numeric_normalizers.npz`：Quality 融合 embedding 及其可重放参数；
- encode 目录中的 `frame_embeddings/ep<episode_id>.npy`：每个已索引 episode（包括
  短 episode）的完整 `[帧数, CLIP 维度]`、`float32` 逐帧特征；
  `frame_embeddings_index.json` 记录顺序、shape 和 SHA-256；
- encode 目录中的 `visual_half_embeddings.npy`：每个候选两个半段的归一化视觉均值；
  graph 目录中的 `half_action_labels.npy` 记录两个半段各自的原始动作标签；

- `selected_manifest.jsonl`：训练入口可直接消费的片段清单；
- `all_clips.parquet`：全池 support、progress、reliability、运动原语与选择诊断；
- `selection_report.json`：coverage、目标分解、任务计数与堆刷新统计；
- `manifest.json`、`run_manifest.json`：Cocore 参数、阶段目录与指纹；
- `resolved_config.yaml`、`environment.json`：`run` 的完整配置与环境。

使用以下命令独立重算 coverage、逐步边际增益、目标、预算、唯一性、堆统计和清单一致性：

```bash
python -m cocore validate \
  --output-dir outputs/cocore/libero90/select-cooccurrence-w1-top10pct \
  --config cocore/config_libero90.yaml
```

校验还会逐个检查帧缓存文件集合、shape、dtype、有限值和 SHA-256。编码中断时临时
缓存会被清理，下次从头执行；只有完整发布的 encode 阶段才会被复用。

当前版本只支持本仓库约定的 8 维 LIBERO `observation.state`。selection 的
parquet/JSONL 导出最终 `prototype_indices`、`prototype_weights`、叶标签、动作标签与
`half_action_labels`，不导出可分解的 action/distance 权重。所有图关系与选择目标直接
消费绝对 `prototype_weights`，不对每行额外归一或截断。
