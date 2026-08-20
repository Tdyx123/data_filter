# Cocore：运动原语关系筛选

Cocore 从 LeRobot v2 LIBERO episode 中选择固定预算的 15 帧片段。它独立管理候选
索引、稀疏图、Quality 风格编码流水线、两级动作原型、配置、缓存与输出产物；
运行时代码不依赖 `quality_filter` 或 `segment_filter_core` 的切片逻辑。

长度为 `L >= 15` 的 episode 固定生成 `N = ceil(L / 15)` 个完整候选。首个候选从
第 0 帧开始，末个候选结束于最后一帧；中间 `N-1` 个起点间隔近似均匀，间隔最多
相差 1，较短间隔集中在前部。例如 16 帧的起点为 `[0,1]`，31 帧为
`[0,8,16]`，207 帧为 `[0,14,28,42,57,...,192]`。同一 episode 中按起点相邻的
候选始终构成 sequence 边，即使两个 15 帧窗口发生重叠。

编码先按全数据 1%/99% 分位将 action 和向量 observation 缩放到 `[0,1]`。每个
episode 只执行一次视觉模型前向；候选片段的视觉特征
`[sum(v0..v14), v14-v0]` 直接用于拟合 128 维 PCA，再与 state/action 的
`mean/std/max` 及 `start/episode_length` 拼接并做行 L2 归一化。该 embedding 用于
可靠性 support 和相似图。Encode 还保留原始逐帧 CLIP 缓存，并将候选拆成共享第 7 帧的
`[0..7]`、`[7..14]` 两个半段，分别缓存原始 CLIP 空间中的 8 帧归一化均值。

动作—视觉原型保留两级解耦：一级表达动作桶，二级表达桶内视觉中心。学习原型时，
对每条完整轨迹生成首尾覆盖、起点间隔最大为 3 的 8 帧窗口 `[t,t+7]`，用
`state[t]→state[t+7]` 分类动作。间隔默认取 3；轨迹长度为 `3x+1` 时最后一个间隔
取 2，长度为 `3x` 时最后两个间隔取 2，长度 9 特取起点 `[0,1]`。例如长度
12、13、14 的起点分别为 `[0,2,4]`、`[0,3,5]`、`[0,3,6]`。设原始逐帧 CLIP
维度为 `D`；聚类复用上述
`visual_pca.npz`，裁剪 `components` 的前 `D` 列，对每帧执行纯矩阵乘法
`v @ components[:, :D].T`，不应用 PCA 的 `mean` 或 `scale`，分量不足时右补零到
128 维。窗口在投影后取 8 帧均值并 L2 归一化，再学习视觉中心；候选两个半段也投影到
同一空间后分配最近中心。设全部窗口数为 `W`，动作保留条件为：

```text
count >= max(400, ceil(0.005 * W))
```

非 `stop` 桶只训练原始标签本身达到门槛的硬窗口；低频窗口不进入任何训练桶。默认
`prototypes.use_stop_bucket: true`，此时 `stop` 只训练原始 `stop` 窗口；设为 `false`
时仍统计原始 `stop` 数量，但不训练 stop 桶、视觉中心或叶原型。令 `M_a` 为桶内硬样本
数，中心数固定为：

```text
K_a = min(M_a, min(30, max(10, floor(4 * log2(M_a) - 30))))
```

外层 `min(M_a, ...)` 只在桶内不足 10 个训练窗口时降低中心数；非 `stop` 桶仍受上述
至少 400 个窗口的动作门槛保护，因此其中心数范围为 10～30。

聚类不使用样本权重。训练窗口数不超过 65,536 的动作桶使用完整 Lloyd KMeans，允许
按动作桶并行且每个模型固定 1 个 OpenMP 线程；超过 65,536 的动作桶使用
MiniBatchKMeans，这些大桶按动作 ID 串行拟合且每个模型固定 4 个 OpenMP 线程。两种
模型都使用一次 k-means++ 初始化，并由 `tol` 收敛条件提前停止，`max_iter` 只作为迭代
上限。中心按硬归属数降序、坐标字典序稳定编号；每桶还记录训练窗口到最近中心的欧氏
距离 q10/q90。

构造 15 帧候选标签时，分别分类 `state[0]→state[7]` 与
`state[7]→state[14]`，两个半段各自独立打一个叶标签。若原始动作未保留，先找原子数
最多的保留子集；多个父动作并列时，在它们的所有中心中选欧氏距离最近的叶原型，距离
相同取较小叶 ID。启用 stop 桶时，没有非空父集会回退 `stop`；关闭时该半段不产生
标签。两个半段都没有标签的候选从 graph 起排除，不参与 KNN、关系图、预算计算或筛选；
只有一个半段有标签的候选仍保留。过滤后的 sequence/cooccurrence 是原候选图的诱导
子图，不跨无标签空洞重连，也不压缩原始共现距离。

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
`selection.method` 默认使用 `lazy_heap`，其余预算使用确定性的惰性最大堆近似优化：

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

`selection.method: random_multibranch` 改用可复现的随机多分支搜索。coverage seed 始终作为
固定集合；初始化 8 个活动分支，各无放回抽取 10 个片段并记作第 1 轮。以后每个父分支
生成 4 个各补入 10 个片段的子分支，从 32 个子分支中保留 8 个。cooccurrence 关系项仍
使用全部固定片段和当前分支片段。sequence 分支在初始化和每次重组后从零开始，只保留
分支局部 sequence 计数：普通子分支继承父分支累计值，对本轮新增片段按顺序只计算其与
全部固定片段、父分支已有活动片段及本轮更早新增片段之间的原始有向 sequence 边。
固定片段内部以及已有活动片段内部的边不会重复计算；numerator 使用该局部计数，
denominator 仍使用全池 sequence 计数。

分支冗余同样从 0 开始，固定片段不超过 100 时以全部固定片段作为相似性参照；超过 100
后，每个初始化、子或重组分支都使用独立随机流无放回抽取 100 个固定片段。普通子分支
继承父分支累计冗余；对本轮新增片段按顺序只计算其与固定参照、父分支已有活动片段及
本轮更早新增片段之间的相似边。固定参照片段内部以及已有活动片段内部的相似边不会重复
计算。最终目标和逐条增益使用获胜分支从最近一次重组开始积累的 sequence 与冗余增量；
coverage seed 和已经提交的固定片段在 sequence 模式下不贡献最终 relation 增量。

首次在第 20 轮重组，之后每 10 轮重组：统计活动片段在 8 个分支中的出现次数，从最高
得分分支内取计数最高的 100 个加入固定集合，再让每个分支从自己的未固定片段中保留
计数最高的 100 个。重组时分支 sequence 与冗余均清零，每个分支保留的 100 个片段整体
作为新批次，按顺序只与新的全部固定集合和更早保留片段计算 sequence，并按相似性抽样
规则计算冗余。计数或分支得分相同时由 `seed` 随机破平局；最后一批不足 10 个时只补足
预算，达到预算后直接采用最高得分分支。主状态只保存固定选择掩码、原型覆盖和任务计数，
分支仅保存稀疏更新；8、4、10、20、10 和两个用途不同的 100 都是固定算法常量，不提供
额外配置项。

增量 sequence 与冗余口径记录在选择算法元数据中，因此旧 `random_multibranch` select
缓存与新口径不兼容。升级后可运行
`python -m cocore select --config <config> --force` 仅重建 select 产物；既有 scan、
encode 和 graph 产物继续复用。sequence 随机多分支产物中，`selection_report.json` 的
`objective.relation`、`weighted_relation`、`total` 及逐片段增益均使用上述分支局部口径。

随机多分支选择会用单调高精度时钟记录每轮分支生成、评分与保留耗时；发生重组时，
计数、提交、主状态重建、片段保留及重组分支重建单独计时，不计入普通轮次。最终结果
重算与导出也不计入轮次，而由既有的选择总耗时和 `select.export` 覆盖。
`selection_report.json` 的 `branch_search.timings` 保存完整明细与算术平均值：

```json
{
  "rounds": [{"round": 1, "seconds": 0.125}],
  "recombinations": [{"round": 20, "seconds": 0.5}],
  "average_round_seconds": 0.125,
  "average_recombination_seconds": 0.5
}
```

命令行不打印逐轮或逐次重组明细，只在存在对应样本时输出
`select.random_multibranch.round_average` 和
`select.random_multibranch.recombination_average` 两条 `cocore_timing` 平均值；既有的
`select.random_multibranch` 总耗时保持不变。没有轮次或重组时，报告中的列表为空、
平均值为 `null`，命令行省略相应平均值。首次使用该计时 schema 时，兼容的旧随机选择
缓存会仅重建 select 产物，继续复用 scan、encode 和 graph。

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
  tol: 1.0e-4
  num_threads: 4
  use_stop_bucket: true

selection:
  method: lazy_heap  # 或 random_multibranch
  ratio: 0.10
  budget: null
  max_refreshes: 100  # 仅 lazy_heap 使用
```

片段长度固定为 15，候选数量、首尾锚定与近似均匀间隔是 Cocore 固定算法的一部分；
配置中不接受 `clip` section。
Quality 风格编码取代了旧关系编码，因此不再接受顶层 `relation` 或 `normalization`；
`encoding.visual_dim` 固定为 128，`pca_fit_max_samples` 可限制 PCA 拟合样本数。
动作门槛、中心数公式、30 个中心上限、距离分位和权重公式都是 Cocore 固定算法，
不可配置；`prototypes` 只接受 `method`、`batch_size`、`max_iter`、正数 `tol`、正整数
`num_threads` 和布尔值 `use_stop_bucket`，并明确拒绝旧 `count`、`top_r` 或
`temperature`。`batch_size` 只影响
超过 65,536 个训练窗口的大桶；`num_threads` 默认为 4，只并行不超过阈值的小桶，
debug 配置固定为 1。它与 `runtime.num_workers` 相互独立，后者仍只控制 episode 读取
进程。

视觉中心训练会一次物化所有保留窗口的 128 维 `float32` 投影。小桶用完整 KMeans
并行拟合，大桶用 MiniBatchKMeans 串行拟合，避免多个大桶同时占用 CPU 和临时内存。
基础额外内存约为“保留窗口数 × 128 × 4 字节”：LIBERO90 约 106 MiB，Bridge V2
约 273 MiB；完整 KMeans 拟合小桶时还会产生有界于 65,536 个窗口的工作副本。改变
`num_threads` 不改变 graph 指纹或产物，因此可复用同一 graph 缓存；改变
`batch_size`、`max_iter`、`tol` 或 `use_stop_bucket` 会使 graph 缓存失效。

可在运行时覆盖选择方法、选择比例、关系类型与关系权重：

```bash
python -m cocore run --config cocore/config_libero90.yaml \
  --selection-method random_multibranch \
  --selection-ratio 0.20 \
  --relation sequence \
  --relation-weight 1.5
```

`build-graph`、`select`、`run` 和 `validate` 还接受单向开关
`--no-use-stop-bucket`，其优先级高于 YAML，并将
`prototypes.use_stop_bucket` 强制覆盖为 `false`。`scan` 和 `encode` 不接受该参数。
例如关闭 stop 桶并重建不兼容缓存：

```bash
python -m cocore run --config cocore/config_libero90.yaml \
  --no-use-stop-bucket \
  --force
```

上述选择写入
`outputs/cocore/libero90/select-sequence-w1p5-top20pct-random-multibranch/`；默认
`lazy_heap` 仍写入原目录名。
`--cooccurrence-weight` 已移除；Cocore 也不接受
RelCore 的 `--reliability-metrics`、`--prototype-method` 或
`--prototype-gain-metrics` 参数。

快速 CPU 检查：

```bash
python -m cocore run --config cocore/config_debug.yaml --force
```

Cocore 0.15.0 使用 prototype schema 9、10～30 个桶内视觉中心、可选 stop 桶、无标签
候选诱导子图、65,536 窗口的混合 KMeans 阈值、最大间隔 3 的动作训练窗口、裁剪 PCA
的 128 维聚类空间、近似均匀候选和原始相邻 sequence 图，并按 episode 持久化完整原始
逐帧 CLIP 特征，并提供 lazy heap 与随机多分支两种选择方法。0.14.x 及更早版本的
scan、encode、graph 和 selection 缓存不迁移；
升级后必须通过 `--force` 重建全部阶段，或使用新的输出目录。

## 输出与校验

输出根目录包含 `scan/`、`encode/`、`graph-17-motion-hard-nearest-pca/` 和一个或多个
`select-<关系>-w<权重>-top<比例>pct/`；随机多分支方法追加
`-random-multibranch`。选择目录包含：

- scan 目录中的 `episodes.parquet` 与 `clips.parquet`：episode 元数据和可重放的
  近似均匀候选；Cocore scan 不再生成未被后续阶段消费的 `normalization.npz`；
- graph 目录中的 `prototype_catalog.json` 与 `prototype_centers.npy`：动作类别、
  原始计数、训练计数、视觉中心数量、距离 q10/q90、投影契约以及按叶 ID 对齐的
  128 维中心；
- graph 目录中的 `source_clip_indices.npy`：每个 eligible graph 节点对应的 scan/encode
  候选行号；关闭 stop 桶后它同时记录被排除候选形成的空洞；
- encode 目录中的 `embeddings.npy`、`visual_pca.npz` 和
  `numeric_normalizers.npz`：Quality 融合 embedding 及其可重放参数；
- encode 目录中的 `frame_embeddings/ep<episode_id>.npy`：每个已索引 episode（包括
  短 episode）的完整 `[帧数, CLIP 维度]`、`float32` 逐帧特征；
  `frame_embeddings_index.json` 记录顺序、shape 和 SHA-256；
- encode 目录中的 `visual_half_embeddings.npy`：每个候选两个半段的归一化视觉均值；
  graph 目录中的 `half_action_labels.npy` 记录两个半段各自的原始动作标签；

- `selected_manifest.jsonl`：训练入口可直接消费的片段清单；
- `all_clips.parquet`：eligible 筛选池的 support、progress、reliability、运动原语与选择
  诊断，不包含无标签候选；
- `selection_report.json`：coverage、目标分解、任务计数、lazy heap 刷新统计或
  `branch_search` 轮次/评估/重组统计、逐轮与逐次重组耗时，以及 scanned、eligible、
  excluded-unlabeled 候选数量；
- `manifest.json`、`run_manifest.json`：Cocore 参数、阶段目录与指纹；
- `resolved_config.yaml`、`environment.json`：`run` 的完整配置与环境。

使用以下命令独立重算 coverage、逐步边际增益、目标、预算、唯一性、方法统计和清单一致性：

```bash
python -m cocore validate \
  --output-dir outputs/cocore/libero90/select-cooccurrence-w1-top10pct \
  --config cocore/config_libero90.yaml \
  --selection-method lazy_heap \
  --no-use-stop-bucket
```

只有生成结果时关闭了 stop 桶，验证时才应传入该开关。若省略 `--config`，CLI 会从
选择目录的 `resolved_config.yaml` 读取重放配置；验证随机多分支结果时必须使用生成时
相同的方法和 seed，校验器会重放每个分支的相似性抽样并核对最终惩罚集合。复用
0.14.x 或其他 stop 设置不同的
输出目录时，应使用 `--force` 重建全部不兼容阶段。

校验还会从 episode 元数据重放近似均匀候选，逐字段核对 `clips.parquet`，逐个检查
帧缓存文件集合、shape、dtype、有限值和 SHA-256，并使用同一 PCA components 重算
投影窗口、中心、距离分位和候选分配。编码中断时临时缓存会被清理，下次从头执行；
只有完整发布的 encode 阶段才会被复用。

当前版本只支持本仓库约定的 8 维 LIBERO `observation.state`。selection 的
parquet/JSONL 导出最终 `prototype_indices`、`prototype_weights`、叶标签、动作标签与
`half_action_labels`，不导出可分解的 action/distance 权重。所有图关系与选择目标直接
消费绝对 `prototype_weights`，不对每行额外归一或截断。
