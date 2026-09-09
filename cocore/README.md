# Cocore：运动原语关系筛选

Cocore 从 LeRobot v2 机器人 episode 中选择固定预算的 profile 固定长度片段：LIBERO
使用 15 帧，BridgeData V2 使用 7 帧。它独立管理候选
索引、稀疏图、Quality 风格编码流水线、两级动作原型、配置、缓存与输出产物；
运行时代码不依赖 `quality_filter` 或 `segment_filter_core` 的切片逻辑。

LIBERO 中长度为 `L >= 15` 的 episode 固定生成 `N = ceil(L / 15)` 个完整候选。首个候选从
第 0 帧开始，末个候选结束于最后一帧；中间 `N-1` 个起点间隔近似均匀，间隔最多
相差 1，较短间隔集中在前部。例如 16 帧的起点为 `[0,1]`，31 帧为
`[0,8,16]`，207 帧为 `[0,14,28,42,57,...,192]`。同一 episode 中按起点相邻的
候选始终构成 sequence 边，即使两个 15 帧窗口发生重叠。Bridge 使用同一近似均匀、
首尾覆盖策略，但将候选长度固定为 7，候选数为 `ceil(L / 7)`。

编码先按全数据 1%/99% 分位将 action 和向量 observation 缩放到 `[0,1]`。每个
episode 只执行一次视觉模型前向；候选片段的视觉特征
`[sum(v0..v(C-1)), v(C-1)-v0]` 直接用于拟合 128 维 PCA，其中 `C` 为 profile 的候选
长度；再与 state/action 的
`mean/std/max` 拼接并做行 L2 归一化，所有 profile 均不包含轨迹位置
`start/episode_length`。旧版含位置的编码缓存指纹不兼容，需使用 `--force`
重新编码并重建下游产物。该 embedding 用于
可靠性 support 和相似图。Encode 还保留原始逐帧 CLIP 缓存。LIBERO 将候选拆成共享
第 7 帧的 `[0..7]`、`[7..14]` 两个 8 帧半段；Bridge 将 7 帧候选拆成共享第 3 帧的
`[0..3]`、`[3..6]` 两个 4 帧半段，分别缓存原始 CLIP 空间中的归一化均值。

support 使用全体候选 embedding 的欧氏距离：令 `k_eff = min(quality.knn, N-1)`，
`d_k(i)` 为第 `k_eff` 个其他候选的距离，统一半径为 `R = median(d_k)`。
令 `count_i` 为距离 `<= R` 的其他候选数，则计入自身后的
`support_i = min(count_i + 1, k_eff + 1) / (k_eff + 1)`，输出为 `[0,1]` 的 float32。
边界点和重复坐标的其他候选均计入；`R=0` 时仍按此规则计算，单候选直接取 1。
graph manifest 记录 `support_mode: median_radius_count_with_self`；旧公式的 graph
及下游选择缓存需用 `--force` 重建，兼容的 scan/encode 缓存可继续复用。

新增可选可靠性指标 `support_old`，复用原先基于距离的 support：
`support_old_i = exp(-d_k(i) / (median(d_k) + 1e-8))`。`d_k` 是全体候选
embedding 欧氏距离空间中的第 `k_eff = min(quality.knn, N-1)` 个非自身近邻距离；
单候选分数为 `1`，输出为 `[0,1]` 的 float32。

`support` 与 `support_old` **不能同时加入** `reliability_metrics`，也可以均不选择；
默认仍启用当前 `support`。两者共用 `quality.knn` / `--support-k K`，与相似图的
`graph.knn` 独立。无论选择哪项，graph 的 `nodes.npz`、`all_clips.parquet` 和
`selected_manifest.jsonl` 均保存两项分数，仅所选项参与几何均值融合及可靠性下限截断。
例如将 `--reliability-metrics support_old progress --support-k 20` 加入运行命令；
验证时使用相同指标及 K。Cocore YAML 等价配置为
`reliability_metrics: [support_old, progress]` 与 `quality: {knn: 20}`。

graph manifest 和缓存指纹新增 `support_old` 公式契约，现有 `support_mode` 仍标识
当前计数公式。缺少新契约或字段的旧产物需使用 `--force` 重建 graph 及下游结果；
兼容的 scan/encode 缓存会复用。校验会从全部候选 embedding 重算两项分数。

`build-graph`、`select`、`run` 和 `validate` 支持 `--support-k K`，K 必须为正整数。
显式传入时覆盖 `quality.knn`；未传时保留 YAML 配置值，默认配置为 10。
该参数不改变 `graph.knn`，计算仍使用 `k_eff = min(K, N-1)`。
例如 `python -m cocore run --support-k 20`；复用旧输出目录且 k 改变时加 `--force`。
`validate --support-k 20` 会检查 k 与产物保存值一致；未指定 `--config` 时读取
输出目录中的 `resolved_config.yaml`。`scan` 和 `encode` 不接受此参数。

动作—视觉原型保留两级解耦：一级表达动作桶，二级表达桶内视觉中心。学习原型时，
对每条完整轨迹生成首尾覆盖的 profile 固定窗口：LIBERO 使用起点最大间隔 3 的
8 帧 `[t,t+7]` 和 `state[t]→state[t+7]`；Bridge 使用起点最大间隔 2 的 4 帧
`[t,t+3]` 和 `state[t]→state[t+3]`。LIBERO 的尾部重排规则保持不变：轨迹长度为
`3x+1` 时最后一个间隔取 2，长度为 `3x` 时最后两个间隔取 2，长度 9 特取起点
`[0,1]`；例如长度 12、13、14 的起点分别为 `[0,2,4]`、`[0,3,5]`、
`[0,3,6]`。Bridge 正常起点每隔 2 帧，尾差为奇数时最后一个间隔取 1；例如长度
7、8、9 的起点分别为 `[0,2,3]`、`[0,2,4]`、`[0,2,4,5]`。设原始逐帧 CLIP
维度为 `D`；聚类复用上述
`visual_pca.npz`，裁剪 `components` 的前 `D` 列，对每帧执行纯矩阵乘法
`v @ components[:, :D].T`，不应用 PCA 的 `mean` 或 `scale`，分量不足时右补零到
128 维。窗口在投影后按 profile 取 8 帧或 4 帧均值并 L2 归一化，再学习视觉中心；候选两个半段也投影到
同一空间后分配最近中心。`prototypes.profile` 固定分类与动作保留契约：

- `libero`（通用 Cocore 默认）：沿用单一严格阈值 `0.03`，不分类 roll，动作保留条件为
  `count >= max(400, ceil(0.005 * W))`；
- `bridge_v2`：xyz 为 `0.03 m`，roll/pitch 为 `0.18 rad`，yaw 为 `0.24 rad`，
  gripper 为 `0.20`；roll 与 yaw 取 `[-π, π)` 最短角差，动作保留条件为
  `count >= max(400, ceil(0.005 * W))`。

Bridge 原子顺序固定为平移、`roll positive/negative`、pitch tilt、yaw rotate、gripper，
仍合成为一个复合标签。所有阈值边界都使用严格 `>`/`<`，等于阈值不激活动作。
设全部窗口数为 `W`，LIBERO 与 Bridge 的保留条件统一为：

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

构造候选标签时，LIBERO 分别分类 `state[0]→state[7]` 与
`state[7]→state[14]`，Bridge 分别分类 `state[0]→state[3]` 与
`state[3]→state[6]`；两个半段各自独立打一个叶标签。若原始动作未保留，先找原子数
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

可靠性目录为 `support`、`support_old`、`progress`、`action_variation`、
`visual_action_consistency`、`non_dwell`、`eef_jerk`、`local_path_efficiency`、`low_high_frequency_jitter`、`low_local_backtracking` 和 `action_jump`，
顶层 `reliability_metrics` 可选择非空、无重复且满足互斥规则的指标子集。默认启用
`support`、`progress`、`action_variation`、`visual_action_consistency` 和 `action_jump`；
`non_dwell` 需要显式配置 `dwell` 阈值。输入顺序会按
上述固定顺序规范化。`action_variation` 在完整 episode 的分位缩放动作上计算：逐步分数
为当前与前一步动作的 L2 差乘 2，再加未来最多 5 步动作逐维总体方差的维度均值；首步
差分为 0，未来少于 2 步时方差为 0。片段原始分数取最高 3 个逐步分数的均值，再使用
全部候选的 `encoding.quantile_low/high` 分位数缩放并截断到 `[0, 1]`。

`visual_action_consistency`（VAC，语义参考
[FrameSkip](https://arxiv.org/html/2605.13757)）复用同一轮编码保存的完整 episode 逐帧视觉特征和
分位缩放动作，不增加视觉模型前向。逐步分数为

```text
VAC_t = ||v_t - v_(t-1)||_2 / (||a_t - a_(t-1)||_2 + encoding.epsilon)
```

`VAC_0` 复制第一个有效分数；片段原始分数同样取最高 3 个逐步分数的均值，再按全部
候选的 `encoding.quantile_low/high` 缩放并截断到 `[0, 1]`。候选原始分数为常量时归零。
高 VAC 表示相对于动作变化出现了更显著的视觉变化，因此保持“越高越重要”的方向。

### 局部路径效率

`local_path_efficiency` 衡量一个片段内末端的实际路径有多接近起终点直线。
使用同一固定坐标系下、未归一化的 `observation.state[:, 0:3]`，对片段的所有
L 个位置样本（L ≥ 2）计算：

```text
D_net  = ||p[L-1] - p[0]||₂
D_path = sum(||p[t] - p[t-1]||₂, t=1,...,L-1)
local_path_efficiency = D_net / D_path   if D_path > delta_path
                        NaN             otherwise
```

分数在 `[0, 1]`，不进行分位数归一化，计算复杂度为 O(L)。例如净位移 0.20 m、
累计路程 0.25 m 时得分为 0.80；这只是距离比，不表示任务完成率或动作价值。
该项默认关闭；启用时必须显式指定有限正阈值 `delta_path`，单位与原始位置相同，
应结合位置噪声和片段时长设定。以下阈值仅为配置示例：

```yaml
local_path_efficiency:
  delta_path: 0.001
reliability_metrics: [support, progress, action_variation, visual_action_consistency, local_path_efficiency]
```

也可用 `--reliability-metrics` 选择该项，但 YAML 中仍需配置阈值。只配置阈值、
不把它加入 `reliability_metrics`，可仅输出诊断结果。
静止和累计路程不超过阈值的片段不参与此项评分：融合时按每个片段的有效指标数
计算几何平均；若只选了此项且无有效值，则可靠性采用中性值 1，不把静止记为零分。

encode 保存 float64 的 `path_position_sequences.npy` 和 `local_path_efficiency.npy`，
阈值、输入定义及计算版本进入缓存契约，并保存 SHA-256 校验。验证会从原始位置重算，
并检查重叠片段的一致性。图节点及筛选明细包含 `local_path_efficiency`；无效值在
NumPy 中为 NaN，在 JSON/Parquet 明细中为 `null`。`selection_report.json` 的
`local_path_efficiency_summary` 分别记录全池和选中片段的有效分数均值、有效数及无效数。

该项适合辅助比较同类、单阶段的点到点移动，不自动识别任务阶段。低分可能来自合理的
绕障、圆弧开门或往复擦拭；闭环回到起点时可为 0。高分不代表目标正确、运动平滑，
也不评价速度、停顿、Jerk 或末端旋转；两个不同位置样本的得分恒为 1。
应统一采样和预处理方式：删除中间点可能缩短累计路程、抬高得分。
这是此处采用的低成本几何代理指标，不是 S2I 原论文的质量评分公式。

### 低变化驻留比例

`dwell_ratio` 表示片段内末端位置、姿态和夹爪同时低变化的时间占比，
`non_dwell = 1 - dwell_ratio` 是可选可靠性分量，不进行分位数归一化。
对片段内部的 `L−1` 对相邻帧计算：

```text
low_t = (||p_t - p_(t-1)|| / dt_t < position_speed_threshold)
        AND (relative_rotation_angle / dt_t < angular_speed_threshold)
        AND (|g_t - g_(t-1)| / dt_t < gripper_speed_threshold)
dwell_ratio = sum(dt_t * low_t) / sum(dt_t)
```

使用未归一化的 `observation.state`：位置为 `[0:3]`、姿态为 `[3:6]`、夹爪为 `[7]`。
LIBERO 姿态按旋转向量解释，Bridge 按 XYZ 欧拉角解释，均计算最短相对旋转角。
时间戳单位为秒，必须有限且严格递增；状态必须有限，片段至少两帧。
等间隔时公式等价于低变化比较次数除以 `L−1`，例如 `70/100 = 0.70`；
片段首帧不与片段外的前一帧比较。

默认不配置 `dwell`，也不计算该指标。下面展示配置结构，`null` 必须替换为按数据集
校准的有限正数，否则配置校验会报错；不提供通用默认阈值：

```yaml
dwell:
  position_speed_threshold: null  # 原始位置单位/秒，LIBERO、Bridge 通常为 m/s
  angular_speed_threshold: null   # rad/s
  gripper_speed_threshold: null   # 原始夹爪单位/秒，不同数据集量纲可能不同
  gripper_mode: continuous
reliability_metrics: [support, progress, action_variation, visual_action_consistency, non_dwell]
```

如仅需诊断，完整配置 `dwell` 后保持原来的四项 `reliability_metrics`。
二值夹爪显式设置 `gripper_mode: binary`，直接比较相邻状态是否相等，此时可以省略
`gripper_speed_threshold`。这里的夹爪来自观测状态，不能用二值动作指令替代连续观测。

启用计算后，encode 新增 `dwell_state_sequences.npy`、`dwell_timestamps.npy`、
`dwell_ratio.npy`、`non_dwell.npy`；原始缓存用 float64 保存，配有 SHA-256 校验。
graph 节点、`all_clips.parquet` 和 `selected_manifest.jsonl` 均包含两项得分。
`selection_report.json` 的 `dwell_summary` 提供全池与选中片段的得分算术均值，
不是将所有片段时长合并计算的比例。配置、profile 和计算版本写入阶段契约及指纹；
修改配置需重建不兼容缓存，`validate` 会复算并检查缓存、节点、行和报告的一致性。

高驻留比例不等于低价值：持物、等待、接触保持均可能合理。`non_dwell` 仅在显式选择时
影响几何均值，不新增仅凭驻留比例直接删除片段的规则。

所选指标逐片段跳过允许无效的 NaN 分量，对剩余有效项取几何均值；无有效项时取 1：

```text
reliability = clip((product(valid_selected_metrics)) ** (1 / max(valid_metric_count, 1)),
                   quality.min_reliability, 1)
```

融合取所选有效指标乘积的 n 次方根，再应用可靠性下限；局部路径效率为 NaN 时，
n 按片段减一，全部无效时采用中性值 1。该
reliability 会同时进入
coverage seed、关系收益、冗余惩罚和分支保留排序。
初始集合为每个可达运动原语选择
`reliability * assignment` 最大的片段并取并集，使所有原型 coverage 达到全池最大值。
其余预算固定使用可复现的随机多分支搜索，并优化以下目标：

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

coverage seed 始终作为
固定集合；初始化 8 个活动分支，各无放回抽取 10 个片段并记作第 1 轮。以后每个父分支
生成 4 个各补入 10 个片段的子分支，从 32 个子分支中保留 8 个。cooccurrence 关系项仍
使用全部固定片段和当前分支片段。sequence 分支在初始化和每次重组后从零开始，只保留
分支局部 sequence 计数：普通子分支继承父分支累计值，对本轮新增片段按顺序只计算其与
全部固定片段、父分支已有活动片段及本轮更早新增片段之间的原始有向 sequence 边。
固定片段内部以及已有活动片段内部的边不会重复计算；numerator 使用该局部计数，
denominator 仍使用全池 sequence 计数。

分支冗余同样从 0 开始。全部固定片段使用 L2 归一化 embedding 建立共享的 FAISS
`IndexFlatIP`；每个分支为已有活动片段建立独立 Flat 索引。对本轮新增片段按顺序执行
余弦相似度阈值 `range_search`，累计其与全部固定片段、父分支已有活动片段及本轮更早
新增片段之间的可靠性加权惩罚。固定片段内部以及已有活动片段内部的相似对不会重复计算。
惩罚继续使用稀疏 graph 的全局冗余分母；FAISS 取代固定片段随机抽样和分支相似边查找，
把阈值以上的精确相似对纳入增量。最终目标和逐条增益使用获胜分支从最近一次重组开始
积累的 sequence 与冗余增量；coverage seed 和已经提交的固定片段在 sequence 模式下不贡献
最终 relation 增量。`random_multibranch` 严格要求 `faiss-cpu>=1.9`，缺失时不会回退到
其他检索后端。

首次在第 20 轮重组，之后每 10 轮重组：统计活动片段在 8 个分支中的出现次数，从最高
得分分支内取计数最高的 100 个加入固定集合，计数相同时由 `seed` 随机破平局。每个分支
再从自己的未固定片段中按出现计数降序、`reliability` 降序、`sample_id` 升序稳定保留
100 个。重组时分支 sequence 与冗余均清零，每个分支保留的 100 个片段整体作为新批次，
按顺序只与新的全部固定集合和更早保留片段计算 sequence，并按 FAISS 阈值检索规则计算
冗余。分支得分相同时仍由 `seed` 随机破平局；最后一批不足 10 个时只补足预算，达到预算
后直接采用最高得分分支。主状态只保存固定选择掩码、原型覆盖和任务计数，分支仅保存稀疏
更新；8、4、10、20、10 和两个用途不同的 100 都是固定算法常量，不提供额外配置项。

增量 sequence、冗余口径与 `recombination_ranking` 排序规则记录在选择算法元数据中，
selection 产物使用独立 schema 1。缺少该 schema、但算法与上游 graph 指纹均匹配的旧
随机多分支缓存会自动只重建 select 产物，继续复用 scan、encode 和 graph；旧堆选择
产物不迁移且 validator 会拒绝。sequence 随机多分支产物中，`selection_report.json` 的
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
python -m cocore select --config cocore/config_libero90.yaml \
  --reliability-metrics support action_variation
python -m cocore run --config cocore/config_libero90.yaml
```

配置必须显式声明关系类型与权重：

```yaml
reliability_metrics: [support, progress, action_variation, visual_action_consistency, action_jump]

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
  profile: libero
  batch_size: 4096
  max_iter: 100
  tol: 1.0e-4
  num_threads: 4
  use_stop_bucket: true

selection:
  ratio: 0.10
  budget: null
```

片段长度由 profile 固定为 LIBERO 15 帧或 Bridge 7 帧，候选数量、首尾锚定与近似均匀
间隔是 Cocore 固定算法的一部分；
配置中不接受 `clip` section。
Quality 风格编码取代了旧关系编码，因此不再接受顶层 `relation` 或 `normalization`；
`encoding.visual_dim` 固定为 128，`pca_fit_max_samples` 可限制 PCA 拟合样本数。
动作门槛、中心数公式、30 个中心上限、距离分位和权重公式都是 Cocore profile 的固定算法，
不可单独配置；`prototypes` 只接受 `method`、`profile`、`batch_size`、`max_iter`、正数 `tol`、正整数
`num_threads` 和布尔值 `use_stop_bucket`，并明确拒绝旧 `count`、`top_r` 或
`temperature`。`profile` 只接受 `libero` 或 `bridge_v2`；通用配置默认 `libero`。
`batch_size` 只影响
超过 65,536 个训练窗口的大桶；`num_threads` 默认为 4，只并行不超过阈值的小桶，
debug 配置固定为 1。它与 `runtime.num_workers` 相互独立，后者仍只控制 episode 读取
进程。

视觉中心训练会一次物化所有保留窗口的 128 维 `float32` 投影。小桶用完整 KMeans
并行拟合，大桶用 MiniBatchKMeans 串行拟合，避免多个大桶同时占用 CPU 和临时内存。
基础额外内存约为“保留窗口数 × 128 × 4 字节”：LIBERO90 约 106 MiB，Bridge V2
按间隔 2 的 4 帧参考基线估算约 257 MiB；完整 KMeans 拟合小桶时还会产生有界于
65,536 个窗口的工作副本。改变
`num_threads` 不改变 graph 指纹或产物，因此可复用同一 graph 缓存；改变
`profile`、`batch_size`、`max_iter`、`tol` 或 `use_stop_bucket` 会使 graph 缓存失效。
profile、分轴阈值、roll 标签、环绕轴和保留公式同时写入 catalog、各级 manifest 与
graph/select 指纹。

Bridge Orig V2 参考数据（排除空任务 episode）包含 38,660 条有效 episode 和 622,782
个四帧窗口；完整集门槛为 3,114。参考结果为 1,181 个复合标签、25 个保留的非 stop
动作桶和约 594 个叶原型，非 stop 精确覆盖约 60.62%，父类回退约 14.22%，无父类
回退不超过 1.23%，原始 stop 约 23.94%；原子动作与 occurrence 保留质量分别至少为
68.79% 和 84.67%。

可在运行时覆盖选择比例、关系类型与关系权重：

```bash
python -m cocore run --config cocore/config_libero90.yaml \
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
`outputs/cocore/libero90/select-sequence-w1p5-top20pct-random-multibranch/`。
`build-graph`、`select`、`run` 和 `validate` 可通过
`--reliability-metrics METRIC [METRIC ...]` 覆盖 YAML；空集合、重复项或未知指标都会报错。
`--cooccurrence-weight` 已移除，且 Cocore 不接受 RelCore 的 `--prototype-method` 或
`--prototype-gain-metrics` 参数。

快速 CPU 检查：

```bash
python -m cocore run --config cocore/config_debug.yaml --force
```

Cocore 0.19.0 使用 prototype schema 10、profile 固定的 15/8（LIBERO）或 7/4（Bridge）
时间几何、10～30 个桶内视觉中心、可选 stop 桶、无标签
候选诱导子图、65,536 窗口的混合 KMeans 阈值、LIBERO 最大间隔 3、Bridge 最大间隔 2
的动作训练窗口、裁剪 PCA
的 128 维聚类空间、近似均匀候选和原始相邻 sequence 图，并按 episode 持久化完整原始
逐帧 CLIP 特征，选择阶段固定使用 selection schema 3 的随机多分支算法。0.19.0 新增
VAC 原始/归一化产物。当前默认融合 support、progress、AVI、VAC 和 `action_jump`，
并支持其余可选指标；各指标计算版本和参数独立进入缓存契约。指标、AVI 与
VAC 契约进入阶段指纹、manifest 和报告，validator 会从动作序列和逐帧视觉缓存重算
VAC，校验 Top-3、分位缩放、图节点、选择输出及融合结果。版本校验保持严格，因此旧
artifact 不会被接受，升级后应使用 `--force` 重建。Bridge 适配器 0.12.0 使用 Cocore
0.19.0/schema 10；prototype schema 不变，selection schema 升至 3。

## 输出与校验

输出根目录包含 `scan/`、`encode/`、`graph-18-motion-hard-nearest-pca/` 和一个或多个
`select-<关系>-w<权重>-top<比例>pct-random-multibranch/`。选择目录包含：

- scan 目录中的 `episodes.parquet` 与 `clips.parquet`：episode 元数据和可重放的
  近似均匀候选；Cocore scan 不再生成未被后续阶段消费的 `normalization.npz`；
- graph 目录中的 `prototype_catalog.json` 与 `prototype_centers.npy`：动作类别、
  profile、分轴阈值、环绕策略、roll 标签、保留策略、原始计数、训练计数、视觉中心
  数量、距离 q10/q90、投影契约以及按叶 ID 对齐的 128 维中心；
- graph 目录中的 `source_clip_indices.npy`：每个 eligible graph 节点对应的 scan/encode
  候选行号；关闭 stop 桶后它同时记录被排除候选形成的空洞；
- encode 目录中的 `embeddings.npy`、`visual_pca.npz` 和
  `numeric_normalizers.npz`：Quality 融合 embedding 及其可重放参数；
- encode 目录中的 `frame_embeddings/ep<episode_id>.npy`：每个已索引 episode（包括
  短 episode）的完整 `[帧数, CLIP 维度]`、`float32` 逐帧特征；
  `frame_embeddings_index.json` 记录顺序、shape 和 SHA-256；
- encode 目录中的 `visual_half_embeddings.npy`：每个候选两个半段的归一化视觉均值；
  graph 目录中的 `half_action_labels.npy` 记录两个半段各自的原始动作标签；
- encode 目录中的 `action_variation_raw.npy`、`action_variation.npy`、
  `visual_action_consistency_raw.npy` 和 `visual_action_consistency.npy`：候选级 AVI/VAC
  原始值和全候选分位归一化值；

- `selected_manifest.jsonl`：训练入口可直接消费的片段清单；
- `all_clips.parquet`：eligible 筛选池的 support、progress、原始/归一化
  `action_variation`、原始/归一化 `visual_action_consistency`、reliability、运动原语与
  选择诊断，不包含无标签候选；
- `selection_report.json`：coverage、目标分解、任务计数、`branch_search`
  轮次/评估/重组统计、逐轮与逐次重组耗时，以及 scanned、eligible、
  excluded-unlabeled 候选数量；
- `manifest.json`、`run_manifest.json`：Cocore 参数、可靠性指标、阶段目录与指纹；
- `resolved_config.yaml`、`environment.json`：`run` 的完整配置与环境。

使用以下命令独立重算 coverage、逐步边际增益、目标、预算、唯一性、方法统计和清单一致性：

```bash
python -m cocore validate \
  --output-dir outputs/cocore/libero90/select-cooccurrence-w1-top10pct-random-multibranch \
  --config cocore/config_libero90.yaml \
  --no-use-stop-bucket
```

只有生成结果时关闭了 stop 桶，验证时才应传入该开关。若省略 `--config`，CLI 会从
选择目录的 `resolved_config.yaml` 读取重放配置；验证时必须使用生成时相同的 seed 和
`reliability_metrics`，
校验器会重放每个分支的 FAISS 阈值检索并核对增量惩罚。复用
旧版本、旧 Bridge 15/8 几何、profile、阈值、环绕策略、可靠性指标或 stop 设置不同的
输出目录时，应使用 `--force` 重建全部不兼容阶段。

校验还会从 episode 元数据重放近似均匀候选，逐字段核对 `clips.parquet`，逐个检查
帧缓存文件集合、shape、dtype、有限值和 SHA-256，并使用同一 PCA components 重算
投影窗口、中心、距离分位和候选分配。编码中断时临时缓存会被清理，下次从头执行；
只有完整发布的 encode 阶段才会被复用。

当前版本支持本仓库约定的 8 维 LIBERO 与 BridgeData V2 `observation.state`；必须选择
对应 profile。selection 的
parquet/JSONL 导出最终 `prototype_indices`、`prototype_weights`、叶标签、动作标签与
`half_action_labels`，不导出可分解的 action/distance 权重。所有图关系与选择目标直接
消费绝对 `prototype_weights`，不对每行额外归一或截断。

### 可选动作—执行偏差

`action_execution_deviation` 衡量原始控制指令要求的末端平移与实际执行位移的差距。
它是控制指令与反馈的一致性检查，不是动作平滑度或片段价值的直接度量。
默认不计算；仅配置时输出诊断，将 `low_action_execution_deviation` 加入
`reliability_metrics` 后才参与可靠性几何平均融合。

```yaml
action_execution_deviation:
  action_source: original_command
  action_semantics: delta_from_observed_position
  action_scale: [1, 1, 1]  # 仅适用于动作前三维已经是米制位移；否则填写实际每轴转换系数
  alignment_confirmed: true

reliability_metrics: [support, progress, action_variation, visual_action_consistency, action_jump, low_action_execution_deviation]
```

四个配置字段均必须显式填写。`alignment_confirmed: true` 表示使用者已确认：
`observation.state[:, :3]` 是以米为单位的实测位置，动作与位置处于同一固定坐标系，
动作 t 对应位置 t→t+1 的执行区间，且控制器限幅已经在上游正确处理。
`action_scale` 必须是三个有限正数，将原始动作前三维转换为米制期望位移；
它不是从当前数据分布拟合出的归一化参数，也不能替代坐标转换。
程序不会推断动作语义、猜测缩放或为每个片段搜索最优延迟。只有 YAML 完整配置后，
现有 `--reliability-metrics` 参数才能用于选择本指标。

使用 float64，在每个片段内按以下顺序计算：

```text
a_pos[t] = raw_action[t, :3] * action_scale
e[t] = ||(p[t+1] - p[t]) - a_pos[t]||₂
action_execution_deviation_raw = mean(e[0:L-1])
```

L 个位置只有 L−1 个执行区间；末帧动作不参与计算，不跨片段或 episode 比较。
必须先逐步取范数再求平均，以免不同方向的偏差抵消。例如指令为 `(10,0,0)` mm，
实际位移为 `(6,3,0)` mm，偏差为 5 mm，即原始输出 `0.005` m。
原始值越大偏差越大，不是 `[0,1]` 比例；算法复杂度为 O(L)，不需要训练模型。

时间戳必须严格递增，允许不等间隔。位置、动作或时间缺失、形状错误、NaN/Inf
均报错；少于两帧、时间不递增和计算溢出分别记录 `too_short`、
`non_increasing_time`、`overflow`。不可计算时缓存数值为 NaN，JSON/Parquet 导出为 null。

归一化使用所有有效扫描候选的原始偏差，沿用 `encoding.quantile_low/high/epsilon`：

```text
low_action_execution_deviation = 1 - clip((raw - q_low) / (q_high - q_low), 0, 1)
```

分位跨度不超过 epsilon 时赋 1，表示当前池无法区分，不能解释为执行无偏差。
只配置诊断不会改变筛选池或融合分数；选择该指标融合时，不可计算片段排除出图，
入图条件与原型资格、已启用 Jerk 的有效性取交集。过滤不补跨空洞的 sequence 边，
比例预算基于最终入图数量；无候选时报错。

encode 缓存保存 `action_execution_deviation_{positions,actions,timestamps,raw,valid,reason}.npy`
及 `low_action_execution_deviation.npy`，包括原始输入、计算契约与 SHA-256。
graph 节点、`all_clips.parquet`、选中清单保留原始值、融合值与状态。
`action_execution_deviation_summary` 分别汇总 scanned、graph、selected 的米制原始均值、
有效/无效数量和无效原因。启用融合时，`excluded_clips.json` 保留所有排除原因，
`excluded_action_execution_deviation_clips` 记录本项无效数量；总排除数按并集计算，
不能与 Jerk/无标签数量直接相加。validator 会检查重叠输入一致性，并重算原始值、
分位缩放、入图掩码、融合分数、导出及报告。更改配置会使 encode 及下游缓存失效，
复用同一目录时需 `--force` 重建。未配置时不要求新增缓存。

首版仅支持相对**当前实测位置**的平移指令；相对上一目标位置、绝对位置、速度、
关节、力矩、旋转和夹爪不适用。位置目标可能需要逐渐跟踪，接触操作中的偏差也可能是
正常控制行为，因此非零偏差不直接等于示范无效。自由空间与接触操作应分开比较；
当前不自动识别接触阶段，也不设置统一异常阈值。

若动作由前后状态差重标注，代入公式得到的零只是标签构造结果，不能作为执行质量证据。
此时必须提供原始下发指令，否则不要启用；`action_source` 声明错误或不明确时会报错。
程序不能凭数值为零判断来源，真实完全跟踪的零偏差仍有效。
参考：[robosuite 控制器](https://robosuite.ai/docs/source/robosuite.controllers.parts.arm.html)、
[Bridge 官方动作重标注代码](https://github.com/rail-berkeley/bridge_data_v2/blob/main/jaxrl_m/data/bridge_dataset.py)。

### 可选末端运动 Jerk

在 `reliability_metrics` 中加入 `eef_jerk` 即启用计算和融合，无需模型或额外阈值：

```yaml
reliability_metrics: [support, progress, action_variation, visual_action_consistency, eef_jerk]
```

CLI 使用 `--reliability-metrics support progress action_variation visual_action_consistency eef_jerk`。
`eef_jerk` 为可选项，未选择时不计算，也不要求 Jerk 缓存。

输入为固定坐标系下实际末端位置 `observation.state[:, :3]`（米）和原始时间戳（秒），
不使用缩放后的 state 或动作指令。使用 float64，在每个片段内部计算：

```text
j_t = (p_t - 3*p_(t-1) + 3*p_(t-2) - p_(t-3)) / Δt³
S_jerk = sum(||j_t||₂) / (L-3)
```

先取向量模长再求平均；原始 `eef_jerk_raw` 单位为 m/s³，非 `[0,1]` 比例。
`Δt` 为相邻时间差中位数，所有时间差必须为正且满足 `rtol=1e-4, atol=1e-8` 的等间隔检查。
片段短于四帧、重复时间、不等间隔或数值溢出时，分别记录 `too_short`、
`non_increasing_time`、`non_uniform_time`、`overflow`。无效值在 NPY 中为 NaN、JSON 中为 null，
不记为零。位置缺失、NaN/Inf 和适配器拒绝的时间戳仍报错；同时配置 dwell 时保留其校验。
当前扫描器仅生成完整的 LIBERO 15 帧或 Bridge 7 帧候选，短 episode 继续按原规则跳过。

融合分数使用扫描候选中 Jerk 有效片段的分位数，沿用 `encoding.quantile_low/high` 和 `epsilon`：

```text
eef_jerk = 1 - clip((eef_jerk_raw - q_low) / max(q_high - q_low, epsilon), 0, 1)
```

分位数跨度不超过 epsilon 时统一赋 1，表示当前池中不能区分。
只有有效片段参与 Jerk 融合，最终入图条件为“有原型标签且 Jerk 有效”。过滤保留原始邻接关系，
不跨被排除片段补 sequence 边。比例预算以入图候选数计算，空候选或预算超限报错。

encode 新增 `eef_jerk_positions.npy`、`eef_jerk_timestamps.npy`、`eef_jerk_raw.npy`、
`eef_jerk.npy`、`eef_jerk_valid.npy`、`eef_jerk_reason.npy`。缓存包含计算契约和 SHA-256，
验证时检查原始缓存重叠一致性并重算数值、有效掩码及分数。graph 的
`prototype_eligible_mask.npy` 保存原型资格，回放核验与 Jerk 掩码的交集。
节点、all-clips 和 selected manifest 中记录原始值、融合值与状态。
`excluded_clips.json` 按 sample_id 记录被排除候选及全部原因。
报告分别记录 `excluded_unlabeled_clips`、`excluded_jerk_clips` 和并集 `excluded_clips`，
前两项可能重叠，不能相加。`eef_jerk_summary` 汇总有效扫描池/选中片段的原始均值及无效原因计数。
启用指标会改变 encode 及下游指纹，复用同一目录时需要 `--force` 重建。

首版不平滑、不计算 RMS 或旋转 Jerk。三阶差分对位置噪声敏感；原始 Jerk 还受运动幅度和时长影响，
同轨迹执行时间延长 k 倍会使 Jerk 缩小到 `1/k³`。归一化分数仅表示本次候选池的相对水平，
不能据此跨任务统一比较；静止片段也可得到零 Jerk，应结合任务进展和交互事件判断数据价值。

### 局部折返率（可选）

`local_backtracking_rate` 衡量片段内相邻两段末端位移出现明显反向的比例。
输入是同一固定坐标系中的实际末端位置 `observation.state[:, :3]`（当前 LIBERO 与
Bridge V2 以米为单位）及秒单位时间戳。使用片段内全部采样点、float64 原始值，
不使用动作指令或归一化 state，不平滑、不重采样，也不跨片段或静止步补算。

```text
d[t] = p[t+1] - p[t]
V = {t: ||d[t]|| > epsilon_p 且 ||d[t+1]|| > epsilon_p}
c[t] = clip(dot(d[t], d[t+1]) / (||d[t]|| * ||d[t+1]||), -1, 1)
local_backtracking_rate = count(c[t] < -eta, t ∈ V) / |V|
low_local_backtracking = 1 - local_backtracking_rate
```

分母是有效方向比较次数 `|V|`，只有所有位移均通过检查时才等于 `L−2`。
例如 80 次有效比较中有 12 次折返，折返率为 `0.15`；它不代表时间或路程浪费了 15%。
默认 `eta=0.5` 对应夹角严格大于 120°，直角及恰好达到阈值的转向不计入。
`epsilon_p` 必须显式填写有限正数，单位与原始位置相同；`eta` 可配置，须满足
`0 ≤ eta < 1`。二者均不接受布尔值，配置块不接受其他键。

```yaml
# 先按位置噪声、采样间隔及运动尺度校准 epsilon_p，再取消注释。
# local_backtracking:
#   epsilon_p: null  # 必填；替换为有限正数，当前数据单位为米
#   eta: 0.5        # 可省略
```

存在配置块即可输出诊断；需要影响可靠性时，再将 `low_local_backtracking` 加入
`reliability_metrics`，或使用 CLI 的 `--reliability-metrics` 选择它。保持原默认分量的示例：

```yaml
reliability_metrics: [support, progress, action_variation, visual_action_consistency, action_jump, low_local_backtracking]
```

该例还需填写上述配置块；只选择融合分量而未配置阈值会报错。默认不计算、不参与融合。
融合使用 `1−折返率` 的逐片段几何平均；无效分量跳过，若选中的所有分量均无效则沿用
中性可靠性 `1`，保留现有最低可靠性裁剪。不会仅因该指标不可评价而排除候选。

导出字段为 `local_backtracking_rate`、`low_local_backtracking`、
`local_backtracking_valid_count`（有效比较次数）、`local_backtracking_count`（折返次数）、
`local_backtracking_valid` 和 `local_backtracking_reason`。两项次数为整数。
无有效比较时，分数为 NaN、次数为零，原因为 `no_valid_comparisons`。
少于三帧、非递增时间、不均匀采样或计算溢出也不可评价，原因依次为 `too_short`、
`non_increasing_time`、`non_uniform_time`、`overflow`。等间隔检查使用时间差中位数，
容差 `rtol=1e-4, atol=1e-8`；位置或时间戳缺失、形状错误及非有限值直接报错。

encode 保存 `local_backtracking_positions.npy`、`local_backtracking_timestamps.npy`
及上述六项字段的 NPY 数组；位置、时间戳及分数为 float64，次数为 int64。
缓存契约记录阈值、公式、采样约定、profile 和 SHA-256；`validate` 重算指标并核对
重叠片段输入、节点映射、融合值与导出记录。配置变化需要按现有机制使用 `--force` 重建。
图节点、`all_clips.parquet` 和 `selected_manifest.jsonl` 均保留六项字段，JSON 无效分数为
`null`。`selection_report.json` 的 `local_backtracking_summary` 分别统计 `scanned`
（全部扫描候选）、`graph`（入图候选）、`selected`（已选片段）：有效／无效片段数、
无效原因计数、有效片段的 `rate_mean`、有效比较总次数 `valid_comparison_count` 和
折返总次数 `backtracking_count`。`rate_mean` 是有效片段等权均值，不是汇总次数之比。

高折返率不等于低价值：擦拭等往复任务或纠偏动作可能需要反向移动，应在同类、同阶段
片段间比较，并结合有效比较次数解释。改变采样频率或预处理会改变结果；此指标不含旋转
或夹爪动作，缓慢绕圈及“前进—停顿—后退”也可能漏检，因此低折返率不能证明没有绕行。
局部路径效率看整体多走多少路，局部折返率看相邻方向反转多频繁。这里采用的是工程代理
指标，不是 S2I 原论文的评分公式。

### 高频抖动能量占比（可选）

`high_frequency_jitter` 分析片段内部固定坐标系下的末端三轴平移速度。
输入为原始 `observation.state[:, :3]` 与秒单位时间戳，相邻位置差分产生 `N=L−1`
个速度样本。逐轴去均值，使用周期 Hann 窗和单边功率谱密度，再合并三轴功率；
不先取速度模长，不包含旋转或夹爪，不做平滑、重采样或零填充。

- `high_frequency_ratio`：`E_HF / (E_total + epsilon)`，分母排除直流，分子严格取 `f > cutoff_hz`。
- `high_frequency_rms`：`sqrt(E_HF)`；`total_fluctuation_rms`：`sqrt(E_total)`，单位均为原始位置单位/秒。
- `high_frequency_resolution_hz`：`fs/N`，采样间隔取片段时间差的中位数。
- `low_high_frequency_jitter`：有效时等于 `1−high_frequency_ratio`，可选参与可靠性融合。
- `high_frequency_valid`、`high_frequency_reason`：有效性与原因。

配置块存在即可输出诊断；加入 `reliability_metrics` 或通过 `--reliability-metrics`
选择 `low_high_frequency_jitter` 才改变融合。三个校准参数必须显式填写有限正数：

```yaml
# 将 null 替换为同机器人、同采样率、同类任务下校准的数值后启用。
# high_frequency_jitter:
#   cutoff_hz: null                    # Hz，必须严格低于 fs/2
#   noise_floor_rms: null              # 总速度波动的噪声底，原始位置单位/秒
#   max_frequency_resolution_hz: null  # 可接受的最大 fs/N，Hz
#   epsilon: 1.0e-12                   # 可省略，单位与积分功率一致
# reliability_metrics: [support, progress, action_variation, visual_action_consistency, low_high_frequency_jitter]
```

速度样本不足 3 个、频点间隔超限、截止频率两侧缺少频点或总波动 RMS 不超过噪声底时，
分别记录 `too_short`、`coarse_resolution`、`missing_band`、`low_fluctuation`，按此顺序
选择首个原因。能计算的原始诊断仍保留，无效融合分量为 NaN；导出 JSON/Parquet 时为 `null`。
几何平均逐片段跳过无效分量；所有启用项均无效时取 `1`。高频指标无效不会排除图节点；
已有 Jerk 无效排除规则仍独立生效。

时间戳必须严格递增且等间隔（`rtol=1e-3, atol=1e-8`，与 Jerk 的容差独立）。
缺失或非有限输入、非法时间戳、超出奈奎斯特范围的截止频率及数值溢出会报出片段标识。
LIBERO 的 15 帧只有 14 个速度样本，Bridge 的 7 帧只有 6 个速度样本，应结合实际
`fs/N` 校准可接受分辨率；不会自动延长片段或把 3 Hz 设为通用阈值。

encode 独立缓存 float64 位置、时间戳、诊断和融合值，以及布尔有效性和原因字符串。
参数与算法契约进入缓存指纹，`validate` 检查校验和、重叠输入一致性并重新计算指标和融合。
`high_frequency_jitter_summary` 分别列出 `scanned`（扫描候选）、`graph`（图内片段）
及 `selected`（选中片段）的有效数、无效数、原因计数和有效片段均值。

这是抖动代理指标，不是片段价值或机械能。高占比不代表大幅抖动，必须结合绝对强度；
往复任务也可能有高频运动。Jerk 衡量加速度变化强度，本指标衡量速度波动的频率组成，
两者不等价。本指标不使用分位数归一化，也不自动按占比删除片段。


### 异常动作跳变率（默认启用）

`action_jump_rate` 衡量片段内部相邻动作变化超过统一阈值的比例；融合分量
`action_jump = 1 - action_jump_rate`，保持越高越好的方向，不再进行分位数归一化。

输入为原始动作，排除 `quality.gripper_action_index` 指定的夹爪维度（默认最后一维）。
不复用截断到 `[0,1]` 的动作编码，避免掩盖极端跳变。用本次加载的全部 episode
（受 `runtime.max_episodes` 限制）的连续动作，按帧统一计算各维总体标准差 `s_i`：

```text
J_t = sqrt(mean_i(((a[t,i] - a[t-1,i]) / max(s_i, epsilon))^2))
tau = quantile(all_episode_internal_J, threshold_quantile, method="linear")
action_jump_rate = count(J_t > tau within clip) / (L - 1)
action_jump = 1 - action_jump_rate
```

尺度不按片段计算；阈值标定不跨 episode，也不会因候选重叠重复计数。片段起始帧不与
片段外前一帧比较。101 帧中有 8 次严格超过阈值时，跳变率为 `0.08`，融合值为 `0.92`；
等于阈值不计异常。静止连续动作的跳变率为零，即使二值夹爪正常切换也不会触发此项。

```yaml
reliability_metrics: [support, progress, action_variation, visual_action_consistency, action_jump]
# 选择 action_jump 时自动补齐以下默认值；可省略整个配置块。
action_jump:
  threshold_quantile: 0.99
  threshold: null       # 非空时优先使用固定阈值，须有限且非负
  epsilon: 1.0e-8       # 须有限且为正；分位参数须满足 0 < q < 1
```

显式选择原四项且不提供配置块时，不计算此项；仅提供配置块而不选择 `action_jump`
时，输出诊断但不改变融合。CLI 可用 `--reliability-metrics` 选择此项，标定参数通过 YAML
配置。其他指标的 NaN 跳过、Jerk 无效候选排除规则保持独立；跳变率不新增硬删除条件。

编码缓存以 float64 保存完整参考池的连续动作拼接数组、尺度、阈值、片段跳变率和
融合值，以 int64 保存 episode ID、边界偏移、连续维度索引和比较样本数，文件均以
`action_jump` 开头。计算契约及 SHA-256 存于 encode manifest；校验时检查扫描参考池
边界并重算标定和分数，核对节点、all-clips、selected 及各阶段契约。启用或修改配置会
改变 encode 及下游指纹，复用不兼容目录须 `--force`；显式旧配置保留原缓存契约。

假设同次运行采用相同动作表示和控制周期，旋转按动作数值直接作差。非有限动作、
维度不一致、非法夹爪索引、无连续维度或计算溢出时报错。单帧 episode 只贡献尺度样本，
不贡献阈值比较；指标函数要求片段至少两帧。

当前数据集 P99 是工程标定起点，并非可信参考集或通用最优阈值，需要结合机器人、
动作表示、控制频率和任务调整。此项衡量动作指令变化；技能切换和快速纠正也可能
产生高跳变率，不能单独据此删除数据，也不能等同于实际运动不平滑。它与使用末端
位置及时间戳的 Jerk、高频抖动指标具有不同含义。
