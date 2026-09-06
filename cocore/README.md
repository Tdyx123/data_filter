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
`mean/std/max` 及 `start/episode_length` 拼接并做行 L2 归一化。该 embedding 用于
可靠性 support 和相似图。Encode 还保留原始逐帧 CLIP 缓存。LIBERO 将候选拆成共享
第 7 帧的 `[0..7]`、`[7..14]` 两个 8 帧半段；Bridge 将 7 帧候选拆成共享第 3 帧的
`[0..3]`、`[3..6]` 两个 4 帧半段，分别缓存原始 CLIP 空间中的归一化均值。

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

可靠性目录为 `support`、`progress`、`action_variation`、
`visual_action_consistency`、`non_dwell`、`eef_jerk` 和 `local_path_efficiency`，
顶层 `reliability_metrics` 可选择非空、无重复的指标子集，默认仅启用前四项；`non_dwell` 需要显式配置 `dwell` 阈值。输入顺序会按
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

任意所选指标集合都统一取几何均值：

```text
reliability = clip((product(selected_metrics)) ** (1 / metric_count),
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
reliability_metrics: [support, progress, action_variation, visual_action_consistency]

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
VAC 原始/归一化产物，默认采用四指标的任意非空子集几何均值融合；另支持显式配置的
第五项 `non_dwell`，其计算版本和阈值独立进入缓存契约。指标、AVI 与
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

### 可选末端运动 Jerk

在 `reliability_metrics` 中加入 `eef_jerk` 即启用计算和融合，无需模型或额外阈值：

```yaml
reliability_metrics: [support, progress, action_variation, visual_action_consistency, eef_jerk]
```

CLI 使用 `--reliability-metrics support progress action_variation visual_action_consistency eef_jerk`。
默认四项不变，未选择 `eef_jerk` 时不计算，也不要求 Jerk 缓存。

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
