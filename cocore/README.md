# Cocore：运动原语关系筛选

Cocore 从 LeRobot v2 LIBERO episode 中选择固定预算的 15 帧片段。它复用 RelCore
的通用编码组件和稀疏图能力，但独立管理编码流水线、两级动作原型、配置、缓存与输出
产物。

每个 15 帧片段固定拆为共享中间帧的前半 `[0..7]` 和后半 `[7..14]`。一次图像遍历和
一次视觉模型前向同时生成原有关系 embedding，以及两个半片段各自 8 帧视觉特征的直接
均值。关系 embedding 继续用于可靠性和相似图；纯视觉均值只用于二级动作聚类。

动作原型分为两级。一级直接分类每个片段的 `[0→7]` 和 `[7→14]` 两个动作，因此动作
占比的统计母体固定为 `2 * 片段数`，不再对 episode 做逐时步 horizon 采样。一级严格
保留半片段占比 `p > 0.005` 的动作，并分别软化前后半的低频标签。二级将每个半片段与
自己的一级动作对齐，在动作桶内使用 L2 归一化后的视觉均值和一级软权重执行加权
MiniBatchKMeans：

```text
K = min(bucket_size, floor(3 + 2 * log2(200p)))
M = min(K, floor(2 + 1.5 * log2(200p)))
```

桶内全部半片段到全部 K 个中心的欧氏距离共同确定 10%/90% 分位点；距离权重从 1 线性
反向映射到 0.3，并截断在 `[0.3, 1]`。每个半片段保留距离权重最大的 M 个子簇，最终
原型权重等于一级动作权重乘以距离权重。结果合回 15 帧图节点时，前后半命中同一叶
原型只保留组合权重较大的结果，平局固定以前半优先。低频 `stop` 继续作为单一 fallback
原型。

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
objective:
  relation: cooccurrence  # 或 sequence
  relation_weight: 1.0

prototypes:
  method: motion_primitives
  batch_size: 4096
  max_iter: 100
```

片段长度与步长是 Cocore 固定算法的一部分，均为 15；配置中不再接受 `clip` section。
二级 K/M、Top-M 和距离权重范围是 Cocore 固定算法，不接受 RelCore 的
`count`、`top_r` 或 `temperature` 配置。

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

Cocore 0.6.0 引入独立编码流水线、半片段视觉均值二级聚类和 catalog v3。0.5.x 及更早
版本生成的 encode、graph 和 selection 缓存不能继续复用或校验；保留原输出目录时需要
通过 `--force` 重新生成。

## 输出与校验

输出根目录包含 `scan/`、`encode/`、`graph-12-motion-primitives/` 和一个或多个
`select-<关系>-w<权重>-top<比例>pct/`。选择目录包含：

- graph 目录中的 `prototype_catalog.json` 与 `prototype_centers.npy`：动作类别、
  半片段二级聚类诊断和按叶原型 ID 对齐的视觉中心；
- encode 目录中的 `visual_half_embeddings.npy`：`[片段, 前后半, 视觉维度]` 的 8 帧
  均值；graph 目录中的 `half_action_labels.npy` 是离线强校验使用的内部原始动作标签；

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

当前版本只支持本仓库约定的 8 维 LIBERO `observation.state`；一级动作不会退回全局
KMeans，二级聚类固定依赖半片段视觉均值。selection 的 parquet/JSONL 不导出内部前后
半来源字段。
