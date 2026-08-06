# SQCN：15 帧片段数据价值评价

SQCN 0.3.0 起，Quality、窗口、融合编码、缩放和多样化 Filter 算法统一由
`segment_filter_core` 提供。旧的 `sqcn.quality`、`sqcn.encoding`、
`sqcn.scaling`、`sqcn.sampling` 和 `sqcn.filtering.algorithm` Python 导入路径
已移除；调用公共算法时应直接导入 `segment_filter_core`。包含旧类路径的 SQCN
pickle/cache 与 0.3.0 不兼容，需要传 `--force` 重建。SQCN 两个命令行入口保持不变。

SQCN（Segment Quality-Coverage-Novelty Score）只评价完整 15 帧片段，不评价
完整轨迹。最终分数固定为：

```text
SQCN = 0.8 * Quality + 0.1 * Coverage + 0.1 * Novelty
```

## 运行

安装独立依赖并运行 LIBERO-90：

```bash
pip install -r sqcn/requirements.txt
.venv/bin/python -m sqcn \
  --config sqcn/config_libero90.yaml \
  --output-dir /data/results/libero90_sqcn \
  --force
```

最终分数文件为
`/data/results/libero90_sqcn/fragment/scores.csv`。快速检查前两个 episode：

```bash
.venv/bin/python -m sqcn \
  --config sqcn/config_libero90.yaml \
  --output-dir /data/results/libero90_sqcn_smoke \
  --max-episodes 2 \
  --force
```

SQCN 强制使用配置中的单路相机和本地 CLIP ViT，不会访问 Hugging Face，模型
无法加载时也不会退回像素编码。数据集必须同时包含动作和至少一路向量状态。

## 片段与 embedding

候选片段采用长度 15、stride 15 的窗口，并追加一个与轨迹末尾对齐的完整窗口。
Coverage 参考片段按轨迹长度确定数量，在合法起点范围内覆盖首尾并均匀放置：

- `15 <= L < 30`：1 个；
- `30 <= L <= 90`：`round_half_up(L / 15)`；
- `91 <= L <= 180`：`round_half_up(L / 30) + 3`；
- `L > 180`：`round_half_up(L / 60) + 6`。

每帧图像经 CLIP ViT 编码并 L2 归一化。片段视觉特征为
`[sum(v_0..v_14), v_14-v_0]`；CLIP ViT-B/32 的 512 维帧特征由此形成
1024 维片段特征，再经标准化和唯一一次 PCA 得到 128 维视觉特征。最终按
`[视觉, 状态 mean/std/max, 动作 mean/std/max, 起点位置/轨迹帧数]` 拼接并做
一次 L2 归一化。最终维度随状态和动作维度推导，LIBERO-90 为 174 维。

## 指标

- Quality：动作平滑、状态转移、运动效率按 `0.4/0.3/0.3` 合成。
- Coverage：候选对全部参考片段的平均 RBF 亲和度，`sigma` 默认使用 median
  heuristic。
- Novelty：候选集合内排除自身的 kNN 平均距离，默认 `k=20`。

三个指标都按候选分布的 1%/99% 分位缩放到 `[0, 1]`。

## 输出

默认输出到 `outputs/sqcn/<dataset>/`：

- `fragment/scores.csv`：候选片段的 Q/C/N/SQCN，按 SQCN 降序；
- `fragment/embeddings.npy`：与 CSV 行严格对齐的最终 embedding；
- `fragment/features.pkl`：最终 L2 前的拼接特征、原始指标和归一化边界；
- `reference/segments.csv`、`reference/embeddings.npy`：Coverage 参考集；
- `encoder_artifacts/`：视觉 PCA 和数值归一化器；
- `run_manifest.json`：数据/配置指纹、模型、规模、参数和输出路径。

传入 `--output-dir PATH` 时，`PATH` 就是最终运行目录，不会再追加数据集名称；
上述所有产物都会写入该目录。未传入时继续使用配置中的
`output.root/<dataset>/`。

## 独立多样化 Filter

完成 SQCN 计算后，Filter 默认根据 `sqcn` 原始分数和同行 embedding 做独立的
多样化重排。Filter 按源 `run_manifest.json` 的 `embedding_dim` 校验实际维度，
并兼容缺少该字段的旧 128 维结果。比例在整个候选片段集合上计算，选择数量按
向上取整确定；原始 `fragment/` 和 `run_manifest.json` 不会被修改。Top 10%
和 Top 20% 分别运行：

```bash
.venv/bin/python -m sqcn.filtering \
  --input-dir /data/results/libero90_sqcn \
  --percent 10

.venv/bin/python -m sqcn.filtering \
  --input-dir /data/results/libero90_sqcn \
  --percent 20
```

默认分别写入 `filter/top10pct/` 和 `filter/top20pct/`。每个目录包含过滤后的
`scores.csv`、与其行顺序严格对齐的 `embeddings.npy`，以及记录源文件哈希、
实际随机 seed、惩罚参数和选择摘要的 `filter_manifest.json`。过滤后的 CSV
在原字段后追加 `filter_rank`、`adjusted_score` 和 `knn_penalty`。0.4.0 版
filter 结果使用固定候选池和单点 silent 晋升状态机；旧结果与当前筛选轨迹不兼容，
需要重新生成。输入片段总数和按比例向上取整后的目标数量都必须至少为 100。

传入 `--quality-only` 时，只把多样性重排使用的基础分数从 `sqcn` 切换成
`quality`；embedding、相似性惩罚、seed 和其余产物流程保持不变：

```bash
.venv/bin/python -m sqcn.filtering \
  --input-dir /data/results/libero90_sqcn \
  --percent 10 \
  --quality-only
```

quality 模式默认写入 `filter/quality-top10pct/`，避免覆盖相同比例的 SQCN
结果。Python 调用可使用
`filter_sqcn_run(input_dir, 10, quality_only=True)`。manifest 的
`algorithm.score_column` 会记录本次使用的 `quality` 或 `sqcn`，ordering
也会同步声明对应的基础分数字段。

未传 `--seed` 时，每次运行都会生成新的随机 seed。seed 只影响 silent 片段晋升
candidate 时用于补齐更新计数的均匀随机抽样。需要复现某次结果时，从该次
`filter_manifest.json` 的 `algorithm.seed` 读取数值并显式传入：

```bash
.venv/bin/python -m sqcn.filtering \
  --input-dir /data/results/libero90_sqcn \
  --percent 10 \
  --seed 123456 \
  --force
```

可以用 `--output-dir PATH` 指定精确输出目录；已有目录必须显式传 `--force`
才能替换。Filter 固定使用 `lambda=1.0`，`sigma` 为当前基础分数 Top 100 条
embedding 的两两欧氏距离均值。Filter 先按基础分数降序、`sample_id` 升序选择
Top 100；目标恰好为 100 时直接返回。否则，所有剩余片段先对这 100 个 selected
片段计算惩罚、记录 `update_count=100` 并进入 frozen silent heap，再取 silent
Top 100 形成 candidate 集合。

之后每轮选择 candidate Top 1，将它加入 selected 并更新其余 candidates；若尚未
达到目标，再从 silent 晋升 Top 1。晋升片段必须满足
`ceil(100 + log2(selected_count - 100))` 次更新，不足部分从完整 selected 集合中
均匀、批内无放回抽样补齐。silent 片段在晋升前不实时更新；silent 耗尽后继续
排空 candidates。每个 candidate 维护已遇到 reference 中 RBF similarity 最高的
5 个不同片段，相似度并列时按 `sample_id` 升序取舍。惩罚为
`sum(similarity * score) / 有效近邻数`，不足 5 个时按实际近邻数计算，且不会
保留历史最大惩罚。

LIBERO-90 的固定规模基线是 35,164 个参考片段、46,705 个候选片段、10,651
个重合片段以及 71,218 个 PCA 去重并集片段。
