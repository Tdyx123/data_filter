# SQCN：15 帧片段数据价值评价

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
`[sum(v_0..v_14), v_14-v_0]`，经标准化、PCA 和 L2 归一化得到 256 维
embedding。该向量与动作/状态的 mean、std、max 池化特征拼接，再经第二级
标准化、PCA 和 L2 归一化得到最终 128 维 embedding。

## 指标

- Quality：动作平滑、状态转移、运动效率按 `0.4/0.3/0.3` 合成。
- Coverage：候选对全部参考片段的平均 RBF 亲和度，`sigma` 默认使用 median
  heuristic。
- Novelty：候选集合内排除自身的 kNN 平均距离，默认 `k=20`。

三个指标都按候选分布的 1%/99% 分位缩放到 `[0, 1]`。

## 输出

默认输出到 `outputs/sqcn/<dataset>/`：

- `fragment/scores.csv`：候选片段的 Q/C/N/SQCN，按 SQCN 降序；
- `fragment/embeddings.npy`：与 CSV 行严格对齐的 128 维 embedding；
- `fragment/features.pkl`：中间融合特征、原始指标和归一化边界；
- `reference/segments.csv`、`reference/embeddings.npy`：Coverage 参考集；
- `encoder_artifacts/`：视觉 PCA、融合 PCA、数值归一化器；
- `run_manifest.json`：数据/配置指纹、模型、规模、参数和输出路径。

传入 `--output-dir PATH` 时，`PATH` 就是最终运行目录，不会再追加数据集名称；
上述所有产物都会写入该目录。未传入时继续使用配置中的
`output.root/<dataset>/`。

## 独立多样化 Filter

完成 SQCN 计算后，可以根据 `sqcn` 原始分数和同行 128 维 embedding 做独立
的多样化重排。比例在整个候选片段集合上计算，选择数量按向上取整确定；原始
`fragment/` 和 `run_manifest.json` 不会被修改。Top 10% 和 Top 20% 分别运行：

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
在原字段后追加 `filter_rank`、`adjusted_score` 和 `max_penalty`。

未传 `--seed` 时，每次运行都会生成新的随机 seed。需要复现某次结果时，从
该次 `filter_manifest.json` 的 `algorithm.seed` 读取数值并显式传入：

```bash
.venv/bin/python -m sqcn.filtering \
  --input-dir /data/results/libero90_sqcn \
  --percent 10 \
  --seed 123456 \
  --force
```

可以用 `--output-dir PATH` 指定精确输出目录；已有目录必须显式传 `--force`
才能替换。Filter 固定使用 `lambda=1.0`，`sigma` 为原始 SQCN Top 100 条
embedding 的两两欧氏距离均值。若目标数量不超过 100，结果按照算法定义直接
取原始分数 Top-N，不进入相似性惩罚循环。

LIBERO-90 的固定规模基线是 35,164 个参考片段、46,705 个候选片段、10,651
个重合片段以及 71,218 个 PCA 去重并集片段。
