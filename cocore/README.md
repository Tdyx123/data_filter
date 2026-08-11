# Cocore：运动原语共现筛选

Cocore 从 LeRobot v2 LIBERO episode 中选择固定预算的 15 帧片段。它复用 RelCore
的两遍编码、关系 embedding、运动原语软分配和稀疏图构建，但拥有独立配置、缓存与
输出根目录。

可靠性固定为 `sqrt(support * progress)`。初始集合为每个可达运动原语选择
`reliability * assignment` 最大的片段并取并集，使所有原型 coverage 达到全池最大值。
其余预算使用固定宽度的束搜索，优化：

```text
score(S) = w * a(S)^T C a(S) - redundancy(S)
a(S) = sum(reliability_i * assignment_i)
```

其中 `C` 是全池运动原语共现先验，冗余与 RelCore 使用相同的可靠性加权相似边定义。
根节点 rollout 8 个结果，后续每个节点 rollout 4 个结果；每层对选择集合精确去重后
保留 top 8。

## 运行

```bash
pip install -r cocore/requirements.txt

python -m cocore scan --config cocore/config_libero90.yaml
python -m cocore encode --config cocore/config_libero90.yaml
python -m cocore build-graph --config cocore/config_libero90.yaml
python -m cocore select --config cocore/config_libero90.yaml
python -m cocore run --config cocore/config_libero90.yaml
```

可在运行时覆盖选择比例与共现权重：

```bash
python -m cocore run --config cocore/config_libero90.yaml \
  --selection-ratio 0.20 \
  --cooccurrence-weight 1.5
```

上述选择写入 `outputs/cocore/libero90/select-w1p5-top20pct/`。Cocore 不接受
RelCore 的 `--reliability-metrics`、`--prototype-method` 或
`--prototype-gain-metrics` 参数。

快速 CPU 检查：

```bash
python -m cocore run --config cocore/config_debug.yaml --force
```

## 输出与校验

输出根目录包含 `scan/`、`encode/`、`graph-12-motion-primitives/` 和一个或多个
`select-w<权重>-top<比例>pct/`。选择目录包含：

- `selected_manifest.jsonl`：训练入口可直接消费的片段清单；
- `all_clips.parquet`：全池 support、progress、reliability、运动原语与选择诊断；
- `selection_report.json`：coverage、目标分解、残余配额、束搜索层统计与最终 beam；
- `manifest.json`、`run_manifest.json`：Cocore 参数、阶段目录与指纹；
- `resolved_config.yaml`、`environment.json`：`run` 的完整配置与环境。

使用以下命令独立重算 coverage、目标、预算、唯一性、残余配额和清单一致性：

```bash
python -m cocore validate \
  --output-dir outputs/cocore/libero90/select-w1-top10pct \
  --config cocore/config_libero90.yaml
```

第一版只支持本仓库约定的 8 维 LIBERO `observation.state`，不会退回 KMeans。
