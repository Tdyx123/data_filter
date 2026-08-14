# Cocore BridgeV2

`cocore_bridge_v2` 是仓库内的 BridgeData V2 专用 Cocore 命令包。它不复制
Cocore 的编码、运动原语、关系目标或惰性最大堆算法，而是固定 Bridge 数据契约后
调用现有 `cocore.pipeline`。输出仍是 Cocore artifact，可直接交给现有训练入口和
`cocore` 校验器消费。

## 固定数据契约

- 默认数据路径：`/data/dwb/datasets/bridge_orig_1.0.0_lerobo`；
- LeRobot `v2.0`、WidowX、5 Hz；
- 只读取 `observation.images.image_0`、8 维 `observation.state` 和 7 维 `action`；
- 排除任务名为空的 episode，再应用 `--max-episodes`；
- 固定使用 15 帧片段、步长 15 和 Cocore 两级动作原型；一级占比直接统计片段的前后
  两半，二级原型使用各半 8 帧的视觉均值，并在对应一级动作桶内单独聚类；
- 全局选择，不施加逐任务配额。

Bridge 为 5 Hz，因此 15 帧片段覆盖约 3 秒，现有运动原语的 7–8 帧比较跨度约为
1.4–1.6 秒。本适配包不重采样，也不改变 Cocore 的帧级语义。

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
  --selection-ratio 0.10
```

关系可选择 `sequence` 或 `cooccurrence`。`select`、`run` 和 `validate` 的
`--selection-ratio` 默认是 `0.10`。执行阶段还支持：

- `--dataset-path PATH`：覆盖默认挂载点，但目标必须满足同一 Bridge schema；
- `--output-dir PATH`：覆盖默认输出根目录；
- `--max-episodes N`：在排除空任务后只处理前 N 条有效 episode；
- `--force`：按 Cocore 的缓存规则重建不兼容阶段。

首版不接受任意 YAML `--config`，以防绕过固定相机、空任务策略或运动原语契约。

## 输出与验证

默认输出根目录是：

```text
outputs/cocore_bridge_v2/bridge_orig_1.0.0
```

其中包含 `scan/`、`encode/`、`graph-12-motion-primitives/` 和
`select-<relation>-w<weight>-top<ratio>pct/`。选择目录继续提供
`selected_manifest.jsonl`、`all_clips.parquet`、`selection_report.json`、
`manifest.json` 和 `run_manifest.json`；encode 目录提供 Quality 融合
`embeddings.npy`、`visual_pca.npz`、`numeric_normalizers.npz`、按 episode 分片的
`frame_embeddings/`、对应索引以及 `visual_half_embeddings.npy`。graph 目录提供分层
`prototype_catalog.json`、`prototype_centers.npy` 和内部校验用
`half_action_labels.npy`。manifest 的生产者仍为 `cocore`。

验证时必须重复传入生成该选择结果时使用的目标与比例。验证只读取 artifact，不访问
源数据集：

```bash
python -m cocore_bridge_v2 validate \
  --output-dir \
    outputs/cocore_bridge_v2/bridge_orig_1.0.0/select-sequence-w1-top10pct \
  --relation sequence --relation-weight 1.0 \
  --selection-ratio 0.10
```

## 运行基线

正式数据预期包含 53,192 条源 episode。排除 14,532 条空任务 episode 后保留
38,660 条、1,305,714 帧；其中 537 条短于 15 帧。最终产生 106,625 个候选片段，
Top 10% 预算为 10,663。

无需 GPU 的 100 episode scan 冒烟：

```bash
python -m cocore_bridge_v2 scan \
  --relation sequence --relation-weight 1.0 \
  --max-episodes 100 \
  --output-dir outputs/cocore_bridge_v2/bridge-smoke-100
```

`encode` 和完整 `run` 会解码 `image_0` AV1 视频并使用单路 CLIP 特征。每个有效
episode 的完整逐帧特征会写入 encode 缓存，并经 128 维视觉 PCA 与 state/action
时序池化特征融合。生产配置固定从 `/data/dwb/models/clip-vit-base-patch32` 本地加载
模型，要求 CUDA；不会访问网络，也不会读取 `image_1`、`image_2` 或 `image_3`。
