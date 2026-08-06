# Quality Filter：分阶段 Quality-only 片段筛选

Quality Filter 对完整 15 帧片段只计算 SQCN 的 Quality，并使用与 SQCN 0.4.0
相同的 embedding 多样化 Filter。算法实现位于公共包 `segment_filter_core`；本包
负责配置、阶段缓存、产物和命令行接口。

## 安装与运行

```bash
pip install -r quality_filter/requirements.txt

python -m quality_filter quality --config quality_filter/config_libero90.yaml
python -m quality_filter encode --config quality_filter/config_libero90.yaml
python -m quality_filter filter --config quality_filter/config_libero90.yaml \
  --percent 10 --seed 1234
python -m quality_filter run --config quality_filter/config_libero90.yaml \
  --percent 10
python -m quality_filter validate \
  --output-dir outputs/quality_filter/libero90 \
  --config quality_filter/config_libero90.yaml \
  --percent 10
```

`--output-dir` 是精确运行目录。`--max-episodes` 可用于 smoke test；不兼容的
已有阶段必须显式传 `--force`。未指定 Filter seed 时，首次构建会生成随机 seed；
兼容缓存复用 manifest 中的实际 seed，`--force` 会重新生成。
程序会在导入 NumPy 前把原生数值库限制为每进程 1 线程；如需调整，设置
`TRAJECTORY_DATA_NUM_THREADS=1..64`。

## 阶段与产物

- `quality/`：两次无图像遍历后生成 `scores.csv`、`numeric_features.npz`、
  `numeric_normalizers.pkl` 和 `manifest.json`。
- `encode/`：在候选与 SQCN reference 窗口去重并集上拟合视觉 PCA，输出与 Quality
  行对齐的 `embeddings.npy`、`visual_pca.pkl` 和 `manifest.json`。逐帧 CLIP 特征
  不会持久化；编码阶段失效时需要重新执行 CLIP。
- `filter/top10pct/`：输出已选 `scores.csv`、同行 `embeddings.npy` 和
  `filter_manifest.json`。结果只包含真实的 Quality 分量与 Filter 诊断，不伪造
  Coverage、Novelty 或 SQCN。
- `run_manifest.json`：记录完成的 Quality+Encode 数据源和阶段指纹；不同筛选比例
  可在同一运行目录下并存。

Filter 使用 Top 100 初始化、100 个 candidate、silent 晋升、5-NN RBF 惩罚和
`lambda=1`。输入片段数与按比例向上取整后的目标数都必须至少为 100。

## 训练接入

all-tasks Octo 训练可直接读取筛选结果：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --prior-prefiltered-scores \
  outputs/quality_filter/libero90/filter/top10pct/scores.csv \
  --output-dir outputs/octo_small_libero_quality_filter_top10pct \
  --preflight-only
```

训练把 CSV 中每一行视为已选片段，只读取 `episode_id`、`start_step`、`end_step`，
忽略 Quality、rank 和其他诊断字段，也不读取相邻的 root/filter manifest。预检会
校验 episode、片段边界、重复项和完整 action window；该参数不能与
`--target-only` 同时使用，训练不按 Quality 加权采样。
