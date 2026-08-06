# TDUS：通用轨迹数据价值评价

TDUS（Trajectory Data Utility Score）是一套无需策略训练、奖励模型或 rollout 的
机器人数据评价工具。它从轨迹本身的动作、状态和图像特征计算 Quality、Coverage、
Diversity、Novelty，并输出逐 trajectory/chunk 的可排序分数。数据读取接口位于
独立的 `trajectory_data` 包；`tdus.dataset` 已删除。

Bridge 只是默认配置使用的首个 LeRobot v2 数据集。核心包、API 和输出均为通用
`tdus`，其他数据集可通过注册 `DatasetAdapter` 接入。

## 安装

建议使用独立 Python 3.10+ 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r tdus/requirements.txt
```

如需指定 CUDA 版本，请先按 PyTorch 官方命令安装 GPU 版 Torch，再安装其余依赖。
`faiss-cpu` 是默认 KNN 后端；环境已安装 `faiss-gpu` 时会自动使用 GPU。若 FAISS
不可用，代码会退回 sklearn，最小环境下还可使用有界内存的 NumPy 实现。

## 配置数据

默认配置指向当前 Bridge LeRobot v2 数据：

```yaml
dataset:
  type: lerobot
  name: bridge_orig
  path: /data/dwb/datasets/bridge_orig_1.0.0_lerobo
```

数据路径需包含 `meta/info.json`、`meta/episodes.jsonl` 和 Parquet episode。
图像既可保存为外部视频（`dtype: video`），也可作为 PNG/JPEG bytes 内嵌在
Parquet（`dtype: image`）。adapter 根据 `info.json` 自动识别维度、相机与
observation，不写死机器人类型。默认编码发现的第一路相机；改成 `all` 可使用
全部相机，将 `dataset.use_images` 设为 `false` 可只使用向量 observation。

### LIBERO-90

仓库提供 [config_libero90.yaml](config_libero90.yaml)，用于
`/data/dwb/datasets/LIBERO_lerobot/libero90`：

```bash
python -m tdus --config tdus/config_libero90.yaml
```

该配置使用 8 维 `observation.state`、7 维 `action` 和 agentview
`observation.images.image`；Parquet 中内嵌的 PNG 会自动解码为 RGB。
完整 trajectory 与 `K=15, S=15` 的 chunk 会同时评分；全量 4,500 个 episode
预计产生 46,705 个长度 15 的 chunk，只有对齐 episode 末尾的最后一个窗口可能
与前一窗口少量重叠。
`/data/dwb/datasets/LIBERO_lerobot` 根目录下另一个 book-caddy target 数据集
不参与这次评分。CLIP 模型与 processor 固定从本地
`/data/dwb/models/clip-vit-base-patch32` 加载，运行 TDUS 不访问 Hugging Face。
快速验证真实数据的前两个 episode：

```bash
python -m tdus --config tdus/config_libero90.yaml --max-episodes 2 --force
```

所有算法参数均位于 [config.yaml](config.yaml)，包括 chunk 的 K/S、CLIP、抽帧数、
PCA、MMD、KNN、GPU、进程数、权重、随机种子、缓存与输出路径。

## 计算 TDUS

```bash
python -m tdus.tdus --config tdus/config.yaml
```

等价的简写：

```bash
python -m tdus --config tdus/config.yaml
```

快速验证前几个 episode：

```bash
python -m tdus --config tdus/config.yaml --max-episodes 8 --force
```

默认同时计算完整 trajectory 和 `K=32, S=16` chunk。CLIP 权重无法加载时会自动
执行 resize + flatten + PCA fallback。兼容缓存会在重复运行时复用；`--force`
强制重算。

TDUS 使用 `trajectory_data` 的共享启动保护，在导入 NumPy 前把 OpenBLAS、
OpenMP、MKL 和 NumExpr 线程池限制为每进程 1 线程，避免高核数服务器上的嵌套
并行。首选使用范围为 1–64 的共享变量 `TRAJECTORY_DATA_NUM_THREADS`：

```bash
TRAJECTORY_DATA_NUM_THREADS=4 python -m tdus --config tdus/config_libero90.yaml
```

原有 `TDUS_NUM_THREADS` 继续兼容；两个变量同时设置时，共享变量优先。

## 仅重算最终分数

已有 `quality`、`coverage`、`diversity`、`novelty` 后，可以用 convert
脚本更换四项权重，只生成新的 CSV，不重新编码轨迹或计算指标：

```bash
python -m tdus.convert \
  --input outputs/tdus/libero90/chunk/scores.csv \
  --output outputs/tdus/libero90/chunk/scores_custom.csv \
  --quality-weight 0.4 \
  --coverage-weight 0.3 \
  --diversity-weight 0.2 \
  --novelty-weight 0.1
```

四个权重都必须提供，取有限非负值且总和为 1。输出路径必须与输入路径不同；
已有输出默认不会被覆盖，需要替换时显式传入 `--force`。

convert 只改写每行的 `tdus`，保留其他字段、列顺序和样本行顺序，因此新 CSV
仍与原来的 `embeddings.npy` 对齐。脚本不会修改标准 `scores.csv`、
`tdus_scores.csv`、embedding、特征缓存或 run manifest。需要按新分数选取样本时，
由 top-k 或训练入口在读取新 CSV 后排序。

## 数据筛选

选择 TDUS 最高的 100 个 chunk：

```bash
python -m tdus.selector --config tdus/config.yaml --mode chunk top-k --k 100
```

在 50,000 timestep 预算下执行动态集合边际选择：

```bash
python -m tdus.selector --config tdus/config.yaml --mode chunk \
  budget --budget 50000
```

Python API：

```python
from tdus import select_top_k, select_budget

top = select_top_k(100, "outputs/tdus/bridge_orig/chunk/tdus_scores.csv")
```

`select_budget` 还接收候选 embedding、trajectory reference embedding、coverage 配置
和 TDUS 权重；CLI 会从 YAML 自动补齐这些参数。

## 根据 LIBERO-10 结果回归 TDUS 权重

权重 sweep 已产生 LIBERO-10 评测结果后，可用 `all` 目录作为未筛选基准，拟合
可重跑的线性/二次 Ridge 响应面：

```bash
python3 scripts/analyze_libero_tdus_weights.py \
  --results-root /data/dwb/octo_small_libero \
  --weights-file weights.jsonl \
  --sweep-manifest /data/dwb/checkpoints/octo_tdus_weight_sweep/sweep_manifest.json \
  --output-dir outputs/tdus_weight_regression
```

脚本只使用 10 个任务均完成、episode 数完整且评测协议与 `all` 一致的模型；部分
结果会列入候选状态，但不会进入回归。每个任务先计算
`clip(model_success_rate / all_success_rate, 0, 2)`，再以相同的 1/10 权重平均，
因此不会让 episode 池化或低基准任务的极端倍数主导目标。

默认在一阶和含平方/交互项的二阶 Ridge 之间用嵌套留一验证自动选阶，并生成：

- `analysis.json`：输入哈希、数据质量、原始权重方程、验证指标、稳定性与最优解；
- `candidate_ranking.csv`：`weights.jsonl` 全部候选的状态、支持范围和预测排名；
- `report.md`：当前实测最佳、支持范围内预测最佳、下一评测候选和连续参考解。

连续解只在完整模型各权重分量的观测范围内求解。报告标为 `provisional` 时，说明
嵌套验证没有优于均值预测，或候选对留一重拟合不稳定；任何预测权重都必须经过新
的训练和 LIBERO-10 评测，不能当作实测性能。

## 分析图

```bash
python -m tdus.analysis --config tdus/config.yaml
```

输出 `tdus_hist.png`、`embedding_tsne.png`、`quality_distribution.png` 和
`trajectory_vs_chunk.png`。

## 输出

每种粒度位于 `outputs/tdus/<dataset.name>/<trajectory|chunk>/`：

- `embeddings.npy`：按得分 CSV 相同行序排列的 128 维 embedding；
- `features.pkl`：数据/配置指纹、样本元数据、原始池化特征、质量与归一化信息；
- `tdus_scores.csv`：按 TDUS 降序的标准结果；
- `scores.csv`：与 `tdus_scores.csv` 内容一致的兼容文件；
- `budget_selection.csv`：预算筛选的加入顺序、边际收益和集合分数。

CSV 字段为：

```text
sample_id,episode_id,start_step,end_step,length,
quality,coverage,diversity,novelty,tdus
```

## 增加其他数据集

实现通用接口并注册工厂：

```python
from trajectory_data import DatasetAdapter, register_dataset_adapter

class MyDatasetAdapter(DatasetAdapter):
    ...

register_dataset_adapter("my_format", MyDatasetAdapter)
```

然后配置：

```yaml
dataset:
  type: my_format
  name: my_robot_data
  path: /data/my_robot_data
```

adapter 通过公共 `iter_episodes()` 产生 `EpisodeData`，共享基类再构造
`TrajectorySegment`；encoder、四项指标、筛选和分析无需修改。

## 测试

```bash
pytest tests/test_tdus.py tests/test_tdus_lerobot.py
```

默认测试使用合成 adapter；带 `real_data` 标记的测试只校验已挂载的真实 LeRobot
数据元信息。
