# TDUS：通用轨迹数据价值评价

TDUS（Trajectory Data Utility Score）是一套无需策略训练、奖励模型或 rollout 的
机器人数据评价工具。它从轨迹本身的动作、状态和图像特征计算 Quality、Coverage、
Diversity、Novelty，并输出逐 trajectory/chunk 的可排序分数。

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

数据路径需包含 `meta/info.json`、`meta/episodes.jsonl`、Parquet episode 和可选视频。
adapter 根据 `info.json` 自动识别维度、相机与 observation，不写死机器人类型。
默认编码 `observation.images.image_0`；改成 `all` 可使用全部相机，将
`dataset.use_images` 设为 `false` 可只使用向量 observation。

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
from tdus.dataset import DatasetAdapter, register_dataset_adapter

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

adapter 只负责产生 `TrajectorySegment`；encoder、四项指标、筛选和分析无需修改。

## 测试

```bash
pytest tests/test_tdus.py tests/test_tdus_lerobot.py
```

默认测试使用合成 adapter；带 `real_data` 标记的测试只校验已挂载的真实 LeRobot
数据元信息。
