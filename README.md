# Qwen3-VL / Qwen3.5 GROOT Bridge

这是一个独立的 BridgeData V2 视觉—语言—动作（VLA）微调项目。它完整加载本地
Qwen3-VL-4B-Instruct 的 36 层文本模型，或 Qwen3.5-0.8B 的 24 层混合
DeltaNet/全注意力文本模型。Qwen3-VL 在全部文本注意力与 MLP 投影上训练 LoRA，
Qwen3.5 在全部文本 token-mixer 上训练 LoRA；两者都从随机初始化的 GROOT 风格
flow-matching DiT 动作头开始训练。视觉塔和主干原始参数始终冻结。

仓库同时包含与训练解耦的通用轨迹数据价值工具
[TDUS](tdus/README.md)，用于直接从 LeRobot trajectory/chunk 计算 Quality、
Coverage、Diversity 和 Novelty；以及仿照 SQCN 两遍数据流、面向 LIBERO 关系图与
集合目标筛选的 [RelCore](relcore/README.md)。另有固定使用 support、progress 和
运动原语、以可配置 sequence/cooccurrence 关系减冗余目标执行有界惰性最大堆选择的
[Cocore](cocore/README.md)。

BridgeData V2 可通过独立的仓库内命令包
[Cocore BridgeV2](cocore_bridge_v2/README.md) 使用同一 Cocore 算法。该适配入口固定
读取 LeRobot v2 的 `image_0`、8 维 state 和 7 维 action，排除空任务 episode，
并保留与 Cocore 相同的选择 artifact 格式。

另有分阶段的 [Quality Filter](quality_filter/README.md)，只计算 SQCN Quality，
再用相同的融合 embedding 和 0.4.0 多样化 Filter 完成片段筛选。SQCN 与
Quality Filter 的共同算法位于 `segment_filter_core`。

它不依赖 Isaac-GR00T 源码，也不兼容 NVIDIA GR00T checkpoint 或 Policy API。

## 环境

需要 Linux、Python 3.12、CUDA 12.8 和支持 BF16 的 NVIDIA GPU。推荐在新虚拟环境中：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0 torchvision==0.24.0
pip install -e ".[test]"
```

使用已有 pyenv Python 3.12 环境时，直接升级该环境并安装项目：

```bash
pyenv shell <python-3.12-env>
python -m pip install --upgrade pip
python -m pip install --upgrade -e ".[test]"
```

Qwen3.5 需要 Transformers 5；项目固定使用 `transformers==5.2.0` 和
`peft==0.18.0`。DeltaNet 默认使用 Transformers 内置的 PyTorch 回退路径，
不要求安装 `fla` 或 `causal-conv1d`。

FlashAttention 2 是可选项；未安装时自动使用 PyTorch SDPA：

```bash
pip install -e ".[flash]"
```

Qwen3.5 在 Transformers 5.2 下的 `auto` 模式固定使用 SDPA，避免该版本
FlashAttention varlen backward 的已知越界问题；Qwen3-VL 的自动选择行为不变。

## 训练

### Octo-small（独立并列配置）

Octo-small 使用独立的纯 PyTorch 路线，不替换 Qwen3-VL，也不复用其 LoRA、
GR00T DiT 或 DeepSpeed 配置。训练数据、双相机视觉 stem、T5-base、12 层
blockwise transformer 和 diffusion action head 均由 PyTorch 执行；正式环境
不安装或导入 TensorFlow、dlimp、JAX、Flax、Optax、Orbax。

#### 安装纯 PyTorch 环境

Octo 路线固定使用独立 Python 3.10 环境：

```bash
python3.10 -m venv .venv-octo-pytorch
source .venv-octo-pytorch/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1
pip install -r requirements-octo-pytorch.txt
```

#### 一次性转换预训练 checkpoint

本地 `/data/dwb/models/octo-small` 是 Orbax 格式。使用独立的一次性 CPU
环境把 step 270000 转成 PyTorch：

```bash
python3.10 -m venv .venv-octo-convert
source .venv-octo-convert/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install -r requirements-octo-convert.txt

bash scripts/convert_octo_small_to_pytorch.sh \
  --source /data/dwb/models/octo-small \
  --output /data/dwb/models/octo-small-pytorch \
  --step 270000
```

转换器直接读取 Orbax 参数树，不依赖官方 Octo 包或 TensorFlow。它会转换
双相机 stem、T5-base、12 层 transformer 和兼容的 diffusion 权重；新增
proprio 参数以及 horizon 4→8 后形状不兼容的 diffusion 投影按 seed 42
初始化。输出包含 `model.safetensors`、`model_config.json`、自包含
`text_encoder/` 和记录源 hash、逐 tensor 映射的 `conversion_manifest.json`。

首次转换默认从 `/data/dwb/models/t5-base` 读取预先下载的
`config.json`、`spiece.model` 和 `tokenizer.json`；T5 模型权重仍来自本地
Octo checkpoint。转换过程不会访问 Hugging Face，最终 artifact 会生成完整的
`text_encoder/tokenizer_config.json`。如需使用其他本地目录：

```bash
bash scripts/convert_octo_small_to_pytorch.sh \
  --t5-source /path/to/t5-base \
  --local-files-only
```

转换完成后可删除一次性环境，训练只使用纯 PyTorch 环境。

纯 CPU 两步模型/优化器 smoke test 不需要 CUDA：

```bash
PYTHONPATH=src pytest -q \
  tests/test_octo_small_pytorch.py \
  -k two_step_cpu_smoke
```

#### BridgeData V2 四卡微调

Octo-small 可直接读取默认的
`/data/dwb/datasets/bridge_orig_1.0.0_lerobo`，无需转换或复制数据。独立入口只使用
`observation.images.image_0` 作为主相机、8 维 `observation.state` 和 7 维
`action`；空语言 episode 会被排除。第七维抓手动作是 `[0,1]` 连续值（接受
`[-1e-5, 1+1e-5]` 的浮点容差并裁剪回 `[0,1]`），按 `2*x-1` 映射到
`[-1,+1]` 且不参与统计标准化；非有限值或明显越界值会触发包含 episode ID 的数据
校验错误。管线不伪造 wrist 图像。

先运行只读预检，再执行两步 smoke test 或正式训练；`--output-dir` 始终必填：

```bash
bash scripts/train_bridge_octo_small_4x4090.sh \
  --output-dir outputs/octo_small_bridge_preflight \
  --preflight-only

bash scripts/train_bridge_octo_small_4x4090.sh \
  --output-dir outputs/octo_small_bridge_smoke \
  --smoke-test

bash scripts/train_bridge_octo_small_4x4090.sh \
  --output-dir outputs/octo_small_bridge
```

默认使用 GPU `0,1,2,3`、每卡 micro-batch 8、梯度累积 4、全局 batch 128、
10,000 optimizer steps、400 步 warmup、峰值学习率 `3e-4` 和 BF16。预检会验证
LeRobot v2/WidowX/5 Hz/AV1、特征和统计维度、源数据计数，并解码有效 episode 的
首、中、尾样本。自定义路径和续训示例：

```bash
bash scripts/train_bridge_octo_small_4x4090.sh \
  --dataset-path /path/to/bridge_lerobot_v2 \
  --model-path /path/to/octo-small-pytorch \
  --gpu-ids 0,1,2,3 \
  --output-dir outputs/octo_small_bridge \
  --resume latest
```

首版使用全部 38,660 条非空语言 episode（1,305,714 帧），不划分验证集，也不接收
Cocore/RelCore 筛选清单或运行 SimplerEnv 评测。训练输出沿用 Octo 的配置、数据和
模型 manifest、metrics、latest/best checkpoint 指针；续训会恢复采样顺序和随机数
状态。

#### 转换 LIBERO

先用独立脚本把原始 LIBERO HDF5 转换为严格的 LeRobotDataset v2.0。默认读取
`/data/dwb/datasets/LIBERO`，转换完整 LIBERO-90 prior 和 Book-caddy 任务中
确定性选出的 5 条 demonstration：

```bash
bash scripts/prepare_libero_lerobot_v2.sh \
  --source /data/dwb/datasets/LIBERO \
  --output /data/dwb/datasets/LIBERO_lerobot
```

输出包含 `libero90/` 和小写目标任务目录。每个 episode 独立保存为 Parquet，
两路 128×128 RGB 图像以无损 PNG 内嵌；`meta/` 下生成 v2.0 的
`info.json`、`episodes.jsonl`、`tasks.jsonl` 和 `stats.json`。数据固定为
10 Hz，`libero90/meta/stats.json` 同时作为 prior 与 target 的 action/proprio
归一化统计。根目录的 `conversion_manifest.json` 记录源文件、转换计数及五条
target demo ID。

已有目标数据集默认不会被覆盖；`--overwrite` 只重建上述两个明确的数据集目录和
转换清单，不会删除输出根目录中的其他文件。不支持从旧格式迁移，必须从原始
LIBERO HDF5 重新转换。

需要把 LIBERO-10 的全部 10 个任务各抽取 5 条轨迹时，使用独立转换入口：

```bash
bash scripts/prepare_libero10_lerobot_v2.sh \
  --source /data/dwb/datasets/LIBERO \
  --output /data/dwb/datasets/LIBERO_lerobot
```

该命令按任务名哈希确定性选择轨迹，合并生成 `libero10_5/`，其中包含 10 个
task 和 50 个 episode。转换详情保存在
`libero10_5_conversion_manifest.json`。默认拒绝覆盖；需要重建时追加
`--overwrite`，其作用范围只包括 `libero10_5/` 和对应 manifest，不会修改
`libero90/`、已有单任务 target 或根目录 `conversion_manifest.json`。
`task_index` 与评测脚本使用同一套官方 LIBERO-10 顺序：

```text
0  put both the alphabet soup and the tomato sauce in the basket
1  put both the cream cheese box and the butter in the basket
2  turn on the stove and put the moka pot on it
3  put the black bowl in the bottom drawer of the cabinet and close it
4  put the white mug on the left plate and the yellow-white mug on the right plate
5  pick up the book and place it in the back compartment of the caddy
6  put the white mug on the plate and the chocolate pudding to its right
7  put both the alphabet soup and the cream cheese box in the basket
8  put both moka pots on the stove
9  put the yellow-white mug in the microwave and close it
```

#### Preflight 与四卡训练

原生 PyTorch DataLoader 直接读取 Parquet 内嵌 PNG，完成 Lanczos resize、
CHW 转换、`[-1,1]` 图像归一化、prior action/proprio 统计归一化和 8 步
action window。每个 worker 使用有界 episode LRU cache；确定性 DDP sampler
保证每个 micro-batch 都按 LIBERO-90 与所选任务的五条 target 轨迹 `1:1`
采样。`--task-index` 是必填的评测 index，例如 book-caddy 为 `5`：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 --output-dir outputs/octo_small_libero_task-5 --preflight-only
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 --output-dir outputs/octo_small_libero_task-5-smoke --smoke-test
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 --output-dir outputs/octo_small_libero_task-5
```

如需让 LIBERO-10 的全部 10 个任务、50 条完整 target 轨迹共同参与训练，使用
独立的四卡全任务脚本。每个 micro-batch 按 `1:1` 采样，由 50% 全任务
target 池和 50% LIBERO-90 prior 组成：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --output-dir outputs/octo_small_libero_4gpu_all-tasks_quality_top10pct --preflight-only
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --output-dir outputs/octo_small_libero_4gpu_all-tasks_quality_top10pct-smoke --smoke-test
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --output-dir outputs/octo_small_libero_4gpu_all-tasks_quality_top10pct
```

如果只使用上述 `libero10_5` 中每个任务 5 条、共 50 条示例轨迹，不采样也不读取
LIBERO-90 prior 或预筛选片段文件，追加 `--target-only`：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --target-only \
  --output-dir outputs/octo_small_libero_4gpu_all-tasks_target-only \
  --preflight-only
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --target-only \
  --output-dir outputs/octo_small_libero_4gpu_all-tasks_target-only-smoke \
  --smoke-test
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --target-only \
  --output-dir outputs/octo_small_libero_4gpu_all-tasks_target-only
```

该模式的每卡 micro-batch 8 条样本全部来自 `libero10_5`，action 与 proprio
归一化也使用 `libero10_5/meta/stats.json`。`--target-only` 不能与
`--sample-weights` 或任何 `--prior-*` 参数同时使用；不传该开关时，脚本仍保持
下面所述的 1:1 target/prior 混合训练行为。

全任务 target 池内部按帧均匀采样；因此不同任务的抽样频率会随其轨迹总帧数
变化。该脚本默认把
`/data/dwb/libero_filter/quality/filter/top10pct/scores.csv` 作为已经筛选完成的
片段清单。文件中的 4,671 行全部入选，不按 Quality 或其他分数二次排序；每个
片段内从 `start_step` 到 `end_step` 的每一帧都贡献训练起点；8-step action
window 可以越过片段末尾继续读取同一条 episode 的真实动作。只有到达 episode
末尾仍不足 8 步时，才重复 episode 最后一个完整动作补满，且补位参与训练 loss。
重叠片段按 episode/frame 起点去重后覆盖 2,974 条 episode，共得到 66,566 个
prior 训练起点。

训练入口只接受 `--prior-prefiltered-scores PATH`。文件可为 CSV 或 JSONL：首个
非空内容以 `{` 开头时按 JSONL 读取，否则按 CSV 读取。两种格式都只要求
`episode_id`、`start_step`、`end_step`，其他分数、rank、`sample_id`、`length`
和诊断字段全部忽略。预检不读取相邻的 filter/run manifest，也不校验算法来源；
它只校验 UTF-8、必需字段、整数、episode、非空片段边界和重复片段，
并把输入文件 SHA256 与展开后的选择摘要写入训练 manifest。`--output-dir` 对训练、
smoke test 和 preflight 都是必填参数：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --prior-prefiltered-scores /path/to/selected.csv \
  --output-dir outputs/custom_prefiltered_run \
  --preflight-only
```

RelCore 的 `selected_manifest.jsonl` 也通过同一参数传入；文件中的每个 JSON
对象都视为已选片段，`selected` 等额外字段不参与训练：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --prior-prefiltered-scores /data/dwb/libero_filter/relcore_top20pct/select/selected_manifest.jsonl \
  --output-dir outputs/octo_small_libero_4gpu_all-tasks_relcore_top20pct \
  --preflight-only
```

已删除训练入口中的 TDUS 排序、SQCN、RelCore 和 Quality 专用参数。需要按某个
分数或算法选择 Top N% 时，必须先由相应离线工具生成只包含已选行的 CSV/JSONL，
再传给统一参数。`--prior-prefiltered-scores` 不能与 `--target-only` 混用。

训练脚本使用 `torchrun`、DDP 和 BF16；默认每卡 micro-batch 8，其中全任务
target 4 条、prior 4 条；梯度累积 4 后，四卡有效全局 batch 为 128，其中
target 64 条、prior 64 条。单任务脚本与权重 sweep 也使用 1:1。预检会验证
v2.0 metadata、全部 Parquet
footer/schema、评测顺序的 10-task 映射、单任务模式的 5 条或全任务模式的
50 条 target episode、抽样 PNG、统计维度、非空 prior、PyTorch
safetensors/转换清单、CUDA 数量和 BF16 支持。
自定义数据根目录使用 `--lerobot-path`；旧数据参数不再接受：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --lerobot-path /data/dwb/datasets/LIBERO_lerobot \
  --output-dir outputs/octo_small_libero_task-5 \
  --preflight-only
```

#### 使用自定义预筛选片段训练

单任务入口使用相同的预筛选文件契约，并继续按 1:1 组成 prior/target
micro-batch。TDUS、SQCN、Quality Filter 或其他算法必须先在训练外完成排序和
截取；训练端使用输入文件中的全部行：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --prior-prefiltered-scores /path/to/selected.csv \
  --output-dir outputs/octo_small_libero_prefiltered
```

`--preflight-only` 和 `--smoke-test` 支持相同参数。续训必须使用内容完全相同的
预筛选文件，否则输入文件 SHA256 和选择摘要会发生变化：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --prior-prefiltered-scores /path/to/selected.csv \
  --output-dir outputs/octo_small_libero_prefiltered \
  --resume latest
```

DataMIL 的整轨 Top-K 清单也通过同一个参数传入，但仅支持 JSONL。每行必须包含
`trajectory_id` 和 `num_frames`；训练端将 `trajectory_id` 映射到同号的 LeRobot
`episode_index`，并把该 episode 的 `0..num_frames-1` 全部帧作为训练起点。
`num_frames` 必须与 `libero90/meta/episodes.jsonl` 中记录的 episode 长度完全一致；
重复轨迹、未知 ID、非正帧数、长度不一致或与片段 schema 混用都会在预检时报错。
`rank`、`score`、`frame_weight`、`demo_id` 等附加字段被忽略，输入文件不会被改写。

例如，显式使用 DataMIL Top 20% 的整轨选择训练单任务模型：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --prior-prefiltered-scores /data/dwb/libero_filter/datamil/selected_topk0.2.jsonl \
  --output-dir outputs/octo_small_libero_task-5_datamil-top20pct
```

manifest 会把这类输入记录为 `prefiltered_trajectories`，保留原始 JSONL 的绝对
路径和 SHA256，并记录选中轨迹数、episode 数与展开后的训练起点数。传统
`episode_id/start_step/end_step` CSV/JSONL 的 manifest 和训练语义保持不变。

#### 批量扫描 TDUS 权重

`weights.jsonl` 中每一行可以作为一组独立的
`quality`、`coverage`、`diversity`、`novelty` 权重，重新换算 chunk 的最终
`tdus` 并训练一个包含 LIBERO-10 全部任务的模型。默认读取
`outputs/tdus/libero90/chunk/scores.csv`、选择重新排序后的 Top 10%，并启动
8 个单卡训练进程：

```bash
python3 scripts/train_libero_octo_small_all_tasks_weight_sweep.py
```

默认使用 GPU `0,1,2,3`，8 个固定 worker 的绑定顺序为
`0,1,2,3,0,1,2,3`，即每张卡同时运行 2 个模型。权重 sweep 固定按
LIBERO-10 target 与 LIBERO-90 prior `1:1` 采样；单模型 micro-batch 8 中包含
4 条 target 和 4 条 prior，梯度累积 16 后有效 batch 128 中包含 64 条 target
和 64 条 prior。运行正式 sweep 前建议先确认同卡 2 个 Octo-small 模型能够
同时装入显存：

```bash
python3 scripts/train_libero_octo_small_all_tasks_weight_sweep.py \
  --preflight-only

python3 scripts/train_libero_octo_small_all_tasks_weight_sweep.py \
  --smoke-test
```

第 1 行权重生成 `scores_weights_001.csv` 和 `model-001/`，依此类推。sweep 会先按
新的 TDUS 分数排序并将 Top N% 物化到该 CSV，再通过统一的
`--prior-prefiltered-scores` 启动训练；原始 `scores.csv` 不会被修改。
模型、日志、批次 manifest 和最终汇总默认写入
`outputs/octo_small_libero_all_tasks_weight_sweep_top10pct/`。可覆盖常用参数：

```bash
python3 scripts/train_libero_octo_small_all_tasks_weight_sweep.py \
  --weights-file /path/to/weights.jsonl \
  --scores /path/to/libero90/chunk/scores.csv \
  --output-root outputs/octo_weight_sweep \
  --gpu-ids 0,1,2,3 \
  --parallel 8 \
  --prior-top-percent 10 \
  --max-steps 5000
```

`--max-steps` 会应用到每一行权重对应的模型，同时作为该模型余弦学习率衰减的
终点，并用于判断已有模型应跳过还是从 `latest` 续训；不传时使用单卡配置中的
10,000 步。

重复运行时，已达到目标步数的模型会跳过，存在完整但未完成 checkpoint 的模型
会从 `latest` 续训。若权重、源 scores 或重新换算后的 scores 与已有 manifest
不一致，或者旧输出不是 1:1 采样，脚本会保留现场并将该行报告为失败。首次使用
1:1 sweep 前需要人工移动或清理同一路径下的旧 3:1 checkpoint。单个模型失败
不会阻止其他行训练；最终 `sweep_summary.json` 会列出失败行，且批量脚本返回
非零状态。

默认训练范围是第 1–10,000 步，前 400 步固定为线性 warmup，随后余弦学习率
在第 10,000 步衰减到 0。`--max-steps N` 会同时覆盖训练和余弦衰减终点，使
余弦 progress 在第 400–N 步重新映射并在第 N 步到达 0；小于等于 400 步的
smoke/短跑只执行 warmup。该行为由 Octo-small 的共享调度器实现，对单卡、四卡、
target-only、mixed 和 weight sweep 均生效。每 1,000 步和最终步保存候选
checkpoint，并使用该保存区间内的平均训练 loss 选择 best。`checkpoints/`
最多保留 latest 与 best 对应的两个 `step-*` 目录；二者重合时只保留一个。
`latest.json` 和 `best.json` 分别记录最新步和最佳步，不会额外创建 `best/`
目录。

训练 checkpoint 使用 safetensors 保存模型，并保存 AdamW、scheduler、
sampler 和 RNG 状态。严格续训：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --output-dir outputs/octo_small_libero_task-5 \
  --resume latest
```

完整配置见 `configs/octo_small_libero_4x4090.yaml`，默认模型路径为
`/data/dwb/models/octo-small-pytorch`。

评测 target-only checkpoint 时必须使用与训练相同的 target 统计，不能沿用评测器
默认的 LIBERO-90 统计：

```bash
bash scripts/evaluate_libero_octo_small.sh \
  --checkpoint /path/to/checkpoint \
  --base-model /data/dwb/models/octo-small-pytorch \
  --statistics /data/dwb/datasets/LIBERO_lerobot/libero10_5/meta/stats.json
```

#### LIBERO checkpoint 闭环评测

评测器只加载 checkpoint，不启动训练，也不解析 `best`、`latest` 或训练运行目录。
在已经能够加载 Octo checkpoint 的 Python 3.10 模型环境中安装固定版本的
LIBERO/robosuite 仿真依赖，然后 checkout 官方 LIBERO 源码；不要在该环境中
安装 LIBERO 的完整历史训练依赖：

```bash
pip install -r requirements-octo-libero-eval.txt
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
export LIBERO_ROOT="$PWD/third_party/LIBERO"
```

评测器通过 `LIBERO_ROOT` 或 `--libero-root` 直接从该 checkout 导入官方
`libero.libero.envs`，无需把 LIBERO 打成 wheel；启动时会严格核对 Git commit。
官方 `.init` 文件只有在当前文件与该 commit 的 Git blob 完全一致后，才会使用
`torch.load(..., weights_only=False)` 读取。

仿真栈固定为 MuJoCo 3.10.0、robosuite 1.4.0、bddl 1.0.1 和
Gymnasium 1.3.0，不再安装旧版 Gym。
环境由 LIBERO 官方 `OffScreenRenderEnv` 创建，并使用官方
`SubprocVectorEnv` 并行执行；场景、机器人控制器、观测和成功谓词均来自
LIBERO/robosuite，不再使用项目内的直连 MuJoCo 实现。
固定 commit 的 LIBERO 向量环境仍写有 `import gym`；评测器在导入 LIBERO 前
将 Gymnasium 注册为进程内的 `gym` 兼容别名。使用 `spawn` 创建并行 worker 时，
可序列化的启动包装器也会在 worker 导入 LIBERO 前先注册该别名，再转交给官方
worker 入口；无需安装旧版 Gym，也不修改第三方 checkout。
MuJoCo 3.10 修改了 `mj_fullM` 的参数签名；评测器在导入 LIBERO 环境前安装
局部兼容层，把 robosuite 1.4.0 的旧调用转换为 3.10 的 `MjData` 调用，同时保留
3.10 原生调用方式。

MuJoCo、robosuite、BDDL 和 Gymnasium 使用精确版本校验；cloudpickle、
easydict、future、matplotlib 和 termcolor 只校验兼容区间，不会因为模型环境中
已安装兼容的新版本而拒绝启动。例如 cloudpickle 3.1.2 可直接使用，无需降级。

自包含 checkpoint 目录应包含 `model.safetensors`、`model_config.json` 和
`text_encoder/`。默认 checkpoint 已指向
`/data/dwb/models/octo-small-pytorch`。先对 book-caddy（LIBERO-10 index 5）
执行单环境预检：

```bash
bash scripts/evaluate_libero_octo_small.sh \
  --indexes 5 \
  --checkpoint /data/dwb/models/octo-small-pytorch \
  --preflight-only
```

仅包含权重的目录或 `.safetensors` 文件必须显式提供自包含基础模型：

```bash
bash scripts/evaluate_libero_octo_small.sh \
  --indexes 5 \
  --checkpoint /path/to/model.safetensors \
  --base-model /data/dwb/models/octo-small-pytorch \
  --preflight-only
```

启动脚本内置固定 commit 的官方 LIBERO-10 顺序。`--indexes` 接受逗号分隔的
index；不传该参数或传入 `all` 时，会串行启动 10 个独立进程。以下命令分别执行
book-caddy 短闭环 smoke test、选择三个任务，以及全部任务：

```bash
bash scripts/evaluate_libero_octo_small.sh --indexes 5 --smoke-test
bash scripts/evaluate_libero_octo_small.sh --indexes 0,2,5
bash scripts/evaluate_libero_octo_small.sh
```

可用 `--help` 查看完整 index 到任务名称的映射。多任务模式下，
`--output-dir` 是结果根目录；例如：

```bash
bash scripts/evaluate_libero_octo_small.sh \
  --indexes 0,1 \
  --output-dir outputs/libero10_selected
```

上述两个任务分别写入 `outputs/libero10_selected/task-0/` 和
`outputs/libero10_selected/task-1/`。显式使用 `--task-name NAME` 时仍只启动一个
评测进程，并保持原有 `--output-dir` 语义；`--task-name` 不能和 `--indexes`
同时使用。

默认不记录视频。传入 `--save-videos-path` 后，每个任务会按全局 `episode_id`
分别选择最早的成功、失败 episode，并把两类视频写入独立目录。未指定
`--record-videos K` 时两类各保存最多 1 条；显式指定 K 时两类各保存最多 K 条，
且 `--record-videos` 不能脱离 `--save-videos-path` 单独使用。例如：

```bash
bash scripts/evaluate_libero_octo_small.sh \
  --indexes 0,1 \
  --save-videos-path outputs/libero10_videos \
  --record-videos 3
```

上述命令将视频写入
`outputs/libero10_videos/task-0/{success,failure}/` 和
`outputs/libero10_videos/task-1/{success,failure}/`；显式使用 `--task-name` 时则直接
写入给定路径下的 `{success,failure}/`。成功视频包含初始帧并录到首次成功 step；
失败视频同样保存完整 episode。录像使用正式评测中生成的动作轨迹进行确定性
单环境回放，不重复模型推理，也不改变正式评测的并行协议。目标目录已有
`episode-*.mp4` 时需要传入 `--overwrite`；覆盖只替换这些生成视频并保留目录中的
其他文件。`--preflight-only` 不运行 episode，因此不会生成视频。

服务器启动脚本默认使用 `MUJOCO_GL=egl`；无 GPU 的主机可在安装 OSMesa 后使用
`MUJOCO_GL=osmesa bash scripts/evaluate_libero_octo_small.sh ...`。并行环境使用
LIBERO 官方 `SubprocVectorEnv`，评测器会在加载模型和初始化 CUDA 前把
multiprocessing 启动方式固定为 `spawn`，避免 EGL/CUDA 状态被 `fork` 到 worker。

worker 关闭采用整组共享的有界回收：先等待正常关闭 10 秒，再依次对仍存活的
worker 执行 `terminate`（最多 5 秒）和 `kill`（最多 5 秒）。因此关闭时长不会随
worker 数量线性累加。被强制回收但最终退出的 worker 会产生 warning，评测结果仍
可完成；SIGKILL 后仍无法回收则按仿真基础设施故障使用退出码 `3`，多任务 launcher
会停止整批任务，`failure.json` 会记录关闭阶段和残留 PID。

如果 EGL worker 无法启动，评测器会保留最早的异常类型和消息，并显示
`CUDA_VISIBLE_DEVICES`、`MUJOCO_EGL_DEVICE_ID` 和实际并行度。默认会清理
失败的整组 worker，并把并行度按 50→25→12→6→3→1 自动降低；成功使用的
并行度会记录在结果的 `runtime.effective_num_envs` 和
`runtime.environment_batch_sizes` 中。也可以显式测试较小并行度：

```bash
bash scripts/evaluate_libero_octo_small.sh \
  --indexes 5 \
  --episodes 3 \
  --num-envs 1 \
  --max-steps 8 \
  --overwrite
```

传入 `--no-auto-reduce-num-envs` 可恢复首次启动失败即退出的严格行为。评测器
不会自动切换渲染后端。普通任务失败使用退出码 `2`；并行度降到 1 后仍存在
EGL、驱动或渲染子进程启动失败时使用退出码 `3`，多任务 launcher 会立即停止
整批任务，错误同时记录在当前任务的 `failure.json`。

评测固定使用随机种子 `3471197683`、`1232873419`、`1448008435`，每个种子都
依次使用相同的 LIBERO-10 初始状态，并采用 20 个稳定步、8 步 action chunk 和
960 步上限。`--episodes` 表示三个种子的总 episode 数，必须能被 3 整除。
默认的 `results.json` 与 `episodes.jsonl` 分别写入
`outputs/octo_small_libero_eval/task-0/` 到 `task-9/`；已有正式结果默认不会被
覆盖，必须显式传入 `--overwrite`。默认每个任务评测 150 episodes（50 个固定
初始状态 × 3 个种子），全部任务共 1500 episodes。动作和 proprio 统计可通过
`--statistics` 指向其他 LeRobot v2 `stats.json`。结果会记录初始状态文件哈希、
MuJoCo/robosuite 实际版本和
LIBERO 仿真栈标识；每条 episode 记录包含全局 `episode_id`、`init_state_id`
和 `seed`，汇总同时给出整体与分 seed 成功率。MuJoCo 3.10.0 与旧版 2.x
的动力学结果可能存在数值差异，不同版本的正式评测结果不应直接混合比较。

### 4×RTX 4090

4 卡配置默认使用物理 GPU `0,1,2,3`，每卡 micro-batch 1、梯度累积 16，
有效 batch 仍为 64。GPU 编号可通过 `--gpu-ids` 设置：

```bash
bash scripts/train_bridge_4x4090.sh \
  --gpu-ids 2,3,6,7 \
  --lora-learning-rate 1e-5 \
  --action-head-learning-rate 1e-4 \
  --model-path /data/dwb/models/Qwen3-VL-4B-Instruct \
  --dataset-path /data/dwb/datasets/bridge_orig_1.0.0_lerobo \
  --output-dir outputs/qwen3_vl_4b_groot_bridge_4gpu
```

BridgeData V2 也可以使用周期性 LoRA 调度入口：

```bash
bash scripts/train_bridge_qwen3_vl_4b_groot_cyclic_lora_4x4090.sh \
  --output-dir outputs/qwen3_vl_4b_groot_bridge_cyclic_lora
```

该入口的第 1–5,000 个 optimizer step 只训练 GR00T 动作头；之后每 100 步的
前 90 步仍只训练动作头，最后 10 步训练 LoRA 与动作头。默认调度可以分别用
`--lora-freeze-steps`、`--lora-cycle-steps` 和 `--lora-active-steps` 覆盖。入口继续
使用 `configs/bridge_4x4090.yaml` 中的 BridgeData V2 数据配置，并允许通过现有
Bridge 训练参数覆盖模型、数据集、GPU 和输出路径。

`--gpu-ids` 是宿主机上的物理编号。启动器会将其写入
`CUDA_VISIBLE_DEVICES`，随后用 4 个 `torchrun` 进程训练；编号数量必须与
`--gpu-count` 一致。例如改成 GPU `4,5,6,7`：

```bash
bash scripts/train_bridge_4x4090.sh --gpu-ids 4,5,6,7
```

LoRA 与 GR00T 动作头使用独立 optimizer parameter group，可分别通过
`--lora-learning-rate` 和 `--action-head-learning-rate` 覆盖；默认分别为
`1e-5` 与 `1e-4`。

### Qwen LIBERO 全任务训练

Qwen LIBERO 使用独立的 Shell/YAML 入口，不读取 Bridge 配置。默认训练
`libero10_5` 的全部 50 条 target 轨迹，并使用 SQCN Top10% LIBERO-90 prior；
4 卡每个全局 micro-step 按 1:1 取得 2 条 target 与 2 条 prior，梯度累积 16
后的有效 batch 64 为 32/32。`--output-dir` 必须显式指定：

```bash
bash scripts/train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh \
  --lora-learning-rate 5e-6 \
  --action-head-learning-rate 2e-4 \
  --output-dir outputs/qwen3_vl_groot_libero
```

Qwen3.5-0.8B 使用独立 Shell/YAML 入口，默认从
`/data/dwb/models/Qwen3.5-0.8B` 加载 24 层、hidden size 1024 的多模态主干，
并复用相同的 LIBERO 数据组合和 12 层 GROOT 动作头：

```bash
bash scripts/train_libero_qwen3_5_0_8b_groot_all_tasks_4x4090.sh \
  --output-dir outputs/qwen3_5_0_8b_groot_libero
```

该入口默认直接读取主干最后一层 hidden state，不计算未使用的 LM logits。LoRA
覆盖 6 个全注意力层的 `q_proj/k_proj/v_proj/o_proj`，以及 18 个 DeltaNet 层的
`in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`；匹配范围严格限制在文本层。

使用周期性 LoRA 调度的新入口：

```bash
bash scripts/train_libero_qwen3_vl_4b_groot_cyclic_lora_all_tasks_4x4090.sh \
  --output-dir outputs/qwen3_vl_groot_libero_cyclic_lora
```

该入口的第 1–5,000 个 optimizer step 只训练 GR00T 动作头。之后每 100 步的
前 90 步仍只训练动作头，最后 10 步训练 LoRA 与动作头；例如 5,001–5,090
只训练动作头，5,091–5,100 训练两者，5,101 步开始下一个周期。这里的“训练
Qwen”只更新覆盖 36 层注意力 `q_proj/k_proj/v_proj/o_proj` 与 MLP
`gate_proj/up_proj/down_proj` 的 LoRA，Qwen3-VL-4B 原始参数始终冻结。
LoRA 的 warmup 与余弦衰减只统计实际启用 LoRA 的 optimizer step。

默认调度可分别用 `--lora-freeze-steps`、`--lora-cycle-steps` 和
`--lora-active-steps` 覆盖；后两个参数必须同时提供，且 active steps 不能超过
cycle steps。`train/lora_enabled`、`train/lora_updates` 和 `train/lora_lr`
分别记录刚完成 step 是否更新 LoRA、累计 LoRA 更新次数和该 step 实际使用的
LoRA 学习率。

默认的 `--prior-prefiltered-scores PATH` 把传入文件中的全部行视为已经完成筛选的
片段，不在训练端重新排序或截取。文件可为 CSV 或 JSONL，只要求
`episode_id`、`start_step`、`end_step`；其他分数、rank 和诊断字段全部忽略。
加载与预检不读取相邻的 `filter_manifest.json` 或 `run_manifest.json`，只校验输入
编码、必需字段、整数、episode、非空片段边界和重复片段，并把
输入文件哈希及展开后的选择摘要写入训练 manifest。`--prior-top-percent`、
`--prior-relcore-manifest` 和 `--prior-quality-filter-scores` 仍保留各自的严格来源与
manifest 校验。

可追加 `--preflight-only`，或用 `--smoke-test` 固定执行 20 个 optimizer step。
只训练 LIBERO-10 时追加 `--target-only`；该模式不读取 prior 或 scores，且不能与
`--sample-weights`/`--prior-*` 混用。LIBERO 不做离线验证，因此只写周期/最终
checkpoint 和 `latest.json`，不生成 `best.json`。

### 8×RTX 4090

8 卡配置：

```bash
bash scripts/train_bridge_8x4090.sh \
  --model-path /data/dwb/models/Qwen3-VL-4B-Instruct \
  --dataset-path /data/dwb/datasets/bridge_orig_1.0.0_lerobo \
  --output-dir outputs/qwen3_vl_4b_groot_bridge
```

启动器先校验模型、Bridge 元数据、三段真实视频和单卡 micro-batch 显存，再启动
对应数量的 `torchrun` 进程。以下命令以 4 卡版为例，只运行预检：

```bash
bash scripts/train_bridge_4x4090.sh --gpu-ids 0,1,2,3 --preflight-only
```

运行 20 个 optimizer step 的验收：

```bash
bash scripts/train_bridge_4x4090.sh --gpu-ids 0,1,2,3 --smoke-test
```

使用 ZeRO-3 和 CPU optimizer offload：

```bash
bash scripts/train_bridge_4x4090.sh --gpu-ids 0,1,2,3 --deepspeed-stage 3
```

训练只保存 LoRA 和动作头权重，不保存优化器、scheduler、随机状态或 DeepSpeed
分片，因此不支持 `--resume` 断点续训。

所有配置默认同时关闭 Qwen 主干与 DiT 动作头的 gradient checkpointing。全部
Qwen3-VL-4B 配置（Bridge 4 卡、Bridge 8 卡和 Qwen LIBERO）默认使用 PyTorch
Inductor 仅编译 DiT 动作头；编译采用动态 shape、允许局部 graph break，首次训练
batch 会产生一次性 TorchInductor 预热延迟。Qwen 主干仍默认不编译，以避开
FlashAttention 动态 shape 路径中的 graph break、按层重复编译和 Inductor 编译失败。
兼容字段 `model.torch_compile.enabled` 保持为 `false`，显式的目标开关优先；可追加
`--no-compile-action-head` 临时关闭动作头编译，或追加 `--compile-qwen-backbone`
单独开启 Qwen 主干编译。Qwen3.5-0.8B 使用独立配置，其默认行为不受此设置影响。

完整参数见：

```bash
python -m qwen3_vl_groot.cli launch --help
```

## 数据约定

Bridge 只读取 `observation.images.image_0`；LIBERO 只读取主视角
`observation.images.image`。数据中前六维 EEF 动作已是相对动作，
第七维 gripper 是绝对值；管线不会再次减去当前状态。训练/验证按 episode 的稳定
哈希切分。q01/q99 仅从训练 episode 计算，保存在输出目录的
`data_cache/normalization.json`，原始数据集不会被修改。

## 输出

输出目录包含：

- `run_config.yaml`、Bridge 的 `data_fingerprint.json` 或 LIBERO 的
  `dataset_manifest.json`，以及归一化统计；
- `metrics.jsonl` 与 TensorBoard event；
- `checkpoints/step-XXXXXXXX/` 下仅含 LoRA、动作头和必要推理元数据的紧凑
  Safetensors checkpoint；
- `checkpoints/latest.json` 与 `checkpoints/best.json` 分别标记最新和验证集
  最优 step。

rank 0 会把 `metrics.jsonl` 中的同一份 JSON 指标同步打印到终端。也可以从另一个
终端直接跟踪指标或启动 TensorBoard：

```bash
tail -f /path/to/output/metrics.jsonl
tensorboard --logdir /path/to/output/tensorboard --port 6006
```

日志按 `train.log_every_steps` 个 optimizer step 输出；首次 batch 前会明确提示是否
正在进行 TorchInductor warm-up。短时间没有新行不等于 NCCL 卡死，应结合指标文件
更新时间和 GPU 利用率判断。

最多保留 latest 与 best 引用的两个 `step-*` 目录；二者指向同一步时只保留
一份。不会额外复制到 `best/` 或 `inference/` 目录。

## 推理

```python
import json
from pathlib import Path

from qwen3_vl_groot.inference import BridgePolicy

checkpoint_root = Path("outputs/qwen3_vl_4b_groot_bridge/checkpoints")
best = json.loads((checkpoint_root / "best.json").read_text())
policy = BridgePolicy.from_pretrained(
    checkpoint_root / best["checkpoint"],
    model_path="/data/dwb/models/Qwen3-VL-4B-Instruct",
)
actions = policy.predict_actions(image, state, instruction, denoising_steps=4)
assert actions.shape == (1, 8, 7)
```

`image` 可为一张 PIL 图像、RGB NumPy 数组或由这些对象组成的 batch；`state`
为 `[8]` 或 `[B,8]`。输出是反归一化后的 `[B,8,7]` 连续动作。

### Qwen LoRA 的 LIBERO 闭环评测

`evaluate_libero_qwen.sh` 只接受训练产出的具体 `step-XXXXXXXX` 紧凑
checkpoint。该目录必须包含 `adapter_model.safetensors`、`policy_config.json` 和
`normalization.json`；评测器根据 manifest 加载本地 Qwen3-VL-4B 或
Qwen3.5-0.8B 基座，再叠加其中的 LoRA 与 GROOT action head，不解析
训练输出根目录、`latest.json` 或 `best.json`。

当前版本要求 Qwen3-VL-4B checkpoint 的 `model.lora.target_modules` 同时包含完整
attention 与 MLP 目标，不迁移旧 attention-only LoRA checkpoint；这类 manifest
会在加载基座和适配器权重前被明确拒绝。Qwen3.5-0.8B checkpoint 的目标结构保持
不变。需要继续评测旧 attention-only Qwen3-VL-4B checkpoint 时，应使用生成它们
的旧版本代码。

先对 LIBERO-10 index 5 执行单环境预检：

```bash
bash scripts/evaluate_libero_qwen.sh \
  --indexes 5 \
  --checkpoint /data/dwb/qwen_small_libero_relcore_top20pct/checkpoints/step-00020000 \
  --preflight-only
```

基础模型默认从 checkpoint 的 `policy_config.json` 读取，也可以用
`--model-path /data/dwb/models/Qwen3-VL-4B-Instruct` 或
`--model-path /data/dwb/models/Qwen3.5-0.8B` 显式覆盖；覆盖路径必须与
manifest 的 `backbone_family` 一致。预检通过后可执行固定三个种子、
每个种子一个初始状态、最多八步的 smoke test：

```bash
bash scripts/evaluate_libero_qwen.sh \
  --indexes 5 \
  --checkpoint /data/dwb/qwen_small_libero_relcore_top20pct/checkpoints/step-00020000 \
  --smoke-test
```

不传 `--indexes` 时按官方顺序串行评测全部十个任务，每个任务默认运行 150 个
episode，并分别写入 `outputs/qwen_libero_eval/task-0/` 到 `task-9/`。单卡模型推理
默认以 `--policy-batch-size 4` 对并行环境分批；该值可根据显存调整，而
`--num-envs` 仍控制 LIBERO 仿真并行度。已有结果不会自动覆盖，重复正式评测需传
`--overwrite`。

### Qwen Bridge 的 SimplerEnv 四任务闭环评测

该入口只评测以下固定 WidowX Bridge 任务：`spoon`、`carrot`、`stack`、
`eggplant`。仓库用 submodule 固定
`third_party/SimplerEnv@06accaca93535902d408da4855f21cece12bceb7`；首次 checkout 后
必须同时初始化其嵌套的 `ManiSkill2_real2sim`：

```bash
git submodule update --init --recursive third_party/SimplerEnv
```

评测必须使用预先准备的 Python 3.10 或 3.11 单进程环境。不要把当前项目的
Python 3.12 `pyproject.toml` 安装进该环境；按专用约束安装 Qwen、SAPIEN 和
ManiSkill2 依赖即可，启动器会直接设置两个源码目录的 `PYTHONPATH`：

```bash
python3.10 -m venv .venv-simpler
.venv-simpler/bin/pip install -r requirements-qwen-simpler-eval.txt
```

脚本不会安装或改写依赖，并会严格检查 Python 版本、NumPy 1.24.4、Qwen 推理栈、
SAPIEN/CUDA、两个源码 commit、overlay 资源以及 checkpoint 契约。先运行预检；它会
对四类环境分别创建、reset、读取相机/base-frame proprio 并执行一次安全零动作，且
让模型完成一次推理：

```bash
bash scripts/evaluate_simpler_qwen.sh \
  --python .venv-simpler/bin/python \
  --checkpoint /data/dwb/qwen_bridge/checkpoints/step-00020000 \
  --preflight-only
```

预检通过后运行四任务 smoke test（每任务 seed 0、object episode 0、最多 8 步）：

```bash
bash scripts/evaluate_simpler_qwen.sh \
  --python .venv-simpler/bin/python \
  --checkpoint /data/dwb/qwen_bridge/checkpoints/step-00020000 \
  --smoke-test
```

完整协议为 4 个任务 × object episode `0..23` × 策略种子 `0,2,4`，共 288 回合；
模型只加载一次，每个回合单独创建/关闭环境，每次预测并执行完整 8 步动作块：

```bash
bash scripts/evaluate_simpler_qwen.sh \
  --python .venv-simpler/bin/python \
  --checkpoint /data/dwb/qwen_bridge/checkpoints/step-00020000 \
  --tasks all \
  --output-dir outputs/qwen_simpler_eval
```

`--tasks` 也接受如 `spoon,eggplant` 的逗号列表；`--action-horizon` 可设为 1–8。
默认不录像；传入 `--save-videos-path outputs/qwen_simpler_videos` 后保存全部执行回合，
路径为 `TASK/seed-SEED/episode-ID_{success|failure}.mp4`，默认 5 FPS。结果目录在运行中
原子更新 `episodes.partial.jsonl`，正常结束后写 `episodes.jsonl` 与 `results.json`；
预检写 `preflight.json`，失败写 `failure.json`。已有输出不会覆盖，需显式传
`--overwrite`。

## 测试

```bash
pytest
pytest -m real_data
pytest -q tests/test_qwen_simpler_evaluation.py tests/test_evaluate_simpler_qwen_script.py
bash -n scripts/evaluate_simpler_qwen.sh
```

真实 4/8 卡 smoke test 必须在能访问 NVIDIA 驱动的训练机运行。
