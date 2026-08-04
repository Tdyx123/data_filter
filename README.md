# Qwen3-VL-4B-GR00T Bridge

这是一个独立的 BridgeData V2 视觉—语言—动作（VLA）微调项目。它完整加载本地
Qwen3-VL-4B-Instruct 的 36 层文本模型，在每一层注意力的
`q_proj/k_proj/v_proj/o_proj` 上训练 LoRA，并从随机初始化的
GR00T 风格 flow-matching DiT 动作头开始训练。

仓库同时包含与训练解耦的通用轨迹数据价值工具
[TDUS](tdus/README.md)，用于直接从 LeRobot trajectory/chunk 计算 Quality、
Coverage、Diversity 和 Novelty。

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

FlashAttention 2 是可选项；未安装时自动使用 PyTorch SDPA：

```bash
pip install -e ".[flash]"
```

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
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
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
bash scripts/train_libero_octo_small_4x4090.sh --task-index 5 --preflight-only
bash scripts/train_libero_octo_small_4x4090.sh --task-index 5 --smoke-test
bash scripts/train_libero_octo_small_4x4090.sh --task-index 5
```

如需让 LIBERO-10 的全部 10 个任务、50 条完整 target 轨迹共同参与训练，使用
独立的四卡全任务脚本。每个 micro-batch 按 `3:1` 采样，由 75% 全任务
target 池和 25% LIBERO-90 prior 组成：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh --preflight-only
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh --smoke-test
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh
```

如果只使用上述 `libero10_5` 中每个任务 5 条、共 50 条示例轨迹，不采样也不读取
LIBERO-90 prior 或 SQCN/TDUS scores，追加 `--target-only`：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --target-only \
  --preflight-only
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --target-only \
  --smoke-test
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --target-only
```

该模式的每卡 micro-batch 8 条样本全部来自 `libero10_5`，action 与 proprio
归一化也使用 `libero10_5/meta/stats.json`。默认输出目录为
`outputs/octo_small_libero_4gpu_all-tasks_target-only`。`--target-only` 不能与
`--sample-weights` 或任何 `--prior-*` 参数同时使用；不传该开关时，脚本仍保持
下面所述的 3:1 target/prior 混合训练行为。

全任务 target 池内部按帧均匀采样；因此不同任务的抽样频率会随其轨迹总帧数
变化。该脚本默认把
`/data/dwb/libero90_sqcn/filter/top10pct/scores.csv` 作为已经筛选完成的 SQCN
片段清单，使用文件中的全部 4671 个片段，不再按分数二次截取。每个片段只
贡献完整落在片段范围内的 8-step action window；重叠片段去重后覆盖 2923 条
episode，共得到 36263 个 prior 训练起点。默认输出目录为
`outputs/octo_small_libero_4gpu_all-tasks_sqcn_top10pct`。

预检会核对同目录 `filter_manifest.json`、SQCN `run_manifest.json`、数据集来源、
文件哈希、选择摘要、连续 `filter_rank` 和片段边界。可以重复传入新参数和输出
参数来覆盖脚本默认值，后出现的值生效：

```bash
bash scripts/train_libero_octo_small_all_tasks_4x4090.sh \
  --prior-prefiltered-scores /path/to/filter/top10pct/scores.csv \
  --output-dir outputs/custom_sqcn_run \
  --preflight-only
```

`--prior-prefiltered-scores` 不能与 TDUS 的 `--prior-top-percent` 或
`--prior-scores` 混用。需要 TDUS 排名筛选时，继续使用单任务入口，或直接调用
`python3 -m octo_small_libero.cli --all-tasks` 并传入 TDUS 参数。

训练脚本使用 `torchrun`、DDP 和 BF16；默认每卡 micro-batch 8，其中全任务
target 6 条、prior 2 条；梯度累积 4 后，四卡有效全局 batch 为 128，其中
target 96 条、prior 32 条。单任务脚本仍保持 1:1；权重 sweep 单独使用
3:1。预检会验证
v2.0 metadata、全部 Parquet
footer/schema、评测顺序的 10-task 映射、单任务模式的 5 条或全任务模式的
50 条 target episode、抽样 PNG、统计维度、非空 prior、PyTorch
safetensors/转换清单、CUDA 数量和 BF16 支持。
自定义数据根目录使用 `--lerobot-path`；旧数据参数不再接受：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --lerobot-path /data/dwb/datasets/LIBERO_lerobot \
  --preflight-only
```

#### 使用 TDUS Top 片段训练

训练入口可根据 `outputs/tdus/libero90/chunk/scores.csv` 的 `tdus` 分数筛选
LIBERO-90 prior，同时保留所选任务完整的 5 条 target 轨迹，并继续按 1:1
组成每个 micro-batch。Top 10% 和 Top 20% 分别运行：

```bash
bash scripts/train_libero_octo_small_4x4090.sh --task-index 5 --prior-top-percent 10
bash scripts/train_libero_octo_small_4x4090.sh --task-index 5 --prior-top-percent 20
```

比例接受 `(0, 100]` 内的任意数值，选中 chunk 数按向上取整计算。排序固定为
`tdus` 降序、`length` 升序、`sample_id` 升序。一个选中 chunk 只贡献完整落在
片段内的 8-step action window 起点，重叠 chunk 的相同 episode/frame 起点只
保留一次。`--prior-top-percent 100` 仍应用这套严格片段边界；不传该参数才是
包含 episode 尾部 padding window 的原始全帧基线。

未显式传 `--output-dir` 时，输出目录会先追加 `_task-5` 等任务后缀，再追加
`_top10pct`、`_top20pct` 等 prior 比例后缀，避免任务或比例互相覆盖。也可以
覆盖 scores 文件和输出位置：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --prior-top-percent 12.5 \
  --prior-scores /path/to/libero90/chunk/scores.csv \
  --output-dir outputs/octo_small_libero_top12p5
```

`--preflight-only` 和 `--smoke-test` 支持相同参数。预检会核对 TDUS
`run_manifest.json` 中的数据源，并报告选中的 chunk、episode 和去重后的训练
起点数。训练 manifest 会记录 scores SHA256 与选择摘要；续训必须使用完全相同
的比例和 scores 内容：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --task-index 5 \
  --prior-top-percent 10 \
  --resume latest
```

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
LIBERO-10 target 与 LIBERO-90 prior `3:1` 采样；单模型 micro-batch 8 中包含
6 条 target 和 2 条 prior，梯度累积 16 后有效 batch 128 中包含 96 条 target
和 32 条 prior。运行正式 sweep 前建议先确认同卡 2 个 Octo-small 模型能够
同时装入显存：

```bash
python3 scripts/train_libero_octo_small_all_tasks_weight_sweep.py \
  --preflight-only

python3 scripts/train_libero_octo_small_all_tasks_weight_sweep.py \
  --smoke-test
```

第 1 行权重生成 `scores_weights_001.csv` 和 `model-001/`，依此类推。
重新换算的 CSV 保存在原 `chunk/` 目录中，原始 `scores.csv` 不会被修改。
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

`--max-steps` 会应用到每一行权重对应的模型，同时用于判断已有模型应跳过还是从
`latest` 续训；不传时使用单卡配置中的 10,000 步。

重复运行时，已达到目标步数的模型会跳过，存在完整但未完成 checkpoint 的模型
会从 `latest` 续训。若权重、源 scores 或重新换算后的 scores 与已有 manifest
不一致，或者旧输出不是 3:1 采样，脚本会保留现场并将该行报告为失败。首次使用
3:1 sweep 前需要人工移动或清理同一路径下的旧 1:1 checkpoint。单个模型失败
不会阻止其他行训练；最终 `sweep_summary.json` 会列出失败行，且批量脚本返回
非零状态。

默认训练范围是第 1–10,000 步，前 400 步 warmup，余弦学习率衰减覆盖完整
10,000 步；`--max-steps` 可以覆盖训练上限。每 1,000 步和最终步保存候选
checkpoint，并使用该保存区间内的平均训练 loss 选择 best。`checkpoints/`
最多保留 latest 与 best 对应的两个 `step-*` 目录；二者重合时只保留一个。
`latest.json` 和 `best.json` 分别记录最新步和最佳步，不会额外创建 `best/`
目录。

训练 checkpoint 使用 safetensors 保存模型，并保存 AdamW、scheduler、
sampler 和 RNG 状态。严格续训：

```bash
bash scripts/train_libero_octo_small_4x4090.sh --task-index 5 --resume latest
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

评测固定使用随机种子 `0`、`1`、`2`，每个种子都依次使用相同的 LIBERO-10
初始状态，并采用 20 个稳定步、8 步 action chunk 和 960 步上限。`--episodes`
表示三个种子的总 episode 数，必须能被 3 整除。
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
  --model-path /data/dwb/models/Qwen3-VL-4B-Instruct \
  --dataset-path /data/dwb/datasets/bridge_orig_1.0.0_lerobo \
  --output-dir outputs/qwen3_vl_4b_groot_bridge_4gpu
```

`--gpu-ids` 是宿主机上的物理编号。启动器会将其写入
`CUDA_VISIBLE_DEVICES`，随后用 4 个 `torchrun` 进程训练；编号数量必须与
`--gpu-count` 一致。例如改成 GPU `4,5,6,7`：

```bash
bash scripts/train_bridge_4x4090.sh --gpu-ids 4,5,6,7
```

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

4 卡和 8 卡配置默认同时关闭 Qwen 主干与 DiT 动作头的 gradient checkpointing，
并使用 PyTorch Inductor 原地编译两个模块。编译采用动态 shape、允许局部 graph
break；首次训练调用以及第 2,000 步 LoRA 解冻后可能出现一次性编译延迟。该配置
减少重复计算并提高稳定阶段吞吐，但会增加激活显存占用。遇到编译器或算子兼容
问题时，可在配置中将 `model.torch_compile.enabled` 改为 `false`。

完整参数见：

```bash
python -m qwen3_vl_groot.cli launch --help
```

## 数据约定

首版只读取 `observation.images.image_0`。数据中前六维 EEF 动作已是相对动作，
第七维 gripper 是绝对值；管线不会再次减去当前状态。训练/验证按 episode 的稳定
哈希切分。q01/q99 仅从训练 episode 计算，保存在输出目录的
`data_cache/normalization.json`，原始数据集不会被修改。

## 输出

输出目录包含：

- `run_config.yaml`、`data_fingerprint.json` 和归一化统计；
- `metrics.jsonl` 与 TensorBoard event；
- `checkpoints/step-XXXXXXXX/` 下仅含 LoRA、动作头和必要推理元数据的紧凑
  Safetensors checkpoint；
- `checkpoints/latest.json` 与 `checkpoints/best.json` 分别标记最新和验证集
  最优 step。

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

## 测试

```bash
pytest
pytest -m real_data
```

真实 4/8 卡 smoke test 必须在能访问 NVIDIA 驱动的训练机运行。
