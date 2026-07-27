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

首次转换默认从 `google-t5/t5-base` 获取公开 tokenizer/config，T5 模型权重
仍来自本地 Octo checkpoint。离线机器可指定已准备好的目录：

```bash
bash scripts/convert_octo_small_to_pytorch.sh \
  --t5-source /data/dwb/models/t5-base \
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
  --output /data/dwb/datasets/LIBERO/lerobot
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

#### Preflight 与四卡训练

原生 PyTorch DataLoader 直接读取 Parquet 内嵌 PNG，完成 Lanczos resize、
CHW 转换、`[-1,1]` 图像归一化、prior action/proprio 统计归一化和 8 步
action window。每个 worker 使用有界 episode LRU cache；确定性 DDP sampler
保证每个 micro-batch 都按 LIBERO-90 与五条 Book-caddy target `1:1` 采样：

```bash
bash scripts/train_libero_octo_small_4x4090.sh --preflight-only
bash scripts/train_libero_octo_small_4x4090.sh --smoke-test
bash scripts/train_libero_octo_small_4x4090.sh
```

训练脚本使用 `torchrun`、DDP 和 BF16；默认每卡 micro-batch 8、梯度累积 4，
四卡有效全局 batch 为 128。预检会验证 v2.0 metadata、全部 Parquet
footer/schema、任务映射、抽样 PNG、统计维度、非空 prior、恰好 5 条 target
episode、PyTorch safetensors/转换清单、CUDA 数量和 BF16 支持。
自定义数据根目录使用 `--lerobot-path`；旧数据参数不再接受：

```bash
bash scripts/train_libero_octo_small_4x4090.sh \
  --lerobot-path /data/dwb/datasets/LIBERO/lerobot \
  --preflight-only
```

训练 checkpoint 使用 safetensors 保存模型，并保存 AdamW、scheduler、
sampler 和 RNG 状态。严格续训：

```bash
bash scripts/train_libero_octo_small_4x4090.sh --resume latest
```

完整配置见 `configs/octo_small_libero_4x4090.yaml`，默认模型路径为
`/data/dwb/models/octo-small-pytorch`。

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

验证 checkpoint 能从第 20 步继续到第 21 步：

```bash
bash scripts/train_bridge_4x4090.sh --gpu-ids 0,1,2,3 --resume latest --max-steps 21
```

续训与 ZeRO-3/CPU optimizer offload：

```bash
bash scripts/train_bridge_4x4090.sh --gpu-ids 0,1,2,3 --resume latest
bash scripts/train_bridge_4x4090.sh --gpu-ids 0,1,2,3 --deepspeed-stage 3
```

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
- `checkpoints/` 下可严格恢复的 DeepSpeed checkpoint；
- `best/` 和 `inference/` 下的 LoRA + DiT 紧凑 Safetensors 推理包。

## 推理

```python
from qwen3_vl_groot.inference import BridgePolicy

policy = BridgePolicy.from_pretrained(
    "outputs/qwen3_vl_4b_groot_bridge/best",
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
