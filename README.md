# Qwen3-VL-4B-GR00T Bridge

这是一个独立的 BridgeData V2 视觉—语言—动作（VLA）微调项目。它完整加载本地
Qwen3-VL-4B-Instruct 的 36 层文本模型，在每一层注意力的
`q_proj/k_proj/v_proj/o_proj` 上训练 LoRA，并从随机初始化的
GR00T 风格 flow-matching DiT 动作头开始训练。

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

默认配置面向 8×RTX 4090：

```bash
bash scripts/train_bridge_8x4090.sh \
  --model-path /data/dwb/models/Qwen3-VL-4B-Instruct \
  --dataset-path /data/dwb/datasets/bridge_orig_1.0.0_lerobo \
  --output-dir outputs/qwen3_vl_4b_groot_bridge
```

启动器先校验模型、Bridge 元数据、三段真实视频和单卡 micro-batch 显存，再启动
8 个 `torchrun` 进程。只运行预检：

```bash
bash scripts/train_bridge_8x4090.sh --preflight-only
```

运行 20 个 optimizer step 的验收：

```bash
bash scripts/train_bridge_8x4090.sh --smoke-test
```

验证 checkpoint 能从第 20 步继续到第 21 步：

```bash
bash scripts/train_bridge_8x4090.sh --resume latest --max-steps 21
```

续训与 ZeRO-3/CPU optimizer offload：

```bash
bash scripts/train_bridge_8x4090.sh --resume latest
bash scripts/train_bridge_8x4090.sh --deepspeed-stage 3
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

真实 8 卡 smoke test 必须在能访问 NVIDIA 驱动的训练机运行。
