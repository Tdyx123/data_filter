# Octo-small BridgeData V2 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个可直接读取 BridgeData V2 LeRobot 数据的 Octo-small 四卡微调入口。

**Architecture:** 在独立 `octo_small_bridge` 包中实现 Bridge 配置、数据、采样、预检和 CLI，复用现有 Octo PyTorch 模型、训练循环和 checkpoint。共享 LeRobot adapter 增加单 episode 读取接口，Octo 模型增加可选 wrist 模态，LIBERO 路线保持兼容。

**Tech Stack:** Python 3.10、PyTorch 2.4.1/DDP、torchvision 0.19.1、PyAV 14.0.1、PyArrow 14.0.2、pytest、Bash

## Global Constraints

- 直接在 `main` 上实现，但必须保留用户已有未提交修改。
- 生产行为遵循 TDD：先写测试并确认因缺失行为失败，再写最小实现。
- 默认只读取 `image_0`、8 维 state、7 维 action，并排除空语言 episode。
- 抓手动作从 `{0,1}` 转换到 `{-1,+1}`，前六维动作和 proprio 使用 Bridge stats。
- 不写源数据、不实现评测、不支持筛选 manifest、不划分验证集。
- 默认四卡、global batch 128、micro-batch 8、累积 4、10,000 步、warmup 400、峰值学习率 `3e-4`、BF16。

---

### Task 1: 单 episode LeRobot 读取接口

**Files:**
- Modify: `trajectory_data/lerobot.py`
- Test: `tests/test_octo_small_bridge.py`

**Interfaces:**
- Produces: `LeRobotDatasetAdapter.load_episode(record: EpisodeRecord, *, load_images: bool = True) -> EpisodeData`。

- [ ] 写测试：已索引 record 可只解码该 episode，未知或被排除的 record 被拒绝。
- [ ] 运行目标测试，确认因 `load_episode` 不存在而失败。
- [ ] 用现有 `_worker_payload`/`_load_lerobot_episode` 实现最小公开方法。
- [ ] 运行目标测试和 `trajectory_data` 回归测试。

### Task 2: Bridge frame dataset 与可恢复 sampler

**Files:**
- Create: `src/octo_small_bridge/data.py`
- Create: `src/octo_small_bridge/__init__.py`
- Test: `tests/test_octo_small_bridge.py`

**Interfaces:**
- Produces: `BridgeFrameRef`、`BridgeFrameDataset`、`BridgeDistributedBatchSampler`、`BridgeTrainingData`、`make_training_dataset(...)`。
- Consumes: Task 1 的 `load_episode` 和 Bridge `meta/stats.json`。

- [ ] 写合成 LeRobot v2/AV1 fixture 和失败测试，覆盖空任务过滤、图像/proprio/action batch、抓手转换和尾部 mask。
- [ ] 运行测试并确认因 Bridge 数据模块缺失而失败。
- [ ] 实现 stats 校验、episode LRU、确定性图像增广、动作窗口和 tokenizer 输出。
- [ ] 写 sampler 失败测试，覆盖固定 seed、rank 不重叠、episode locality 和 state round-trip。
- [ ] 实现按 epoch/rank 确定性重建的 batch sampler，并保持目标测试全绿。

### Task 3: Octo 单相机模型与共享训练循环

**Files:**
- Modify: `src/octo_small_libero/torch_model.py`
- Modify: `src/octo_small_libero/modeling.py`
- Modify: `src/octo_small_libero/training.py`
- Create: `src/octo_small_bridge/training.py`
- Test: `tests/test_octo_small_pytorch.py`
- Test: `tests/test_octo_small_bridge.py`

**Interfaces:**
- `OctoSmallPolicy.encode_observation(batch)` 接受可选 `image_wrist`。
- `load_pytorch_model(..., observation_tokenizers=("primary", "wrist"))` 可冻结未使用 wrist 参数。
- 共享训练函数接受 `training_data_builder` 与 `dataset_manifest_builder` 回调。

- [ ] 写无 wrist 前后向和冻结参数失败测试，同时保留双相机测试。
- [ ] 运行测试并确认当前实现因缺少 `image_wrist` 失败。
- [ ] 最小修改 token 拼接与模型加载逻辑，使单/双相机测试通过。
- [ ] 写 Bridge training wrapper/manifest 与恢复失败测试。
- [ ] 抽取共享训练注入点并实现 Bridge wrapper，运行 Octo 训练/checkpoint 回归。

### Task 4: 配置、预检、CLI 和启动脚本

**Files:**
- Create: `src/octo_small_bridge/config.py`
- Create: `src/octo_small_bridge/preflight.py`
- Create: `src/octo_small_bridge/cli.py`
- Create: `configs/octo_small_bridge_v2_4x4090.yaml`
- Create: `scripts/train_bridge_octo_small_4x4090.sh`
- Test: `tests/test_octo_small_bridge.py`
- Test: `tests/test_train_bridge_octo_script.py`

**Interfaces:**
- Python CLI 支持已批准的八个覆盖参数，要求显式 `--output-dir`。
- Shell 预检调用 `python3 -m octo_small_bridge.cli`，训练调用四进程 `torchrun`。

- [ ] 写配置/CLI 校验和 fake executable Shell 失败测试。
- [ ] 运行测试并确认因模块/脚本缺失而失败。
- [ ] 实现 YAML 校验、路径解析、预检报告和 CLI 调度。
- [ ] 实现 Shell 启动器并运行目标测试与 `bash -n`。

### Task 5: 文档、依赖与完整验证

**Files:**
- Modify: `requirements-octo-pytorch.txt`
- Modify: `README.md`
- Create: `tests/test_octo_small_bridge_real.py`

- [ ] 增加 torchvision/PyAV 固定版本和 Octo Bridge 使用说明。
- [ ] 写真实数据测试，断言默认挂载数据有效/排除计数并解码首中尾有效 episode。
- [ ] 运行 Bridge/Octo/trajectory_data 目标测试、全部 pytest 和 Shell 语法检查。
- [ ] 在具备 Octo 环境与 GPU 时运行 `--preflight-only` 和 2-step `--smoke-test`；无 GPU 时明确记录未运行原因。
- [ ] 检查 `git diff`，确认没有覆盖任务外修改，再进入 verification/code-review/branch finishing 流程。
