# Octo-small BridgeData V2 训练适配设计

## 目标

新增独立的 Octo-small BridgeData V2 微调入口，直接读取已有 LeRobot v2 数据，
不转换或复制数据，也不改变现有 LIBERO 训练入口。默认入口面向 4×RTX 4090，
支持预检、两步 smoke test、正式训练和严格断点续训。

## 命令与配置

启动脚本为 `scripts/train_bridge_octo_small_4x4090.sh`，配置文件为
`configs/octo_small_bridge_v2_4x4090.yaml`，Python 入口为
`python -m octo_small_bridge.cli`。命令必须显式传入 `--output-dir`，并接受
`--dataset-path`、`--model-path`、`--gpu-ids`、`--max-steps`、`--resume`、
`--preflight-only` 和 `--smoke-test`。

预检由单个 Python 进程完成；正式训练通过四个 `torchrun` 进程执行。默认模型为
`/data/dwb/models/octo-small-pytorch`，默认数据集为
`/data/dwb/datasets/bridge_orig_1.0.0_lerobo`。

## 数据契约

适配器要求 LeRobot v2.0、WidowX、5 Hz、AV1 视频，并固定读取：

- `observation.images.image_0` 作为 256×256 `image_primary`；
- 8 维 `observation.state` 作为 proprio；
- 7 维 `action` 作为八步动作窗口。

只保留语言任务非空的 episode。默认挂载数据应从 53,192 条源 episode 中排除
14,532 条空任务 episode，保留 38,660 条和 1,305,714 帧。`image_1/2/3` 不进入
Octo batch，缺失 wrist 使用 Octo 的缺失模态语义。

图像训练增广沿用本地 Octo checkpoint 的 primary 配置：随机缩放裁剪 scale
`[0.8, 1.0]`、ratio `[0.9, 1.1]`，brightness `0.1`、contrast/saturation
`[0.9, 1.1]`、hue `0.05`，输出 256×256；预检使用确定性 resize。

proprio 和动作前六维使用 Bridge `meta/stats.json` 的 mean/std 标准化，零方差维使用
epsilon。第七维抓手动作是 `[0,1]` 连续值，并排除统计标准化；仅接受有限值及
`[-1e-5, 1+1e-5]` 范围，先将容差内值裁剪到 `[0,1]`，再按 `2*x-1` 转换为
Octo Bridge 约定的 `[-1,+1]`。非有限值或明显越界值必须抛出包含 episode ID 的
`DatasetValidationError`。episode 尾部动作窗口重复最后一个完整动作作为占位，
同时通过 `action_pad_mask` 排除 loss。

## 采样、模型与恢复

Bridge 使用 map-style frame dataset 和 episode-aware 分布式 batch sampler。每个
epoch 确定性打乱有效 episode，各 rank 获得不重叠分片，episode 内帧顺序也由 seed
确定；local batch 尽量从同一 episode 取帧，使 worker 的 LRU 缓存只解码一次
Parquet/AV1。索引携带 epoch、episode 和 frame position，使图像增广不依赖 worker
调度。

sampler 的 checkpoint 状态只记录已发出的 batch 数，并能按 rank、seed 和 metadata
重建下一批，保证续训数据顺序一致。训练 checkpoint 继续保存模型、AdamW、scheduler、
sampler 和 RNG 状态。

`OctoSmallPolicy` 接受可选 `image_wrist`。Bridge 前向只拼接 primary、proprio 和
readout token；加载 Bridge 训练模型时冻结所有 `wrist_*` 参数。LIBERO 仍提供双相机
batch，行为保持不变。

## 训练默认值与输出

默认 global batch 128、每卡 micro-batch 8、梯度累积 4、10,000 optimizer steps、
400 步 warmup、峰值学习率 `3e-4`、BF16。首版使用全部非空 episode，不划分离线
验证集；`best` 继续按保存区间平均训练 loss 选择。输出沿用现有 Octo 格式，包括
配置、数据/model manifest、metrics、latest/best 指针和最多两个 step checkpoint。

## 范围边界与测试

首版不实现 SimplerEnv 评测，不接受 Cocore/RelCore 选择清单，不包含空语言 episode，
也不修改源数据集。测试覆盖 Shell/CLI、schema 与预检、合成 AV1 数据、抓手转换、
尾部 mask、sampler 多 rank 隔离/复现/恢复、单相机模型前后向、LIBERO 双相机回归、
checkpoint round-trip 以及默认挂载真实数据的首中尾 episode 解码。
