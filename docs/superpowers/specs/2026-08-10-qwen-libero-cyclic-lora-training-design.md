# Qwen LIBERO 周期性 LoRA 训练设计

## 目标

新增一个基于现有 `train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh` 的训练入口。前 5000 个 optimizer step 仅训练 GR00T 动作头；之后每 100 步中的前 90 步仍仅训练动作头，最后 10 步训练 LoRA 与动作头。Qwen3-VL-4B 原始参数始终冻结。

## 入口与兼容性

- 新脚本 `scripts/train_libero_qwen3_vl_4b_groot_cyclic_lora_all_tasks_4x4090.sh` 包装现有 LIBERO 全任务脚本，默认注入 `lora_freeze_steps=5000`、`lora_cycle_steps=100`、`lora_active_steps=10`，并继续透传调用方参数。
- 新入口继承现有模型、4 卡、LIBERO target/prior 1:1 采样、SQCN Top10% prior 和显式输出目录约定。
- CLI 和训练配置增加对应的可选调度字段。未配置周期字段时，现有 Bridge 与 LIBERO 入口保持“冻结结束后持续训练 LoRA”的行为。

## 调度语义

- 步数使用 1-based optimizer step，不按 micro-step 计数；同一梯度累积窗口内 LoRA 状态保持不变。
- 第 1–5000 步关闭 LoRA。首个周期为 5001–5100，其中 5001–5090 关闭 LoRA，5091–5100 开启 LoRA；后续每 100 步重复。
- 动作头始终训练，并继续按全局 optimizer step 执行原有 warmup 与余弦衰减。
- LoRA 关闭时同时设为不可求导并将该 optimizer group 的调度 LR 置零；开启时恢复求导。
- LoRA warmup 与余弦衰减只按累计有效 LoRA 更新次数推进。默认 20000 个总步数下共有 1500 个 LoRA 更新步，前 500 个有效更新用于 warmup。

## 配置校验与日志

- `lora_freeze_steps` 必须是非负整数；周期长度必须是正整数；有效步数必须满足 `0 < active <= cycle`；周期长度与有效步数必须同时提供或同时省略。
- 日志中的 `train/lora_enabled` 表示刚完成的 optimizer step 是否更新了 LoRA，并记录累计 LoRA 更新步数；`train/lora_lr` 反映该步实际使用的 LoRA 学习率。

## 测试与验收

- 单元测试覆盖 5000/5001、5090/5091、5100/5101 等边界、多个周期及不完整尾周期。
- scheduler 测试验证冻结步 LR 为零、LoRA warmup/衰减仅按有效更新推进，并验证现有非周期配置行为不变。
- 配置测试覆盖合法组合、缺失成对字段和非法范围。
- shell 测试验证新入口继承原 LIBERO 默认参数并注入 5000/100/10，同时允许显式 CLI 参数覆盖默认值。
- 运行相关 pytest、shell 语法检查和代码静态检查；不启动真实多卡训练。
