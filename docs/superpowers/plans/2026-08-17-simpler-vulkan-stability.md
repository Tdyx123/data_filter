# SimplerEnv Vulkan 长跑稳定性修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. All production changes must follow RED → GREEN TDD.

**Goal:** 三类模型在每个任务内复用唯一的离屏 SAPIEN 环境，通过逻辑 CUDA 设备参数隔离模型和模拟器 GPU，并保持既有评测、错误与 IPC 协议不变。

**Architecture:** 共享 runner 把环境生命周期从 episode 提升到 task；每个 task 进入时创建一次环境，在所有 seed/object episode 完成或异常时关闭一次。公共 settings 携带独立的 `sim_device`，环境 builder 统一创建离屏 renderer。Qwen、Octo-small、StarVLA Python 入口及三个 shell 启动器只负责验证和转发该参数。

**Tech Stack:** Python 3.12/3.10、pytest、SAPIEN/SimplerEnv、Bash、Ruff。

## Global Constraints

- 每个任务只创建一个 SAPIEN 环境；每个回合仍按 seed/object 顺序调用 `env.reset()`。
- 每个 `task × policy_seed` 只创建一个策略 generator。
- 每个完成回合后仍更新 `episodes.partial.jsonl`，完成后仍清除 partial。
- reset/renderer/环境创建故障仍以 `SimplerInfrastructureError` 立即终止；策略错误继续使用现有 task error 语义。
- `SimplerRunSettings` 与 Qwen 对应 settings 的 `sim_device` 默认值均为 `"cuda:0"`。
- `--sim-device` 仅接受 `cuda:<非负逻辑编号>`，默认 `cuda:0`。
- `create_simpler_environment(..., sim_device="cuda:0")` 必须向 SAPIEN 传递 `renderer_kwargs={"offscreen_only": True, "device": sim_device}`。
- StarVLA 的模型 `--device` 与模拟器 `--sim-device` 完全独立。
- 结果协议必须新增 `sim_renderer_device`、`sim_renderer_offscreen_only: true`、`environment_lifecycle: "one_per_task"`。
- 保持 StarVLA IPC v2、动作协议、任务/种子/对象集合、固定源码版本及逐步推理不变。
- 不增加 Vulkan 自动重试、交互式 viewer 或模拟器分批重启。
- 不恢复现有 partial RNG 状态；正式结果从 seed 0 使用 `--overwrite` 完整重跑。
- `CUDA_VISIBLE_DEVICES` 顺序定义逻辑 CUDA 编号。

---

### Task 1: 共享 runner 生命周期、renderer 设置与结果协议

**Files:**
- Modify: `src/simpler_bridge/evaluation.py`
- Modify: `tests/test_simpler_bridge_evaluation.py`
- Modify: `tests/test_qwen_simpler_evaluation.py`

**Requirements:**

- [ ] 先写失败测试：两任务、多 seed、多 object 时，environment factory 每任务只调用一次。
- [ ] 先写失败测试：同一 task 环境严格按 seed/object 顺序重复 reset，任务结束或异常时只 close 一次。
- [ ] 先写失败测试：generator 仍按每个 `task × seed` 创建一次，partial 每完成一个回合更新一次。
- [ ] 先写失败测试：策略错误、reset 失败、环境创建失败分别保持现有 task error、立即基础设施失败语义。
- [ ] 将环境创建移到 task 循环，使用 `try/finally` 保证完成和异常路径恰好关闭一次；不改变 episode 顺序和策略推理路径。
- [ ] 给 `SimplerRunSettings` 增加 `sim_device: str = "cuda:0"`，验证其严格匹配 `cuda:<非负十进制整数>`。
- [ ] 扩展 `create_simpler_environment(..., sim_device="cuda:0")`，准确传入 `renderer_kwargs={"offscreen_only": True, "device": sim_device}`。
- [ ] 在 protocol/report 中写入 `sim_renderer_device`、`sim_renderer_offscreen_only: true`、`environment_lifecycle: "one_per_task"`。
- [ ] 运行聚焦测试确认 RED，再做最小实现并确认 GREEN。
- [ ] 提交本任务及本计划文档。

### Task 2: 三个 Python 模型入口统一 `--sim-device`

**Files:**
- Modify: `src/qwen3_vl_groot/simpler_evaluation.py`
- Modify: `src/qwen3_vl_groot/evaluate_simpler.py`
- Modify: `src/octo_small_bridge/evaluate_simpler.py`
- Modify: `src/starvla_bridge/evaluate_simpler.py`
- Modify: `tests/test_qwen_simpler_evaluation.py`
- Modify: `tests/test_evaluate_simpler_qwen_script.py`
- Modify: `tests/test_octo_small_simpler_evaluation.py`
- Modify: `tests/test_starvla_simpler_evaluation.py`

**Requirements:**

- [ ] 先写失败测试：Qwen 对应 settings 新增 `sim_device: str = "cuda:0"`，转换到 shared settings 时不丢失。
- [ ] 先写失败测试：Qwen、Octo-small、StarVLA parser 默认 `cuda:0`，接受合法逻辑编号，拒绝负数、裸 `cuda`、物理编号或其他设备格式。
- [ ] 先写失败测试：三个入口把 settings 的 `sim_device` 传给环境 builder；preflight 与正式 evaluation 一致。
- [ ] 实现统一的公共参数验证/解析路径，避免三个入口出现不同语义。
- [ ] StarVLA 继续把模型运行时元数据保留为 remote device，模拟器单独使用 `sim_device`。
- [ ] 不改变 StarVLA IPC v2、动作转换、任务集合、seed/object 集合或源码版本验证。
- [ ] 运行聚焦测试确认 RED，再做最小实现并确认 GREEN。
- [ ] 提交本任务。

### Task 3: 三个启动器转发、清理语义与静态验收

**Files:**
- Modify: `scripts/evaluate_simpler_qwen.sh`
- Modify: `scripts/evaluate_simpler_octo_small.sh`
- Modify: `scripts/evaluate_simpler_starvla.sh`
- Modify: `tests/test_evaluate_simpler_qwen_script.py`
- Modify: `tests/test_evaluate_simpler_starvla_script.py`
- Modify/Create: Octo-small 启动器对应测试（遵循现有测试布局）

**Requirements:**

- [ ] 先写失败测试：三个启动器默认/显式转发 `--sim-device`，并拒绝非法值、缺值和重复值。
- [ ] Qwen、Octo-small 启动器显式解析后统一转发；StarVLA 帮助文本与参数解析清晰区分模型 `--device` 和模拟器 `--sim-device`。
- [ ] StarVLA 双进程清理、socket 管理、退出码传播保持不变。
- [ ] 运行共享 runner、Qwen、Octo-small、StarVLA、三个启动器清理测试。
- [ ] 运行 Ruff、`git diff --check`，并对三个脚本执行 `bash -n`。
- [ ] 提交本任务。

## GPU 验收（全部代码任务与评审完成后由主控制器执行）

- 使用 `CUDA_VISIBLE_DEVICES=4,5`。
- 模型使用 `--device cuda:1`（物理 GPU 5）。
- 模拟器使用 `--sim-device cuda:0`（物理 GPU 4）。
- 先用 spoon 单任务从 seed 0 完整运行 72 回合，确认越过原第 32 次 renderer 创建故障。
- 再用原正式输出目录和 `--overwrite` 从头完整运行 288 回合，不续跑既有 31 条 partial。
- 正式结果必须满足：`episodes.jsonl` 为 288 条、`task_errors` 为空、日志无 `ErrorExtensionNotPresent`、partial 文件已清除、两个子进程正常退出。
