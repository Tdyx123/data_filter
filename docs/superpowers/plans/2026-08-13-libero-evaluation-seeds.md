# LIBERO 固定评测种子更新实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Octo-small 与复用其协议的 Qwen LIBERO 评测器的三个固定随机种子更新为 `3471197683`、`1232873419`、`1448008435`。

**Architecture:** 保留现有共享协议边界：`src/octo_small_libero/evaluation.py` 中的 `EVALUATION_SEEDS` 仍是唯一事实来源，Octo 与 Qwen 设置对象继续复用它。只同步依赖该协议的测试和用户文档，不新增可覆盖种子的命令行接口。

**Tech Stack:** Python 3、pytest、Bash、Markdown

## Global Constraints

- 固定种子必须严格为 `(3471197683, 1232873419, 1448008435)`，顺序不变。
- Qwen LIBERO 评测器继续复用 Octo 的 `EVALUATION_SEEDS`。
- episode 数量、初始状态顺序、并行度和其他评测参数保持不变。
- 不增加 `--seeds` 或其他种子覆盖接口。
- 不运行需要 GPU 或 MuJoCo 的真实闭环评测。

---

### Task 1: 更新共享固定种子协议及其回归覆盖

**Files:**
- Modify: `tests/test_octo_small_evaluation.py:435-472,475-621,1310-1390`
- Modify: `tests/test_qwen_libero_evaluation.py:400-455`
- Modify: `tests/test_evaluate_libero_script.py:236-243`
- Modify: `src/octo_small_libero/evaluation.py:47`
- Modify: `src/octo_small_libero/evaluate.py:34-41`
- Modify: `scripts/evaluate_libero_octo_small.sh:41-44`
- Modify: `README.md:510-512`

**Interfaces:**
- Consumes: `EVALUATION_SEEDS: tuple[int, int, int]`，由 Octo 与 Qwen 的设置类和评测循环读取。
- Produces: `EVALUATION_SEEDS == (3471197683, 1232873419, 1448008435)`，并在结果记录、生成器初始化、环境 reset 和帮助文本中保持同一顺序。

- [ ] **Step 1: 先把回归测试期望改为新协议**

在 `tests/test_octo_small_evaluation.py` 中定义测试期望并更新协议、循环和报告断言：

```python
EXPECTED_EVALUATION_SEEDS = (3471197683, 1232873419, 1448008435)


def test_evaluation_protocol_uses_three_fixed_seeds_and_balanced_episodes():
    settings = EvaluationSettings(episodes=150, num_envs=50)

    assert settings.seeds == EXPECTED_EVALUATION_SEEDS
```

将 episode 顺序断言中的 seed 依次改为上述三个值，并让 `_make_report` 测试数据及 `summary.by_seed` 使用相同值。在 CLI 默认测试中增加：

```python
help_text = parser.format_help()
for seed in EXPECTED_EVALUATION_SEEDS:
    assert str(seed) in help_text
```

在 `tests/test_qwen_libero_evaluation.py` 中导入或定义相同的期望元组，并将生成器、环境 reset 和 episode 行断言更新为三个新值：

```python
assert tuple(policy.generator_seeds) == EXPECTED_EVALUATION_SEEDS
assert tuple(environment.seeds) == EXPECTED_EVALUATION_SEEDS
```

在 `tests/test_evaluate_libero_script.py::test_help_lists_every_task_without_starting_python` 中增加：

```python
for seed in (3471197683, 1232873419, 1448008435):
    assert str(seed) in completed.stdout
```

- [ ] **Step 2: 运行定向测试并确认 RED**

Run:

```bash
pytest -q \
  tests/test_octo_small_evaluation.py::test_evaluation_protocol_uses_three_fixed_seeds_and_balanced_episodes \
  tests/test_octo_small_evaluation.py::test_evaluation_cli_defaults_and_checkpoint_arguments \
  tests/test_qwen_libero_evaluation.py::test_qwen_evaluation_repeats_fixed_states_and_writes_qwen_report \
  tests/test_evaluate_libero_script.py::test_help_lists_every_task_without_starting_python
```

Expected: FAIL；默认设置仍返回 `(0, 1, 2)`，且 CLI/shell 帮助尚未包含新种子。

- [ ] **Step 3: 最小化更新实现和说明**

在 `src/octo_small_libero/evaluation.py` 中更新唯一事实来源：

```python
EVALUATION_SEEDS = (3471197683, 1232873419, 1448008435)
```

将 `src/octo_small_libero/evaluate.py` 的 `--episodes` help、`scripts/evaluate_libero_octo_small.sh` 的协议说明以及 README 的中文说明同步为这三个十进制值。保持“3 个种子、每个任务默认 150 episodes、每个种子 50 个初始状态”的现有语义。

- [ ] **Step 4: 运行定向测试并确认 GREEN**

Run:

```bash
pytest -q tests/test_octo_small_evaluation.py tests/test_qwen_libero_evaluation.py tests/test_evaluate_libero_script.py -m "not gpu and not slow"
```

Expected: PASS，且没有新的 warning 或错误。

- [ ] **Step 5: 执行静态检查和旧协议残留检查**

Run:

```bash
bash -n scripts/evaluate_libero_octo_small.sh
git diff --check
rg -n 'fixed seeds 0, 1, and 2|种子 `0`、`1`、`2`|EVALUATION_SEEDS = \(0, 1, 2\)' src scripts tests README.md
```

Expected: 前两个命令退出码为 0；`rg` 无输出且退出码为 1，说明目标范围内没有旧协议描述。

- [ ] **Step 6: 提交实现**

```bash
git add README.md scripts/evaluate_libero_octo_small.sh src/octo_small_libero/evaluate.py src/octo_small_libero/evaluation.py tests/test_evaluate_libero_script.py tests/test_octo_small_evaluation.py tests/test_qwen_libero_evaluation.py docs/superpowers/plans/2026-08-13-libero-evaluation-seeds.md
git commit -m "fix: update LIBERO evaluation seeds"
```
