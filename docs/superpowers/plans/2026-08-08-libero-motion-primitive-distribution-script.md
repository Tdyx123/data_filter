# LIBERO Motion Primitive Distribution Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个命令行脚本，从完整 LeRobot v2 数据集逐 episode 生成 LIBERO 运动原语，并向指定目录写出全局类别数量和占比 CSV。

**Architecture:** 脚本复用 `LeRobotDatasetAdapter`，关闭图像加载并仅选择 `observation.state`；每个 episode 独立调用现有运动原语 API，最后用现有统计函数汇总并原子写入 CSV。测试通过临时构造的最小 LeRobot v2 Parquet 数据集，从 CLI 层验证边界、输出和失败行为。

**Tech Stack:** Python 3.12、NumPy、PyArrow、argparse、pytest。

## Global Constraints

- 直接在当前 `main` 分支修改，不创建 worktree 或功能分支。
- 不加载图像，不跨 episode 计算，不修改现有分类规则。
- 不新增依赖；输出固定为 UTF-8 CSV `motion_primitive_distribution.csv`。
- `--horizon` 默认 `4` 且范围 `3..8`；`--threshold` 默认 `0.03` 且为有限非负数；`--tail-strategy` 默认 `truncate`。

---

### Task 1: CLI 行为与数据集汇总

**Files:**
- Create: `tests/test_generate_libero_motion_primitive_distribution_script.py`
- Create: `scripts/generate_libero_motion_primitive_distribution.py`

**Interfaces:**
- Consumes: `LeRobotDatasetAdapter`, `make_libero_config()`, `generate_motion_primitives()`, `compute_primitive_statistics()`。
- Produces: `generate_distribution(dataset_root: Path, *, horizon: int, threshold: float, tail_strategy: TailStrategy) -> tuple[list[tuple[str, int, float]], int, int]`、`write_distribution_csv(statistics, output_dir: Path) -> Path` 和 `main(argv: Sequence[str] | None = None) -> int`。

- [ ] **Step 1: 写出正常汇总和 episode 边界的失败测试**

```python
def test_cli_writes_global_distribution_without_crossing_episode_boundaries(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _write_dataset(
        dataset,
        [
            _axis_states(5, axis=0, step=0.02),
            _axis_states(5, axis=1, step=-0.02),
        ],
    )
    output = tmp_path / "nested" / "output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root", str(dataset),
            "--output-dir", str(output),
            "--horizon", "3",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with (output / "motion_primitive_distribution.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"primitive": "move forward", "count": "2", "proportion": "0.5"},
        {"primitive": "move right", "count": "2", "proportion": "0.5"},
    ]
    assert "Processed episodes: 2" in result.stdout
    assert "Generated labels: 4" in result.stdout
```

- [ ] **Step 2: 运行测试并确认因脚本缺失而失败**

Run: `.venv/bin/python -m pytest tests/test_generate_libero_motion_primitive_distribution_script.py::test_cli_writes_global_distribution_without_crossing_episode_boundaries -q`

Expected: FAIL，子进程返回非零状态并报告找不到 `scripts/generate_libero_motion_primitive_distribution.py`。

- [ ] **Step 3: 实现最小 CLI、逐 episode 汇总和原子 CSV 写入**

```python
def generate_distribution(dataset_root, *, horizon, threshold, tail_strategy):
    adapter = LeRobotDatasetAdapter({
        "path": str(dataset_root),
        "use_images": False,
        "feature_keys": {"vector_observations": [STATE_KEY]},
    })
    config = make_libero_config(
        horizon=horizon,
        threshold=threshold,
        tail_strategy=tail_strategy,
    )
    labels = []
    for episode in adapter.iter_episodes(num_workers=0, load_images=False):
        labels.extend(generate_motion_primitives(episode.observations[STATE_KEY], config))
    return compute_primitive_statistics(labels), len(adapter.episodes()), len(labels)
```

`write_distribution_csv()` 先创建输出目录，再用同目录 `NamedTemporaryFile` 写入
`primitive,count,proportion`，成功后 `os.replace()`，异常时删除临时文件。
`main()` 解析必填的 `--dataset-root`、`--output-dir` 及三个分类参数，成功时打印
episode 数、标签数和 CSV 绝对路径；捕获数据校验、I/O 和配置错误后向 stderr 打印
`error: ...` 并返回 `1`。

- [ ] **Step 4: 运行正常汇总测试并确认通过**

Run: `.venv/bin/python -m pytest tests/test_generate_libero_motion_primitive_distribution_script.py::test_cli_writes_global_distribution_without_crossing_episode_boundaries -q`

Expected: PASS。

- [ ] **Step 5: 写出短轨迹、参数传递和错误数据集测试**

```python
def test_cli_writes_header_only_when_no_labels_are_generated(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, [np.zeros((3, 8), dtype=np.float32)])
    output = tmp_path / "output"
    result = _run_cli(dataset, output, "--horizon", "3")
    assert result.returncode == 0
    assert (output / OUTPUT_NAME).read_text() == "primitive,count,proportion\n"
    assert "Generated labels: 0" in result.stdout


def test_cli_passes_tail_strategy_and_threshold_to_classifier(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, [_axis_states(3, axis=0, step=0.02)])
    output = tmp_path / "output"
    result = _run_cli(
        dataset, output,
        "--horizon", "3", "--threshold", "0.03", "--tail-strategy", "clip",
    )
    assert result.returncode == 0
    assert _read_rows(output) == [
        {"primitive": "stop", "count": "2", "proportion": str(2 / 3)},
        {"primitive": "move forward", "count": "1", "proportion": str(1 / 3)},
    ]


def test_cli_rejects_dataset_without_state_and_leaves_no_csv(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, [np.zeros((4, 8), dtype=np.float32)], include_state=False)
    output = tmp_path / "output"
    result = _run_cli(dataset, output)
    assert result.returncode == 1
    assert "observation.state" in result.stderr
    assert not (output / OUTPUT_NAME).exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--horizon", "2"),
        ("--threshold", "-0.01"),
        ("--threshold", "nan"),
    ],
)
def test_cli_rejects_invalid_classifier_settings(tmp_path: Path, arguments: tuple[str, str]):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, [np.zeros((4, 8), dtype=np.float32)])
    output = tmp_path / "output"
    result = _run_cli(dataset, output, *arguments)
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert not (output / OUTPUT_NAME).exists()
```

- [ ] **Step 6: 运行新增脚本的全部测试并修正最小实现**

Run: `.venv/bin/python -m pytest tests/test_generate_libero_motion_primitive_distribution_script.py -q`

Expected: 全部 PASS，且无 warning。

- [ ] **Step 7: 提交脚本和测试**

```bash
git add scripts/generate_libero_motion_primitive_distribution.py tests/test_generate_libero_motion_primitive_distribution_script.py
git commit -m "feat: add LIBERO motion primitive distribution script"
```

### Task 2: 使用文档和全量验证

**Files:**
- Modify: `libero_motion_primitives/README.md`

**Interfaces:**
- Consumes: Task 1 的 CLI 参数和固定 CSV 文件名。
- Produces: 可复制运行命令、参数说明和 CSV 示例。

- [ ] **Step 1: 更新 README 的运行说明**

在“快速使用”后增加“完整 LeRobot 数据集汇总”章节，包含：

```bash
.venv/bin/python scripts/generate_libero_motion_primitive_distribution.py \
  --dataset-root /data/dwb/datasets/LIBERO_lerobot/libero10_5 \
  --output-dir outputs/libero_motion_primitives \
  --horizon 4 \
  --threshold 0.03 \
  --tail-strategy truncate
```

并说明输出文件为 `outputs/libero_motion_primitives/motion_primitive_distribution.csv`，
字段为 `primitive,count,proportion`，统计为全数据集汇总但 episode 独立分类。

- [ ] **Step 2: 运行脚本测试和原有运动原语回归测试**

Run: `.venv/bin/python -m pytest tests/test_generate_libero_motion_primitive_distribution_script.py libero_motion_primitives/tests/test_motion_primitives.py -q`

Expected: 全部 PASS。

- [ ] **Step 3: 在真实 LIBERO-10 数据集上运行脚本**

Run: `.venv/bin/python scripts/generate_libero_motion_primitive_distribution.py --dataset-root /data/dwb/datasets/LIBERO_lerobot/libero10_5 --output-dir /tmp/libero-motion-primitive-verification`

Expected: 返回 `0`，打印 `Processed episodes: 50`、正数标签总数和 CSV 绝对路径；CSV 数量之和等于打印的标签总数，占比之和在浮点误差内等于 `1.0`。

- [ ] **Step 4: 检查差异并提交文档**

```bash
git diff --check
git add libero_motion_primitives/README.md docs/superpowers/plans/2026-08-08-libero-motion-primitive-distribution-script.md
git commit -m "docs: explain LIBERO primitive distribution command"
```
