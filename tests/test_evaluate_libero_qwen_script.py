import os
import subprocess
from pathlib import Path

import pytest

from octo_small_libero.libero10_tasks import LIBERO_10_TASK_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_libero_qwen.sh"
CHECKPOINT = "/models/qwen-run/checkpoints/step-00020000"


def _fake_python(tmp_path: Path) -> tuple[dict[str, str], Path]:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir(parents=True)
    executable = executable_dir / "python3"
    executable.write_text(
        """#!/usr/bin/env bash
set -u
{
  for argument in "$@"; do
    printf '%s\\037' "${argument}"
  done
  printf '\\n'
} >> "${QWEN_EVAL_TEST_CALLS:?}"

for argument in "$@"; do
  if [[ -n "${QWEN_EVAL_TEST_FAIL_OUTPUT:-}" ]] \\
    && [[ "${argument}" == "${QWEN_EVAL_TEST_FAIL_OUTPUT}" ]]; then
    exit "${QWEN_EVAL_TEST_FAIL_STATUS:-23}"
  fi
done
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    calls_path = tmp_path / "calls"
    environment = os.environ.copy()
    environment["PATH"] = f"{executable_dir}:{environment['PATH']}"
    environment["QWEN_EVAL_TEST_CALLS"] = str(calls_path)
    return environment, calls_path


def _run_script(
    tmp_path: Path,
    *arguments: str,
    fail_output: str | None = None,
    fail_status: int = 23,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    environment, calls_path = _fake_python(tmp_path)
    if fail_output is not None:
        environment["QWEN_EVAL_TEST_FAIL_OUTPUT"] = fail_output
        environment["QWEN_EVAL_TEST_FAIL_STATUS"] = str(fail_status)
    completed = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = []
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            calls.append(line.split("\x1f")[:-1])
    return completed, calls


def _option(call: list[str], option: str) -> str:
    position = call.index(option)
    return call[position + 1]


def test_checkpoint_is_required_before_python_starts(tmp_path):
    completed, calls = _run_script(tmp_path, "--indexes", "5")

    assert completed.returncode == 2
    assert calls == []
    assert "--checkpoint is required" in completed.stderr


def test_default_runs_all_tasks_with_explicit_checkpoint(tmp_path):
    completed, calls = _run_script(tmp_path, "--checkpoint", CHECKPOINT)

    assert completed.returncode == 0
    assert len(calls) == 10
    assert all(call[:2] == ["-m", "qwen3_vl_groot.evaluate"] for call in calls)
    assert [_option(call, "--task-name") for call in calls] == list(LIBERO_10_TASK_NAMES)
    assert [_option(call, "--output-dir") for call in calls] == [
        f"outputs/qwen_libero_eval/task-{index}" for index in range(10)
    ]
    assert all(_option(call, "--checkpoint") == CHECKPOINT for call in calls)


def test_selected_indexes_forward_qwen_options(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--indexes=0,5,9",
        f"--checkpoint={CHECKPOINT}",
        "--model-path",
        "/models/Qwen3-VL-4B-Instruct",
        "--policy-batch-size",
        "2",
        "--smoke-test",
    )

    assert completed.returncode == 0
    assert [_option(call, "--task-name") for call in calls] == [
        LIBERO_10_TASK_NAMES[0],
        LIBERO_10_TASK_NAMES[5],
        LIBERO_10_TASK_NAMES[9],
    ]
    assert all(_option(call, "--model-path") == "/models/Qwen3-VL-4B-Instruct" for call in calls)
    assert all(_option(call, "--policy-batch-size") == "2" for call in calls)
    assert all("--smoke-test" in call for call in calls)


def test_explicit_task_preserves_single_output_semantics(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--checkpoint",
        CHECKPOINT,
        "--task-name",
        "custom_task",
        "--output-dir",
        "/tmp/qwen-one-task",
    )

    assert completed.returncode == 0
    assert len(calls) == 1
    assert _option(calls[0], "--task-name") == "custom_task"
    assert _option(calls[0], "--output-dir") == "/tmp/qwen-one-task"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--checkpoint", CHECKPOINT, "--checkpoint", CHECKPOINT),
        ("--checkpoint", CHECKPOINT, "--indexes", "10"),
        ("--checkpoint", CHECKPOINT, "--indexes", "0,0"),
        ("--checkpoint", CHECKPOINT, "--indexes", "0,1", "--task-name", "custom"),
    ],
)
def test_invalid_launcher_arguments_fail_before_python(tmp_path, arguments):
    completed, calls = _run_script(tmp_path, *arguments)

    assert completed.returncode == 2
    assert calls == []
    assert "error:" in completed.stderr


def test_normal_failure_continues_but_simulation_failure_aborts(tmp_path):
    failing_output = "outputs/qwen_libero_eval/task-5"
    completed, calls = _run_script(
        tmp_path,
        "--checkpoint",
        CHECKPOINT,
        "--indexes",
        "0,5,9",
        fail_output=failing_output,
    )
    assert completed.returncode == 1
    assert [_option(call, "--task-name") for call in calls] == [
        LIBERO_10_TASK_NAMES[0],
        LIBERO_10_TASK_NAMES[5],
        LIBERO_10_TASK_NAMES[9],
    ]
    assert "5(exit=23)" in completed.stderr

    aborted, aborted_calls = _run_script(
        tmp_path / "abort",
        "--checkpoint",
        CHECKPOINT,
        "--indexes",
        "0,5,9",
        fail_output=failing_output,
        fail_status=3,
    )
    assert aborted.returncode == 3
    assert [_option(call, "--task-name") for call in aborted_calls] == [
        LIBERO_10_TASK_NAMES[0],
        LIBERO_10_TASK_NAMES[5],
    ]

    contract_error, contract_calls = _run_script(
        tmp_path / "contract",
        "--checkpoint",
        CHECKPOINT,
        "--indexes",
        "0,5,9",
        fail_output=failing_output,
        fail_status=2,
    )
    assert contract_error.returncode == 2
    assert [_option(call, "--task-name") for call in contract_calls] == [
        LIBERO_10_TASK_NAMES[0],
        LIBERO_10_TASK_NAMES[5],
    ]
