import json
import os
import subprocess
from pathlib import Path

import pytest

from octo_small_libero.libero10_tasks import LIBERO_10_TASK_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_libero_octo_small.sh"
TASKS = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    (
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_"
        "yellow_and_white_mug_on_the_right_plate"
    ),
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    (
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_"
        "chocolate_pudding_to_the_right_of_the_plate"
    ),
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)


def test_evaluator_shell_mapping_matches_shared_training_order():
    assert TASKS == LIBERO_10_TASK_NAMES


def _fake_python(tmp_path: Path) -> tuple[dict[str, str], Path]:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "python3"
    executable.write_text(
        """#!/usr/bin/env bash
set -u
{
  for argument in "$@"; do
    printf '%s\\037' "${argument}"
  done
  printf '\\n'
} >> "${OCTO_TEST_CALLS:?}"

for argument in "$@"; do
  if [[ -n "${OCTO_TEST_FAIL_OUTPUT:-}" ]] \
    && [[ "${argument}" == "${OCTO_TEST_FAIL_OUTPUT}" ]]; then
    exit "${OCTO_TEST_FAIL_STATUS:-23}"
  fi
done
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    calls_path = tmp_path / "calls"
    environment = os.environ.copy()
    environment["PATH"] = f"{executable_dir}:{environment['PATH']}"
    environment["OCTO_TEST_CALLS"] = str(calls_path)
    return environment, calls_path


def _run_script(
    tmp_path: Path,
    *arguments: str,
    fail_output: str | None = None,
    fail_status: int = 23,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    environment, calls_path = _fake_python(tmp_path)
    if fail_output is not None:
        environment["OCTO_TEST_FAIL_OUTPUT"] = fail_output
        environment["OCTO_TEST_FAIL_STATUS"] = str(fail_status)
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


def test_default_runs_all_tasks_in_official_order(tmp_path):
    completed, calls = _run_script(tmp_path)

    assert completed.returncode == 0
    assert len(calls) == 10
    assert [_option(call, "--task-name") for call in calls] == list(TASKS)
    assert [_option(call, "--output-dir") for call in calls] == [
        f"outputs/octo_small_libero_eval/task-{index}" for index in range(10)
    ]
    assert all(call[:2] == ["-m", "octo_small_libero.evaluate"] for call in calls)


def test_selected_indexes_keep_order_and_forward_common_arguments(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--indexes=0,5,9",
        "--checkpoint",
        "/models/checkpoint",
        "--smoke-test",
    )

    assert completed.returncode == 0
    assert [_option(call, "--task-name") for call in calls] == [
        TASKS[0],
        TASKS[5],
        TASKS[9],
    ]
    assert [_option(call, "--output-dir") for call in calls] == [
        "outputs/octo_small_libero_eval/task-0",
        "outputs/octo_small_libero_eval/task-5",
        "outputs/octo_small_libero_eval/task-9",
    ]
    assert all("--smoke-test" in call for call in calls)
    assert all(_option(call, "--checkpoint") == "/models/checkpoint" for call in calls)


def test_two_smoke_tasks_run_in_separate_processes_and_both_complete(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--indexes",
        "0,1",
        "--smoke-test",
    )

    assert completed.returncode == 0
    assert [_option(call, "--task-name") for call in calls] == [TASKS[0], TASKS[1]]
    assert all("--smoke-test" in call for call in calls)
    assert "task 0 completed successfully" in completed.stdout
    assert "evaluating LIBERO-10 task 1" in completed.stdout
    assert "task 1 completed successfully" in completed.stdout


def test_indexes_all_matches_default_and_honors_custom_output_root(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--indexes",
        "all",
        "--output-dir",
        "/tmp/libero-results",
    )

    assert completed.returncode == 0
    assert [_option(call, "--task-name") for call in calls] == list(TASKS)
    assert [_option(call, "--output-dir") for call in calls] == [
        f"/tmp/libero-results/task-{index}" for index in range(10)
    ]


def test_explicit_task_name_preserves_single_task_output_semantics(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--task-name",
        "custom_task",
        "--output-dir",
        "/tmp/one-task",
        "--smoke-test",
    )

    assert completed.returncode == 0
    assert len(calls) == 1
    assert _option(calls[0], "--task-name") == "custom_task"
    assert _option(calls[0], "--output-dir") == "/tmp/one-task"
    assert "--smoke-test" in calls[0]


@pytest.mark.parametrize(
    "arguments",
    [
        ("--indexes", "10"),
        ("--indexes", "0,0"),
        ("--indexes", ""),
        ("--indexes", "0,,1"),
        ("--indexes", "zero"),
        ("--indexes", "0,1", "--task-name", "custom_task"),
    ],
)
def test_invalid_indexes_fail_before_python_is_started(tmp_path, arguments):
    completed, calls = _run_script(tmp_path, *arguments)

    assert completed.returncode != 0
    assert calls == []
    assert "error:" in completed.stderr


def test_failed_task_does_not_prevent_later_tasks(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--indexes",
        "0,5,9",
        fail_output="outputs/octo_small_libero_eval/task-5",
    )

    assert completed.returncode != 0
    assert [_option(call, "--task-name") for call in calls] == [
        TASKS[0],
        TASKS[5],
        TASKS[9],
    ]
    assert "5(exit=23)" in completed.stderr


def test_simulation_infrastructure_failure_aborts_remaining_tasks(tmp_path):
    completed, calls = _run_script(
        tmp_path,
        "--indexes",
        "0,5,9",
        fail_output="outputs/octo_small_libero_eval/task-5",
        fail_status=3,
    )

    assert completed.returncode == 3
    assert [_option(call, "--task-name") for call in calls] == [
        TASKS[0],
        TASKS[5],
    ]
    assert "simulation infrastructure failure; aborting the batch" in completed.stderr


def test_help_lists_every_task_without_starting_python(tmp_path):
    completed, calls = _run_script(tmp_path, "--help")

    assert completed.returncode == 0
    assert calls == []
    assert "--indexes LIST" in completed.stdout
    for seed in (3471197683, 1232873419, 1448008435):
        assert str(seed) in completed.stdout
    for index, task in enumerate(TASKS):
        assert f"{index}  {task}" in completed.stdout


@pytest.mark.gpu
@pytest.mark.slow
def test_real_two_task_smoke_batch_exits_without_worker_processes(tmp_path):
    checkpoint = os.environ.get("OCTO_LIBERO_EVAL_CHECKPOINT")
    statistics = os.environ.get("OCTO_LIBERO_EVAL_STATISTICS")
    if not checkpoint or not statistics:
        pytest.skip(
            "set OCTO_LIBERO_EVAL_CHECKPOINT and OCTO_LIBERO_EVAL_STATISTICS "
            "to run the real LIBERO two-task smoke regression"
        )

    output_root = tmp_path / "real-two-task-smoke"
    command = [
        "bash",
        str(SCRIPT),
        "--indexes",
        "0,1",
        "--smoke-test",
        "--checkpoint",
        checkpoint,
        "--statistics",
        statistics,
        "--output-dir",
        str(output_root),
    ]
    libero_root = os.environ.get("LIBERO_ROOT")
    if libero_root:
        command.extend(["--libero-root", libero_root])
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, 9)
        stdout, stderr = process.communicate()
        pytest.fail(f"two-task smoke batch timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")

    assert process.returncode == 0, stderr
    assert "task 0 completed successfully" in stdout
    assert "evaluating LIBERO-10 task 1" in stdout
    assert "task 1 completed successfully" in stdout
    for index in (0, 1):
        report = json.loads(
            (output_root / f"task-{index}" / "results.json").read_text(encoding="utf-8")
        )
        assert report["status"] == "complete"
        assert report["runtime"]["libero_multiprocessing_start_method"] == "spawn"

    process_table = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,cmd="],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    remaining_group = [
        line for line in process_table if line.split(maxsplit=2)[1] == str(process.pid)
    ]
    assert remaining_group == []
