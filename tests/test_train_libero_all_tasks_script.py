import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_tasks_script_injects_selection_and_forwards_preflight_arguments(
    tmp_path,
):
    calls = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$OCTO_TEST_CALLS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$OCTO_TEST_CALLS\"\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["OCTO_TEST_CALLS"] = str(calls)
    script = (
        PROJECT_ROOT
        / "scripts"
        / "train_libero_octo_small_all_tasks_4x4090.sh"
    )

    subprocess.run(
        [
            "bash",
            str(script),
            "--preflight-only",
            "--max-steps",
            "7",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["-m", "octo_small_libero.cli"]
    assert "--all-tasks" in arguments
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["3", "1"]
    assert "--task-index" not in arguments
    assert "--preflight-only" in arguments
    assert arguments[arguments.index("--max-steps") + 1] == "7"

    subprocess.run(
        [
            "bash",
            str(script),
            "--max-steps",
            "9",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    training_arguments = calls.read_text(encoding="utf-8").splitlines()
    assert "--all-tasks" in training_arguments
    weight_index = training_arguments.index("--sample-weights")
    assert training_arguments[weight_index + 1 : weight_index + 3] == ["3", "1"]
    assert training_arguments[training_arguments.index("--max-steps") + 1] == "9"


def test_single_task_and_standalone_single_gpu_keep_default_sample_weights():
    for name in (
        "train_libero_octo_small_4x4090.sh",
        "train_libero_octo_small_all_tasks_1x4090.sh",
    ):
        source = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "--sample-weights" not in source
