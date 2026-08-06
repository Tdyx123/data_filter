import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFILTERED_SCORES = "/data/dwb/libero_filter/quality/filter/top10pct/scores.csv"


def _assert_prefiltered_defaults(arguments):
    scores_index = arguments.index("--prior-prefiltered-scores")
    assert arguments[scores_index + 1] == PREFILTERED_SCORES


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
            "--output-dir",
            "outputs/preflight",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["-m", "octo_small_libero.cli"]
    assert "--all-tasks" in arguments
    _assert_prefiltered_defaults(arguments)
    output_indexes = [
        index for index, value in enumerate(arguments) if value == "--output-dir"
    ]
    assert output_indexes == [arguments.index("--output-dir")]
    assert arguments[output_indexes[0] + 1] == "outputs/preflight"
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
            "--output-dir",
            "outputs/training",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    training_arguments = calls.read_text(encoding="utf-8").splitlines()
    assert "--all-tasks" in training_arguments
    _assert_prefiltered_defaults(training_arguments)
    output_indexes = [
        index
        for index, value in enumerate(training_arguments)
        if value == "--output-dir"
    ]
    assert len(output_indexes) == 1
    assert training_arguments[output_indexes[0] + 1] == "outputs/training"
    weight_index = training_arguments.index("--sample-weights")
    assert training_arguments[weight_index + 1 : weight_index + 3] == ["3", "1"]
    assert training_arguments[training_arguments.index("--max-steps") + 1] == "9"


def test_single_task_script_keeps_default_sample_weights():
    source = (
        PROJECT_ROOT / "scripts" / "train_libero_octo_small_4x4090.sh"
    ).read_text(encoding="utf-8")
    assert "--sample-weights" not in source


def test_all_tasks_script_target_only_omits_prior_defaults_and_uses_isolated_output(
    tmp_path,
):
    calls = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for executable in ("python3", "torchrun"):
        fake = fake_bin / executable
        fake.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$OCTO_TEST_CALLS\"\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["OCTO_TEST_CALLS"] = str(calls)
    script = PROJECT_ROOT / "scripts" / "train_libero_octo_small_all_tasks_4x4090.sh"

    subprocess.run(
        [
            "bash",
            str(script),
            "--target-only",
            "--output-dir",
            "outputs/target-only",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert "--target-only" in arguments
    assert "--sample-weights" not in arguments
    assert "--prior-prefiltered-scores" not in arguments
    output_index = arguments.index("--output-dir")
    assert arguments[output_index + 1] == "outputs/target-only"

    custom_output = "outputs/custom-target-only"
    subprocess.run(
        [
            "bash",
            str(script),
            "--target-only",
            "--output-dir",
            custom_output,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    training_arguments = calls.read_text(encoding="utf-8").splitlines()
    assert "--target-only" in training_arguments
    assert "--sample-weights" not in training_arguments
    assert "--prior-prefiltered-scores" not in training_arguments
    output_indexes = [
        index for index, value in enumerate(training_arguments) if value == "--output-dir"
    ]
    assert training_arguments[output_indexes[-1] + 1] == custom_output


def test_all_tasks_script_explicit_prefiltered_scores_override_default(tmp_path):
    calls = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$OCTO_TEST_CALLS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["OCTO_TEST_CALLS"] = str(calls)
    script = PROJECT_ROOT / "scripts" / "train_libero_octo_small_all_tasks_4x4090.sh"
    scores = "/data/custom/selected.csv"

    subprocess.run(
        [
            "bash",
            str(script),
            "--prior-prefiltered-scores",
            scores,
            "--output-dir",
            "outputs/quality-filter",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    scores_index = arguments.index("--prior-prefiltered-scores")
    assert arguments[scores_index + 1] == scores
    assert PREFILTERED_SCORES not in arguments
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["3", "1"]

    subprocess.run(
        [
            "bash",
            str(script),
            f"--prior-prefiltered-scores={scores}",
            "--output-dir",
            "outputs/custom-equals",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    equals_arguments = calls.read_text(encoding="utf-8").splitlines()
    assert f"--prior-prefiltered-scores={scores}" in equals_arguments
    assert PREFILTERED_SCORES not in equals_arguments
