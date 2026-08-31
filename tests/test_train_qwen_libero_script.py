import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh"
)
LIBERO_8X_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_libero_qwen3_vl_4b_groot_all_tasks_8x4090.sh"
)
CYCLIC_LIBERO_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_libero_qwen3_vl_4b_groot_cyclic_lora_all_tasks_4x4090.sh"
)
QWEN35_LIBERO_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_libero_qwen3_5_0_8b_groot_all_tasks_4x4090.sh"
)
BRIDGE_SCRIPT = PROJECT_ROOT / "scripts" / "train_bridge_4x4090.sh"
CYCLIC_BRIDGE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_bridge_qwen3_vl_4b_groot_cyclic_lora_4x4090.sh"
)
LIBERO_CONFIG = PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
LIBERO_8X_CONFIG = (
    PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_8x4090.yaml"
)
QWEN35_LIBERO_CONFIG = (
    PROJECT_ROOT / "configs" / "qwen3_5_0_8b_groot_libero_4x4090.yaml"
)
BRIDGE_CONFIG = PROJECT_ROOT / "configs" / "bridge_4x4090.yaml"
SQCN_SCORES = "/data/dwb/libero90_sqcn/filter/top10pct/scores.csv"


def _fake_python_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    calls = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$QWEN_TEST_CALLS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["QWEN_TEST_CALLS"] = str(calls)
    return environment, calls


@pytest.mark.parametrize(
    ("script", "config"),
    [(LIBERO_SCRIPT, LIBERO_CONFIG), (LIBERO_8X_SCRIPT, LIBERO_8X_CONFIG)],
)
def test_libero_script_uses_only_libero_config_and_full_prior_by_default(
    tmp_path,
    script,
    config,
):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(script),
            "--output-dir",
            "outputs/libero",
            "--lora-learning-rate",
            "5e-6",
            "--action-head-learning-rate",
            "2e-4",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["-m", "qwen3_vl_groot.cli", "launch"]
    assert arguments[arguments.index("--config") + 1] == str(config)
    assert str(BRIDGE_CONFIG) not in arguments
    assert "--all-tasks" in arguments
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["1", "1"]
    assert "--prior-prefiltered-scores" not in arguments
    assert arguments[arguments.index("--lora-learning-rate") + 1] == "5e-6"
    assert arguments[arguments.index("--action-head-learning-rate") + 1] == "2e-4"


@pytest.mark.parametrize("script", [LIBERO_SCRIPT, LIBERO_8X_SCRIPT])
def test_libero_target_only_does_not_inject_prior_or_sample_weights(tmp_path, script):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(script),
            "--target-only",
            "--output-dir",
            "outputs/libero-target-only",
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


def test_bridge_script_keeps_bridge_config_and_forwards_learning_rates(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(BRIDGE_SCRIPT),
            "--lora-learning-rate",
            "7e-6",
            "--action-head-learning-rate",
            "3e-4",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--config") + 1] == str(BRIDGE_CONFIG)
    assert str(LIBERO_CONFIG) not in arguments
    assert "--all-tasks" not in arguments
    assert "--sample-weights" not in arguments
    assert "--prior-prefiltered-scores" not in arguments
    assert arguments[arguments.index("--lora-learning-rate") + 1] == "7e-6"
    assert arguments[arguments.index("--action-head-learning-rate") + 1] == "3e-4"


def test_cyclic_bridge_script_injects_schedule_and_bridge_defaults(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(CYCLIC_BRIDGE_SCRIPT),
            "--output-dir",
            "outputs/bridge-cyclic",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--config") + 1] == str(BRIDGE_CONFIG)
    assert str(LIBERO_CONFIG) not in arguments
    assert "--all-tasks" not in arguments
    assert "--sample-weights" not in arguments
    assert "--prior-prefiltered-scores" not in arguments
    assert arguments[arguments.index("--lora-freeze-steps") + 1] == "5000"
    assert arguments[arguments.index("--lora-cycle-steps") + 1] == "100"
    assert arguments[arguments.index("--lora-active-steps") + 1] == "10"
    assert arguments[arguments.index("--micro-batch-size") + 1] == "1"
    assert arguments[arguments.index("--gradient-accumulation-steps") + 1] == "16"
    assert "--qwen-context-forward" not in arguments
    assert "--no-compile-qwen-backbone" in arguments
    assert "--no-compile-action-head" not in arguments
    assert arguments[arguments.index("--episode-cache-size") + 1] == "2"


def test_cyclic_bridge_script_allows_explicit_schedule_overrides(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(CYCLIC_BRIDGE_SCRIPT),
            "--lora-freeze-steps",
            "6000",
            "--lora-cycle-steps",
            "200",
            "--lora-active-steps",
            "20",
            "--output-dir",
            "outputs/bridge-cyclic-custom",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()

    def values_for(option):
        return [arguments[index + 1] for index, value in enumerate(arguments) if value == option]

    assert values_for("--lora-freeze-steps") == ["5000", "6000"]
    assert values_for("--lora-cycle-steps") == ["100", "200"]
    assert values_for("--lora-active-steps") == ["10", "20"]


def test_libero_script_preserves_explicit_weights_and_prior_mode(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)
    relcore = "/data/relcore/selected_manifest.jsonl"

    subprocess.run(
        [
            "bash",
            str(LIBERO_SCRIPT),
            "--sample-weights",
            "3",
            "1",
            "--prior-relcore-manifest",
            relcore,
            "--output-dir",
            "outputs/libero-relcore",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments.count("--sample-weights") == 1
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["3", "1"]
    assert arguments[arguments.index("--prior-relcore-manifest") + 1] == relcore
    assert "--prior-prefiltered-scores" not in arguments


def test_qwen35_libero_script_uses_independent_config_and_full_prior_by_default(
    tmp_path,
):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(QWEN35_LIBERO_SCRIPT),
            "--output-dir",
            "outputs/qwen35-libero",
            "--lora-learning-rate",
            "5e-6",
            "--action-head-learning-rate",
            "2e-4",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["-m", "qwen3_vl_groot.cli", "launch"]
    assert arguments[arguments.index("--config") + 1] == str(QWEN35_LIBERO_CONFIG)
    assert str(LIBERO_CONFIG) not in arguments
    assert "--all-tasks" in arguments
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["1", "1"]
    assert "--prior-prefiltered-scores" not in arguments
    assert arguments[arguments.index("--lora-learning-rate") + 1] == "5e-6"
    assert arguments[arguments.index("--action-head-learning-rate") + 1] == "2e-4"
    assert arguments[-1] == "--preflight-only"


def test_qwen35_libero_target_only_skips_mixture_defaults(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(QWEN35_LIBERO_SCRIPT),
            "--target-only",
            "--output-dir",
            "outputs/qwen35-target-only",
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


def test_qwen35_libero_preserves_explicit_weights_and_prior(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)
    relcore = "/data/relcore/qwen35-selected.jsonl"

    subprocess.run(
        [
            "bash",
            str(QWEN35_LIBERO_SCRIPT),
            "--sample-weights",
            "3",
            "1",
            "--prior-relcore-manifest",
            relcore,
            "--output-dir",
            "outputs/qwen35-relcore",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments.count("--sample-weights") == 1
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["3", "1"]
    assert arguments[arguments.index("--prior-relcore-manifest") + 1] == relcore
    assert "--prior-prefiltered-scores" not in arguments


@pytest.mark.parametrize(
    "script",
    [LIBERO_SCRIPT, LIBERO_8X_SCRIPT, QWEN35_LIBERO_SCRIPT],
)
def test_qwen_libero_scripts_forward_explicit_prefiltered_scores_once(
    tmp_path,
    script,
):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(script),
            "--prior-prefiltered-scores",
            SQCN_SCORES,
            "--output-dir",
            "outputs/libero-prefiltered",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments.count("--prior-prefiltered-scores") == 1
    assert arguments[arguments.index("--prior-prefiltered-scores") + 1] == SQCN_SCORES


def test_cyclic_libero_script_uses_full_prior_and_preserves_schedule_defaults(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(CYCLIC_LIBERO_SCRIPT),
            "--output-dir",
            "outputs/libero-cyclic",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert "--all-tasks" in arguments
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["1", "1"]
    assert "--prior-prefiltered-scores" not in arguments
    assert arguments[arguments.index("--lora-freeze-steps") + 1] == "5000"
    assert arguments[arguments.index("--lora-cycle-steps") + 1] == "100"
    assert arguments[arguments.index("--lora-active-steps") + 1] == "10"
    assert arguments[arguments.index("--micro-batch-size") + 1] == "1"
    assert arguments[arguments.index("--gradient-accumulation-steps") + 1] == "16"
    assert "--qwen-context-forward" not in arguments
    assert "--no-compile-qwen-backbone" in arguments
    assert "--no-compile-action-head" not in arguments
    assert arguments[arguments.index("--episode-cache-size") + 1] == "2"


def test_cyclic_libero_script_allows_explicit_schedule_overrides(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(CYCLIC_LIBERO_SCRIPT),
            "--lora-freeze-steps",
            "6000",
            "--lora-cycle-steps",
            "200",
            "--lora-active-steps",
            "20",
            "--output-dir",
            "outputs/libero-cyclic-custom",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()

    def values_for(option):
        return [arguments[index + 1] for index, value in enumerate(arguments) if value == option]

    assert values_for("--lora-freeze-steps") == ["5000", "6000"]
    assert values_for("--lora-cycle-steps") == ["100", "200"]
    assert values_for("--lora-active-steps") == ["10", "20"]


@pytest.mark.parametrize("script", [CYCLIC_BRIDGE_SCRIPT, CYCLIC_LIBERO_SCRIPT])
def test_cyclic_qwen3_vl_4b_scripts_forward_explicit_action_head_disable_once(
    tmp_path,
    script,
):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(script),
            "--no-compile-action-head",
            "--output-dir",
            "outputs/cyclic-no-head-compile",
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments.count("--no-compile-action-head") == 1
