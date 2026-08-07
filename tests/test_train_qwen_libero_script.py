import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBERO_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "train_libero_qwen3_vl_4b_groot_all_tasks_4x4090.sh"
)
BRIDGE_SCRIPT = PROJECT_ROOT / "scripts" / "train_bridge_4x4090.sh"
LIBERO_CONFIG = PROJECT_ROOT / "configs" / "qwen3_vl_4b_groot_libero_4x4090.yaml"
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


def test_libero_script_uses_only_libero_config_and_injects_one_to_one_defaults(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(LIBERO_SCRIPT),
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
    assert arguments[arguments.index("--config") + 1] == str(LIBERO_CONFIG)
    assert str(BRIDGE_CONFIG) not in arguments
    assert "--all-tasks" in arguments
    weight_index = arguments.index("--sample-weights")
    assert arguments[weight_index + 1 : weight_index + 3] == ["1", "1"]
    assert arguments[arguments.index("--prior-prefiltered-scores") + 1] == SQCN_SCORES
    assert arguments[arguments.index("--lora-learning-rate") + 1] == "5e-6"
    assert arguments[arguments.index("--action-head-learning-rate") + 1] == "2e-4"


def test_libero_target_only_does_not_inject_prior_or_sample_weights(tmp_path):
    environment, calls = _fake_python_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(LIBERO_SCRIPT),
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
