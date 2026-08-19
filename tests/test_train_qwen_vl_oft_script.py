import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "train_qwenvl_oft_4x4090.sh"
CONFIG = PROJECT_ROOT / "configs" / "qwenvl_oft_bridge_4x4090.yaml"


def test_oft_launcher_uses_independent_module_config_and_forwards_overrides(tmp_path):
    calls = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$OFT_TEST_CALLS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["OFT_TEST_CALLS"] = str(calls)

    subprocess.run(
        [
            "bash",
            str(SCRIPT),
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
    assert arguments[:3] == ["-m", "qwen_vl_oft.cli", "launch"]
    assert arguments[arguments.index("--config") + 1] == str(CONFIG)
    assert arguments[arguments.index("--lora-learning-rate") + 1] == "7e-6"
    assert arguments[arguments.index("--action-head-learning-rate") + 1] == "3e-4"


def test_oft_cli_exposes_launch_train_and_warm_start_contract():
    from qwen_vl_oft.cli import build_parser

    parser = build_parser()
    launch = parser.parse_args(
        [
            "launch",
            "--config",
            str(CONFIG),
            "--gpu-ids",
            "2,3,6,7",
            "--smoke-test",
            "--warm-start-checkpoint",
            "/tmp/step-00000007",
        ]
    )
    train = parser.parse_args(
        ["train", "--config", str(CONFIG), "--warm-start-checkpoint", "/tmp/step-00000007"]
    )

    assert launch.gpu_ids == [2, 3, 6, 7]
    assert launch.smoke_test is True
    assert launch.warm_start_checkpoint == "/tmp/step-00000007"
    assert train.warm_start_checkpoint == "/tmp/step-00000007"
