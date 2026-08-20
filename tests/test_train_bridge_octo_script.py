from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "train_bridge_octo_small_4x4090.sh"


def _fake_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
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
    return environment, calls


def test_bridge_octo_script_uses_single_process_for_preflight(tmp_path: Path) -> None:
    environment, calls = _fake_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--output-dir",
            "outputs/preflight",
            "--dataset-path",
            "/data/bridge",
            "--learning-rate",
            "2e-4",
            "--warmup-steps",
            "400",
            "--preflight-only",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["-m", "octo_small_bridge.cli"]
    assert arguments[arguments.index("--config") + 1] == str(
        PROJECT_ROOT / "configs" / "octo_small_bridge_v2_4x4090.yaml"
    )
    assert arguments[arguments.index("--output-dir") + 1] == "outputs/preflight"
    assert arguments[arguments.index("--learning-rate") + 1] == "2e-4"
    assert arguments[arguments.index("--warmup-steps") + 1] == "400"
    assert "--preflight-only" in arguments


def test_bridge_octo_script_uses_four_torchrun_processes(tmp_path: Path) -> None:
    environment, calls = _fake_environment(tmp_path)

    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--output-dir",
            "outputs/train",
            "--max-steps",
            "7",
            "--learning-rate",
            "1e-4",
            "--warmup-steps",
            "0",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )

    arguments = calls.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["--standalone", "--nproc_per_node=4", "-m"]
    assert arguments[3] == "octo_small_bridge.cli"
    assert arguments[arguments.index("--output-dir") + 1] == "outputs/train"
    assert arguments[arguments.index("--max-steps") + 1] == "7"
    assert arguments[arguments.index("--learning-rate") + 1] == "1e-4"
    assert arguments[arguments.index("--warmup-steps") + 1] == "0"
