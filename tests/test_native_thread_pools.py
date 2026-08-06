from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (
        *NATIVE_THREAD_VARIABLES,
        "TRAJECTORY_DATA_NUM_THREADS",
        "TDUS_NUM_THREADS",
    ):
        environment.pop(variable, None)
    return environment


def _import_probe(package: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    code = f"""
import os
import sys

variables = {NATIVE_THREAD_VARIABLES!r}

class NumpyImportProbe:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy":
            print("before_numpy=" + "|".join(os.environ[name] for name in variables))
            sys.meta_path.remove(self)
        return None

sys.meta_path.insert(0, NumpyImportProbe())
__import__({package!r})
print("after_import=" + "|".join(os.environ[name] for name in variables))
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "package",
    ["trajectory_data", "relcore", "sqcn", "quality_filter"],
)
def test_command_packages_limit_native_threads_before_numpy_import(package: str):
    environment = _clean_environment()

    result = _import_probe(package, environment)

    assert result.returncode == 0, result.stderr
    assert "before_numpy=1|1|1|1" in result.stdout
    assert "after_import=1|1|1|1" in result.stdout


def test_shared_limit_overrides_dangerous_inherited_native_thread_values():
    environment = _clean_environment()
    environment.update({variable: "256" for variable in NATIVE_THREAD_VARIABLES})

    result = _import_probe("trajectory_data", environment)

    assert result.returncode == 0, result.stderr
    assert "before_numpy=1|1|1|1" in result.stdout
    assert "after_import=1|1|1|1" in result.stdout


def test_shared_thread_override_applies_before_numpy_import():
    environment = _clean_environment()
    environment["TRAJECTORY_DATA_NUM_THREADS"] = "4"

    result = _import_probe("relcore", environment)

    assert result.returncode == 0, result.stderr
    assert "before_numpy=4|4|4|4" in result.stdout
    assert "after_import=4|4|4|4" in result.stdout


@pytest.mark.parametrize("configured", ["invalid", "0", "65"])
def test_shared_thread_override_rejects_invalid_values(configured: str):
    environment = _clean_environment()
    environment["TRAJECTORY_DATA_NUM_THREADS"] = configured

    result = _import_probe("trajectory_data", environment)

    assert result.returncode != 0
    assert "TRAJECTORY_DATA_NUM_THREADS must be an integer in [1, 64]" in result.stderr


def test_tdus_legacy_thread_override_remains_supported():
    environment = _clean_environment()
    environment["TDUS_NUM_THREADS"] = "3"

    result = _import_probe("tdus", environment)

    assert result.returncode == 0, result.stderr
    assert "before_numpy=3|3|3|3" in result.stdout
    assert "after_import=3|3|3|3" in result.stdout


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("invalid", "1"), ("0", "1"), ("65", "64")],
)
def test_tdus_legacy_thread_override_preserves_fallback_and_clamping(
    configured: str,
    expected: str,
):
    environment = _clean_environment()
    environment["TDUS_NUM_THREADS"] = configured

    result = _import_probe("tdus", environment)

    assert result.returncode == 0, result.stderr
    expected_values = "|".join([expected] * len(NATIVE_THREAD_VARIABLES))
    assert f"before_numpy={expected_values}" in result.stdout
    assert f"after_import={expected_values}" in result.stdout


def test_shared_thread_override_takes_precedence_over_tdus_legacy_value():
    environment = _clean_environment()
    environment["TRAJECTORY_DATA_NUM_THREADS"] = "5"
    environment["TDUS_NUM_THREADS"] = "3"

    result = _import_probe("tdus", environment)

    assert result.returncode == 0, result.stderr
    assert "before_numpy=5|5|5|5" in result.stdout
    assert "after_import=5|5|5|5" in result.stdout


def test_spawned_worker_inherits_shared_native_thread_limit(tmp_path: Path):
    script = tmp_path / "spawn_thread_probe.py"
    script.write_text(
        f"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

VARIABLES = {NATIVE_THREAD_VARIABLES!r}

def worker_environment():
    return tuple(os.environ[name] for name in VARIABLES)

if __name__ == "__main__":
    import trajectory_data

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        print("|".join(pool.submit(worker_environment).result()))
""",
        encoding="utf-8",
    )
    environment = _clean_environment()
    environment["TRAJECTORY_DATA_NUM_THREADS"] = "4"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else str(ROOT) + os.pathsep + existing_pythonpath
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4|4|4|4"
