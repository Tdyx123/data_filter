"""Exercise the shell entrypoint with real spawn workers, without loading a dataset."""

import os
from pathlib import Path
import shlex
import subprocess
import sys


def test_manifest_preview_precedes_spawn_safe_run(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    launcher = tmp_path / "python"
    launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    launcher.chmod(0o755)
    # Replace only the expensive pipeline; retain the real CLI and module entrypoint.
    (tmp_path / "sitecustomize.py").write_text('''
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from cocore_ablation import cli

def synthetic_pipeline(config, *, output_dir, subfolder_name, force):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert pool.submit(abs, -7).result(timeout=10) == 7
    print("spawn_worker=ok", flush=True)
    return Path(output_dir) / subfolder_name / "select"

cli.run_pipeline = synthetic_pipeline
''')
    env = dict(os.environ)
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(repo)])
    output = tmp_path / "output with spaces"
    result = subprocess.run(
        ["bash", str(repo / "run_ablation.sh"), "no-relation",
         "--output-dir", str(output), "--relation-weight", "0"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"selected_manifest_path={output}/no-relation/select/selected_manifest.jsonl"
    assert lines[1] == "spawn_worker=ok"
