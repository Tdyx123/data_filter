"""Stable fingerprints and atomic stage publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: str | Path) -> str:
    """Hash every regular file below a local model directory."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"local model directory contains no files: {root}")
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cache_is_valid(destination: Path, fingerprint: str, required: Sequence[str]) -> bool:
    manifest = destination / "manifest.json"
    if not manifest.is_file() or any(not (destination / item).is_file() for item in required):
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "complete" or payload.get("fingerprint") != fingerprint:
        return False
    return True


def publish_stage(
    destination: Path,
    *,
    fingerprint: str,
    required: Sequence[str],
    force: bool,
    resume: bool,
    build: Callable[[Path], None],
) -> bool:
    """Build and atomically publish a stage; return true when newly built."""

    if resume and cache_is_valid(destination, fingerprint, required):
        return False
    if destination.exists() and not force:
        raise FileExistsError(f"relcore stage cache is incompatible: {destination}; pass --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.relcore-", dir=destination.parent)
    )
    try:
        build(temporary)
        if any(not (temporary / item).is_file() for item in required):
            raise RuntimeError(f"stage did not create all required artifacts: {required}")
        backup: Path | None = None
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.relcore-backup-",
                    dir=destination.parent,
                )
            )
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return True
