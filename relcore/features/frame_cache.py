"""Persistent, content-validated per-episode visual feature cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trajectory_data import EpisodeRecord


def _file_sha256(path: Path) -> str:
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class FrameFeatureCache:
    """Store one atomically completed float32 feature array per episode."""

    def __init__(self, root: str | Path, *, fingerprint: str):
        self.root = Path(root)
        self.fingerprint = str(fingerprint)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid frame cache manifest: {manifest}") from error
            if payload.get("fingerprint") != self.fingerprint:
                raise ValueError(f"frame cache fingerprint mismatch: {manifest}")
        else:
            _atomic_json(
                manifest,
                {"status": "active", "fingerprint": self.fingerprint},
            )

    @staticmethod
    def _stem(record: EpisodeRecord) -> str:
        return f"ep{record.episode_id:06d}"

    def feature_path(self, record: EpisodeRecord) -> Path:
        return self.root / f"{self._stem(record)}.npy"

    def metadata_path(self, record: EpisodeRecord) -> Path:
        return self.root / f"{self._stem(record)}.json"

    def store(self, record: EpisodeRecord, features: np.ndarray) -> None:
        values = np.asarray(features, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[0] != record.length
            or values.shape[1] <= 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"episode {record.episode_id}: frame features must be finite [frames, dim]"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._stem(record)}.",
            suffix=".npy.tmp",
            dir=self.root,
        )
        temporary = Path(temporary_name)
        target = self.feature_path(record)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _atomic_json(
                self.metadata_path(record),
                {
                    "status": "complete",
                    "fingerprint": self.fingerprint,
                    "episode_id": record.episode_id,
                    "frames": record.length,
                    "feature_dim": int(values.shape[1]),
                    "dtype": "float32",
                    "sha256": _file_sha256(target),
                },
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def is_valid(self, record: EpisodeRecord, *, output_dim: int) -> bool:
        feature_path = self.feature_path(record)
        metadata_path = self.metadata_path(record)
        if not feature_path.is_file() or not metadata_path.is_file():
            return False
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload != {
                "status": "complete",
                "fingerprint": self.fingerprint,
                "episode_id": record.episode_id,
                "frames": record.length,
                "feature_dim": int(output_dim),
                "dtype": "float32",
                "sha256": payload.get("sha256"),
            }:
                return False
            values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (record.length, int(output_dim)) or values.dtype != np.float32:
                return False
            if not np.all(np.isfinite(values)):
                return False
            return _file_sha256(feature_path) == payload["sha256"]
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return False

    def valid_prefix(self, records: Sequence[EpisodeRecord], *, output_dim: int) -> int:
        for index, record in enumerate(records):
            if not self.is_valid(record, output_dim=output_dim):
                return index
        return len(records)

    def load(self, record: EpisodeRecord, *, output_dim: int) -> np.ndarray:
        if not self.is_valid(record, output_dim=output_dim):
            raise ValueError(f"episode {record.episode_id}: invalid frame feature cache entry")
        return np.asarray(np.load(self.feature_path(record), allow_pickle=False), dtype=np.float32)

    def publish_features(
        self,
        records: Sequence[EpisodeRecord],
        destination: str | Path,
        *,
        output_dim: int,
    ) -> list[str]:
        target_root = Path(destination)
        target_root.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for record in records:
            if not self.is_valid(record, output_dim=output_dim):
                raise ValueError(f"episode {record.episode_id}: cannot publish invalid frame cache")
            source = self.feature_path(record)
            target = target_root / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            names.append(source.name)
        return names
