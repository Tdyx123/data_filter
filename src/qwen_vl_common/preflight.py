from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Any

from .backbone import inspect_qwen_config
from .data import BridgeMetadata, bridge_collate, validate_episode


class PreflightError(RuntimeError):
    """Raised when the target host cannot run the requested configuration."""


MAX_RESERVED_MEMORY_GIB = 22.0


def _build_memory_probe_batch(
    dataset: Any,
    *,
    micro_batch_size: int,
) -> dict[str, Any]:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        if len(dataset) < micro_batch_size:
            raise PreflightError(
                f"Memory probe needs {micro_batch_size} distinct samples, "
                f"but the dataset has only {len(dataset)}"
            )
        samples = [dataset[index] for index in range(micro_batch_size)]
    else:
        samples = list(islice(iter(dataset), micro_batch_size))
        if len(samples) != micro_batch_size:
            raise PreflightError(
                f"Memory probe requested {micro_batch_size} samples, got {len(samples)}"
            )
    identities = [
        (
            sample.get("dataset_name"),
            int(sample["episode_index"]),
            int(sample["frame_index"]),
        )
        for sample in samples
    ]
    if len(set(identities)) != micro_batch_size:
        raise PreflightError("Memory probe samples must be distinct")
    return bridge_collate(samples)


def _memory_probe_result(
    *,
    loss: float,
    peak_allocated: int,
    peak_reserved: int,
    total: int,
    micro_batch_size: int,
) -> dict[str, Any]:
    limit_bytes = int(MAX_RESERVED_MEMORY_GIB * 2**30)
    if peak_reserved > limit_bytes:
        raise PreflightError(
            "Memory probe reserved "
            f"{peak_reserved / 2**30:.3f} GiB, above the "
            f"{MAX_RESERVED_MEMORY_GIB:.1f} GiB candidate limit"
        )
    if peak_reserved >= total:
        raise PreflightError("Memory probe reached or exceeded physical GPU memory")
    return {
        "loss": loss,
        "micro_batch_size": micro_batch_size,
        "lora_active": True,
        "peak_allocated_gib": round(peak_allocated / 2**30, 3),
        "peak_reserved_gib": round(peak_reserved / 2**30, 3),
        "reserved_limit_gib": MAX_RESERVED_MEMORY_GIB,
        "reserved_headroom_gib": round((limit_bytes - peak_reserved) / 2**30, 3),
        "total_gib": round(total / 2**30, 3),
    }


def validate_paths_and_data(
    config: dict[str, Any],
    *,
    decode_samples: bool = True,
) -> dict[str, Any]:
    model_path = Path(config["paths"]["model"]).expanduser().resolve()
    family = str(config["model"].get("backbone_family", "qwen3_vl"))
    inspect_qwen_config(model_path, expected_family=family)
    dataset_path = Path(config["paths"]["dataset"]).expanduser().resolve()
    metadata = BridgeMetadata(dataset_path, config["data"])
    episodes = metadata.episodes
    sample_positions = sorted({0, len(episodes) // 2, len(episodes) - 1})
    samples = [
        validate_episode(metadata, episodes[position], decode=decode_samples)
        for position in sample_positions
    ]
    train_episodes, validation_episodes = metadata.split()
    return {
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "qwen_text_layers": int(config["model"]["text_layers"]),
        "backbone_family": family,
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "sampled_episodes": samples,
        "data_fingerprint": metadata.fingerprint(),
    }


def validate_gpu_host(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise PreflightError("PyTorch is not installed in this environment") from error
    required = int(config["train"]["gpu_count"])
    available = torch.cuda.device_count()
    if available < required:
        raise PreflightError(f"Requested {required} GPUs, but torch sees {available}")
    devices = []
    physical_ids = config["train"].get("gpu_ids")
    for index in range(required):
        properties = torch.cuda.get_device_properties(index)
        if properties.major < 8:
            raise PreflightError(f"GPU {index} does not provide the requested BF16 support")
        devices.append(
            {
                "index": index,
                "configured_physical_id": (
                    int(physical_ids[index]) if physical_ids is not None else None
                ),
                "name": properties.name,
                "memory_gib": round(properties.total_memory / 2**30, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices}


__all__ = [
    "MAX_RESERVED_MEMORY_GIB",
    "PreflightError",
    "validate_gpu_host",
    "validate_paths_and_data",
]
