"""Cosine-based sample and subset diversity."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1.0e-8)


def scaled_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Map cosine similarity from [-1, 1] to [0, 1]."""

    return np.clip((l2_normalize(left) @ l2_normalize(right).T + 1.0) / 2.0, 0.0, 1.0)


def sample_diversity(
    embeddings: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int = 42,
) -> np.ndarray:
    """Per-sample dissimilarity to a deterministic candidate reference pool."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) <= 1:
        return np.zeros((len(embeddings),), dtype=np.float32)
    rng = np.random.default_rng(seed)
    reference_size = min(len(embeddings), int(config.get("reference_size", 4096)))
    reference_indices = np.sort(
        rng.choice(len(embeddings), size=reference_size, replace=False)
    )
    reference = l2_normalize(embeddings[reference_indices])
    normalized = l2_normalize(embeddings)
    batch_size = int(config.get("batch_size", 2048))
    result = np.empty((len(embeddings),), dtype=np.float32)
    reference_lookup = {int(index): pos for pos, index in enumerate(reference_indices)}
    device = str(config.get("device", "auto"))
    torch_device: str | None = None
    try:
        import torch

        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            torch_device = "cuda"
            reference_tensor = torch.as_tensor(reference, device=torch_device)
    except ImportError:
        torch_device = None
    for start in range(0, len(embeddings), batch_size):
        block = normalized[start : start + batch_size]
        if torch_device:
            block_tensor = torch.as_tensor(block, device=torch_device)
            similarities = (
                ((block_tensor @ reference_tensor.T + 1.0) / 2.0)
                .clamp_(0.0, 1.0)
                .cpu()
                .numpy()
            )
        else:
            similarities = np.clip((block @ reference.T + 1.0) / 2.0, 0.0, 1.0)
        sums = similarities.sum(axis=1)
        denominators = np.full(len(block), reference_size, dtype=np.float32)
        # Exclude self when the row is part of the sampled reference.
        for local, global_index in enumerate(range(start, start + len(block))):
            if global_index in reference_lookup:
                sums[local] -= 1.0
                denominators[local] -= 1.0
        means = np.divide(
            sums,
            denominators,
            out=np.ones_like(sums),
            where=denominators > 0,
        )
        result[start : start + len(block)] = 1.0 - means
    return np.clip(result, 0.0, 1.0)


def subset_diversity(embeddings: np.ndarray) -> float:
    """Exact pairwise diversity for a selected subset."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    count = len(embeddings)
    if count < 2:
        return 0.0
    normalized = l2_normalize(embeddings)
    cosine_sum = float((normalized @ normalized.T).sum() - count) / 2.0
    pair_count = count * (count - 1) / 2.0
    scaled_mean = (cosine_sum / pair_count + 1.0) / 2.0
    return float(np.clip(1.0 - scaled_mean, 0.0, 1.0))
