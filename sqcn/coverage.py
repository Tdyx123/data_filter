"""Reference-fragment RBF affinity coverage."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .scaling import robust_unit_interval


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float32)
    right_values = np.asarray(right, dtype=np.float32)
    distances = (
        np.sum(left_values * left_values, axis=1, keepdims=True)
        + np.sum(right_values * right_values, axis=1)[None, :]
        - 2.0 * left_values @ right_values.T
    )
    return np.maximum(distances, 0.0)


def rbf_kernel(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    return np.exp(-_squared_distances(left, right) / (2.0 * sigma * sigma))


def median_heuristic(
    embeddings: np.ndarray,
    *,
    sample_size: int = 4096,
    pair_samples: int = 100_000,
    seed: int = 42,
    epsilon: float = 1.0e-8,
) -> float:
    values = np.asarray(embeddings, dtype=np.float32)
    if len(values) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    count = min(len(values), int(sample_size))
    indices = np.sort(rng.choice(len(values), size=count, replace=False))
    sampled = values[indices]
    pair_count = min(int(pair_samples), count * (count - 1) // 2)
    left = rng.integers(0, count, size=pair_count)
    right = rng.integers(0, count, size=pair_count)
    keep = left != right
    distances = np.linalg.norm(sampled[left[keep]] - sampled[right[keep]], axis=1)
    distances = distances[distances > epsilon]
    return float(np.median(distances)) if len(distances) else 1.0


def _use_cuda(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import torch

        return torch.cuda.is_available() if device in {"auto", "cuda"} else False
    except ImportError:
        return False


def reference_affinity(
    candidates: np.ndarray,
    reference: np.ndarray,
    sigma: float,
    *,
    batch_size: int = 2048,
    device: str = "auto",
) -> np.ndarray:
    """Return each candidate's mean RBF affinity to all reference fragments."""

    left = np.asarray(candidates, dtype=np.float32)
    right = np.asarray(reference, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
        raise ValueError("candidate/reference embeddings must share shape [samples, dim]")
    if len(right) == 0:
        raise ValueError("reference embeddings cannot be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = np.empty((len(left),), dtype=np.float64)
    if _use_cuda(device):
        import torch

        torch_device = "cuda" if device == "auto" else device
        right_tensor = torch.as_tensor(right, device=torch_device)
        right_norm = (right_tensor * right_tensor).sum(dim=1)
        denominator = 2.0 * sigma * sigma
        for start in range(0, len(left), batch_size):
            block = torch.as_tensor(left[start : start + batch_size], device=torch_device)
            accumulator = torch.zeros(len(block), dtype=torch.float64, device=torch_device)
            left_norm = (block * block).sum(dim=1, keepdim=True)
            for other_start in range(0, len(right), batch_size):
                other = right_tensor[other_start : other_start + batch_size]
                distances = (
                    left_norm
                    + right_norm[other_start : other_start + batch_size][None, :]
                    - 2.0 * block @ other.T
                ).clamp_min_(0.0)
                accumulator += torch.exp(-distances / denominator).double().sum(dim=1)
            output[start : start + len(block)] = (
                accumulator / len(right)
            ).cpu().numpy()
        return output.astype(np.float32)

    for start in range(0, len(left), batch_size):
        block = left[start : start + batch_size]
        accumulator = np.zeros((len(block),), dtype=np.float64)
        for other_start in range(0, len(right), batch_size):
            other = right[other_start : other_start + batch_size]
            accumulator += rbf_kernel(block, other, sigma).sum(axis=1)
        output[start : start + len(block)] = accumulator / len(right)
    return output.astype(np.float32)


def coverage_scores(
    candidates: np.ndarray,
    reference: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float]:
    """Return normalized coverage, raw affinity, bounds, and RBF sigma."""

    sigma_config = config.get("sigma", "median")
    sigma = (
        median_heuristic(
            reference,
            sample_size=int(config.get("sigma_sample_size", 4096)),
            pair_samples=int(config.get("sigma_pair_samples", 100_000)),
            seed=seed,
        )
        if str(sigma_config).lower() == "median"
        else float(sigma_config)
    )
    raw = reference_affinity(
        candidates,
        reference,
        sigma,
        batch_size=int(config.get("batch_size", 2048)),
        device=str(config.get("device", "auto")),
    )
    normalized, bounds = robust_unit_interval(
        raw,
        low=float(config.get("quantile_low", 0.01)),
        high=float(config.get("quantile_high", 0.99)),
    )
    return normalized, raw, bounds, sigma
