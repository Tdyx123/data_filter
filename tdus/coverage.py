"""RBF-MMD coverage with bounded-memory block computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    distances = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0)


def rbf_kernel(left: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    """Return one in-memory RBF kernel block."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return np.exp(-_squared_distances(left, right) / (2.0 * sigma * sigma))


def median_heuristic(
    embeddings: np.ndarray,
    *,
    sample_size: int = 4096,
    pair_samples: int = 100_000,
    seed: int = 42,
    epsilon: float = 1.0e-8,
) -> float:
    """Estimate RBF sigma from a deterministic sample of non-zero distances."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(embeddings) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    count = min(len(embeddings), sample_size)
    indices = np.sort(rng.choice(len(embeddings), size=count, replace=False))
    sampled = embeddings[indices]
    # Avoid materializing a 4096^2 matrix merely to estimate one scalar.
    pair_count = min(pair_samples, count * (count - 1) // 2)
    left = rng.integers(0, count, size=pair_count)
    right = rng.integers(0, count, size=pair_count)
    keep = left != right
    distances = np.linalg.norm(sampled[left[keep]] - sampled[right[keep]], axis=1)
    distances = distances[distances > epsilon]
    return float(np.median(distances)) if len(distances) else 1.0


def _torch_available(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import torch

        return torch.cuda.is_available() if device in {"auto", "cuda"} else False
    except ImportError:
        return False


def kernel_row_means(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    *,
    batch_size: int = 2048,
    device: str = "auto",
) -> np.ndarray:
    """Compute mean_j K(left_i, right_j) without allocating the full matrix."""

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if len(right) == 0:
        raise ValueError("right kernel population cannot be empty")
    output = np.empty((len(left),), dtype=np.float64)
    use_torch = _torch_available(device)
    if use_torch:
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


def mmd_squared(
    reference: np.ndarray,
    subset: np.ndarray,
    sigma: float,
    *,
    batch_size: int = 2048,
    device: str = "auto",
) -> float:
    """Biased MMD². The biased estimator is non-negative in exact arithmetic."""

    reference = np.asarray(reference, dtype=np.float32)
    subset = np.asarray(subset, dtype=np.float32)
    if len(reference) == 0 or len(subset) == 0:
        raise ValueError("MMD requires non-empty reference and subset")
    dd = float(
        kernel_row_means(
            reference,
            reference,
            sigma,
            batch_size=batch_size,
            device=device,
        ).mean()
    )
    ss = float(
        kernel_row_means(
            subset, subset, sigma, batch_size=batch_size, device=device
        ).mean()
    )
    ds = float(
        kernel_row_means(
            subset, reference, sigma, batch_size=batch_size, device=device
        ).mean()
    )
    return max(dd + ss - 2.0 * ds, 0.0)


@dataclass
class CoverageModel:
    """Reusable reference distribution and its MMD sufficient statistics."""

    reference: np.ndarray
    sigma: float
    reference_kernel_mean: float
    batch_size: int = 2048
    device: str = "auto"

    @classmethod
    def fit(
        cls,
        reference: np.ndarray,
        config: Mapping[str, Any],
        *,
        seed: int = 42,
    ) -> "CoverageModel":
        reference = np.asarray(reference, dtype=np.float32)
        maximum = config.get("reference_max_samples")
        if maximum and len(reference) > int(maximum):
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(len(reference), size=int(maximum), replace=False)
            )
            reference = reference[indices]
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
        batch_size = int(config.get("batch_size", 2048))
        device = str(config.get("device", "auto"))
        row_means = kernel_row_means(
            reference,
            reference,
            sigma,
            batch_size=batch_size,
            device=device,
        )
        return cls(reference, sigma, float(row_means.mean()), batch_size, device)

    def reference_affinity(self, samples: np.ndarray) -> np.ndarray:
        return kernel_row_means(
            samples,
            self.reference,
            self.sigma,
            batch_size=self.batch_size,
            device=self.device,
        )

    def score_samples(self, samples: np.ndarray) -> np.ndarray:
        """Coverage(D, {x}); singleton self-kernel is exactly one."""

        affinity = self.reference_affinity(samples)
        mmd2 = np.maximum(self.reference_kernel_mean + 1.0 - 2.0 * affinity, 0.0)
        return np.exp(-mmd2).astype(np.float32)

    def score_subset(self, subset: np.ndarray) -> float:
        subset = np.asarray(subset, dtype=np.float32)
        if len(subset) == 0:
            return 0.0
        subset_mean = float(
            kernel_row_means(
                subset,
                subset,
                self.sigma,
                batch_size=self.batch_size,
                device=self.device,
            ).mean()
        )
        cross_mean = float(self.reference_affinity(subset).mean())
        mmd2 = max(
            self.reference_kernel_mean + subset_mean - 2.0 * cross_mean, 0.0
        )
        return float(np.exp(-mmd2))

    def gain(self, subset: np.ndarray, sample: np.ndarray) -> float:
        subset = np.asarray(subset, dtype=np.float32)
        sample = np.asarray(sample, dtype=np.float32).reshape(1, -1)
        current = self.score_subset(subset)
        updated = sample if len(subset) == 0 else np.concatenate([subset, sample])
        return self.score_subset(updated) - current


def coverage(
    reference: np.ndarray,
    subset: np.ndarray,
    *,
    sigma: float,
    batch_size: int = 2048,
    device: str = "auto",
) -> float:
    return float(
        np.exp(
            -mmd_squared(
                reference, subset, sigma, batch_size=batch_size, device=device
            )
        )
    )


def coverage_gain(
    reference: np.ndarray,
    subset: np.ndarray,
    sample: np.ndarray,
    *,
    sigma: float,
    batch_size: int = 2048,
    device: str = "auto",
) -> float:
    model = CoverageModel.fit(
        reference,
        {"sigma": sigma, "batch_size": batch_size, "device": device},
    )
    return model.gain(subset, sample)
