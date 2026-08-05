"""Median/IQR numeric normalization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustNormalizer:
    median: np.ndarray
    iqr: np.ndarray
    epsilon: float = 1.0e-6

    @classmethod
    def fit(
        cls,
        arrays: Iterable[np.ndarray],
        *,
        epsilon: float = 1.0e-6,
    ) -> "RobustNormalizer":
        materialized = [np.asarray(values, dtype=np.float32) for values in arrays]
        if not materialized:
            raise ValueError("cannot fit normalization without arrays")
        merged = np.concatenate(materialized, axis=0)
        if merged.ndim != 2 or not np.all(np.isfinite(merged)):
            raise ValueError("normalization inputs must be finite [time, dim] arrays")
        median = np.median(merged, axis=0).astype(np.float32)
        first, third = np.quantile(merged, (0.25, 0.75), axis=0)
        return cls(median, (third - first).astype(np.float32), float(epsilon))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(self.median):
            raise ValueError("normalizer input dimension does not match fitted statistics")
        if not np.all(np.isfinite(array)):
            raise ValueError("normalizer input contains NaN or infinity")
        return ((array - self.median) / (self.iqr + self.epsilon)).astype(np.float32)
