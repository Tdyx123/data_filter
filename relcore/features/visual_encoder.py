"""Pluggable per-frame visual encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np


class VisualEncoder(Protocol):
    output_dim: int

    def encode(self, images: np.ndarray) -> np.ndarray: ...


class DummyVisualEncoder:
    """Deterministic image statistics encoder used only by tests and debug runs."""

    output_dim = 6

    def encode(self, images: np.ndarray) -> np.ndarray:
        values = np.asarray(images, dtype=np.float32)
        if values.ndim != 4 or values.shape[-1] != 3 or len(values) == 0:
            raise ValueError("images must have shape [time, height, width, 3]")
        means = values.mean(axis=(1, 2))
        stds = values.std(axis=(1, 2))
        return np.concatenate([means, stds], axis=1).astype(np.float32)


def _clip_feature_tensor(output: object):
    import torch

    if torch.is_tensor(output):
        return output
    pooled = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooled):
        return pooled
    raise TypeError("CLIP get_image_features returned no tensor feature")


class FrozenClipEncoder:
    """Strict local CLIP image encoder with no network or pixel fallback."""

    def __init__(self, config: Mapping[str, object]):
        if config.get("local_files_only") is not True:
            raise ValueError("FrozenClipEncoder requires local_files_only=true")
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            configured_device = str(config.get("device", "auto"))
            self.device = (
                "cuda"
                if configured_device == "auto" and torch.cuda.is_available()
                else configured_device
            )
            if self.device == "auto":
                self.device = "cpu"
            model_path = str(config["model"])
            self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = (
                CLIPModel.from_pretrained(model_path, local_files_only=True).eval().to(self.device)
            )
            self.output_dim = int(self.model.config.projection_dim)
            self.batch_size = int(config.get("batch_size", 64))
        except Exception as error:
            raise RuntimeError(f"local CLIP encoder could not be loaded: {error}") from error

    def encode(self, images: np.ndarray) -> np.ndarray:
        import torch

        values = np.asarray(images)
        if values.ndim != 4 or values.shape[-1] != 3 or len(values) == 0:
            raise ValueError("images must have shape [time, height, width, 3]")
        outputs: list[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            inputs = self.processor(
                images=[frame for frame in values[start : start + self.batch_size]],
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = _clip_feature_tensor(self.model.get_image_features(**inputs))
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            outputs.append(features.cpu().numpy().astype(np.float32))
        result = np.concatenate(outputs, axis=0)
        if not np.all(np.isfinite(result)):
            raise ValueError("CLIP produced NaN or infinity")
        return result
