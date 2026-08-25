from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from simpler_bridge.evaluation import SimplerEvaluationError, select_first_action

from .checkpoint import (
    OFFICIAL_BRIDGE_ACTION_MASK,
    OFFICIAL_BRIDGE_ACTION_MEAN,
    OFFICIAL_BRIDGE_ACTION_STD,
    OfficialCheckpointReport,
    sha256_file,
    validate_official_or_finetuned_checkpoint,
)


ACTION_SHAPE = (1, 4, 7)
IMAGE_SIZE = 256
LANGUAGE_TOKENS = 16


@dataclass(frozen=True)
class OfficialActionStatistics:
    path: Path
    mean: np.ndarray
    std: np.ndarray
    mask: np.ndarray
    sha256: str
    dataset_name: str = "bridge_dataset"

    def denormalize_actions(self, value: Any) -> np.ndarray:
        actions = np.asarray(value, dtype=np.float32)
        if actions.ndim < 1 or actions.shape[-1] != 7:
            raise SimplerEvaluationError(
                f"Official actions must end in dimension 7, found {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise SimplerEvaluationError("Official actions contain NaN or infinity")
        return np.where(
            self.mask,
            actions * self.std + self.mean,
            actions,
        ).astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "dataset_name": self.dataset_name,
            "normalization": "mean_std",
            "mask": self.mask.tolist(),
        }


def load_official_action_statistics(
    path: str | Path,
) -> OfficialActionStatistics:
    target = Path(path).expanduser().resolve()
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
        action = document["bridge_dataset"]["action"]
        mean = np.asarray(action["mean"], dtype=np.float32)
        std = np.asarray(action["std"], dtype=np.float32)
        mask = np.asarray(action["mask"], dtype=np.bool_)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SimplerEvaluationError(
            f"Could not load official Bridge action statistics from {target}: {error}"
        ) from error
    if mean.shape != (7,) or std.shape != (7,) or mask.shape != (7,):
        raise SimplerEvaluationError(
            "Official bridge_dataset action mean/std/mask must have shape (7,)"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise SimplerEvaluationError("Official Bridge statistics must be finite")
    if np.any(std <= 0):
        raise SimplerEvaluationError("Official Bridge action std must be positive")
    expected_mask = np.asarray(OFFICIAL_BRIDGE_ACTION_MASK, dtype=np.bool_)
    if not np.array_equal(mask, expected_mask):
        raise SimplerEvaluationError(
            "Official Bridge action mask must be six continuous dimensions and one gripper"
        )
    if not np.array_equal(
        mean, np.asarray(OFFICIAL_BRIDGE_ACTION_MEAN, dtype=np.float32)
    ) or not np.array_equal(std, np.asarray(OFFICIAL_BRIDGE_ACTION_STD, dtype=np.float32)):
        raise SimplerEvaluationError(
            "Official Bridge action mean/std must match the released Octo-small statistics"
        )
    return OfficialActionStatistics(
        path=target,
        mean=mean,
        std=std,
        mask=mask,
        sha256=sha256_file(target),
    )


@lru_cache(maxsize=16)
def _tensorflow_lanczos3_spans(
    input_size: int,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build TensorFlow ScaleAndTranslate Lanczos3 spans in float32."""

    scale = np.float32(output_size / input_size)
    inverse_scale = np.float32(1.0) / scale
    kernel_scale = np.maximum(inverse_scale, np.float32(1.0))
    radius = np.float32(3.0) * kernel_scale
    samples = (
        np.arange(output_size, dtype=np.float32) + np.float32(0.5)
    ) * inverse_scale - np.float32(0.5)
    starts: list[int] = []
    span_weights: list[np.ndarray] = []
    pi = np.float32(3.14159265359)
    for sample in samples:
        start = max(0, int(np.ceil(sample - radius)))
        stop = min(input_size, int(np.floor(sample + radius)) + 1)
        distance = np.abs(sample - np.arange(start, stop, dtype=np.float32)) / kernel_scale
        weights = np.empty_like(distance)
        near_zero = distance <= np.float32(1e-3)
        weights[near_zero] = np.float32(1.0)
        regular = ~near_zero
        regular_distance = distance[regular]
        weights[regular] = (
            np.float32(3.0)
            * np.sin(pi * regular_distance)
            * np.sin(pi * regular_distance / np.float32(3.0))
            / (pi * pi * regular_distance * regular_distance)
        )
        weights[distance > np.float32(3.0)] = np.float32(0.0)
        weights /= np.sum(weights, dtype=np.float32)
        starts.append(start)
        span_weights.append(weights)
    max_span = max(len(weights) for weights in span_weights)
    indices = np.zeros((output_size, max_span), dtype=np.int64)
    weights = np.zeros((output_size, max_span), dtype=np.float32)
    for output_index, (start, values) in enumerate(zip(starts, span_weights, strict=True)):
        count = len(values)
        indices[output_index, :count] = np.arange(start, start + count)
        weights[output_index, :count] = values
    return indices, weights


def resize_image_tensorflow_lanczos3(
    image: np.ndarray,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    """Match pinned SimplerEnv's TF Lanczos3+antialias uint8 pipeline."""

    output_height, output_width = size
    if image.shape[:2] == size:
        return image.copy()
    row_indices, row_weights = _tensorflow_lanczos3_spans(image.shape[0], output_height)
    column_indices, column_weights = _tensorflow_lanczos3_spans(image.shape[1], output_width)
    source = image.astype(np.float32)
    column_samples = source[:, column_indices, :]
    horizontally_resized = np.sum(
        column_samples * column_weights[None, :, :, None],
        axis=2,
        dtype=np.float32,
    )
    row_samples = horizontally_resized[row_indices, :, :]
    resized = np.sum(
        row_samples * row_weights[:, :, None, None],
        axis=1,
        dtype=np.float32,
    )
    return np.clip(np.round(resized), 0, 255).astype(np.uint8)


def preprocess_primary_image(value: Any) -> np.ndarray:
    image_array = np.asarray(value)
    if (
        image_array.dtype != np.uint8
        or image_array.ndim != 3
        or image_array.shape[0] <= 0
        or image_array.shape[1] <= 0
        or image_array.shape[2] != 3
    ):
        raise SimplerEvaluationError(
            "Official Octo expects a uint8 HWC RGB image; "
            f"found {image_array.shape} {image_array.dtype}"
        )
    image = resize_image_tensorflow_lanczos3(
        image_array,
        size=(IMAGE_SIZE, IMAGE_SIZE),
    )
    channel_first = image.astype(np.float32).transpose(2, 0, 1)
    return (channel_first / np.float32(127.5) - np.float32(1.0)).astype(np.float32, copy=False)


class OctoOfficialModelPolicy:
    policy_name = "Octo-small official-semantics PyTorch baseline"
    gripper_threshold = 0.5

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        statistics: OfficialActionStatistics,
        device: str,
        precision: str,
        generator_factory: Callable[[str, int], Any] | None = None,
        sampler: Callable[[Mapping[str, np.ndarray], Any], Any] | None = None,
        baseline: str = "official_parity",
    ) -> None:
        if precision not in {"bf16", "fp32"}:
            raise SimplerEvaluationError("precision must be bf16 or fp32")
        self.model = model
        self.tokenizer = tokenizer
        self.statistics = statistics
        self.device = str(device)
        self.precision = precision
        if baseline not in {"official_parity", "official_finetuned"}:
            raise SimplerEvaluationError(
                "baseline must be official_parity or official_finetuned"
            )
        self.baseline = baseline
        self._generator_factory = generator_factory
        self._sampler = sampler
        self._images: deque[np.ndarray] = deque(maxlen=2)
        self._instruction: str | None = None
        self._language: dict[str, np.ndarray] | None = None

    def make_generator(self, seed: int) -> Any:
        if self._generator_factory is not None:
            return self._generator_factory(self.device, int(seed))
        import torch

        return torch.Generator(device=self.device).manual_seed(int(seed))

    def begin_episode(self, instruction: str) -> None:
        text = str(instruction).strip()
        if not text:
            raise SimplerEvaluationError("Official Octo instruction must be non-empty")
        encoded = self.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=LANGUAGE_TOKENS,
            return_tensors="np",
        )
        try:
            input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.bool_)
        except (KeyError, TypeError, ValueError) as error:
            raise SimplerEvaluationError("Octo tokenizer returned invalid arrays") from error
        if input_ids.shape != (1, LANGUAGE_TOKENS) or attention_mask.shape != (
            1,
            LANGUAGE_TOKENS,
        ):
            raise SimplerEvaluationError(
                "Octo tokenizer must return input_ids and attention_mask with shape (1, 16)"
            )
        self._images.clear()
        self._instruction = text
        self._language = {
            "language_input_ids": input_ids,
            "language_attention_mask": attention_mask,
        }

    def prepare_observation(self, image: Any, instruction: str) -> dict[str, np.ndarray]:
        text = str(instruction).strip()
        if self._instruction is None or self._language is None:
            raise SimplerEvaluationError("begin_episode must be called before inference")
        if text != self._instruction:
            raise SimplerEvaluationError(
                "infer instruction must match the instruction cached by begin_episode"
            )
        self._images.append(preprocess_primary_image(image))
        history = np.stack(tuple(self._images), axis=0)[None]
        return {
            "image_primary": history,
            "timestep_pad_mask": np.ones((1, history.shape[1]), dtype=np.bool_),
            **self._language,
        }

    def describe_observation(self, prepared: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {
            "model_image_shape": list(prepared["image_primary"].shape),
            "image_history_length": int(prepared["image_primary"].shape[1]),
            "image_history_horizon": 2,
            "use_proprio": False,
            "observation_tokenizers": ["primary"],
        }

    def _sample_actions(self, prepared: Mapping[str, np.ndarray], generator: Any) -> Any:
        if self._sampler is not None:
            return self._sampler(prepared, generator)
        import torch

        if "proprio" in prepared:
            raise SimplerEvaluationError("Official model batch must not contain proprio")
        batch = {
            key: torch.from_numpy(value).to(self.device, non_blocking=True)
            for key, value in prepared.items()
        }
        use_bf16 = self.precision == "bf16"
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ),
        ):
            return self.model.sample_actions(batch, generator=generator)

    def predict_actions(self, prepared: Mapping[str, np.ndarray], *, generator: Any) -> np.ndarray:
        if "proprio" in prepared:
            raise SimplerEvaluationError("Official model batch must not contain proprio")
        actions = self._sample_actions(prepared, generator)
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        normalized = np.asarray(actions, dtype=np.float32)
        if normalized.shape != ACTION_SHAPE:
            raise SimplerEvaluationError(
                f"Official checkpoint returned {normalized.shape}; expected {ACTION_SHAPE}"
            )
        if not np.all(np.isfinite(normalized)):
            raise SimplerEvaluationError("Official checkpoint returned NaN or infinity")
        return self.statistics.denormalize_actions(normalized)

    def select_action(self, actions: np.ndarray) -> np.ndarray:
        return select_first_action(actions)

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "native_action_chunk_size": 4,
            "action_dim": 7,
            "image_history_horizon": 2,
            "use_proprio": False,
            "diffusion_steps": 20,
            "model_action_gripper": "continuous_model_prediction",
            "precision": self.precision,
            "statistics": self.statistics.as_dict(),
            "rng": {
                "backend": "torch.Generator",
                "jax_seed_bitwise_equivalent": False,
            },
        }


def load_official_policy(
    checkpoint: str | Path,
    *,
    device: str,
    precision: str,
    torch_module: Any | None = None,
) -> tuple[OfficialCheckpointReport, OctoOfficialModelPolicy]:
    report = validate_official_or_finetuned_checkpoint(checkpoint)
    if precision not in {"bf16", "fp32"}:
        raise SimplerEvaluationError("precision must be bf16 or fp32")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise SimplerEvaluationError("PyTorch is required for Octo evaluation") from error
    resolved_device = torch_module.device(device)
    if str(resolved_device.type) == "cuda":
        if not torch_module.cuda.is_available():
            raise SimplerEvaluationError(f"CUDA is unavailable for requested device {device}")
        if precision == "bf16" and not torch_module.cuda.is_bf16_supported():
            raise SimplerEvaluationError(f"CUDA device does not support BF16: {device}")
    elif precision == "bf16":
        raise SimplerEvaluationError("BF16 evaluation requires CUDA; use --precision fp32")
    try:
        from safetensors.torch import load_model
        from transformers import AutoTokenizer, T5Config, T5EncoderModel

        from .model import OctoSmallOfficialPolicy

        text_root = report.path / "text_encoder"
        text_config = T5Config.from_pretrained(text_root, local_files_only=True)
        model = OctoSmallOfficialPolicy(T5EncoderModel(text_config), report.config)
        load_model(
            model,
            str(report.weights_path),
            strict=True,
            device="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(text_root, local_files_only=True)
        model.to(resolved_device)
        model.eval()
    except SimplerEvaluationError:
        raise
    except Exception as error:
        raise SimplerEvaluationError(
            f"Could not load official PyTorch Octo-small checkpoint: {error}"
        ) from error
    return report, OctoOfficialModelPolicy(
        model=model,
        tokenizer=tokenizer,
        statistics=load_official_action_statistics(report.statistics_path),
        device=str(device),
        precision=precision,
        baseline=report.checkpoint_kind,
    )
