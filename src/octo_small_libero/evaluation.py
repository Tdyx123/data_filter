from __future__ import annotations

import hashlib
import io
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .libero10_tasks import LIBERO_10_TASK_NAMES


DEFAULT_CHECKPOINT = Path("/data/dwb/models/octo-small-pytorch")
DEFAULT_STATISTICS = Path("/data/dwb/datasets/LIBERO_lerobot/libero90/meta/stats.json")
DEFAULT_TASK_NAME = (
    "study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
)
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
MUJOCO_VERSION = "3.10.0"
ROBOSUITE_VERSION = "1.4.0"
BDDL_VERSION = "1.0.1"
GYMNASIUM_VERSION = "1.3.0"
CLOUDPICKLE_REQUIREMENT = ">=2.1.0,<4"
EASYDICT_REQUIREMENT = ">=1.9,<2"
FUTURE_REQUIREMENT = ">=0.18.2,<2"
MATPLOTLIB_REQUIREMENT = ">=3.5.3,<4"
TERMCOLOR_REQUIREMENT = ">=2.4.0,<4"
MUJOCO_COMPATIBILITY = "libero-robosuite-1.4.0-mujoco-3.10.0-gymnasium-1.3.0-v1"
LIBERO_MULTIPROCESSING_START_METHOD = "spawn"
RESULT_SCHEMA_VERSION = 3
EVALUATION_SEEDS = (3471197683, 1232873419, 1448008435)
ACTION_HORIZON = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
LANGUAGE_TOKENS = 16
LIBERO_WORKER_GRACEFUL_CLOSE_TIMEOUT_SECONDS = 10.0
LIBERO_WORKER_TERMINATE_TIMEOUT_SECONDS = 5.0
LIBERO_WORKER_KILL_TIMEOUT_SECONDS = 5.0
_LIBERO_ORIGINAL_WORKER_ATTRIBUTE = "_octo_small_libero_original_worker"
_LIBERO_ORIGINAL_CLOUDPICKLE_WRAPPER_ATTRIBUTE = "_octo_small_libero_original_cloudpickle_wrapper"


class EvaluationError(RuntimeError):
    """Raised when a checkpoint or LIBERO evaluation contract is invalid."""


class SimulationInfrastructureError(EvaluationError):
    """Raised when MuJoCo workers cannot initialize the simulation runtime."""


class UnrecoverableSimulationShutdownError(SimulationInfrastructureError):
    """Raised when LIBERO workers remain alive after SIGKILL."""

    def __init__(self, message: str, *, worker_pids: Sequence[int]):
        super().__init__(message)
        self.worker_pids = tuple(int(pid) for pid in worker_pids)


def configure_evaluation_multiprocessing() -> str:
    """Force LIBERO workers to start without inheriting the CUDA parent state."""
    current = multiprocessing.get_start_method(allow_none=True)
    if current != LIBERO_MULTIPROCESSING_START_METHOD:
        multiprocessing.set_start_method(
            LIBERO_MULTIPROCESSING_START_METHOD,
            force=current is not None,
        )
    actual = multiprocessing.get_start_method(allow_none=False)
    if actual != LIBERO_MULTIPROCESSING_START_METHOD:
        raise EvaluationError(
            "LIBERO evaluation requires multiprocessing start method "
            f"{LIBERO_MULTIPROCESSING_START_METHOD!r}; found {actual!r}"
        )
    return actual


def _multiprocessing_start_method() -> str:
    return configure_evaluation_multiprocessing()


@dataclass(frozen=True)
class SimulationStartupFailure:
    pid: int
    exception_type: str
    message: str
    mujoco_gl: str | None
    cuda_visible_devices: str | None
    mujoco_egl_device_id: str | None


class _FailedOffscreenEnvironment:
    """Pickle-safe carrier that lets a failed worker report its startup error."""

    def __init__(self, failure: SimulationStartupFailure):
        self._octo_startup_failure = failure

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class CheckpointSpec:
    requested_path: Path
    weights_path: Path
    base_model_path: Path
    self_contained: bool
    model_config_path: Path
    text_encoder_path: Path
    weights_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_path": str(self.requested_path),
            "weights_path": str(self.weights_path),
            "base_model_path": str(self.base_model_path),
            "self_contained": self.self_contained,
            "model_config_path": str(self.model_config_path),
            "text_encoder_path": str(self.text_encoder_path),
            "weights_sha256": self.weights_sha256,
        }


@dataclass(frozen=True)
class NormalizationStatistics:
    path: Path
    action_mean: np.ndarray
    action_std: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    sha256: str

    def normalize_proprio(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != PROPRIO_DIM:
            raise EvaluationError(f"Expected proprio[..., {PROPRIO_DIM}], found {array.shape}")
        return (array - self.proprio_mean) / (self.proprio_std + np.float32(1.0e-8))

    def actions_to_environment(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != ACTION_DIM:
            raise EvaluationError(f"Expected actions[..., {ACTION_DIM}], found {array.shape}")
        result = array.copy()
        result[..., :6] = result[..., :6] * self.action_std[:6] + self.action_mean[:6]
        result[..., 6] = 1.0 - 2.0 * result[..., 6]
        if not np.all(np.isfinite(result)):
            raise EvaluationError("Action denormalization produced NaN or infinity")
        return result


@dataclass(frozen=True)
class EvaluationSettings:
    checkpoint: Path = DEFAULT_CHECKPOINT
    base_model: Path | None = None
    statistics: Path = DEFAULT_STATISTICS
    output_dir: Path = Path("outputs/octo_small_libero_eval")
    save_videos_path: Path | None = None
    task_name: str = DEFAULT_TASK_NAME
    episodes: int = 150
    num_envs: int = 50
    auto_reduce_num_envs: bool = True
    max_steps: int = 960
    settle_steps: int = 20
    seeds: tuple[int, int, int] = EVALUATION_SEEDS
    device: str = "cuda:0"
    precision: str = "bf16"
    record_videos: int = 0
    video_fps: int = 30
    libero_root: Path | None = None
    libero_config_path: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class LiberoTask:
    task_id: int
    name: str
    language: str
    bddl_file: Path
    init_states_file: Path
    init_states_sha256: str
    init_states_git_blob: str
    init_states: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _resolve_statistics_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target.is_dir():
        candidates = (target / "meta" / "stats.json", target / "stats.json")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    if not target.is_file():
        raise EvaluationError(f"LIBERO statistics file does not exist: {target}")
    return target


def load_statistics(path: str | Path) -> NormalizationStatistics:
    target = _resolve_statistics_path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Could not read LIBERO statistics: {target}") from error

    try:
        action = value["action"]
        proprio = value["observation.state"]
        action_mean = np.asarray(action["mean"], dtype=np.float32)
        action_std = np.asarray(action["std"], dtype=np.float32)
        proprio_mean = np.asarray(proprio["mean"], dtype=np.float32)
        proprio_std = np.asarray(proprio["std"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(
            f"{target} must contain action and observation.state mean/std arrays"
        ) from error
    expected = (
        ("action.mean", action_mean, (ACTION_DIM,)),
        ("action.std", action_std, (ACTION_DIM,)),
        ("observation.state.mean", proprio_mean, (PROPRIO_DIM,)),
        ("observation.state.std", proprio_std, (PROPRIO_DIM,)),
    )
    for name, array, shape in expected:
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise EvaluationError(f"{target}: {name} must be a finite array with shape {shape}")
        if name.endswith(".std") and np.any(array < 0):
            raise EvaluationError(f"{target}: {name} must be non-negative")
    return NormalizationStatistics(
        path=target,
        action_mean=action_mean,
        action_std=action_std,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        sha256=_sha256(target),
    )


def _required_model_files(root: Path) -> dict[str, Path]:
    text = root / "text_encoder"
    return {
        "weights": root / "model.safetensors",
        "config": root / "model_config.json",
        "text_config": text / "config.json",
        "tokenizer_config": text / "tokenizer_config.json",
        "tokenizer_model": text / "spiece.model",
        "tokenizer_json": text / "tokenizer.json",
    }


def _is_self_contained_model(root: Path) -> bool:
    return root.is_dir() and all(path.is_file() for path in _required_model_files(root).values())


def _validate_model_config(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Could not read model config: {path}") from error
    expected = {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "proprio_dim": PROPRIO_DIM,
        "language_tokens": LANGUAGE_TOKENS,
    }
    for key, expected_value in expected.items():
        try:
            actual = int(config[key])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError(f"{path}: missing integer {key}") from error
        if actual != expected_value:
            raise EvaluationError(f"{path}: expected {key}={expected_value}, found {actual}")


def resolve_checkpoint(
    checkpoint: str | Path,
    *,
    base_model: str | Path | None = None,
) -> CheckpointSpec:
    if Path(checkpoint).name.lower() in {"best", "latest"}:
        raise EvaluationError(
            "Training pointer aliases such as best/latest are not supported; "
            "pass an explicit checkpoint directory or .safetensors file"
        )
    requested = Path(checkpoint).expanduser().resolve()
    if not requested.exists():
        raise EvaluationError(f"Checkpoint does not exist: {requested}")
    if requested.is_dir():
        weights = requested / "model.safetensors"
        artifact_root = requested
    else:
        if requested.suffix != ".safetensors":
            raise EvaluationError("A checkpoint file must have the .safetensors extension")
        weights = requested
        artifact_root = requested.parent
    if not weights.is_file():
        raise EvaluationError(f"Checkpoint weights do not exist: {weights}")

    self_contained = _is_self_contained_model(artifact_root)
    if self_contained:
        if base_model is not None:
            raise EvaluationError("--base-model is only valid for a non-self-contained checkpoint")
        base_root = artifact_root
    else:
        if base_model is None:
            raise EvaluationError(f"{requested} contains weights only; provide --base-model")
        base_root = Path(base_model).expanduser().resolve()
        if not _is_self_contained_model(base_root):
            missing = [
                str(path)
                for path in _required_model_files(base_root).values()
                if not path.is_file()
            ]
            raise EvaluationError(
                f"Base model is not self-contained: {base_root}; missing {missing}"
            )

    files = _required_model_files(base_root)
    _validate_model_config(files["config"])
    return CheckpointSpec(
        requested_path=requested,
        weights_path=weights,
        base_model_path=base_root,
        self_contained=self_contained,
        model_config_path=files["config"],
        text_encoder_path=base_root / "text_encoder",
        weights_sha256=_sha256(weights),
    )


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert a finite xyzw quaternion to an axis-angle vector."""
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise EvaluationError(f"Expected a finite xyzw quaternion, found {quat}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator).astype(np.float32)


def observation_to_proprio(observation: Mapping[str, Any]) -> np.ndarray:
    required = {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
    missing = required.difference(observation)
    if missing:
        raise EvaluationError(f"LIBERO observation is missing keys: {sorted(missing)}")
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if position.shape != (3,) or len(gripper) < 1:
        raise EvaluationError(
            f"Invalid LIBERO proprio shapes: position={position.shape}, gripper={gripper.shape}"
        )
    value = np.concatenate(
        [
            position,
            quaternion_to_axis_angle(observation["robot0_eef_quat"]),
            np.zeros(1, dtype=np.float32),
            gripper[:1],
        ]
    ).astype(np.float32)
    if value.shape != (PROPRIO_DIM,) or not np.all(np.isfinite(value)):
        raise EvaluationError("LIBERO proprio conversion produced invalid values")
    return value


def resize_flipped_rgb(value: Any, size: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise EvaluationError("Pillow is required for LIBERO image preprocessing") from error
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise EvaluationError(f"Expected an HWC RGB image, found {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise EvaluationError(f"Expected an integer RGB image, found {array.dtype}")
    array = np.flip(array.astype(np.uint8, copy=False), axis=0)
    image = Image.fromarray(array)
    if image.size != (size[1], size[0]):
        image = image.resize((size[1], size[0]), resample=Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8).copy()


def observation_batch_to_list(observations: Any) -> list[Mapping[str, Any]]:
    if isinstance(observations, Mapping):
        if not observations:
            raise EvaluationError("Received an empty batched observation")
        if {
            "robot0_eef_pos",
            "robot0_eef_quat",
            "agentview_image",
        }.issubset(observations):
            return [observations]
        first = np.asarray(next(iter(observations.values())))
        if first.ndim == 0:
            return [observations]
        batch_size = len(first)
        return [
            {key: np.asarray(value)[index] for key, value in observations.items()}
            for index in range(batch_size)
        ]
    if isinstance(observations, np.ndarray):
        values = observations.tolist()
    else:
        values = list(observations)
    if isinstance(values, Mapping):
        values = [values]
    if not values or not all(isinstance(value, Mapping) for value in values):
        raise EvaluationError("LIBERO vector observations must contain dictionaries")
    return list(values)


def build_model_batch(
    observations: Any,
    *,
    instruction: str,
    tokenizer: Any,
    statistics: NormalizationStatistics,
    language_tokens: int = LANGUAGE_TOKENS,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("PyTorch is required for checkpoint evaluation") from error
    values = observation_batch_to_list(observations)
    primary = []
    wrist = []
    proprio = []
    for observation in values:
        missing = {"agentview_image", "robot0_eye_in_hand_image"}.difference(observation)
        if missing:
            raise EvaluationError(f"LIBERO observation is missing image keys: {sorted(missing)}")
        primary.append(resize_flipped_rgb(observation["agentview_image"], (256, 256)))
        wrist.append(resize_flipped_rgb(observation["robot0_eye_in_hand_image"], (128, 128)))
        proprio.append(observation_to_proprio(observation))

    primary_array = np.stack(primary).astype(np.float32) / 127.5 - 1.0
    wrist_array = np.stack(wrist).astype(np.float32) / 127.5 - 1.0
    proprio_array = statistics.normalize_proprio(np.stack(proprio))
    encoded = tokenizer(
        [instruction] * len(values),
        padding="max_length",
        truncation=True,
        max_length=language_tokens,
        return_tensors="pt",
    )
    return {
        "image_primary": torch.from_numpy(primary_array).permute(0, 3, 1, 2).unsqueeze(1),
        "image_wrist": torch.from_numpy(wrist_array).permute(0, 3, 1, 2).unsqueeze(1),
        "proprio": torch.from_numpy(proprio_array).unsqueeze(1),
        "language_input_ids": encoded["input_ids"].to(dtype=torch.long),
        "language_attention_mask": encoded["attention_mask"].to(dtype=torch.bool),
    }


class LoadedCheckpointPolicy:
    def __init__(
        self,
        *,
        checkpoint: CheckpointSpec,
        model: Any,
        tokenizer: Any,
        statistics: NormalizationStatistics,
        device: Any,
        precision: str,
    ):
        self.checkpoint = checkpoint
        self.model = model
        self.tokenizer = tokenizer
        self.statistics = statistics
        self.device = device
        self.precision = precision

    def make_generator(self, seed: int) -> Any:
        import torch

        return torch.Generator(device=self.device).manual_seed(int(seed))

    def predict_action_chunk(
        self,
        observations: Any,
        instruction: str,
        *,
        generator: Any,
    ) -> np.ndarray:
        import torch

        batch = build_model_batch(
            observations,
            instruction=instruction,
            tokenizer=self.tokenizer,
            statistics=self.statistics,
            language_tokens=int(self.model.config.language_tokens),
        )
        batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}
        use_bf16 = self.precision == "bf16"
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                actions = self.model.sample_actions(batch, generator=generator)
        actions = actions.float().cpu().numpy()
        expected = (len(observation_batch_to_list(observations)), ACTION_HORIZON, ACTION_DIM)
        if actions.shape != expected:
            raise EvaluationError(
                f"Checkpoint returned actions with shape {actions.shape}; expected {expected}"
            )
        if not np.all(np.isfinite(actions)):
            raise EvaluationError("Checkpoint returned NaN or infinite actions")
        return self.statistics.actions_to_environment(actions)


def load_checkpoint_policy(
    checkpoint: CheckpointSpec,
    *,
    statistics: NormalizationStatistics,
    device: str,
    precision: str,
) -> LoadedCheckpointPolicy:
    try:
        import torch
        from safetensors.torch import load_model
    except ImportError as error:
        raise EvaluationError(
            "torch and safetensors are required for checkpoint evaluation"
        ) from error
    from .torch_model import OctoSmallPolicy

    if precision not in {"bf16", "fp32"}:
        raise EvaluationError("precision must be bf16 or fp32")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise EvaluationError(f"CUDA device requested but CUDA is unavailable: {device}")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise EvaluationError(f"CUDA device does not support BF16: {device}")
    elif precision == "bf16":
        raise EvaluationError("BF16 evaluation requires a CUDA device; use --precision fp32")

    model, tokenizer = OctoSmallPolicy.from_pretrained(
        checkpoint.base_model_path,
        device="cpu",
    )
    base_weights = (checkpoint.base_model_path / "model.safetensors").resolve()
    if checkpoint.weights_path.resolve() != base_weights:
        load_model(
            model,
            str(checkpoint.weights_path),
            strict=True,
            device="cpu",
        )
    config = model.config
    actual = {
        "action_dim": int(config.action_dim),
        "action_horizon": int(config.action_horizon),
        "proprio_dim": int(config.proprio_dim),
        "language_tokens": int(config.language_tokens),
    }
    expected = {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "proprio_dim": PROPRIO_DIM,
        "language_tokens": LANGUAGE_TOKENS,
    }
    if actual != expected:
        raise EvaluationError(f"Loaded checkpoint model contract is {actual}; expected {expected}")
    model.to(resolved_device)
    model.eval()
    return LoadedCheckpointPolicy(
        checkpoint=checkpoint,
        model=model,
        tokenizer=tokenizer,
        statistics=statistics,
        device=resolved_device,
        precision=precision,
    )


def _is_libero_package_root(path: Path) -> bool:
    required = (path / "bddl_files", path / "init_files", path / "assets")
    return path.is_dir() and all(item.is_dir() for item in required)


def _find_libero_package_root(source_root: Path | None = None) -> Path:
    requested = source_root
    if requested is None and os.environ.get("LIBERO_ROOT"):
        requested = Path(os.environ["LIBERO_ROOT"])
    if requested is None:
        bundled = Path(__file__).resolve().parents[2] / "third_party" / "LIBERO"
        if bundled.is_dir():
            requested = bundled
    if requested is not None:
        source = requested.expanduser().resolve()
        candidates = (source / "libero" / "libero", source / "libero", source)
        for root in candidates:
            if not _is_libero_package_root(root):
                continue
            if root.name != "libero" or root.parent.name != "libero":
                raise EvaluationError(f"LIBERO source has an unsupported package layout: {source}")
            import_root = root.parent.parent
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
            return root
        raise EvaluationError(f"LIBERO source is missing libero/libero benchmark assets: {source}")

    package = importlib.util.find_spec("libero")
    if package is None or not package.submodule_search_locations:
        raise EvaluationError(
            "LIBERO source was not found; pass --libero-root PATH or set LIBERO_ROOT "
            f"to the official checkout at commit {LIBERO_COMMIT}"
        )
    outer = Path(next(iter(package.submodule_search_locations))).resolve()
    root = outer / "libero"
    if not _is_libero_package_root(root):
        raise EvaluationError(f"Installed LIBERO package is missing benchmark assets: {root}")
    return root


def _verify_libero_commit(root: Path) -> str:
    repository = root.parent.parent
    if not (repository / ".git").exists():
        raise EvaluationError(
            "LIBERO must be an official Git checkout so its pinned commit can be verified; "
            f"found benchmark assets under {root}"
        )
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EvaluationError(f"Could not read LIBERO Git commit: {detail}")
    commit = result.stdout.strip()
    if commit != LIBERO_COMMIT:
        raise EvaluationError(
            f"LIBERO checkout is at {commit}; expected pinned commit {LIBERO_COMMIT}"
        )
    return commit


def _verify_libero_init_file(root: Path, path: Path) -> tuple[str, bytes]:
    repository = root.parent.parent.resolve()
    init_root = (root / "init_files").resolve()
    target = path.expanduser().resolve()
    try:
        target.relative_to(init_root)
        relative = target.relative_to(repository).as_posix()
    except ValueError as error:
        raise EvaluationError(
            f"LIBERO initial-state file must be inside {init_root}: {target}"
        ) from error
    if not target.is_file():
        raise EvaluationError(f"LIBERO initial-state file does not exist: {target}")
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise EvaluationError(f"Could not read LIBERO initial-state file: {target}") from error

    expected = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", f"HEAD:{relative}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if expected.returncode != 0:
        raise EvaluationError(
            f"LIBERO initial-state file is not tracked by the pinned commit: {target}"
        )
    actual = subprocess.run(
        ["git", "-C", str(repository), "hash-object", "--stdin"],
        check=False,
        input=payload,
        capture_output=True,
        timeout=10,
    )
    if actual.returncode != 0:
        detail = (actual.stderr.strip() or actual.stdout.strip()).decode(
            "utf-8",
            errors="replace",
        )
        raise EvaluationError(f"Could not hash LIBERO initial-state file: {detail}")
    expected_blob = expected.stdout.strip()
    actual_blob = actual.stdout.decode("ascii").strip()
    if actual_blob != expected_blob:
        raise EvaluationError(
            "LIBERO initial-state file differs from the pinned Git commit: "
            f"{target} (expected blob {expected_blob}, found {actual_blob})"
        )
    return actual_blob, payload


def load_trusted_libero_init_states(
    root: Path,
    path: Path,
) -> tuple[np.ndarray, str, str]:
    _verify_libero_commit(root)
    target = path.expanduser().resolve()
    git_blob, payload = _verify_libero_init_file(root, target)
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("PyTorch is required to load LIBERO initial states") from error
    try:
        value = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise EvaluationError(f"Could not load verified LIBERO initial states: {target}") from error
    states = np.asarray(value)
    if (
        states.ndim != 2
        or states.shape[0] <= 0
        or not np.issubdtype(states.dtype, np.number)
        or not np.all(np.isfinite(states))
    ):
        raise EvaluationError(
            f"LIBERO initial states must be a non-empty finite numeric matrix: "
            f"shape={states.shape}, dtype={states.dtype}"
        )
    return states.copy(), hashlib.sha256(payload).hexdigest(), git_blob


def configure_libero(
    *,
    output_dir: Path,
    datasets_path: Path,
    source_root: Path | None = None,
    config_path: Path | None = None,
) -> tuple[Path, Path, str]:
    root = _find_libero_package_root(source_root)
    commit = _verify_libero_commit(root)
    target = (
        config_path.expanduser().resolve()
        if config_path is not None
        else (output_dir / ".libero").resolve()
    )
    target.mkdir(parents=True, exist_ok=True)
    config_file = target / "config.yaml"
    config = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(datasets_path),
        "assets": str(root / "assets"),
    }
    _atomic_write_json(config_file, config)
    os.environ["LIBERO_CONFIG_PATH"] = str(target)
    imported = sys.modules.get("libero.libero")
    if imported is not None:
        imported_config = Path(str(getattr(imported, "config_file", ""))).resolve()
        if imported_config != config_file:
            raise EvaluationError("LIBERO was imported before LIBERO_CONFIG_PATH was configured")
    return root, config_file, commit


def resolve_libero_task(task_name: str, *, libero_root: Path) -> LiberoTask:
    requested = task_name.strip().lower()
    matches: list[tuple[int, str, str]] = []
    for task_id, name in enumerate(LIBERO_10_TASK_NAMES):
        scene_marker = name.index("_SCENE")
        language_start = name.index("_", scene_marker + 1) + 1
        language = name[language_start:].replace("_", " ")
        candidates = {
            name.lower(),
            language.lower(),
            name.lower().removesuffix("_demo"),
        }
        if requested in candidates:
            matches.append((task_id, name, language))
    if len(matches) != 1:
        raise EvaluationError(
            f"LIBERO-10 task {task_name!r} resolved to {len(matches)} tasks; "
            f"available tasks are {list(LIBERO_10_TASK_NAMES)}"
        )
    task_id, name, language = matches[0]
    bddl_root = (libero_root / "bddl_files").resolve()
    bddl_file = (bddl_root / "libero_10" / f"{name}.bddl").resolve()
    try:
        bddl_file.relative_to(bddl_root)
    except ValueError as error:
        raise EvaluationError(
            f"LIBERO BDDL file escapes the benchmark root: {bddl_file}"
        ) from error
    if not bddl_file.is_file():
        raise EvaluationError(f"LIBERO BDDL file does not exist: {bddl_file}")
    init_states_file = (libero_root / "init_files" / "libero_10" / f"{name}.pruned_init").resolve()
    init_states, init_states_sha256, init_states_git_blob = load_trusted_libero_init_states(
        libero_root, init_states_file
    )
    return LiberoTask(
        task_id=task_id,
        name=name,
        language=language,
        bddl_file=bddl_file,
        init_states_file=init_states_file,
        init_states_sha256=init_states_sha256,
        init_states_git_blob=init_states_git_blob,
        init_states=init_states,
    )


def validate_simulation_dependencies() -> dict[str, str]:
    exact = {
        "mujoco": MUJOCO_VERSION,
        "robosuite": ROBOSUITE_VERSION,
        "bddl": BDDL_VERSION,
        "gymnasium": GYMNASIUM_VERSION,
    }
    compatible = {
        "cloudpickle": CLOUDPICKLE_REQUIREMENT,
        "easydict": EASYDICT_REQUIREMENT,
        "future": FUTURE_REQUIREMENT,
        "matplotlib": MATPLOTLIB_REQUIREMENT,
        "termcolor": TERMCOLOR_REQUIREMENT,
    }
    versions: dict[str, str] = {}
    for distribution, required in exact.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise EvaluationError(
                f"Missing evaluation dependency {distribution}=={required}; "
                "install requirements-octo-libero-eval.txt"
            ) from error
        if actual != required:
            raise EvaluationError(f"Evaluation requires {distribution}=={required}, found {actual}")
        versions[distribution] = actual
    for distribution, requirement in compatible.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise EvaluationError(
                f"Missing evaluation dependency {distribution}{requirement}; "
                "install requirements-octo-libero-eval.txt"
            ) from error
        try:
            supported = Version(actual) in SpecifierSet(requirement)
        except (InvalidVersion, InvalidSpecifier) as error:
            raise EvaluationError(
                f"Could not validate evaluation dependency {distribution} "
                f"version {actual!r} against {requirement}"
            ) from error
        if not supported:
            raise EvaluationError(
                f"Evaluation requires {distribution}{requirement}, found {actual}"
            )
        versions[distribution] = actual
    return versions


def _install_libero_gymnasium_compatibility(
    *,
    gymnasium_module: Any | None = None,
    module_registry: dict[str, Any] | None = None,
) -> bool:
    if gymnasium_module is None:
        try:
            import gymnasium as gymnasium_module
        except ImportError as error:
            raise EvaluationError(f"Could not import Gymnasium {GYMNASIUM_VERSION}") from error
    actual_version = str(getattr(gymnasium_module, "__version__", ""))
    if actual_version != GYMNASIUM_VERSION:
        raise EvaluationError(
            f"LIBERO Gymnasium compatibility requires {GYMNASIUM_VERSION}, "
            f"found {actual_version or 'unknown'}"
        )

    registry = sys.modules if module_registry is None else module_registry
    existing = registry.get("gym")
    if existing is gymnasium_module:
        return False
    if existing is not None:
        raise EvaluationError(
            "Legacy gym was imported before the LIBERO Gymnasium compatibility alias was installed"
        )
    registry["gym"] = gymnasium_module
    return True


class _LiberoWorkerCloudpickleWrapper:
    """Serialize LIBERO environment factories without importing LIBERO in a worker."""

    def __init__(self, data: Any):
        self.data = data

    def __getstate__(self) -> bytes:
        import cloudpickle

        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        import cloudpickle

        self.data = cloudpickle.loads(data)


def _run_libero_subprocess_worker_with_gymnasium(
    parent: Any,
    pipe: Any,
    env_fn_wrapper: Any,
    observation_buffers: Any = None,
) -> None:
    """Install the Gymnasium alias before a spawned worker imports LIBERO."""
    _install_libero_gymnasium_compatibility()
    try:
        from libero.libero.envs import venv as libero_venv
    except ImportError as error:
        raise EvaluationError(
            "Could not import the pinned LIBERO subprocess worker after "
            "installing Gymnasium compatibility"
        ) from error

    original_worker = getattr(
        libero_venv,
        _LIBERO_ORIGINAL_WORKER_ATTRIBUTE,
        getattr(libero_venv, "_worker", None),
    )
    if original_worker is None or original_worker is _run_libero_subprocess_worker_with_gymnasium:
        raise EvaluationError("Could not resolve the original LIBERO subprocess worker")
    original_worker(parent, pipe, env_fn_wrapper, observation_buffers)


def _install_libero_subprocess_worker_bootstrap(libero_venv: Any) -> bool:
    """Make LIBERO spawn workers import this module before their own venv module."""
    current_worker = getattr(libero_venv, "_worker", None)
    current_wrapper = getattr(libero_venv, "CloudpickleWrapper", None)
    worker_installed = current_worker is _run_libero_subprocess_worker_with_gymnasium
    wrapper_installed = current_wrapper is _LiberoWorkerCloudpickleWrapper
    if worker_installed and wrapper_installed:
        return False
    if worker_installed != wrapper_installed:
        raise EvaluationError("LIBERO subprocess worker bootstrap is only partially installed")
    if current_worker is None or current_wrapper is None:
        raise EvaluationError("Pinned LIBERO subprocess worker internals are unavailable")

    setattr(libero_venv, _LIBERO_ORIGINAL_WORKER_ATTRIBUTE, current_worker)
    setattr(
        libero_venv,
        _LIBERO_ORIGINAL_CLOUDPICKLE_WRAPPER_ATTRIBUTE,
        current_wrapper,
    )
    libero_venv._worker = _run_libero_subprocess_worker_with_gymnasium
    libero_venv.CloudpickleWrapper = _LiberoWorkerCloudpickleWrapper
    return True


def _install_robosuite_mujoco_310_compatibility(
    *,
    mujoco_module: Any | None = None,
    mjdata_wrapper: type[Any] | None = None,
) -> bool:
    if mujoco_module is None:
        try:
            import mujoco as mujoco_module
        except ImportError as error:
            raise EvaluationError(f"Could not import MuJoCo {MUJOCO_VERSION}") from error
    actual_version = str(getattr(mujoco_module, "__version__", ""))
    if actual_version != MUJOCO_VERSION:
        raise EvaluationError(
            f"MuJoCo compatibility layer requires {MUJOCO_VERSION}, "
            f"found {actual_version or 'unknown'}"
        )
    if mjdata_wrapper is None:
        try:
            from robosuite.utils.binding_utils import MjData as mjdata_wrapper
        except ImportError as error:
            raise EvaluationError(
                f"Could not import robosuite {ROBOSUITE_VERSION} MuJoCo bindings"
            ) from error

    current_full_m = mujoco_module.mj_fullM
    if getattr(current_full_m, "_octo_mujoco_310_compatibility", False):
        return False

    native_data_type = mujoco_module.MjData

    def compatible_full_m(model: Any, second: Any, third: Any) -> None:
        if isinstance(second, native_data_type):
            current_full_m(model, second, third)
            return
        if isinstance(third, native_data_type):
            current_full_m(model, third, second)
            return
        raise TypeError("MuJoCo 3.10 mj_fullM compatibility expected an MjData argument")

    compatible_full_m._octo_mujoco_310_compatibility = True
    compatible_full_m._octo_native_mj_fullM = current_full_m
    mjdata_wrapper.qM = property(lambda instance: instance._data)
    mujoco_module.mj_fullM = compatible_full_m
    return True


def _process_is_alive(process: Any) -> bool:
    try:
        return bool(process.is_alive())
    except (AssertionError, OSError, ValueError):
        return False


def _join_processes_until(processes: Sequence[Any], deadline: float) -> None:
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.join(remaining)
        except (AssertionError, OSError, ValueError):
            continue


def _close_libero_subprocess_vector_environment(environment: Any) -> None:
    """Close every LIBERO worker within group-wide graceful/TERM/KILL deadlines."""
    if bool(getattr(environment, "is_closed", False)):
        return
    environment.is_closed = True
    workers = list(getattr(environment, "workers", ()))
    worker_records: list[tuple[Any, Any, Any]] = []
    for worker in workers:
        worker.is_closed = True
        process = getattr(worker, "process", None)
        remote = getattr(worker, "parent_remote", None)
        if process is not None:
            worker_records.append((worker, process, remote))

    processes = [process for _, process, _ in worker_records]
    pending_remotes: list[Any] = []
    terminated_pids: list[int] = []
    killed_pids: list[int] = []
    try:
        for _, process, remote in worker_records:
            if remote is None or not _process_is_alive(process):
                continue
            try:
                remote.send(["close", None])
                pending_remotes.append(remote)
            except (BrokenPipeError, ConnectionError, EOFError, OSError, ValueError):
                continue

        graceful_deadline = time.monotonic() + LIBERO_WORKER_GRACEFUL_CLOSE_TIMEOUT_SECONDS
        for remote in pending_remotes:
            remaining = max(0.0, graceful_deadline - time.monotonic())
            try:
                if remote.poll(remaining):
                    remote.recv()
            except (BrokenPipeError, ConnectionError, EOFError, OSError, ValueError):
                continue
        _join_processes_until(processes, graceful_deadline)

        alive = [process for process in processes if _process_is_alive(process)]
        for process in alive:
            if process.pid is not None:
                terminated_pids.append(int(process.pid))
            try:
                process.terminate()
            except (AttributeError, OSError, ProcessLookupError):
                continue
        terminate_deadline = time.monotonic() + LIBERO_WORKER_TERMINATE_TIMEOUT_SECONDS
        _join_processes_until(alive, terminate_deadline)

        alive = [process for process in alive if _process_is_alive(process)]
        for process in alive:
            if process.pid is not None:
                killed_pids.append(int(process.pid))
            try:
                process.kill()
            except (AttributeError, OSError, ProcessLookupError):
                continue
        kill_deadline = time.monotonic() + LIBERO_WORKER_KILL_TIMEOUT_SECONDS
        _join_processes_until(alive, kill_deadline)
    finally:
        for _, _, remote in worker_records:
            if remote is None:
                continue
            try:
                remote.close()
            except (AttributeError, OSError, ValueError):
                pass

    if terminated_pids:
        print(
            f"[warning] forced LIBERO worker shutdown with terminate: pids={terminated_pids}",
            file=sys.stderr,
            flush=True,
        )
    if killed_pids:
        print(
            f"[warning] forced LIBERO worker shutdown with kill: pids={killed_pids}",
            file=sys.stderr,
            flush=True,
        )

    surviving_pids = [
        int(process.pid)
        for process in processes
        if process.pid is not None and _process_is_alive(process)
    ]
    if surviving_pids:
        raise UnrecoverableSimulationShutdownError(
            "LIBERO workers remained alive after close, terminate, and kill "
            f"(worker_pids={surviving_pids}, num_envs={len(workers)}, "
            f"MUJOCO_GL={os.environ.get('MUJOCO_GL')!r})",
            worker_pids=surviving_pids,
        )


def _bounded_subprocess_vector_environment_class(base: type[Any]) -> type[Any]:
    class BoundedSubprocVectorEnv(base):
        def close(self) -> None:
            _close_libero_subprocess_vector_environment(self)

    BoundedSubprocVectorEnv.__name__ = f"Bounded{base.__name__}"
    BoundedSubprocVectorEnv.__qualname__ = BoundedSubprocVectorEnv.__name__
    return BoundedSubprocVectorEnv


def _libero_environment_classes() -> tuple[type[Any], type[Any]]:
    validate_simulation_dependencies()
    _install_libero_gymnasium_compatibility()
    _install_robosuite_mujoco_310_compatibility()
    try:
        from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
        from libero.libero.envs import venv as libero_venv
    except ImportError as error:
        raise EvaluationError(
            "Could not import the pinned LIBERO robosuite environment stack; "
            "install requirements-octo-libero-eval.txt and provide the official "
            f"LIBERO checkout at commit {LIBERO_COMMIT}"
        ) from error
    _install_libero_subprocess_worker_bootstrap(libero_venv)
    return OffScreenRenderEnv, _bounded_subprocess_vector_environment_class(SubprocVectorEnv)


def _make_offscreen_environment(environment_kwargs: dict[str, Any]) -> Any:
    offscreen_environment, _ = _libero_environment_classes()
    environment = offscreen_environment(**environment_kwargs)
    environment._octo_startup_failure = None
    return environment


def _make_subprocess_offscreen_environment(
    environment_kwargs: dict[str, Any],
) -> Any:
    try:
        return _make_offscreen_environment(environment_kwargs)
    except Exception as error:
        return _FailedOffscreenEnvironment(
            SimulationStartupFailure(
                pid=os.getpid(),
                exception_type=type(error).__name__,
                message=str(error),
                mujoco_gl=os.environ.get("MUJOCO_GL"),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                mujoco_egl_device_id=os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            )
        )


def _format_simulation_startup_error(
    failures: Sequence[tuple[int, SimulationStartupFailure]],
    *,
    num_envs: int,
) -> str:
    _, first = failures[0]
    failed_workers = ", ".join(f"{index}(pid={failure.pid})" for index, failure in failures)
    root_message = first.message.rstrip(".")
    return (
        "LIBERO offscreen worker startup failed "
        f"(workers={failed_workers}, num_envs={num_envs}, "
        f"MUJOCO_GL={first.mujoco_gl!r}): "
        f"{first.exception_type}: {root_message}. "
        "Check the NVIDIA driver and EGL availability (for example, nvidia-smi), "
        f"CUDA_VISIBLE_DEVICES={first.cuda_visible_devices!r}, and "
        f"MUJOCO_EGL_DEVICE_ID={first.mujoco_egl_device_id!r}. "
        "Retry with a smaller --num-envs value to test GPU rendering resource "
        "pressure."
    )


def _validate_vector_environment_startup(
    environment: Any,
    *,
    num_envs: int,
) -> None:
    try:
        startup_results = environment.get_env_attr("_octo_startup_failure")
    except (BrokenPipeError, ConnectionError, EOFError, OSError) as error:
        raise SimulationInfrastructureError(
            "A LIBERO offscreen worker exited before completing its startup "
            f"health check (num_envs={num_envs}, MUJOCO_GL="
            f"{os.environ.get('MUJOCO_GL')!r}): {type(error).__name__}: {error}. "
            "Check the worker traceback, NVIDIA driver/EGL availability, "
            "CUDA_VISIBLE_DEVICES, and MUJOCO_EGL_DEVICE_ID. Use --num-envs to "
            "test a smaller explicit value."
        ) from error
    failures = [
        (index, result)
        for index, result in enumerate(startup_results)
        if isinstance(result, SimulationStartupFailure)
    ]
    if failures:
        raise SimulationInfrastructureError(
            _format_simulation_startup_error(failures, num_envs=num_envs)
        )


def make_vector_environment(task: LiberoTask, num_envs: int) -> Any:
    _, vector_environment = _libero_environment_classes()

    kwargs = {
        "bddl_file_name": str(task.bddl_file),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    factories = [partial(_make_subprocess_offscreen_environment, kwargs) for _ in range(num_envs)]
    environment = vector_environment(factories)
    try:
        _validate_vector_environment_startup(environment, num_envs=num_envs)
    except Exception as startup_error:
        try:
            environment.close()
        except UnrecoverableSimulationShutdownError as shutdown_error:
            raise shutdown_error from startup_error
        except Exception:
            pass
        raise
    return environment


def make_vector_environment_with_backoff(
    task: LiberoTask,
    num_envs: int,
    *,
    auto_reduce: bool,
) -> tuple[Any, int]:
    candidate = num_envs
    while True:
        try:
            return make_vector_environment(task, candidate), candidate
        except UnrecoverableSimulationShutdownError:
            raise
        except SimulationInfrastructureError as error:
            if not auto_reduce or candidate == 1:
                raise
            reduced = max(1, candidate // 2)
            print(
                "[warning] LIBERO offscreen startup failed at "
                f"num_envs={candidate}; retrying with num_envs={reduced}: {error}",
                file=sys.stderr,
                flush=True,
            )
            candidate = reduced


def settle_vector_environment(
    environment: Any,
    *,
    init_states: Any,
    statistics: NormalizationStatistics,
    seed: int,
    settle_steps: int,
) -> Any:
    environment.reset()
    environment.seed(int(seed))
    observations = environment.set_init_state(init_states)
    batch_size = len(observation_batch_to_list(observations))
    dummy = np.zeros((batch_size, ACTION_DIM), dtype=np.float32)
    dummy[:, -1] = 1.0
    dummy = statistics.actions_to_environment(dummy)
    for _ in range(settle_steps):
        observations = environment.step(dummy)[0]
    return observations


def rollout_action_chunks(
    environment: Any,
    observations: Any,
    *,
    predictor: Callable[[Any], np.ndarray],
    episode_ids: Sequence[int],
    init_state_ids: Sequence[int],
    seed: int,
    max_steps: int,
    action_horizon: int = ACTION_HORIZON,
    frame_callback: Callable[[Any, int], None] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    if max_steps <= 0 or action_horizon <= 0:
        raise EvaluationError("max_steps and action_horizon must be positive")
    batch_size = len(observation_batch_to_list(observations))
    if len(episode_ids) != batch_size:
        raise EvaluationError("episode_ids length must match the environment batch")
    if len(init_state_ids) != batch_size:
        raise EvaluationError("init_state_ids length must match the environment batch")
    success_mask = np.zeros(batch_size, dtype=bool)
    first_success_step = np.full(batch_size, -1, dtype=np.int64)
    executed_steps = 0
    if frame_callback is not None:
        frame_callback(observations, executed_steps)

    while executed_steps < max_steps and not np.all(success_mask):
        actions = np.asarray(predictor(observations), dtype=np.float32)
        expected = (batch_size, action_horizon, ACTION_DIM)
        if actions.shape != expected:
            raise EvaluationError(
                f"Policy returned action shape {actions.shape}; expected {expected}"
            )
        if not np.all(np.isfinite(actions)):
            raise EvaluationError("Policy returned NaN or infinite actions")
        chunk_steps = min(action_horizon, max_steps - executed_steps)
        for offset in range(chunk_steps):
            result = environment.step(actions[:, offset])
            if not isinstance(result, tuple) or len(result) not in {4, 5}:
                raise EvaluationError("LIBERO environment returned an invalid step tuple")
            observations = result[0]
            executed_steps += 1
            successes = np.asarray(environment.check_success(), dtype=bool)
            if successes.shape != (batch_size,):
                raise EvaluationError(
                    f"LIBERO success shape is {successes.shape}; expected {(batch_size,)}"
                )
            newly_successful = successes & ~success_mask
            first_success_step[newly_successful] = executed_steps
            success_mask |= successes
            if frame_callback is not None:
                frame_callback(observations, executed_steps)
            if np.all(success_mask):
                break

    episodes = []
    for index, (episode_id, init_state_id) in enumerate(
        zip(episode_ids, init_state_ids, strict=True)
    ):
        success = bool(success_mask[index])
        episodes.append(
            {
                "episode_id": int(episode_id),
                "init_state_id": int(init_state_id),
                "seed": int(seed),
                "success": success,
                "steps": (int(first_success_step[index]) if success else int(executed_steps)),
                "first_success_step": (int(first_success_step[index]) if success else None),
                "termination": "success" if success else "max_steps",
            }
        )
    return episodes, observations


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "transformers",
        "safetensors",
        "numpy",
        "Pillow",
        "mujoco",
        "robosuite",
        "bddl",
        "gymnasium",
        "cloudpickle",
        "easydict",
        "future",
        "matplotlib",
        "termcolor",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def validate_settings(settings: EvaluationSettings) -> None:
    if settings.episodes <= 0:
        raise EvaluationError("episodes must be positive")
    if settings.seeds != EVALUATION_SEEDS:
        raise EvaluationError(f"seeds must be the fixed evaluation seeds {EVALUATION_SEEDS}")
    if settings.episodes % len(settings.seeds) != 0:
        raise EvaluationError(f"episodes must be divisible by {len(settings.seeds)} fixed seeds")
    if settings.num_envs <= 0 or settings.num_envs > settings.episodes:
        raise EvaluationError("num_envs must be in [1, episodes]")
    if settings.max_steps <= 0:
        raise EvaluationError("max_steps must be positive")
    if settings.settle_steps < 0:
        raise EvaluationError("settle_steps must be non-negative")
    if settings.record_videos < 0 or settings.record_videos > settings.episodes:
        raise EvaluationError("record_videos must be in [0, episodes]")
    if settings.video_fps <= 0:
        raise EvaluationError("video_fps must be positive")


def _validate_video_settings(settings: EvaluationSettings) -> None:
    if settings.save_videos_path is None:
        if settings.record_videos != 0:
            raise EvaluationError("record_videos requires save_videos_path")
        return
    if settings.record_videos <= 0:
        raise EvaluationError("record_videos must be positive when save_videos_path is set")


def _check_output_targets(settings: EvaluationSettings) -> None:
    existing = [
        settings.output_dir / "results.json",
        settings.output_dir / "episodes.jsonl",
    ]
    if not settings.overwrite and any(path.exists() for path in existing):
        raise EvaluationError(
            f"Evaluation output already exists in {settings.output_dir}; use --overwrite"
        )


def _check_video_targets(settings: EvaluationSettings) -> None:
    if settings.save_videos_path is None or settings.overwrite:
        return
    existing = [
        path
        for outcome in ("success", "failure")
        for path in (settings.save_videos_path / outcome).glob("episode-*.mp4")
    ]
    if existing:
        raise EvaluationError(
            f"Video output already exists in {settings.save_videos_path}; use --overwrite"
        )


def configure_transformers_offline(output_dir: Path) -> Path:
    cache = (output_dir / ".hf-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = (output_dir / ".matplotlib-cache").resolve()
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    numba_cache = (output_dir / ".numba-cache").resolve()
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    return cache


def _frame_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    if "agentview_image" not in observation:
        raise EvaluationError("LIBERO observation has no agentview_image for video")
    return resize_flipped_rgb(observation["agentview_image"], (256, 256))


def _save_videos(
    output_dir: Path,
    frames: Mapping[int, Sequence[np.ndarray]],
    *,
    fps: int,
) -> list[str]:
    if not frames:
        return []
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise EvaluationError("imageio and imageio-ffmpeg are required to record videos") from error
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for episode_id, episode_frames in sorted(frames.items()):
        target = video_dir / f"episode-{episode_id:03d}.mp4"
        with imageio.get_writer(target, fps=fps) as writer:
            for frame in episode_frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        paths.append(str(target))
    return paths


def _select_video_episodes(
    episodes: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    if limit <= 0:
        raise EvaluationError("video selection limit must be positive")
    selected: list[Mapping[str, Any]] = []
    counts = {True: 0, False: 0}
    for episode in sorted(episodes, key=lambda item: int(item["episode_id"])):
        outcome = bool(episode["success"])
        if counts[outcome] >= limit:
            continue
        selected.append(episode)
        counts[outcome] += 1
    return selected


def _video_action_limit(episode: Mapping[str, Any]) -> int:
    return int(episode["steps"])


def _replay_episode_frames(
    environment: Any,
    observations: Any,
    actions: Sequence[np.ndarray],
    *,
    expected_success: bool,
) -> Iterator[np.ndarray]:
    values = observation_batch_to_list(observations)
    if len(values) != 1:
        raise EvaluationError("Video replay requires exactly one environment")
    yield _frame_from_observation(values[0])

    for step, action in enumerate(actions, start=1):
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (ACTION_DIM,):
            raise EvaluationError(
                f"Video replay action shape is {action_array.shape}; expected {(ACTION_DIM,)}"
            )
        result = environment.step(action_array[None, :])
        if not isinstance(result, tuple) or len(result) not in {4, 5}:
            raise EvaluationError("LIBERO video replay returned an invalid step tuple")
        observations = result[0]
        successes = np.asarray(environment.check_success(), dtype=bool)
        if successes.shape != (1,):
            raise EvaluationError(
                f"LIBERO video replay success shape is {successes.shape}; expected {(1,)}"
            )
        succeeded = bool(successes[0])
        values = observation_batch_to_list(observations)
        if len(values) != 1:
            raise EvaluationError("Video replay requires exactly one observation")
        if succeeded and not expected_success:
            raise EvaluationError(f"Failed episode reproduced success at step {step}")
        yield _frame_from_observation(values[0])
        if succeeded:
            return

    if expected_success:
        raise EvaluationError(
            f"Successful episode did not reproduce success after {len(actions)} steps"
        )


def _write_video(path: Path, frames: Iterable[np.ndarray], *, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise EvaluationError("imageio and imageio-ffmpeg are required to record videos") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(f"Could not encode LIBERO video {path}: {error}") from error


def _record_replay_videos(
    settings: EvaluationSettings,
    *,
    task: LiberoTask,
    statistics: NormalizationStatistics,
    episodes: Sequence[Mapping[str, Any]],
    action_trajectories: Mapping[int, Sequence[np.ndarray]],
) -> list[str]:
    if settings.save_videos_path is None:
        return []
    _validate_video_settings(settings)
    _check_video_targets(settings)
    selected = _select_video_episodes(episodes, limit=settings.record_videos)
    video_root = settings.save_videos_path
    video_root.parent.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []

    with tempfile.TemporaryDirectory(
        prefix=f".{video_root.name or 'octo-videos'}-",
        dir=video_root.parent,
    ) as temporary_directory:
        staging_root = Path(temporary_directory)
        environment = None
        try:
            if selected:
                environment, _ = make_vector_environment_with_backoff(
                    task,
                    1,
                    auto_reduce=settings.auto_reduce_num_envs,
                )
            for episode in selected:
                episode_id = int(episode["episode_id"])
                init_state_id = int(episode["init_state_id"])
                replay_steps = _video_action_limit(episode)
                trajectory = list(action_trajectories.get(episode_id, ()))
                if len(trajectory) < replay_steps:
                    raise EvaluationError(
                        f"Episode {episode_id} has {len(trajectory)} recorded actions; "
                        f"video replay requires {replay_steps}"
                    )
                observations = settle_vector_environment(
                    environment,
                    init_states=task.init_states[[init_state_id]],
                    statistics=statistics,
                    seed=int(episode["seed"]) + init_state_id,
                    settle_steps=settings.settle_steps,
                )
                outcome = "success" if bool(episode["success"]) else "failure"
                staged_path = staging_root / outcome / f"episode-{episode_id:03d}.mp4"
                _write_video(
                    staged_path,
                    _replay_episode_frames(
                        environment,
                        observations,
                        trajectory[:replay_steps],
                        expected_success=bool(episode["success"]),
                    ),
                    fps=settings.video_fps,
                )
                final_paths.append(video_root / outcome / f"episode-{episode_id:03d}.mp4")
        finally:
            if environment is not None:
                environment.close()

        for outcome in ("success", "failure"):
            target_directory = video_root / outcome
            target_directory.mkdir(parents=True, exist_ok=True)
            if settings.overwrite:
                for existing in target_directory.glob("episode-*.mp4"):
                    existing.unlink()
            staged_directory = staging_root / outcome
            if staged_directory.is_dir():
                for staged_path in sorted(staged_directory.glob("episode-*.mp4")):
                    staged_path.replace(target_directory / staged_path.name)

    return [str(path) for path in final_paths]


def _make_report(
    *,
    status: str,
    settings: EvaluationSettings,
    checkpoint: CheckpointSpec,
    statistics: NormalizationStatistics,
    task: LiberoTask,
    episodes: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    videos: Sequence[str],
    libero_commit: str,
    environment_batch_sizes: Sequence[int] = (),
    error: str | None = None,
) -> dict[str, Any]:
    successes = sum(bool(episode["success"]) for episode in episodes)
    summaries_by_seed = []
    for seed in settings.seeds:
        seeded_episodes = [episode for episode in episodes if episode.get("seed") == seed]
        seeded_successes = sum(bool(episode["success"]) for episode in seeded_episodes)
        summaries_by_seed.append(
            {
                "seed": seed,
                "completed_episodes": len(seeded_episodes),
                "successes": seeded_successes,
                "failures": len(seeded_episodes) - seeded_successes,
                "success_rate": (
                    seeded_successes / len(seeded_episodes) if seeded_episodes else None
                ),
            }
        )
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "route": "octo-small-pytorch-libero-checkpoint-eval",
        "checkpoint": checkpoint.as_dict(),
        "statistics": {
            "path": str(statistics.path),
            "sha256": statistics.sha256,
        },
        "task": {
            "suite": "libero_10",
            "task_id": task.task_id,
            "name": task.name,
            "language": task.language,
            "bddl_file": str(task.bddl_file),
            "init_states_file": str(task.init_states_file),
            "init_states_sha256": task.init_states_sha256,
            "init_states_git_blob": task.init_states_git_blob,
        },
        "protocol": {
            "episodes": settings.episodes,
            "episodes_per_seed": settings.episodes // len(settings.seeds),
            "seeds": list(settings.seeds),
            "num_envs": settings.num_envs,
            "auto_reduce_num_envs": settings.auto_reduce_num_envs,
            "max_steps": settings.max_steps,
            "settle_steps": settings.settle_steps,
            "action_horizon": ACTION_HORIZON,
        },
        "summary": {
            "completed_episodes": len(episodes),
            "successes": successes,
            "failures": len(episodes) - successes,
            "success_rate": successes / len(episodes) if episodes else None,
            "by_seed": summaries_by_seed,
        },
        "runtime": {
            "device": settings.device,
            "precision": settings.precision,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "libero_expected_commit": LIBERO_COMMIT,
            "libero_commit": libero_commit,
            "mujoco_expected_version": MUJOCO_VERSION,
            "simulation_compatibility": MUJOCO_COMPATIBILITY,
            "libero_multiprocessing_start_method": (_multiprocessing_start_method()),
            "effective_num_envs": (
                max(environment_batch_sizes) if environment_batch_sizes else None
            ),
            "environment_batch_sizes": list(environment_batch_sizes),
            "elapsed_seconds": elapsed_seconds,
        },
        "videos": list(videos),
    }
    if error is not None:
        report["error"] = error
    return report


def run_preflight(
    settings: EvaluationSettings,
    *,
    checkpoint: CheckpointSpec,
    statistics: NormalizationStatistics,
    task: LiberoTask,
    policy: LoadedCheckpointPolicy,
    libero_commit: str,
) -> dict[str, Any]:
    try:
        environment = _make_offscreen_environment(
            {
                "bddl_file_name": str(task.bddl_file),
                "camera_heights": 128,
                "camera_widths": 128,
            }
        )
    except Exception as error:
        raise SimulationInfrastructureError(
            "Could not create the LIBERO offscreen environment "
            f"(num_envs=1, MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}): "
            f"{type(error).__name__}: {error}. Check the NVIDIA driver and EGL "
            "availability (for example, nvidia-smi), CUDA_VISIBLE_DEVICES, and "
            "MUJOCO_EGL_DEVICE_ID. The evaluator did not change the rendering "
            "backend or reduce parallelism automatically."
        ) from error
    try:
        environment.reset()
        environment.seed(settings.seeds[0])
        observation = environment.set_init_state(task.init_states[0])
        dummy = np.zeros((1, ACTION_DIM), dtype=np.float32)
        dummy[:, -1] = 1.0
        raw_dummy = statistics.actions_to_environment(dummy)[0]
        for _ in range(settings.settle_steps):
            observation = environment.step(raw_dummy)[0]
        generator = policy.make_generator(settings.seeds[0])
        actions = policy.predict_action_chunk(
            [observation],
            task.language,
            generator=generator,
        )
    finally:
        environment.close()
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "passed",
        "checkpoint": checkpoint.as_dict(),
        "statistics": {
            "path": str(statistics.path),
            "sha256": statistics.sha256,
        },
        "task": {
            "task_id": task.task_id,
            "name": task.name,
            "init_states": len(task.init_states),
            "init_states_file": str(task.init_states_file),
            "init_states_sha256": task.init_states_sha256,
            "init_states_git_blob": task.init_states_git_blob,
        },
        "protocol": {
            "seeds": list(settings.seeds),
        },
        "sample_action_shape": list(actions.shape),
        "runtime": {
            "device": settings.device,
            "precision": settings.precision,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "packages": _package_versions(),
            "libero_expected_commit": LIBERO_COMMIT,
            "libero_commit": libero_commit,
            "mujoco_expected_version": MUJOCO_VERSION,
            "simulation_compatibility": MUJOCO_COMPATIBILITY,
            "libero_multiprocessing_start_method": _multiprocessing_start_method(),
        },
    }
    _atomic_write_json(settings.output_dir / "preflight.json", report)
    return report


def evaluate_checkpoint(
    settings: EvaluationSettings,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    configure_evaluation_multiprocessing()
    validate_settings(settings)
    _validate_video_settings(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    configure_transformers_offline(settings.output_dir)
    validate_simulation_dependencies()
    checkpoint = resolve_checkpoint(
        settings.checkpoint,
        base_model=settings.base_model,
    )
    statistics = load_statistics(settings.statistics)
    datasets_path = statistics.path.parent.parent.parent
    libero_root, libero_config_file, libero_commit = configure_libero(
        output_dir=settings.output_dir,
        datasets_path=datasets_path,
        source_root=settings.libero_root,
        config_path=settings.libero_config_path,
    )
    task = resolve_libero_task(settings.task_name, libero_root=libero_root)
    episodes_per_seed = settings.episodes // len(settings.seeds)
    if episodes_per_seed > len(task.init_states):
        raise EvaluationError(
            f"Requested {episodes_per_seed} episodes per seed but task has "
            f"{len(task.init_states)} fixed initial states; the maximum total "
            f"for {len(settings.seeds)} seeds is "
            f"{len(task.init_states) * len(settings.seeds)}"
        )
    policy = load_checkpoint_policy(
        checkpoint,
        statistics=statistics,
        device=settings.device,
        precision=settings.precision,
    )
    if preflight_only:
        report = run_preflight(
            settings,
            checkpoint=checkpoint,
            statistics=statistics,
            task=task,
            policy=policy,
            libero_commit=libero_commit,
        )
        report["libero_config_file"] = str(libero_config_file)
        _atomic_write_json(settings.output_dir / "preflight.json", report)
        return report

    _check_output_targets(settings)
    _check_video_targets(settings)
    started = time.monotonic()
    episodes: list[dict[str, Any]] = []
    action_trajectories: dict[int, list[np.ndarray]] = {}
    record_actions = settings.save_videos_path is not None
    videos: list[str] = []
    environment = None
    environment_batch_size = 0
    environment_batch_sizes: list[int] = []
    active_num_envs = settings.num_envs
    try:
        for seed_index, seed in enumerate(settings.seeds):
            generator = policy.make_generator(seed)
            seed_start = 0
            while seed_start < episodes_per_seed:
                batch_size = min(active_num_envs, episodes_per_seed - seed_start)
                if environment is None or environment_batch_size != batch_size:
                    if environment is not None:
                        environment.close()
                        environment = None
                        environment_batch_size = 0
                    environment, batch_size = make_vector_environment_with_backoff(
                        task,
                        batch_size,
                        auto_reduce=settings.auto_reduce_num_envs,
                    )
                    active_num_envs = min(active_num_envs, batch_size)
                    environment_batch_size = batch_size
                environment_batch_sizes.append(batch_size)
                init_state_ids = list(range(seed_start, seed_start + batch_size))
                episode_start = seed_index * episodes_per_seed + seed_start
                episode_ids = list(range(episode_start, episode_start + batch_size))
                observations = settle_vector_environment(
                    environment,
                    init_states=task.init_states[init_state_ids],
                    statistics=statistics,
                    seed=seed + seed_start,
                    settle_steps=settings.settle_steps,
                )

                if record_actions:
                    for episode_id in episode_ids:
                        action_trajectories[episode_id] = []

                def predict_actions(current: Any) -> np.ndarray:
                    actions = policy.predict_action_chunk(
                        current,
                        task.language,
                        generator=generator,
                    )
                    if record_actions:
                        for local_index, episode_id in enumerate(episode_ids):
                            action_trajectories[episode_id].extend(
                                np.asarray(actions[local_index], dtype=np.float32).copy()
                            )
                    return actions

                group_episodes, _ = rollout_action_chunks(
                    environment,
                    observations,
                    predictor=predict_actions,
                    episode_ids=episode_ids,
                    init_state_ids=init_state_ids,
                    seed=seed,
                    max_steps=settings.max_steps,
                    action_horizon=ACTION_HORIZON,
                )
                episodes.extend(group_episodes)
                _atomic_write_jsonl(
                    settings.output_dir / "episodes.partial.jsonl",
                    episodes,
                )
                seed_start += batch_size
        if environment is not None:
            environment.close()
            environment = None
            environment_batch_size = 0
        videos = _record_replay_videos(
            settings,
            task=task,
            statistics=statistics,
            episodes=episodes,
            action_trajectories=action_trajectories,
        )
    except Exception as error:
        elapsed = time.monotonic() - started
        partial = _make_report(
            status="failed",
            settings=settings,
            checkpoint=checkpoint,
            statistics=statistics,
            task=task,
            episodes=episodes,
            elapsed_seconds=elapsed,
            videos=videos,
            libero_commit=libero_commit,
            environment_batch_sizes=environment_batch_sizes,
            error=f"{type(error).__name__}: {error}",
        )
        _atomic_write_json(settings.output_dir / "failure.json", partial)
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except UnrecoverableSimulationShutdownError as error:
                elapsed = time.monotonic() - started
                partial = _make_report(
                    status="failed",
                    settings=settings,
                    checkpoint=checkpoint,
                    statistics=statistics,
                    task=task,
                    episodes=episodes,
                    elapsed_seconds=elapsed,
                    videos=videos,
                    libero_commit=libero_commit,
                    environment_batch_sizes=environment_batch_sizes,
                    error=f"{type(error).__name__}: {error}",
                )
                _atomic_write_json(settings.output_dir / "failure.json", partial)
                raise

    elapsed = time.monotonic() - started
    report = _make_report(
        status="complete",
        settings=settings,
        checkpoint=checkpoint,
        statistics=statistics,
        task=task,
        episodes=episodes,
        elapsed_seconds=elapsed,
        videos=videos,
        libero_commit=libero_commit,
        environment_batch_sizes=environment_batch_sizes,
    )
    _atomic_write_jsonl(settings.output_dir / "episodes.jsonl", episodes)
    _atomic_write_json(settings.output_dir / "results.json", report)
    partial_path = settings.output_dir / "episodes.partial.jsonl"
    if partial_path.exists():
        partial_path.unlink()
    return report


def settings_as_dict(settings: EvaluationSettings) -> dict[str, Any]:
    value = asdict(settings)
    return {key: str(item) if isinstance(item, Path) else item for key, item in value.items()}
