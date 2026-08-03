"""Regression analysis for TDUS weight sweeps evaluated on LIBERO-10."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np


COMPONENTS = ("quality", "coverage", "diversity", "novelty")
RunStatus = Literal["complete", "partial", "not_started"]
ModelOrder = Literal["linear", "quadratic"]
RequestedModelOrder = Literal["auto", "linear", "quadratic"]
DEFAULT_ALPHAS = tuple([0.0, *np.logspace(-6, 3, 37).tolist()])
FEATURE_NAMES: dict[ModelOrder, tuple[str, ...]] = {
    "linear": ("quality", "coverage", "diversity"),
    "quadratic": (
        "quality",
        "coverage",
        "diversity",
        "quality^2",
        "coverage^2",
        "diversity^2",
        "quality*coverage",
        "quality*diversity",
        "coverage*diversity",
    ),
}


class RegressionError(ValueError):
    """Raised when TDUS regression inputs are missing or incompatible."""


@dataclass(frozen=True)
class WeightCandidate:
    """One JSONL weight row and its deterministic sweep model identifier."""

    line_number: int
    model_id: str
    weights: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(COMPONENTS, self.weights, strict=True))


@dataclass(frozen=True)
class RunObservation:
    """Evaluation state and optional complete ten-task observation."""

    model_id: str
    line_number: int
    weights: tuple[float, float, float, float]
    status: RunStatus
    task_rates: tuple[float, ...] | None
    task_successes: tuple[int, ...] | None
    task_episodes: tuple[int, ...] | None
    score: float | None
    exclusion_reason: str | None


@dataclass(frozen=True)
class ExperimentData:
    """Validated baseline, candidate inventory, and complete regression rows."""

    results_root: Path
    weights_path: Path
    weights_sha256: str
    sweep_manifest_path: Path | None
    sweep_manifest_sha256: str | None
    baseline_name: str
    baseline_rates: tuple[float, ...]
    baseline_successes: tuple[int, ...]
    baseline_episodes: tuple[int, ...]
    baseline_score: float
    clip_max: float
    candidates: tuple[WeightCandidate, ...]
    runs: Mapping[str, RunObservation]

    @property
    def complete_runs(self) -> tuple[RunObservation, ...]:
        return tuple(
            self.runs[candidate.model_id]
            for candidate in self.candidates
            if self.runs[candidate.model_id].status == "complete"
        )


@dataclass(frozen=True)
class _TaskResult:
    task_id: int
    rate: float
    successes: int
    episodes: int
    signature: Mapping[str, Any]


@dataclass(frozen=True)
class ValidationMetrics:
    """Prediction error metrics for one validation path."""

    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class ModelSelectionResult:
    """Best leave-one-out regularization result for one polynomial order."""

    order: ModelOrder
    alpha: float
    validation: ValidationMetrics


@dataclass(frozen=True)
class RegressionFit:
    """Selected ridge response surface in standardized and raw coordinates."""

    order: ModelOrder
    alpha: float
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    standardized_coefficients: tuple[float, ...]
    raw_coefficients: Mapping[str, float]
    equation: str
    training_predictions: tuple[float, ...]
    residuals: tuple[float, ...]
    nested_predictions: tuple[float, ...]
    nested_validation: ValidationMetrics
    nested_selected_orders: tuple[ModelOrder, ...]
    nested_selected_alphas: tuple[float, ...]
    selection_results: tuple[ModelSelectionResult, ...]

    def predict(self, weights: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        values = _validate_weight_matrix(weights)
        features = _feature_matrix(values, self.order)
        raw = np.array(
            [self.raw_coefficients[name] for name in self.feature_names],
            dtype=np.float64,
        )
        return float(self.raw_coefficients["intercept"]) + features @ raw


@dataclass(frozen=True)
class OptimizationResult:
    """Maximum of a fitted response surface over a bounded weight simplex."""

    weights: tuple[float, float, float, float]
    predicted_score: float
    active_constraints: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegressionError(f"could not read {description} {path}: {error}") from error


def load_weight_candidates(path: str | Path) -> tuple[WeightCandidate, ...]:
    """Load valid non-negative unit-sum TDUS weights from JSONL."""

    weights_path = Path(path).expanduser().resolve()
    try:
        lines = weights_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RegressionError(f"could not read weights file {weights_path}: {error}") from error
    if not lines:
        raise RegressionError(f"weights file is empty: {weights_path}")

    expected = set(COMPONENTS)
    candidates: list[WeightCandidate] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RegressionError(f"weights line {line_number} is empty")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise RegressionError(
                f"weights line {line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(raw, dict) or set(raw) != expected:
            raise RegressionError(
                f"weights line {line_number} must contain exactly {list(COMPONENTS)}"
            )
        values: list[float] = []
        for component in COMPONENTS:
            value = raw[component]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RegressionError(
                    f"weights line {line_number} field {component!r} must be numeric"
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise RegressionError(
                    f"weights line {line_number} field {component!r} "
                    "must be finite and non-negative"
                )
            values.append(number)
        if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise RegressionError(
                f"weights line {line_number} must sum to 1, got {sum(values):.12g}"
            )
        candidates.append(
            WeightCandidate(
                line_number=line_number,
                model_id=f"model-{line_number:03d}",
                weights=tuple(values),  # type: ignore[arg-type]
            )
        )
    return tuple(candidates)


def _validate_sweep_manifest(
    path: str | Path,
    candidates: tuple[WeightCandidate, ...],
    weights_sha256: str,
) -> tuple[Path, str]:
    manifest_path = Path(path).expanduser().resolve()
    raw = _load_json(manifest_path, "sweep manifest")
    manifest = _require_mapping(raw, manifest_path, "root")
    declared_sha256 = manifest.get("weights_sha256")
    if declared_sha256 != weights_sha256:
        raise RegressionError(
            f"sweep manifest {manifest_path}: weights_sha256={declared_sha256!r} "
            f"does not match weights file {weights_sha256!r}"
        )
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise RegressionError(f"sweep manifest {manifest_path}: jobs must be a list")
    if len(jobs) != len(candidates):
        raise RegressionError(
            f"sweep manifest {manifest_path}: expected {len(candidates)} jobs, found {len(jobs)}"
        )
    for candidate, raw_job in zip(candidates, jobs, strict=True):
        job = _require_mapping(raw_job, manifest_path, f"jobs[{candidate.line_number - 1}]")
        if job.get("line_number") != candidate.line_number:
            raise RegressionError(
                f"sweep manifest {manifest_path}: {candidate.model_id} line_number mismatch"
            )
        if job.get("model_id") != candidate.model_id:
            raise RegressionError(
                f"sweep manifest {manifest_path}: expected {candidate.model_id}, "
                f"found {job.get('model_id')!r}"
            )
        raw_weights = job.get("weights")
        if not isinstance(raw_weights, dict) or set(raw_weights) != set(COMPONENTS):
            raise RegressionError(
                f"sweep manifest {manifest_path}: {candidate.model_id} weights are invalid"
            )
        manifest_weights = tuple(float(raw_weights[name]) for name in COMPONENTS)
        if any(
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in zip(manifest_weights, candidate.weights, strict=True)
        ):
            raise RegressionError(
                f"sweep manifest {manifest_path}: {candidate.model_id} weights "
                f"{manifest_weights!r} do not match {candidate.weights!r}"
            )
    return manifest_path, _sha256(manifest_path)


def _require_mapping(value: Any, path: Path, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RegressionError(f"{path}: field {field} must be an object")
    return value


def _read_complete_task(path: Path, expected_task_id: int) -> _TaskResult | None:
    if not path.is_file():
        return None
    raw = _load_json(path, "evaluation result")
    result = _require_mapping(raw, path, "root")
    if result.get("status") != "complete":
        return None
    task = _require_mapping(result.get("task"), path, "task")
    protocol = _require_mapping(result.get("protocol"), path, "protocol")
    summary = _require_mapping(result.get("summary"), path, "summary")
    statistics = _require_mapping(result.get("statistics"), path, "statistics")

    task_id = task.get("task_id")
    if task_id != expected_task_id:
        raise RegressionError(f"{path}: task_id={task_id!r} does not match task-{expected_task_id}")
    episodes = protocol.get("episodes")
    completed = summary.get("completed_episodes")
    successes = summary.get("successes")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise RegressionError(f"{path}: protocol.episodes must be a positive integer")
    if completed != episodes:
        return None
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise RegressionError(f"{path}: summary.successes must be an integer")
    if not 0 <= successes <= completed:
        raise RegressionError(f"{path}: summary.successes={successes} is outside [0, {completed}]")
    expected_rate = successes / completed
    rate = summary.get("success_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise RegressionError(f"{path}: summary.success_rate must be numeric")
    rate_number = float(rate)
    if not math.isfinite(rate_number) or not math.isclose(
        rate_number, expected_rate, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RegressionError(
            f"{path}: summary.success_rate={rate_number!r} does not match "
            f"successes/completed_episodes={expected_rate!r}"
        )

    signature = {
        "schema_version": result.get("schema_version"),
        "route": result.get("route"),
        "suite": task.get("suite"),
        "task_name": task.get("name"),
        "init_states_sha256": task.get("init_states_sha256"),
        "statistics_sha256": statistics.get("sha256"),
        "episodes": episodes,
        "episodes_per_seed": protocol.get("episodes_per_seed"),
        "seeds": protocol.get("seeds"),
        "max_steps": protocol.get("max_steps"),
        "settle_steps": protocol.get("settle_steps"),
        "action_horizon": protocol.get("action_horizon"),
    }
    return _TaskResult(
        task_id=expected_task_id,
        rate=rate_number,
        successes=successes,
        episodes=episodes,
        signature=signature,
    )


def _read_run(root: Path, run_name: str, task_count: int) -> tuple[_TaskResult, ...] | None:
    tasks: list[_TaskResult] = []
    for task_id in range(task_count):
        task = _read_complete_task(
            root / run_name / f"task-{task_id}" / "results.json",
            task_id,
        )
        if task is None:
            return None
        tasks.append(task)
    return tuple(tasks)


def _compare_protocol(
    baseline: tuple[_TaskResult, ...],
    candidate: tuple[_TaskResult, ...],
    model_id: str,
) -> None:
    for baseline_task, candidate_task in zip(baseline, candidate, strict=True):
        for field, expected in baseline_task.signature.items():
            actual = candidate_task.signature.get(field)
            if actual != expected:
                raise RegressionError(
                    f"{model_id}/task-{baseline_task.task_id}: {field}={actual!r} "
                    f"does not match baseline {expected!r}"
                )


def _equal_task_score(
    task_rates: tuple[float, ...],
    baseline_rates: tuple[float, ...],
    clip_max: float,
) -> float:
    clipped = [
        min(max(rate / baseline, 0.0), clip_max)
        for rate, baseline in zip(task_rates, baseline_rates, strict=True)
    ]
    return math.fsum(clipped) / len(clipped)


def load_experiment_data(
    results_root: str | Path,
    weights_path: str | Path,
    *,
    baseline_name: str = "all",
    clip_max: float = 2.0,
    task_count: int = 10,
    sweep_manifest: str | Path | None = None,
) -> ExperimentData:
    """Load comparable complete runs and retain partial candidate status."""

    if not math.isfinite(clip_max) or clip_max < 1.0:
        raise RegressionError("clip_max must be at least 1 and finite")
    if task_count <= 0:
        raise RegressionError("task_count must be positive")
    root = Path(results_root).expanduser().resolve()
    candidates = load_weight_candidates(weights_path)
    resolved_weights = Path(weights_path).expanduser().resolve()
    weights_sha256 = _sha256(resolved_weights)
    if sweep_manifest is None:
        manifest_path = None
        manifest_sha256 = None
    else:
        manifest_path, manifest_sha256 = _validate_sweep_manifest(
            sweep_manifest,
            candidates,
            weights_sha256,
        )
    baseline = _read_run(root, baseline_name, task_count)
    if baseline is None:
        raise RegressionError(
            f"baseline {baseline_name!r} does not have {task_count} complete tasks under {root}"
        )
    baseline_rates = tuple(task.rate for task in baseline)
    for task_id, rate in enumerate(baseline_rates):
        if rate == 0.0:
            raise RegressionError(f"baseline task-{task_id} success rate is zero")

    runs: dict[str, RunObservation] = {}
    for candidate in candidates:
        run_root = root / candidate.model_id
        if not run_root.exists():
            status: RunStatus = "not_started"
            tasks = None
            exclusion = "result directory does not exist"
        else:
            tasks = _read_run(root, candidate.model_id, task_count)
            if tasks is None:
                status = "partial"
                exclusion = "not all tasks have complete episode counts"
            else:
                status = "complete"
                exclusion = None
        if tasks is not None:
            _compare_protocol(baseline, tasks, candidate.model_id)
            rates = tuple(task.rate for task in tasks)
            successes = tuple(task.successes for task in tasks)
            episodes = tuple(task.episodes for task in tasks)
            score = _equal_task_score(rates, baseline_rates, clip_max)
        else:
            rates = None
            successes = None
            episodes = None
            score = None
        runs[candidate.model_id] = RunObservation(
            model_id=candidate.model_id,
            line_number=candidate.line_number,
            weights=candidate.weights,
            status=status,
            task_rates=rates,
            task_successes=successes,
            task_episodes=episodes,
            score=score,
            exclusion_reason=exclusion,
        )

    return ExperimentData(
        results_root=root,
        weights_path=resolved_weights,
        weights_sha256=weights_sha256,
        sweep_manifest_path=manifest_path,
        sweep_manifest_sha256=manifest_sha256,
        baseline_name=baseline_name,
        baseline_rates=baseline_rates,
        baseline_successes=tuple(task.successes for task in baseline),
        baseline_episodes=tuple(task.episodes for task in baseline),
        baseline_score=_equal_task_score(baseline_rates, baseline_rates, clip_max),
        clip_max=clip_max,
        candidates=candidates,
        runs=runs,
    )


def _validate_weight_matrix(
    weights: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 4:
        raise RegressionError("weights must be a two-dimensional array with four columns")
    if not np.isfinite(values).all():
        raise RegressionError("weights must be finite")
    if np.any(values < -1e-12):
        raise RegressionError("weights must be non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9):
        raise RegressionError("every weight row must sum to 1")
    return values


def _validate_targets(targets: Sequence[float] | np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or len(values) != rows:
        raise RegressionError(f"targets must contain exactly {rows} values")
    if not np.isfinite(values).all():
        raise RegressionError("targets must be finite")
    return values


def _feature_matrix(weights: np.ndarray, order: ModelOrder) -> np.ndarray:
    quality = weights[:, 0]
    coverage = weights[:, 1]
    diversity = weights[:, 2]
    linear = [quality, coverage, diversity]
    if order == "linear":
        return np.column_stack(linear)
    return np.column_stack(
        [
            *linear,
            quality * quality,
            coverage * coverage,
            diversity * diversity,
            quality * coverage,
            quality * diversity,
            coverage * diversity,
        ]
    )


def _fit_core(
    weights: np.ndarray,
    targets: np.ndarray,
    order: ModelOrder,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = _feature_matrix(weights, order)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale <= 1e-15, 1.0, scale)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    penalty = np.diag([0.0, *([1.0] * features.shape[1])])
    normal = design.T @ design + alpha * penalty
    right = design.T @ targets
    if alpha == 0.0:
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    else:
        try:
            coefficients = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(normal, right, rcond=None)[0]
    return coefficients, mean, scale


def _predict_standardized(
    weights: np.ndarray,
    order: ModelOrder,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    features = _feature_matrix(weights, order)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return design @ coefficients


def _validation_metrics(actual: np.ndarray, predicted: np.ndarray) -> ValidationMetrics:
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors * errors)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    residual = float(np.sum(errors * errors))
    if denominator <= 1e-30:
        r2 = 1.0 if residual <= 1e-30 else 0.0
    else:
        r2 = 1.0 - residual / denominator
    return ValidationMetrics(mae=mae, rmse=rmse, r2=r2)


def _minimum_rows(order: ModelOrder, *, nested: bool = False) -> int:
    return len(FEATURE_NAMES[order]) + (3 if nested else 2)


def _eligible_orders(
    rows: int,
    requested: RequestedModelOrder,
    *,
    nested: bool = False,
) -> tuple[ModelOrder, ...]:
    desired: tuple[ModelOrder, ...]
    if requested == "auto":
        desired = ("linear", "quadratic")
    elif requested in ("linear", "quadratic"):
        desired = (requested,)
    else:
        raise RegressionError(f"unknown model_order: {requested!r}")
    eligible = tuple(order for order in desired if rows >= _minimum_rows(order, nested=nested))
    if not eligible:
        requirements = ", ".join(
            f"{order}={_minimum_rows(order, nested=nested)}" for order in desired
        )
        raise RegressionError(
            f"not enough complete runs ({rows}) for requested regression; "
            f"minimum rows: {requirements}"
        )
    return eligible


def _validate_alphas(alphas: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(alpha) for alpha in alphas)
    if not values:
        raise RegressionError("alphas must not be empty")
    if any(not math.isfinite(alpha) or alpha < 0.0 for alpha in values):
        raise RegressionError("alphas must be finite and non-negative")
    return tuple(sorted(set(values)))


def _loo_predictions(
    weights: np.ndarray,
    targets: np.ndarray,
    order: ModelOrder,
    alpha: float,
) -> np.ndarray:
    predictions = np.empty(len(targets), dtype=np.float64)
    for held_out in range(len(targets)):
        keep = np.arange(len(targets)) != held_out
        coefficients, mean, scale = _fit_core(weights[keep], targets[keep], order, alpha)
        predictions[held_out] = _predict_standardized(
            weights[[held_out]], order, coefficients, mean, scale
        )[0]
    return predictions


def _best_for_order(
    weights: np.ndarray,
    targets: np.ndarray,
    order: ModelOrder,
    alphas: tuple[float, ...],
) -> tuple[ModelSelectionResult, np.ndarray]:
    options: list[tuple[float, float, ValidationMetrics, np.ndarray]] = []
    for alpha in alphas:
        predictions = _loo_predictions(weights, targets, order, alpha)
        metrics = _validation_metrics(targets, predictions)
        options.append((metrics.rmse, alpha, metrics, predictions))
    _, alpha, metrics, predictions = min(
        options,
        key=lambda item: (round(item[0], 12), -item[1]),
    )
    return ModelSelectionResult(order=order, alpha=alpha, validation=metrics), predictions


def _select_hyperparameters(
    weights: np.ndarray,
    targets: np.ndarray,
    requested: RequestedModelOrder,
    alphas: tuple[float, ...],
) -> tuple[ModelSelectionResult, tuple[ModelSelectionResult, ...]]:
    results = tuple(
        _best_for_order(weights, targets, order, alphas)[0]
        for order in _eligible_orders(len(targets), requested)
    )
    selected = min(
        results,
        key=lambda result: (
            round(result.validation.rmse, 12),
            0 if result.order == "linear" else 1,
            -result.alpha,
        ),
    )
    return selected, results


def _nested_loo_predictions(
    weights: np.ndarray,
    targets: np.ndarray,
    requested: RequestedModelOrder,
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, tuple[ModelOrder, ...], tuple[float, ...]]:
    predictions = np.empty(len(targets), dtype=np.float64)
    orders: list[ModelOrder] = []
    selected_alphas: list[float] = []
    for held_out in range(len(targets)):
        keep = np.arange(len(targets)) != held_out
        selected, _ = _select_hyperparameters(weights[keep], targets[keep], requested, alphas)
        coefficients, mean, scale = _fit_core(
            weights[keep], targets[keep], selected.order, selected.alpha
        )
        predictions[held_out] = _predict_standardized(
            weights[[held_out]], selected.order, coefficients, mean, scale
        )[0]
        orders.append(selected.order)
        selected_alphas.append(selected.alpha)
    return predictions, tuple(orders), tuple(selected_alphas)


def _raw_coefficients(
    order: ModelOrder,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    slopes = coefficients[1:] / scale
    intercept = float(coefficients[0] - np.sum(coefficients[1:] * mean / scale))
    return {
        "intercept": intercept,
        **{name: float(value) for name, value in zip(FEATURE_NAMES[order], slopes, strict=True)},
    }


def _equation(raw: Mapping[str, float], order: ModelOrder) -> str:
    terms = [f"{raw['intercept']:.12g}"]
    for name in FEATURE_NAMES[order]:
        coefficient = float(raw[name])
        sign = "+" if coefficient >= 0.0 else "-"
        terms.append(f" {sign} {abs(coefficient):.12g}*{name}")
    return "score_hat =" + "".join(terms)


def fit_regression(
    weights: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    model_order: RequestedModelOrder = "auto",
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> RegressionFit:
    """Select and fit a linear or quadratic ridge response surface."""

    weight_values = _validate_weight_matrix(weights)
    target_values = _validate_targets(targets, len(weight_values))
    alpha_values = _validate_alphas(alphas)
    top_level_orders = _eligible_orders(len(target_values), model_order, nested=True)
    effective_order: RequestedModelOrder = (
        "auto" if len(top_level_orders) == 2 else top_level_orders[0]
    )
    selected, selection_results = _select_hyperparameters(
        weight_values, target_values, effective_order, alpha_values
    )
    nested_predictions, nested_orders, nested_alphas = _nested_loo_predictions(
        weight_values, target_values, effective_order, alpha_values
    )
    coefficients, mean, scale = _fit_core(
        weight_values, target_values, selected.order, selected.alpha
    )
    raw = _raw_coefficients(selected.order, coefficients, mean, scale)
    training_predictions = _predict_standardized(
        weight_values, selected.order, coefficients, mean, scale
    )
    return RegressionFit(
        order=selected.order,
        alpha=selected.alpha,
        feature_names=FEATURE_NAMES[selected.order],
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        standardized_coefficients=tuple(float(value) for value in coefficients),
        raw_coefficients=raw,
        equation=_equation(raw, selected.order),
        training_predictions=tuple(float(value) for value in training_predictions),
        residuals=tuple(
            float(actual - predicted)
            for actual, predicted in zip(target_values, training_predictions, strict=True)
        ),
        nested_predictions=tuple(float(value) for value in nested_predictions),
        nested_validation=_validation_metrics(target_values, nested_predictions),
        nested_selected_orders=nested_orders,
        nested_selected_alphas=nested_alphas,
        selection_results=selection_results,
    )


def fit_fixed_regression(
    weights: Sequence[Sequence[float]] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    order: ModelOrder,
    alpha: float,
) -> RegressionFit:
    """Refit an already-selected order and alpha without another CV search."""

    weight_values = _validate_weight_matrix(weights)
    target_values = _validate_targets(targets, len(weight_values))
    alpha_value = _validate_alphas((alpha,))[0]
    coefficients, mean, scale = _fit_core(weight_values, target_values, order, alpha_value)
    raw = _raw_coefficients(order, coefficients, mean, scale)
    predictions = _predict_standardized(weight_values, order, coefficients, mean, scale)
    metrics = _validation_metrics(target_values, predictions)
    return RegressionFit(
        order=order,
        alpha=alpha_value,
        feature_names=FEATURE_NAMES[order],
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        standardized_coefficients=tuple(float(value) for value in coefficients),
        raw_coefficients=raw,
        equation=_equation(raw, order),
        training_predictions=tuple(float(value) for value in predictions),
        residuals=tuple(
            float(actual - predicted)
            for actual, predicted in zip(target_values, predictions, strict=True)
        ),
        nested_predictions=(),
        nested_validation=metrics,
        nested_selected_orders=(),
        nested_selected_alphas=(),
        selection_results=(),
    )


def _quadratic_form(fitted: RegressionFit) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.array(
        [
            fitted.raw_coefficients["quality"],
            fitted.raw_coefficients["coverage"],
            fitted.raw_coefficients["diversity"],
        ],
        dtype=np.float64,
    )
    hessian = np.zeros((3, 3), dtype=np.float64)
    if fitted.order == "quadratic":
        for index, name in enumerate(("quality^2", "coverage^2", "diversity^2")):
            hessian[index, index] = 2.0 * fitted.raw_coefficients[name]
        for left, right, name in (
            (0, 1, "quality*coverage"),
            (0, 2, "quality*diversity"),
            (1, 2, "coverage*diversity"),
        ):
            hessian[left, right] = fitted.raw_coefficients[name]
            hessian[right, left] = fitted.raw_coefficients[name]
    return gradient, hessian


def optimize_bounded_simplex(
    fitted: RegressionFit,
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    reference_weights: Sequence[float] | None = None,
) -> OptimizationResult:
    """Exactly enumerate KKT candidates on a bounded four-weight simplex."""

    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if lower.shape != (4,) or upper.shape != (4,):
        raise RegressionError("lower_bounds and upper_bounds must have four values")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RegressionError("simplex bounds must be finite")
    if np.any(lower < 0.0) or np.any(upper > 1.0) or np.any(lower > upper):
        raise RegressionError("simplex bounds must satisfy 0 <= lower <= upper <= 1")
    if float(lower.sum()) > 1.0 + 1e-12 or float(upper.sum()) < 1.0 - 1e-12:
        raise RegressionError("simplex bounds do not contain any unit-sum weights")
    if reference_weights is None:
        reference = (lower + upper) / 2.0
        reference = reference / reference.sum()
    else:
        reference = np.asarray(reference_weights, dtype=np.float64)
        if reference.shape != (4,):
            raise RegressionError("reference_weights must have four values")

    rows: list[np.ndarray] = []
    limits: list[float] = []
    names: list[str] = []
    for index, component in enumerate(COMPONENTS[:3]):
        unit = np.zeros(3, dtype=np.float64)
        unit[index] = 1.0
        rows.extend([unit, -unit])
        limits.extend([float(upper[index]), float(-lower[index])])
        names.extend([f"{component}=upper", f"{component}=lower"])
    rows.extend([np.ones(3), -np.ones(3)])
    limits.extend([float(1.0 - lower[3]), float(-(1.0 - upper[3]))])
    names.extend(["novelty=lower", "novelty=upper"])
    inequality = np.vstack(rows)
    limit = np.asarray(limits)
    gradient, hessian = _quadratic_form(fitted)

    candidates: dict[tuple[float, ...], tuple[np.ndarray, tuple[str, ...]]] = {}
    for active_count in range(4):
        for active_indices in itertools.combinations(range(len(names)), active_count):
            if active_indices:
                active = inequality[list(active_indices)]
                active_limits = limit[list(active_indices)]
                if np.linalg.matrix_rank(active) != active_count:
                    continue
                kkt = np.block(
                    [
                        [hessian, active.T],
                        [active, np.zeros((active_count, active_count))],
                    ]
                )
                right = np.concatenate([-gradient, active_limits])
            else:
                kkt = hessian
                right = -gradient
            solution, _, _, _ = np.linalg.lstsq(kkt, right, rcond=None)
            if np.linalg.norm(kkt @ solution - right, ord=np.inf) > 1e-8:
                continue
            independent = solution[:3]
            if np.any(inequality @ independent - limit > 1e-9):
                continue
            weights = np.append(independent, 1.0 - independent.sum())
            if np.any(weights < lower - 1e-9) or np.any(weights > upper + 1e-9):
                continue
            weights[np.abs(weights) < 1e-12] = 0.0
            key = tuple(np.round(weights, 12))
            candidates[key] = (
                weights,
                tuple(names[index] for index in active_indices),
            )
    if not candidates:
        raise RegressionError("could not find a feasible bounded-simplex optimum")

    evaluated: list[tuple[float, float, tuple[float, ...], np.ndarray, tuple[str, ...]]] = []
    for weights, active_names in candidates.values():
        score = float(fitted.predict(weights.reshape(1, 4))[0])
        distance = float(np.sum((weights - reference) ** 2))
        evaluated.append((score, -distance, tuple(-weights), weights, active_names))
    _, _, _, best_weights, best_active = max(evaluated, key=lambda item: item[:3])
    best_score = float(fitted.predict(best_weights.reshape(1, 4))[0])
    return OptimizationResult(
        weights=tuple(float(value) for value in best_weights),  # type: ignore[arg-type]
        predicted_score=best_score,
        active_constraints=best_active,
    )
