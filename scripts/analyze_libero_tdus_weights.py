#!/usr/bin/env python3
"""Fit and report a TDUS weight response surface from complete LIBERO-10 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_value = str(import_root)
    if import_value not in sys.path:
        sys.path.insert(0, import_value)

from tdus.regression import (  # noqa: E402
    COMPONENTS,
    DEFAULT_ALPHAS,
    ExperimentData,
    RegressionError,
    RegressionFit,
    fit_fixed_regression,
    fit_regression,
    load_experiment_data,
    optimize_bounded_simplex,
)


DEFAULT_RESULTS_ROOT = Path("/data/dwb/octo_small_libero")
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tdus_weight_regression"
SCHEMA_VERSION = 1


def _sha256_files(root: Path, relative_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths, key=lambda path: path.as_posix()):
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _result_fingerprint(dataset: ExperimentData) -> tuple[str, int]:
    relative_paths = [
        Path(dataset.baseline_name) / f"task-{task_id}" / "results.json"
        for task_id in range(len(dataset.baseline_rates))
    ]
    for run in dataset.complete_runs:
        relative_paths.extend(
            Path(run.model_id) / f"task-{task_id}" / "results.json"
            for task_id in range(len(dataset.baseline_rates))
        )
    return _sha256_files(dataset.results_root, relative_paths), len(relative_paths)


def _weights_dict(values: Sequence[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(COMPONENTS, values, strict=True)}


def _in_support(weights: Sequence[float], lower: np.ndarray, upper: np.ndarray) -> bool:
    values = np.asarray(weights, dtype=np.float64)
    return bool(np.all(values >= lower - 1e-12) and np.all(values <= upper + 1e-12))


def _validation_dict(metrics: Any) -> dict[str, float]:
    return {
        "mae": float(metrics.mae),
        "rmse": float(metrics.rmse),
        "r2": float(metrics.r2),
    }


def _rank_candidates(
    dataset: ExperimentData,
    fitted: RegressionFit,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[dict[str, Any]]:
    matrix = np.asarray([candidate.weights for candidate in dataset.candidates])
    predictions = fitted.predict(matrix)
    rows: list[dict[str, Any]] = []
    for candidate, prediction in zip(dataset.candidates, predictions, strict=True):
        run = dataset.runs[candidate.model_id]
        rows.append(
            {
                "line_number": candidate.line_number,
                "model_id": candidate.model_id,
                "status": run.status,
                "in_support": _in_support(candidate.weights, lower, upper),
                "weights": candidate.as_dict(),
                "observed_score": run.score,
                "predicted_score": float(prediction),
                "residual": (None if run.score is None else float(run.score - prediction)),
            }
        )
    ranked = sorted(rows, key=lambda row: (-row["predicted_score"], row["line_number"]))
    rank_by_model = {row["model_id"]: rank for rank, row in enumerate(ranked, start=1)}
    for row in rows:
        row["rank"] = rank_by_model[row["model_id"]]
    return rows


def _best_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    observed: bool = False,
    require_support: bool = False,
    require_incomplete: bool = False,
) -> Mapping[str, Any] | None:
    eligible = []
    for row in rows:
        if observed and row["observed_score"] is None:
            continue
        if require_support and not row["in_support"]:
            continue
        if require_incomplete and row["status"] == "complete":
            continue
        eligible.append(row)
    if not eligible:
        return None
    score_field = "observed_score" if observed else "predicted_score"
    return min(
        eligible,
        key=lambda row: (-float(row[score_field]), int(row["line_number"])),
    )


def _summary_candidate(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "line_number": int(row["line_number"]),
        "model_id": str(row["model_id"]),
        "status": str(row["status"]),
        "weights": dict(row["weights"]),
        "observed_score": row["observed_score"],
        "predicted_score": float(row["predicted_score"]),
        "in_support": bool(row["in_support"]),
    }


def _stability(
    dataset: ExperimentData,
    fitted: RegressionFit,
    candidate_rows: Sequence[Mapping[str, Any]],
    lower: np.ndarray,
    upper: np.ndarray,
    final_best_model_id: str,
) -> dict[str, Any]:
    complete = dataset.complete_runs
    weights = np.asarray([run.weights for run in complete], dtype=np.float64)
    targets = np.asarray([run.score for run in complete], dtype=np.float64)
    all_candidates = np.asarray(
        [candidate.weights for candidate in dataset.candidates], dtype=np.float64
    )
    supported_indices = [index for index, row in enumerate(candidate_rows) if row["in_support"]]
    counts: Counter[str] = Counter()
    continuous: list[tuple[float, float, float, float]] = []
    reference = weights.mean(axis=0)
    for held_out in range(len(complete)):
        keep = np.arange(len(complete)) != held_out
        refitted = fit_fixed_regression(
            weights[keep],
            targets[keep],
            order=fitted.order,
            alpha=fitted.alpha,
        )
        predictions = refitted.predict(all_candidates)
        best_index = min(
            supported_indices,
            key=lambda index: (
                -float(predictions[index]),
                dataset.candidates[index].line_number,
            ),
        )
        counts[dataset.candidates[best_index].model_id] += 1
        optimum = optimize_bounded_simplex(
            refitted,
            lower_bounds=lower,
            upper_bounds=upper,
            reference_weights=reference,
        )
        continuous.append(optimum.weights)
    continuous_values = np.asarray(continuous, dtype=np.float64)
    ranges = {
        component: {
            "min": float(continuous_values[:, index].min()),
            "median": float(np.median(continuous_values[:, index])),
            "max": float(continuous_values[:, index].max()),
        }
        for index, component in enumerate(COMPONENTS)
    }
    return {
        "folds": len(complete),
        "candidate_selection_counts": dict(sorted(counts.items())),
        "top_candidate_frequency": counts[final_best_model_id] / len(complete),
        "continuous_weight_ranges": ranges,
    }


def analyze(dataset: ExperimentData, *, model_order: str = "auto") -> dict[str, Any]:
    complete = dataset.complete_runs
    weights = np.asarray([run.weights for run in complete], dtype=np.float64)
    targets = np.asarray([run.score for run in complete], dtype=np.float64)
    fitted = fit_regression(
        weights,
        targets,
        model_order=model_order,  # type: ignore[arg-type]
        alphas=DEFAULT_ALPHAS,
    )
    lower = weights.min(axis=0)
    upper = weights.max(axis=0)
    candidate_rows = _rank_candidates(dataset, fitted, lower, upper)
    observed_best = _best_row(candidate_rows, observed=True)
    predicted_best = _best_row(candidate_rows, require_support=True)
    if predicted_best is None:
        raise RegressionError("no candidate weight row lies inside observed support")
    next_evaluation = _best_row(
        candidate_rows,
        require_support=True,
        require_incomplete=True,
    )
    reference = weights.mean(axis=0)
    continuous = optimize_bounded_simplex(
        fitted,
        lower_bounds=lower,
        upper_bounds=upper,
        reference_weights=reference,
    )
    stability = _stability(
        dataset,
        fitted,
        candidate_rows,
        lower,
        upper,
        str(predicted_best["model_id"]),
    )
    provisional_reasons: list[str] = []
    if fitted.nested_validation.r2 <= 0.0:
        provisional_reasons.append(
            "nested leave-one-out R² is non-positive; the regression does not "
            "outperform predicting the training mean"
        )
    if stability["top_candidate_frequency"] < 0.5:
        provisional_reasons.append(
            "the selected supported candidate appears in fewer than 50% of "
            "leave-one-model-out refits"
        )
    results_sha256, result_file_count = _result_fingerprint(dataset)
    status_counts = Counter(run.status for run in dataset.runs.values())
    observations = []
    prediction_by_model = {row["model_id"]: row["predicted_score"] for row in candidate_rows}
    for run in complete:
        observations.append(
            {
                "model_id": run.model_id,
                "line_number": run.line_number,
                "weights": _weights_dict(run.weights),
                "task_rates": list(run.task_rates or ()),
                "score": run.score,
                "predicted_score": prediction_by_model[run.model_id],
                "residual": run.score - prediction_by_model[run.model_id],
            }
        )
    baseline_tasks = [
        {
            "task_id": task_id,
            "successes": successes,
            "episodes": episodes,
            "success_rate": rate,
        }
        for task_id, (successes, episodes, rate) in enumerate(
            zip(
                dataset.baseline_successes,
                dataset.baseline_episodes,
                dataset.baseline_rates,
                strict=True,
            )
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "results_root": str(dataset.results_root),
            "results_sha256": results_sha256,
            "result_file_count": result_file_count,
            "weights_path": str(dataset.weights_path),
            "weights_sha256": dataset.weights_sha256,
            "sweep_manifest_path": (
                None if dataset.sweep_manifest_path is None else str(dataset.sweep_manifest_path)
            ),
            "sweep_manifest_sha256": dataset.sweep_manifest_sha256,
        },
        "objective": {
            "formula": "mean_task(clip(model_success_rate / all_success_rate, 0, clip_max))",
            "clip_min": 0.0,
            "clip_max": dataset.clip_max,
            "task_count": len(dataset.baseline_rates),
            "task_contribution": 1.0 / len(dataset.baseline_rates),
            "baseline_score": dataset.baseline_score,
        },
        "baseline": {
            "name": dataset.baseline_name,
            "tasks": baseline_tasks,
        },
        "data_quality": {
            "candidate_count": len(dataset.candidates),
            "complete_model_count": status_counts["complete"],
            "partial_model_count": status_counts["partial"],
            "not_started_model_count": status_counts["not_started"],
            "included_models": [run.model_id for run in complete],
            "excluded_models": [
                {
                    "model_id": run.model_id,
                    "status": run.status,
                    "reason": run.exclusion_reason,
                }
                for run in dataset.runs.values()
                if run.status != "complete"
            ],
        },
        "observations": observations,
        "support_bounds": {
            "lower": _weights_dict(lower),
            "upper": _weights_dict(upper),
        },
        "model": {
            "requested_order": model_order,
            "selected_order": fitted.order,
            "alpha": fitted.alpha,
            "equation": fitted.equation,
            "raw_coefficients": dict(fitted.raw_coefficients),
            "nested_validation": _validation_dict(fitted.nested_validation),
            "selection_results": [
                {
                    "order": result.order,
                    "alpha": result.alpha,
                    "validation": _validation_dict(result.validation),
                }
                for result in fitted.selection_results
            ],
            "nested_selected_order_counts": dict(
                sorted(Counter(fitted.nested_selected_orders).items())
            ),
        },
        "candidates": candidate_rows,
        "optima": {
            "observed_best": _summary_candidate(observed_best),
            "predicted_supported_best": _summary_candidate(predicted_best),
            "next_evaluation": _summary_candidate(next_evaluation),
            "continuous": {
                "weights": _weights_dict(continuous.weights),
                "predicted_score": continuous.predicted_score,
                "active_constraints": list(continuous.active_constraints),
            },
        },
        "stability": stability,
        "provisional": bool(provisional_reasons),
        "warnings": provisional_reasons,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _csv_bytes(analysis: Mapping[str, Any]) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "rank",
        "line_number",
        "model_id",
        "status",
        "in_support",
        *COMPONENTS,
        "observed_score",
        "predicted_score",
        "residual",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for candidate in sorted(analysis["candidates"], key=lambda row: row["rank"]):
        writer.writerow(
            {
                "rank": candidate["rank"],
                "line_number": candidate["line_number"],
                "model_id": candidate["model_id"],
                "status": candidate["status"],
                "in_support": str(candidate["in_support"]).lower(),
                **candidate["weights"],
                "observed_score": candidate["observed_score"],
                "predicted_score": candidate["predicted_score"],
                "residual": candidate["residual"],
            }
        )
    return output.getvalue().encode("utf-8")


def _candidate_markdown(candidate: Mapping[str, Any] | None) -> str:
    if candidate is None:
        return "无"
    weights = ", ".join(f"{name}={candidate['weights'][name]:.4f}" for name in COMPONENTS)
    return (
        f"`{candidate['model_id']}` ({weights}); "
        f"predicted={candidate['predicted_score']:.6f}, "
        f"status={candidate['status']}"
    )


def _markdown(analysis: Mapping[str, Any]) -> bytes:
    model = analysis["model"]
    quality = analysis["data_quality"]
    optima = analysis["optima"]
    continuous = optima["continuous"]
    continuous_weights = ", ".join(
        f"{name}={continuous['weights'][name]:.4f}" for name in COMPONENTS
    )
    warning_lines = (
        [f"- {warning}" for warning in analysis["warnings"]] if analysis["warnings"] else ["- 无"]
    )
    lines = [
        "# TDUS Weight Regression Report",
        "",
        f"结论状态：**{'provisional' if analysis['provisional'] else 'validated'}**",
        "",
        "## 数据与目标",
        "",
        f"- 完整模型：{quality['complete_model_count']}；部分模型："
        f"{quality['partial_model_count']}；未开始：{quality['not_started_model_count']}",
        f"- 基准：`{analysis['baseline']['name']}`，基准归一化分数为 "
        f"{analysis['objective']['baseline_score']:.6f}",
        "- 每任务分数：`clip(model_success_rate / all_success_rate, 0, 2)`；十个任务等权平均。",
        "",
        "## 回归方程",
        "",
        f"- 选择模型：{model['selected_order']} ridge，alpha={model['alpha']:.12g}",
        f"- `{model['equation']}`",
        f"- 嵌套留一：RMSE={model['nested_validation']['rmse']:.6f}，"
        f"MAE={model['nested_validation']['mae']:.6f}，"
        f"R²={model['nested_validation']['r2']:.6f}",
        "",
        "## 最优参数",
        "",
        f"- 当前实测最佳：{_candidate_markdown(optima['observed_best'])}",
        f"- 支持范围内预测最佳：{_candidate_markdown(optima['predicted_supported_best'])}",
        f"- 下一评测候选：{_candidate_markdown(optima['next_evaluation'])}",
        f"- 连续参考解：{continuous_weights}; predicted={continuous['predicted_score']:.6f}",
        "",
        "## 警告",
        "",
        *warning_lines,
        "",
        "连续解和 provisional 预测必须经过新的训练/评测验证，不能当作实测性能。",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def write_outputs(analysis: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    outputs = {
        "analysis": root / "analysis.json",
        "ranking": root / "candidate_ranking.csv",
        "report": root / "report.md",
    }
    json_bytes = (json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write(outputs["analysis"], json_bytes)
    _atomic_write(outputs["ranking"], _csv_bytes(analysis))
    _atomic_write(outputs["report"], _markdown(analysis))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a task-equal TDUS weight regression from complete LIBERO-10 results"
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--weights-file", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--baseline-name", default="all")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--clip-max", type=float, default=2.0)
    parser.add_argument(
        "--model-order",
        choices=("auto", "linear", "quadratic"),
        default="auto",
    )
    parser.add_argument("--sweep-manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        dataset = load_experiment_data(
            arguments.results_root,
            arguments.weights_file,
            baseline_name=arguments.baseline_name,
            clip_max=arguments.clip_max,
            sweep_manifest=arguments.sweep_manifest,
        )
        result = analyze(dataset, model_order=arguments.model_order)
        outputs = write_outputs(result, arguments.output_dir)
    except (OSError, RegressionError, ValueError) as error:
        parser.error(str(error))
        return 2
    print(f"analysis: {outputs['analysis']}")
    print(f"ranking: {outputs['ranking']}")
    print(f"report: {outputs['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
