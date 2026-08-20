import hashlib
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tdus.regression import (
    RegressionError,
    fit_regression,
    load_experiment_data,
    optimize_bounded_simplex,
)


COMPONENTS = ("quality", "coverage", "diversity", "novelty")


def _write_weights(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_result(
    root: Path,
    run_name: str,
    task_id: int,
    *,
    successes: int,
    episodes: int = 10,
    status: str = "complete",
    completed_episodes: int | None = None,
    max_steps: int = 960,
) -> None:
    task_root = root / run_name / f"task-{task_id}"
    task_root.mkdir(parents=True, exist_ok=True)
    completed = episodes if completed_episodes is None else completed_episodes
    result = {
        "schema_version": 3,
        "status": status,
        "route": "octo-small-pytorch-libero-checkpoint-eval",
        "statistics": {"sha256": "stats-sha"},
        "task": {
            "suite": "libero_10",
            "task_id": task_id,
            "name": f"task-name-{task_id}",
            "init_states_sha256": f"init-sha-{task_id}",
        },
        "protocol": {
            "episodes": episodes,
            "episodes_per_seed": episodes,
            "seeds": [0],
            "max_steps": max_steps,
            "settle_steps": 20,
            "action_horizon": 8,
        },
        "summary": {
            "completed_episodes": completed,
            "successes": successes,
            "failures": completed - successes,
            "success_rate": successes / completed if completed else 0.0,
        },
    }
    (task_root / "results.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )


def _write_complete_run(
    root: Path,
    run_name: str,
    successes: list[int],
) -> None:
    assert len(successes) == 10
    for task_id, task_successes in enumerate(successes):
        _write_result(
            root,
            run_name,
            task_id,
            successes=task_successes,
        )


def test_loader_uses_equal_task_clipped_ratios_and_excludes_partial_runs(
    tmp_path: Path,
) -> None:
    weights_path = tmp_path / "weights.jsonl"
    _write_weights(
        weights_path,
        [
            dict(zip(COMPONENTS, (0.4, 0.3, 0.2, 0.1), strict=True)),
            dict(zip(COMPONENTS, (0.1, 0.2, 0.3, 0.4), strict=True)),
        ],
    )
    results_root = tmp_path / "results"
    _write_complete_run(results_root, "all", [1] + [5] * 9)
    _write_complete_run(results_root, "model-001", [3] + [5] * 9)
    (results_root / "model-001" / "task-0" / "failure.json").write_text(
        "{}",
        encoding="utf-8",
    )
    for task_id in range(9):
        _write_result(
            results_root,
            "model-002",
            task_id,
            successes=5,
        )

    dataset = load_experiment_data(results_root, weights_path, clip_max=2.0)

    assert dataset.baseline_rates == pytest.approx((0.1,) + (0.5,) * 9)
    assert dataset.baseline_score == pytest.approx(1.0)
    assert dataset.runs["model-001"].status == "complete"
    assert dataset.runs["model-001"].score == pytest.approx(1.1)
    assert dataset.runs["model-002"].status == "partial"
    assert dataset.runs["model-002"].score is None
    assert [run.model_id for run in dataset.complete_runs] == ["model-001"]


def test_loader_rejects_complete_run_with_incompatible_protocol(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.jsonl"
    _write_weights(
        weights_path,
        [dict(zip(COMPONENTS, (0.4, 0.3, 0.2, 0.1), strict=True))],
    )
    results_root = tmp_path / "results"
    _write_complete_run(results_root, "all", [5] * 10)
    _write_complete_run(results_root, "model-001", [5] * 10)
    _write_result(
        results_root,
        "model-001",
        4,
        successes=5,
        max_steps=480,
    )

    with pytest.raises(RegressionError, match="model-001/task-4.*max_steps"):
        load_experiment_data(results_root, weights_path)


def test_loader_rejects_zero_baseline_success_rate(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.jsonl"
    _write_weights(
        weights_path,
        [dict(zip(COMPONENTS, (0.4, 0.3, 0.2, 0.1), strict=True))],
    )
    results_root = tmp_path / "results"
    _write_complete_run(results_root, "all", [0] + [5] * 9)

    with pytest.raises(RegressionError, match="baseline task-0 success rate is zero"):
        load_experiment_data(results_root, weights_path)


def test_loader_rejects_clip_max_that_changes_baseline_score(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.jsonl"
    _write_weights(
        weights_path,
        [dict(zip(COMPONENTS, (0.4, 0.3, 0.2, 0.1), strict=True))],
    )
    results_root = tmp_path / "results"
    _write_complete_run(results_root, "all", [5] * 10)

    with pytest.raises(RegressionError, match="clip_max must be at least 1"):
        load_experiment_data(results_root, weights_path, clip_max=0.5)


def test_loader_rejects_sweep_manifest_weight_mapping_conflict(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.jsonl"
    weights = dict(zip(COMPONENTS, (0.4, 0.3, 0.2, 0.1), strict=True))
    _write_weights(weights_path, [weights])
    manifest_path = tmp_path / "sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                "jobs": [
                    {
                        "line_number": 1,
                        "model_id": "model-001",
                        "weights": {**weights, "quality": 0.5, "coverage": 0.2},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    _write_complete_run(results_root, "all", [5] * 10)

    with pytest.raises(RegressionError, match="sweep manifest.*model-001.*weights"):
        load_experiment_data(
            results_root,
            weights_path,
            sweep_manifest=manifest_path,
        )


def _simplex_design(count: int = 24) -> np.ndarray:
    return np.random.default_rng(731).dirichlet(np.ones(4), size=count)


@pytest.mark.skip(reason="Long-running automatic regression model selection test")
def test_auto_regression_selects_linear_and_restores_raw_equation() -> None:
    weights = _simplex_design()
    target = 0.7 + 0.4 * weights[:, 0] - 0.2 * weights[:, 1] + 0.1 * weights[:, 2]

    fitted = fit_regression(weights, target, model_order="auto")

    assert fitted.order == "linear"
    assert fitted.alpha == pytest.approx(0.0)
    assert fitted.raw_coefficients == pytest.approx(
        {
            "intercept": 0.7,
            "quality": 0.4,
            "coverage": -0.2,
            "diversity": 0.1,
        },
        abs=1e-10,
    )
    assert fitted.predict(weights) == pytest.approx(target, abs=1e-10)
    assert fitted.nested_validation.rmse == pytest.approx(0.0, abs=1e-10)


@pytest.mark.skip(reason="Long-running automatic regression model selection test")
def test_auto_regression_selects_quadratic_for_curved_response() -> None:
    weights = _simplex_design(30)
    quality = weights[:, 0]
    coverage = weights[:, 1]
    diversity = weights[:, 2]
    target = (
        1.2 - 2.0 * quality**2 - 1.5 * coverage**2 - 1.2 * diversity**2 + 0.8 * quality * coverage
    )

    fitted = fit_regression(weights, target, model_order="auto")

    assert fitted.order == "quadratic"
    assert fitted.predict(weights) == pytest.approx(target, abs=1e-9)
    assert fitted.raw_coefficients["quality^2"] == pytest.approx(-2.0, abs=1e-8)
    assert fitted.raw_coefficients["quality*coverage"] == pytest.approx(0.8, abs=1e-8)


def test_bounded_simplex_optimizer_finds_linear_boundary_optimum() -> None:
    weights = _simplex_design()
    target = weights[:, 0] + 2.0 * weights[:, 1]
    fitted = fit_regression(
        weights,
        target,
        model_order="linear",
        alphas=(0.0,),
    )

    optimum = optimize_bounded_simplex(
        fitted,
        lower_bounds=(0.0, 0.0, 0.0, 0.0),
        upper_bounds=(1.0, 1.0, 1.0, 1.0),
    )

    assert optimum.weights == pytest.approx((0.0, 1.0, 0.0, 0.0), abs=1e-9)
    assert optimum.predicted_score == pytest.approx(2.0, abs=1e-9)


def test_bounded_simplex_optimizer_finds_quadratic_interior_optimum() -> None:
    weights = _simplex_design(30)
    quality = weights[:, 0]
    coverage = weights[:, 1]
    diversity = weights[:, 2]
    target = -((quality - 0.2) ** 2 + (coverage - 0.3) ** 2 + (diversity - 0.1) ** 2)
    fitted = fit_regression(
        weights,
        target,
        model_order="quadratic",
        alphas=(0.0,),
    )

    optimum = optimize_bounded_simplex(
        fitted,
        lower_bounds=(0.0, 0.0, 0.0, 0.0),
        upper_bounds=(1.0, 1.0, 1.0, 1.0),
    )

    assert optimum.weights == pytest.approx((0.2, 0.3, 0.1, 0.4), abs=1e-8)
    assert optimum.predicted_score == pytest.approx(0.0, abs=1e-9)


def test_linear_regression_accepts_six_rows_for_nested_leave_one_out() -> None:
    weights = _simplex_design(6)
    target = 0.8 + weights[:, 0] - weights[:, 2]

    fitted = fit_regression(
        weights,
        target,
        model_order="linear",
        alphas=(0.0, 1.0),
    )

    assert len(fitted.nested_predictions) == 6


def test_cli_writes_deterministic_json_csv_and_markdown_reports(tmp_path: Path) -> None:
    from scripts.analyze_libero_tdus_weights import main

    weights = _simplex_design(100)
    weights[6] = weights[:6].mean(axis=0)
    weights_path = tmp_path / "weights.jsonl"
    _write_weights(
        weights_path,
        [dict(zip(COMPONENTS, row.tolist(), strict=True)) for row in weights],
    )
    results_root = tmp_path / "results"
    _write_complete_run(results_root, "all", [5] * 10)
    for index, successes in enumerate((3, 4, 5, 6, 7, 8), start=1):
        _write_complete_run(
            results_root,
            f"model-{index:03d}",
            [successes] * 10,
        )
    output_dir = tmp_path / "analysis"
    arguments = [
        "--results-root",
        str(results_root),
        "--weights-file",
        str(weights_path),
        "--output-dir",
        str(output_dir),
        "--model-order",
        "linear",
    ]

    assert main(arguments) == 0
    first_bytes = {
        name: (output_dir / name).read_bytes()
        for name in ("analysis.json", "candidate_ranking.csv", "report.md")
    }
    assert main(arguments) == 0
    assert first_bytes == {
        name: (output_dir / name).read_bytes()
        for name in ("analysis.json", "candidate_ranking.csv", "report.md")
    }

    analysis = json.loads(first_bytes["analysis.json"])
    assert analysis["schema_version"] == 1
    assert analysis["objective"]["baseline_score"] == pytest.approx(1.0)
    assert analysis["data_quality"]["complete_model_count"] == 6
    assert len(analysis["candidates"]) == 100
    assert analysis["candidates"][6]["status"] == "not_started"
    assert analysis["candidates"][6]["in_support"] is True
    assert sum(analysis["optima"]["continuous"]["weights"].values()) == pytest.approx(1.0)
    rows = list(csv.DictReader(first_bytes["candidate_ranking.csv"].decode().splitlines()))
    assert len(rows) == 100
    assert rows[0]["rank"] == "1"
    report = first_bytes["report.md"].decode()
    assert "# TDUS Weight Regression Report" in report
    assert "score_hat =" in report
