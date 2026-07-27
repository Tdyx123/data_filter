"""Static top-k and dynamic timestep-budget TDUS selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .coverage import CoverageModel, rbf_kernel
from .diversity import l2_normalize
from .tdus import load_config


def _read_scores(scores: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(scores) if not isinstance(scores, pd.DataFrame) else scores.copy()
    required = {
        "sample_id",
        "length",
        "quality",
        "coverage",
        "diversity",
        "novelty",
        "tdus",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scores are missing columns: {missing}")
    return frame


def select_top_k(
    k: int,
    scores: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Return exactly the highest-scoring k rows using deterministic ties."""

    if k < 0:
        raise ValueError("k must be non-negative")
    frame = _read_scores(scores)
    return (
        frame.sort_values(
            ["tdus", "length", "sample_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .head(k)
        .reset_index(drop=True)
    )


def _subset_utility_vectors(
    *,
    selected_count: int,
    selected_length_quality: float,
    selected_length_novelty: float,
    selected_kernel_sum: float,
    selected_reference_sum: float,
    candidate_selected_kernel: np.ndarray,
    reference_affinity: np.ndarray,
    reference_kernel_mean: float,
    selected_cosine_sum: float,
    candidate_selected_cosine: np.ndarray,
    lengths: np.ndarray,
    quality: np.ndarray,
    novelty: np.ndarray,
    budget: int,
    weights: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_count = selected_count + 1
    new_kernel_sum = selected_kernel_sum + 2.0 * candidate_selected_kernel + 1.0
    new_reference_sum = selected_reference_sum + reference_affinity
    mmd2 = np.maximum(
        reference_kernel_mean
        + new_kernel_sum / (new_count * new_count)
        - 2.0 * new_reference_sum / new_count,
        0.0,
    )
    coverage = np.exp(-mmd2)
    if new_count < 2:
        diversity = np.zeros_like(coverage)
    else:
        pair_count = new_count * (new_count - 1) / 2.0
        mean_cosine = (
            selected_cosine_sum + candidate_selected_cosine
        ) / pair_count
        diversity = np.clip(1.0 - (mean_cosine + 1.0) / 2.0, 0.0, 1.0)
    utility = (
        float(weights["quality"])
        * (selected_length_quality + lengths * quality)
        / budget
        + float(weights["coverage"]) * coverage
        + float(weights["diversity"]) * diversity
        + float(weights["novelty"])
        * (selected_length_novelty + lengths * novelty)
        / budget
    )
    return utility, coverage, diversity


def select_budget(
    budget: int,
    scores: str | Path | pd.DataFrame,
    embeddings: str | Path | np.ndarray,
    reference_embeddings: str | Path | np.ndarray,
    *,
    coverage_config: Mapping[str, Any],
    weights: Mapping[str, float],
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Greedily maximize marginal subset TDUS per timestep.

    The score CSV and embedding rows must be aligned; the TDUS pipeline writes
    them in the same deterministic sorted order.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    frame = _read_scores(scores).reset_index(drop=True)
    values = (
        np.load(embeddings) if isinstance(embeddings, (str, Path)) else np.asarray(embeddings)
    ).astype(np.float32)
    reference = (
        np.load(reference_embeddings)
        if isinstance(reference_embeddings, (str, Path))
        else np.asarray(reference_embeddings)
    ).astype(np.float32)
    if len(frame) != len(values):
        raise ValueError("scores and embeddings have different row counts")

    coverage_model = CoverageModel.fit(reference, coverage_config, seed=seed)
    reference_affinity = coverage_model.reference_affinity(values)
    normalized = l2_normalize(values)
    lengths = frame["length"].to_numpy(dtype=np.int64)
    quality = frame["quality"].to_numpy(dtype=np.float64)
    novelty = frame["novelty"].to_numpy(dtype=np.float64)
    static_tdus = frame["tdus"].to_numpy(dtype=np.float64)
    sample_ids = frame["sample_id"].astype(str).to_numpy()

    available = np.ones(len(frame), dtype=bool)
    selected_indices: list[int] = []
    trace: list[dict[str, Any]] = []
    selected_count = 0
    total_length = 0
    length_quality = 0.0
    length_novelty = 0.0
    kernel_sum = 0.0
    reference_sum = 0.0
    cosine_sum = 0.0
    candidate_kernel_sum = np.zeros(len(frame), dtype=np.float64)
    candidate_cosine_sum = np.zeros(len(frame), dtype=np.float64)
    current_utility = 0.0

    while True:
        fits = available & (lengths <= budget - total_length)
        candidates = np.flatnonzero(fits)
        if len(candidates) == 0:
            break
        candidate_utility, candidate_coverage, candidate_diversity = (
            _subset_utility_vectors(
                selected_count=selected_count,
                selected_length_quality=length_quality,
                selected_length_novelty=length_novelty,
                selected_kernel_sum=kernel_sum,
                selected_reference_sum=reference_sum,
                candidate_selected_kernel=candidate_kernel_sum[candidates],
                reference_affinity=reference_affinity[candidates],
                reference_kernel_mean=coverage_model.reference_kernel_mean,
                selected_cosine_sum=cosine_sum,
                candidate_selected_cosine=candidate_cosine_sum[candidates],
                lengths=lengths[candidates],
                quality=quality[candidates],
                novelty=novelty[candidates],
                budget=budget,
                weights=weights,
            )
        )
        gains = candidate_utility - current_utility
        gain_per_step = gains / lengths[candidates]
        # lexsort uses the last key as primary: gain density, static TDUS,
        # shorter length, then lexical sample id.
        rank = np.lexsort(
            (
                sample_ids[candidates],
                lengths[candidates],
                -static_tdus[candidates],
                -gain_per_step,
            )
        )
        local = int(rank[0])
        index = int(candidates[local])
        gain = float(gains[local])
        if gain <= 0.0:
            break

        selected_count += 1
        selected_indices.append(index)
        available[index] = False
        total_length += int(lengths[index])
        length_quality += float(lengths[index] * quality[index])
        length_novelty += float(lengths[index] * novelty[index])
        kernel_sum += 2.0 * float(candidate_kernel_sum[index]) + 1.0
        reference_sum += float(reference_affinity[index])
        cosine_sum += float(candidate_cosine_sum[index])
        current_utility = float(candidate_utility[local])
        trace.append(
            {
                "selection_order": selected_count,
                **frame.iloc[index].to_dict(),
                "marginal_gain": gain,
                "marginal_gain_per_step": gain / lengths[index],
                "cumulative_length": total_length,
                "subset_coverage": float(candidate_coverage[local]),
                "subset_diversity": float(candidate_diversity[local]),
                "subset_tdus": current_utility,
            }
        )

        # One vector-to-all update per selected sample; no NxN matrix is retained.
        candidate_kernel_sum += rbf_kernel(
            values, values[index : index + 1], coverage_model.sigma
        ).reshape(-1)
        candidate_cosine_sum += normalized @ normalized[index]

    result = pd.DataFrame(trace)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result


def _mode_root(config: Mapping[str, Any], mode: str) -> Path:
    return (
        Path(str(config["output"]["root"]))
        / str(config["dataset"].get("name", "dataset"))
        / mode
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select samples using TDUS")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--mode", choices=["trajectory", "chunk"], default="chunk")
    subparsers = parser.add_subparsers(dest="command", required=True)
    top = subparsers.add_parser("top-k")
    top.add_argument("--k", type=int, required=True)
    top.add_argument("--output", default=None)
    budget = subparsers.add_parser("budget")
    budget.add_argument("--budget", type=int, required=True)
    budget.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    mode_root = _mode_root(config, args.mode)
    scores_path = mode_root / "tdus_scores.csv"
    if args.command == "top-k":
        result = select_top_k(args.k, scores_path)
        output = Path(args.output) if args.output else mode_root / f"top_{args.k}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    else:
        output = (
            Path(args.output) if args.output else mode_root / "budget_selection.csv"
        )
        dataset_root = mode_root.parent
        reference_mode = "trajectory"
        manifest_path = dataset_root / "run_manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                reference_mode = str(
                    json.load(handle).get("coverage_reference_mode", reference_mode)
                )
        reference_path = dataset_root / reference_mode / "embeddings.npy"
        if not reference_path.is_file():
            reference_path = mode_root / "embeddings.npy"
        result = select_budget(
            args.budget,
            scores_path,
            mode_root / "embeddings.npy",
            reference_path,
            coverage_config=config["coverage"],
            weights=config["weights"],
            seed=int(config["runtime"].get("seed", 42)),
            output_path=output,
        )
    print(f"selected={len(result)} output={output}")


if __name__ == "__main__":
    main()
