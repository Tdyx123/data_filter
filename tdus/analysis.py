"""Generate deterministic TDUS distribution and embedding diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .tdus import load_config


def _load_modes(
    dataset_root: Path, modes: Sequence[str]
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    loaded: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for mode in modes:
        mode_root = dataset_root / mode
        scores_path = mode_root / "tdus_scores.csv"
        embeddings_path = mode_root / "embeddings.npy"
        if scores_path.is_file() and embeddings_path.is_file():
            scores = pd.read_csv(scores_path)
            embeddings = np.load(embeddings_path)
            if len(scores) != len(embeddings):
                raise ValueError(f"{mode}: scores and embeddings are not aligned")
            loaded[mode] = (scores, embeddings)
    if not loaded:
        raise FileNotFoundError(f"No completed TDUS mode outputs found under {dataset_root}")
    return loaded


def generate_plots(config: Mapping[str, Any]) -> dict[str, Path]:
    """Write the requested PNG analysis bundle and return its paths."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    dataset_root = (
        Path(str(config["output"]["root"]))
        / str(config["dataset"].get("name", "dataset"))
    )
    modes = tuple(str(mode) for mode in config["segmentation"]["modes"])
    loaded = _load_modes(dataset_root, modes)
    plot_root = dataset_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    dpi = int(config["analysis"].get("dpi", 160))
    sns.set_theme(style="whitegrid")
    outputs: dict[str, Path] = {}

    fig, axis = plt.subplots(figsize=(8, 5))
    for mode, (scores, _) in loaded.items():
        sns.histplot(
            scores["tdus"],
            bins=int(config["analysis"].get("histogram_bins", 40)),
            stat="density",
            element="step",
            fill=False,
            label=mode,
            ax=axis,
        )
    axis.set(xlabel="TDUS", ylabel="Density", title="TDUS distribution")
    axis.legend()
    fig.tight_layout()
    outputs["tdus_hist"] = plot_root / "tdus_hist.png"
    fig.savefig(outputs["tdus_hist"], dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric in zip(axes, ("quality", "diversity", "novelty")):
        for mode, (scores, _) in loaded.items():
            sns.kdeplot(scores[metric], label=mode, fill=False, ax=axis, warn_singular=False)
        axis.set(xlim=(0, 1), title=f"{metric.title()} distribution")
        axis.legend()
    fig.tight_layout()
    outputs["quality_distribution"] = plot_root / "quality_distribution.png"
    fig.savefig(outputs["quality_distribution"], dpi=dpi)
    plt.close(fig)

    seed = int(config["runtime"].get("seed", 42))
    maximum = int(config["analysis"].get("tsne_max_samples", 5000))
    rng = np.random.default_rng(seed)
    sampled_embeddings: list[np.ndarray] = []
    sampled_modes: list[str] = []
    sampled_scores: list[np.ndarray] = []
    per_mode = max(1, maximum // len(loaded))
    for mode, (scores, embeddings) in loaded.items():
        count = min(len(embeddings), per_mode)
        indices = np.sort(rng.choice(len(embeddings), size=count, replace=False))
        sampled_embeddings.append(embeddings[indices])
        sampled_modes.extend([mode] * count)
        sampled_scores.append(scores["tdus"].to_numpy()[indices])
    combined = np.concatenate(sampled_embeddings)
    combined_scores = np.concatenate(sampled_scores)
    if len(combined) >= 3:
        from sklearn.manifold import TSNE

        configured_perplexity = float(config["analysis"].get("tsne_perplexity", 30))
        perplexity = min(configured_perplexity, max(1.0, (len(combined) - 1) / 3.0))
        coordinates = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(combined)
    else:
        coordinates = np.pad(combined[:, :2], ((0, 0), (0, max(0, 2 - combined.shape[1]))))
    fig, axis = plt.subplots(figsize=(8, 6))
    markers = {"trajectory": "o", "chunk": "^"}
    sampled_modes_array = np.asarray(sampled_modes)
    for mode in loaded:
        mask = sampled_modes_array == mode
        scatter = axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=combined_scores[mask],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=12,
            alpha=0.65,
            marker=markers.get(mode, "o"),
            label=mode,
        )
    fig.colorbar(scatter, ax=axis, label="TDUS")
    axis.set(title="t-SNE of trajectory embeddings", xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.legend()
    fig.tight_layout()
    outputs["embedding_tsne"] = plot_root / "embedding_tsne.png"
    fig.savefig(outputs["embedding_tsne"], dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    comparison = pd.concat(
        [
            scores.assign(segment_type=mode)
            for mode, (scores, _) in loaded.items()
        ],
        ignore_index=True,
    )
    sns.boxplot(data=comparison, x="segment_type", y="tdus", ax=axes[0])
    axes[0].set(title="TDUS by segment type", xlabel="", ylabel="TDUS")
    summary = (
        comparison.groupby("segment_type")[["quality", "coverage", "diversity", "novelty"]]
        .mean()
        .reset_index()
        .melt(id_vars="segment_type", var_name="metric", value_name="mean_score")
    )
    sns.barplot(
        data=summary,
        x="metric",
        y="mean_score",
        hue="segment_type",
        ax=axes[1],
    )
    axes[1].set(title="Mean component scores", xlabel="", ylabel="Mean score", ylim=(0, 1))
    fig.tight_layout()
    outputs["trajectory_vs_chunk"] = plot_root / "trajectory_vs_chunk.png"
    fig.savefig(outputs["trajectory_vs_chunk"], dpi=dpi)
    plt.close(fig)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TDUS analysis plots")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = generate_plots(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
