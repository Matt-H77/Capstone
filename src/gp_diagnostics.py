"""
GP diagnostics and plotting utilities.

This module contains reusable plotting and reporting functions for Gaussian Process
posterior diagnostics, candidate acquisition diagnostics, and 2D posterior slices.

Typical usage in a notebook:

    from gp_diagnostics import run_gp_diagnostics

    run_gp_diagnostics(
        X=X,
        y=y,
        trained_models=trained_models,
        best_input=best_input,
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
        final_score=final_score,
        best_idx=best_idx,
        candidates=candidates,
        week_dataset=week_dataset,
        n_dimensions=n_dimensions,
    )
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_gp_posterior_slices(
    X: np.ndarray,
    y: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    ensemble_mean: np.ndarray,
    best_idx: int,
    week_dataset: int,
    n_dimensions: Optional[int] = None,
    beta: float = 2.0,
    grid_size: int = 500,
) -> None:
    """Plot 1D GP posterior slices for each input dimension.

    Each plot varies one input dimension from 0 to 1 while holding all other
    dimensions fixed at best_input.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    best_input = np.asarray(best_input)
    NN_best_input = np.asarray(NN_best_input)
    if n_dimensions is None:
        n_dimensions = X.shape[1]

    x_grid = np.linspace(0.0, 1.0, grid_size)
    recent_count = min(int(week_dataset), len(X))

    for dim in range(n_dimensions):
        fig, ax = plt.subplots(figsize=(15, 7))

        X_slice = np.tile(best_input, (grid_size, 1))
        X_slice[:, dim] = x_grid

        for i in range(1, recent_count + 1):
            ax.scatter(
                X[-i, dim],
                y[-i],
                c="b",
                marker="*",
                s=100,
                label="Recent observed samples" if i == 1 else None,
                alpha=0.7,
            )

        ax.axhline(y.max(), linestyle="--", color="gray", label="Max observed output")
        ax.axhline(y.min(), linestyle="--", color="gray", label="Min observed output")

        ax.scatter(
            X[:, dim],
            y,
            c="r",
            marker="x",
            label="Observed samples (projection only)",
            s=80,
            alpha=0.35,
        )

        for kernel_name, gp in trained_models.items():
            mean, std = gp.predict(X_slice, return_std=True)
            ax.plot(x_grid, mean, label=f"{kernel_name} posterior mean")
            ax.fill_between(
                x_grid,
                mean - beta * std,
                mean + beta * std,
                alpha=0.2,
                label=f"{kernel_name} +/- {beta:g} std",
            )

        ax.axvline(best_input[dim], linestyle="--", label="Slice centre")

        ax.scatter(
            best_input[dim],
            ensemble_mean[best_idx],
            marker="o",
            s=80,
            linewidths=3,
            label="Final selected point",
        )
        ax.scatter(
            NN_best_input[dim],
            ensemble_mean[best_idx],
            marker="s",
            s=80,
            linewidths=3,
            label="NN candidate",
        )

        ax.set_xlabel(f"Input dimension {dim}")
        ax.set_ylabel("Predicted output")
        ax.set_title(
            f"GP posterior slice for input dimension {dim}\n"
            "Other dimensions fixed at best_input"
        )
    
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.grid(True)
        plt.show()

def plot_gp_posterior_slices_grid(
    X: np.ndarray,
    y: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    ensemble_mean: np.ndarray,
    best_idx: int,
    week_dataset: int,
    n_dimensions: Optional[int] = None,
    beta: float = 2.0,
    grid_size: int = 500,
) -> None:
    """Plot 1D GP posterior slices for all dimensions in a subplot grid."""

    from matplotlib.lines import Line2D
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    y = np.asarray(y)
    best_input = np.asarray(best_input)
    NN_best_input = np.asarray(NN_best_input)

    if n_dimensions is None:
        n_dimensions = X.shape[1]

    x_grid = np.linspace(0.0, 1.0, grid_size)
    recent_count = min(int(week_dataset), len(X))

    ncols = int(np.ceil(np.sqrt(n_dimensions)))
    nrows = int(np.ceil(n_dimensions / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    kernel_legend_handles = {}

    for dim in range(n_dimensions):
        ax = axes[dim]

        X_slice = np.tile(best_input, (grid_size, 1))
        X_slice[:, dim] = x_grid

        for i in range(1, recent_count + 1):
            ax.scatter(
                X[-i, dim],
                y[-i],
                c="b",
                marker="*",
                s=60,
                alpha=0.7,
            )

        ax.axhline(y.max(), linestyle="--", color="gray", alpha=0.7)
        ax.axhline(y.min(), linestyle="--", color="gray", alpha=0.7)

        ax.scatter(
            X[:, dim],
            y,
            c="r",
            marker="x",
            s=30,
            alpha=0.35,
        )

        for kernel_name, gp in trained_models.items():
            mean, std = gp.predict(X_slice, return_std=True)

            line, = ax.plot(
                x_grid,
                mean,
                label=f"{kernel_name} posterior mean",
            )

            if kernel_name not in kernel_legend_handles:
                kernel_legend_handles[kernel_name] = line

            ax.fill_between(
                x_grid,
                mean - beta * std,
                mean + beta * std,
                alpha=0.15,
            )

        ax.axvline(
            best_input[dim],
            linestyle="--",
            color="k",
            alpha=0.6,
        )

        ax.scatter(
            best_input[dim],
            ensemble_mean[best_idx],
            marker="o",
            s=80,
            linewidths=2,
        )

        ax.scatter(
            NN_best_input[dim],
            ensemble_mean[best_idx],
            marker="s",
            s=80,
            linewidths=2,
            facecolors="none",
            edgecolors="red",
        )

        ax.scatter(
            Thompson_best_input[dim],
            ensemble_mean[best_idx],
            marker="D",
            s=80,
            linewidths=2,
            facecolors="none",
            edgecolors="blue",
        )

        ax.set_title(f"Dimension {dim+1}")
        ax.set_xlabel(f"Input dimension {dim+1}")
        ax.set_ylabel("Predicted output")
        ax.grid(True)

    for idx in range(n_dimensions, len(axes)):
        axes[idx].axis("off")

    legend_handles = list(kernel_legend_handles.values()) + [
        Line2D([], [], color="gray", linestyle="--", label="Min / max observed"),
        Line2D([], [], color="k", linestyle="--", label="Slice centre"),
        Line2D([], [], marker="x", linestyle="None", color="red", label="Observed samples"),
        Line2D([], [], marker="*", linestyle="None", color="blue", markersize=10, label="Recent samples"),
        Line2D([], [], marker="o", linestyle="None", color="red",markersize=8, label="Final selected point"),
        Line2D([], [], marker="D", linestyle="None", color="blue",markersize=8, label="Thompson candidate"),
        Line2D([], [], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="red", markeredgewidth=2, markersize=8, label="NN candidate", ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
    )

    fig.subplots_adjust(
        bottom=0.24,
        hspace=0.35,
        wspace=0.25,
    )

    plt.show()


def plot_training_fit(X: np.ndarray, y: np.ndarray, trained_models: Mapping[str, object]) -> None:
    """Plot actual vs predicted training outputs for each GP model."""
    plt.figure(figsize=(8, 6))

    for kernel_name, gp in trained_models.items():
        mean_train, _ = gp.predict(X, return_std=True)
        plt.scatter(y, mean_train, label=kernel_name)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Training Fit")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_training_std_report(X: np.ndarray, trained_models: Mapping[str, object]) -> None:
    """Print maximum posterior std on the training inputs for each GP model."""
    for kernel_name, gp in trained_models.items():
        _, std_train = gp.predict(X, return_std=True)
        print(f"{kernel_name} max training std:", np.max(std_train))


def plot_acquisition_score_distribution(final_score: np.ndarray, best_idx: int) -> None:
    """Plot the distribution of final acquisition scores."""
    plt.figure(figsize=(8, 6))
    plt.hist(final_score, bins=50)
    plt.axvline(final_score[best_idx], linestyle="--", label="Selected candidate")
    plt.xlabel("Final acquisition score")
    plt.ylabel("Candidate count")
    plt.title("Distribution of acquisition scores")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_candidate_score_report(
    final_score: np.ndarray,
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    candidates: np.ndarray,
    best_idx: int,
    top_n: int = 10,
) -> None:
    """Print percentile, score gap, top candidates, and top-candidate spread."""
    rank = np.sum(final_score <= final_score[best_idx])
    percentile = 100.0 * rank / len(final_score)

    print(f"Selected candidate percentile: {percentile:.4f}%")

    sorted_scores = np.sort(final_score)
    best = sorted_scores[-1]
    second = sorted_scores[-2] if len(sorted_scores) > 1 else np.nan

    print("Best score:", best)
    print("Second best:", second)
    print("Gap:", best - second if len(sorted_scores) > 1 else np.nan)

    top_idx = np.argsort(final_score)[-top_n:][::-1]

    print("\nTop candidates")
    print("-" * 60)

    for rank_num, idx in enumerate(top_idx, start=1):
        print(
            f"{rank_num:2d}: "
            f"score={final_score[idx]:.6f}, "
            f"mean={ensemble_mean[idx]:.3f}, "
            f"std={ensemble_std[idx]:.3f}, "
            f"x={candidates[idx]}"
        )

    top_points = candidates[np.argsort(final_score)[-top_n:]]

    print(f"\nTop-{top_n} candidate spread")
    print("Min:", np.min(top_points, axis=0))
    print("Max:", np.max(top_points, axis=0))
    print("Std:", np.std(top_points, axis=0))


def print_dimension_importance_report(
    final_score: np.ndarray,
    candidates: np.ndarray,
    n_dimensions: Optional[int] = None,
    top_n: int = 100,
) -> None:
    """Print dimension spread/importance based on the top scoring candidates."""
    if n_dimensions is None:
        n_dimensions = candidates.shape[1]

    top_n = min(top_n, len(final_score))
    top_idx = np.argsort(final_score)[-top_n:]
    top_points = candidates[top_idx]

    importance = np.std(top_points, axis=0)
    max_importance = np.max(importance)
    if max_importance > 0:
        importance = importance / max_importance

    print("\nDimension importance from top candidates")
    for d, imp in enumerate(importance):
        print(f"Dimension {d}: {imp:.3f}")

    print("\nTop candidate input distribution")
    for d in range(n_dimensions):
        print(
            f"Dim {d}: "
            f"mean={np.mean(top_points[:, d]):.3f}, "
            f"std={np.std(top_points[:, d]):.3f}"
        )


def plot_candidate_mean_vs_uncertainty(
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    final_score: np.ndarray,
    best_idx: int,
    thompson_best_idx: int,
) -> None:
    """Plot candidate mean vs uncertainty, with and without score colouring."""
    plt.figure(figsize=(8, 6))
    plt.scatter(ensemble_mean, ensemble_std, alpha=0.25, s=10)
    plt.scatter(
        ensemble_mean[best_idx],
        ensemble_std[best_idx],
        s=120,
        marker="o",
        label="Selected candidate",
    )
    plt.scatter(
    ensemble_mean[thompson_best_idx],
    ensemble_std[thompson_best_idx],
    s=120,
    marker="s",
    facecolors="none",
    edgecolors="red",
    linewidths=2,
    label="Thompson candidate",
    zorder=10,
    )
    plt.xlabel("Ensemble predicted mean")
    plt.ylabel("Ensemble predicted std")
    plt.title("Candidate mean vs uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(ensemble_mean, ensemble_std, c=final_score, s=5, alpha=0.3)
    plt.scatter(
        ensemble_mean[best_idx],
        ensemble_std[best_idx],
        s=120,
        marker="o",
        label="Selected candidate",
    )
    plt.scatter(
    ensemble_mean[thompson_best_idx],
    ensemble_std[thompson_best_idx],
    s=120,
    marker="s",
    facecolors="none",
    edgecolors="red",
    linewidths=2,
    label="Thompson candidate",
    zorder=10,
    )
    plt.colorbar(label="Final acquisition score")
    plt.xlabel("Ensemble predicted mean")
    plt.ylabel("Ensemble predicted std")
    plt.title("Candidate mean vs uncertainty colored by acquisition score")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_posterior_slices(
    X: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    n_dimensions: Optional[int] = None,
    model_name: str = "Matern",
    grid_size: int = 100,
) -> None:
    """Plot 2D posterior mean slices for all input-dimension pairs."""
    if n_dimensions is None:
        n_dimensions = X.shape[1]

    if model_name not in trained_models:
        available = ", ".join(trained_models.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: {available}")

    gp = trained_models[model_name]

    for dim_a, dim_b in combinations(range(n_dimensions), 2):
        x1 = np.linspace(0.0, 1.0, grid_size)
        x2 = np.linspace(0.0, 1.0, grid_size)
        xx, yy = np.meshgrid(x1, x2)

        X_grid = np.tile(best_input, (grid_size * grid_size, 1))
        X_grid[:, dim_a] = xx.ravel()
        X_grid[:, dim_b] = yy.ravel()

        mean_grid, _ = gp.predict(X_grid, return_std=True)
        mean_grid = mean_grid.reshape(grid_size, grid_size)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(xx, yy, mean_grid, levels=30)
        plt.colorbar(contour, label="Predicted mean")

        plt.scatter(X[:, dim_a], X[:, dim_b], marker="x", label="Observed samples")
        plt.scatter(
            best_input[dim_a],
            best_input[dim_b],
            s=120,
            marker="o",
            label="Selected candidate",
        )
        plt.scatter(
            NN_best_input[dim_a],
            NN_best_input[dim_b],
            s=120,
            marker="s",
            label="NN candidate",
        )

        plt.xlabel(f"Input dimension {dim_a}")
        plt.ylabel(f"Input dimension {dim_b}")
        plt.title(
            f"{model_name} 2D posterior mean slice\n"
            f"Dims {dim_a} vs {dim_b}, other dims fixed at best_input"
        )
        plt.legend()
        plt.grid(True)
        plt.show()


def run_gp_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    final_score: np.ndarray,
    best_idx: int,
    thompson_best_idx: int,
    candidates: np.ndarray,
    week_dataset: int,
    n_dimensions: Optional[int] = None,
    beta: float = 2.0,
    slice_grid_size: int = 500,
    posterior_2d_model_name: str = "Matern",
    posterior_2d_grid_size: int = 100,
    top_n: int = 10,
) -> None:
    """Run the full GP diagnostic plotting/reporting sequence."""
    if n_dimensions is None:
        n_dimensions = np.asarray(X).shape[1]

    plot_gp_posterior_slices_grid(
        X=X,
        y=y,
        trained_models=trained_models,
        best_input=best_input,
        NN_best_input=NN_best_input,
        Thompson_best_input=Thompson_best_input,
        ensemble_mean=ensemble_mean,
        best_idx=best_idx,
        week_dataset=week_dataset,
        n_dimensions=n_dimensions,
        beta=beta,
        grid_size=slice_grid_size,
    )

    plot_training_fit(X=X, y=y, trained_models=trained_models)
    print_training_std_report(X=X, trained_models=trained_models)
    plot_acquisition_score_distribution(final_score=final_score, best_idx=best_idx)

    print_candidate_score_report(
        final_score=final_score,
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
        candidates=candidates,
        best_idx=best_idx,
        top_n=top_n,
    )

    print_dimension_importance_report(
        final_score=final_score,
        candidates=candidates,
        n_dimensions=n_dimensions,
        top_n=100,
    )

    plot_candidate_mean_vs_uncertainty(
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
        final_score=final_score,
        best_idx=best_idx,
        thompson_best_idx=thompson_best_idx,
    )

    plot_2d_posterior_slices(
        X=X,
        trained_models=trained_models,
        best_input=best_input,
        NN_best_input=NN_best_input,
        n_dimensions=n_dimensions,
        model_name=posterior_2d_model_name,
        grid_size=posterior_2d_grid_size,
    )

    print("\nProcess complete.")
