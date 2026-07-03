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
from matplotlib.colors import Normalize
import numpy as np
import math


def _nearest_candidate_index(candidates: np.ndarray, point: np.ndarray) -> int:
    """Return the index of the candidate closest to a supplied point."""
    candidates = np.asarray(candidates)
    point = np.asarray(point)

    if candidates.ndim != 2:
        raise ValueError("candidates must be a 2D array.")
    if point.shape[0] != candidates.shape[1]:
        raise ValueError(
            f"point has length {point.shape[0]}, "
            f"but candidates have {candidates.shape[1]} dimensions."
        )

    return int(np.argmin(np.linalg.norm(candidates - point, axis=1)))


def plot_gp_posterior_slices(
    X: np.ndarray,
    y: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    ensemble_mean: np.ndarray,
    best_idx: int,
    week_dataset: int,
    thompson_best_idx: Optional[int] = None,
    nn_best_idx: Optional[int] = None,
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
    Thompson_best_input = np.asarray(Thompson_best_input)
    if n_dimensions is None:
        n_dimensions = X.shape[1]

    x_grid = np.linspace(0.0, 1.0, grid_size)
    recent_count = min(int(week_dataset), len(X))
    gp_candidate_y = ensemble_mean[best_idx]
    thompson_candidate_y = (
        ensemble_mean[thompson_best_idx]
        if thompson_best_idx is not None
        else gp_candidate_y
    )
    nn_candidate_y = (
        ensemble_mean[nn_best_idx]
        if nn_best_idx is not None
        else gp_candidate_y
    )

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
            gp_candidate_y,
            marker="o",
            s=80,
            linewidths=3,
            label="GP / acquisition candidate",
        )
        ax.scatter(
            Thompson_best_input[dim],
            thompson_candidate_y,
            marker="D",
            s=80,
            linewidths=3,
            facecolors="none",
            label="Thompson candidate",
        )
        ax.scatter(
            NN_best_input[dim],
            nn_candidate_y,
            marker="s",
            s=80,
            linewidths=3,
            facecolors="none",
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
    SVM_best_input: np.ndarray,
    ensemble_mean: np.ndarray,
    best_idx: int,
    week_dataset: int,
    thompson_best_idx: Optional[int] = None,
    nn_best_idx: Optional[int] = None,
    SVM_best_idx: Optional[int] = None,
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
    Thompson_best_input = np.asarray(Thompson_best_input)
    SVM_best_input = np.asarray(SVM_best_input)
    if n_dimensions is None:
        n_dimensions = X.shape[1]

    x_grid = np.linspace(0.0, 1.0, grid_size)
    recent_count = min(int(week_dataset), len(X))
    gp_candidate_y = ensemble_mean[best_idx]
    thompson_candidate_y = (
        ensemble_mean[thompson_best_idx]
        if thompson_best_idx is not None
        else gp_candidate_y
    )
    nn_candidate_y = (
        ensemble_mean[nn_best_idx]
        if nn_best_idx is not None
        else gp_candidate_y
    )
    svm_candidate_y = (
        ensemble_mean[SVM_best_idx]
        if SVM_best_idx is not None
        else gp_candidate_y
    )

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
            gp_candidate_y,
            marker="o",
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
        )

        ax.scatter(
            NN_best_input[dim],
            nn_candidate_y,
            marker="s",
            s=80,
            linewidths=2,
            facecolors="none",
            edgecolors="green",
        )

        ax.scatter(
            Thompson_best_input[dim],
            thompson_candidate_y,
            marker="D",
            s=80,
            linewidths=2,
            facecolors="none",
            edgecolors="blue",
        )

        ax.scatter(
            SVM_best_input[dim],
            svm_candidate_y,
            marker="^",
            s=80,
            linewidths=2,
            facecolors="none",
            edgecolors="purple",
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
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="red", markersize=8, label="GP / acquisition candidate"),
        Line2D([], [], marker="D", linestyle="None", markerfacecolor="none", markeredgecolor="blue", markersize=8, label="Thompson candidate"),
        Line2D([], [], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="green", markeredgewidth=2, markersize=8, label="NN candidate", ),
        Line2D([], [], marker="^", linestyle="None", markerfacecolor="none", markeredgecolor="purple", markeredgewidth=2, markersize=8, label="SVM candidate", ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
    )

    fig.suptitle(
        f"GP Posterior Slices Fixed Slice at GP Centre: {np.round(best_input, 3)}",
        fontsize=14,
        y=0.98,
    )

    fig.subplots_adjust(
        bottom=0.24,
        hspace=0.35,
        wspace=0.25,
    )

    plt.show()

def plot_gp_posterior_slices_grid_fixed(
    X: np.ndarray,
    y: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    ensemble_mean: np.ndarray,
    best_idx: int,
    week_dataset: int,
    thompson_best_idx: Optional[int] = None,
    nn_best_idx: Optional[int] = None,
    slice_center: Optional[np.ndarray] = None,
    n_dimensions: Optional[int] = None,
    beta: float = 2.0,
    grid_size: int = 500,
) -> None:
    """Plot 1D GP posterior slices using a fixed slice centre."""

    from matplotlib.lines import Line2D
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    y = np.asarray(y)
    best_input = np.asarray(best_input)
    NN_best_input = np.asarray(NN_best_input)
    Thompson_best_input = np.asarray(Thompson_best_input)

    if n_dimensions is None:
        n_dimensions = X.shape[1]

    if slice_center is None:
        slice_center = best_input.copy()
    else:
        slice_center = np.asarray(slice_center).copy()

    if slice_center.shape[0] != X.shape[1]:
        raise ValueError(
            f"slice_center has length {slice_center.shape[0]}, "
            f"but X has {X.shape[1]} dimensions."
        )

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

    gp_candidate_y = ensemble_mean[best_idx]
    thompson_candidate_y = (
        ensemble_mean[thompson_best_idx]
        if thompson_best_idx is not None
        else gp_candidate_y
    )
    nn_candidate_y = (
        ensemble_mean[nn_best_idx]
        if nn_best_idx is not None
        else gp_candidate_y
    )

    for dim in range(n_dimensions):
        ax = axes[dim]

        # Fixed slice position
        X_slice = np.tile(slice_center, (grid_size, 1))
        X_slice[:, dim] = x_grid

        # Observed samples
        ax.scatter(
            X[:, dim],
            y,
            c="r",
            marker="x",
            s=30,
            alpha=0.35,
        )

        # Recent samples
        for i in range(1, recent_count + 1):
            ax.scatter(
                X[-i, dim],
                y[-i],
                c="b",
                marker="*",
                s=60,
                alpha=0.7,
            )

        # Observed min/max
        ax.axhline(y.max(), linestyle="--", color="gray", alpha=0.7)
        ax.axhline(y.min(), linestyle="--", color="gray", alpha=0.7)

        # GP posterior slices
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

        # Fixed slice centre marker
        ax.axvline(
            slice_center[dim],
            linestyle="--",
            color="k",
            alpha=0.6,
        )

        # Candidate markers
        ax.scatter(
            best_input[dim],
            gp_candidate_y,
            marker="o",
            s=80,
            linewidths=2,
            color="red",
        )

        ax.scatter(
            Thompson_best_input[dim],
            thompson_candidate_y,
            marker="D",
            s=80,
            linewidths=2,
            facecolors="none",
            edgecolors="blue",
        )

        ax.scatter(
            NN_best_input[dim],
            nn_candidate_y,
            marker="s",
            s=80,
            linewidths=2,
            facecolors="none",
            edgecolors="red",
        )

        ax.set_title(f"Dimension {dim + 1}")
        ax.set_xlabel(f"Input dimension {dim + 1}")
        ax.set_ylabel("Predicted output")
        ax.grid(True)

    for idx in range(n_dimensions, len(axes)):
        axes[idx].axis("off")

    legend_handles = list(kernel_legend_handles.values()) + [
        Line2D([], [], color="gray", linestyle="--", label="Min / max observed"),
        Line2D([], [], color="k", linestyle="--", label="Fixed slice centre"),
        Line2D([], [], marker="x", linestyle="None", color="red", label="Observed samples"),
        Line2D([], [], marker="*", linestyle="None", color="blue", markersize=10, label="Recent samples"),
        Line2D([], [], marker="o", linestyle="None", color="red", markersize=8, label="GP / acquisition candidate"),
        Line2D([], [], marker="D", linestyle="None", color="blue", markersize=8, label="Thompson candidate"),
        Line2D([], [], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="red", markeredgewidth=2, markersize=8, label="NN candidate"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
    )

    fig.suptitle(
        f"GP Posterior Slices at Fixed Slice Centre: {np.round(slice_center, 3)}",
        fontsize=14,
        y=0.98,
    )

    fig.subplots_adjust(
        bottom=0.24,
        top=0.90,
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


def plot_acquisition_score_distribution(
    final_score: np.ndarray,
    best_idx: int,
    thompson_best_idx: Optional[int] = None,
    nn_best_idx: Optional[int] = None,
    SVM_best_idx: Optional[int] = None,
) -> None:
    """Plot final acquisition scores and mark GP, Thompson, NN, and SVM candidates."""
    plt.figure(figsize=(8, 6))
    plt.hist(final_score, bins=50)

    plt.axvline(
        final_score[best_idx],
        linestyle="--",
        color="red",
        label="GP / acquisition candidate",
    )

    if thompson_best_idx is not None:
        plt.axvline(
            final_score[thompson_best_idx],
            linestyle=":",
            color="blue",
            label="Thompson candidate",
        )

    if nn_best_idx is not None:
        plt.axvline(
            final_score[nn_best_idx],
            linestyle="-.",
            color="green",
            label="NN candidate",
        )

    if SVM_best_idx is not None:
        plt.axvline(
            final_score[SVM_best_idx],
            linestyle="--",
            color="purple",
            label="SVM candidate",
        )

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
    nn_best_idx: Optional[int] = None,
    SVM_best_idx: Optional[int] = None,
) -> None:
    """Plot candidate mean vs uncertainty, marking GP, Thompson, NN, and SVM candidates."""

    def add_candidate_markers() -> None:
        plt.scatter(
            ensemble_mean[best_idx],
            ensemble_std[best_idx],
            s=140,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="GP / acquisition candidate",
            zorder=10,
        )
        plt.scatter(
            ensemble_mean[thompson_best_idx],
            ensemble_std[thompson_best_idx],
            s=140,
            marker="D",
            facecolors="none",
            edgecolors="blue",
            linewidths=2,
            label="Thompson candidate",
            zorder=11,
        )
        if nn_best_idx is not None:
            plt.scatter(
                ensemble_mean[nn_best_idx],
                ensemble_std[nn_best_idx],
                s=140,
                marker="s",
                facecolors="none",
                edgecolors="green",
                linewidths=2,
                label="NN candidate",
                zorder=12,
            )
        if SVM_best_idx is not None:
            plt.scatter(
                ensemble_mean[SVM_best_idx],
                ensemble_std[SVM_best_idx],
                s=140,
                marker="^",
                facecolors="none",
                edgecolors="purple",
                linewidths=2,
                label="SVM candidate",
                zorder=13,
            )

    plt.figure(figsize=(8, 6))
    plt.scatter(ensemble_mean, ensemble_std, alpha=0.25, s=10)
    add_candidate_markers()
    plt.xlabel("Ensemble predicted mean")
    plt.ylabel("Ensemble predicted std")
    plt.title("Candidate mean vs uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(ensemble_mean, ensemble_std, c=final_score, s=5, alpha=0.3)
    add_candidate_markers()
    plt.colorbar(label="Final acquisition score")
    plt.xlabel("Ensemble predicted mean")
    plt.ylabel("Ensemble predicted std")
    plt.title("Candidate mean vs uncertainty colored by acquisition score")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_posterior_slices_old(
    X: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    n_dimensions: Optional[int] = None,
    model_name: str = "Matern",
    grid_size: int = 100,
) -> None:
    """Plot 2D posterior mean slices for all input-dimension pairs."""
    X = np.asarray(X)
    best_input = np.asarray(best_input)
    NN_best_input = np.asarray(NN_best_input)
    Thompson_best_input = np.asarray(Thompson_best_input)

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
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="GP / acquisition candidate",
        )
        plt.scatter(
            Thompson_best_input[dim_a],
            Thompson_best_input[dim_b],
            s=120,
            marker="D",
            facecolors="none",
            edgecolors="blue",
            linewidths=2,
            label="Thompson candidate",
        )
        plt.scatter(
            NN_best_input[dim_a],
            NN_best_input[dim_b],
            s=120,
            marker="s",
            facecolors="none",
            edgecolors="green",
            linewidths=2,
            label="NN candidate",
        )

        plt.xlabel(f"Input dimension {dim_a + 1}")
        plt.ylabel(f"Input dimension {dim_b + 1}")
        plt.title(
            f"{model_name} 2D posterior mean slice\n"
            f"Dims {dim_a + 1} vs {dim_b + 1}, other dims fixed at best_input"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_2d_posterior_slices(
    X: np.ndarray,
    trained_models,
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    n_dimensions=None,
    model_name="Matern",
    grid_size=100,
):
    """
    Plot every pairwise 2D posterior slice in a single figure.
    """

    X = np.asarray(X)
    best_input = np.asarray(best_input)
    NN_best_input = np.asarray(NN_best_input)
    Thompson_best_input = np.asarray(Thompson_best_input)

    if n_dimensions is None:
        n_dimensions = X.shape[1]

    if model_name not in trained_models:
        raise KeyError(
            f"'{model_name}' not found. Available: {list(trained_models.keys())}"
        )

    gp = trained_models[model_name]

    pairs = list(combinations(range(n_dimensions), 2))
    n_plots = len(pairs)

    # Choose a nearly-square grid
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    contour = None

    for ax, (dim_a, dim_b) in zip(axes.flat, pairs):

        x1 = np.linspace(0, 1, grid_size)
        x2 = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x1, x2)

        X_grid = np.tile(best_input, (grid_size * grid_size, 1))
        X_grid[:, dim_a] = xx.ravel()
        X_grid[:, dim_b] = yy.ravel()

        mean_grid, _ = gp.predict(X_grid, return_std=True)
        mean_grid = mean_grid.reshape(grid_size, grid_size)

        contour = ax.contourf(
            xx,
            yy,
            mean_grid,
            levels=30,
            cmap="viridis",
        )

        ax.scatter(
            X[:, dim_a],
            X[:, dim_b],
            marker="x",
            c="black",
            s=25,
            label="Observed samples",
        )

        ax.scatter(
            best_input[dim_a],
            best_input[dim_b],
            marker="o",
            s=120,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="GP / acquisition candidate",
        )

        ax.scatter(
            Thompson_best_input[dim_a],
            Thompson_best_input[dim_b],
            marker="D",
            s=120,
            facecolors="none",
            edgecolors="blue",
            linewidths=2,
            label="Thompson candidate",
        )

        ax.scatter(
            NN_best_input[dim_a],
            NN_best_input[dim_b],
            marker="s",
            s=120,
            facecolors="none",
            edgecolors="green",
            linewidths=2,
            label="NN candidate",
        )

        ax.set_xlabel(f"Dimension {dim_a+1}")
        ax.set_ylabel(f"Dimension {dim_b+1}")
        ax.set_title(f"Dims {dim_a+1} vs {dim_b+1}", fontsize=10)
        ax.grid(True)

    # Hide unused axes
    for ax in axes.flat[n_plots:]:
        ax.set_visible(False)

    # Shared colour bar
    cbar = fig.colorbar(
        contour,
        ax=axes.ravel().tolist(),
        shrink=0.9,
        pad=0.02,
    )
    cbar.set_label("Predicted mean")

    # Single legend
    handles, labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
    )

    # Overall title
    fig.suptitle(
        f"{model_name} GP Posterior Mean Slices\n"
        "Other dimensions fixed at the current best input",
        fontsize=16,
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0.06, 0.93, 0.95])
    plt.show()

def plot_posterior_pair_matrix(
    X,
    trained_models,
    best_input,
    NN_best_input,
    Thompson_best_input,
    model_name="Matern",
    grid_size=80,
):
    X = np.asarray(X)
    best_input = np.asarray(best_input)
    NN_best_input = np.asarray(NN_best_input)
    Thompson_best_input = np.asarray(Thompson_best_input)

    n_dims = X.shape[1]

    if model_name not in trained_models:
        raise KeyError(
            f"{model_name} not found. Available models: {list(trained_models.keys())}"
        )

    gp = trained_models[model_name]

    fig, axes = plt.subplots(
        n_dims,
        n_dims,
        figsize=(3.2 * n_dims, 3.2 * n_dims),
        squeeze=False,
    )

    contour = None

    for row in range(n_dims):
        for col in range(n_dims):
            ax = axes[row, col]

            if row == col:
                ax.text(
                    0.5,
                    0.5,
                    f"Dim {row + 1}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            dim_y = row
            dim_x = col

            x_vals = np.linspace(0.0, 1.0, grid_size)
            y_vals = np.linspace(0.0, 1.0, grid_size)
            xx, yy = np.meshgrid(x_vals, y_vals)

            X_grid = np.tile(best_input, (grid_size * grid_size, 1))
            X_grid[:, dim_x] = xx.ravel()
            X_grid[:, dim_y] = yy.ravel()

            mean_grid, _ = gp.predict(X_grid, return_std=True)
            mean_grid = mean_grid.reshape(grid_size, grid_size)

            contour = ax.contourf(xx, yy, mean_grid, levels=30)

            ax.scatter(
                X[:, dim_x],
                X[:, dim_y],
                marker="x",
                s=18,
                label="Observed samples",
            )

            ax.scatter(
                best_input[dim_x],
                best_input[dim_y],
                s=90,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                label="GP / acquisition candidate",
            )

            ax.scatter(
                Thompson_best_input[dim_x],
                Thompson_best_input[dim_y],
                s=90,
                marker="D",
                facecolors="none",
                edgecolors="blue",
                linewidths=2,
                label="Thompson candidate",
            )

            ax.scatter(
                NN_best_input[dim_x],
                NN_best_input[dim_y],
                s=90,
                marker="s",
                facecolors="none",
                edgecolors="green",
                linewidths=2,
                label="NN candidate",
            )

            if row == n_dims - 1:
                ax.set_xlabel(f"Dim {dim_x + 1}")

            if col == 0:
                ax.set_ylabel(f"Dim {dim_y + 1}")

            ax.grid(True, alpha=0.3)

    # Single colour bar
    cbar = fig.colorbar(
        contour,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        pad=0.02,
    )
    cbar.set_label("Predicted mean")

    # Single shared legend
    handles, labels = axes[0, 1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
    )

    fig.suptitle(
        f"{model_name} posterior pairwise matrix\n"
        "Each subplot shows one dimension against another; other dimensions fixed at best_input",
        fontsize=16,
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0.05, 0.93, 0.95])
    plt.show()

def plot_posterior_pair_matrix_lower_triangle(
    X: np.ndarray,
    trained_models: Mapping[str, Any],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    SVM_best_input: np.ndarray,
    n_dimensions: Optional[int] = None,
    model_name: str = "Matern",
    grid_size: int = 80,
    levels: int = 30,
    figsize_per_dim: float = 3.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot lower-triangle pairwise 2D GP posterior mean slices.

    Each lower-triangle subplot shows one input dimension against another.
    The diagonal shows dimension labels.
    The upper triangle is hidden.

    Other dimensions are fixed at best_input.

    Parameters
    ----------
    X:
        Observed input samples, shape (n_samples, n_dimensions).
    trained_models:
        Mapping of model names to trained GP-like models.
        The selected model must support predict(X_grid, return_std=True).
    best_input:
        Current GP/acquisition candidate or best known input.
    NN_best_input:
        Neural-network candidate input.
    Thompson_best_input:
        Thompson-sampling candidate input.
    n_dimensions:
        Number of dimensions to plot. If None, uses X.shape[1].
    model_name:
        Key into trained_models.
    grid_size:
        Number of grid samples per axis for each 2D slice.
    levels:
        Number of contour levels.
    figsize_per_dim:
        Figure scaling factor per dimension.
    save_path:
        Optional file path to save the figure, e.g. "figures/posterior_matrix.png".
    show:
        Whether to display the plot with plt.show().
    """

    X = np.asarray(X)
    best_input = np.asarray(best_input).ravel()
    NN_best_input = np.asarray(NN_best_input).ravel()
    Thompson_best_input = np.asarray(Thompson_best_input).ravel()

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    if n_dimensions is None:
        n_dimensions = X.shape[1]

    if n_dimensions < 2:
        raise ValueError("At least 2 dimensions are required for pairwise plotting.")

    if n_dimensions > X.shape[1]:
        raise ValueError(
            f"n_dimensions={n_dimensions} exceeds X.shape[1]={X.shape[1]}"
        )

    for name, arr in {
        "best_input": best_input,
        "NN_best_input": NN_best_input,
        "Thompson_best_input": Thompson_best_input,
        "SVM_best_input": SVM_best_input,
    }.items():
        if arr.size < n_dimensions:
            raise ValueError(
                f"{name} has length {arr.size}, but n_dimensions={n_dimensions}"
            )

    if model_name not in trained_models:
        available = ", ".join(trained_models.keys())
        raise KeyError(
            f"Model '{model_name}' not found. Available models: {available}"
        )

    gp = trained_models[model_name]

    # ------------------------------------------------------------------
    # First pass: compute all posterior mean grids so every subplot uses
    # the same colour scale. This makes colours comparable across panels.
    # ------------------------------------------------------------------
    x_vals = np.linspace(0.0, 1.0, grid_size)
    y_vals = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x_vals, y_vals)

    mean_grids = {}
    global_min = np.inf
    global_max = -np.inf

    for row in range(n_dimensions):
        for col in range(row):
            dim_y = row
            dim_x = col

            X_grid = np.tile(best_input, (grid_size * grid_size, 1))
            X_grid[:, dim_x] = xx.ravel()
            X_grid[:, dim_y] = yy.ravel()

            mean_grid, _ = gp.predict(X_grid, return_std=True)
            mean_grid = np.asarray(mean_grid).reshape(grid_size, grid_size)

            mean_grids[(row, col)] = mean_grid
            global_min = min(global_min, float(np.nanmin(mean_grid)))
            global_max = max(global_max, float(np.nanmax(mean_grid)))

    if not mean_grids:
        raise RuntimeError("No lower-triangle posterior grids were generated.")

    if np.isclose(global_min, global_max):
        pad = 1e-6 if global_min == 0 else abs(global_min) * 1e-6
        global_min -= pad
        global_max += pad

    norm = Normalize(vmin=global_min, vmax=global_max)
    contour_levels = np.linspace(global_min, global_max, levels)

    # ------------------------------------------------------------------
    # Plotting pass.
    # constrained_layout avoids tight_layout warnings caused by shared
    # colourbars, figure-level legends, and hidden axes.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        n_dimensions,
        n_dimensions,
        figsize=(figsize_per_dim * n_dimensions, figsize_per_dim * n_dimensions),
        squeeze=False,
        constrained_layout=True,
    )

    contour = None
    legend_handles = None
    legend_labels = None

    for row in range(n_dimensions):
        for col in range(n_dimensions):
            ax = axes[row, col]

            # Upper triangle: hidden because it duplicates the lower triangle.
            if col > row:
                ax.set_visible(False)
                continue

            # Diagonal: dimension labels only.
            if row == col:
                ax.text(
                    0.5,
                    0.5,
                    f"Dim {row + 1}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                continue

            dim_y = row
            dim_x = col
            mean_grid = mean_grids[(row, col)]

            contour = ax.contourf(
                xx,
                yy,
                mean_grid,
                levels=contour_levels,
                norm=norm,
            )

            ax.scatter(
                X[:, dim_x],
                X[:, dim_y],
                marker="x",
                s=20,
                c="black",
                label="Observed samples",
            )

            ax.scatter(
                best_input[dim_x],
                best_input[dim_y],
                s=110,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                label="GP / acquisition candidate",
            )

            ax.scatter(
                Thompson_best_input[dim_x],
                Thompson_best_input[dim_y],
                s=110,
                marker="D",
                facecolors="none",
                edgecolors="blue",
                linewidths=2,
                label="Thompson candidate",
            )

            ax.scatter(
                NN_best_input[dim_x],
                NN_best_input[dim_y],
                s=110,
                marker="s",
                facecolors="none",
                edgecolors="green",
                linewidths=2,
                label="NN candidate",
            )

            ax.scatter(
                SVM_best_input[dim_x],
                SVM_best_input[dim_y],
                s=110,
                marker="^",
                facecolors="none",
                edgecolors="purple",
                linewidths=2,
                label="SVM candidate",
            )

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

            if row == n_dimensions - 1:
                ax.set_xlabel(f"Dim {dim_x + 1}")
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(f"Dim {dim_y + 1}")
            else:
                ax.set_yticklabels([])

            ax.set_title(f"Dim {dim_y + 1} vs Dim {dim_x + 1}", fontsize=9)
            ax.grid(True, alpha=0.3)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle(
        f"{model_name} GP posterior mean pairwise matrix\n"
        "Lower triangle only; other dimensions fixed at best_input",
        fontsize=16,
    )

    if contour is not None:
        cbar = fig.colorbar(
            contour,
            ax=axes,
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Predicted mean")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=True,
        )

    plt.show()

    #if save_path is not None:
    #    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    #if show:
    #    plt.show()
    #else:
    #    plt.close(fig)

def run_gp_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    trained_models: Mapping[str, object],
    best_input: np.ndarray,
    NN_best_input: np.ndarray,
    Thompson_best_input: np.ndarray,
    SVM_best_input: np.ndarray,
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    final_score: np.ndarray,
    best_idx: int,
    thompson_best_idx: int,
    SVM_best_idx: int,
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

    nn_best_idx = _nearest_candidate_index(candidates, NN_best_input)

    plot_gp_posterior_slices_grid(
        X=X,
        y=y,
        trained_models=trained_models,
        best_input=best_input,
        NN_best_input=NN_best_input,
        Thompson_best_input=Thompson_best_input,
        SVM_best_input=SVM_best_input,
        ensemble_mean=ensemble_mean,
        best_idx=best_idx,
        week_dataset=week_dataset,
        thompson_best_idx=thompson_best_idx,
        nn_best_idx=nn_best_idx,
        SVM_best_idx=SVM_best_idx,
        n_dimensions=n_dimensions,
        beta=beta,
        grid_size=slice_grid_size,
    )

    fixed_slice_center = np.full(X.shape[1], 0.5)

    #plot_gp_posterior_slices_grid_fixed(
    #X=X,
    #y=y,
    #trained_models=trained_models,
    #best_input=best_input,
    #NN_best_input=NN_best_input,
    #Thompson_best_input=Thompson_best_input,
    #ensemble_mean=ensemble_mean,
    #best_idx=best_idx,
    #week_dataset=week_dataset,
    #thompson_best_idx=thompson_best_idx,
    #nn_best_idx=nn_best_idx,
    #slice_center=fixed_slice_center,
#)

    plot_training_fit(X=X, y=y, trained_models=trained_models)
    print_training_std_report(X=X, trained_models=trained_models)
    plot_acquisition_score_distribution(
        final_score=final_score,
        best_idx=best_idx,
        thompson_best_idx=thompson_best_idx,
        nn_best_idx=nn_best_idx,
        SVM_best_idx=SVM_best_idx
    )

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
        nn_best_idx=nn_best_idx,
        SVM_best_idx=SVM_best_idx
    )

    #plot_2d_posterior_slices(
    #    X=X,
    #    trained_models=trained_models,
    #    best_input=best_input,
    #    NN_best_input=NN_best_input,
    #    Thompson_best_input=Thompson_best_input,
    #    n_dimensions=n_dimensions,
    #    model_name=posterior_2d_model_name,
    #    grid_size=posterior_2d_grid_size,
   # )

    #plot_posterior_pair_matrix(
    #    X=X,
    #    trained_models=trained_models,
    #    best_input=best_input,
    #    NN_best_input=NN_best_input,
    #    Thompson_best_input=Thompson_best_input,
   #     model_name=posterior_2d_model_name,
    #    grid_size=posterior_2d_grid_size,
    #)

    #plot_posterior_pair_matrix_lower_triangle(
    #X=X,
    #trained_models=trained_models,
    #best_input=best_input,
    #NN_best_input=NN_best_input,
    #Thompson_best_input=Thompson_best_input,
    #model_name="Matern",
   # grid_size=80,
    #)
    
    plot_posterior_pair_matrix_lower_triangle(
    X=X,
    trained_models=trained_models,
    best_input=best_input,
    NN_best_input=NN_best_input,
    Thompson_best_input=Thompson_best_input,
    SVM_best_input=SVM_best_input,
    model_name="RBF",
    grid_size=80,
)

    print("\nProcess complete.")
