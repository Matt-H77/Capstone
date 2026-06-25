import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

# This module implements an SVM-based filter to classify high-yield vs low-yield candidates.
def train_svm_high_yield_classifier(
    load_dataset,
    X_train,
    y_train,
    high_yield_percent=0.25,
    label_mode="percentile",
    threshold_value=1e-5
):
    #Train an RBF SVM to classify high-yield vs low-yield samples.
    y_train = np.asarray(y_train).ravel()

    if label_mode == "threshold":
        threshold = float(threshold_value)
        labels = (y_train > threshold).astype(int)
    else:
        threshold = np.quantile(y_train, 1.0 - high_yield_percent)
        labels = (y_train >= threshold).astype(int)

    n_high = int(labels.sum())
    n_low = int(len(labels) - n_high)

    if n_high < 2 or n_low < 2:
        raise ValueError(
            "Not enough high/low examples for SVM. "
            f"Got high={n_high}, low={n_low}. "
            "Adjust svm threshold/percentile or collect more observations."
        )

    if load_dataset == 1:
        # Dataset 1 is simpler, so we can use a less complex SVM configuration.
        base_svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=2.0,
                gamma=5.0,
                class_weight="balanced",
                random_state=0
            )
        )

        svm_model = CalibratedClassifierCV(
            estimator=base_svm,
            method="sigmoid",
            cv=3
        )
        
    elif load_dataset == 7:
        # Dataset 7 is more complex and sparse, so we use a more aggressive SVM 
        # configuration to try to capture the high-yield region better.
        base_svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=5.0,
                gamma=2.0,
                class_weight="balanced",
                random_state=0
            )
        )

        svm_model = CalibratedClassifierCV(
            estimator=base_svm,
            method="sigmoid",
            cv=3
        )
    else:
        # For other datasets, we use a default configuration that is more aggressive 
        # than dataset 1 but less than dataset 7, as a starting point.
        base_svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=5.0,
                gamma="scale",
                class_weight="balanced",
                random_state=0
            )
        )

        svm_model = CalibratedClassifierCV(
            estimator=base_svm,
            method="sigmoid",
            cv=3
        )

    svm_model.fit(X_train, labels)
    return svm_model, labels, threshold

# This function applies the trained SVM model to score candidate points and filter 
# them based on the predicted probability of being high-yield. It uses a multi-step 
# strategy to ensure that a reasonable number of candidates are retained, even if 
# the initial probability threshold is too strict or too lenient.
def filter_candidates_with_svm(
    candidates_in,
    svm_model,
    keep_fraction=0.10,
    min_probability=0.60,
    min_keep=10000
):
    
    # Score generated candidates with the SVM and keep likely high-yield candidates.

    # Strategy:
    #  1. Keep candidates where P(high-yield) >= min_probability.
    #  2. If too few pass, fall back to the top min_keep candidates by probability.
    #  3. If too many pass, cap the result using keep_fraction while keeping the highest probabilities.

    # This multi-step strategy ensures that we retain a reasonable number of candidates for the next iteration,
    # even if the SVM probabilities are not perfectly calibrated or if the distribution of candidates is very skewed.
    candidates_in = np.asarray(candidates_in)
    p_high = svm_model.predict_proba(candidates_in)[:, 1]

    # First pass: probability threshold.
    keep_idx = np.where(p_high >= min_probability)[0]

    # Fallback: threshold too strict, so keep the best min_keep candidates.
    if len(keep_idx) < min_keep:
        n_keep = min(min_keep, len(candidates_in))
        keep_idx = np.argsort(p_high)[-n_keep:]
    else:
        # Cap the number of retained candidates using keep_fraction.
        max_keep = max(min_keep, int(len(candidates_in) * keep_fraction))

        if len(keep_idx) > max_keep:
            local_order = np.argsort(p_high[keep_idx])[-max_keep:]
            keep_idx = keep_idx[local_order]

    return candidates_in[keep_idx], p_high[keep_idx], p_high

# This function mixes the SVM-guided candidates with random candidates from the original global pool.
def mix_svm_candidates_with_global_exploration(
    svm_candidates,
    svm_scores,
    original_candidates,
    svm_model,
    random_fraction=0.0,
    random_state=0
):
    
    # Mix SVM-guided candidates with random candidates from the original global pool.

    # This keeps SVM as a soft guide rather than a hard gate. It is especially useful
    # for sparse/contamination-style functions where undiscovered islands may exist.
    svm_candidates = np.asarray(svm_candidates)
    original_candidates = np.asarray(original_candidates)

    # If random_fraction is 0 or there are no original candidates, just return the SVM candidates.
    if random_fraction <= 0.0 or len(original_candidates) == 0:
        return svm_candidates, svm_scores, 0

    # Set up random number generator for reproducibility.
    rng = np.random.default_rng(random_state)

    # Add random global points as a fraction of the SVM-guided set size.
    n_random = int(np.ceil(random_fraction * len(svm_candidates)))
    n_random = max(1, min(n_random, len(original_candidates)))

    # Randomly sample from the original candidates without replacement.
    random_idx = rng.choice(len(original_candidates), size=n_random, replace=False)
    random_candidates = original_candidates[random_idx]

    # Combine the SVM candidates with the random candidates.
    mixed_candidates = np.vstack([svm_candidates, random_candidates])

    # Remove exact duplicates introduced by mixing.
    mixed_candidates = np.unique(mixed_candidates, axis=0)

    # Re-score the mixed pool so logging and later diagnostics describe the final pool.
    mixed_scores = svm_model.predict_proba(mixed_candidates)[:, 1]

    return mixed_candidates, mixed_scores, n_random

# This function visualizes the SVM decision boundaries in 2D slices of the input space. 
# It creates contour plots of the predicted probability of being high-yield, along with 
# the observed training points and the slice center. This can help to understand how the 
# SVM is classifying the input space and where the high-yield regions are located.
def plot_svm_decision_boundaries(
    svm_model,
    X_train,
    y_train,
    labels,
    threshold,
    slice_center=None,
    grid_size=150,
    max_pairs=6
):
    
    # Plot 2D pairwise SVM decision-boundary slices.
    # For dimensions not shown, values are fixed at slice_center.
    # In more than 2D this is a slice, not the full high-dimensional boundary.
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    n_dims = X_train.shape[1]

    # If slice_center is not provided, use the best observed point (highest yield) as the center of the slice.
    if slice_center is None:
        slice_center = X_train[np.argmax(y_train)]

    pairs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            pairs.append((i, j))
    pairs = pairs[:max_pairs]

    # For each pair of dimensions, we create a grid of points and evaluate the SVM probability of 
    # being high-yield across that grid. We then plot the contour of these probabilities, 
    # along with the observed training points and the slice center.
    for dim_i, dim_j in pairs:
        xi = np.linspace(0.0, 1.0, grid_size)
        xj = np.linspace(0.0, 1.0, grid_size)
        xx, yy_grid = np.meshgrid(xi, xj)

        X_plot = np.tile(slice_center, (grid_size * grid_size, 1))
        X_plot[:, dim_i] = xx.ravel()
        X_plot[:, dim_j] = yy_grid.ravel()

        p_high = svm_model.predict_proba(X_plot)[:, 1].reshape(grid_size, grid_size)

        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(xx, yy_grid, p_high, levels=20, alpha=0.65)
        fig.colorbar(contour, ax=ax, label="SVM probability of high yield")

        ax.contour(xx, yy_grid, p_high, levels=[0.5], linewidths=2)

        low_mask = labels == 0
        high_mask = labels == 1

        ax.scatter(
            X_train[low_mask, dim_i],
            X_train[low_mask, dim_j],
            marker="x",
            s=80,
            label="Observed low yield"
        )
        ax.scatter(
            X_train[high_mask, dim_i],
            X_train[high_mask, dim_j],
            marker="*",
            s=160,
            label="Observed high yield"
        )

        ax.scatter(
            slice_center[dim_i],
            slice_center[dim_j],
            marker="o",
            s=120,
            facecolors="none",
            linewidths=2,
            label="Slice centre / best observed"
        )

        ax.set_xlabel(f"Input dimension {dim_i}")
        ax.set_ylabel(f"Input dimension {dim_j}")
        ax.set_title(
            "SVM high-yield decision boundary slice\n"
            f"High yield = top {100.0 * (labels.mean()):.1f}% observed, threshold={threshold:.6g}"
        )
        ax.legend()
        ax.grid(True)
        plt.show()

def plot_svm_decision_boundaries_grid(
    svm_model,
    X_train,
    y_train,
    labels,
    threshold,
    slice_center=None,
    grid_size=150,
    max_pairs=6
):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    labels = np.asarray(labels)

    n_dims = X_train.shape[1]

    if slice_center is None:
        slice_center = X_train[np.argmax(y_train)]

    pairs = [(i, j) for i in range(n_dims) for j in range(i + 1, n_dims)]
    pairs = pairs[:max_pairs]

    n_plots = len(pairs)

    if n_plots == 0:
        print("No dimension pairs available to plot.")
        return

    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()
    contour = None

    for plot_idx, (dim_i, dim_j) in enumerate(pairs):
        ax = axes[plot_idx]

        xi = np.linspace(0.0, 1.0, grid_size)
        xj = np.linspace(0.0, 1.0, grid_size)
        xx, yy_grid = np.meshgrid(xi, xj)

        X_plot = np.tile(slice_center, (grid_size * grid_size, 1))
        X_plot[:, dim_i] = xx.ravel()
        X_plot[:, dim_j] = yy_grid.ravel()

        p_high = svm_model.predict_proba(X_plot)[:, 1]
        p_high = p_high.reshape(grid_size, grid_size)

        contour = ax.contourf(
            xx,
            yy_grid,
            p_high,
            levels=20,
            alpha=0.75
        )

        ax.contour(
            xx,
            yy_grid,
            p_high,
            levels=[0.5],
            colors="black",
            linewidths=2
        )

        low_mask = labels == 0
        high_mask = labels == 1

        ax.scatter(
            X_train[low_mask, dim_i],
            X_train[low_mask, dim_j],
            marker="x",
            s=50
        )

        ax.scatter(
            X_train[high_mask, dim_i],
            X_train[high_mask, dim_j],
            marker="*",
            s=100
        )

        ax.scatter(
            slice_center[dim_i],
            slice_center[dim_j],
            marker="o",
            s=100,
            facecolors="none",
            edgecolors="black",
            linewidths=2
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Dim {dim_i}")
        ax.set_ylabel(f"Dim {dim_j}")
        ax.set_title(f"Dimensions {dim_i} vs {dim_j}")
        ax.grid(True, alpha=0.3)

    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    fig.subplots_adjust(
        left=0.06,
        right=0.88,
        bottom=0.06,
        top=0.86,
        hspace=0.35,
        wspace=0.22
    )

    cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.62])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label("SVM probability of high yield")

    legend_handles = [
        Line2D([0], [0], marker="x", linestyle="None", markersize=8, label="Low yield"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=10, label="High yield"),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2,
            label="Slice centre"
        )
    ]

    fig.suptitle(
        "SVM Decision Boundary Slices\n"
        f"High yield = top {100.0 * labels.mean():.1f}% observed "
        f"(threshold={threshold:.6g})",
        fontsize=14,
        y=0.985
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=True
    )

    plt.show()

def plot_svm_decision_boundary_matrix(
    svm_model,
    X_train,
    y_train,
    labels,
    threshold,
    slice_center=None,
    grid_size=100,
    show_margins=True
):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    labels = np.asarray(labels)

    n_dims = X_train.shape[1]

    if slice_center is None:
        slice_center = X_train[np.argmax(y_train)]

    fig, axes = plt.subplots(
        n_dims,
        n_dims,
        figsize=(3.0 * n_dims, 3.0 * n_dims)
    )

    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_dims == 2:
        axes = np.asarray(axes)

    contour = None

    for row in range(n_dims):
        for col in range(n_dims):
            ax = axes[row, col]

            if row == col:
                ax.text(
                    0.5, 0.5,
                    f"Dim {row}",
                    ha="center",
                    va="center",
                    fontsize=11
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue

            if col > row:
                ax.axis("off")
                continue

            dim_i = col
            dim_j = row

            xi = np.linspace(0.0, 1.0, grid_size)
            xj = np.linspace(0.0, 1.0, grid_size)
            xx, yy_grid = np.meshgrid(xi, xj)

            X_plot = np.tile(slice_center, (grid_size * grid_size, 1))
            X_plot[:, dim_i] = xx.ravel()
            X_plot[:, dim_j] = yy_grid.ravel()

            p_high = svm_model.predict_proba(X_plot)[:, 1]
            p_high = p_high.reshape(grid_size, grid_size)

            contour = ax.contourf(
                xx,
                yy_grid,
                p_high,
                levels=20,
                alpha=0.75
            )

            # Decision boundary using probability
            ax.contour(
                xx,
                yy_grid,
                p_high,
                levels=[0.5],
                colors="black",
                linewidths=1.5
            )

            # SVM margin lines using decision_function: -1 and +1
            if show_margins and hasattr(svm_model, "decision_function"):
                decision_values = svm_model.decision_function(X_plot)
                decision_values = decision_values.reshape(grid_size, grid_size)

                ax.contour(
                    xx,
                    yy_grid,
                    decision_values,
                    levels=[-1, 1],
                    colors="black",
                    linestyles="--",
                    linewidths=1.2
                )

            low_mask = labels == 0
            high_mask = labels == 1

            ax.scatter(
                X_train[low_mask, dim_i],
                X_train[low_mask, dim_j],
                marker="x",
                s=25,
                label="Low yield"
            )

            ax.scatter(
                X_train[high_mask, dim_i],
                X_train[high_mask, dim_j],
                marker="*",
                s=60,
                label="High yield"
            )

            ax.scatter(
                slice_center[dim_i],
                slice_center[dim_j],
                marker="o",
                s=60,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label="Slice centre"
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)

            if row == n_dims - 1:
                ax.set_xlabel(f"Dim {dim_i}")
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(f"Dim {dim_j}")
            else:
                ax.set_yticklabels([])

    if n_dims <= 2:
        top_space = 0.72
        legend_y = 0.86
        title_y = 0.97
        cbar_bottom = 0.18
        cbar_height = 0.55
    else:
        top_space = 0.88
        legend_y = 0.94
        title_y = 0.99
        cbar_bottom = 0.15
        cbar_height = 0.65

    fig.subplots_adjust(
        left=0.08,
        right=0.84,
        bottom=0.08,
        top=top_space,
        hspace=0.18,
        wspace=0.18
    )

    if contour is not None:
        cbar_ax = fig.add_axes([0.88, cbar_bottom, 0.03, cbar_height])
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label("SVM probability of high yield")

    legend_handles = [
        Line2D([0], [0], marker="x", linestyle="None", markersize=7, label="Low yield"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=9, label="High yield"),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.5,
            label="Slice centre"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Decision boundary"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="SVM margins"
        )
    ]

    fig.suptitle(
        "SVM Pairwise Decision Boundary Matrix\n"
        f"High yield = top {100.0 * labels.mean():.1f}% observed "
        f"(threshold={threshold:.6g})",
        fontsize=14,
        y=title_y
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=5,
        frameon=True
    )

    plt.show()
    
# This function extends the previous one by also plotting the decision boundary (where the SVM decision function is zero) 
# and the margin lines (where the decision function is ±1). This provides a more detailed visualization of how the SVM 
# is classifying the input space, showing not only the probability of being high-yield but also the regions where the 
# SVM is most confident in its classification.
def plot_svm_decision_boundaries_with_margin(
    svm_model,
    X_train,
    y_train,
    labels,
    threshold,
    slice_center=None,
    grid_size=150,
    max_pairs=6
):
    """
    Plot SVM probability surface together with
    decision boundary and margin lines.

    Solid line  : decision boundary (f(x)=0)
    Dashed lines: margin boundaries (f(x)=±1)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    labels = np.asarray(labels).ravel()

    n_dims = X_train.shape[1]

    if slice_center is None:
        slice_center = X_train[np.argmax(y_train)]

    pairs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            pairs.append((i, j))

    pairs = pairs[:max_pairs]

    for dim_i, dim_j in pairs:

        xi = np.linspace(0.0, 1.0, grid_size)
        xj = np.linspace(0.0, 1.0, grid_size)
        xx, yy_grid = np.meshgrid(xi, xj)

        X_plot = np.tile(slice_center, (grid_size * grid_size, 1))
        X_plot[:, dim_i] = xx.ravel()
        X_plot[:, dim_j] = yy_grid.ravel()

        p_high = svm_model.predict_proba(X_plot)[:, 1].reshape(
            grid_size,
            grid_size
        )

        decision_values = svm_model.decision_function(X_plot).reshape(
            grid_size,
            grid_size
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        contour = ax.contourf(
            xx,
            yy_grid,
            p_high,
            levels=20,
            alpha=0.65
        )

        fig.colorbar(
            contour,
            ax=ax,
            label="Probability(high yield)"
        )

        contours = ax.contour(
            xx,
            yy_grid,
            decision_values,
            levels=[-1, 0, 1],
            colors=["white", "black", "white"],
            linewidths=[1.5, 3.0, 1.5],
            linestyles=["--", "-", "--"]
        )

        ax.clabel(
            contours,
            fmt={
                -1: "margin -1",
                0: "boundary",
                1: "margin +1"
            },
            fontsize=8
        )

        low_mask = labels == 0
        high_mask = labels == 1

        ax.scatter(
            X_train[low_mask, dim_i],
            X_train[low_mask, dim_j],
            marker="x",
            s=80,
            label="Observed low yield"
        )

        ax.scatter(
            X_train[high_mask, dim_i],
            X_train[high_mask, dim_j],
            marker="*",
            s=160,
            label="Observed high yield"
        )

        ax.scatter(
            slice_center[dim_i],
            slice_center[dim_j],
            marker="o",
            s=120,
            facecolors="none",
            linewidths=2,
            label="Slice centre"
        )

        ax.set_xlabel(f"Input dimension {dim_i}")
        ax.set_ylabel(f"Input dimension {dim_j}")

        ax.set_title(
            "SVM probability surface with decision margin\n"
            f"High yield threshold = {threshold:.6g}"
        )

        ax.legend()
        ax.grid(True)

        plt.show()