import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
        svm_model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=2.0,
                gamma=5.0,
                probability=True,
                class_weight="balanced",
                random_state=0
            )
        )
    elif load_dataset == 7:
        # Dataset 7 is more complex and sparse, so we use a more aggressive SVM 
        # configuration to try to capture the high-yield region better.
        svm_model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=5.0,
                gamma=2.0,
                probability=True,
                class_weight="balanced",
                random_state=0
            )
        )
    else:
        # For other datasets, we use a default configuration that is more aggressive 
        # than dataset 1 but less than dataset 7, as a starting point.
        svm_model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=5.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=0
            )
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