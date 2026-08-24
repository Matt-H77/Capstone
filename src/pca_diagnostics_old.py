"""
PCA diagnostic helpers for the BBO Capstone notebook.

The main entry point is `run_pca_diagnostics`.
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display


def run_pca_diagnostics(
    X,
    y,
    candidates,
    load_dataset,
    week_dataset,
    best_input=None,
    best_input_thompson=None,
    nn_candidate=None,
    svm_enabled=False,
    best_SVM_candidate_input=None,
    target_variance=0.90,
    max_pca_candidate_points=10000,
):
    """Run PCA diagnostics and visual comparisons for an optimisation dataset.

    Parameters
    ----------
    X : array-like
        Observed input samples, shape (n_samples, n_dimensions).
    y : array-like
        Observed/transformed optimisation outputs.
    candidates : array-like
        Generated candidate input points.
    load_dataset : int or str
        Function/dataset identifier used in plot titles.
    week_dataset : int
        Current week/iteration identifier.
    best_input : array-like, optional
        GP / hybrid selected candidate.
    best_input_thompson : array-like, optional
        Thompson sampling selected candidate.
    nn_candidate : array-like, optional
        Neural-network selected candidate.
    svm_enabled : bool, default False
        Whether an SVM-selected candidate should be included.
    best_SVM_candidate_input : array-like, optional
        SVM selected candidate.
    target_variance : float, default 0.90
        Minimum cumulative observed-input variance to retain.
    max_pca_candidate_points : int, default 10000
        Maximum generated candidates shown in PCA plots.

    Returns
    -------
    dict
        PCA model, retained component count/variance, explained-variance table,
        loadings, transformed observed inputs, and candidate comparison table.
    """
    pca_full = None
    n_retained = 0
    retained_variance = np.nan
    explained_variance_table = None
    pca_loadings = None
    X_pca_retained = None
    pca_candidate_comparison = None

    print("==================================================")
    print(f"PCA DIAGNOSTICS - >={target_variance:.0%} VARIANCE COVERAGE")
    print("==================================================")

    X_pca_source = np.asarray(X, dtype=float)
    y_pca_source = np.asarray(y).ravel()

    if X_pca_source.ndim != 2 or X_pca_source.shape[1] < 2 or X_pca_source.shape[0] < 2:
        print("PCA diagnostics skipped: at least 2 samples and 2 input dimensions are required.")
    else:
        # ------------------------------------------------------
        # 1. Full PCA and automatic >=90% variance threshold
        # ------------------------------------------------------
        # PCA is fitted ONLY on observed inputs, not generated candidates.
        n_pca_components = min(X_pca_source.shape[0], X_pca_source.shape[1])
        pca_full = PCA(n_components=n_pca_components)
        X_pca_full = pca_full.fit_transform(X_pca_source)

        explained = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        n_retained = int(np.searchsorted(cumulative, target_variance) + 1)
        n_retained = min(n_retained, len(explained))
        retained_variance = float(cumulative[n_retained - 1])

        explained_variance_table = pd.DataFrame({
            "Principal Component": [f"PC{i+1}" for i in range(len(explained))],
            "Explained Variance": explained,
            "Cumulative Variance": cumulative,
            "Retained for target": [i < n_retained for i in range(len(explained))],
        })

        print("\nExplained variance by principal component:")
        display(explained_variance_table.style.format({
            "Explained Variance": "{:.3f}",
            "Cumulative Variance": "{:.3f}",
        }))

        print(
            f"\nMinimum components required for target variance: {n_retained} "
            f"(PC1-PC{n_retained}), explaining {retained_variance:.1%}."
        )

        plt.figure(figsize=(9, 5))
        component_numbers = np.arange(1, len(explained) + 1)
        plt.plot(component_numbers, cumulative, marker="o", label="Cumulative explained variance")
        plt.bar(component_numbers, explained, alpha=0.35, label="Individual explained variance")
        plt.axhline(target_variance, linestyle="--", linewidth=1.2, label=f"{target_variance:.0%} variance target")
        plt.axvline(n_retained, linestyle=":", linewidth=1.2, label=f"Retain PC1-PC{n_retained}")
        plt.xlabel("Principal component")
        plt.ylabel("Fraction of observed input variance")
        plt.title(f"Function {load_dataset}: PCA explained variance")
        plt.xticks(component_numbers)
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.show()

        # ------------------------------------------------------
        # 2. Retained PCA subspace and loadings
        # ------------------------------------------------------
        # Reuse the already-fitted full PCA basis and retain only PC1 ... PCn.
        X_pca_retained = X_pca_full[:, :n_retained]
        retained_explained = explained[:n_retained]

        loading_labels = [f"x{i+1}" for i in range(X_pca_source.shape[1])]
        retained_pc_names = [f"PC{i+1}" for i in range(n_retained)]

        pca_loadings = pd.DataFrame(
            pca_full.components_[:n_retained].T,
            index=loading_labels,
            columns=[f"{pc} loading" for pc in retained_pc_names],
        )

        print(f"\nPCA loadings for retained components PC1-PC{n_retained}:")
        display(pca_loadings.style.format("{:.3f}"))

        # ------------------------------------------------------
        # 3. Project a representative sample of generated candidates
        # ------------------------------------------------------
        candidate_array = np.asarray(candidates, dtype=float)

        if len(candidate_array) > max_pca_candidate_points:
            pca_rng = np.random.default_rng(load_dataset * 1000 + week_dataset)
            candidate_plot_indices = pca_rng.choice(
                len(candidate_array),
                size=max_pca_candidate_points,
                replace=False,
            )
            candidate_plot = candidate_array[candidate_plot_indices]
        else:
            candidate_plot = candidate_array

        candidate_plot_pca = pca_full.transform(candidate_plot)[:, :n_retained]

        # ------------------------------------------------------
        # 4. Collect model-selected candidates safely
        # ------------------------------------------------------
        selected_candidates = {}

        def add_pca_candidate(name, value):
            if value is None:
                return
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.shape[0] == X_pca_source.shape[1] and np.all(np.isfinite(arr)):
                selected_candidates[name] = arr

        add_pca_candidate("GP / Hybrid", best_input)
        add_pca_candidate("Thompson", best_input_thompson)
        add_pca_candidate("NN", nn_candidate)

        if svm_enabled:
            add_pca_candidate("SVM", best_SVM_candidate_input)

        selected_pca = {
            name: pca_full.transform(value.reshape(1, -1))[0, :n_retained]
            for name, value in selected_candidates.items()
        }

        marker_map = {
            "GP / Hybrid": "*",
            "Thompson": "P",
            "NN": "D",
            "SVM": "s",
        }
        color_map = {
            "GP / Hybrid": "tab:blue",
            "Thompson": "tab:orange",
            "NN": "tab:green",
            "SVM": "tab:red",
        }

        # Common observed-point masks used on every pairwise projection.
        high_y_threshold = np.quantile(y_pca_source, 0.75)
        high_y_mask = y_pca_source >= high_y_threshold
        best_observed_idx_pca = int(np.argmax(y_pca_source))

        # ------------------------------------------------------
        # 5. Lower-triangle matrix for ALL retained PCs
        # ------------------------------------------------------
        # The lower triangle shows every unique pair of retained PCs.
        # The diagonal shows the observed-data distribution for each PC.
        # The upper triangle is intentionally hidden to avoid duplicate plots.
        if n_retained < 2:
            print("Only one component was required for >=90% variance, so no 2D PC-pair matrix is available.")
        else:
            print(
                f"\nCreating lower-triangle PCA matrix for PC1-PC{n_retained} "
                f"({n_retained * (n_retained - 1) // 2} unique pairwise projections)."
            )

            matrix_size = max(3.0 * n_retained, 8.0)
            fig, axes = plt.subplots(
                n_retained,
                n_retained,
                figsize=(matrix_size, matrix_size),
                squeeze=False,
            )

            # Shared colour scale for observed y across every lower-triangle panel.
            y_min = float(np.nanmin(y_pca_source))
            y_max = float(np.nanmax(y_pca_source))

            # Store one observed scatter artist for a single shared colourbar.
            colourbar_artist = None

            for row in range(n_retained):
                for col in range(n_retained):
                    ax = axes[row, col]

                    # Hide duplicate upper-triangle panels.
                    if col > row:
                        ax.set_visible(False)
                        continue

                    # Diagonal: distribution of each retained PC for observed data.
                    if row == col:
                        ax.hist(
                            X_pca_retained[:, col],
                            bins=min(10, max(4, int(np.sqrt(len(X_pca_retained))))),
                            alpha=0.75,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        # Candidate positions along this individual PC.
                        # These lines are excluded from the automatic legend; the figure
                        # uses explicit marker-based legend handles below so model styles
                        # match the pairwise scatter panels.
                        ax.axvline(
                            X_pca_retained[best_observed_idx_pca, col],
                            color="black",
                            linestyle="--",
                            linewidth=1.4,
                        )
                        for name, coords in selected_pca.items():
                            ax.axvline(
                                coords[col],
                                color=color_map.get(name, "black"),
                                linestyle=":",
                                linewidth=1.3,
                            )
                        ax.set_ylabel("Count")
                        ax.grid(alpha=0.2)

                    # Lower triangle: pairwise PC projections.
                    else:
                        ax.scatter(
                            candidate_plot_pca[:, col],
                            candidate_plot_pca[:, row],
                            s=5,
                            alpha=0.06,
                            label="Generated candidate cloud",
                            zorder=1,
                        )

                        observed_scatter = ax.scatter(
                            X_pca_retained[:, col],
                            X_pca_retained[:, row],
                            c=y_pca_source,
                            s=45,
                            cmap="viridis",
                            vmin=y_min,
                            vmax=y_max,
                            edgecolors="black",
                            linewidths=0.35,
                            label="Observed inputs",
                            zorder=3,
                        )
                        colourbar_artist = observed_scatter

                        ax.scatter(
                            X_pca_retained[high_y_mask, col],
                            X_pca_retained[high_y_mask, row],
                            facecolors="none",
                            edgecolors="red",
                            linewidths=1.4,
                            s=85,
                            label="Top 25% observed y",
                            zorder=4,
                        )

                        ax.scatter(
                            X_pca_retained[best_observed_idx_pca, col],
                            X_pca_retained[best_observed_idx_pca, row],
                            marker="X",
                            s=130,
                            c="black",
                            label="Best observed",
                            zorder=6,
                        )

                        for name, coords in selected_pca.items():
                            ax.scatter(
                                coords[col],
                                coords[row],
                                marker=marker_map.get(name, "o"),
                                s=135,
                                c=color_map.get(name, "black"),
                                edgecolors="black",
                                linewidths=0.7,
                                label=f"{name} candidate",
                                zorder=7,
                            )

                        ax.grid(alpha=0.2)

                    # Labels only on the outside edges to keep the matrix readable.
                    if row == n_retained - 1:
                        ax.set_xlabel(
                            f"PC{col + 1}\n({retained_explained[col]:.1%})"
                        )
                    else:
                        ax.tick_params(labelbottom=False)

                    if col == 0 and row > 0:
                        ax.set_ylabel(
                            f"PC{row + 1}\n({retained_explained[row]:.1%})"
                        )
                    elif col != 0:
                        ax.tick_params(labelleft=False)

            fig.suptitle(
                f"Function {load_dataset}, week {week_dataset}: "
                f"PCA lower-triangle matrix (PC1-PC{n_retained}, {retained_variance:.1%} variance)",
                y=0.995,
                fontsize=14,
            )

            # One shared colourbar rather than repeating it in every panel.
            if colourbar_artist is not None:
                visible_axes = [
                    axes[r, c]
                    for r in range(n_retained)
                    for c in range(n_retained)
                    if c <= r
                ]
                cbar = fig.colorbar(
                    colourbar_artist,
                    ax=visible_axes,
                    fraction=0.025,
                    pad=0.02,
                )
                cbar.set_label("Transformed optimisation y")

            # Explicit figure-level legend.  We do not harvest artists from the
            # axes because the diagonal uses vertical lines, which otherwise causes
            # the model candidates to appear as identical dotted lines in the legend.
            legend_handles = [
                Line2D(
                    [0], [0], marker="X", linestyle="None", markersize=10,
                    markerfacecolor="black", markeredgecolor="black",
                    label="Best observed",
                ),
                Line2D(
                    [0], [0], marker=marker_map["GP / Hybrid"], linestyle="None", markersize=11,
                    markerfacecolor=color_map["GP / Hybrid"], markeredgecolor="black",
                    label="GP / Hybrid candidate",
                ),
                Line2D(
                    [0], [0], marker=marker_map["Thompson"], linestyle="None", markersize=10,
                    markerfacecolor=color_map["Thompson"], markeredgecolor="black",
                    label="Thompson candidate",
                ),
            ]

            if "NN" in selected_pca:
                legend_handles.append(
                    Line2D(
                        [0], [0], marker=marker_map["NN"], linestyle="None", markersize=9,
                        markerfacecolor=color_map["NN"], markeredgecolor="black",
                        label="NN candidate",
                    )
                )

            if "SVM" in selected_pca:
                legend_handles.append(
                    Line2D(
                        [0], [0], marker=marker_map["SVM"], linestyle="None", markersize=9,
                        markerfacecolor=color_map["SVM"], markeredgecolor="black",
                        label="SVM candidate",
                    )
                )

            legend_handles.extend([
                Line2D(
                    [0], [0], marker=".", linestyle="None", markersize=8,
                    markerfacecolor="tab:blue", markeredgecolor="none", alpha=0.25,
                    label="Generated candidate cloud",
                ),
                Line2D(
                    [0], [0], marker="o", linestyle="None", markersize=7,
                    markerfacecolor="tab:blue", markeredgecolor="black",
                    label="Observed inputs",
                ),
                Line2D(
                    [0], [0], marker="o", linestyle="None", markersize=8,
                    markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.5,
                    label="Top 25% observed y",
                ),
            ])

            fig.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.965),
                fontsize=9,
                frameon=True,
            )

            # Reserve space explicitly instead of calling tight_layout().  Shared
            # colourbar axes plus hidden upper-triangle axes can make tight_layout
            # emit a compatibility warning in Matplotlib.
            fig.subplots_adjust(
                left=0.055, right=0.82, bottom=0.055, top=0.94,
                wspace=0.18, hspace=0.18,
            )
            plt.show()

        # ------------------------------------------------------
        # 6. Cross-model comparison in the FULL retained PCA subspace
        # ------------------------------------------------------
        if selected_pca:
            gp_coords = selected_pca.get("GP / Hybrid")
            gp_original = selected_candidates.get("GP / Hybrid")
            pca_rows = []

            for name, coords in selected_pca.items():
                original = selected_candidates[name]
                retained_pca_distance = np.nan
                original_distance = np.nan

                if gp_coords is not None:
                    retained_pca_distance = float(np.linalg.norm(coords - gp_coords))
                if gp_original is not None:
                    original_distance = float(np.linalg.norm(original - gp_original))

                row = {
                    "Method": name,
                    f"PC1-PC{n_retained} distance from GP": retained_pca_distance,
                    "Original-space distance from GP": original_distance,
                }

                for i, coord in enumerate(coords):
                    row[f"PC{i+1}"] = coord

                pca_rows.append(row)

            pca_candidate_comparison = pd.DataFrame(pca_rows)

            # Put component coordinates before distance columns for readability.
            ordered_columns = (
                ["Method"]
                + [f"PC{i+1}" for i in range(n_retained)]
                + [f"PC1-PC{n_retained} distance from GP", "Original-space distance from GP"]
            )
            pca_candidate_comparison = pca_candidate_comparison[ordered_columns]

            print(f"\nCross-model candidate positions in retained PCA space (PC1-PC{n_retained}):")
            numeric_format = {col: "{:.4f}" for col in ordered_columns if col != "Method"}
            display(pca_candidate_comparison.style.format(numeric_format))

            print(
                f"\nThe retained PCA distance uses all {n_retained} components needed to explain "
                f"{retained_variance:.1%} of observed input variance. It remains an interpretation aid; "
                "the original-space distance is still the primary geometric measure."
            )

    return {
        "pca": pca_full,
        "n_retained": n_retained,
        "retained_variance": retained_variance,
        "explained_variance_table": explained_variance_table,
        "loadings": pca_loadings,
        "X_pca_retained": X_pca_retained,
        "candidate_comparison": pca_candidate_comparison,
    }
