
# PCA diagnostic helpers for the BBO Capstone notebook.

# The main entry point is `run_pca_diagnostics`.


from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display

#------------------------------------------------------
#  Main PCA diagnostics function
#------------------------------------------------------
def run_pca_diagnostics(
    X, # Observed input samples, shape (n_samples, n_dimensions).
    y, # Observed/transformed optimisation outputs.
    candidates, # Generated candidate input points.
    load_dataset, # Function/dataset identifier used in plot titles.
    week_dataset, # Current week/iteration identifier.
    best_input=None, # GP / hybrid selected candidate.
    best_input_thompson=None, # Thompson sampling selected candidate.
    nn_candidate=None, # Neural-network selected candidate.
    svm_enabled=False, # Whether an SVM-selected candidate should be included.
    best_SVM_candidate_input=None, # SVM selected candidate.
    target_variance=0.90, # Minimum cumulative observed-input variance to retain.
    max_pca_candidate_points=10000, # Maximum generated candidates shown in PCA plots.
):
    # Run PCA diagnostics and visual comparisons for an optimisation dataset.

    # Parameters
    # ----------
    # X : array-like
    #     Observed input samples, shape (n_samples, n_dimensions).
    # y : array-like
    #     Observed/transformed optimisation outputs.
    # candidates : array-like
    #     Generated candidate input points.
    # load_dataset : int or str
    #     Function/dataset identifier used in plot titles.
    # week_dataset : int
    #     Current week/iteration identifier.
    # best_input : array-like, optional
    #     GP / hybrid selected candidate.
    # best_input_thompson : array-like, optional
    #     Thompson sampling selected candidate.
    # nn_candidate : array-like, optional
    #     Neural-network selected candidate.
    # svm_enabled : bool, default False
    #     Whether an SVM-selected candidate should be included.
    # best_SVM_candidate_input : array-like, optional
    #     SVM selected candidate.
    # target_variance : float, default 0.90
    #     Minimum cumulative observed-input variance to retain.
    # max_pca_candidate_points : int, default 10000
    #     Maximum generated candidates shown in PCA plots.

    # Returns
    # -------
    # dict
    #    PCA model, retained component count/variance, explained-variance table,
    #     loadings, transformed observed inputs, and candidate comparison table.
    
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

    #------------------------------------------------------
    # Convert inputs to numpy arrays and validate shapes.
    #------------------------------------------------------
    X_pca_source = np.asarray(X, dtype=float)
    y_pca_source = np.asarray(y).ravel()

    #------------------------------------------------------
    # Skip PCA if there are too few samples or dimensions.
    #------------------------------------------------------
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

        #------------------------------------------------------
        # Determine how many components are needed to reach the target variance.
        #------------------------------------------------------
        explained = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        #------------------------------------------------------
        # Retain the minimum number of components that reach the target variance.
        #------------------------------------------------------
        n_retained = int(np.searchsorted(cumulative, target_variance) + 1)
        n_retained = min(n_retained, len(explained))
        retained_variance = float(cumulative[n_retained - 1])

        #------------------------------------------------------
        # Create a table of explained variance for each principal component.
        #------------------------------------------------------
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
        "X_pca_source": X_pca_source,
        "y_pca_source": y_pca_source,
        "X_pca_retained": X_pca_retained,
        "retained_explained": retained_explained if X_pca_retained is not None else None,
        "retained_pc_names": retained_pc_names if X_pca_retained is not None else [],
        "selected_candidates": selected_candidates if X_pca_retained is not None else {},
        "selected_pca": selected_pca if X_pca_retained is not None else {},
        "high_y_mask": high_y_mask if X_pca_retained is not None else None,
        "best_observed_idx_pca": best_observed_idx_pca if X_pca_retained is not None else None,
        "candidate_comparison": pca_candidate_comparison,
    }

def run_extended_pca_diagnostics(
    pca_results,
    load_dataset,
):
    """Run extended PCA reporting using results from run_pca_diagnostics()."""

    print("==================================================")
    print("EXTENDED PCA DIAGNOSTIC REPORTING")
    print("==================================================")

    if not isinstance(pca_results, dict):
        raise TypeError(
            "pca_results must be the dictionary returned by run_pca_diagnostics()."
        )

    required_keys = [
        "X_pca_source",
        "y_pca_source",
        "X_pca_retained",
        "selected_pca",
        "selected_candidates",
        "n_retained",
        "retained_variance",
        "high_y_mask",
        "best_observed_idx_pca",
        "retained_pc_names",
        "retained_explained",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in pca_results or pca_results[key] is None
    ]

    if missing_keys:
        print("Extended PCA diagnostics skipped.")
        print("Missing PCA result values:", missing_keys)
        return None

    X_pca_source = pca_results["X_pca_source"]
    y_pca_source = pca_results["y_pca_source"]
    X_pca_retained = pca_results["X_pca_retained"]
    selected_pca = pca_results["selected_pca"]
    selected_candidates = pca_results["selected_candidates"]
    n_retained = pca_results["n_retained"]
    retained_variance = pca_results["retained_variance"]
    high_y_mask = pca_results["high_y_mask"]
    best_observed_idx_pca = pca_results["best_observed_idx_pca"]
    retained_pc_names = pca_results["retained_pc_names"]
    retained_explained = pca_results["retained_explained"]

    # ------------------------------------------------------
    # 1. Observed-region summary
    # ------------------------------------------------------
    high_X_pca = X_pca_retained[high_y_mask]
    high_y_values = y_pca_source[high_y_mask]

    all_centroid_pca = np.mean(X_pca_retained, axis=0)
    high_centroid_pca = np.mean(high_X_pca, axis=0)
    best_observed_pca = X_pca_retained[best_observed_idx_pca]

    high_distances_to_centroid = np.linalg.norm(
        high_X_pca - high_centroid_pca,
        axis=1,
    )
    high_region_mean_radius = float(np.mean(high_distances_to_centroid))
    high_region_median_radius = float(np.median(high_distances_to_centroid))
    high_region_max_radius = float(np.max(high_distances_to_centroid))

    high_y_threshold_report = float(np.quantile(y_pca_source, 0.75))

    print("\n1. HIGH-PERFORMING OBSERVED REGION")
    print(f"Observed samples: {len(X_pca_source)}")
    print(f"Top-25% threshold: y >= {high_y_threshold_report:.6f}")
    print(f"High-y samples: {int(np.sum(high_y_mask))}")
    print(f"Retained PCA subspace: PC1-PC{n_retained} ({retained_variance:.1%} variance)")
    print(f"High-y region mean radius from its centroid:   {high_region_mean_radius:.4f}")
    print(f"High-y region median radius from its centroid: {high_region_median_radius:.4f}")
    print(f"High-y region maximum radius from its centroid:{high_region_max_radius:.4f}")

    centroid_table = pd.DataFrame({
        "Principal Component": retained_pc_names,
        "All-observation centroid": all_centroid_pca,
        "Top-25% y centroid": high_centroid_pca,
        "Best observed": best_observed_pca,
        "High-y centroid shift": high_centroid_pca - all_centroid_pca,
    })
    display(centroid_table.style.format({
        "All-observation centroid": "{:.4f}",
        "Top-25% y centroid": "{:.4f}",
        "Best observed": "{:.4f}",
        "High-y centroid shift": "{:.4f}",
    }))

    # ------------------------------------------------------
    # 2. PC association with observed y
    # ------------------------------------------------------
    # PCA is unsupervised, so these correlations are post-hoc diagnostics only.
    pc_y_rows = []
    for i, pc_name in enumerate(retained_pc_names):
        pc_values = X_pca_retained[:, i]
        pearson = float(pd.Series(pc_values).corr(pd.Series(y_pca_source), method="pearson"))
        spearman = float(pd.Series(pc_values).corr(pd.Series(y_pca_source), method="spearman"))

        high_mean = float(np.mean(pc_values[high_y_mask]))
        other_mean = float(np.mean(pc_values[~high_y_mask])) if np.any(~high_y_mask) else np.nan
        pooled_std = float(np.std(pc_values, ddof=1)) if len(pc_values) > 1 else np.nan
        standardized_shift = (
            (high_mean - other_mean) / pooled_std
            if np.isfinite(pooled_std) and pooled_std > 0 and np.isfinite(other_mean)
            else np.nan
        )

        pc_y_rows.append({
            "PC": pc_name,
            "Explained variance": float(retained_explained[i]),
            "Pearson corr with y": pearson,
            "Spearman corr with y": spearman,
            "Top-25% mean PC score": high_mean,
            "Other-observation mean PC score": other_mean,
            "Standardized high-y shift": standardized_shift,
        })

    pc_y_diagnostics = pd.DataFrame(pc_y_rows)
    pc_y_diagnostics["|Spearman| rank"] = (
        pc_y_diagnostics["Spearman corr with y"].abs()
        .rank(ascending=False, method="min")
        .astype(int)
    )
    pc_y_diagnostics = pc_y_diagnostics.sort_values("|Spearman| rank")

    print("\n2. RETAINED-PC ASSOCIATION WITH OBSERVED OUTPUT")
    print("Note: PCA itself does not use y; these are post-hoc associations, not causal effects.")
    display(pc_y_diagnostics.style.format({
        "Explained variance": "{:.1%}",
        "Pearson corr with y": "{:+.3f}",
        "Spearman corr with y": "{:+.3f}",
        "Top-25% mean PC score": "{:+.4f}",
        "Other-observation mean PC score": "{:+.4f}",
        "Standardized high-y shift": "{:+.3f}",
    }))

    # ------------------------------------------------------
    # 3. Candidate relationship to observed high-y region
    # ------------------------------------------------------
    candidate_region_rows = []
    high_X_original = X_pca_source[high_y_mask]
    all_centroid_original = np.mean(X_pca_source, axis=0)
    high_centroid_original = np.mean(high_X_original, axis=0)
    best_observed_original = X_pca_source[best_observed_idx_pca]

    for name, coords in selected_pca.items():
        original = selected_candidates[name]

        pca_to_high_centroid = float(np.linalg.norm(coords - high_centroid_pca))
        pca_to_best = float(np.linalg.norm(coords - best_observed_pca))
        pca_to_all_centroid = float(np.linalg.norm(coords - all_centroid_pca))
        pca_nearest_high = float(np.min(np.linalg.norm(high_X_pca - coords, axis=1)))
        pca_nearest_observed = float(np.min(np.linalg.norm(X_pca_retained - coords, axis=1)))

        original_to_high_centroid = float(np.linalg.norm(original - high_centroid_original))
        original_to_best = float(np.linalg.norm(original - best_observed_original))
        original_to_all_centroid = float(np.linalg.norm(original - all_centroid_original))
        original_nearest_high = float(np.min(np.linalg.norm(high_X_original - original, axis=1)))
        original_nearest_observed = float(np.min(np.linalg.norm(X_pca_source - original, axis=1)))

        # <1 means the candidate is closer to the high-y centroid than the
        # average top-quartile observation is, relative to the high-y region radius.
        normalized_high_region_distance = (
            pca_to_high_centroid / high_region_mean_radius
            if high_region_mean_radius > 0 else np.nan
        )

        candidate_region_rows.append({
            "Method": name,
            "PCA dist to high-y centroid": pca_to_high_centroid,
            "PCA dist to best observed": pca_to_best,
            "PCA nearest high-y point": pca_nearest_high,
            "PCA nearest observed point": pca_nearest_observed,
            "PCA dist to all-observation centroid": pca_to_all_centroid,
            "High-y radius units": normalized_high_region_distance,
            "Original dist to high-y centroid": original_to_high_centroid,
            "Original dist to best observed": original_to_best,
            "Original nearest high-y point": original_nearest_high,
            "Original nearest observed point": original_nearest_observed,
            "Original dist to all-observation centroid": original_to_all_centroid,
        })

    candidate_region_diagnostics = pd.DataFrame(candidate_region_rows)
    if not candidate_region_diagnostics.empty:
        candidate_region_diagnostics["High-y proximity rank"] = (
            candidate_region_diagnostics["PCA dist to high-y centroid"]
            .rank(ascending=True, method="min")
            .astype(int)
        )
        candidate_region_diagnostics = candidate_region_diagnostics.sort_values("High-y proximity rank")

    print("\n3. MODEL CANDIDATE PROXIMITY TO THE HIGH-PERFORMING REGION")
    print("'High-y radius units' compares candidate distance to the top-25% centroid with the mean radius of that region.")
    display(candidate_region_diagnostics.style.format({
        col: "{:.4f}"
        for col in candidate_region_diagnostics.columns
        if col not in ["Method", "High-y proximity rank"]
    }))

    # ------------------------------------------------------
    # 4. Candidate-to-candidate agreement matrices
    # ------------------------------------------------------
    candidate_names = list(selected_pca.keys())

    pca_pairwise = pd.DataFrame(
        np.zeros((len(candidate_names), len(candidate_names))),
        index=candidate_names,
        columns=candidate_names,
    )
    original_pairwise = pca_pairwise.copy()

    for name_a in candidate_names:
        for name_b in candidate_names:
            pca_pairwise.loc[name_a, name_b] = np.linalg.norm(
                selected_pca[name_a] - selected_pca[name_b]
            )
            original_pairwise.loc[name_a, name_b] = np.linalg.norm(
                selected_candidates[name_a] - selected_candidates[name_b]
            )

    print(f"\n4. CROSS-MODEL AGREEMENT: PAIRWISE DISTANCE IN PC1-PC{n_retained}")
    display(pca_pairwise.style.format("{:.4f}"))

    print("\nOriginal-space candidate pairwise distance:")
    display(original_pairwise.style.format("{:.4f}"))

    # Mean distance to all other model candidates gives a simple consensus score.
    consensus_rows = []
    for name in candidate_names:
        other_names = [other for other in candidate_names if other != name]
        mean_pca_distance = (
            float(np.mean([pca_pairwise.loc[name, other] for other in other_names]))
            if other_names else 0.0
        )
        mean_original_distance = (
            float(np.mean([original_pairwise.loc[name, other] for other in other_names]))
            if other_names else 0.0
        )
        consensus_rows.append({
            "Method": name,
            "Mean PCA distance to other models": mean_pca_distance,
            "Mean original distance to other models": mean_original_distance,
        })

    model_consensus = pd.DataFrame(consensus_rows)
    model_consensus["PCA consensus rank"] = (
        model_consensus["Mean PCA distance to other models"]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    model_consensus = model_consensus.sort_values("PCA consensus rank")

    print("\nModel consensus summary (smaller mean distance = greater agreement with the other models):")
    display(model_consensus.style.format({
        "Mean PCA distance to other models": "{:.4f}",
        "Mean original distance to other models": "{:.4f}",
    }))

    # ------------------------------------------------------
    # 5. Per-PC candidate deviation from the high-y centroid
    # ------------------------------------------------------
    high_pc_std = np.std(high_X_pca, axis=0, ddof=1) if len(high_X_pca) > 1 else np.full(n_retained, np.nan)
    per_pc_rows = []

    for name, coords in selected_pca.items():
        for i, pc_name in enumerate(retained_pc_names):
            diff = float(coords[i] - high_centroid_pca[i])
            std_units = (
                diff / high_pc_std[i]
                if np.isfinite(high_pc_std[i]) and high_pc_std[i] > 0
                else np.nan
            )
            per_pc_rows.append({
                "Method": name,
                "PC": pc_name,
                "Candidate score": float(coords[i]),
                "High-y centroid": float(high_centroid_pca[i]),
                "Difference": diff,
                "Difference in high-y SD units": std_units,
            })

    candidate_pc_deviation = pd.DataFrame(per_pc_rows)
    print("\n5. PER-PC CANDIDATE DEVIATION FROM THE TOP-25% CENTROID")
    display(candidate_pc_deviation.style.format({
        "Candidate score": "{:+.4f}",
        "High-y centroid": "{:+.4f}",
        "Difference": "{:+.4f}",
        "Difference in high-y SD units": "{:+.2f}",
    }))

    # ------------------------------------------------------
    # 6. Compact visual diagnostics
    # ------------------------------------------------------
    if not candidate_region_diagnostics.empty:
        plot_order = candidate_region_diagnostics.sort_values(
            "PCA dist to high-y centroid"
        )
        plt.figure(figsize=(8, 4.5))
        plt.bar(
            plot_order["Method"],
            plot_order["PCA dist to high-y centroid"],
        )
        plt.axhline(
            high_region_mean_radius,
            linestyle="--",
            linewidth=1.2,
            label="Mean top-25% region radius",
        )
        plt.ylabel(f"Distance in PC1-PC{n_retained} space")
        plt.title(f"Function {load_dataset}: candidate distance to high-y PCA centroid")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.show()

    pc_plot_order = pc_y_diagnostics.sort_values("PC")
    plt.figure(figsize=(8, 4.5))
    x_positions = np.arange(len(pc_plot_order))
    width = 0.36
    plt.bar(
        x_positions - width / 2,
        pc_plot_order["Pearson corr with y"],
        width=width,
        label="Pearson",
    )
    plt.bar(
        x_positions + width / 2,
        pc_plot_order["Spearman corr with y"],
        width=width,
        label="Spearman",
    )
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x_positions, pc_plot_order["PC"])
    plt.ylabel("Correlation with observed y")
    plt.title(f"Function {load_dataset}: retained-PC association with observed output")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.show()

    # ------------------------------------------------------
    # 7. Automated diagnostic interpretation summary
    # ------------------------------------------------------
    print("\n6. PCA DIAGNOSTIC INTERPRETATION SUMMARY")
    print("--------------------------------------------------")

    if not candidate_region_diagnostics.empty:
        nearest_method = candidate_region_diagnostics.iloc[0]["Method"]
        nearest_distance = candidate_region_diagnostics.iloc[0]["PCA dist to high-y centroid"]
        nearest_radius_units = candidate_region_diagnostics.iloc[0]["High-y radius units"]
        print(
            f"Closest model candidate to the top-25% PCA centroid: {nearest_method} "
            f"(distance {nearest_distance:.4f}, {nearest_radius_units:.2f} high-y mean-radius units)."
        )

    if len(model_consensus) > 0:
        consensus_method = model_consensus.iloc[0]["Method"]
        consensus_distance = model_consensus.iloc[0]["Mean PCA distance to other models"]
        print(
            f"Greatest cross-model consensus: {consensus_method} "
            f"(mean retained-PCA distance to other model candidates {consensus_distance:.4f})."
        )

    strongest_pc_row = pc_y_diagnostics.iloc[0]
    print(
        f"Strongest monotonic PC-y association among retained components: "
        f"{strongest_pc_row['PC']} "
        f"(Spearman {strongest_pc_row['Spearman corr with y']:+.3f}; "
        f"explained variance {strongest_pc_row['Explained variance']:.1%})."
    )

    if high_region_mean_radius > 0:
        print(
            f"Top-25% observed points have a mean PCA radius of {high_region_mean_radius:.4f}; "
            "use this as a rough scale when judging whether a candidate lies inside, near, or outside "
            "the established high-performing region."
        )

    print(
        "Interpretation caution: PCA is fitted only to X and is unsupervised. PC-y correlations, "
        "centroid distances, and proximity ranks describe the current observed sample; they do not "
        "prove causality or guarantee that the nearest candidate will produce the highest next y."
    )

    return {
        "high_y_threshold": high_y_threshold_report,
        "all_centroid_pca": all_centroid_pca,
        "high_centroid_pca": high_centroid_pca,
        "best_observed_pca": best_observed_pca,
        "high_region_mean_radius": high_region_mean_radius,
        "high_region_median_radius": high_region_median_radius,
        "high_region_max_radius": high_region_max_radius,
        "centroid_table": centroid_table,
        "pc_y_diagnostics": pc_y_diagnostics,
        "candidate_region_diagnostics": candidate_region_diagnostics,
        "pca_pairwise": pca_pairwise,
        "original_pairwise": original_pairwise,
        "model_consensus": model_consensus,
        "candidate_pc_deviation": candidate_pc_deviation,
    }

