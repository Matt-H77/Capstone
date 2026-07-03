import numpy as np
import pandas as pd


def build_candidate_comparison_table(
    candidates,
    best_idx,
    best_input_thompson,
    nn_candidate,
    ensemble_mean,
    ensemble_std,
    ei_final,
    pi_final,
    ucb_final,
    svm_candidate_scores=None,
    include_svm=True,
):
    """
    Build a comparison table for GP, Thompson, NN and optional SVM candidates.
    """

    candidates = np.asarray(candidates)

    gp_idx = int(best_idx)

    thompson_idx = int(
        np.argmin(np.linalg.norm(candidates - best_input_thompson, axis=1))
    )

    nn_idx = int(
        np.argmin(np.linalg.norm(candidates - nn_candidate, axis=1))
    )

    if include_svm and svm_candidate_scores is not None:
        svm_idx = int(np.argmax(svm_candidate_scores))
        svm_ranks = (
            pd.Series(svm_candidate_scores)
            .rank(ascending=False, method="min")
            .astype(int)
        )
    else:
        svm_idx = None
        svm_ranks = None

    gp_mean_ranks = (
        pd.Series(ensemble_mean)
        .rank(ascending=False, method="min")
        .astype(int)
    )

    uncertainty_ranks = (
        pd.Series(ensemble_std)
        .rank(ascending=False, method="min")
        .astype(int)
    )

    def candidate_summary(method_name, idx):
        row = {
            "Method": method_name,
            "Candidate Index": idx,
            "Candidate Input": np.round(candidates[idx], 3),

            "GP Mean": float(ensemble_mean[idx]),
            "GP Std": float(ensemble_std[idx]),

            "EI": float(ei_final[idx]),
            "PI": float(pi_final[idx]),
            "UCB": float(ucb_final[idx]),

            "Distance to GP": float(
                np.linalg.norm(candidates[idx] - candidates[gp_idx])
            ),

            "GP Mean Rank": int(gp_mean_ranks.iloc[idx]),
            "Uncertainty Rank": int(uncertainty_ranks.iloc[idx]),
        }

        if include_svm and svm_candidate_scores is not None:
            row["SVM Score"] = float(svm_candidate_scores[idx])
            row["SVM Rank"] = int(svm_ranks.iloc[idx])

        return row

    rows = [
        candidate_summary("Final GP Choice", gp_idx),
        candidate_summary("Thompson Sampling", thompson_idx),
        candidate_summary("Neural Network", nn_idx),
    ]

    if svm_idx is not None:
        rows.append(candidate_summary("SVM", svm_idx))

    comparison_table = pd.DataFrame(rows)

    comparison_table = comparison_table.sort_values(
        "GP Mean",
        ascending=False
    ).reset_index(drop=True)

    #numeric_cols = [
    #    "GP Mean",
    #    "GP Std",
    #    "EI",
    #    "PI",
    #    "UCB",
    #    "Distance to GP",
    #]

    #if "SVM Score" in comparison_table.columns:
    #    numeric_cols.append("SVM Score")

    #comparison_table[numeric_cols] = comparison_table[numeric_cols].round(4)

    #return comparison_table

    best_gp_mean_value = comparison_table["GP Mean"].max()
    gp_std_reference = comparison_table.loc[
        comparison_table["Method"] == "Final GP Choice", "GP Std"
    ].iloc[0]

    comparison_table["GP Mean Loss"] = (
        best_gp_mean_value - comparison_table["GP Mean"]
    )

    comparison_table["Loss per Distance"] = np.where(
        comparison_table["Distance to GP"] > 0,
        comparison_table["GP Mean Loss"] / comparison_table["Distance to GP"],
        0.0
    )

    comparison_table["Uncertainty Gain vs GP"] = (
        comparison_table["GP Std"] - gp_std_reference
    )

    comparison_table["Exploration per Loss"] = np.where(
        comparison_table["Uncertainty Gain vs GP"] > 0,
        comparison_table["Uncertainty Gain vs GP"] / comparison_table["GP Mean Loss"],
        0.0
    )

    comparison_table["Mean-Std Tradeoff"] = (
        comparison_table["GP Mean"] + comparison_table["GP Std"]
    )

    comparison_table["Candidate Role"] = np.select(
    [
        comparison_table["Method"].eq("Final GP Choice"),
        (comparison_table["GP Mean Loss"] <= 0.1) & (comparison_table["Distance to GP"] > 0.5),
        comparison_table["Uncertainty Gain vs GP"] > 0.05,
    ],
    [
        "Exploitation baseline",
        "Low-cost diverse alternative",
        "Exploratory candidate",
    ],
    default="Alternative candidate"
)

    numeric_cols = [
        "GP Mean",
        "GP Std",
        "EI",
        "PI",
        "UCB",
        "Distance to GP",
        "GP Mean Loss",
        "Loss per Distance",
        "Uncertainty Gain vs GP",
        "Exploration per Loss",
        "Mean-Std Tradeoff",
        "Candidate Role",
    ]

    if "SVM Score" in comparison_table.columns:
        numeric_cols.append("SVM Score")

    comparison_table[numeric_cols] = comparison_table[numeric_cols].round(4)

    return comparison_table