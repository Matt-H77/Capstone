
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Configuration
# ============================================================

JSON_DIR = Path("outputs/json")

st.set_page_config(
    page_title="BBO Optimisation Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("Black-Box Bayesian Optimisation Dashboard")
st.caption(
    "Human-facing results use the original/raw black-box objective. "
    "Transformed optimisation-space values are retained for diagnostics."
)


# ============================================================
# Helpers
# ============================================================

def load_json_files():
    files = sorted(JSON_DIR.glob("function_*_week_*.json"))
    runs = []

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["_source_file"] = file.name
            runs.append(data)

        except Exception as exc:
            st.warning(f"Could not load {file.name}: {exc}")

    return runs


def safe_float(value, digits=6):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def vector_dataframe(values, prefix="x"):
    if values is None:
        return pd.DataFrame()

    return pd.DataFrame(
        [values],
        columns=[f"{prefix}{i + 1}" for i in range(len(values))],
    )


def objective_label(direction):
    if direction == "minimise":
        return "Minimise ↓"
    if direction == "maximise":
        return "Maximise ↑"
    if direction == "maximise_transformed":
        return "Maximise transformed objective ↑"
    return direction or "Unknown"


def get_best_raw_output(run):
    best = run.get("best_observed", {})
    if "raw_output" in best:
        return best.get("raw_output")
    # Backward compatibility with older JSON files
    return best.get("output")


def get_best_transformed_output(run):
    best = run.get("best_observed", {})
    return best.get("transformed_output")


def get_objective_direction(run):
    objective = run.get("objective", {})
    best = run.get("best_observed", {})

    return (
        objective.get("direction")
        or best.get("objective_direction")
        or "unknown"
    )


def get_gp_predicted_raw(run):
    candidate = run.get("gp_hybrid_candidate", {})

    if "predicted_output" in candidate:
        return candidate.get("predicted_output")

    # Backward compatibility
    return candidate.get("predicted_mean")


def get_gp_predicted_transformed(run):
    candidate = run.get("gp_hybrid_candidate", {})

    if "predicted_output_transformed" in candidate:
        return candidate.get("predicted_output_transformed")

    return candidate.get("predicted_mean")


def get_submission_candidate(run):
    """
    Return the actual candidate intended for submission.

    If a special final submission candidate exists, use it.
    Otherwise fall back to the normal GP hybrid candidate.
    """
    final_candidate = run.get("final_submission_candidate")

    if final_candidate:
        return final_candidate

    return run.get("gp_hybrid_candidate", {})


def get_submission_method(run):
    final_candidate = run.get("final_submission_candidate")

    if final_candidate:
        return final_candidate.get(
            "method",
            "Final submission",
        )

    return "GP Hybrid"


def get_submission_predicted_raw(run):
    """
    Return the final submission prediction in original/raw objective space.

    Special final submission candidates may contain either a raw prediction
    or an optimisation-space prediction. When only the transformed value is
    available, convert it back consistently with the history chart logic.
    """
    function_id = run.get("run", {}).get("function")
    candidate = get_submission_candidate(run)

    if candidate.get("predicted_output") is not None:
        return candidate.get("predicted_output")

    pred = candidate.get("predicted_output_transformed")

    if pred is None:
        pred = candidate.get("predicted_mean")

    if pred is None:
        return None

    pred = float(pred)

    if function_id in [3, 6]:
        return -pred

    if function_id == 1:
        return float(np.expm1(pred))

    return pred


def get_gp_prediction_for_history(run):
    """
    Return the GP candidate prediction in a form comparable with the
    raw/original objective used by the history chart.

    Function 1:
        y = log1p(abs(raw_y))
        -> recover predicted magnitude with expm1(y)

    Functions 3 and 6:
        y = -raw_y
        -> recover raw prediction with -y

    Other functions:
        transformed and raw spaces are the same.
    """
    function_id = run.get("run", {}).get("function")
    candidate = run.get("gp_hybrid_candidate", {})

    # Prefer the internal/transformed prediction because this lets us
    # apply the inverse consistently even for older JSON exports.
    pred = candidate.get("predicted_output_transformed")

    if pred is None:
        pred = candidate.get("predicted_mean")

    if pred is None:
        pred = candidate.get("predicted_output")

    if pred is None:
        return None

    pred = float(pred)

    if function_id in [3, 6]:
        return -pred

    if function_id == 1:
        return float(np.expm1(pred))

    return pred


def build_summary_dataframe(runs):
    rows = []

    for run in runs:
        run_info = run.get("run", {})
        strategy = run.get("strategy", {})

        rows.append(
            {
                "Function": run_info.get("function"),
                "Week": run_info.get("week"),
                "Objective": objective_label(
                    get_objective_direction(run)
                ),
                "Samples": run_info.get("samples"),
                "Dimensions": run_info.get("dimensions"),
                "Best Raw Output": get_best_raw_output(run),
                "Progress": strategy.get("progress"),
                "Exploration": strategy.get("exploration_weight"),
                "Exploitation": strategy.get("exploitation_weight"),
                "File": run.get("_source_file"),
            }
        )

    return pd.DataFrame(rows)


def build_best_values_dataframe(runs):
    """Return the latest cumulative best value for each function."""
    latest_runs = {}

    for run in runs:
        function_id = run.get("run", {}).get("function")
        week = run.get("run", {}).get("week")

        if function_id is None:
            continue

        if (
            function_id not in latest_runs
            or week is not None
            and week > latest_runs[function_id].get("run", {}).get("week", -1)
        ):
            latest_runs[function_id] = run

    rows = [
        {
            "Function": function_id,
            "Best Value": get_best_raw_output(run),
        }
        for function_id, run in sorted(latest_runs.items())
    ]

    return pd.DataFrame(rows)


# ============================================================
# Load data
# ============================================================

runs = load_json_files()

if not runs:
    st.warning(
        "No JSON optimisation files were found.\n\n"
        "Expected files such as:\n"
        "`outputs/json/function_2_week_12.json`"
    )
    st.stop()

summary_df = build_summary_dataframe(runs)


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.header("Dashboard Controls")

available_functions = sorted(
    int(x)
    for x in summary_df["Function"].dropna().unique()
)

selected_function = st.sidebar.selectbox(
    "Function",
    available_functions,
    format_func=lambda x: f"Function {x}",
)

function_runs = [
    run
    for run in runs
    if run.get("run", {}).get("function") == selected_function
]

available_weeks = sorted(
    int(run.get("run", {}).get("week"))
    for run in function_runs
    if run.get("run", {}).get("week") is not None
)

selected_week = st.sidebar.selectbox(
    "Week",
    available_weeks,
    index=len(available_weeks) - 1,
)

selected_run = next(
    run
    for run in function_runs
    if run.get("run", {}).get("week") == selected_week
)


# ============================================================
# Overview across all functions
# ============================================================

st.subheader("Best Value by Function")

best_values_df = build_best_values_dataframe(runs)

best_values_col, _ = st.columns([1, 1])

centred_best_values_df = (
    best_values_df.style
    .set_properties(**{"text-align": "center"})
    .set_table_styles(
        [
            {
                "selector": "th",
                "props": [("text-align", "center")],
            }
        ]
    )
)

with best_values_col:
    st.dataframe(
        centred_best_values_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Function": st.column_config.NumberColumn(
                "Function",
                format="%d",
            ),
            "Best Value": st.column_config.NumberColumn(
                "Best Value",
                format="%.6f",
            ),
        },
    )

st.subheader("All Functions")

overview_df = summary_df.copy()

for col in ["Progress", "Exploration", "Exploitation"]:
    if col in overview_df.columns:
        overview_df[col] = overview_df[col].apply(
            lambda x: f"{float(x) * 100:.1f}%"
            if pd.notna(x)
            else ""
        )

st.dataframe(
    overview_df.sort_values(["Function", "Week"]),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# Selected run headline metrics
# ============================================================

run_info = selected_run.get("run", {})
best = selected_run.get("best_observed", {})
strategy = selected_run.get("strategy", {})
gp_candidate = selected_run.get("gp_hybrid_candidate", {})

submission_candidate = get_submission_candidate(
    selected_run
)

submission_method = get_submission_method(
    selected_run
)

objective_direction = get_objective_direction(selected_run)

st.divider()

st.header(
    f"Function {selected_function} — Week {selected_week}"
)

metric_cols = st.columns(6)

metric_cols[0].metric(
    "Objective",
    objective_label(objective_direction),
)

metric_cols[1].metric(
    "Samples",
    run_info.get("samples", "N/A"),
)

metric_cols[2].metric(
    "Dimensions",
    run_info.get("dimensions", "N/A"),
)

metric_cols[3].metric(
    "Best Raw Output",
    safe_float(get_best_raw_output(selected_run)),
)

progress = strategy.get("progress")
metric_cols[4].metric(
    "Progress",
    f"{float(progress) * 100:.1f}%"
    if progress is not None
    else "N/A",
)

metric_cols[5].metric(
    "Final Predicted Output",
    safe_float(
        get_submission_predicted_raw(
            selected_run
        )
    ),
)


# ============================================================
# Tabs
# ============================================================

tab_overview, tab_history, tab_candidates, tab_strategy, tab_gp, tab_pca, tab_diag = st.tabs(
    [
        "Overview",
        "History",
        "Candidates",
        "Strategy",
        "GP Models",
        "PCA",
        "Diagnostics",
    ]
)


# ============================================================
# Overview tab
# ============================================================

with tab_overview:

    st.subheader("Best Observed Input")

    best_input_df = vector_dataframe(
        best.get("input")
    )

    if not best_input_df.empty:
        st.dataframe(
            best_input_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No best-observed input was found in this JSON file.")

    cols = st.columns(3)

    cols[0].metric(
        "Best Raw Output",
        safe_float(get_best_raw_output(selected_run)),
    )

    cols[1].metric(
        "Transformed Optimiser Output",
        safe_float(get_best_transformed_output(selected_run)),
    )

    cols[2].metric(
        "Objective",
        objective_label(objective_direction),
    )

    st.divider()

    st.subheader("Final Submission")

    st.write(
        f"**Selection method:** {submission_method}"
    )

    submission_input_df = vector_dataframe(
        submission_candidate.get("input")
    )

    if not submission_input_df.empty:

        submission_column_config = {
            col: st.column_config.NumberColumn(
                col,
                format="%.6f",
            )
            for col in submission_input_df.columns
        }

        st.dataframe(
            submission_input_df,
            use_container_width=True,
            hide_index=True,
            column_config=submission_column_config,
        )
    else:
        st.info(
            "No final submission input was found."
        )

    final_cols = st.columns(3)

    final_cols[0].metric(
        "Predicted Output",
        safe_float(
            get_submission_predicted_raw(
                selected_run
            )
        ),
    )

    final_cols[1].metric(
        "Predicted Std",
        safe_float(
            submission_candidate.get(
                "predicted_std"
            )
        ),
    )

    final_cols[2].metric(
        "Distance From Best Observed",
        safe_float(
            submission_candidate.get(
                "distance_from_best_observed"
            )
        ),
    )

    selection_reason = submission_candidate.get(
        "selection_reason"
    )

    if selection_reason:
        st.info(selection_reason)

    st.divider()

    st.subheader("GP Hybrid Recommendation")

    gp_input_df = vector_dataframe(
        gp_candidate.get("input")
    )

    if not gp_input_df.empty:
        st.dataframe(
            gp_input_df,
            use_container_width=True,
            hide_index=True,
        )

    cols = st.columns(6)

    cols[0].metric(
        "Predicted Raw Output",
        safe_float(get_gp_predicted_raw(selected_run)),
    )
    cols[1].metric(
        "Predicted Std",
        safe_float(gp_candidate.get("predicted_std")),
    )
    cols[2].metric(
        "EI",
        safe_float(gp_candidate.get("ei")),
    )
    cols[3].metric(
        "UCB",
        safe_float(gp_candidate.get("ucb")),
    )
    cols[4].metric(
        "PI",
        safe_float(gp_candidate.get("pi")),
    )
    cols[5].metric(
        "Final Score",
        safe_float(gp_candidate.get("final_score")),
    )


# ============================================================
# History tab
# ============================================================

with tab_history:

    history_rows = []

    for run in function_runs:
        info = run.get("run", {})
        strat = run.get("strategy", {})

        history_rows.append(
            {
                "Week": info.get("week"),
                "Samples": info.get("samples"),
                "Objective": objective_label(
                    get_objective_direction(run)
                ),
                "Best Raw Output": get_best_raw_output(run),
                "GP Candidate Prediction": get_gp_prediction_for_history(run),
                "Best Transformed Output": get_best_transformed_output(run),
                "Progress": strat.get("progress"),
                "Exploration": strat.get("exploration_weight"),
                "Exploitation": strat.get("exploitation_weight"),
            }
        )

    history_df = (
        pd.DataFrame(history_rows)
        .sort_values("Week")
        .reset_index(drop=True)
    )

    st.subheader("Optimisation Progress")

    history_chart_cols = [
        col
        for col in [
            "Best Raw Output",
            "GP Candidate Prediction",
        ]
        if col in history_df.columns
        and history_df[col].notna().any()
    ]

    if history_chart_cols:
        plot_df = history_df[
            ["Week"] + history_chart_cols
        ].melt(
            id_vars="Week",
            var_name="Series",
            value_name="Output",
        )

        plot_df = plot_df.dropna(subset=["Output"])

        y_min = plot_df["Output"].min()
        y_max = plot_df["Output"].max()

        if pd.notna(y_min) and pd.notna(y_max):
            y_range = y_max - y_min

            if y_range == 0:
                padding = abs(y_min) * 0.05
                if padding == 0:
                    padding = 0.01
            else:
                padding = y_range * 0.10

            y_domain = [
                float(y_min - padding),
                float(y_max + padding),
            ]

            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "Week:O",
                        title="Week",
                    ),
                    y=alt.Y(
                        "Output:Q",
                        title="Objective Output",
                        scale=alt.Scale(
                            domain=y_domain,
                            zero=False,
                        ),
                    ),
                    color=alt.Color(
                        "Series:N",
                        title="",
                    ),
                    tooltip=[
                        alt.Tooltip("Week:O", title="Week"),
                        alt.Tooltip("Series:N", title="Series"),
                        alt.Tooltip(
                            "Output:Q",
                            title="Output",
                            format=".6f",
                        ),
                    ],
                )
                .properties(
                    title="Optimisation Progress — autoscaled Y-axis",
                    height=450,
                )
            )

            st.altair_chart(
                chart,
                use_container_width=True,
            )

        if selected_function == 1:
            st.caption(
                "For Function 1, the GP history curve is the predicted raw-output "
                "magnitude, reconstructed with expm1() from log1p(abs(raw_y)). "
                "The original sign cannot be recovered because abs() removed it. "
                "The Y-axis is autoscaled to make small changes easier to see."
            )
        elif selected_function in [3, 6]:
            st.caption(
                "For Functions 3 and 6, the GP history curve is converted back "
                "to original minimisation space by negating the transformed GP prediction. "
                "The Y-axis is autoscaled to make small changes easier to see."
            )
        else:
            st.caption(
                "The Y-axis is autoscaled to the visible history range so that "
                "smaller optimisation changes are easier to see."
            )

    st.subheader("Run History")

    display_history = history_df.copy()

    for col in ["Progress", "Exploration", "Exploitation"]:
        display_history[col] = display_history[col].apply(
            lambda x: f"{float(x) * 100:.1f}%"
            if pd.notna(x)
            else ""
        )

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Candidates tab
# ============================================================

with tab_candidates:

    st.subheader("Candidate Comparison")

    comparison = selected_run.get("candidate_comparison")

    if comparison:
        comparison_df = pd.DataFrame(comparison)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No candidate comparison table was exported.")

    st.subheader("Individual Candidate Inputs")

    candidate_sections = {}

    if selected_run.get("final_submission_candidate"):
        candidate_sections["Final Submission"] = selected_run.get(
            "final_submission_candidate",
            {},
        )

    candidate_sections.update(
        {
            "GP Hybrid": selected_run.get("gp_hybrid_candidate", {}),
            "Thompson": selected_run.get("thompson_candidate", {}),
            "Neural Network": selected_run.get("neural_network_candidate", {}),
            "SVM": selected_run.get("svm_candidate", {}),
        }
    )

    for name, candidate in candidate_sections.items():
        with st.expander(
            name,
            expanded=(
                name == "Final Submission"
                if selected_run.get("final_submission_candidate")
                else name == "GP Hybrid"
            ),
        ):
            if candidate.get("input") is not None:

                candidate_input_df = vector_dataframe(
                    candidate.get("input")
                )

                candidate_column_config = {
                    col: st.column_config.NumberColumn(
                        col,
                        format="%.6f",
                    )
                    for col in candidate_input_df.columns
                }

                st.dataframe(
                    candidate_input_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=candidate_column_config,
                )

            additional = {
                key: value
                for key, value in candidate.items()
                if key != "input"
            }

            if additional:
                st.json(additional)


# ============================================================
# Strategy tab
# ============================================================

with tab_strategy:

    st.subheader("Exploration vs Exploitation")

    exploration = strategy.get("exploration_weight")
    exploitation = strategy.get("exploitation_weight")

    cols = st.columns(4)

    cols[0].metric(
        "Progress",
        f"{float(progress) * 100:.1f}%"
        if progress is not None
        else "N/A",
    )

    cols[1].metric(
        "Exploration",
        f"{float(exploration) * 100:.1f}%"
        if exploration is not None
        else "N/A",
    )

    cols[2].metric(
        "Exploitation",
        f"{float(exploitation) * 100:.1f}%"
        if exploitation is not None
        else "N/A",
    )

    cols[3].metric(
        "Xi",
        safe_float(strategy.get("xi")),
    )

    candidate_pool = selected_run.get("candidate_pool", {})

    st.subheader("Candidate Pool")

    pool_cols = st.columns(2)

    pool_cols[0].metric(
        "Before SVM",
        candidate_pool.get("before_svm", "N/A"),
    )

    pool_cols[1].metric(
        "Final Candidates",
        candidate_pool.get("final", "N/A"),
    )


# ============================================================
# GP Models tab
# ============================================================

with tab_gp:

    st.subheader("Gaussian Process Ensemble")

    gp_models = selected_run.get("gp_models", {})

    if gp_models:

        rows = []

        for name, model in gp_models.items():
            rows.append(
                {
                    "Model": name,
                    "Fitted Kernel": model.get("kernel"),
                    "Log Marginal Likelihood":
                        model.get("log_marginal_likelihood"),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No GP model information was exported.")


# ============================================================
# PCA tab
# ============================================================

with tab_pca:

    st.subheader("PCA Diagnostics")

    pca = selected_run.get("pca", {})
    extended_pca = selected_run.get("extended_pca", {})

    if not pca and not extended_pca:
        st.info("No PCA results were exported for this run.")

    else:

        explained_variance = None

        for key in [
            "explained_variance_ratio",
            "explained_variance_ratio_",
            "explained_variance",
        ]:
            if isinstance(pca, dict) and key in pca:
                explained_variance = pca[key]
                break

        if isinstance(explained_variance, list) and explained_variance:

            variance_df = pd.DataFrame(
                {
                    "Principal Component": [
                        f"PC{i + 1}"
                        for i in range(len(explained_variance))
                    ],
                    "Explained Variance":
                        explained_variance,
                }
            )

            variance_df["Cumulative Variance"] = (
                variance_df["Explained Variance"]
                .cumsum()
            )

            st.subheader("Explained Variance")

            st.bar_chart(
                variance_df.set_index(
                    "Principal Component"
                )[["Explained Variance"]]
            )

            st.line_chart(
                variance_df.set_index(
                    "Principal Component"
                )[["Cumulative Variance"]]
            )

            st.dataframe(
                variance_df,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Raw PCA Export")

        with st.expander("PCA results"):
            st.json(pca)

        with st.expander("Extended PCA results"):
            st.json(extended_pca)


# ============================================================
# Diagnostics tab
# ============================================================

with tab_diag:

    st.subheader("Objective Transformation")

    objective = selected_run.get("objective", {})

    st.write(
        f"**Direction:** {objective_label(objective_direction)}"
    )

    st.write(
        f"**Output transform:** "
        f"`{objective.get('output_transform', best.get('output_transform', 'N/A'))}`"
    )

    diag_cols = st.columns(2)

    diag_cols[0].metric(
        "Best Raw Output",
        safe_float(get_best_raw_output(selected_run)),
    )

    diag_cols[1].metric(
        "Best Transformed Output",
        safe_float(get_best_transformed_output(selected_run)),
    )

    st.subheader("GP Prediction Spaces")

    gp_cols = st.columns(2)

    gp_cols[0].metric(
        "Predicted Raw Output",
        safe_float(get_gp_predicted_raw(selected_run)),
    )

    gp_cols[1].metric(
        "Predicted Transformed Output",
        safe_float(get_gp_predicted_transformed(selected_run)),
    )

    st.caption(
        "Raw values are intended for interpretation and presentation. "
        "Transformed values are the targets used internally by the optimiser "
        "and acquisition functions."
    )

    final_submission = selected_run.get(
        "final_submission_candidate"
    )
    local_gp_diagnostics = selected_run.get(
        "local_gp_diagnostics"
    )

    if final_submission:
        st.subheader("Final Submission Diagnostics")
        st.json(final_submission)

    if local_gp_diagnostics:
        st.subheader("Local GP Diagnostics")
        st.json(local_gp_diagnostics)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    f"Loaded from: {selected_run.get('_source_file', 'unknown file')}"
)
