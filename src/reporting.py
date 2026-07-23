# Function to print a detailed report of the automatic optimisation process, 
# including dataset summary, search strategy, GP training quality, acquisition 
# suggestions, final hybrid decision, cross-model agreement, convergence diagnostics, 
# decision type, interpretation, and optimiser confidence.  
def print_automatic_optimisation_report(
    X,
    y,
    trained_models,
    results,
    best_input,
    ensemble_mean,
    ensemble_std,
    best_idx,
    final_score,
    ei_final,
    ucb_final,
    pi_final,
    load_dataset,
    week_dataset,
    progress,
    **kwargs,
):
    import numpy as np

    # Helper functions for formatting and classification
    # Optional values such as exploration_weight, candidates, nn_candidate, etc.
    # are passed through **kwargs, not globals().
    def has(name):
        return name in kwargs and kwargs[name] is not None

    def get(name, default=None):
        return kwargs.get(name, default)

    # Helper function to format arrays for printing
    def arr(x):
        if x is None:
            return "None"
        return np.array2string(np.asarray(x), precision=6, suppress_small=True)

    # Helper function to classify search behaviour based on distance to best observed input
    def classify_search_behaviour(distance):
        if distance < 0.02:
            return "Strong local refinement"
        elif distance < 0.10:
            return "Local exploitation"
        elif distance < 0.30:
            return "Regional exploration / exploitation"
        else:
            return "Global exploration"

    # Helper function to label agreement based on distance
    def agreement_label(distance):
        if distance < 0.05:
            return "Very high"
        elif distance < 0.15:
            return "High"
        elif distance < 0.30:
            return "Moderate"
        else:
            return "Low"

    # Extract an ARD length-scale vector from a trained sklearn GP kernel.
    # This supports compound kernels such as ConstantKernel * RBF or
    # ConstantKernel * Matern.
    def extract_length_scales_from_kernel(kernel, n_dimensions):
        if kernel is None:
            return None

        # Search recursively through compound kernels.
        for attribute_name in ("k1", "k2"):
            child_kernel = getattr(kernel, attribute_name, None)

            if child_kernel is not None:
                result = extract_length_scales_from_kernel(
                    child_kernel,
                    n_dimensions
                )

                if result is not None:
                    return result

        if not hasattr(kernel, "length_scale"):
            return None

        length_scale = np.asarray(
            kernel.length_scale,
            dtype=float
        )

        # Scalar/isotropic kernels cannot provide dimension-specific
        # ARD distances.
        if length_scale.ndim == 0 or length_scale.size == 1:
            return None

        length_scale = length_scale.reshape(-1)

        if length_scale.size != n_dimensions:
            return None

        # Avoid division by zero or invalid scales.
        if not np.all(np.isfinite(length_scale)):
            return None

        if np.any(length_scale <= 0):
            return None

        return length_scale


    def select_reference_ard_length_scales(trained_models, n_dimensions):
        """
        Prefer Matern ARD scales, then RBF, then any available
        anisotropic GP kernel.
        """
        preferred_names = (
            "Matern",
            "Matérn",
            "RBF"
        )

        for preferred_name in preferred_names:
            for kernel_name, gp in trained_models.items():
                if preferred_name.lower() in kernel_name.lower():
                    length_scales = extract_length_scales_from_kernel(
                        getattr(gp, "kernel_", None),
                        n_dimensions
                    )

                    if length_scales is not None:
                        return kernel_name, length_scales

        for kernel_name, gp in trained_models.items():
            length_scales = extract_length_scales_from_kernel(
                getattr(gp, "kernel_", None),
                n_dimensions
            )

            if length_scales is not None:
                return kernel_name, length_scales

        return None, None


    def scaled_distance(point_a, point_b, length_scales):
        point_a = np.asarray(point_a, dtype=float)
        point_b = np.asarray(point_b, dtype=float)

        if length_scales is None:
            return float(
                np.linalg.norm(point_a - point_b)
            )

        length_scales = np.asarray(
            length_scales,
            dtype=float
        )

        return float(
            np.sqrt(
                np.sum(
                    (
                        (point_a - point_b)
                        / length_scales
                    ) ** 2
                )
            )
        )


    def scaled_nearest_sample_distance(samples, point, length_scales):
        samples = np.asarray(samples, dtype=float)
        point = np.asarray(point, dtype=float)

        if length_scales is None:
            return float(
                np.min(
                    np.linalg.norm(
                        samples - point,
                        axis=1
                    )
                )
            )

        scaled_differences = (
            samples - point
        ) / length_scales

        return float(np.min(np.linalg.norm(scaled_differences, axis=1)))
        
    # Helper function to label uncertainty based on standard deviation
    def uncertainty_label(std_value):
        if std_value < 0.001:
            return "Very low"
        elif std_value < 0.05:
            return "Low"
        elif std_value < 0.5:
            return "Moderate"
        else:
            return "High"

    # Helper function to label confidence based on score
    def confidence_label(score):
        if score >= 5.0:
            return "VERY HIGH"
        elif score >= 4.0:
            return "HIGH"
        elif score >= 2.5:
            return "MODERATE"
        elif score >= 1.5:
            return "LOW / MODERATE"
        else:
            return "LOW"

    # Helper function to extract top candidate inputs from results dictionary
    def extract_top_candidate_inputs(results_dict):
        inputs = []
        for _, result in results_dict.items():
            if isinstance(result, dict) and "input" in result:
                inputs.append(np.asarray(result["input"], dtype=float))
        if len(inputs) == 0:
            return None
        return np.vstack(inputs)

    # Helper function to compute mean pairwise distance between points
    def mean_pairwise_distance(points):
        if points is None or len(points) < 2:
            return None

        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distances.append(np.linalg.norm(points[i] - points[j]))

        return float(np.mean(distances)) if distances else None

    # Helper function to safely format scalar values that may be missing
    def fmt(value, precision=6):
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return str(value)

    # Helper function to safely format integer/ranking values that may be missing
    def fmt_int(value):
        if value is None:
            return "N/A"
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)

    # Helper function to compute 1-based descending rank within an array.
    # Higher scores get better/lower rank numbers.
    def descending_rank(values, index):
        if values is None or index is None:
            return None
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            return None
        index = int(index)
        if index < 0 or index >= len(values):
            return None
        return int(1 + np.sum(values > values[index]))

    # Helper function to find the nearest candidate index for a supplied point.
    def nearest_candidate_index(candidate_array, point):
        if candidate_array is None or point is None:
            return None
        candidate_array = np.asarray(candidate_array, dtype=float)
        point = np.asarray(point, dtype=float)
        if candidate_array.ndim != 2 or candidate_array.shape[0] == 0:
            return None
        return int(np.argmin(np.linalg.norm(candidate_array - point, axis=1)))

    # Helper function to fetch a per-candidate value either from an explicit scalar
    # or from a full vector and candidate index.
    def candidate_value(scalar_name, vector_name, candidate_index):
        if has(scalar_name):
            return get(scalar_name)
        if has(vector_name) and candidate_index is not None:
            values = np.asarray(get(vector_name), dtype=float)
            if 0 <= int(candidate_index) < len(values):
                return float(values[int(candidate_index)])
        return None

    print("==================================================")
    print("OPTIMISATION REPORT")
    print("==================================================")
    print("==================================================")
    print("1. DATASET SUMMARY")
    print("==================================================")
    print(f"Function / dataset: {load_dataset}")
    print(f"Week dataset: {week_dataset}")
    print(f"Samples used: {X.shape[0]}")
    print(f"Input dimensions: {X.shape[1]}")
    print(f"Samples per dimension: {X.shape[0] / X.shape[1]:.4f}")
    print(f"Observed y min: {np.min(y):.6f}")
    print(f"Observed y max: {np.max(y):.6f}")
    print(f"Observed y mean: {np.mean(y):.6f}")

    # Identify the best observed input and output from the dataset
    best_observed_idx = int(np.argmax(y))
    best_observed_input = X[best_observed_idx]
    best_observed_output = y[best_observed_idx]

    print(f"Best observed input: {arr(best_observed_input)}")
    print(f"Best observed output: {best_observed_output:.6f}")

    print("==================================================")
    print("\n2. SEARCH STRATEGY")
    print("==================================================")
    if progress is not None:
        print(f"Progress through budget: {get('progress', progress):.4f}")
    if has("exploration_weight"):
        print(f"Exploration weight / UCB: {get('exploration_weight'):.4f}")
    if has("exploitation_weight"):
        print(f"Exploitation weight / EI: {get('exploitation_weight'):.4f}")
    if has("xi"):
        print(f"EI / PI xi: {get('xi'):.6g}")
    if has("num_candidates_before_svm"):
        print(f"Generated candidates before filtering: {get('num_candidates_before_svm')}")
    elif has("num_candidates"):
        print(f"Generated candidates before filtering: {get('num_candidates')}")
    if has("candidates"):
        print(f"Candidates used after filtering: {len(get('candidates'))}")
    if has("svm_enabled"):
        print(f"SVM filtering enabled: {get('svm_enabled')}")

    print("==================================================")
    print("\n3. SVM CANDIDATE COMPARISON")
    print("==================================================")

    svm_enabled = bool(get("svm_enabled", False))
    candidates = get("candidates") if has("candidates") else None
    candidates_array = None if candidates is None else np.asarray(candidates, dtype=float)
    candidates_after_svm = len(candidates_array) if candidates_array is not None else None

    # Optional: if you pass the original candidate pool before SVM filtering,
    # the report can count it directly. Otherwise it uses num_candidates_before_svm.
    candidates_before_array = None
    if has("candidates_before_svm"):
        candidates_before_array = np.asarray(get("candidates_before_svm"), dtype=float)
    elif has("all_candidates_before_svm"):
        candidates_before_array = np.asarray(get("all_candidates_before_svm"), dtype=float)

    if candidates_before_array is not None:
        candidates_before_svm = len(candidates_before_array)
    else:
        candidates_before_svm = get("num_candidates_before_svm", get("num_candidates", None))

    # Optional: if you pass an sklearn SVM model, compute margins automatically
    # for the filtered candidates. This avoids needing to create svm_scores in
    # the notebook. If no model is supplied, the report still prints SVM counts.
    auto_svm_decision_values = None
    svm_model = get("svm_model", get("svm_classifier", None))
    if svm_model is not None and candidates_array is not None:
        try:
            if hasattr(svm_model, "decision_function"):
                auto_svm_decision_values = np.asarray(
                    svm_model.decision_function(candidates_array), dtype=float
                )
            elif hasattr(svm_model, "predict_proba"):
                proba = np.asarray(svm_model.predict_proba(candidates_array), dtype=float)
                auto_svm_decision_values = proba[:, -1]
        except Exception:
            auto_svm_decision_values = None

    if has("svm_enabled"):
        print(f"SVM filtering enabled: {svm_enabled}")
    else:
        print("SVM filtering enabled: not supplied")

    if candidates_before_svm is not None:
        print(f"Candidates before SVM filtering: {int(candidates_before_svm)}")
    if candidates_after_svm is not None:
        print(f"Candidates after SVM filtering: {int(candidates_after_svm)}")
    if candidates_before_svm is not None and candidates_after_svm is not None:
        rejected = int(candidates_before_svm) - int(candidates_after_svm)
        retention = 100.0 * candidates_after_svm / max(int(candidates_before_svm), 1)
        print(f"Candidates rejected by SVM: {rejected}")
        print(f"SVM retention rate: {retention:.2f}%")

    # Candidate indices in the filtered candidate set. These can be supplied explicitly,
    # or inferred by nearest-neighbour lookup if candidates are available.
    gp_candidate_idx = int(get("best_idx" if has("best_idx") else "gp_candidate_idx", best_idx))
    thompson_candidate_idx = get("thompson_best_idx", None)
    nn_candidate_idx = get("nn_best_idx", None)

    if thompson_candidate_idx is None and has("best_input_thompson"):
        thompson_candidate_idx = nearest_candidate_index(candidates_array, get("best_input_thompson"))
    if nn_candidate_idx is None and has("nn_candidate"):
        nn_candidate_idx = nearest_candidate_index(candidates_array, get("nn_candidate"))

    # SVM scores can be provided as explicit scalars or as arrays over candidates.
    # Supported scalar names: gp_svm_score, thompson_svm_score, nn_svm_score.
    # Supported vector names: svm_scores and svm_decision_values.
    # If svm_model is supplied, svm_decision_values are computed automatically.
    svm_scores = np.asarray(get("svm_scores"), dtype=float) if has("svm_scores") else None
    svm_decision_values = (
        np.asarray(get("svm_decision_values"), dtype=float)
        if has("svm_decision_values")
        else auto_svm_decision_values
    )

    # If only decision values are available, use them as the SVM score too.
    if svm_scores is None and svm_decision_values is not None:
        svm_scores = svm_decision_values

    gp_svm_score = candidate_value("gp_svm_score", "svm_scores", gp_candidate_idx)
    thompson_svm_score = candidate_value("thompson_svm_score", "svm_scores", thompson_candidate_idx)
    nn_svm_score = candidate_value("nn_svm_score", "svm_scores", nn_candidate_idx)

    # candidate_value reads from kwargs, so handle auto-computed margins here.
    gp_svm_margin = candidate_value("gp_svm_decision_value", "svm_decision_values", gp_candidate_idx)
    thompson_svm_margin = candidate_value("thompson_svm_decision_value", "svm_decision_values", thompson_candidate_idx)
    nn_svm_margin = candidate_value("nn_svm_decision_value", "svm_decision_values", nn_candidate_idx)

    if gp_svm_margin is None and svm_decision_values is not None and gp_candidate_idx is not None:
        gp_svm_margin = float(svm_decision_values[int(gp_candidate_idx)])
    if thompson_svm_margin is None and svm_decision_values is not None and thompson_candidate_idx is not None:
        thompson_svm_margin = float(svm_decision_values[int(thompson_candidate_idx)])
    if nn_svm_margin is None and svm_decision_values is not None and nn_candidate_idx is not None:
        nn_svm_margin = float(svm_decision_values[int(nn_candidate_idx)])

    if gp_svm_score is None and svm_scores is not None and gp_candidate_idx is not None:
        gp_svm_score = float(svm_scores[int(gp_candidate_idx)])
    if thompson_svm_score is None and svm_scores is not None and thompson_candidate_idx is not None:
        thompson_svm_score = float(svm_scores[int(thompson_candidate_idx)])
    if nn_svm_score is None and svm_scores is not None and nn_candidate_idx is not None:
        nn_svm_score = float(svm_scores[int(nn_candidate_idx)])

    svm_score_count = len(svm_scores) if svm_scores is not None else candidates_after_svm

    gp_svm_rank = get("gp_svm_rank", descending_rank(svm_scores, gp_candidate_idx))
    thompson_svm_rank = get("thompson_svm_rank", descending_rank(svm_scores, thompson_candidate_idx))
    nn_svm_rank = get("nn_svm_rank", descending_rank(svm_scores, nn_candidate_idx))

    gp_mean_rank = get("gp_mean_rank", descending_rank(ensemble_mean, gp_candidate_idx))
    gp_std_rank = get("gp_uncertainty_rank", descending_rank(ensemble_std, gp_candidate_idx))
    gp_ei_rank = get("gp_ei_rank", descending_rank(ei_final, gp_candidate_idx))
    gp_ucb_rank = get("gp_ucb_rank", descending_rank(ucb_final, gp_candidate_idx))
    gp_pi_rank = get("gp_pi_rank", descending_rank(pi_final, gp_candidate_idx))

    thompson_mean_rank = get("thompson_mean_rank", descending_rank(ensemble_mean, thompson_candidate_idx))
    thompson_std_rank = get("thompson_uncertainty_rank", descending_rank(ensemble_std, thompson_candidate_idx))
    thompson_ei_rank = get("thompson_ei_rank", descending_rank(ei_final, thompson_candidate_idx))
    thompson_ucb_rank = get("thompson_ucb_rank", descending_rank(ucb_final, thompson_candidate_idx))
    thompson_pi_rank = get("thompson_pi_rank", descending_rank(pi_final, thompson_candidate_idx))

    nn_mean_rank = get("nn_mean_rank", descending_rank(ensemble_mean, nn_candidate_idx))
    nn_std_rank = get("nn_uncertainty_rank", descending_rank(ensemble_std, nn_candidate_idx))
    nn_ei_rank = get("nn_ei_rank", descending_rank(ei_final, nn_candidate_idx))
    nn_ucb_rank = get("nn_ucb_rank", descending_rank(ucb_final, nn_candidate_idx))
    nn_pi_rank = get("nn_pi_rank", descending_rank(pi_final, nn_candidate_idx))

    print("\nCandidate SVM support")
    print("Candidate         Accepted  SVM Score   SVM Margin   SVM Rank")
    print("--------------------------------------------------------------")

    def accepted_label(score, margin):
        if margin is not None:
            return "Yes" if float(margin) >= 0 else "No"
        if score is not None:
            return "Yes" if float(score) >= 0 else "No"
        return "N/A"

    print(
        f"{'GP Hybrid':<16} "
        f"{accepted_label(gp_svm_score, gp_svm_margin):<8} "
        f"{fmt(gp_svm_score):>9} "
        f"{fmt(gp_svm_margin):>12} "
        f"{fmt_int(gp_svm_rank):>8}"
    )
    if has("best_input_thompson") or thompson_candidate_idx is not None:
        print(
            f"{'Thompson':<16} "
            f"{accepted_label(thompson_svm_score, thompson_svm_margin):<8} "
            f"{fmt(thompson_svm_score):>9} "
            f"{fmt(thompson_svm_margin):>12} "
            f"{fmt_int(thompson_svm_rank):>8}"
        )
    if has("nn_candidate") or nn_candidate_idx is not None:
        print(
            f"{'Neural Network':<16} "
            f"{accepted_label(nn_svm_score, nn_svm_margin):<8} "
            f"{fmt(nn_svm_score):>9} "
            f"{fmt(nn_svm_margin):>12} "
            f"{fmt_int(nn_svm_rank):>8}"
        )

    print("\nCandidate acquisition comparison")
    print("Candidate         GP Mean    GP Std        EI       UCB        PI  Final Score")
    print("----------------------------------------------------------------------------")

    def print_candidate_row(label, idx, mean_value=None, std_value=None):
        if idx is not None and 0 <= int(idx) < len(ensemble_mean):
            idx = int(idx)
            mean_value = ensemble_mean[idx] if mean_value is None else mean_value
            std_value = ensemble_std[idx] if std_value is None else std_value
            ei_value = ei_final[idx]
            ucb_value = ucb_final[idx]
            pi_value = pi_final[idx]
            score_value = final_score[idx]
        else:
            ei_value = ucb_value = pi_value = score_value = None
        print(
            f"{label:<16} "
            f"{fmt(mean_value):>8} "
            f"{fmt(std_value):>9} "
            f"{fmt(ei_value):>9} "
            f"{fmt(ucb_value):>9} "
            f"{fmt(pi_value):>9} "
            f"{fmt(score_value):>12}"
        )

    print_candidate_row("GP Hybrid", gp_candidate_idx)
    if has("best_input_thompson") or thompson_candidate_idx is not None:
        print_candidate_row("Thompson", thompson_candidate_idx)
    if has("nn_candidate") or nn_candidate_idx is not None:
        print_candidate_row(
            "Neural Network",
            nn_candidate_idx,
            get("nn_gp_ensemble_mean", None),
            get("nn_gp_ensemble_std", None),
        )

    print("\nCandidate ranking within filtered candidate set")
    total_rank_count = svm_score_count if svm_score_count is not None else len(ensemble_mean)
    print(f"Ranking denominator: {total_rank_count}")
    print("Candidate         Mean Rank  Std Rank   EI Rank  UCB Rank   PI Rank  SVM Rank")
    print("---------------------------------------------------------------------------")
    print(
        f"{'GP Hybrid':<16} "
        f"{fmt_int(gp_mean_rank):>9} "
        f"{fmt_int(gp_std_rank):>9} "
        f"{fmt_int(gp_ei_rank):>9} "
        f"{fmt_int(gp_ucb_rank):>9} "
        f"{fmt_int(gp_pi_rank):>9} "
        f"{fmt_int(gp_svm_rank):>9}"
    )
    if has("best_input_thompson") or thompson_candidate_idx is not None:
        print(
            f"{'Thompson':<16} "
            f"{fmt_int(thompson_mean_rank):>9} "
            f"{fmt_int(thompson_std_rank):>9} "
            f"{fmt_int(thompson_ei_rank):>9} "
            f"{fmt_int(thompson_ucb_rank):>9} "
            f"{fmt_int(thompson_pi_rank):>9} "
            f"{fmt_int(thompson_svm_rank):>9}"
        )
    if has("nn_candidate") or nn_candidate_idx is not None:
        print(
            f"{'Neural Network':<16} "
            f"{fmt_int(nn_mean_rank):>9} "
            f"{fmt_int(nn_std_rank):>9} "
            f"{fmt_int(nn_ei_rank):>9} "
            f"{fmt_int(nn_ucb_rank):>9} "
            f"{fmt_int(nn_pi_rank):>9} "
            f"{fmt_int(nn_svm_rank):>9}"
        )

    if gp_svm_score is not None or gp_svm_margin is not None:
        print("\nSVM interpretation")
        if gp_svm_margin is not None and float(gp_svm_margin) >= 0:
            print("The selected GP candidate lies inside the SVM-accepted region.")
        elif gp_svm_margin is not None:
            print("The selected GP candidate has a negative SVM margin, so check whether it bypassed the SVM filter.")
        elif gp_svm_score is not None:
            print("The selected GP candidate has an available SVM score, so it can be compared against alternative candidates.")

        if thompson_svm_score is not None:
            diff = float(gp_svm_score) - float(thompson_svm_score) if gp_svm_score is not None else None
            print(f"Difference in SVM score, GP minus Thompson: {fmt(diff)}")
        if nn_svm_score is not None:
            diff = float(gp_svm_score) - float(nn_svm_score) if gp_svm_score is not None else None
            print(f"Difference in SVM score, GP minus NN: {fmt(diff)}")

        if svm_enabled and candidates_before_svm is not None and candidates_after_svm is not None:
            print(
                "The SVM filter acts as a feasibility or quality gate before the "
                "GP/EI/UCB/PI scoring chooses among the surviving candidates."
            )
    else:
        print("\nSVM interpretation")
        print(
            "No per-candidate SVM scores were supplied. Pass svm_scores, "
            "svm_decision_values, or explicit gp/thompson/nn SVM scores to fill this section."
        )

    print("==================================================")
    print("\n4. GP TRAINING QUALITY")
    print("==================================================")
    training_errors = []

    y_scale = float(np.max(y) - np.min(y))
    if y_scale <= 0:
        y_scale = 1.0

    for kernel_name, gp in trained_models.items():
        train_mean, train_std = gp.predict(X, return_std=True)
        mae = np.mean(np.abs(train_mean - y))
        rmse = np.sqrt(np.mean((train_mean - y) ** 2))
        relative_rmse = rmse / y_scale
        training_errors.append(rmse)

        print(f"\n{kernel_name}")
        print(f"  Training MAE: {mae:.6f}")
        print(f"  Training RMSE: {rmse:.6f}")
        print(f"  Relative RMSE: {relative_rmse:.6f}")
        print(f"  Max training std: {np.max(train_std):.6f}")
        print(f"  Learned kernel: {gp.kernel_}")

    median_training_rmse = float(np.median(training_errors)) if training_errors else None

    # Use the strongest available anisotropic kernel geometry for
    # high-dimensional distance diagnostics.
    ard_kernel_name, ard_length_scales = (select_reference_ard_length_scales(trained_models, X.shape[1]))

    print("==================================================")
    print("\n5. INDIVIDUAL GP ACQUISITION SUGGESTIONS")
    print("==================================================")
    for method, result in results.items():
        print(f"\n{method}")
        print(f"  Input: {arr(result['input'])}")
        print(f"  Predicted mean: {result['mean']:.6f}")
        print(f"  Predicted std: {result['std']:.6f}")
        print(f"  Acquisition score: {result['score']:.6g}")

    acquisition_points = extract_top_candidate_inputs(results)
    acquisition_spread = mean_pairwise_distance(acquisition_points)
    # A second consensus measure focused on suggestions from kernels
    # with anisotropic relevance information. This usually gives a
    # more meaningful summary in high-dimensional functions.
    core_gp_points = []

    for method, result in results.items():
        method_lower = method.lower()

        if (
            "matern" in method_lower
            or "matérn" in method_lower
            or "rbf" in method_lower
        ):
            if isinstance(result, dict) and "input" in result:
                core_gp_points.append(
                    np.asarray(
                        result["input"],
                        dtype=float
                    )
                )

    if len(core_gp_points) >= 2:
        core_gp_points = np.vstack(core_gp_points)

        if ard_length_scales is not None:
            core_gp_pairwise_distances = []

            for i in range(len(core_gp_points)):
                for j in range(i + 1, len(core_gp_points)):
                    core_gp_pairwise_distances.append(
                        scaled_distance(
                            core_gp_points[i],
                            core_gp_points[j],
                            ard_length_scales
                        )
                    )

            core_gp_spread = float(
                np.mean(core_gp_pairwise_distances)
            )
        else:
            core_gp_spread = mean_pairwise_distance(
                core_gp_points
            )
    else:
        core_gp_spread = None

    print("==================================================")
    print("\n6. FINAL HYBRID DECISION")
    print("==================================================")
    print(f"Best next input: {arr(best_input)}")
    print(f"Ensemble predicted mean: {ensemble_mean[best_idx]:.6f}")
    print(f"Ensemble predicted std: {ensemble_std[best_idx]:.6f}")
    print(f"Estimated EI: {ei_final[best_idx]:.6g}")
    print(f"Estimated UCB: {ucb_final[best_idx]:.6f}")
    print(f"Estimated PI: {pi_final[best_idx]:.6g}")
    print(f"Final acquisition score: {final_score[best_idx]:.6f}")
    print(
        "Note: The final acquisition score is a normalized relative score "
        "within the candidate set, not an absolute probability of success."
    )

    print("==================================================")
    print("\n7. CROSS-MODEL AGREEMENT")
    print("==================================================")
    print(f"GP / hybrid candidate: {arr(best_input)}")

    #distance_to_best_observed = np.linalg.norm(best_input - best_observed_input)
    #nearest_sample_distance = float(np.min(np.linalg.norm(X - best_input, axis=1)))

    #print(f"Distance to best observed input: {distance_to_best_observed:.6f}")
    #print(f"Distance to nearest observed sample: {nearest_sample_distance:.6f}")
    #print(f"Search behaviour: {classify_search_behaviour(distance_to_best_observed)}")

    distance_to_best_observed = float(
        np.linalg.norm(
            best_input - best_observed_input
        )
    )

    nearest_sample_distance = float(
        np.min(
            np.linalg.norm(
                X - best_input,
                axis=1
            )
        )
    )

    ard_distance_to_best_observed = scaled_distance(
        best_input,
        best_observed_input,
        ard_length_scales
    )

    ard_nearest_sample_distance = (
        scaled_nearest_sample_distance(
            X,
            best_input,
            ard_length_scales
        )
    )

    print(
        f"Raw distance to best observed input: "
        f"{distance_to_best_observed:.6f}"
    )

    print(
        f"Raw distance to nearest observed sample: "
        f"{nearest_sample_distance:.6f}"
    )

    if ard_length_scales is not None:
        print(
            f"ARD reference kernel: "
            f"{ard_kernel_name}"
        )

        print(
            f"ARD-scaled distance to best observed input: "
            f"{ard_distance_to_best_observed:.6f}"
        )

        print(
            f"ARD-scaled distance to nearest observed sample: "
            f"{ard_nearest_sample_distance:.6f}"
        )

        search_distance = ard_distance_to_best_observed

        print(
            "Search behaviour uses ARD-scaled distance because "
            "the trained GP has dimension-specific length scales."
        )
    else:
        search_distance = distance_to_best_observed

    print(
        f"Search behaviour: "
        f"{classify_search_behaviour(search_distance)}"
    )

    if acquisition_spread is not None:
        print(f"Mean pairwise distance between acquisition suggestions: {acquisition_spread:.6f}")
        print(f"Acquisition-function agreement: {agreement_label(acquisition_spread)}")
        if core_gp_spread is not None:
            print(
                f"Core Matérn/RBF acquisition spread: "
                f"{core_gp_spread:.6f}"
            )

            print(
                f"Core GP acquisition agreement: "
                f"{agreement_label(core_gp_spread)}"
            )

    d_thompson = None
    d_nn = None
    d_svm = None
    best_input_thompson = None
    nn_candidate = None
    svm_candidate = None

    if has("best_input_thompson"):
        best_input_thompson = np.asarray(get("best_input_thompson"), dtype=float)
        #d_thompson = np.linalg.norm(best_input - best_input_thompson)
        d_thompson_raw = float(np.linalg.norm(best_input - best_input_thompson))

        d_thompson = scaled_distance(
            best_input,
            best_input_thompson,
            ard_length_scales
        )

        print(f"Raw distance GP to Thompson: " f"{d_thompson_raw:.6f}" )

        if ard_length_scales is not None:
            print(f"ARD-scaled distance GP to Thompson: " f"{d_thompson:.6f}")
            
        print(f"Thompson candidate: {arr(best_input_thompson)}")
        print(f"Distance GP to Thompson: {d_thompson:.6f}")
        print(f"GP / Thompson agreement: {agreement_label(d_thompson)}")

    if has("nn_candidate"):
        nn_candidate = np.asarray(get("nn_candidate"), dtype=float)
        #d_nn = np.linalg.norm(best_input - nn_candidate)
        #print(f"NN candidate: {arr(nn_candidate)}")
        #print(f"Distance GP to NN: {d_nn:.6f}")
        #print(f"GP / NN agreement: {agreement_label(d_nn)}")
        d_nn_raw = float(
            np.linalg.norm(
                best_input - nn_candidate
            )
        )

        d_nn = scaled_distance(
            best_input,
            nn_candidate,
            ard_length_scales
        )

        print(f"NN candidate: {arr(nn_candidate)}")
        print(
            f"Raw distance GP to NN: "
            f"{d_nn_raw:.6f}"
        )

        if ard_length_scales is not None:
            print(
                f"ARD-scaled distance GP to NN: "
                f"{d_nn:.6f}"
            )

        print(
            f"GP / NN agreement: "
            f"{agreement_label(d_nn)}"
        )

        if has("nn_gp_ensemble_mean"):
            print(f"GP mean at NN candidate: {get('nn_gp_ensemble_mean'):.6f}")
        if has("nn_gp_ensemble_std"):
            print(f"GP std at NN candidate: {get('nn_gp_ensemble_std'):.6f}")

    if has("svm_candidate"):
        svm_candidate = np.asarray(get("svm_candidate"), dtype=float)
        #d_svm = np.linalg.norm(best_input - svm_candidate)
        #print(f"SVM candidate: {arr(svm_candidate)}")
        #print(f"Distance GP to SVM: {d_svm:.6f}")
        #print(f"GP / SVM agreement: {agreement_label(d_svm)}")
        d_svm_raw = float(
            np.linalg.norm(
                best_input - svm_candidate
            )
        )

        d_svm = scaled_distance(
            best_input,
            svm_candidate,
            ard_length_scales
        )

        print(f"SVM candidate: {arr(svm_candidate)}")
        print(
            f"Raw distance GP to SVM: "
            f"{d_svm_raw:.6f}"
        )

        if ard_length_scales is not None:
            print(
                f"ARD-scaled distance GP to SVM: "
                f"{d_svm:.6f}"
            )

        print(
            f"GP / SVM agreement: "
            f"{agreement_label(d_svm)}"
        )

        if has("svm_gp_mean"):
            print(f"GP mean at SVM candidate: {get('svm_gp_mean'):.6f}")
        if has("svm_gp_std"):
            print(f"GP std at SVM candidate: {get('svm_gp_std'):.6f}")
        if has("svm_gp_ei"):
            print(f"EI at SVM candidate: {get('svm_gp_ei'):.6g}")

    # Candidate agreement matrix across all available candidate generators.
    agreement_candidates = [("GP", np.asarray(best_input, dtype=float))]
    if best_input_thompson is not None:
        agreement_candidates.append(("Thompson", best_input_thompson))
    if nn_candidate is not None:
        agreement_candidates.append(("NN", nn_candidate))
    if svm_candidate is not None:
        agreement_candidates.append(("SVM", svm_candidate))

    if len(agreement_candidates) >= 3:
        names = [name for name, _ in agreement_candidates]
        points = [point for _, point in agreement_candidates]
        width = max(10, max(len(name) for name in names) + 2)

        #print("\nCandidate agreement distance matrix")
        if ard_length_scales is not None:
            print(
                "\nCandidate agreement distance matrix "
                "(ARD-scaled)"
            )
        else:
            print(
                "\nCandidate agreement distance matrix"
            )

        print("".ljust(width) + "".join(name.rjust(12) for name in names))
        pairwise_distances = []
        for i, name_i in enumerate(names):
            row = name_i.ljust(width)
            for j in range(len(names)):
                #distance = float(np.linalg.norm(points[i] - points[j]))
                distance = scaled_distance(points[i], points[j], ard_length_scales)
                row += f"{distance:12.6f}"
                if i < j:
                    pairwise_distances.append(distance)
            print(row)

        if pairwise_distances:
            mean_consensus_distance = float(np.mean(pairwise_distances))
            max_consensus_distance = float(np.max(pairwise_distances))
            print(f"Mean cross-model distance: {mean_consensus_distance:.6f}")
            print(f"Max cross-model distance: {max_consensus_distance:.6f}")
            print(f"Overall candidate consensus: {agreement_label(mean_consensus_distance)}")

            if mean_consensus_distance < 0.05:
                print("Consensus summary: all available candidate generators are concentrated in almost the same region.")
            elif mean_consensus_distance < 0.15:
                print("Consensus summary: the available candidate generators broadly support the same region.")
            elif mean_consensus_distance < 0.30:
                print("Consensus summary: the available candidate generators show partial agreement but with visible spread.")
            else:
                print("Consensus summary: the available candidate generators disagree, so treat the final query as less strongly supported.")

    print("==================================================")
    print("\n8. CONVERGENCE DIAGNOSTICS")
    print("==================================================")

    selected_std = ensemble_std[best_idx]
    selected_ei = ei_final[best_idx]
    selected_pi = pi_final[best_idx]

    print(f"Posterior uncertainty at selected point: {selected_std:.6f}")
    print(f"Uncertainty level: {uncertainty_label(selected_std)}")

    if selected_ei < 1e-12:
        print(
            "Expected Improvement has effectively converged. "
            "The surrogate predicts that further improvement over the current best "
            "observation is unlikely."
        )
    elif selected_ei < 1e-6:
        print(
            "Expected Improvement is very small, suggesting limited predicted gain "
            "over the current best observation."
        )
    else:
        print(
            "Expected Improvement is still active, suggesting the model sees some "
            "possibility of improvement."
        )

    if selected_pi < 1e-12:
        print("Probability of Improvement has effectively converged and is close to zero.")
    elif selected_pi < 1e-6:
        print("Probability of Improvement is very small.")
    else:
        print("Probability of Improvement is still non-negligible.")

    #if distance_to_best_observed < 0.02:
    if search_distance < 0.02:
        print(
            "The suggested query lies extremely close to the current best observation, "
            "indicating strong local refinement."
        )

    nearest_distance_for_interpretation = (
        ard_nearest_sample_distance
        if ard_length_scales is not None
        else nearest_sample_distance
        )

    print("==================================================")
    print("\n9. DECISION TYPE")
    print("==================================================")

    #decision_type = classify_search_behaviour(distance_to_best_observed)
    decision_type = classify_search_behaviour(search_distance)
    print(f"Strategy: {decision_type}")

    #if decision_type == "Strong local refinement":
    #    print(
    #        "Reason: The selected point is almost coincident with the current best "
    #        "observation, so this recommendation is a fine local refinement step."
    #    )
    #elif decision_type == "Local exploitation":
    #    print(
    #        "Reason: The selected point remains close to the best observation, so the "
    #        "optimiser is exploiting the current promising basin."
    #    )
    #elif decision_type == "Regional exploration / exploitation":
    #    print(
    #        "Reason: The selected point is in the same broad region as the current best "
    #        "sample but far enough away to test the local shape of the response surface."
    #    )
    #else:
    #    print(
    #        "Reason: The selected point is far from the current best observation. "
    #        "If the GP and acquisition functions agree, this should be interpreted as "
    #        "a deliberate exploration of a second promising basin rather than a failure."
    #    )

    if decision_type == "Strong local refinement":
        if ard_length_scales is not None:
            print(
                "Reason: Although some raw coordinates may differ, the "
                "candidate is almost coincident with the current best "
                "observation in the GP's learned ARD geometry. This is "
                "a fine model-relative local refinement."
            )
        else:
            print(
                "Reason: The selected point is almost coincident with "
                "the current best observation, so this recommendation "
                "is a fine local refinement step."
            )

    elif decision_type == "Local exploitation":
        if ard_length_scales is not None:
            print(
                "Reason: The candidate remains close to the best "
                "observation after accounting for dimension-specific "
                "GP length scales, so the optimiser is exploiting the "
                "current promising basin."
            )
        else:
            print(
                "Reason: The selected point remains close to the best "
                "observation, so the optimiser is exploiting the "
                "current promising basin."
            )

    elif decision_type == "Regional exploration / exploitation":
        print(
            "Reason: The candidate remains in the same broad GP-supported "
            "region but is displaced enough to test the response surface "
            "and add useful information."
        )

    else:
        print(
            "Reason: The selected point is far from the current best "
            "observation even after accounting for the GP's learned "
            "length scales. This is genuine global exploration."
        )


    if nearest_distance_for_interpretation  < 0.02:
        print(
            "Nearest-sample note: the candidate is also very close to an existing sample, "
            "so this is a dense local refinement."
        )
    elif nearest_distance_for_interpretation  < 0.10:
        print(
            "Nearest-sample note: the candidate is near previously sampled data, "
            "so it is refining a known region rather than jumping into completely new space."
        )
    else:
        print(
            "Nearest-sample note: the candidate is not close to existing samples, "
            "so the query will add information in a relatively unexplored part of the domain."
        )

    print("==================================================")
    print("\n10. INTERPRETATION")
    print("==================================================")

    if selected_std < 0.5:
        print("The selected candidate has low-to-moderate uncertainty.")
    else:
        print("The selected candidate has high uncertainty and should be treated as exploratory.")

    if d_thompson is not None:
        if d_thompson < 0.15:
            print("The hybrid GP and Thompson candidates broadly agree.")
        elif d_thompson < 0.30:
            print("The Thompson candidate gives partial support, but is displaced from the final GP candidate.")
        else:
            print("The hybrid GP and Thompson candidates differ noticeably.")

    #if d_nn is not None:
    #    if d_nn < 0.15:
    #        print("The GP ensemble and NN surrogate broadly agree on the same promising region.")
    #    elif d_nn < 0.30:
    #        print("The NN surrogate gives partial support, but is displaced from the final GP candidate.")
    #    else:
    #        print("The GP ensemble and NN surrogate disagree, so the neural surrogate is not supporting this query.")
    if d_nn is not None:
        nn_gp_mean = get(
            "nn_gp_ensemble_mean",
            None
        )

        nn_gp_ei = None

        if nn_candidate_idx is not None:
            nn_gp_ei = float(
                ei_final[int(nn_candidate_idx)]
            )

        selected_gp_mean = float(
            ensemble_mean[best_idx]
        )

        if (
            nn_gp_mean is not None
            and float(nn_gp_mean) < 0.95 * selected_gp_mean
        ):
            print(
                "The NN proposes a point in a broadly related GP-scaled "
                "region, but the GP assigns it a materially lower predicted "
                "mean than the selected candidate."
            )

            if nn_gp_ei is not None:
                print(
                    f"GP EI at NN candidate: "
                    f"{nn_gp_ei:.6g}"
                )

            print(
                "The NN disagreement therefore lowers confidence slightly "
                "but should not override the GP/Thompson recommendation."
            )

        elif d_nn < 0.15:
            print(
                "The GP ensemble and NN surrogate broadly agree on "
                "the same promising region."
            )

        elif d_nn < 0.30:
            print(
                "The NN surrogate gives partial support, but is "
                "displaced from the final GP candidate."
            )

        else:
            print(
                "The GP ensemble and NN surrogate disagree, so the "
                "neural surrogate is not directly supporting this query."
            )

    #if d_svm is not None:
    #    if d_svm < 0.15:
    #        print("The SVM candidate supports the same promising region as the GP hybrid candidate.")
    #    elif d_svm < 0.30:
    #        print("The SVM candidate gives partial support, but is displaced from the final GP candidate.")
    #    else:
    #        print("The SVM candidate differs noticeably from the GP hybrid candidate, so the SVM is not giving strong spatial support.")

    if d_svm is not None:
        if gp_svm_margin is not None and float(gp_svm_margin) >= 0:
            print(
                "The selected GP candidate passes the SVM quality "
                "gate. Spatial disagreement with the standalone SVM "
                "candidate does not invalidate it because the SVM is "
                "used primarily as a filter rather than as the final "
                "regression surrogate."
            )

            if d_svm < 0.15:
                print(
                    "The standalone SVM candidate also lies in the "
                    "same GP-scaled region."
                )
            elif d_svm < 0.30:
                print(
                    "The standalone SVM optimum is somewhat displaced, "
                    "but still provides partial regional support."
                )
            else:
                print(
                    "The standalone SVM optimum is spatially different, "
                    "but the GP candidate remains within the accepted "
                    "SVM region."
                )
        else:
            print(
                "The selected candidate does not have clear positive "
                "SVM support, so inspect the filtering stage before "
                "submitting it."
            )

    print("==================================================")
    print("\n11. OPTIMISER CONFIDENCE")
    print("==================================================")

    confidence_score = 0.0
    confidence_reasons = []

    if median_training_rmse is not None:
        relative_rmse = median_training_rmse / y_scale
        if relative_rmse < 0.05:
            confidence_score += 1.0
            confidence_reasons.append("GP training error is small relative to the observed output range")
        elif relative_rmse < 0.15:
            confidence_score += 0.5
            confidence_reasons.append("GP training error is moderate relative to the observed output range")

    if selected_std < 0.5:
        confidence_score += 1.0
        confidence_reasons.append(f"posterior uncertainty is {uncertainty_label(selected_std).lower()} at the selected point")

    #if acquisition_spread is not None:
    #    if acquisition_spread < 0.15:
    #        confidence_score += 1.0
    #        confidence_reasons.append("GP acquisition functions broadly agree")
    #    elif acquisition_spread < 0.30:
    #        confidence_score += 0.5
    #        confidence_reasons.append("GP acquisition functions show moderate agreement")

    agreement_spread_for_confidence = (
    core_gp_spread
    if core_gp_spread is not None
    else acquisition_spread
    )

    if agreement_spread_for_confidence is not None:
        if agreement_spread_for_confidence < 0.15:
            confidence_score += 1.0
            confidence_reasons.append(
                "core GP acquisition functions broadly agree"
            )
        elif agreement_spread_for_confidence < 0.30:
            confidence_score += 0.5
            confidence_reasons.append(
                "core GP acquisition functions show moderate agreement"
            )

        if d_thompson is not None:
            if d_thompson < 0.15:
                confidence_score += 1.0
                confidence_reasons.append("Thompson sampling supports the GP candidate")
            elif d_thompson < 0.30:
                confidence_score += 0.5
                confidence_reasons.append("Thompson sampling gives partial support")

    #if d_nn is not None:
    #    if d_nn < 0.15:
    #        confidence_score += 1.0
    #        confidence_reasons.append("neural surrogate supports the GP candidate")
    #    elif d_nn < 0.30:
    #        confidence_score += 0.5
    #        confidence_reasons.append("neural surrogate gives partial support")

    if d_svm is not None:
        if d_svm < 0.15:
            confidence_score += 1.0
            confidence_reasons.append("SVM candidate supports the GP candidate")
        elif d_svm < 0.30:
            confidence_score += 0.5
            confidence_reasons.append("SVM candidate gives partial support")

    print(f"Overall optimiser confidence: {confidence_label(confidence_score)}")
    print(f"Confidence score: {confidence_score:.1f} / 6.0")
    print("Reasons:")

    if confidence_reasons:
        for reason in confidence_reasons:
            print(f"  - {reason}")
    else:
        print("  - no strong agreement signals were detected")

    print("==================================================")
    print("\n12. FINAL RECOMMENDATION")
    print("==================================================")
    print(f"Submit next query: {arr(best_input)}")
    print("\n===============================================================\n")
