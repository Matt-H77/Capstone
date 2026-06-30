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
    def has(name):
        return name in globals()

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
        if score >= 4.0:
            return "VERY HIGH"
        elif score >= 3.0:
            return "HIGH"
        elif score >= 2.0:
            return "MODERATE"
        elif score >= 1.0:
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

    print("==================================================")
    print("OPTIMISATION REPORT")
    print("==================================================")
    print("1. DATASET SUMMARY")
    print(f"Function / dataset: {load_dataset if has('load_dataset') else 'unknown'}")
    print(f"Week dataset: {week_dataset if has('week_dataset') else 'unknown'}")
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

    print("\n2. SEARCH STRATEGY")
    if has("progress"):
        print(f"Progress through budget: {progress:.4f}")
    if has("exploration_weight"):
        print(f"Exploration weight / UCB: {exploration_weight:.4f}")
    if has("exploitation_weight"):
        print(f"Exploitation weight / EI: {exploitation_weight:.4f}")
    if has("xi"):
        print(f"EI / PI xi: {xi:.6g}")
    if has("num_candidates"):
        print(f"Generated candidates before filtering: {num_candidates}")
    if has("candidates"):
        print(f"Candidates used after filtering: {len(candidates)}")
    if has("svm_enabled"):
        print(f"SVM filtering enabled: {svm_enabled}")

    print("\n3. GP TRAINING QUALITY")
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

    print("\n4. INDIVIDUAL GP ACQUISITION SUGGESTIONS")
    for method, result in results.items():
        print(f"\n{method}")
        print(f"  Input: {arr(result['input'])}")
        print(f"  Predicted mean: {result['mean']:.6f}")
        print(f"  Predicted std: {result['std']:.6f}")
        print(f"  Acquisition score: {result['score']:.6g}")

    acquisition_points = extract_top_candidate_inputs(results)
    acquisition_spread = mean_pairwise_distance(acquisition_points)

    print("\n5. FINAL HYBRID DECISION")
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

    print("\n6. CROSS-MODEL AGREEMENT")
    print(f"GP / hybrid candidate: {arr(best_input)}")

    distance_to_best_observed = np.linalg.norm(best_input - best_observed_input)
    nearest_sample_distance = float(np.min(np.linalg.norm(X - best_input, axis=1)))

    print(f"Distance to best observed input: {distance_to_best_observed:.6f}")
    print(f"Distance to nearest observed sample: {nearest_sample_distance:.6f}")
    print(f"Search behaviour: {classify_search_behaviour(distance_to_best_observed)}")

    if acquisition_spread is not None:
        print(f"Mean pairwise distance between acquisition suggestions: {acquisition_spread:.6f}")
        print(f"Acquisition-function agreement: {agreement_label(acquisition_spread)}")

    d_thompson = None
    d_nn = None

    if has("best_input_thompson"):
        d_thompson = np.linalg.norm(best_input - best_input_thompson)
        print(f"Thompson candidate: {arr(best_input_thompson)}")
        print(f"Distance GP to Thompson: {d_thompson:.6f}")
        print(f"GP / Thompson agreement: {agreement_label(d_thompson)}")

    if has("nn_candidate") and nn_candidate is not None:
        d_nn = np.linalg.norm(best_input - nn_candidate)
        print(f"NN candidate: {arr(nn_candidate)}")
        print(f"Distance GP to NN: {d_nn:.6f}")
        print(f"GP / NN agreement: {agreement_label(d_nn)}")

        if has("nn_gp_ensemble_mean"):
            print(f"GP mean at NN candidate: {nn_gp_ensemble_mean:.6f}")
        if has("nn_gp_ensemble_std"):
            print(f"GP std at NN candidate: {nn_gp_ensemble_std:.6f}")

    print("\n7. CONVERGENCE DIAGNOSTICS")

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

    if distance_to_best_observed < 0.02:
        print(
            "The suggested query lies extremely close to the current best observation, "
            "indicating strong local refinement."
        )

    print("\n8. DECISION TYPE")

    decision_type = classify_search_behaviour(distance_to_best_observed)
    print(f"Strategy: {decision_type}")

    if decision_type == "Strong local refinement":
        print(
            "Reason: The selected point is almost coincident with the current best "
            "observation, so this recommendation is a fine local refinement step."
        )
    elif decision_type == "Local exploitation":
        print(
            "Reason: The selected point remains close to the best observation, so the "
            "optimiser is exploiting the current promising basin."
        )
    elif decision_type == "Regional exploration / exploitation":
        print(
            "Reason: The selected point is in the same broad region as the current best "
            "sample but far enough away to test the local shape of the response surface."
        )
    else:
        print(
            "Reason: The selected point is far from the current best observation. "
            "If the GP and acquisition functions agree, this should be interpreted as "
            "a deliberate exploration of a second promising basin rather than a failure."
        )

    if nearest_sample_distance < 0.02:
        print(
            "Nearest-sample note: the candidate is also very close to an existing sample, "
            "so this is a dense local refinement."
        )
    elif nearest_sample_distance < 0.10:
        print(
            "Nearest-sample note: the candidate is near previously sampled data, "
            "so it is refining a known region rather than jumping into completely new space."
        )
    else:
        print(
            "Nearest-sample note: the candidate is not close to existing samples, "
            "so the query will add information in a relatively unexplored part of the domain."
        )

    print("\n9. INTERPRETATION")

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

    if d_nn is not None:
        if d_nn < 0.15:
            print("The GP ensemble and NN surrogate broadly agree on the same promising region.")
        elif d_nn < 0.30:
            print("The NN surrogate gives partial support, but is displaced from the final GP candidate.")
        else:
            print("The GP ensemble and NN surrogate disagree, so the neural surrogate is not supporting this query.")

    print("\n10. OPTIMISER CONFIDENCE")

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

    if acquisition_spread is not None:
        if acquisition_spread < 0.15:
            confidence_score += 1.0
            confidence_reasons.append("GP acquisition functions broadly agree")
        elif acquisition_spread < 0.30:
            confidence_score += 0.5
            confidence_reasons.append("GP acquisition functions show moderate agreement")

    if d_thompson is not None:
        if d_thompson < 0.15:
            confidence_score += 1.0
            confidence_reasons.append("Thompson sampling supports the GP candidate")
        elif d_thompson < 0.30:
            confidence_score += 0.5
            confidence_reasons.append("Thompson sampling gives partial support")

    if d_nn is not None:
        if d_nn < 0.15:
            confidence_score += 1.0
            confidence_reasons.append("neural surrogate supports the GP candidate")
        elif d_nn < 0.30:
            confidence_score += 0.5
            confidence_reasons.append("neural surrogate gives partial support")

    print(f"Overall optimiser confidence: {confidence_label(confidence_score)}")
    print(f"Confidence score: {confidence_score:.1f} / 5.0")
    print("Reasons:")

    if confidence_reasons:
        for reason in confidence_reasons:
            print(f"  - {reason}")
    else:
        print("  - no strong agreement signals were detected")

    print("\n11. FINAL RECOMMENDATION")
    print(f"Submit next query: {arr(best_input)}")
    print("\n===============================================================\n")