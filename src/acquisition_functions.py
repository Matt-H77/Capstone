# Acquisition functions for Bayesian Optimization
import numpy as np
from scipy.stats import norm

# Calculate Upper Confidence Bound score
def ucb(mean, std, beta=2.0):
    return mean + beta * std

# Calculate Expected Improvement score
def expected_improvement(mean, std, best_y, xi=0.01):
    std = np.maximum(std, 1e-12)
    improvement = (mean - best_y - xi)
    z = improvement / std
    return (improvement * norm.cdf(z) + std * norm.pdf(z))

# Calculate Probability of Improvement score
def probability_improvement(mean, std, best_y, xi=0.01):
    std = np.maximum(std, 1e-12)
    z = (mean - best_y - xi) / std
    return norm.cdf(z)

#   Normalize scores to [0, 1] range for fair combination
def normalize_score(score):
    score = np.asarray(score)
    return ((score - np.min(score)) / (np.max(score) - np.min(score) + 1e-12))

#  Normalize scores to [0, 1] range for fair combination with additional checks
def normalize_score_safe(score, min_range=1e-9, relative_tol=1e-3):
    score = np.asarray(score, dtype=float)

    s_min = np.nanmin(score)
    s_max = np.nanmax(score)
    s_range = s_max - s_min

    scale = max(abs(s_max), abs(s_min), 1e-12)

    if (not np.isfinite(s_range)) or s_range < min_range or s_range < relative_tol * scale:
        return np.zeros_like(score)

    return (score - s_min) / s_range