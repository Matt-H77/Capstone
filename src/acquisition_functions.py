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