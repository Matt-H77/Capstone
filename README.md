# Black-Box Optimisation (BBO) Capstone Project

## 1. Project Overview

This capstone project is a Black-Box Optimisation (BBO) challenge based on Bayesian optimisation principles. The objective is to find the maximum of eight unknown functions using only a limited set of initial observations and a restricted number of future queries.

Each function represents a realistic optimisation problem, including radiation source detection, drug discovery, warehouse logistics, recipe optimisation, chemical yield improvement, and machine learning hyperparameter tuning. In all cases, the underlying function is hidden, evaluations are expensive, and only a small number of samples can be collected.

The project mirrors many real-world machine learning and engineering problems where exhaustive search is impractical. Instead of knowing the exact relationship between inputs and outputs, I must build surrogate models that estimate the function behaviour and guide future sampling decisions.

This project is valuable for my professional development because it strengthens skills in model-based optimisation, uncertainty quantification, experimental design, hyperparameter tuning, and decision-making under uncertainty.

---

## 2. Inputs and Outputs

The project provides initial observations as NumPy arrays:

```python
X = np.load("initial_inputs.npy")
y = np.load("initial_outputs.npy")
```

Where:

```python
X.shape = (n_samples, n_dimensions)
y.shape = (n_samples,)
```

The dimensionality varies across functions from 2D to 8D.

Example input:

```python
[0.42, 0.71, 0.18]
```

Example output:

```python
0.684
```

The input represents a candidate query point in the search space, while the output represents the observed performance score returned by the black-box function.

All tasks are framed as maximisation problems.

---

## 3. Challenge Objectives

The goal is to identify input combinations that maximise the unknown function value while using as few queries as possible.

Key constraints include:

- Limited query budget
- Unknown function structure
- Noisy observations
- Multiple local optima
- Increasing difficulty with dimensionality
- Expensive evaluations

The challenge is to learn from previous observations and make informed decisions about where to sample next.

---

## 4. Technical Approach

### Surrogate Modelling

I use Gaussian Process (GP) regression as the primary surrogate model because it provides both predicted function values and uncertainty estimates.

Kernel families evaluated include:

- RBF
- Matern
- Rational Quadratic

### Candidate Generation

Candidate points are generated using Latin Hypercube Sampling (LHS) to achieve broad coverage of the search space.

### Acquisition Functions

I combine multiple acquisition functions:

- Upper Confidence Bound (UCB)
- Expected Improvement (EI)
- Probability of Improvement (PI)

### Adaptive Exploration vs Exploitation

Exploration and exploitation weights are adjusted dynamically. Early rounds prioritise exploration, while later rounds increasingly focus on promising regions identified by the model.

### Hyperparameter Optimisation

Model hyperparameters are tuned automatically using:

- Leave-One-Out Cross Validation
- Kernel comparison
- Noise parameter tuning
- Length-scale optimisation

### Ensemble Modelling

Predictions from multiple Gaussian Process models are combined into an ensemble. This provides more robust mean predictions and uncertainty estimates.

### Future Considerations

Potential future work includes:

- Soft-margin SVM classification of high- and low-performing regions
- Kernel SVMs for non-linear response surfaces
- Alternative surrogate models for higher-dimensional problems

### Current Understanding

My current strategy treats optimisation as a sequential learning problem. Each query is selected using surrogate modelling, uncertainty estimation, validation, and adaptive acquisition functions to maximise the value gained from a limited evaluation budget.
