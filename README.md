# Black-Box Optimisation (BBO) Capstone Project

## 1. Project Overview

This capstone project is a Black-Box Optimisation (BBO) challenge based on Bayesian optimisation principles. The objective is to find the maximum of eight unknown functions using only a limited set of initial observations and a restricted number of future queries.

Each function represents a realistic optimisation problem where evaluations are expensive and only a small number of samples can be collected. The project mirrors many real-world machine learning and engineering problems where exhaustive search is impractical.

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

---

## 4. Technical Approach

### Data Management

For each optimisation task, the current set of observations is loaded and updated with newly acquired query results. Outputs are transformed where necessary so that all tasks are framed as maximisation problems.

### Candidate Generation

A large pool of candidate points is generated using Latin Hypercube Sampling (LHS). The number of candidates scales with problem dimensionality (2D–8D).

### SVM-Guided Candidate Filtering

An RBF Support Vector Machine is used to classify observed samples into high-yield and low-yield regions.

- Promising candidates are retained based on predicted probability of high yield.
- Random candidates can be reintroduced to maintain exploration.
- Computational effort is focused on the most promising regions.

### Gaussian Process Surrogate Ensemble

Gaussian Process Regression is used as the primary surrogate modelling framework.

Kernel families evaluated include:

- RBF
- Matern
- Rational Quadratic

Multiple GP configurations are combined into an ensemble to improve robustness.

### Automatic Hyperparameter Optimisation

GP hyperparameters are tuned automatically using Leave-One-Out Cross Validation (LOOCV).

The search includes:

- Kernel selection
- Observation noise (alpha)
- Length-scale bounds
- Matern smoothness parameter (ν)
- Rational Quadratic alpha

### Acquisition Functions

Candidate points are evaluated using:

- Expected Improvement (EI)
- Upper Confidence Bound (UCB)
- Probability of Improvement (PI)

### Adaptive Exploration vs Exploitation

Normalised acquisition scores are combined into a hybrid acquisition function that gradually shifts from exploration to exploitation as more data becomes available.

### Thompson Sampling Comparison

Posterior samples are drawn from the GP ensemble to generate an independent candidate recommendation.

### Neural Network Surrogate

A neural-network ensemble is trained as an alternative surrogate model. Gradient ascent is performed on the learned response surface to generate candidate recommendations.

### Diagnostic Analysis

Before submission, diagnostic tools are used to assess model behaviour and candidate quality:

- GP posterior slice visualisations
- Training fit diagnostics
- Acquisition score distributions
- Mean versus uncertainty analysis
- Thompson sampling comparisons
- Neural-network candidate comparisons
- SVM decision-boundary visualisations

---

## 5. Project Architecture

```text
┌─────────────────────────┐
│ Initial Dataset         │
│ X, y observations       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Data Update Stage       │
│ Append weekly results   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Candidate Generation    │
│ Latin Hypercube         │
│ Sampling (LHS)          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ SVM High-Yield Filter   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ GP Hyperparameter       │
│ Optimisation            │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ GP Ensemble             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ EI + UCB + PI           │
│ Acquisition Functions   │
└───────┬─────────┬───────┘
        │         │
        ▼         ▼
┌────────────┐ ┌────────────┐
│ Thompson   │ │ Neural Net │
│ Sampling   │ │ Surrogate  │
└─────┬──────┘ └─────┬──────┘
      └──────┬───────┘
             ▼
┌─────────────────────────┐
│ Diagnostics             │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Final Submission        │
└─────────────────────────┘
```

---

## 6. Evolution of the Optimisation Strategy

### Week 1 – Initial Gaussian Process Surrogate

The first iteration focused on implementing a Gaussian Process surrogate model. Candidate selection was based on GP predictions and acquisition functions.

**Key achievements:**

- Implemented GP regression.
- Generated LHS candidate pools.
- Established the Bayesian optimisation workflow.

### Week 2 – Gaussian Process Ensemble

The second iteration expanded the approach from a single GP model to an ensemble of Gaussian Processes.

**Key improvements:**

- Added RBF, Matern and Rational Quadratic kernels.
- Implemented kernel comparison and model averaging.
- Incorporated model disagreement into uncertainty estimates.

### Week 3 – SVM-Guided Candidate Filtering

An SVM classifier was introduced to identify high-yield regions before GP evaluation.

**Key improvements:**

- Implemented RBF SVM classification.
- Reduced evaluation of low-quality candidates.
- Added probability-based candidate filtering.
- Preserved global exploration through random reinjection.

### Week 4 – Neural Network Surrogate Exploration

A neural-network surrogate was introduced to provide an alternative view of the response surface.

**Key improvements:**

- Implemented a neural-network surrogate.
- Added neural-network ensemble averaging.
- Compared NN predictions against GP predictions.

### Week 5 – Neural Network Optimisation and Code Refactoring

The fifth iteration refined the neural-network surrogate and improved project structure.

**Key improvements:**

- Added gradient-based optimisation on the NN response surface.
- Compared GP, Thompson Sampling and NN candidate recommendations.
- Refactored functionality into reusable modules:
  - Data loading
  - Acquisition functions
  - GP diagnostics
  - GP hyperparameter tuning
  - SVM filtering
  - Neural-network surrogates
- Improved maintainability and readability of the codebase.

---

## 7. Current Strategy

The final workflow combines:

1. Data updating and preprocessing.
2. Latin Hypercube candidate generation.
3. SVM-guided candidate filtering.
4. GP hyperparameter optimisation.
5. GP ensemble modelling.
6. Hybrid acquisition scoring (EI + UCB + PI).
7. Thompson Sampling validation.
8. Neural-network surrogate comparison.
9. Diagnostic analysis and candidate review.

This creates a robust Bayesian optimisation framework that balances exploration, exploitation, uncertainty quantification, and model diversity while operating under a limited evaluation budget.
