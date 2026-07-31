# Model Card
## Adaptive Bayesian Black-Box Optimisation Framework

## Overview

**Model name:** Adaptive Bayesian Optimisation using Gaussian Process Ensembles with SVM Candidate Filtering

**Model type:** Sequential Bayesian Optimisation Framework

**Version:** 1.0 (Final Capstone Submission)

---

## Intended Use

### Suitable tasks
This framework is designed for:
- Expensive black-box optimisation
- Continuous optimisation problems
- Small-data optimisation
- Hyperparameter optimisation
- Engineering design optimisation
- Scientific experimentation

### Unsuitable tasks
The framework is not intended for:
- Very high-dimensional optimisation
- Discrete combinatorial optimisation
- Real-time optimisation requiring thousands of evaluations
- Highly noisy objective functions
- Problems requiring guaranteed global optima

---

## Model Details

### Overall strategy
Across ten optimisation rounds the framework:

1. Updated the dataset with the latest observation.
2. Retrained Gaussian Process surrogate models.
3. Optimised kernel hyperparameters.
4. Generated a large Latin Hypercube candidate set.
5. Applied SVM filtering to remove low-probability regions.
6. Predicted mean and uncertainty using a GP ensemble.
7. Evaluated Expected Improvement, Probability of Improvement and Upper Confidence Bound.
8. Compared GP, Thompson Sampling, neural network and SVM recommendations.
9. Selected the final submission using predicted improvement, uncertainty and candidate diversity.

### Evolution
The optimisation framework evolved considerably throughout the project.

Initial versions used a single Gaussian Process with Expected Improvement.

Later iterations introduced:
- Gaussian Process ensembles
- Dynamic exploration scheduling
- Adaptive EI parameters
- SVM high-yield filtering
- Thompson Sampling
- Neural network candidate comparison
- Candidate ranking tables
- Automated optimisation reports
- Basin identification for multimodal functions

As the evaluation budget reduced, the strategy progressively shifted from exploration toward exploitation.

---

## Performance

### Metrics
Performance was assessed using:
- Best observed objective value
- Improvement over previous rounds
- Predicted GP mean
- Predictive standard deviation
- Expected Improvement
- Probability of Improvement
- Upper Confidence Bound
- Distance between candidate recommendations
- SVM confidence
- Candidate rankings

### Summary
The framework consistently identified strong candidate locations while operating under a limited evaluation budget.

Performance generally improved as more observations were collected, although convergence depended on the complexity and dimensionality of each benchmark function.

SVM filtering significantly reduced the search space while retaining promising candidate regions, improving computational efficiency.

---

## Assumptions and Limitations

### Assumptions
The optimisation framework assumes:
- Continuous objective functions
- Correlated nearby observations
- Gaussian Processes provide suitable surrogate models
- Candidate generation adequately samples the search space
- Available observations reasonably represent the optimisation landscape

### Limitations
Current limitations include:
- Extremely limited evaluation budget
- Increasing computational cost as datasets grow
- Possible rejection of valuable regions by SVM filtering
- Dependence on accurate uncertainty estimation
- Sensitivity to kernel and hyperparameter selection

---

## Ethical Considerations

Although the benchmark functions are synthetic and contain no personal or sensitive information, transparency remains important.

Documenting model assumptions, optimisation strategy, acquisition functions, candidate filtering and evaluation metrics improves reproducibility, supports responsible deployment and enables adaptation to real-world optimisation problems.

### Documentation Reflection
This model card provides sufficient information to understand the optimisation framework, reproduce the overall workflow and evaluate its strengths and limitations. Additional implementation details, such as exact kernel parameters, could improve developer reproducibility but are not essential for communicating the system's intended use, behaviour and constraints.
