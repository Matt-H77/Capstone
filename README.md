# Black-Box Optimisation (BBO) Capstone Project

## 1. Project Overview

This capstone project is a Black-Box Optimisation (BBO) challenge based on Bayesian optimisation principles. The objective is to find the maximum of eight unknown functions using only a limited set of initial observations and a restricted number of future queries.

Each function represents a realistic optimisation problem where evaluations are expensive and only a small number of samples can be collected. The project mirrors many real-world machine learning and engineering problems where exhaustive search is impractical.

- 📊 **Dataset Datasheet** – [`BBO_Dataset_Datasheet.md`](BBO_Dataset_Datasheet.md)
- 🤖 **Model Card** – [`BBO_Model_Card.md`](BBO_Model_Card.md)
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
**Data Management**

For each optimisation task, the current set of observations is loaded and updated with newly acquired query results. Objective values are transformed where necessary so that all optimisation tasks are formulated as maximisation problems. Duplicate observations are identified, and candidate points that are too close to existing samples are excluded during the optimisation process to encourage exploration of previously unexplored regions.

**Candidate Generation**

A large pool of candidate points is generated using Latin Hypercube Sampling (LHS). The number of generated candidates scales with problem dimensionality (2D–8D), providing broad coverage of the search space while maintaining computational efficiency. This initial candidate pool forms the basis for subsequent filtering and optimisation.

**SVM-Guided Candidate Filtering**

A Radial Basis Function (RBF) Support Vector Machine (SVM) is trained using the observed data to distinguish between high-yield and low-yield regions of the search space.

The trained classifier is used to guide candidate selection by:

* retaining candidates with a high predicted probability of belonging to high-yield regions;
* reintroducing a proportion of randomly selected candidates to preserve global exploration; and
* concentrating computational effort on the most promising regions while maintaining sufficient diversity within the candidate pool.

**Gaussian Process Surrogate Ensemble**

Gaussian Process Regression (GPR) provides the primary surrogate modelling framework used throughout the optimisation process.

Multiple Gaussian Process models employing different kernel configurations are evaluated, including:

* Radial Basis Function (RBF)
* Matérn
* Rational Quadratic

The resulting Gaussian Process models are combined into an ensemble to improve predictive robustness and reduce dependence on any individual kernel configuration. Ensemble predictions provide both the estimated objective value and the associated predictive uncertainty for every candidate point.

**Automatic Hyperparameter Optimisation**

Gaussian Process hyperparameters are selected automatically using Leave-One-Out Cross Validation (LOOCV) as the model selection criterion.

The optimisation process considers:

* kernel selection;
* observation noise (α);
* kernel length-scale bounds;
* Matérn smoothness parameter (ν); and
* Rational Quadratic α parameter.

The best-performing configurations are incorporated into the final Gaussian Process ensemble.

**Acquisition Functions**

Candidate points are evaluated using three standard Bayesian optimisation acquisition functions:

* Expected Improvement (EI)
* Upper Confidence Bound (UCB)
* Probability of Improvement (PI)

Each acquisition function captures different exploration and exploitation characteristics, allowing the optimisation process to balance searching promising regions against investigating uncertain areas of the search space.

**Adaptive Exploration–Exploitation Strategy**

The acquisition function outputs are normalised and combined into a single hybrid acquisition score. Adaptive weighting gradually shifts the optimisation strategy from exploration towards exploitation as additional observations become available.

This dynamic weighting enables broader exploration during the early stages of optimisation while increasingly focusing on high-performing regions as confidence in the surrogate model improves.

**Thompson Sampling Comparison**

Posterior samples are drawn from the Gaussian Process ensemble to generate an independent candidate recommendation using Thompson Sampling. This provides an alternative Bayesian optimisation strategy that samples directly from the surrogate posterior and serves as an independent comparison with the primary hybrid acquisition approach.

**Neural Network Surrogate**

A neural-network ensemble is trained as an alternative surrogate model using the observed data. The learned response surface is subsequently optimised to generate an additional candidate recommendation.

Rather than replacing the primary Bayesian optimisation framework, the neural-network recommendation is used as an independent comparison to assess agreement between different surrogate modelling approaches and to provide additional confidence in the selected candidate.

**Candidate Comparison and Recommendation**

Candidate recommendations produced by the hybrid Gaussian Process acquisition function, Thompson Sampling, the neural-network surrogate and the highest-confidence SVM prediction are compared before each weekly submission.

For each candidate, the following information is evaluated:

* predicted objective value;
* predictive uncertainty;
* acquisition score;
* SVM confidence;
* distance from previously evaluated samples; and
* candidate feasibility.

The hybrid Gaussian Process acquisition function provides the primary automatic recommendation, while the remaining methods offer independent comparisons that support the final submission decision.

**Diagnostic Analysis**

A comprehensive suite of diagnostic tools is used throughout the optimisation process to evaluate model behaviour and candidate quality. These diagnostics include:

* Gaussian Process posterior slice visualisations;
* training fit diagnostics;
* acquisition function score distributions;
* predicted mean versus uncertainty analysis;
* Thompson Sampling comparisons;
* neural-network candidate comparisons;
* SVM confidence and decision-boundary visualisations; and
* candidate comparison tables summarising recommendations from all optimisation methods.

These diagnostics provide insight into surrogate model performance, exploration behaviour and candidate selection, helping to ensure that each submitted query is supported by multiple complementary analyses

---

## 5. Project Architecture

<img width="1024" height="1536" alt="WEEKLY BAYESIAN OPTIMISATION PIPELINE" src="https://github.com/user-attachments/assets/e3496c7e-7798-4172-b77b-6fd8e0f50b05" />


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

### Week 6 - Improved reporting and plots

Worked on improving reporting and explanation of next choice.

**Key improvements:**

- Added more detailed report on models and reasoning for next choice of candidate.
- GP diagnostics, tweaked plots for better visual inspection.

Summary

### Week 7 - Improved reporting and code refactoring

During Week 7, the focus shifted from developing new optimisation algorithms to improving the robustness, interpretability, and usability of the optimisation framework. The Bayesian Optimisation pipeline was enhanced with additional machine learning support, expanded diagnostics, and improved reporting to provide greater confidence when selecting the next experimental evaluation. The framework now not only identifies promising candidates but also explains the reasoning behind each recommendation using evidence from multiple models.

**Key Improvements:**

- Enhanced Candidate Comparison
- Developed a comprehensive comparison table showing recommendations from the Gaussian Process, Thompson Sampling, Neural Network, and SVM.
- Added rankings, uncertainty measures, acquisition function values, and distances between model recommendations to simplify candidate evaluation.
- Added a new reporting section that measures agreement between the Gaussian Process, Thompson Sampling, Neural Network, and SVM recommendations.
- Improved confidence in candidate selection by identifying when multiple independent models converge on the same region of the search space.
- Extended the automatically generated optimisation report with additional diagnostic sections, including SVM candidate analysis and cross-model agreement.
- Improved the interpretability of optimisation results by providing justification for the recommended experimental point rather than only reporting the final candidate.
- Applied the enhanced reporting framework to analyse optimisation behaviour on benchmark functions, providing deeper insight into convergence, uncertainty, and the exploration–exploitation balance.
- Continued modularising the project by moving analysis and reporting functionality from the notebook into reusable Python helper modules.
- Improved code readability, maintainability, and ease of future development.

## Week 8

This week focused on validating the optimisation results by re-evaluating the best candidate from the previous week for each function, with the exception of Function 2 where the alternative high-potential basin was intentionally explored. The repeated evaluations were used to determine whether the objective functions behaved deterministically or exhibited observation noise.

### Key Improvements

* Validated the current optimum for Functions 1, 4, 5, 7 and 8 by obtaining identical outputs from repeated evaluations, confirming deterministic behaviour.
* Investigated the secondary basin identified by the surrogate model for Function 2, reducing uncertainty in an unexplored region of the search space.
* Identified observation noise in Functions 3 and 6, where repeated evaluations of the same input produced different outputs, providing an empirical estimate of measurement variability.
* Updated the optimisation strategy to remove exact duplicate observations from deterministic datasets while retaining repeated measurements with differing outputs for noisy functions.
* Increased confidence in the surrogate models by experimentally validating the assumptions underlying the Bayesian optimisation process.

## Week 9

This week focused on moving the optimisation framework from a working Bayesian optimiser into a more robust and defensible optimisation system. Rather than introducing new optimisation algorithms, the emphasis was on validating the current approach, improving model stability and developing a more informed strategy for selecting the final evaluation point.

Work completed
Gaussian Process stability improvements
* Investigated and resolved several Gaussian Process fitting issues, including convergence warnings.
* Improved kernel behaviour and model robustness so that the GP fits more reliably as additional observations are collected.
* Continued monitoring posterior behaviour through prediction surfaces and optimisation reports.
Analysis of Function 1
* Analysed the Week 9 optimisation report.
* Determined that the GP is converging towards a single optimum with decreasing uncertainty.
* Concluded that exploration should now be reduced and future evaluations should largely follow the GP recommendation.
Analysis of Function 2
* Performed a detailed analysis of the optimisation landscape.
* Examined optimisation reports, posterior plots and candidate rankings.
* Investigated whether a second unexplored basin exists.
* Concluded that although there is some evidence of another possible region, the current data strongly supports continuing to exploit the known optimum until later in the remaining evaluation budget.
* Established a strategy of delaying any large exploratory move until only a few evaluations remain.
Candidate selection strategy

Rather than selecting the next point purely from the acquisition function, we refined the decision process by comparing multiple candidate generation methods, including:

* Gaussian Process optimum
* Thompson Sampling
* Neural Network proposal
* SVM-filtered candidate

These comparisons helped determine when alternative methods provide genuinely new information versus simply duplicating the GP recommendation.

Optimisation report interpretation

Considerable effort was spent understanding what the optimisation reports reveal, including:

* exploration vs exploitation weights
* expected improvement behaviour
* uncertainty estimates
* SVM candidate filtering
* posterior confidence
* evidence for multiple optima

This has made the optimisation process much easier to interpret rather than simply accepting the highest acquisition value.

## Key improvements

* Improved Gaussian Process stability and convergence.
* Better understanding of when exploration is still worthwhile.
* Developed a more structured decision process for choosing the next evaluation point.
* Increased confidence that Function 1 is approaching convergence.
* Identified a practical strategy for handling the possible second basin in Function 2.
* Improved interpretation of optimisation reports, making decisions more evidence-driven rather than relying solely on acquisition function values.
* Continued documenting the optimisation process with reflections that connect theoretical machine learning concepts to practical Bayesian optimisation decisions.

Overall, this week represented a shift from developing the optimisation framework to critically evaluating its behaviour and using that understanding to make more informed decisions about the remaining evaluation budget.

## Week 10

This week focused on improving the robustness and reliability of the Bayesian Optimisation pipeline by investigating why the Gaussian Process (GP) repeatedly selected the same candidate point. Analysis showed that while duplicate observations were removed from the training dataset, previously evaluated inputs could still remain in the optimisation candidate pool, allowing the acquisition function to repeatedly recommend the same location.

## Key Improvements
* Added candidate filtering to remove previously evaluated (or near-evaluated) inputs before GP prediction and acquisition optimisation.
* Implemented an efficient KD-Tree nearest-neighbour search to identify and exclude candidates within a configurable minimum distance of existing observations.
* Improved duplicate handling by distinguishing between:
  * duplicate observations in the training data (removed to avoid redundant GP updates), and
  * duplicate candidate locations (removed to prevent repeated evaluations).
* Added diagnostic reporting showing:
  * the number of candidates removed,
  * the number of remaining candidates, and
  * the minimum distance between retained candidates and previous observations.
* Reviewed the candidate generation process and identified that using a fixed Latin Hypercube seed produced the same candidate set each iteration. This was documented, with support added for using iteration-specific seeds while maintaining reproducibility if required.
* Added validation checks to ensure the final GP-selected candidate satisfies the minimum-distance constraint before submission.

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
