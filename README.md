# Black-Box Optimisation (BBO) Capstone Project

## 1. Project Overview

This capstone project is a Black-Box Optimisation (BBO) challenge based on Bayesian optimisation principles. The objective is to find the maximum of eight unknown functions using only a limited set of initial observations and a restricted number of future queries.

Each function represents a realistic optimisation problem where evaluations are expensive and only a small number of samples can be collected. The project mirrors many real-world machine learning and engineering problems where exhaustive search is impractical.

- 📊 **Dataset Datasheet** – [`BBO_Dataset_Datasheet.md`](BBO_Dataset_Datasheet.md)
- 🤖 **Model Card** – [`BBO_Model_Card.md`](BBO_Model_Card.md)
- 📄 **Licence** – [`licence.md`](licence.md)
- A streamLit dashboard for this project is available at : https://capstone-m24jqknpen5vtbbpmqfusd.streamlit.app/
- 📈 [Jump to Results](#8-results)
- 📄 A reflection of the final project - [Final Project Reflection](Final_Project_Reflection.md)

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

**Principal Component Analysis (PCA) Diagnostics**

PCA is used as an additional diagnostic tool to improve interpretation
of the optimisation search space without changing the underlying
optimisation models or automatic candidate selection.

PCA is fitted to the observed input data and automatically retains the
minimum number of principal components required to explain at least 90%
of the observed input variance. Observations and candidate
recommendations from the hybrid GP, Thompson Sampling, neural-network
surrogate and SVM can then be projected into the same PCA space.

The PCA diagnostics provide:

* explained variance and cumulative explained variance;
* principal-component loadings;
* automatic selection of enough components to explain at least 90% of
the variance;
* visualisation of observations and model-selected candidates in
lower-dimensional PCA space;
* highlighting of the best observed point and high-performing
observations;
* comparison of candidate distances in PCA space and the original
input space;
* cross-model agreement analysis;
* assessment of whether high-performing observations form compact
regions or clusters; and
* extended diagnostics examining relationships between retained
principal components and observed objective values.

PCA is used only for diagnostic interpretation. It does not alter the
Gaussian Process, SVM, neural-network surrogate, Thompson Sampling,
acquisition functions or final automatic recommendation.

These diagnostics provide insight into surrogate model performance,
exploration behaviour, search-space structure and candidate selection,
helping to ensure that each submitted query is supported by multiple
complementary analyses.

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

## Week 11 -- PCA Diagnostics and Search-Space Structure

PCA was introduced to provide a new diagnostic view of the observed
search space and model recommendations.

## Key improvements:

* Added PCA diagnostics without altering the optimisation model.
* Automatically retained enough PCs to explain at least 90% of
  observed input variance.
* Added PCA loading and explained-variance reporting.
* Projected observed samples and GP, Thompson, NN, and SVM candidates
  into retained PCA space.
* Added lower-triangle PCA projection matrices.
* Added top-performing observation highlights.
* Added extended diagnostics for candidate-to-high-performance-region
  distance.
* Added cross-model agreement measurements in retained PCA space.
* Added analysis of high-performing-region compactness and PC/output
  association.

## Week 12 -- Final Terminal Exploitation and Candidate Refinement

The final week focused on converting the optimisation strategy from
exploration-driven Bayesian optimisation into terminal exploitation.
Because no future evaluations remained after this round, the objective
was changed from learning more about the search space to selecting the
candidate with the highest expected final performance.

The Gaussian Process ensemble remained the primary surrogate model, but
candidate selection was based directly on maximum posterior mean rather
than exploration-weighted acquisition scores. EI, UCB and PI were
retained as diagnostics, while Thompson Sampling was disabled for the
final round because additional exploratory information could no longer
be exploited in later iterations.

To reduce dependence on any single Latin Hypercube random seed, the
candidate pools from all previous weeks were reconstructed and rescored
using the final retrained GP ensemble. This provided a much larger and
more representative terminal search space.

### Key improvements:

* Changed the final-week optimisation objective to pure terminal
  exploitation using maximum GP ensemble posterior mean.

* Retained EI, UCB and PI as diagnostic measures but removed their
  influence from the final candidate-selection decision.

* Disabled Thompson Sampling for the final round because there was no
  remaining evaluation budget in which exploratory information could
  provide future benefit.

* Reconstructed and combined the Latin Hypercube candidate pools from
  previous weeks and rescored them using the final GP ensemble,
  reducing sensitivity to a single random candidate seed.

* Reduced the minimum candidate-to-observation distance for the final
  round, allowing much tighter exploitation around previously identified
  high-performing regions while still preventing exact duplicate
  submissions.

* Added explicit boundary candidates where the optimisation evidence
  suggested that the optimum was approaching the edge of the search
  space.

* Added terminal local-search diagnostics for functions where the global
  candidate pool did not provide sufficient resolution around the
  incumbent.

* For Function 4, identified that the global GP ensemble was
  over-smoothing a narrow high-performing region. A local Matérn GP was
  fitted to nearby observations and used for final terminal refinement.

* For Function 7, confirmed that the global GP accurately reproduced the
  incumbent and therefore retained the existing surrogate. A dense local
  search around the incumbent identified a nearby point with a slightly
  higher posterior mean, with a line-search diagnostic confirming a
  smooth increase toward the selected candidate.

* For Function 8, performed a dense local refinement followed by a fine
  one-dimensional line search. This identified a posterior-mean maximum
  within the high-performing region that was not resolved by the broader
  multi-million-point candidate pool.

* Preserved the original GP candidate recommendations in the exported
  JSON reports while adding a separate `final_submission_candidate`
  field for functions requiring specialist terminal refinement.

* Updated the final reporting workflow so that Streamlit and JSON output
  clearly distinguish between the broad GP recommendation and the
  actual final submission candidate.

Overall, Week 12 represented the final transition from exploration and
model learning to exploitation of the best-supported regions discovered
throughout the project. The final selections were therefore based on the
best available posterior evidence, supported by local diagnostics,
PCA interpretation, cross-model comparisons and targeted refinement
where necessary.
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
9. PCA and extended PCA diagnostic analysis.
10. Diagnostic analysis and candidate review.

PCA is used as a supporting diagnostic rather than an optimisation or
candidate-selection method. It helps interpret search-space structure,
high-performing regions and agreement between the GP, Thompson Sampling,
neural-network and SVM recommendations.

This creates a robust Bayesian optimisation framework that balances
exploration, exploitation, uncertainty quantification, model diversity
and search-space interpretation while operating under a limited
evaluation budget.
---

## 8. Results

The final optimisation results are summarised below. Each function was treated as a
maximisation problem and evaluated under a limited query budget. The reported input
is the final recommended point, while the output is the best observed transformed
objective value.

| Function | Dimensions | Initial samples | Final samples | Best value | Recommended input |
|---|---:|---:|---:|---:|---|
| Function 1 | 2D | 10 | 23 | `-0.015465` | `[0.634176, 0.677097]` |
| Function 2 | 2D | 10 | 23 | `0.730244` | `[0.706241, 0.932154]` |
| Function 3 | 3D | 15 | 28 | `-0.476330` | `[0.01547, 0.872867, 0.99982]` |
| Function 4 | 4D | 30 | 43 | `0.672076` | `[0.421802, 0.358916, 0.419253, 0.377334]` |
| Function 5 | 4D | 20 | 33 | `8662.482500` | `[1.0, 1.0, 1.0, 1.0]` |
| Function 6 | 5D | 20 | 33 | `-3.608531` | `[1.0, 1.0, 0.0, 0.0, 1.0]` |
| Function 7 | 6D | 30 | 43 | `2.845336` | `[0.184378, 0.236845, 0.520662, 0.201635, 0.376267, 0.713072]` |
| Function 8 | 8D | 40 | 53 | `9.998474` | `[0.109209, 0.16213, 0.138848, 0.154742, 0.842353, 0.500099, 0.200835, 0.576853]` |

### Weekly Results

Detailed reports and diagnostic outputs for each optimisation week are available here:

- [Week 4 results](weekly_results/week4/)
- [Week 5 results](weekly_results/week5/)
- [Week 6 results](weekly_results/week6/)
- [Week 7 results](weekly_results/week7/)
- [Week 8 results](weekly_results/week8/)
- [Week 9 results](weekly_results/week9/)
- [Week 10 results](weekly_results/week10/)
- [Week 11 results](weekly_results/week11/)
- [Week 12 results](weekly_results/week12/)
