# Final BBO Capstone Reflection
## Initial codebase
I began with a relatively simple Gaussian Process (GP)-based Bayesian optimisation implementation. I built the initial codebase from scratch using the concepts covered in the module, rather than taking it directly from an existing public project. I felt that developing the code myself would be an effective way to reinforce my learning and understand how the different components of Bayesian optimisation worked together.

I chose this starting point because Gaussian Processes are well suited to black-box optimisation when evaluations are expensive and only a limited number of observations are available. They provide both a predicted mean and an estimate of uncertainty, making it possible to balance exploitation of promising regions with exploration of uncertain areas.

The first version was intentionally simple. It scaled the input variables, trained a GP surrogate using the available observations, generated candidate points and selected new queries using an acquisition function. This provided a clear baseline that could be evaluated and improved throughout the project.
 

## Code modifications
The implementation developed considerably over the course of the project, with each week adding functionality in response to limitations identified in the previous iteration.

In Week 1, I implemented a single Gaussian Process surrogate and generated candidate points using Latin Hypercube Sampling. Candidate selection was based on GP predictions and acquisition functions. This established a working Bayesian optimisation pipeline and provided a baseline for later improvements.

In Week 2, I expanded the approach from a single GP to an ensemble using RBF, Matérn and Rational Quadratic kernels. I introduced kernel comparison and model averaging so that the optimiser was less dependent on the assumptions of one particular kernel. Model disagreement also contributed to the uncertainty estimates, making the recommendations more robust across functions with different levels of smoothness.

In Week 3, I added an RBF-based SVM classifier to identify high-yield regions before evaluating candidates with the GP. This reduced the number of low-quality candidates being considered and made the large candidate pools more computationally manageable. Because an overly restrictive filter could remove unexplored but valuable regions, I also introduced probability-based filtering and randomly reintroduced some candidates to preserve global exploration.

In Week 4, I introduced a neural-network surrogate as an independent view of the response surface. In Week 5, I extended this work by adding a neural-network ensemble and gradient-based optimisation of the neural-network response surface. The neural-network recommendations were compared with the GP and Thompson Sampling candidates rather than replacing the main GP-based approach. During Week 5, I also refactored the notebook into reusable modules for data loading, acquisition functions, GP diagnostics, hyperparameter tuning, SVM filtering and neural-network surrogates. This improved the maintainability and readability of the project.

In Week 6, I focused on improving the reporting and visualisation of the optimisation process. I added more detailed explanations of the models and the reasoning behind each candidate recommendation. I also improved the GP diagnostic plots so that posterior behaviour, uncertainty and candidate locations could be inspected more effectively.

In Week 7, I developed a more comprehensive candidate-comparison and reporting system. Recommendations from the GP, Thompson Sampling, neural network and SVM were compared using predicted objective values, uncertainty, acquisition scores, SVM confidence and distances from existing observations. Cross-model agreement analysis helped identify when several independent methods were converging on a similar region. This made the final query decisions more evidence-based instead of relying only on the highest acquisition score.

In Week 8, I used repeated evaluations to investigate whether the functions were deterministic or noisy. Repeated evaluations produced identical outputs for several functions, while Functions 3 and 6 showed evidence of observation noise. I updated the data-handling process so that exact duplicate observations were removed for deterministic functions, while repeated observations with different outputs were retained when they provided information about noise. Function 2 was treated differently because I intentionally explored a secondary high-potential basin rather than simply repeating its current best candidate.

In Week 9, I focused on GP stability, convergence and interpretation. I investigated convergence warnings and improved the reliability of the GP fits as more observations were added. I also developed a more structured decision process by comparing the GP, Thompson Sampling, neural-network and SVM recommendations. For Function 1, the analysis suggested that the model was approaching a single optimum, so exploration could be reduced. For Function 2, I identified evidence of a possible second basin and planned to delay a major exploratory move until later in the remaining evaluation budget.

In Week 10, I addressed a problem where the GP could repeatedly select the same candidate. Although duplicate observations had been removed from the training data, previously evaluated inputs could still remain in the candidate pool. I added distance-based filtering using a KD-Tree to remove candidates that were identical or too close to existing observations. I also added diagnostics showing how many candidates were removed and the minimum distance between retained candidates and previous samples. This prevented redundant submissions and improved the efficiency of the remaining query budget.

In Week 11, I introduced PCA diagnostics to provide an additional view of the search-space structure. PCA was not used to replace the optimisation models, but it helped identify the amount of variance explained by the input dimensions, the main loading directions, the compactness of high-performing regions and the agreement between candidate recommendations in a reduced-dimensional space. This improved my interpretation of the search process without changing the underlying candidate-selection method.

In Week 12, the strategy changed from exploration and model learning to terminal exploitation because no further evaluations would be available after the final submission. The GP ensemble remained the primary surrogate, but the final candidate was selected using the maximum posterior mean rather than exploration-weighted acquisition scores. EI, UCB and PI were retained as diagnostics, while Thompson Sampling was disabled because there was no future evaluation budget in which its exploratory information could be used.

For the final round, I reconstructed and combined candidate pools from previous weeks and rescored them using the final GP ensemble. This reduced dependence on a single Latin Hypercube seed. I also allowed tighter local searches around promising regions, added boundary candidates where appropriate and used specialist local refinement for selected functions. For Function 4, a local Matérn GP was used because the global ensemble appeared to over-smooth a narrow high-performing region. Functions 7 and 8 also received dense local searches and line-search diagnostics to refine the final candidate around their incumbent solutions.

The changes with the greatest practical impact were the GP ensemble, SVM candidate filtering, improved duplicate handling, cross-model candidate comparison and the final transition to terminal exploitation. The diagnostics and reporting improvements were also important because they helped explain why a candidate was selected and allowed the strategy to be adapted to the behaviour of each individual function.

 

## Final result
The final weeks produced more consistent and targeted queries than the early rounds because my strategy gradually shifted from predominantly exploratory behaviour towards greater exploitation. Initially, many queries were exploratory because there was little information about the shape of each function. As the data set grew, the optimiser increasingly concentrated on regions associated with high observed values. The acquisition weighting gradually shifted towards Expected Improvement, while still retaining uncertainty-based exploration through UCB and Thompson sampling.

The differences in my leaderboard positions across the functions suggest that the strategy performed better on some search landscapes than others. My strongest result was second place on Function 4. This indicates that the later improvements to the surrogate models, candidate selection and exploitation strategy were particularly effective for that function. The results on Functions 7 and 8 were also encouraging, especially because these were higher-dimensional problems where uncertainty remained significant. I feel that the knowledge gained throughout this process would allow me to analyse my models more effectively and produce a stronger overall strategy if I were to take part in another black-box optimisation challenge.

Some functions appeared to have relatively clear high-performing regions, while others contained multiple peaks or more irregular behaviour. For example, Function 2 showed evidence of a second promising region, so concentrating entirely on the current best point would have risked missing a better basin. In the higher-dimensional functions, uncertainty remained significant even near the end, which justified retaining some exploration.

If I had more time, I would allocate more evaluations to learning the global structure during the early and middle stages. I would also test the optimisation strategy on simulated functions with known optima before applying it to the capstone functions. This would help distinguish genuine improvements from changes caused by limited observations.

With a fresh start, I would design the data-management and experiment-tracking system earlier, including automatic recording of every model configuration, candidate source and evaluation result. I also did not know all of the available techniques at the beginning of the project, so some early submissions may have been less effective than they could have been. In particular, the initial strategy was not sufficiently robust, and some query opportunities were likely spent before I had developed the more systematic ensemble, filtering and diagnostic methods used later in the project.

 

## Trade-offs and decisions
The main trade-off was between exploration and exploitation. Exploitation was attractive in the final weeks because the remaining query budget was limited and the current best regions had already produced good results. However, excessive exploitation could repeatedly sample the same basin, particularly when the surrogate was overconfident. I therefore used a hybrid strategy that favoured the best predicted regions while preserving a controlled amount of random, Thompson-sampling and UCB-based exploration. In hindsight, I may have continued exploring too much as the weeks progressed and should probably have shifted towards exploitation earlier for some functions.

Another trade-off concerned model complexity. A single GP was easier to interpret and faster to run, but it could be too dependent on the assumptions of its selected kernel. An ensemble was more robust, but required more computation and introduced additional decisions about how to combine the predictions. Similarly, the neural surrogate added flexibility but was not always reliable when trained on very small data sets.

Candidate filtering created another important trade-off. The SVM reduced computational cost and focused the optimiser on regions similar to those associated with previous successes. However, an overly strict filter could remove the global optimum if it lay in a previously unexplored region. I addressed this by using probability thresholds, minimum candidate counts and a random exploration fraction.

Finally, I had to decide whether to keep all observations, including duplicate inputs. Exact duplicates with identical outputs added little new information, whereas duplicate inputs with different outputs helped estimate the noise in the objective function. I therefore treated these cases differently rather than removing all duplicates automatically.

 

## Learning and application
The most important lesson was that Bayesian optimisation is not just about selecting an acquisition function. The quality of the complete pipeline depends on data preprocessing, surrogate assumptions, candidate generation, noise handling, diagnostics and the decision about how much uncertainty to tolerate. A theoretically strong method can still perform poorly if the candidate pool is biased, the model is overconfident or the search becomes too narrow too early.

I would apply this lesson to future competitions by starting with a reliable baseline, recording results systematically and making one controlled change at a time. In real-world ML projects, this is equally important. In my work on broadcast graphics and sports video systems, evaluations can be expensive because they may involve processing large video data sets or running GPU-based models. Surrogate modelling and active experimentation could reduce the number of tests needed when tuning tracker parameters, calibration settings or model configurations.

The project also reinforced the importance of visual diagnostics. Looking at posterior surfaces, uncertainty and candidate rankings often revealed behaviour that was not obvious from a single score. This is directly applicable to computer vision and graphics pipelines, where understanding failure regions can be as important as maximising average performance.

What surprised me most was how quickly the optimisation process became dependent on the quality of early decisions. A few successful observations could strongly influence the apparent structure of the search space, even when the true function had another promising region elsewhere. I was also surprised by how different strategies could perform across functions: a method that worked well for a smooth low-dimensional function was not necessarily best for a noisy or high-dimensional one.

Overall, the capstone changed my view of optimisation from repeatedly trying promising values to managing uncertainty and information gain. The final system was more systematic, more interpretable and better suited to adapting its behaviour as new evidence became available.
