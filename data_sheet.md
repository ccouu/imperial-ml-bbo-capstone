# Datasheet: Black-Box Optimization Portfolio (Functions 1–8)

## Function Overview
This datasheet describes the full suite of **Functions 1 through 8**, which simulate a variety of complex real-world optimization scenarios. The domains include contamination detection (`F1`), noisy machine learning log-likelihoods (`F2`), drug discovery side-effect minimization (`F3`), dynamic warehouse product placement (`F4`), chemical factory yield optimization (`F5`), recipe tuning (`F6`), and high-dimensional machine learning hyperparameter tuning (`F7` and `F8`). 

The dimensionality of the inputs scales across the portfolio, ranging from `2D` up to `8D` arrays. In all cases, the initial dataset provided a sparse uniform seed of observations, and the output represents a `1D` scalar score where the universal objective is **maximization** (often achieved by maximizing the negative of a cost or side-effect).

## Nature of the Data
The input structures consist of `N x D` arrays of coordinates strictly confined within a `[0, 1]` hypercube, returning a `1D` array of scalar performance scores. 

Over the weekly iterations, the datasets evolved from broad, space-filling exploration to highly clustered sampling around specific high-yield regions as the acquisition policy shifted from exploration to exploitation. The presence of noise, randomness, and topographical complexity varies drastically across the functions based on their official definitions:
*   **Highly Noisy and Multimodal:** **Function 2** explicitly contains high aleatoric noise and deceptive local optima, making it highly sensitive to early sampling traps. **Function 4** is incredibly dynamic and packed with local optima.
*   **Smooth and Unimodal:** **Function 5** features a single, distinct peak representing optimal chemical yield. **Function 8** proved to be exceptionally smooth and deterministic, perfectly mapping to a concave, second-degree polynomial manifold without aleatoric noise spikes.
*   **Sparse Signals:** **Function 1** requires exact proximity to trigger non-zero readings, meaning the vast majority of the search space returns a flat response. 
*   **High-Dimensional Sparsity:** Mid-to-high dimensional spaces (`F6`, `F7`, `F8`) present severe data sparsity challenges, where distance metrics degrade and global optimization becomes exceedingly difficult without assuming underlying parametric trends.

## Optimisation Strategy
The primary approach utilized an **Ensemble Regression-Kriging** architecture paired with an **Upper Confidence Bound (UCB)** acquisition function. Standard Gaussian Processes typically fail in higher dimensions (`4D` to `8D`) due to the curse of dimensionality, so Regression-Kriging was deployed to capture the macro-slopes of the space using parametric baselines (such as `Support Vector Regressors`, `Polynomials`, and `XGBoost`). To protect the dense `2D` functions (`F1`, `F2`) from over-parameterized model bias, an **automated fallback gate** was implemented to revert to a standard Gaussian Process if the ensemble's cross-validation error exceeded a safe threshold.

Exploration was aggressively forced in early rounds by setting the UCB exploration factor (`kappa`) between `2.0` and `3.0`. As diagnostics confirmed structural legibility in specific functions (like the polynomial structure in `F5` and `F8`), `kappa` was decayed to `0.0`. The strategy then executed a critical pivot for structured landscapes: abandoning stochastic Bayesian sampling entirely in favor of **bounded L-BFGS-B deterministic gradient optimization** calculated directly on the fitted global trend pipeline.

## Data Handling and Preprocessing
To strictly enforce parameter boundaries during analytical solving, all inputs were processed through a `MinMaxScaler` to maintain `[0, 1]` constraints. The surrogate models utilized a two-tier structure:
*   **Base Regressor:** Determined via a 5-fold cross-validation tournament.
*   **Residual Model:** A standard `Gaussian Process` fitted strictly to the out-of-fold errors.

Target outputs were stabilized using a **sign-preserving logarithmic transformation** to prevent covariance matrix conditioning failures, which was especially critical for functions spanning massive orders of magnitude. 

## Weekly Iteration and Learning
New data points rapidly confirmed that mid-to-high dimensional spaces were better treated as structured geometric shapes rather than chaotic black boxes. Early **boundary probes** (evaluating the extreme edges and corners of the hypercube) proved to be the most informative queries, as they quickly established global gradients and prevented monotonic assumptions drawn from narrow sub-regions.

To ensure the solver did not get trapped in high-dimensional saddle points, analytical optimizations utilized **multi-start random restarts**. In hindsight, systematically auditing every function with multi-degree polynomial hypothesis testing by Week 3 would have allowed the highly structured functions to be solved analytically much earlier in the campaign.

## Performance and Results
The dynamic architectural routing achieved excellent outcomes across the portfolio, most notably in higher dimensions. 
*   **Function 8 (8D):** Achieved **1st place** with a perfect score of `1` by diagnosing a second-degree polynomial and using `L-BFGS-B` to extract the exact analytical maximum.
*   **Function 6 (5D):** Secured **2nd place** using Support Vector Regression combined with Kriging.
*   **Function 5 (4D):** Reached the **Top 5%** using polynomial trend estimators.

Confidence in these high-dimensional peaks is extremely high, as the cross-validation MSE of the selected surrogate models was practically zero, and deterministic solvers found the exact mathematical peaks of those verified surfaces. 

## Ethical, Practical, and General Considerations
This optimization task perfectly mirrors industrial scenarios like automated hyperparameter tuning for Large Language Models (LLMs), expensive drug compound testing, and chemical yield maximization, where evaluating every combination is financially impossible. The **Ensemble Regression-Kriging** strategy scales exceptionally well to these real-world problems by pairing an aggressive global trend estimator with a localized GP to extract maximum information from minimal queries.

However, a limitation of these synthetic functions is that real-world deep learning loss landscapes or biological systems can exhibit massive flat plateaus and steep cliffs that defy simple polynomial modeling. The primary risk for a future user analyzing this framework is **over-engineering**—applying complex models like Deep Neural Networks to small, sparse sample budgets will result in severe overfitting, completely missing the underlying simplicity of the actual landscape.
