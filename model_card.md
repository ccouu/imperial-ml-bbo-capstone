# Model Card

## Model Description

**Input:** 
The model accepts an `N x D` array of continuous, numerical features, where `D` ranges from `2` to `8` dimensions depending on the specific optimization task. All input coordinates represent physical or categorical hyperparameters that are strictly bounded within a defined search space (conceptually a `[0, 1]` unit hypercube).

**Output:** 
The model outputs a `1D` array containing a continuous scalar value representing the objective score to be maximized. Depending on the task, this represents metrics such as chemical yield, model validation accuracy, or the negative value of adverse side effects.

**Model Architecture:** 
The system utilizes a two-tier **Ensemble Regression-Kriging (ERK)** architecture, combined with a dynamic **Upper Confidence Bound (UCB)** acquisition strategy:
1.  **Tier 1 (Global Trend Estimator):** A competitive 5-fold cross-validation tournament evaluates multiple parametric models (`Support Vector Regressors`, `XGBoost`, `Polynomials`, `Linear Models`). The model with the lowest Mean Squared Error (MSE) is selected to capture the macro-level slopes of the landscape.
2.  **Tier 2 (Spatial Error Corrector):** A standard `Gaussian Process` (typically using an anisotropic `Matérn-5/2` kernel) is fitted strictly to the out-of-fold residual errors of the winning base regressor.
3.  **Fallback Gate:** To protect against over-parameterization on simple landscapes, if the ensemble's cross-validation `MSE` exceeds `0.5`, the pipeline automatically bypasses the base regressor and reverts to a standard Gaussian Process.
4.  **Acquisition / Solver:** The pipeline uses a decaying `UCB` exploration factor (`kappa`). For landscapes that are diagnosed as mathematically structured (e.g., highly concave polynomials), the probabilistic UCB sampling is bypassed entirely in favor of a bounded `L-BFGS-B` deterministic gradient solver to find the analytical peak.

## Performance

Performance was evaluated against a synthetic benchmark suite of 8 hidden black-box functions (`F1` to `F8`) spanning 2D to 8D parameter spaces. The primary metric is the normalized leaderboard score achieved on a strictly limited query budget. Below are the absolute peak scores recorded across the 12-week evaluation period.

| Function | Dimension | Winning Architecture | Benchmark Standing | Peak Score |
| -------- | --------- | -------------------- | ------------------ | ---------- |
| **F1**   | `2D`      | `XGBoost` + `Residual GP` | Competitive (Avg)  | `2.96e-08` |
| **F2**   | `2D`      | `XGBoost` + `Residual GP` | Average (High Noise) | `0.5840`   |
| **F3**   | `3D`      | `SVR` + `Residual GP`     | **Top 15%**        | `-0.0051`  |
| **F4**   | `4D`      | `Polynomial` + `Residual GP` | **Top 25%**        | `0.6402`   |
| **F5**   | `4D`      | `Polynomial` + `Residual GP` | **Top 5%**         | `8662.4825`|
| **F6**   | `5D`      | `SVR` + `Residual GP`     | **2nd Place**      | `-0.0786`  |
| **F7**   | `6D`      | `SVR` + `Residual GP`     | Competitive (Avg)  | `2.2608`   |
| **F8**   | `8D`      | `2nd-Degree Polynomial` + `L-BFGS-B` | **1st Place** | `10.0`     |

*Note: The architecture dominates in mid-to-high dimensional, sparse domains (F5, F6, F8) where capturing global parametric trends circumvents the curse of dimensionality.*

## Limitations

*   **Tree-Based Partition Bias:** When `XGBoost` or Decision Trees win the base regressor tournament in low-dimensional, continuous manifolds (like `2D`), they introduce an artificial step-wise blockiness. This partition bias can slightly degrade the smooth interpolation capabilities of the residual Gaussian Process.
*   **Capacity Restrictions:** Deep Neural Networks and highly complex non-linear estimators suffer from severe overfitting when constrained to small sample budgets (`N < 50`). Consequently, the architecture intentionally limits expressiveness by pruning deep learning models in favor of simpler, regularized baselines like SVR and low-degree polynomials.
*   **Solver Constraints:** The `L-BFGS-B` analytical solver cannot be utilized if the chosen base regressor is non-differentiable (e.g., XGBoost). In these cases, the system must rely on stochastic heuristic search over dense grids.

## Trade-offs

*   **Computation Time vs. Query Efficiency:** Executing a 5-fold cross-validation tournament across multiple model families at every single step is computationally expensive. We trade rapid execution time for highly efficient, information-dense queries—which is the correct decision when evaluating the actual black-box is assumed to be financially or temporally expensive.
*   **Exploration Penalty:** The system deliberately sacrifices short-term leaderboard gains in the early rounds by maintaining a high exploration factor (`kappa = 2.0 to 3.0`). This trades immediate exploitation for variance reduction, ensuring the model builds an accurate global map required for high-confidence analytical exploitation in the final rounds.
*   **Global Rigidity vs. Local Flexibility:** Using a global trend regressor assumes the macro-landscape behaves uniformly. If a landscape contains completely disjointed regimes, a parametric base model could distort the spatial modeling of the residual Gaussian Process.
