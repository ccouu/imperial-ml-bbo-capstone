## 📊 Ensemble Regression-Kriging for Black-Box Optimization
### 1. NON-TECHNICAL EXPLANATION OF YOUR PROJECT

Imagine trying to find the highest peak in a vast, foggy mountain range, but every step you take costs a fortune. This project builds an "intelligent digital prospector" to solve this problem. Instead of wandering randomly, the system creates a self-updating 3D elevation map of the terrain based on past steps. It mathematically calculates where the highest peaks are most likely hiding, systematically balancing whether to mine a proven gold vein or explore unmapped territory. In the real world, this exact approach is used to design drugs, tune AI models, and engineer aerospace components without wasting millions on trial-and-error testing.

### 2. DATA

The dataset consists of eight continuous, unobservable synthetic benchmark functions (**F1 to F8**) spanning **2D to 8D** parameter spaces. This data was provided as part of the Black-Box Optimization Capstone Challenge for the Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence. The true mathematical formulations of the functions were hidden, and the system was strictly limited to a small budget of noisy evaluation queries to simulate highly expensive real-world industrial testing.

### 3. MODEL

The project utilizes a **hybrid Ensemble Regression-Kriging with Upper Confidence Bound (ERK-UCB)** architecture.

I chose this because standard Gaussian Processes (GPs) suffer severely from the curse of dimensionality; in sparse **6D to 8D** spaces, distance-based kernels become isotropic and uninformative. To solve this, my model runs a competitive **5-fold cross-validation** tournament across parametric global trend estimators (**Support Vector Regressors, XGBoost, and Polynomials**) to capture macro-level slopes. A Gaussian Process is then fitted strictly to the out-of-fold residual errors. To protect dense 2D manifolds from over-parameterized model bias, I engineered an automated fallback gate that reverts to a standard GP if the ensemble's cross-validation error exceeds a critical threshold (MSE > *0.5*).

### 4. HYPERPARAMETER OPTIMISATION

Target Scaling: Raw targets spanned values from thousands down to extreme scientific notation (10^-160). I optimized the data space itself using a StandardScalar or yeo-johnson.

Gaussian Process Tuning: GP hyperparameters, specifically the Matérn-5/2 kernel lengthscales and the noise floor (WhiteKernel), were optimized using multi-start L-BFGS-B to maximize the log-marginal likelihood, preventing the optimizer from getting trapped in local minima.

Exploration vs. Exploitation (kappa): The Upper Confidence Bound exploration factor (kappa) was scheduled dynamically. It was initialized high (**2.0 to 3.0**) in early rounds to map the hypercube boundaries and epistemic variance, and systematically decayed toward **0.0** in final rounds to force aggressive local peak exploitation.

### 5. RESULTS

The framework demonstrated that inductive bias and continuous kernel regularization dominate mid-to-high dimensional spaces.
| Function | Dimension | Dominant Architecture | Benchmark Standing |
| -------- | --------- | --------------------- | ------------------ |
| **F1** | `2D` | `XGBoost` + `Residual GP` | Competitive (Avg) |
| **F2** | `2D` | `XGBoost` + `Residual GP` | Average (High Noise) |
| **F3** | `3D` | `Support Vector Regression` + `Residual GP` | **Top 15%** |
| **F4** | `4D` | `Polynomial Trend` + `Residual GP` | **Top 25%** |
| **F5** | `4D` | `Polynomial Trend` + `Residual GP` | **Top 5%** |
| **F6** | `5D` | `Support Vector Regression` + `Residual GP` | **2nd Place** |
| **F7** | `6D` | `Support Vector Regression` + `Residual GP` | Competitive (Avg) |
| **F8** | `8D` | `2nd-Degree Polynomial` + `L-BFGS-B Solver` | **1st Place (~0.997 Score)** |

### 6. REPOSITORY MAP

```text
├── README.md                      ← you are here
│
├── data/                          🗄️ Optimization datasets
│   ├── initial_data/              seed data provided at the start of the challenge
│   └── weekly_data/               iterative weekly feedback and newly sampled points
│
└── src/                           ⚙️ Core pipeline and execution
    │
    ├── capstone.ipynb             ⭐ main execution flow, strategy logic, and weekly optimization loop
    │
    └── lib/                       🛠️ Custom utility modules
        ├── data_prep.py           data loading, bounds enforcement, and log-transformations
        ├── metrics.py             cross-validation MSE, UCB/EI acquisition, and scoring routines
        ├── model.py               Ensemble Regression-Kriging, fallback gates, and L-BFGS-B solver
        └── plotting.py            visualization tools for surrogate landscapes and convergence tracking
```


### CONTACT DETAILS
Hi, I'm Charles. I am a full-stack software engineer with over a decade of experience, actively transitioning into machine learning, data engineering, and MLOps. If you want to discuss Bayesian optimization, active learning, or ML infrastructure, feel free to reach out.

GitHub: @ccouu
LinkedIn: https://www.linkedin.com/in/o-u-chan-7b199b3b/
