## 📊 Dataset Composition
### 1. Motivation

This dataset tracks sequential candidate queries and outputs across eight unknown continuous synthetic objective functions (F1 to F8). Created by the student for the Imperial College London ML/AI capstone, it simulates expensive industrial optimization scenarios (e.g., hyperparameter tuning, lab synthesis) to evaluate surrogate modeling, Bayesian acquisition strategies, and LLM in-context optimization.

### 2. Composition

The dataset contains 13 sequential evaluations per function across 8 benchmark functions (114 total instances). Dimensionality varies by function, ranging from 2D to 8D continuous input vectors, with all input coordinates bounded strictly between 0.0 and 1.0. Outputs are continuous scalar values whose scale varies drastically across functions, spanning negative numbers, near-zero values in scientific notation, and positive values into the thousands. Spatial coverage exhibits sampling gaps in unmapped sub-regions due to late-stage exploitation around identified global peaks.

### 3. Collection Process

Data was gathered iteratively across 10 module rounds. Initial Round 0 provided baseline datasets varying from 10 to 40 data points across different functions to seed initial landscape mapping. Subsequent queries evolved using a competitive 5-Fold Cross-Validation surrogate tournament (SVR, XGBoost, Neural Networks, Linear Models) combined with a Gaussian Process trained on out-of-fold residual errors (Regression-Kriging). Decisions were guided by an Upper Confidence Bound (UCB) acquisition function with a decaying exploration parameter (kappa). Candidates were evaluated by the course benchmark engine to retrieve ground-truth target values.

### 4. Preprocessing and Uses

Some targets were transformed using a sign-preserving log-transform (y_trans = -sign(y) * log10(|y| + epsilon)) to prevent matrix inversion collapse on extreme scales, alongside MinMax feature scaling. Raw Cartesian coordinates were preserved to avoid introducing manual inductive bias. Intended for sparse-sample BO benchmarking and academic evaluation; inappropriate for high-dimensional (greater than 10D) or real-world physical deployment.

### 5. Distribution and Maintenance

Archived locally in Jupyter notebooks and submitted via the Imperial College London VLE portal under academic use terms. Maintained by the student author; underlying benchmark definitions remain proprietary to the course organizers.


---

## ⚡ Model Architecture
### 1. Overview & Intended Use

The Ensemble Regression-Kriging with Upper Confidence Bound (ERK-UCB) framework is designed for low-budget, continuous Black-Box Optimization (BBO) across 2D–8D search spaces bounded between 0.0 and 1.0. It is suitable for expensive physical or simulation tuning where objective evaluations are constrained. It should be avoided for high-dimensional spaces (greater than 10D), non-stationary landscapes with sharp discontinuities, or ultra-fast real-time control loops.

### 2. Details & Round Evolution

Across 10 rounds, the strategy evolved from initial space-filling exploration into automated, metric-driven optimization:

Rounds 1–3 (Initialization & GP Baseline): Established spatial bounds using the initial 10–40 data points per function via stationary Gaussian Process (GP) regression.

Rounds 4–8 (Surrogate Tournament & Regression-Kriging): Implemented competitive 5-Fold Cross-Validation across Linear Models, SVR, XGBoost, and Neural Networks to fit global trends. Residual errors were modeled using a GP to capture spatial epistemic uncertainty.

Rounds 9–10 (Decayed Exploitation & Fine Tuning): Shifted acquisition focus from global exploration to aggressive local exploitation by decaying the UCB exploration factor (kappa = 0) on high-confidence functions.

### 3. Performance

Evaluated across functions F1–F8 using Out-of-Fold (OOF) Mean Squared Error (MSE), normalized fitness score (0.00 to 1.00), and regret. High-confidence functions (F3–F6, F8) achieved normalized scores up to 0.997 with MSE under 10% of target variance, enabling pure posterior mean exploitation. Complex functions like F1 required falling back to pure GP modeling to handle extreme spatial variance.

### 4. Assumptions, Limitations & Ethics

Assumptions: Input representations are optimal without feature engineering; target distributions can be stabilized using sign-preserving log-transformations.

Limitations: High computational overhead during model tournaments; potential boundary clustering when epistemic uncertainty is extreme in sparse regions.

Ethical Considerations & Decision Transparency: Transparency in model selection and residual tracking prevents "black-box inside a black-box" opacity, ensuring academic reproducibility and safe deployment in physical domain adaptation.
