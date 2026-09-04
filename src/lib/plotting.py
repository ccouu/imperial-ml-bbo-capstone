import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_xy(x_raw, y_raw):
    """
    Loads separate .npy files for X and Y data, pairs them to plot
    their relationship, and saves the resulting figure to disk.
    """
    try:
        # 2. Clean dimensions by removing trailing dummy axes (e.g., (N, 1) -> (N,))
        x = x_raw.squeeze()
        y = y_raw.squeeze()

        df = pd.DataFrame(x, columns=[f"Feature_{i+1}" for i in range(x.shape[1])])
        df["Target"] = y
                # 4. Initialize visual configuration
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        g = sns.pairplot(df, hue='Target', palette='bright', diag_kind='kde')
        for ax in g.axes.flatten():
            if ax is not None:  # Guard against empty panels in non-square or upper-triangle grids
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        plt.show()

    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")
    finally:
        plt.close()

def plot_linear(x_train, y_train):
    model_1st = LinearRegression()
    model_1st.fit(x_train, y_train)
    pred_1st = model_1st.predict(x_train)
    residuals_1st = y_train - pred_1st
    # Use negative MAPE so scikit-learn maximizes it natively
    scores = cross_val_score(model_1st, x_train, y_train, cv=LeaveOneOut(), scoring='neg_mean_absolute_percentage_error')
    true_mape = -np.mean(scores)
    print(f"1st degree error: {true_mape}")


    # 2. Fit 2nd-Degree Model & Calculate Residuals
    model_2nd = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('reg', LinearRegression())
    ])
    model_2nd.fit(x_train, y_train)
    pred_2nd = model_2nd.predict(x_train)
    residuals_2nd = y_train - pred_2nd
    scores = cross_val_score(model_2nd, x_train, y_train, cv=LeaveOneOut(), scoring='neg_mean_absolute_percentage_error')
    true_mape = -np.mean(scores)
    print(f"2st degree error: {true_mape}")

    # 3. Plot them side-by-side for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1st-Degree Plot (Actual vs Residual)
    ax1.scatter(y_train, residuals_1st, color='blue', s=120, edgecolors='black', zorder=3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_title('1st-Degree: Actual vs Residual')
    ax1.set_xlabel('Actual Target Value (y_train)')
    ax1.set_ylabel('Residual Error')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2nd-Degree Plot (Actual vs Residual)
    ax2.scatter(y_train, residuals_2nd, color='purple', s=120, edgecolors='black', zorder=3)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('2nd-Degree: Actual vs Residual')
    ax2.set_xlabel('Actual Target Value (y_train)')
    ax2.set_ylabel('Residual Error')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_performance_gain(historical_max: float, weekly_new_data: list, historical_std: float = None, use_zscore: bool = False, question: str= ""):
    """
    Plots the cumulative performance gain of weekly submissions relative to Week 0.

    Parameters:
    -----------
    historical_max : float
        The baseline highest output (Week 0).
    weekly_new_data : list of float
        The continuous outputs obtained from Week 1 to Week N.
    historical_std : float, optional
        The standard deviation of the initial data pool (needed if use_zscore=True).
    use_zscore : bool, default False
        If True, plots gains scaled by Standard Deviation. If False, plots % Gain.
    """
    weekly_new_data = list(weekly_new_data)

    # Calculate gains relative to the historical max baseline
    if use_zscore:
        if not historical_std or historical_std == 0:
            raise ValueError("Must provide a non-zero historical_std to compute Z-score gains.")
        # Your Formula: (Y_new - Y_baseline) / STD
        gains = [(y - historical_max) / historical_std for y in weekly_new_data]
        y_label = "Performance Gain (Z-Score Scale / $\sigma$)"
        title_metric = "Z-Score Scale"
        baseline_val = 0.0
    else:
        # Percentage Gain Formula: ((Y_new - Y_baseline) / Y_baseline) * 100
        gains = [((y - historical_max) / abs(historical_max)) * 100 for y in weekly_new_data]
        y_label = "Performance Gain (% Improvement)"
        title_metric = f"Percentage Improvement Scale - Q{question}"
        baseline_val = 0.0

    # Week 0 represents the baseline, where gain is exactly 0

    #gains = np.clip(gains, a_min=0, a_max=max(gains))
    y_values = [baseline_val] + gains
    x_values = list(range(len(y_values)))

    # Plot performance trajectory
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='#2ca02c', linewidth=2.5, label='Model Gain')

    # Draw the fixed baseline anchor at 0
    plt.axhline(y=baseline_val, color='black', linestyle='-', alpha=0.4, label='Historical Baseline (Week 0)')

    # Highlight the breakthrough zone
    plt.fill_between(x_values, y_values, baseline_val, where=(np.array(y_values) >= baseline_val),
                     color='green', alpha=0.1, label='Breakthrough Zone')


    # Layout and Text
    plt.xlabel('Optimization Timeline (Weeks Passed)', fontsize=11, labelpad=10)
    plt.ylabel(y_label, fontsize=11, labelpad=10)
    plt.title(f'Cumulative Performance Gain ({title_metric})', fontsize=13, pad=15, fontweight='bold')

    plt.xticks(x_values)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True)
    plt.savefig(title_metric, bbox_inches='tight', dpi=150)
    plt.show()

def plot_model_function_shape( x, y, predictions, resolution = 500, feature_to_audit=0):
    linear_x = np.linspace(0.0, 1.0, resolution)

    # 5. Plot the completely clean, razor-sharp line
    plt.figure(figsize=(8, 4))
    plt.plot(linear_x, predictions, label="Model Function Shape", color='blue', lw=2)
    plt.scatter(x[:, feature_to_audit], y, color='red', label="Historical Datapoints", zorder=5)

    plt.xlabel(f"Feature {feature_to_audit} Value")
    plt.ylabel("Raw Predicted Y")
    plt.title(f"Pure 1D Landscape Slice across Feature {feature_to_audit}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()   