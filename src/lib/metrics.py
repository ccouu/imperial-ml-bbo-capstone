import numpy as np
import pandas as pd

from scipy.stats import norm

def expected_improvement(X_grid, y_max, default_mean=None, default_std=None):
    """
    Computes the Expected Improvement (EI) over X_grid for maximization.
    Preserves true statistical scales for accurate exploration/exploitation balances.
    """
    mean = default_mean
    # Keep the actual scale of std, do NOT min-max normalize it
    std = default_std

    # Create a safe mask where std is effectively zero to avoid division-by-zero warnings
    # A tiny epsilon protects the division without changing the true variance scale
    safe_std = np.maximum(std, 1e-9)

    # 1. Calculate the standard score Z for maximization
    # If mean > y_max, Z is positive (exploitation choice)
    # If std is large, it dampens Z but enhances the right-hand PDF term (exploration choice)
    Z = (mean - y_max) / safe_std
    
    # 2. Standard textbook closed-form Expected Improvement equation
    ei = (mean - y_max) * norm.cdf(Z) + safe_std * norm.pdf(Z)

    # 3. Clean boundary protection: force areas with zero true uncertainty to zero EI
    ei[std <= 1e-9] = 0.0
    
    return ei

def ucb(post_mean, post_std, beta=1.69):
    acquisition_function = post_mean + beta * post_std
    return acquisition_function

def mape(y, y_pred):
    return  np.sum(np.abs((y-y_pred)/y)) / len(y)

def process_and_filter_top_m(X, Y, new_X, new_Y, M):
    """
    Concatenates original and new data arrays, appends them horizontally,
    sorts the unified table by Y, adds a source origin label, and returns the top M rows.
    """
    # 1. Combine X and Y horizontally for both datasets
    # Shape of current_data: (n, m + 1)
    current_data = np.hstack((X, Y.reshape(-1,1)))
    # Shape of incoming_data: (n2, m + 1)
    incoming_data = np.hstack((new_X, new_Y.reshape(-1,1)))

    # 2. Convert to DataFrames and add the tracking source label
    m_features = X.shape[1]
    column_names = [f"feature_{i}" for i in range(m_features)] + ["target_Y"]

    df_current = pd.DataFrame(current_data, columns=column_names)
    df_current["source_label"] = "original"

    df_incoming = pd.DataFrame(incoming_data, columns=column_names)
    df_incoming["source_label"] = "new"

    # 3. Vertically stack both DataFrames together
    # Shape of unified_df: (n + n2, m + 2)
    unified_df = pd.concat([df_current, df_incoming], axis=0, ignore_index=True)

    # 4. Sort by Y and extract the top M records
    # Change ascending=False if you want the highest values of Y at the top
    sorted_df = unified_df.sort_values(by="target_Y", ascending=False)

    return sorted_df.head(M)

