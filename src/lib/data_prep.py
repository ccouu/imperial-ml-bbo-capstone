import numpy as np
import os
import pandas as pd
import re

def load_xy(x_file, y_file):
    if not os.path.exists(x_file) or not os.path.exists(y_file):
        raise FileNotFoundError("One or both of the input data paths do not exist.")
    x_raw = np.load(x_file)
    y_raw = np.load(y_file)
    return x_raw, y_raw

def load_new_xy(x_file, y_file):
    if not os.path.exists(x_file) or not os.path.exists(y_file):
        raise FileNotFoundError("One or both of the input data paths do not exist.")
        
    # 1. Read the input and output text logs
    with open(x_file, "r", encoding="utf-8") as file:
        x_raw_text = file.read()    
    with open(y_file, "r", encoding="utf-8") as file:
        y_raw_text = file.read()    

    # 2. Parse Inputs (Ragged arrays per row block)
    x_pattern = r"array\(\[(.*?)\]\)"
    weekly_input = []
    # Split the file into individual lines to isolate each submission row
    for line in x_raw_text.strip().split('\n'):
        if line.strip():
            # Find all numbers within np.float64(...) for THIS line only
            line_numbers = re.findall(x_pattern, line)
            if line_numbers:
                line_arrays = []
                for line_string in line_numbers:
                    numbers = [float(num) for num in line_string.split(',') if num.strip()]
                    line_arrays.append(numbers)
                weekly_input.append(line_arrays)

    # 3. FIX: Parse Outputs line-by-line to preserve the 2D [idx][q] structure
    #y_pattern = r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)"
    y_pattern = r"(?:np\.float64\()?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\)?)?"
    weekly_output = []
    
    # Split the file into individual lines to isolate each submission row
    for line in y_raw_text.strip().split('\n'):
        if line.strip():
            # Find all numbers within np.float64(...) for THIS line only
            line_numbers = re.findall(y_pattern, line)
            if line_numbers:
                # Append this line as a separate row of floats
                weekly_output.append([float(num) for num in line_numbers])

    # Registries setup
    input_registry = {}
    output_registry = {}
    input_registry_dict = {}
    output_registry_dict = {}
    
    # 4. Reconstruct and sort data by Question ID
    for idx, whole_data in enumerate(weekly_input):
        # Determine which question index this array maps to based on its position in the log block
        for q, weekly_data in enumerate(whole_data):
            # Dynamic grouping (e.g., array 0 goes to Q1, array 1 goes to Q2)
            input_registry_dict.setdefault(f"Q{q + 1}", []).append(weekly_data)
            
            # This now successfully indexes into [row_index][column_index] without crashing!
            output_registry_dict.setdefault(f"Q{q + 1}", []).append(weekly_output[idx][q])

    # Convert lists to solid NumPy matrices for your modeling steps
    for question, score in input_registry_dict.items():
        input_registry[question] = np.array(score)
        output_registry[question] = np.array(output_registry_dict[question])
        
    return input_registry, output_registry


def transform_targets(y):
    y = np.clip(y, a_min=1e-80, a_max=None)
    log_y = np.log10(y)
    return log_y

def generate_x_grid(dimensions, m=100):
    # 1. Create the 1D linear ticks from 0 to 1
    axis_space = np.linspace(0.0, 1.0, num=m)

    # 2. Duplicate the 1D space for the specified number of dimensions
    spaces = [axis_space] * dimensions
    # 3. Generate the dense coordinate meshes
    # 'ij' matrix indexing guarantees that column order matches dimension order
    meshes = np.meshgrid(*spaces, indexing='ij')

    # 4. Flatten each mesh grid using .ravel() for speed and stack them column-wise
    # .T transposes the stacked rows into a clean (N, dimensions) shape
    X_grid = np.vstack([mesh.ravel() for mesh in meshes]).T

    return X_grid

def remove_outliers(X, y, method="IQR", THRESHOLD = 2.5):
    if method == "IQR":
        # 1. Calculate the 25th (Q1) and 75th (Q3) percentiles of your targets
        q25, q75 = np.percentile(y, 25), np.percentile(y, 75)
        iqr = q75 - q25

        # 2. Define the fence multiplier
        # 1.5 is standard (Tukey's Fences), 3.0 detects only extreme anomalies
        multiplier = 2.5
        lower_bound = q25 - (multiplier * iqr)
        upper_bound = q75 + (multiplier * iqr)
        # 3. Identify coordinates that fall inside the robust boundaries
        is_safe_index = (y >= lower_bound) & (y <= upper_bound)

        # 4. Filter your matrices for your weekly optimization script
        X_clean = X[is_safe_index]
        y_clean = y[is_safe_index]
    elif method == "std":
        mean_y = np.mean(y)
        std_y = np.std(y)

        # Calculate individual Z-scores for every historical target row
        z_scores = (y - mean_y) / std_y

        # 4. Create a boolean mask of rows that sit WITHIN the confidence bound
        is_within_confidence_bound = np.abs(z_scores) < THRESHOLD
        # 5. Filter the s cleanly
        X_clean = X[is_within_confidence_bound]
        y_clean = y[is_within_confidence_bound]
    elif method == "mad":
        median_y = np.median(y)
        
        # Calculate absolute deviations from the median
        abs_deviation = np.abs(y - median_y)
        
        # MAD (scaled by 1.4826 to match standard normal distribution properties)
        mad = 1.4826 * np.median(abs_deviation)
        
        # Calculate modified Z-scores
        # If mad is 0 (all points identical), handle division by zero safely
        if mad == 0:
            return X, y
            
        modified_z_scores = (y - median_y) / mad
        
        # Keep only points within the threshold (typically 2.5 to 3.5)
        negative_mask = modified_z_scores > -THRESHOLD
        positive_mask = modified_z_scores < 100*THRESHOLD
        clean_mask = negative_mask 
        
        print(f"Dropped {len(y) - np.sum(clean_mask)} outlier(s) using robust MAD.") 
        X_clean = X[clean_mask]
        y_clean = y[clean_mask]          
    return X_clean, y_clean
