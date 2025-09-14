# --- START OF FILE utils.py ---

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model, load_model
import matplotlib.pyplot as plt

# Define the feature set to be used by all models
FEATURE_COLS = [
    'xF', 'R', 'B',            # Original features
    'R_over_R_plus_1',         # Engineered features
    'R_times_xF',
    'alpha_approx',
    'R_min_approx',
    'R_factor'
]

def engineer_features(df):
    """
    Adds new, physics-informed features to a dataframe. This can be called from any script.
    """
    df_out = df.copy()
    
    # 1. Dimensionless Group for Reflux Ratio
    df_out['R_over_R_plus_1'] = df_out['R'] / (df_out['R'] + 1)
    
    # 2. Interaction Term
    df_out['R_times_xF'] = df_out['R'] * df_out['xF']
    
    # 3. Physics-Inspired Features based on Relative Volatility (alpha)
    df_out['alpha_approx'] = -5.7 * df_out['xF']**2 + 3.1 * df_out['xF'] + 3.6
    
    # 4. Minimum Reflux Ratio (approximated)
    df_out['R_min_approx'] = 1 / (df_out['alpha_approx'] - 1)
    
    # 5. The "Golden Feature": The Reflux Ratio Factor
    df_out['R_factor'] = df_out['R'] / df_out['R_min_approx']
    
    # Clean up any infinite/NaN values that might arise
    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_out.dropna(inplace=True)
    
    return df_out

def load_data_from_path(path):
    """Loads the raw processed data from a given path."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: The data file was not found at {path}")
        print("Please run the data preparation script first.")
        return None

def split_features_targets(df, target_cols=['xD', 'QR']):
    """Splits the dataframe into features (X) and targets (y)."""
    # Ensure all required feature columns are present before trying to split
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column '{col}' during split.")
    X = df[FEATURE_COLS]
    y = df[target_cols]
    return X, y

def load_data_block_split(path='data/processed/master_data_featured.csv', target_cols=['xD', 'QR'], split_col='xF', split_val=0.8):
    """
    Loads featured data and performs a block-based split for testing extrapolation.
    """
    df = load_data_from_path(path)
    if df is None: return None, None, None, None
    
    train_df = df[df[split_col] < split_val]
    test_df = df[df[split_col] >= split_val]
    
    X_train, y_train = split_features_targets(train_df, target_cols)
    X_test, y_test = split_features_targets(test_df, target_cols)
    
    print(f"Block split on '{split_col} < {split_val}'. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def load_data_gap_split(path='data/processed/master_data_featured.csv', target_cols=['xD', 'QR'], gap_col='R', gap_range=[3.5, 4.5]):
    """
    Loads featured data and creates a split for the generalization test.
    """
    df = load_data_from_path(path)
    if df is None: return None, None, None, None
    
    test_df = df[(df[gap_col] >= gap_range[0]) & (df[gap_col] <= gap_range[1])]
    train_df = df[(df[gap_col] < gap_range[0]) | (df[gap_col] > gap_range[1])]
    
    X_train, y_train = split_features_targets(train_df, target_cols)
    X_test, y_test = split_features_targets(test_df, target_cols)
    
    print(f"Gap split on '{gap_col}' in range {gap_range}. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def save_sklearn_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_sklearn_model(path):
    return joblib.load(path)

def save_tf_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(model, path)
    print(f"Model saved to {path}")

def load_tf_model(path):
    return load_model(path, compile=False)

def save_plot(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(f"Plot saved to {path}")
    plt.close(fig)

def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")

# --- END OF FILE utils.py ---