import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import save_model, load_model
import matplotlib.pyplot as plt

def load_data(path='data/processed/master_data.csv', target_cols=['xD', 'QR'], test_size=0.2, random_state=42):
    """
    Loads the master data, splits it into features (X) and targets (y),
    and performs a train-test split.
    """
    df = pd.read_csv(path)
    
    X = df[['xF', 'R', 'B']]
    y = df[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def load_data_block_split(path='data/processed/master_data.csv', target_cols=['xD', 'QR'], split_col='xF', split_val=0.8):
    """
    Loads data and performs a block-based split for testing extrapolation.
    Trains on data below split_val and tests on data above it.
    """
    df = pd.read_csv(path)
    
    train_df = df[df[split_col] < split_val]
    test_df = df[df[split_col] >= split_val]
    X_train = train_df[['xF', 'R', 'B']]
    y_train = train_df[target_cols]
    
    X_test = test_df[['xF', 'R', 'B']]
    y_test = test_df[target_cols]
    
    print(f"Block split on '{split_col} < {split_val}'. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def load_data_gap_split(path='data/processed/master_data.csv', target_cols=['xD', 'QR'], gap_col='R', gap_range=[3.5, 4.5]):
    """
    Loads data and creates a split for the generalization test, where the model
    is trained on data outside a specific range and tested on data inside it.
    """
    df = pd.read_csv(path)
    
    # The gap is the test set
    test_df = df[(df[gap_col] >= gap_range[0]) & (df[gap_col] <= gap_range[1])]
    
    # The training set is everything outside the gap
    train_df = df[(df[gap_col] < gap_range[0]) | (df[gap_col] > gap_range[1])]
    
    X_train = train_df[['xF', 'R', 'B']]
    y_train = train_df[target_cols]
    
    X_test = test_df[['xF', 'R', 'B']]
    y_test = test_df[target_cols]
    
    print(f"Gap split on '{gap_col}' in range {gap_range}. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def save_sklearn_model(model, path):
    """Saves a scikit-learn, pipeline, or XGBoost model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_sklearn_model(path):
    """Loads a scikit-learn, pipeline, or XGBoost model."""
    return joblib.load(path)

def save_tf_model(model, path):
    """Saves a TensorFlow/Keras model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(model, path)
    print(f"Model saved to {path}")

def load_tf_model(path):
    """Loads a TensorFlow/Keras model without compilation to avoid version compatibility issues."""
    return load_model(path, compile=False)

def save_plot(fig, path):
    """Saves a matplotlib figure to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(f"Plot saved to {path}")
    plt.close(fig)  # Close the figure to free up memory

def save_metrics(metrics, path):
    """Saves a dictionary of metrics to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")