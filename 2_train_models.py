# --- START OF FILE 2_train_models.py ---

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from utils import (
    load_data_block_split, 
    load_data_gap_split,
    save_sklearn_model, 
    save_tf_model
)

def train_and_tune_models():
    """
    Loads data with engineered features using a block split, tunes hyperparameters 
    using Bayesian Optimization, trains the final models, and saves them. 
    Also trains models for the generalization test.
    """
    print("--- Starting Model Training and Tuning on Featured Data ---")

    # --- 1. Standard Training using Block Split ---
    print("\n--- Phase 1: Standard Model Training ---")
    X_train, X_test, y_train, y_test = load_data_block_split()
    
    # Exit if data loading failed
    if X_train is None:
        print("Halting training due to data loading error.")
        sys.exit(1)
        
    print(f"Training with {X_train.shape[1]} features: {X_train.columns.tolist()}")

    # Separate targets
    y_train_xD = y_train[['xD']]
    y_train_QR = y_train[['QR']]

    # Fit and save scalers
    scaler_X = StandardScaler().fit(X_train)
    scaler_y_xD = StandardScaler().fit(y_train_xD)
    scaler_y_QR = StandardScaler().fit(y_train_QR)
    save_sklearn_model(scaler_X, 'models/scaler_X.joblib')
    save_sklearn_model(scaler_y_xD, 'models/scaler_y_xD.joblib')
    save_sklearn_model(scaler_y_QR, 'models/scaler_y_QR.joblib')

    # Scale data for ANN
    X_train_sc = scaler_X.transform(X_train)
    y_train_xD_sc = scaler_y_xD.transform(y_train_xD)
    y_train_QR_sc = scaler_y_QR.transform(y_train_QR)

    # --- Polynomial Regression Tuning & Training ---
    print("\nTuning Polynomial Regression...")
    poly_pipeline = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("linreg", LinearRegression())
    ])
    # Note: High-degree polynomials can become very slow and unstable with more features.
    # A smaller degree range is advisable.
    param_space_poly = {'poly__degree': Integer(2, 4)} 
    
    # Tune for xD
    bayes_search_poly_xD = BayesSearchCV(poly_pipeline, param_space_poly, n_iter=10, cv=5, n_jobs=-1, random_state=42)
    bayes_search_poly_xD.fit(X_train, y_train_xD)
    print(f"Best Poly xD params: {bayes_search_poly_xD.best_params_}")
    save_sklearn_model(bayes_search_poly_xD.best_estimator_, 'models/polynomial_model_xD.joblib')
    
    # Tune for QR
    bayes_search_poly_QR = BayesSearchCV(poly_pipeline, param_space_poly, n_iter=10, cv=5, n_jobs=-1, random_state=42)
    bayes_search_poly_QR.fit(X_train, y_train_QR)
    print(f"Best Poly QR params: {bayes_search_poly_QR.best_params_}")
    save_sklearn_model(bayes_search_poly_QR.best_estimator_, 'models/polynomial_model_QR.joblib')

    # --- XGBoost Tuning & Training ---
    print("\nTuning XGBoost...")
    param_space_xgb = {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'gamma': Real(0, 5)
    }
    
    # Tune for xD
    bayes_search_xgb_xD = BayesSearchCV(XGBRegressor(random_state=42), param_space_xgb, n_iter=30, cv=5, n_jobs=-1, random_state=42)
    bayes_search_xgb_xD.fit(X_train, y_train_xD.values.ravel()) # .ravel() for XGBoost
    print(f"Best XGBoost xD params: {bayes_search_xgb_xD.best_params_}")
    save_sklearn_model(bayes_search_xgb_xD.best_estimator_, 'models/xgboost_model_xD.joblib')

    # Tune for QR
    bayes_search_xgb_QR = BayesSearchCV(XGBRegressor(random_state=42), param_space_xgb, n_iter=30, cv=5, n_jobs=-1, random_state=42)
    bayes_search_xgb_QR.fit(X_train, y_train_QR.values.ravel()) # .ravel() for XGBoost
    print(f"Best XGBoost QR params: {bayes_search_xgb_QR.best_params_}")
    save_sklearn_model(bayes_search_xgb_QR.best_estimator_, 'models/xgboost_model_QR.joblib')

    # --- ANN Training (fixed architecture as tuning is more complex) ---
    print("\nTraining ANNs...")
    # Model for xD
    ann_model_xD = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model_xD.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    ann_model_xD.fit(X_train_sc, y_train_xD_sc, epochs=300, batch_size=16, verbose=0, validation_split=0.1)
    save_tf_model(ann_model_xD, 'models/ann_model_xD.h5')

    # Model for QR
    ann_model_QR = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model_QR.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    ann_model_QR.fit(X_train_sc, y_train_QR_sc, epochs=300, batch_size=16, verbose=0, validation_split=0.1)
    save_tf_model(ann_model_QR, 'models/ann_model_QR.h5')
    
    print("\n--- Standard model training complete. ---")

    # --- 2. Generalization Test Training using Gap Split ---
    print("\n--- Phase 2: Generalization (Gap) Model Training ---")
    X_train_gap, _, y_train_gap, _ = load_data_gap_split()
    
    if X_train_gap is None:
        print("Halting generalization training due to data loading error.")
        sys.exit(1)
        
    y_train_xD_gap = y_train_gap[['xD']]
    y_train_QR_gap = y_train_gap[['QR']]

    # Re-train best models on gapped data
    print("Re-training models on gapped data...")
    # Poly
    best_poly_xD = bayes_search_poly_xD.best_estimator_
    best_poly_xD.fit(X_train_gap, y_train_xD_gap)
    save_sklearn_model(best_poly_xD, 'models/polynomial_model_xD_gap.joblib')
    
    # XGBoost
    best_xgb_xD = bayes_search_xgb_xD.best_estimator_
    best_xgb_xD.fit(X_train_gap, y_train_xD_gap.values.ravel())
    save_sklearn_model(best_xgb_xD, 'models/xgboost_model_xD_gap.joblib')

    # ANN (requires re-scaling)
    scaler_X_gap = StandardScaler().fit(X_train_gap)
    scaler_y_xD_gap = StandardScaler().fit(y_train_xD_gap)
    X_train_gap_sc = scaler_X_gap.transform(X_train_gap)
    y_train_xD_gap_sc = scaler_y_xD_gap.transform(y_train_xD_gap)
    
    ann_model_xD_gap = Sequential([
        Input(shape=(X_train_gap.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model_xD_gap.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    ann_model_xD_gap.fit(X_train_gap_sc, y_train_xD_gap_sc, epochs=300, batch_size=16, verbose=0)
    save_tf_model(ann_model_xD_gap, 'models/ann_model_xD_gap.h5')
    # Save the specific scalers for this test
    save_sklearn_model(scaler_X_gap, 'models/scaler_X_gap.joblib')
    save_sklearn_model(scaler_y_xD_gap, 'models/scaler_y_xD_gap.joblib')
    
    print("\n--- All training phases complete. ---")

if __name__ == '__main__':
    train_and_tune_models()

# --- END OF FILE 2_train_models.py ---