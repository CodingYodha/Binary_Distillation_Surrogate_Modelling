import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from utils import load_data, save_sklearn_model, save_tf_model

def train_all_models():
    """
    Loads data, trains Polynomial, XGBoost, and ANN models for both xD and QR,
    and saves the models and scalers.
    """
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_data()

    y_train_xD = y_train[['xD']]
    y_train_QR = y_train[['QR']]

    # --- Scalers ---
    print("Fitting and saving scalers...")
    scaler_X = StandardScaler().fit(X_train)
    scaler_y_xD = StandardScaler().fit(y_train_xD)
    scaler_y_QR = StandardScaler().fit(y_train_QR)

    save_sklearn_model(scaler_X, 'models/scaler_X.joblib')
    save_sklearn_model(scaler_y_xD, 'models/scaler_y_xD.joblib')
    save_sklearn_model(scaler_y_QR, 'models/scaler_y_QR.joblib')

    X_train_sc = scaler_X.transform(X_train)
    y_train_xD_sc = scaler_y_xD.transform(y_train_xD)
    y_train_QR_sc = scaler_y_QR.transform(y_train_QR)

    # --- 1. Polynomial Regression ---
    print("\nTraining Polynomial Regression models...")
    poly_model_xD = Pipeline([
        ("poly", PolynomialFeatures(degree=6, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    poly_model_xD.fit(X_train, y_train_xD)
    save_sklearn_model(poly_model_xD, 'models/polynomial_model_xD.joblib')

    poly_model_QR = Pipeline([
        ("poly", PolynomialFeatures(degree=8, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    poly_model_QR.fit(X_train, y_train_QR)
    save_sklearn_model(poly_model_QR, 'models/polynomial_model_QR.joblib')

    # --- 2. XGBoost ---
    print("\nTraining XGBoost models...")
    xgb_model_xD = XGBRegressor(max_depth=7, eta=0.7, objective='reg:squarederror', n_estimators=300)
    xgb_model_xD.fit(X_train, y_train_xD)
    save_sklearn_model(xgb_model_xD, 'models/xgboost_model_xD.joblib')
    
    xgb_model_QR = XGBRegressor(max_depth=10, eta=0.7, objective='reg:squarederror', n_estimators=300)
    xgb_model_QR.fit(X_train, y_train_QR)
    save_sklearn_model(xgb_model_QR, 'models/xgboost_model_QR.joblib')

    # --- 3. Artificial Neural Network (ANN) ---
    print("\nTraining ANN models...")
    # Model for xD
    ann_model_xD = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model_xD.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    ann_model_xD.fit(X_train_sc, y_train_xD_sc, epochs=300, batch_size=8, verbose=0)
    save_tf_model(ann_model_xD, 'models/ann_model_xD.h5')

    # Model for QR
    ann_model_QR = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model_QR.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    ann_model_QR.fit(X_train_sc, y_train_QR_sc, epochs=300, batch_size=8, verbose=0)
    save_tf_model(ann_model_QR, 'models/ann_model_QR.h5')

    print("\nAll models trained and saved successfully.")

if __name__ == '__main__':
    train_all_models()