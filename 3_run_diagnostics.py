import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_data, load_sklearn_model, load_tf_model, save_plot, save_metrics

def run_all_diagnostics():
    """
    Loads trained models and scalers, evaluates them on test data,
    runs physical consistency checks, and saves plots and metrics.
    """
    print("--- Running Diagnostics ---")
    
    # --- Load Data and Models ---
    print("Loading data, models, and scalers...")
    X_train, X_test, y_train, y_test = load_data()
    
    models = {
        'poly_xD': load_sklearn_model('models/polynomial_model_xD.joblib'),
        'poly_QR': load_sklearn_model('models/polynomial_model_QR.joblib'),
        'xgb_xD': load_sklearn_model('models/xgboost_model_xD.joblib'),
        'xgb_QR': load_sklearn_model('models/xgboost_model_QR.joblib'),
        'ann_xD': load_tf_model('models/ann_model_xD.h5'),
        'ann_QR': load_tf_model('models/ann_model_QR.h5')
    }
    
    scaler_X = load_sklearn_model('models/scaler_X.joblib')
    scaler_y_xD = load_sklearn_model('models/scaler_y_xD.joblib')
    scaler_y_QR = load_sklearn_model('models/scaler_y_QR.joblib')

    X_test_sc = scaler_X.transform(X_test)
    metrics = {}

    # --- 1. Performance Metrics ---
    print("\nCalculating performance metrics...")
    for model_name, model in models.items():
        target = 'xD' if 'xD' in model_name else 'QR'
        y_true = y_test[[target]]
        
        if 'ann' in model_name:
            y_pred_sc = model.predict(X_test_sc, verbose=0)
            scaler_y = scaler_y_xD if target == 'xD' else scaler_y_QR
            y_pred = scaler_y.inverse_transform(y_pred_sc)
        else:
            y_pred = model.predict(X_test)
            
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        metrics[model_name] = {'R2_Score': r2, 'MSE': mse}
        print(f"{model_name}: R2 = {r2:.6f}, MSE = {mse:.6f}")

    save_metrics(metrics, 'results/metrics.json')

    # --- 2. EDA Plots ---
    print("\nGenerating EDA plots...")
    df = pd.read_csv('data/processed/master_data.csv')
    df_unique = df.drop_duplicates(subset=['R', 'B'])
    
    fig_heatmap, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df_pivot_xD = df_unique.pivot(index='R', columns='B', values='xD')
    df_pivot_QR = df_unique.pivot(index='R', columns='B', values='QR')
    sns.heatmap(df_pivot_xD, ax=ax1).set_title('EDA Heatmap: xD vs (R, B)')
    sns.heatmap(df_pivot_QR, ax=ax2).set_title('EDA Heatmap: QR vs (R, B)')
    save_plot(fig_heatmap, 'results/plots/eda_heatmaps.png')
    plt.close(fig_heatmap)

    # --- 3. Physical Diagnostics ---
    print("\nRunning physical consistency checks...")
    
    # a) Monotonicity Check (xD vs. R)
    print("Checking monotonicity...")
    R_range = np.linspace(0.8, 5.0, 50)
    # Fix: Ensure column order matches training data ['xF', 'R', 'B']
    df_mono = pd.DataFrame({'xF': [0.5]*50, 'R': R_range, 'B': [2.0]*50})
    
    fig_mono, axes_mono = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig_mono.suptitle('Diagnostic: Monotonicity Check (xD vs. R)')

    # Poly
    y_pred_poly = models['poly_xD'].predict(df_mono)
    sns.lineplot(x=df_mono['R'], y=y_pred_poly.ravel(), ax=axes_mono[0]).set_title('Polynomial')
    # XGB
    y_pred_xgb = models['xgb_xD'].predict(df_mono)
    sns.lineplot(x=df_mono['R'], y=y_pred_xgb.ravel(), ax=axes_mono[1]).set_title('XGBoost')
    # ANN
    X_sc = scaler_X.transform(df_mono)
    y_pred_ann_sc = models['ann_xD'].predict(X_sc, verbose=0)
    y_pred_ann = scaler_y_xD.inverse_transform(y_pred_ann_sc)
    sns.lineplot(x=df_mono['R'], y=y_pred_ann.ravel(), ax=axes_mono[2]).set_title('ANN')
    save_plot(fig_mono, 'results/plots/diag_monotonicity.png')
    plt.close(fig_mono)

    # b) Energy Tradeoff Check (QR vs. R)
    print("Checking energy tradeoff...")
    fig_energy, axes_energy = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig_energy.suptitle('Diagnostic: Energy Tradeoff (QR vs. R)')
    
    # Poly
    y_pred_poly_qr = models['poly_QR'].predict(df_mono)
    sns.lineplot(x=df_mono['R'], y=y_pred_poly_qr.ravel(), ax=axes_energy[0]).set_title('Polynomial')
    # XGB
    y_pred_xgb_qr = models['xgb_QR'].predict(df_mono)
    sns.lineplot(x=df_mono['R'], y=y_pred_xgb_qr.ravel(), ax=axes_energy[1]).set_title('XGBoost')
    # ANN
    y_pred_ann_qr_sc = models['ann_QR'].predict(X_sc, verbose=0)
    y_pred_ann_qr = scaler_y_QR.inverse_transform(y_pred_ann_qr_sc)
    sns.lineplot(x=df_mono['R'], y=y_pred_ann_qr.ravel(), ax=axes_energy[2]).set_title('ANN')
    save_plot(fig_energy, 'results/plots/diag_energy_tradeoff.png')
    plt.close(fig_energy)

    # c) Extrapolation Check
    print("Checking extrapolation behavior...")
    R_extrap = np.linspace(0.8, 6.0, 50)
    # Fix: Ensure column order matches training data ['xF', 'R', 'B']
    df_extrap = pd.DataFrame({'xF': [0.5]*50, 'R': R_extrap, 'B': [4.0]*50})
    
    fig_extrap, axes_extrap = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig_extrap.suptitle('Diagnostic: Extrapolation Check (xD vs. R beyond training range)')

    # Poly
    y_pred_poly_ex = models['poly_xD'].predict(df_extrap)
    sns.lineplot(x=df_extrap['R'], y=y_pred_poly_ex.ravel(), ax=axes_extrap[0]).set_title('Polynomial')
    axes_extrap[0].axvline(x=5.0, color='r', linestyle='--', label='Training Max R')
    axes_extrap[0].legend()
    # XGB
    y_pred_xgb_ex = models['xgb_xD'].predict(df_extrap)
    sns.lineplot(x=df_extrap['R'], y=y_pred_xgb_ex.ravel(), ax=axes_extrap[1]).set_title('XGBoost')
    axes_extrap[1].axvline(x=5.0, color='r', linestyle='--', label='Training Max R')
    axes_extrap[1].legend()
    # ANN
    X_sc_ex = scaler_X.transform(df_extrap)
    y_pred_ann_sc_ex = models['ann_xD'].predict(X_sc_ex, verbose=0)
    y_pred_ann_ex = scaler_y_xD.inverse_transform(y_pred_ann_sc_ex)
    sns.lineplot(x=df_extrap['R'], y=y_pred_ann_ex.ravel(), ax=axes_extrap[2]).set_title('ANN')
    axes_extrap[2].axvline(x=5.0, color='r', linestyle='--', label='Training Max R')
    axes_extrap[2].legend()
    save_plot(fig_extrap, 'results/plots/diag_extrapolation.png')
    plt.close(fig_extrap)
    
    print("\n--- Diagnostics Complete ---")

if __name__ == '__main__':
    run_all_diagnostics()