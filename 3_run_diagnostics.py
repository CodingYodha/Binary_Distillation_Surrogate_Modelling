import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error 
from utils import (
    load_data_block_split, 
    load_data_gap_split,
    load_sklearn_model, 
    load_tf_model, 
    save_plot, 
    save_metrics
)

def run_all_diagnostics():
    """
    Loads all trained models and scalers, evaluates them on test data,
    runs all required physical consistency checks, and saves plots and metrics.
    """
    print("--- Running All Diagnostics ---")
    
    # --- Load Data, Models, and Scalers ---
    print("Loading data, models, and scalers...")
    X_train, X_test, y_train, y_test = load_data_block_split()
    
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

    # --- 1. Performance Metrics & Bounds Check ---
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
            
        # Ensure y_pred is 2D array like y_true
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
            
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics[model_name] = {'R2_Score': r2, 'MSE': mse , 'MAE':mae}

        # Bounds Check for xD
        if target == 'xD':
            violations = np.sum((y_pred < 0) | (y_pred > 1))
            metrics[model_name]['Bounds_Violations'] = int(violations)
            print(f"{model_name}: R2={r2:.4f}, MSE={mse:.2e}, MAE={mae:.2e}, Bounds Violations={violations}")
        else:
            print(f"{model_name}: R2={r2:.4f}, MSE={mse:.2e}, MAE={mae:.2e}")
    
    # --- 2. Error Slices (High-Purity Region) ---
    print("\nCalculating metrics for high-purity slice (xD >= 0.95)...")
    high_purity_mask = y_test['xD'] >= 0.95
    X_test_hp = X_test[high_purity_mask]
    y_test_hp = y_test[high_purity_mask]
    
    if not X_test_hp.empty:
        X_test_hp_sc = scaler_X.transform(X_test_hp)
        
        for model_prefix in ['poly', 'xgb', 'ann']:
            model_xD = models[f'{model_prefix}_xD']
            y_true_hp = y_test_hp[['xD']]
            
            if model_prefix == 'ann':
                y_pred_hp_sc = model_xD.predict(X_test_hp_sc, verbose=0)
                y_pred_hp = scaler_y_xD.inverse_transform(y_pred_hp_sc)
            else:
                y_pred_hp = model_xD.predict(X_test_hp)

            # Ensure y_pred_hp is 2D
            if y_pred_hp.ndim == 1:
                y_pred_hp = y_pred_hp.reshape(-1, 1)

            r2_hp = r2_score(y_true_hp, y_pred_hp)
            mse_hp = mean_squared_error(y_true_hp, y_pred_hp)
            mae_hp = mean_absolute_error(y_true_hp, y_pred_hp)
            metrics[f'{model_prefix}_xD']['R2_Score_High_Purity'] = r2_hp
            metrics[f'{model_prefix}_xD']['MSE_High_Purity'] = mse_hp
            metrics[f'{model_prefix}_xD']['MAE_High_Purity'] = mae_hp
            print(f"{model_prefix}_xD (High Purity): R2={r2_hp:.4f}, MSE={mse_hp:.2e}, MAE={mae_hp:.2e}")
    else:
        print("No high-purity data in the test set to evaluate.")

    save_metrics(metrics, 'results/metrics.json')

    # --- 3. EDA Plots ---
    print("\nGenerating EDA plots...")
    df = pd.read_csv('data/processed/master_data.csv')
    df_unique = df.drop_duplicates(subset=['R', 'B'])
    
    fig_heatmap, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df_pivot_xD = df_unique.pivot(index='R', columns='B', values='xD')
    df_pivot_QR = df_unique.pivot(index='R', columns='B', values='QR')
    sns.heatmap(df_pivot_xD, ax=ax1)
    ax1.set_title('EDA Heatmap: xD vs (R, B)')
    sns.heatmap(df_pivot_QR, ax=ax2)
    ax2.set_title('EDA Heatmap: QR vs (R, B)')
    save_plot(fig_heatmap, 'results/plots/eda_heatmaps.png')

    # --- 4. Generate All Plots ---
    print("\nGenerating diagnostic plots...")

    # a) Parity Plots
    fig_parity, axes_parity = plt.subplots(2, 3, figsize=(18, 10))
    fig_parity.suptitle('Diagnostic: Parity Plots (Predicted vs. Actual)', fontsize=16)
    
    for i, model_prefix in enumerate(['poly', 'xgb', 'ann']):
        # xD
        ax = axes_parity[0, i]
        model_xD = models[f'{model_prefix}_xD']
        y_true_xD = y_test[['xD']]
        if model_prefix == 'ann':
            y_pred_xD = scaler_y_xD.inverse_transform(model_xD.predict(X_test_sc, verbose=0))
        else:
            y_pred_xD = model_xD.predict(X_test)
        
        # Ensure consistent shapes
        if y_pred_xD.ndim == 1:
            y_pred_xD = y_pred_xD.reshape(-1, 1)
            
        ax.scatter(y_true_xD.values.ravel(), y_pred_xD.ravel(), alpha=0.5)
        min_val = min(y_true_xD.values.min(), y_pred_xD.min())
        max_val = max(y_true_xD.values.max(), y_pred_xD.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title(f'{model_prefix.upper()} - xD')
        ax.set_xlabel('Actual xD')
        ax.set_ylabel('Predicted xD')
        
        # QR
        ax = axes_parity[1, i]
        model_QR = models[f'{model_prefix}_QR']
        y_true_QR = y_test[['QR']]
        if model_prefix == 'ann':
            y_pred_QR = scaler_y_QR.inverse_transform(model_QR.predict(X_test_sc, verbose=0))
        else:
            y_pred_QR = model_QR.predict(X_test)
            
        # Ensure consistent shapes
        if y_pred_QR.ndim == 1:
            y_pred_QR = y_pred_QR.reshape(-1, 1)
            
        ax.scatter(y_true_QR.values.ravel(), y_pred_QR.ravel(), alpha=0.5)
        min_val = min(y_true_QR.values.min(), y_pred_QR.min())
        max_val = max(y_true_QR.values.max(), y_pred_QR.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title(f'{model_prefix.upper()} - QR')
        ax.set_xlabel('Actual QR')
        ax.set_ylabel('Predicted QR')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig_parity, 'results/plots/diag_parity_plots.png')

    # b) Residual Plots
    for target in ['xD', 'QR']:
        fig_res, axes_res = plt.subplots(3, 3, figsize=(18, 15))
        fig_res.suptitle(f'Diagnostic: Residuals vs. Inputs for {target}', fontsize=16)
        
        for i, model_prefix in enumerate(['poly', 'xgb', 'ann']):
            y_true = y_test[[target]]
            model = models[f'{model_prefix}_{target}']
            if model_prefix == 'ann' and target == 'xD':
                y_pred = scaler_y_xD.inverse_transform(model.predict(X_test_sc, verbose=0))
            elif model_prefix == 'ann' and target == 'QR':
                y_pred = scaler_y_QR.inverse_transform(model.predict(X_test_sc, verbose=0))
            else:
                y_pred = model.predict(X_test)
            
            # Ensure consistent shapes for residual calculation
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            residuals = y_true.values - y_pred

            for j, input_col in enumerate(['R', 'B', 'xF']):
                ax = axes_res[i,j]
                # Ensure both arrays have the same length
                x_vals = X_test[input_col].values
                res_vals = residuals.ravel()
                
                # Handle any length mismatch (should not occur, but safety check)
                min_len = min(len(x_vals), len(res_vals))
                x_vals = x_vals[:min_len]
                res_vals = res_vals[:min_len]
                
                ax.scatter(x_vals, res_vals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_title(f'{model_prefix.upper()} vs {input_col}')
                ax.set_xlabel(input_col)
                ax.set_ylabel('Residual')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_plot(fig_res, f'results/plots/diag_residual_plots_{target}.png')
    
    # c) Sensitivity Plot (xD vs. xF)
    xF_range = np.linspace(0.2, 0.95, 50)
    # Ensure column order matches training data ['xF', 'R', 'B']
    df_sens = pd.DataFrame({'xF': xF_range, 'R': [2.9]*50, 'B': [2.0]*50})
    fig_sens, axes_sens = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig_sens.suptitle('Diagnostic: Sensitivity Check (xD vs. xF)')

    # Poly, XGB, ANN
    for i, model_prefix in enumerate(['poly', 'xgb', 'ann']):
        ax = axes_sens[i]
        model = models[f'{model_prefix}_xD']
        if model_prefix == 'ann':
            X_sc = scaler_X.transform(df_sens)
            y_pred_sc = model.predict(X_sc, verbose=0)
            y_pred = scaler_y_xD.inverse_transform(y_pred_sc)
        else:
            y_pred = model.predict(df_sens)
            
        # Ensure y_pred is 1D for plotting
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
            
        ax.plot(df_sens['xF'], y_pred)
        ax.set_title(model_prefix.upper())
        ax.set_xlabel('xF')
        ax.set_ylabel('Predicted xD')
        ax.grid(True)
    
    save_plot(fig_sens, 'results/plots/diag_sensitivity_xF.png')

    # --- 5. Physical Diagnostics ---
    print("\nRunning physical consistency checks...")
    
    # a) Monotonicity Check (xD vs. R)
    print("Checking monotonicity...")
    R_range = np.linspace(0.8, 5.0, 50)
    # Ensure column order matches training data ['xF', 'R', 'B']
    df_mono = pd.DataFrame({'xF': [0.5]*50, 'R': R_range, 'B': [2.0]*50})
    
    fig_mono, axes_mono = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig_mono.suptitle('Diagnostic: Monotonicity Check (xD vs. R)')

    # Poly
    y_pred_poly = models['poly_xD'].predict(df_mono)
    axes_mono[0].plot(df_mono['R'], y_pred_poly.ravel())
    axes_mono[0].set_title('Polynomial')
    axes_mono[0].set_xlabel('R')
    axes_mono[0].set_ylabel('Predicted xD')
    axes_mono[0].grid(True)
    
    # XGB
    y_pred_xgb = models['xgb_xD'].predict(df_mono)
    axes_mono[1].plot(df_mono['R'], y_pred_xgb.ravel())
    axes_mono[1].set_title('XGBoost')
    axes_mono[1].set_xlabel('R')
    axes_mono[1].set_ylabel('Predicted xD')
    axes_mono[1].grid(True)
    
    # ANN
    X_sc = scaler_X.transform(df_mono)
    y_pred_ann_sc = models['ann_xD'].predict(X_sc, verbose=0)
    y_pred_ann = scaler_y_xD.inverse_transform(y_pred_ann_sc)
    axes_mono[2].plot(df_mono['R'], y_pred_ann.ravel())
    axes_mono[2].set_title('ANN')
    axes_mono[2].set_xlabel('R')
    axes_mono[2].set_ylabel('Predicted xD')
    axes_mono[2].grid(True)
    
    save_plot(fig_mono, 'results/plots/diag_monotonicity.png')

    # b) Energy Tradeoff Check (QR vs. R)
    print("Checking energy tradeoff...")
    fig_energy, axes_energy = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig_energy.suptitle('Diagnostic: Energy Tradeoff (QR vs. R)')
    
    # Poly
    y_pred_poly_qr = models['poly_QR'].predict(df_mono)
    axes_energy[0].plot(df_mono['R'], y_pred_poly_qr.ravel())
    axes_energy[0].set_title('Polynomial')
    axes_energy[0].set_xlabel('R')
    axes_energy[0].set_ylabel('Predicted QR')
    axes_energy[0].grid(True)
    
    # XGB
    y_pred_xgb_qr = models['xgb_QR'].predict(df_mono)
    axes_energy[1].plot(df_mono['R'], y_pred_xgb_qr.ravel())
    axes_energy[1].set_title('XGBoost')
    axes_energy[1].set_xlabel('R')
    axes_energy[1].set_ylabel('Predicted QR')
    axes_energy[1].grid(True)
    
    # ANN
    y_pred_ann_qr_sc = models['ann_QR'].predict(X_sc, verbose=0)
    y_pred_ann_qr = scaler_y_QR.inverse_transform(y_pred_ann_qr_sc)
    axes_energy[2].plot(df_mono['R'], y_pred_ann_qr.ravel())
    axes_energy[2].set_title('ANN')
    axes_energy[2].set_xlabel('R')
    axes_energy[2].set_ylabel('Predicted QR')
    axes_energy[2].grid(True)
    
    save_plot(fig_energy, 'results/plots/diag_energy_tradeoff.png')

    # c) Extrapolation Check
    print("Checking extrapolation behavior...")
    R_extrap = np.linspace(0.8, 6.0, 50)
    # Ensure column order matches training data ['xF', 'R', 'B']
    df_extrap = pd.DataFrame({'xF': [0.5]*50, 'R': R_extrap, 'B': [4.0]*50})
    
    fig_extrap, axes_extrap = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig_extrap.suptitle('Diagnostic: Extrapolation Check (xD vs. R beyond training range)')

    # Poly
    y_pred_poly_ex = models['poly_xD'].predict(df_extrap)
    axes_extrap[0].plot(df_extrap['R'], y_pred_poly_ex.ravel())
    axes_extrap[0].set_title('Polynomial')
    axes_extrap[0].axvline(x=5.0, color='r', linestyle='--', label='Training Max R')
    axes_extrap[0].set_xlabel('R')
    axes_extrap[0].set_ylabel('Predicted xD')
    axes_extrap[0].legend()
    axes_extrap[0].grid(True)
    
    # XGB
    y_pred_xgb_ex = models['xgb_xD'].predict(df_extrap)
    axes_extrap[1].plot(df_extrap['R'], y_pred_xgb_ex.ravel())
    axes_extrap[1].set_title('XGBoost')
    axes_extrap[1].axvline(x=5.0, color='r', linestyle='--', label='Training Max R')
    axes_extrap[1].set_xlabel('R')
    axes_extrap[1].set_ylabel('Predicted xD')
    axes_extrap[1].legend()
    axes_extrap[1].grid(True)
    
    # ANN
    X_sc_ex = scaler_X.transform(df_extrap)
    y_pred_ann_sc_ex = models['ann_xD'].predict(X_sc_ex, verbose=0)
    y_pred_ann_ex = scaler_y_xD.inverse_transform(y_pred_ann_sc_ex)
    axes_extrap[2].plot(df_extrap['R'], y_pred_ann_ex.ravel())
    axes_extrap[2].set_title('ANN')
    axes_extrap[2].axvline(x=5.0, color='r', linestyle='--', label='Training Max R')
    axes_extrap[2].set_xlabel('R')
    axes_extrap[2].set_ylabel('Predicted xD')
    axes_extrap[2].legend()
    axes_extrap[2].grid(True)
    
    save_plot(fig_extrap, 'results/plots/diag_extrapolation.png')
    
    # --- 6. Generalization Test ---
    print("\nRunning generalization test on held-out region...")
    try:
        _, X_test_gap, _, y_test_gap = load_data_gap_split()
        y_test_xD_gap = y_test_gap[['xD']]
        
        models_gap = {
            'poly': load_sklearn_model('models/polynomial_model_xD_gap.joblib'),
            'xgb': load_sklearn_model('models/xgboost_model_xD_gap.joblib'),
            'ann': load_tf_model('models/ann_model_xD_gap.h5')
        }
        
        # Need scalers specific to the gapped training data for the ANN
        scaler_X_gap = load_sklearn_model('models/scaler_X_gap.joblib')
        scaler_y_xD_gap = load_sklearn_model('models/scaler_y_xD_gap.joblib')

        fig_gap, axes_gap = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig_gap.suptitle('Diagnostic: Generalization Test (Interpolation Across Gap in R)', fontsize=16)

        for i, (model_name, model) in enumerate(models_gap.items()):
            ax = axes_gap[i]
            if model_name == 'ann':
                X_test_gap_sc = scaler_X_gap.transform(X_test_gap)
                y_pred_gap_sc = model.predict(X_test_gap_sc, verbose=0)
                y_pred_gap = scaler_y_xD_gap.inverse_transform(y_pred_gap_sc)
            else:
                y_pred_gap = model.predict(X_test_gap)
            
            # Ensure consistent shapes
            if y_pred_gap.ndim > 1:
                y_pred_gap = y_pred_gap.ravel()
            
            ax.scatter(X_test_gap['R'], y_test_xD_gap.values.ravel(), label='Actual Data', alpha=0.6)
            # Sort for line plot
            sorted_indices = X_test_gap['R'].argsort()
            ax.plot(X_test_gap['R'].iloc[sorted_indices], y_pred_gap[sorted_indices], 'r-', label='Model Prediction')
            ax.set_title(f'{model_name.upper()} Model')
            ax.set_xlabel('Reflux Ratio (R)')
            ax.set_ylabel('Distillate Purity (xD)')
            ax.legend()
            ax.grid(True)
            
        save_plot(fig_gap, 'results/plots/diag_generalization_gap_test.png')
        
    except Exception as e:
        print(f"Generalization test failed: {e}")
        print("Continuing with remaining diagnostics...")

    print("\n--- All Diagnostics Complete ---")

if __name__ == '__main__':
    run_all_diagnostics()