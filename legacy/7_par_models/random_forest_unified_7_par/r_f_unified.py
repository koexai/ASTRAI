import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from functions_BRR import get_rmse, get_mae, get_r_squared, get_rrmse
import LSST_sampling as lsst 

# Configuration
BASE_PATH = r"../ASTRAI/seven parameters dataset/ASTRAI DATASET/dataset_preprocessed.csv"
N_DAYS = 1601
N_PARAMS = 7
N_FOLD = 10
N_TREES = 10 

def apply_lsst_masking(X_batch):
    """
    Apply realistic LSST sampling.
    Unobserved regions are set to 0.
    """
    masked_batch = X_batch.copy()
    for i in range(len(masked_batch)):
        calendar = np.arange(N_DAYS) / lsst.DIG_SAMPLES_X_DAY
        
        # Masking generation from LSST_sampling.py
        sun_mask = lsst.sun_masking_np(calendar)
        cloud_mask = lsst.random_cloud_masking(np.ones_like(calendar))
        
        # Combined mask
        combined_mask = (1 - sun_mask) * (1 - cloud_mask)
        
        # Application of the mask to the curve
        masked_batch[i] = masked_batch[i] * combined_mask
        
    return masked_batch

# --- 2. Data Loading and Preparation ---
print(f" Loading dataset: {BASE_PATH}...")
df = pd.read_csv(BASE_PATH)

param_names = ['Raggio', 'Massa', 'Energia', 'Nickel', 'Mcsm', 'Rcsm', 'Slope']
curve_cols = [str(i) for i in range(N_DAYS)]

X_curves_raw = df[curve_cols].values
Y_params_raw = df[param_names].copy()

# Log-transformation of physical parameters (target for char, input for gen)
for col in param_names:
    Y_params_raw[col] = np.log1p(Y_params_raw[col])

Y_params_values = Y_params_raw.values

# 10-Fold Cross-Validation
kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=42)

results_char = []
results_gen = []
best_r2_char = -np.inf
best_r2_gen = -np.inf

print(f"\n Starting 10-Fold Unified CV (Random Forest with {N_TREES} trees)...")

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_curves_raw), 1):
    # Data Split (Test set remains 'untouched' / clean)
    X_train_clean, X_test_clean = X_curves_raw[train_idx], X_curves_raw[test_idx]
    Y_train, Y_test = Y_params_values[train_idx], Y_params_values[test_idx]
    
    # --- AUGMENTATION (Training only) ---
    # LSST Masking for Characterization model training
    X_train_masked = apply_lsst_masking(X_train_clean)
    
    # Scaling (performed inside the fold to prevent leakage)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # For Characterization: X = Masked Curves, Y = Parameters
    X_train_char_scaled = scaler_x.fit_transform(X_train_masked)
    X_test_char_scaled = scaler_x.transform(X_test_clean)
    
    # For Generation: X = Parameters, Y = Clean Curves
    Y_train_scaled = scaler_y.fit_transform(Y_train)
    Y_test_scaled = scaler_y.transform(Y_test)
    
    # TASK 1: CHARACTERIZATION (Curves -> Params)
    rf_char = RandomForestRegressor(n_estimators=N_TREES, max_depth=15, n_jobs=-1, random_state=42)
    rf_char.fit(X_train_char_scaled, Y_train_scaled)
    
    pred_params_scaled = rf_char.predict(X_test_char_scaled)
    
    # Characterization Metrics
    r2_c = get_r_squared(Y_test_scaled, pred_params_scaled)
    results_char.append({
        'RMSE': get_rmse(Y_test_scaled, pred_params_scaled),
        'RRMSE': get_rrmse(Y_test_scaled, pred_params_scaled),
        'MAE': get_mae(Y_test_scaled, pred_params_scaled),
        'R2': r2_c
    })
    
    # TASK 2: GENERATION (Params -> Curves)
    rf_gen = RandomForestRegressor(n_estimators=N_TREES, max_depth=15, n_jobs=-1, random_state=42)
    # Note: the target here is the clean curve (X_train_clean)
    rf_gen.fit(Y_train_scaled, X_train_clean)
    
    pred_curves = rf_gen.predict(Y_test_scaled)
    
    # Generation Metrics (comparison with X_test_clean)
    r2_g = get_r_squared(X_test_clean, pred_curves)
    results_gen.append({
        'RMSE': get_rmse(X_test_clean, pred_curves),
        'RRMSE': get_rrmse(X_test_clean, pred_curves),
        'MAE': get_mae(X_test_clean, pred_curves),
        'R2': r2_g
    })
    
    print(f"Fold {fold_idx}/10 completed. [R2 Char: {r2_c:.4f} | R2 Gen: {r2_g:.4f}]")
    
    # Saving Best Models
    if r2_c > best_r2_char:
        best_r2_char = r2_c
        joblib.dump(rf_char, "best_rf_char_local.joblib")
    
    if r2_g > best_r2_gen:
        best_r2_gen = r2_g
        joblib.dump(rf_gen, "best_rf_gen_local.joblib")

# Final report ---
def print_report(name, metrics_list):
    print(f"\n--- {name} ---")
    df_m = pd.DataFrame(metrics_list)
    means = df_m.mean()
    print(f"Average R2:    {means['R2']:.4f}")
    print(f"Average RMSE:  {means['RMSE']:.4f}")
    print(f"Average RRMSE: {means['RRMSE']:.4f}")
    print(f"Average MAE:   {means['MAE']:.4f}")

print("\n" + "="*40)
print("📊 FINAL PERFORMANCE REPORT (CPU)")
print("="*40)
print_report("CHARACTERIZATION (Curves -> Params)", results_char)
print_report("GENERATION (Params -> Curves)", results_gen)
print("="*40)