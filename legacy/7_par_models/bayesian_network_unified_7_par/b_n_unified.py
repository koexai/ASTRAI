import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from functions_BRR import get_rmse, get_mae, get_r_squared, get_rrmse
from data_corruption_v2 import add_gaussian_noise_v2
import LSST_sampling as lsst

# Configuration
base_path = r"../ASTRAI/seven parameters dataset/ASTRAI DATASET/dataset_preprocessed.csv"
N_DAYS = 1601
N_PARAMS = 7
N_FOLD = 10

def apply_lsst_interpolation(curves_batch):
    """
    Applies LSST mask and INTERPOLATES missing data.
    Used for both INPUT (Characterization) and TARGET (Generation).
    Reconstructs the curve shape required for linear models.
    """
    augmented_batch = curves_batch.copy()
    calendar = np.arange(N_DAYS) / lsst.DIG_SAMPLES_X_DAY
    
    for i in range(len(augmented_batch)):
        sun_mask = lsst.sun_masking_np(calendar)
        cloud_mask = lsst.random_cloud_masking(np.ones_like(calendar))
        combined_mask = (1 - sun_mask) * (1 - cloud_mask)
        
        original_curve = augmented_batch[i]
        valid_indices = np.where(combined_mask == 1)[0]
        
        # Fallback if the curve is totally obscured (rare but possible)
        if len(valid_indices) < 2:
            continue 
            
        valid_values = original_curve[valid_indices]
        # Linear reconstruction over all 1601 points
        interp_curve = np.interp(np.arange(N_DAYS), valid_indices, valid_values)
        augmented_batch[i] = interp_curve
        
    return augmented_batch

# Data loading
print(f"Loading unique dataset: {base_path}...")
df = pd.read_csv(base_path)

param_names = ['Raggio', 'Massa', 'Energia', 'Nickel', 'Mcsm', 'Rcsm', 'Slope']
curve_cols = [str(i) for i in range(N_DAYS)]

C_raw = df[curve_cols].values
P_raw = df[param_names].copy()

print(f"Dataset loaded. Total rows: {len(df)}")

# Parameter Pre-processing 
for col in param_names:
    P_raw[col] = np.log1p(P_raw[col])

P_values = P_raw.values

# 10-Fold Cross-Validation
kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=42)

results_char = []
results_gen = []

print(f"\n Starting {N_FOLD}-Fold Unified CV (Bayesian Networks)...")

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(C_raw), 1):
    # Raw data split
    C_train_raw, C_test_raw = C_raw[train_idx], C_raw[test_idx]
    P_train, P_test = P_values[train_idx], P_values[test_idx]
    
    # =========================================================
    # TASK A: CHARACTERIZATION (Curves -> Parameters)
    # =========================================================
    
    # 1. Noise Injection (On Raw data)
    C_train_noisy = add_gaussian_noise_v2(C_train_raw, noise_std=0.05)
    
    # 2. LSST Sampling + interpolation
    C_train_interp_input = apply_lsst_interpolation(C_train_noisy)
    
    # 3. Scaling
    scaler_x_char = StandardScaler()
    scaler_y_char = StandardScaler()
    
    X_train_char = scaler_x_char.fit_transform(C_train_interp_input)
    X_test_char = scaler_x_char.transform(C_test_raw) 
    
    Y_train_char = scaler_y_char.fit_transform(P_train)
    Y_test_char = scaler_y_char.transform(P_test)
    
    # 4. Training
    fold_metrics_char = {'R2': [], 'RMSE': [], 'MAE': [], 'RRMSE': []}
    
    for i in range(N_PARAMS):
        model_char = BayesianRidge(max_iter=500, tol=1e-4)
        model_char.fit(X_train_char, Y_train_char[:, i])
        
        y_pred = model_char.predict(X_test_char)
        
        # Calculate metrics per parameter
        fold_metrics_char['R2'].append(get_r_squared(Y_test_char[:, i], y_pred))
        fold_metrics_char['RMSE'].append(get_rmse(Y_test_char[:, i], y_pred))
        fold_metrics_char['MAE'].append(get_mae(Y_test_char[:, i], y_pred))
        fold_metrics_char['RRMSE'].append(get_rrmse(Y_test_char[:, i], y_pred))
    
    results_char.append({
        'fold': fold_idx,
        'mean_r2': np.mean(fold_metrics_char['R2']),
        'mean_rmse': np.mean(fold_metrics_char['RMSE']),
        'mean_mae': np.mean(fold_metrics_char['MAE']),
        'mean_rrmse': np.mean(fold_metrics_char['RRMSE'])
    })

    # =========================================================
    # TASK B: GENERATION (Paramters -> Curves)
    # =========================================================
    
    # 1. LSST Sampling + Interpolation (On Target)
    C_train_interp_target = apply_lsst_interpolation(C_train_raw)
    
    # 2. Scaling
    scaler_x_gen = StandardScaler()
    scaler_y_gen = StandardScaler()
    
    X_train_gen = scaler_x_gen.fit_transform(P_train)
    X_test_gen = scaler_x_gen.transform(P_test)
    
    Y_train_gen = scaler_y_gen.fit_transform(C_train_interp_target)
    Y_test_gen = scaler_y_gen.transform(C_test_raw) 
    
    # 3. Training
    model_gen = MultiOutputRegressor(BayesianRidge(max_iter=500, tol=1e-4))
    model_gen.fit(X_train_gen, Y_train_gen)
    
    Y_pred_gen = model_gen.predict(X_test_gen)
    
    results_gen.append({
        'fold': fold_idx,
        'r2': get_r_squared(Y_test_gen, Y_pred_gen),
        'rmse': get_rmse(Y_test_gen, Y_pred_gen),
        'mae': get_mae(Y_test_gen, Y_pred_gen),
        'rrmse': get_rrmse(Y_test_gen, Y_pred_gen)
    })
    
    print(f"Fold {fold_idx} | Char R2: {results_char[-1]['mean_r2']:.4f} | Gen R2: {results_gen[-1]['r2']:.4f}")

# Final report

def print_detailed_report(task_name, results_list, keys):
    """
    Prints a detailed table and final averages for all metrics.
    keys: dictionary mapping metric name to the key in the results dictionary
    e.g., {'R2': 'mean_r2', 'RMSE': 'mean_rmse'}
    """
    print("\n" + "="*80)
    print(f" final report: {task_name}")
    print("="*80)
    
    # Table header
    header = f"{'Fold':<6} | " + " | ".join([f"{k:<10}" for k in keys.keys()])
    print(header)
    print("-" * 80)
    
    # Table rows
    metric_values = {k: [] for k in keys.keys()}
    
    for row in results_list:
        line = f"{row['fold']:<6} | "
        for metric_name, dict_key in keys.items():
            val = row[dict_key]
            metric_values[metric_name].append(val)
            line += f"{val:<10.4f} | "
        print(line)
        
    print("-" * 80)
    
    # Averages Row
    mean_line = f"{'MEDIA':<6} | "
    for metric_name in keys.keys():
        mean_val = np.mean(metric_values[metric_name])
        mean_line += f"{mean_val:<10.4f} | "
    print(mean_line)
    print("="*80)

# Key configuration for Characterization
keys_char = {
    'R2': 'mean_r2',
    'RMSE': 'mean_rmse',
    'MAE': 'mean_mae',
    'RRMSE': 'mean_rrmse'
}

# Key configuration for Generation (keys are direct)
keys_gen = {
    'R2': 'r2',
    'RMSE': 'rmse',
    'MAE': 'mae',
    'RRMSE': 'rrmse'
}

# Print reports
print_detailed_report("CHARACTERIZATION (Curves -> Params)", results_char, keys_char)
print_detailed_report("GENERATION (Params -> Curves)", results_gen, keys_gen)

# Comparative R2 Plot
plt.figure(figsize=(12, 6))
folds = range(1, 11)
r2_char = [x['mean_r2'] for x in results_char]
r2_gen = [x['r2'] for x in results_gen]

plt.plot(folds, r2_char, 'o-', label='Characterization (R2)', color='blue')
plt.plot(folds, r2_gen, 's-', label='Generation (R2)', color='orange')
plt.axhline(y=np.mean(r2_char), color='blue', linestyle='--', alpha=0.5, label='Char Mean')
plt.axhline(y=np.mean(r2_gen), color='orange', linestyle='--', alpha=0.5, label='Gen Mean')

plt.title("Unified Bayesian Network Performance (Interpolated LSST)")
plt.xlabel("Fold")
plt.ylabel("R2 Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(folds)
plt.savefig("performance_unified_bayesian_interp.png")
print("\n Chart saved as 'performance_unified_bayesian_interp.png'")