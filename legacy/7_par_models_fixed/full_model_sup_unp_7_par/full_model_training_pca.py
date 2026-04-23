import pandas as pd
import numpy as np
import torch
import os
import time
import joblib
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlpres import SplitMLPRegressor, MLPWithResiduals, UnifiedModel
from functions_BRR import get_rmse, get_mae, get_r_squared, get_rrmse
import LSST_sampling as lsst

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = r"../../7_par_dataset/ASTRAI DATASET/dataset_preprocessed.parquet"
N_DAYS = 1601
N_PARAMS = 7
N_PCA_COMPONENTS = 50
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 0.001
NOISE_STD = 0.05 

# Configurable loss weights
ALPHA_CHAR = 1.0
ALPHA_GEN = 7 / 50  # old 1 

PARAM_NAMES = ['Raggio', 'Massa', 'Energia', 'Nickel', 'Mcsm', 'Rcsm', 'Slope']

# Augmentation functions
def add_gaussian_noise_fast(X, noise_std):
    """Adds Gaussian noise (vectorized)."""
    noise = np.random.randn(*X.shape)
    return X + noise_std * noise

def apply_lsst_pipeline(curves_batch):
    """
    Pipeline: Noise -> LSST Mask -> Interpolation.
    """
    augmented = curves_batch.copy()

    # 1. Add noise
    augmented = add_gaussian_noise_fast(augmented, NOISE_STD)

    # Pre-calculate calendar
    calendar = np.arange(N_DAYS) / lsst.DIG_SAMPLES_X_DAY

    for i in range(len(augmented)):
        sun_mask = lsst.sun_masking_np(calendar)
        cloud_mask = lsst.random_cloud_masking(np.ones_like(calendar))
        combined_mask = (1 - sun_mask) * (1 - cloud_mask)

        curve = augmented[i]
        valid_idx = np.where(combined_mask == 1)[0]

        if len(valid_idx) < 2:
            continue

        valid_vals = curve[valid_idx]
        augmented[i] = np.interp(np.arange(N_DAYS), valid_idx, valid_vals)

    return augmented

def print_final_stats(name, history):
    print(f"\n--- {name} (10-Fold Mean) ---")
    print(f"R2:    {np.mean(history['R2']):.4f}  (± {np.std(history['R2']):.4f})")
    print(f"RMSE:  {np.mean(history['RMSE']):.4f}  (± {np.std(history['RMSE']):.4f})")
    print(f"RRMSE: {np.mean(history['RRMSE']):.4f}  (± {np.std(history['RRMSE']):.4f})")
    print(f"MAE:   {np.mean(history['MAE']):.4f}  (± {np.std(history['MAE']):.4f})")

def print_per_param_table(header, all_folds_metrics):
    """
    Prints a table with per-parameter metrics averaged across folds.
    all_folds_metrics: list of dicts, one per fold. Each dict has keys = param names,
                       values = dict with RMSE, RRMSE, MAE, R2.
    """
    print(f"\n{'='*80}")
    print(f"  {header} - Per-Parameter Metrics (10-Fold Mean ± Std)")
    print(f"{'='*80}")
    print(f"  {'Parameter':<12} {'R2':>12} {'RMSE':>12} {'RRMSE':>12} {'MAE':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for p in PARAM_NAMES:
        r2_vals = [f[p]['R2'] for f in all_folds_metrics]
        rmse_vals = [f[p]['RMSE'] for f in all_folds_metrics]
        rrmse_vals = [f[p]['RRMSE'] for f in all_folds_metrics]
        mae_vals = [f[p]['MAE'] for f in all_folds_metrics]

        print(f"  {p:<12} "
              f"{np.mean(r2_vals):>5.4f}±{np.std(r2_vals):.4f} "
              f"{np.mean(rmse_vals):>5.4f}±{np.std(rmse_vals):.4f} "
              f"{np.mean(rrmse_vals):>5.4f}±{np.std(rrmse_vals):.4f} "
              f"{np.mean(mae_vals):>5.4f}±{np.std(mae_vals):.4f}")
    print(f"{'='*80}")

def print_per_point_summary(header, all_folds_point_metrics):
    """
    Prints summary statistics for per-point (per time-step) generation metrics.
    all_folds_point_metrics: list of dicts, one per fold. Each dict has keys =
                             metric names, values = arrays of shape (N_DAYS,).
    """
    print(f"\n{'='*80}")
    print(f"  {header} - Per-Point Generation Metrics (across {N_DAYS} time steps)")
    print(f"{'='*80}")

    # Average across folds first: shape (n_folds, N_DAYS) -> (N_DAYS,)
    r2_per_point = np.mean([f['R2'] for f in all_folds_point_metrics], axis=0)
    rmse_per_point = np.mean([f['RMSE'] for f in all_folds_point_metrics], axis=0)
    rrmse_per_point = np.mean([f['RRMSE'] for f in all_folds_point_metrics], axis=0)
    mae_per_point = np.mean([f['MAE'] for f in all_folds_point_metrics], axis=0)

    print(f"  {'Metric':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, vals in [('R2', r2_per_point), ('RMSE', rmse_per_point),
                       ('RRMSE', rrmse_per_point), ('MAE', mae_per_point)]:
        print(f"  {name:<8} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
              f"{np.min(vals):>10.4f} {np.max(vals):>10.4f} {np.median(vals):>10.4f}")
    print(f"{'='*80}")

# --- MAIN FUNCTION ---
def main():
    # Data loading
    print(f"Loading data on {DEVICE}...")
    df = pd.read_parquet(BASE_PATH)
    curve_cols = [str(i) for i in range(N_DAYS)]

    X_raw = df[curve_cols].values.astype('float32')
    y_raw = df[PARAM_NAMES].values.astype('float32')

    # Parameter log-transformation
    y_raw = np.log1p(y_raw)

    # 10-Fold Cross-Validation
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    # Dictionaries to accumulate metrics across all folds
    history_char = {'RMSE': [], 'RRMSE': [], 'MAE': [], 'R2': []}
    history_gen = {'RMSE': [], 'RRMSE': [], 'MAE': [], 'R2': []}

    # Per-parameter and per-point metrics across folds
    all_folds_char_per_param = []
    all_folds_gen_per_point = []

    best_global_r2 = -np.inf

    print(f"Starting Training (Split-MLP Char + Single-MLP Gen) with PCA ({N_PCA_COMPONENTS} components)...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_raw), 1):
        start_time = time.time()

        # Split
        X_train_clean, X_test_clean = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        # --- AUGMENTATION (just train) ---
        print(f"    [Fold {fold_idx}] Applying LSST augmentation...", end="\r")
        X_train_aug = apply_lsst_pipeline(X_train_clean)

        # --- SCALING ---
        # Fit scaler on CLEAN data, then transform both clean and augmented
        x_scaler = StandardScaler()
        x_scaler.fit(X_train_clean)
        X_train_clean_scaled = x_scaler.transform(X_train_clean)
        X_train_aug_scaled = x_scaler.transform(X_train_aug)
        X_test_scaled = x_scaler.transform(X_test_clean)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # --- PCA COMPRESSION ---
        # Fit PCA on clean scaled data, then transform all sets
        pca = PCA(n_components=N_PCA_COMPONENTS)
        pca.fit(X_train_clean_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"    [Fold {fold_idx}] PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")

        X_train_clean_pca = pca.transform(X_train_clean_scaled)
        X_train_aug_pca = pca.transform(X_train_aug_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Concatenate clean + augmented data (model sees both) in PCA space
        X_train_combined = np.vstack([X_train_clean_pca, X_train_aug_pca])
        y_train_combined = np.vstack([y_train_scaled, y_train_scaled])

        train_ds = TensorDataset(torch.FloatTensor(X_train_combined), torch.FloatTensor(y_train_combined))

        # Use pin_memory only if CUDA is available
        use_pin_memory = True if str(DEVICE) == 'cuda' else False

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory
        )

        # Model (input/output dim = N_PCA_COMPONENTS instead of N_DAYS)
        regressor = SplitMLPRegressor(input_dim=N_PCA_COMPONENTS, width=256, num_params=N_PARAMS, depth=3, dropout=0.1)
        generator = MLPWithResiduals(input_dim=N_PARAMS, width=256, out_dim=N_PCA_COMPONENTS, depth=3, dropout=0.1)
        model = UnifiedModel(regressor, generator).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.MSELoss()

        # CosineAnnealingLR scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        # Training: loss computed in PCA space (50D)
        model.fit(train_loader, optimizer, criterion, criterion, DEVICE,
                  epochs=EPOCHS, alpha_char=ALPHA_CHAR, alpha_gen=ALPHA_GEN,
                  scheduler=scheduler)

        # Evaluation (reconstruct to original 1601D space for metrics)
        model.eval()
        with torch.no_grad():
            X_test_pca_tensor = torch.FloatTensor(X_test_pca).to(DEVICE)
            y_test_tensor = torch.FloatTensor(y_test_scaled).to(DEVICE)

            # Predictions in PCA space
            pred_params_sc = model.regressor(X_test_pca_tensor).cpu().numpy()
            pred_curves_pca = model.generator(y_test_tensor).cpu().numpy()

            # Characterization: inverse scaling of parameters
            pred_params = y_scaler.inverse_transform(pred_params_sc)
            true_params = y_test

            # Generation: PCA inverse -> scaler inverse -> original 1601D space
            pred_curves_scaled = pca.inverse_transform(pred_curves_pca)
            pred_curves = x_scaler.inverse_transform(pred_curves_scaled)
            true_curves = X_test_clean

        # --- 1. CHARACTERIZATION Metrics (per-parameter) ---
        fold_char_per_param = {}
        for i, p in enumerate(PARAM_NAMES):
            fold_char_per_param[p] = {
                'RMSE': get_rmse(true_params[:, i], pred_params[:, i]),
                'RRMSE': get_rrmse(true_params[:, i], pred_params[:, i]),
                'MAE': get_mae(true_params[:, i], pred_params[:, i]),
                'R2': get_r_squared(true_params[:, i], pred_params[:, i]),
            }
        all_folds_char_per_param.append(fold_char_per_param)

        # Aggregate (mean across params) for backward-compatible summary
        rmse_c = np.mean([fold_char_per_param[p]['RMSE'] for p in PARAM_NAMES])
        rrmse_c = np.mean([fold_char_per_param[p]['RRMSE'] for p in PARAM_NAMES])
        mae_c = np.mean([fold_char_per_param[p]['MAE'] for p in PARAM_NAMES])
        r2_c = np.mean([fold_char_per_param[p]['R2'] for p in PARAM_NAMES])

        history_char['RMSE'].append(rmse_c)
        history_char['RRMSE'].append(rrmse_c)
        history_char['MAE'].append(mae_c)
        history_char['R2'].append(r2_c)

        # --- 2. GENERATION Metrics (per time-step point) ---
        # true_curves, pred_curves: shape (n_samples, N_DAYS)
        # Compute metrics column-wise (for each of the 1601 time steps)
        fold_gen_per_point = {
            'RMSE': np.array([get_rmse(true_curves[:, t], pred_curves[:, t]) for t in range(N_DAYS)]),
            'RRMSE': np.array([get_rrmse(true_curves[:, t], pred_curves[:, t]) for t in range(N_DAYS)]),
            'MAE': np.array([get_mae(true_curves[:, t], pred_curves[:, t]) for t in range(N_DAYS)]),
            'R2': np.array([get_r_squared(true_curves[:, t], pred_curves[:, t]) for t in range(N_DAYS)]),
        }
        all_folds_gen_per_point.append(fold_gen_per_point)

        # Aggregate (global) for backward-compatible summary
        rmse_g = get_rmse(true_curves.ravel(), pred_curves.ravel())
        rrmse_g = get_rrmse(true_curves.ravel(), pred_curves.ravel())
        mae_g = get_mae(true_curves.ravel(), pred_curves.ravel())
        r2_g = get_r_squared(true_curves.ravel(), pred_curves.ravel())

        history_gen['RMSE'].append(rmse_g)
        history_gen['RRMSE'].append(rrmse_g)
        history_gen['MAE'].append(mae_g)
        history_gen['R2'].append(r2_g)

        elapsed = time.time() - start_time
        print(f"Fold {fold_idx} | {elapsed:.0f}s | Char R2: {r2_c:.4f} | Gen R2: {r2_g:.4f}")

        # Best Model Saving (includes PCA object)
        if r2_c > best_global_r2:
            best_global_r2 = r2_c
            torch.save(model.state_dict(), "best_split_mlp_model.pth")
            joblib.dump(x_scaler, "best_split_mlp_x_scaler.pkl")
            joblib.dump(y_scaler, "best_split_mlp_y_scaler.pkl")
            joblib.dump(pca, "best_split_mlp_pca.pkl")

    # Final Report
    print("\n" + "="*80)
    print("FINAL PERFORMANCE REPORT (Un-scaled metrics)")
    print(f"PCA Components: {N_PCA_COMPONENTS}")
    print("="*80)
    print_final_stats("CHARACTERIZATION (Mean across params)", history_char)
    print_final_stats("GENERATION (Global)", history_gen)

    # Detailed per-parameter characterization table
    print_per_param_table("CHARACTERIZATION", all_folds_char_per_param)

    # Detailed per-point generation summary
    print_per_point_summary("GENERATION", all_folds_gen_per_point)

# --- PROTECTION FOR MULTIPROCESSING ---
if __name__ == "__main__":
    main()
