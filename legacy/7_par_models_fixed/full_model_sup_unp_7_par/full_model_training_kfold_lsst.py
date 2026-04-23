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
from mlpres import SplitMLPRegressor, MLPWithResiduals, UnifiedModel
from functions_BRR import get_rmse, get_mae, get_r_squared, get_rrmse
import LSST_sampling as lsst

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = r"../../7_par_dataset/ASTRAI DATASET/dataset_preprocessed.parquet"
N_DAYS = 1601
N_PARAMS = 7
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
NOISE_STD = 0.05

# FIX 7: Configurable loss weights (default 1.0/1.0)
ALPHA_CHAR = 1.0
ALPHA_GEN = 1.0

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

# --- MAIN FUNCTION ---
# Moving all executive code inside main to support multiprocessing
def main():
    # Data loading
    # FIX 5: Fixed typo "tarting" -> "Starting"
    print(f"Loading data on {DEVICE}...")
    df = pd.read_parquet(BASE_PATH)
    param_names = ['Raggio', 'Massa', 'Energia', 'Nickel', 'Mcsm', 'Rcsm', 'Slope']
    curve_cols = [str(i) for i in range(N_DAYS)]

    X_raw = df[curve_cols].values.astype('float32')
    y_raw = df[param_names].values.astype('float32')

    # Parameter log-transformation
    y_raw = np.log1p(y_raw)

    # 10-Fold Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Dictionaries to accumulate metrics across all folds
    history_char = {'RMSE': [], 'RRMSE': [], 'MAE': [], 'R2': []}
    history_gen = {'RMSE': [], 'RRMSE': [], 'MAE': [], 'R2': []}

    best_global_r2 = -np.inf

    print(f"Starting Training (Split-MLP Char + Single-MLP Gen)...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_raw), 1):
        start_time = time.time()

        # Split
        X_train_clean, X_test_clean = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        # --- AUGMENTATION (just train) ---
        print(f"    [Fold {fold_idx}] Applying LSST augmentation...", end="\r")
        X_train_aug = apply_lsst_pipeline(X_train_clean)

        # --- SCALING ---
        # FIX 1: Fit scaler on CLEAN data, then transform both clean and augmented
        x_scaler = StandardScaler()
        x_scaler.fit(X_train_clean)
        X_train_clean_scaled = x_scaler.transform(X_train_clean)
        X_train_aug_scaled = x_scaler.transform(X_train_aug)
        X_test_scaled = x_scaler.transform(X_test_clean)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # FIX 2: Concatenate clean + augmented data (model sees both)
        X_train_combined = np.vstack([X_train_clean_scaled, X_train_aug_scaled])
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

        # Model
        regressor = SplitMLPRegressor(input_dim=N_DAYS, width=256, num_params=N_PARAMS, depth=3, dropout=0.1)
        generator = MLPWithResiduals(input_dim=N_PARAMS, width=256, out_dim=N_DAYS, depth=3, dropout=0.1)
        model = UnifiedModel(regressor, generator).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.MSELoss()

        # FIX 3: Add CosineAnnealingLR scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Training (with alpha weights and scheduler)
        model.fit(train_loader, optimizer, criterion, criterion, DEVICE,
                  epochs=EPOCHS, alpha_char=ALPHA_CHAR, alpha_gen=ALPHA_GEN,
                  scheduler=scheduler)

        # Evalutation (Reverting scaling)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
            y_test_tensor = torch.FloatTensor(y_test_scaled).to(DEVICE)

            # Scaled Predictions
            pred_params_sc = model.regressor(X_test_tensor).cpu().numpy()
            pred_curves_sc = model.generator(y_test_tensor).cpu().numpy()

            # Inverse Scaling (Returning to log-parameter domain and magnitudes)
            pred_params = y_scaler.inverse_transform(pred_params_sc)
            true_params = y_test

            pred_curves = x_scaler.inverse_transform(pred_curves_sc)
            true_curves = X_test_clean

        # 1. CHARACTERIZATION Metrics (Average over 7 parameters)
        rmse_c = np.mean([get_rmse(true_params[:, i], pred_params[:, i]) for i in range(N_PARAMS)])
        rrmse_c = np.mean([get_rrmse(true_params[:, i], pred_params[:, i]) for i in range(N_PARAMS)])
        mae_c = np.mean([get_mae(true_params[:, i], pred_params[:, i]) for i in range(N_PARAMS)])
        r2_c = np.mean([get_r_squared(true_params[:, i], pred_params[:, i]) for i in range(N_PARAMS)])

        history_char['RMSE'].append(rmse_c)
        history_char['RRMSE'].append(rrmse_c)
        history_char['MAE'].append(mae_c)
        history_char['R2'].append(r2_c)

        # 2. GENERATION Metrics (Global curve metrics)
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

        # Best Model Saving
        if r2_c > best_global_r2:
            best_global_r2 = r2_c
            # FIX 4: Save scalers alongside the model
            torch.save(model.state_dict(), "best_split_mlp_model.pth")
            joblib.dump(x_scaler, "best_split_mlp_x_scaler.pkl")
            joblib.dump(y_scaler, "best_split_mlp_y_scaler.pkl")

    # Final Report
    print("\n" + "="*50)
    print("FINAL PERFORMANCE REPORT (Un-scaled metrics)")
    print("="*50)
    print_final_stats("CHARACTERIZATION", history_char)
    print_final_stats("GENERATION", history_gen)
    print("="*50)

# --- PROTECTION FOR MULTIPROCESSING ---
if __name__ == "__main__":
    main()
