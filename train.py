"""
train.py - K-Fold cross-validated training of the ASTRAI unified model.

Orchestrates the full training pipeline:
1. Load configuration from YAML.
2. Read and log-transform the target physical parameters.
3. For each K-Fold split:
   a. Apply LSST augmentation to the training curves.
   b. Standardize features and targets; fit PCA on clean training data.
   c. Instantiate the UnifiedModel (SplitMLP regressor + residual generator).
   d. Train with a weighted composite loss (characterization + generation).
   e. Evaluate on the held-out fold and record metrics.
4. Report aggregate cross-validation statistics and persist the best checkpoint.

Usage::

    python train.py                              # uses configs/default.yaml
    python train.py --config configs/4par.yaml   # 4-parameter synthetic dataset
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import time
import joblib
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from astrai.models import SplitMLPRegressor, MLPWithResiduals, UnifiedModel
from astrai.metrics import get_rmse, get_mae, get_r_squared, get_rrmse
from astrai.augmentation import apply_lsst_pipeline
from log_experiments import create_experiment_dir, save_code, save_config


def load_config(path="configs/default.yaml"):
    """Load and parse the YAML training configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Nested configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    """Load curves and physical parameters from the configured data source.

    Supports two formats:
    - ``parquet``: a single parquet file with curve columns (``"0"``..``"n_days-1"``)
      and named parameter columns.
    - ``npy_csv``: a ``.npy`` file for curves and a ``.csv`` file for parameters.

    Parameters
    ----------
    cfg : dict
        The full configuration dictionary.

    Returns
    -------
    X_raw : numpy.ndarray
        Light curves, shape ``(n_samples, n_days)``.
    y_raw : numpy.ndarray
        Physical parameters, shape ``(n_samples, n_params)``.
    """
    data_cfg = cfg["data"]
    fmt = data_cfg.get("format", "parquet")
    n_days = data_cfg["n_days"]
    param_names = data_cfg["param_names"]

    if fmt == "parquet":
        df = pd.read_parquet(data_cfg["path"])
        curve_cols = [str(i) for i in range(n_days)]
        X_raw = df[curve_cols].values.astype("float32")
        y_raw = df[param_names].values.astype("float32")

    elif fmt == "npy_csv":
        X_raw = np.load(data_cfg["curves_path"]).astype("float32")
        sep = data_cfg.get("params_csv_sep", ",")
        params_df = pd.read_csv(data_cfg["params_path"], sep=sep)
        y_raw = params_df[param_names].values.astype("float32")

    else:
        raise ValueError(f"Unknown data format: {fmt!r}. Use 'parquet' or 'npy_csv'.")

    return X_raw, y_raw


def print_final_stats(name, history):
    """Print mean +/- std of all tracked metrics across K folds.

    Parameters
    ----------
    name : str
        Label for the metric block (e.g. "CHARACTERIZATION").
    history : dict
        Mapping from metric name to list of per-fold values.
    """
    print(f"\n--- {name} (10-Fold Mean) ---")
    print(f"R2:    {np.mean(history['R2']):.4f}  (+/- {np.std(history['R2']):.4f})")
    print(f"RMSE:  {np.mean(history['RMSE']):.4f}  (+/- {np.std(history['RMSE']):.4f})")
    print(f"RRMSE: {np.mean(history['RRMSE']):.4f}  (+/- {np.std(history['RRMSE']):.4f})")
    print(f"MAE:   {np.mean(history['MAE']):.4f}  (+/- {np.std(history['MAE']):.4f})")


def main():
    """Entry point: load config, run K-Fold training, and save the best model."""
    parser = argparse.ArgumentParser(description="ASTRAI unified model training")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file (default: configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create experiment directory and save code + config
    exp_dir = create_experiment_dir()
    save_code(exp_dir)
    save_config(exp_dir, config_path=args.config)
    print(f"Experiment directory: {exp_dir}")

    # Unpack configuration sections
    n_days = cfg["data"]["n_days"]
    n_params = cfg["data"]["n_params"]
    param_names = cfg["data"]["param_names"]
    samples_per_day = cfg["data"].get("samples_per_day", 4)
    n_pca = cfg["model"]["pca_components"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    n_splits = cfg["training"]["n_splits"]
    seed = cfg["training"]["random_seed"]
    noise_std = cfg["augmentation"]["noise_std"]
    alpha_char = cfg["loss"]["alpha_char"]
    alpha_gen = cfg["loss"]["alpha_gen"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    dropout = cfg["model"]["dropout"]

    # Data loading and log1p transform on targets
    print(f"Loading data on {device}...")
    X_raw, y_raw = load_data(cfg)
    y_raw = np.log1p(y_raw)

    # K-Fold Cross-Validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    history_char = {"RMSE": [], "RRMSE": [], "MAE": [], "R2": []}
    history_gen = {"RMSE": [], "RRMSE": [], "MAE": [], "R2": []}
    best_global_r2 = -np.inf

    print(f"Starting Training (Split-MLP Char + Single-MLP Gen) with PCA ({n_pca} components)...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_raw), 1):
        start_time = time.time()

        X_train_clean, X_test_clean = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        # LSST augmentation on training set only (test remains clean)
        print(f"    [Fold {fold_idx}] Applying LSST augmentation...", end="\r")
        X_train_aug = apply_lsst_pipeline(X_train_clean, n_days, noise_std,
                                          samples_per_day=samples_per_day)

        # Feature standardization (fit on clean train, apply to augmented + test)
        x_scaler = StandardScaler()
        x_scaler.fit(X_train_clean)
        X_train_clean_scaled = x_scaler.transform(X_train_clean)
        X_train_aug_scaled = x_scaler.transform(X_train_aug)
        X_test_scaled = x_scaler.transform(X_test_clean)

        # Target standardization
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # PCA dimensionality reduction (fit on clean train only)
        pca = PCA(n_components=n_pca)
        pca.fit(X_train_clean_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"    [Fold {fold_idx}] PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")

        X_train_clean_pca = pca.transform(X_train_clean_scaled)
        X_train_aug_pca = pca.transform(X_train_aug_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Concatenate clean + augmented training data (targets duplicated accordingly)
        X_train_combined = np.vstack([X_train_clean_pca, X_train_aug_pca])
        y_train_combined = np.vstack([y_train_scaled, y_train_scaled])

        train_ds = TensorDataset(torch.FloatTensor(X_train_combined), torch.FloatTensor(y_train_combined))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Model instantiation: split regressor + residual generator wrapped in UnifiedModel
        regressor = SplitMLPRegressor(input_dim=n_pca, width=width, num_params=n_params, depth=depth, dropout=dropout)
        generator = MLPWithResiduals(input_dim=n_params, width=width, out_dim=n_pca, depth=depth, dropout=dropout)
        model = UnifiedModel(regressor, generator).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        model.fit(
            train_loader, optimizer, criterion, criterion, device,
            epochs=epochs, alpha_char=alpha_char, alpha_gen=alpha_gen,
            scheduler=scheduler,
        )

        # Evaluation on held-out fold
        model.eval()
        with torch.no_grad():
            X_test_pca_tensor = torch.FloatTensor(X_test_pca).to(device)
            y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

            # Characterization: predict physical params from PCA curves
            pred_params_sc = model.regressor(X_test_pca_tensor).cpu().numpy()
            # Generation: reconstruct PCA curves from ground-truth params
            pred_curves_pca = model.generator(y_test_tensor).cpu().numpy()

            # Inverse-transform to original scale for metric computation
            pred_params = y_scaler.inverse_transform(pred_params_sc)
            true_params = y_test

            pred_curves_scaled = pca.inverse_transform(pred_curves_pca)
            pred_curves = x_scaler.inverse_transform(pred_curves_scaled)
            true_curves = X_test_clean

        # Characterization metrics (averaged across physical parameters)
        rmse_c = np.mean([get_rmse(true_params[:, i], pred_params[:, i]) for i in range(n_params)])
        rrmse_c = np.mean([get_rrmse(true_params[:, i], pred_params[:, i]) for i in range(n_params)])
        mae_c = np.mean([get_mae(true_params[:, i], pred_params[:, i]) for i in range(n_params)])
        r2_c = np.mean([get_r_squared(true_params[:, i], pred_params[:, i]) for i in range(n_params)])

        history_char["RMSE"].append(rmse_c)
        history_char["RRMSE"].append(rrmse_c)
        history_char["MAE"].append(mae_c)
        history_char["R2"].append(r2_c)

        # Generation metrics (flattened across all time-steps and samples)
        rmse_g = get_rmse(true_curves.ravel(), pred_curves.ravel())
        rrmse_g = get_rrmse(true_curves.ravel(), pred_curves.ravel())
        mae_g = get_mae(true_curves.ravel(), pred_curves.ravel())
        r2_g = get_r_squared(true_curves.ravel(), pred_curves.ravel())

        history_gen["RMSE"].append(rmse_g)
        history_gen["RRMSE"].append(rrmse_g)
        history_gen["MAE"].append(mae_g)
        history_gen["R2"].append(r2_g)

        elapsed = time.time() - start_time
        print(f"Fold {fold_idx} | {elapsed:.0f}s | Char R2: {r2_c:.4f} | Gen R2: {r2_g:.4f}")

        # Persist checkpoint of the best-performing fold (by characterization R2)
        if r2_c > best_global_r2:
            best_global_r2 = r2_c
            torch.save(model.state_dict(), os.path.join(exp_dir, cfg["checkpoint"]["model"]))
            joblib.dump(x_scaler, os.path.join(exp_dir, cfg["checkpoint"]["x_scaler"]))
            joblib.dump(y_scaler, os.path.join(exp_dir, cfg["checkpoint"]["y_scaler"]))
            joblib.dump(pca, os.path.join(exp_dir, cfg["checkpoint"]["pca"]))

    # Final cross-validation report
    print("\n" + "=" * 50)
    print("FINAL PERFORMANCE REPORT (Un-scaled metrics)")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    print_final_stats("CHARACTERIZATION", history_char)
    print_final_stats("GENERATION", history_gen)
    print("=" * 50)


if __name__ == "__main__":
    main()
