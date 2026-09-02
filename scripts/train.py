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
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from models.split_mlp import SplitMLPRegressor, MLPWithResiduals
from models.unified_model import UnifiedModel
from utils.metrics import get_rmse, get_mae, get_r_squared, get_rrmse
from utils.checkpoints import load_config, load_data, save_model_checkpoint
from utils.augmentation import apply_lsst_pipeline
from utils.log_experiments import create_experiment_dir, save_code, save_config


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
    for k, v in history.items():
        print(f"  {k}: {np.mean(v):.4f}  (+/- {np.std(v):.4f})")


def _preprocess_fold(
    x_train_clean,
    x_test_clean,
    y_train,
    y_test,
    n_pca,
    noise_std,
    n_days,
    samples_per_day,
    fold_idx,
):
    """Augment, scale, and PCA-transform a single fold's data.
    Parameters
    ----------
    x_train_clean : np.ndarray
        Clean training curves, shape (n_train_clean, n_timepoints).
    x_test_clean : np.ndarray
        Clean test curves, shape (n_test, n_timepoints).
    y_train : np.ndarray
        Training parameters, shape (n_train, n_params).
    y_test : np.ndarray
        Test parameters, shape (n_test, n_params).
    n_pca : int
        Number of PCA components to keep.
    noise_std : float
        Standard deviation of Gaussian noise for augmentation.
    n_days : int
        Number of days in the light curve (length of time series).
    samples_per_day : int
        Number of augmented samples to generate per clean curve.
    fold_idx : int
        Index of the current fold (for logging purposes).
    Returns
    -------
    x_train_combined : np.ndarray
        PCA-transformed training curves (clean + augmented),
        shape (n_train_combined, n_pca).
    y_train_combined : np.ndarray
        Scaled training parameters (duplicated for augmented data),
        shape (n_train_combined, n_params).
    x_test_pca : np.ndarray
        PCA-transformed test curves, shape (n_test, n_pca).
    y_test_scaled : np.ndarray
        Scaled test parameters, shape (n_test, n_params).
    x_scaler : StandardScaler
        Fitted scaler for input curves (trained on clean training data).
    y_scaler : StandardScaler
        Fitted scaler for parameters (trained on clean training data).
    pca : PCA
        Fitted PCA object (trained on clean training data).
    """
    print(f"    [Fold {fold_idx}] Applying LSST augmentation...", end="\r")
    x_train_aug, _ = apply_lsst_pipeline(
        x_train_clean,
        n_days,
        noise_std,
        samples_per_day=samples_per_day,
    )

    x_scaler = StandardScaler()
    x_scaler.fit(x_train_clean)
    x_train_clean_scaled = x_scaler.transform(x_train_clean)
    x_train_aug_scaled = x_scaler.transform(x_train_aug)
    x_test_scaled = x_scaler.transform(x_test_clean)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    pca = PCA(n_components=n_pca)
    pca.fit(x_train_clean_scaled)
    expl_var = pca.explained_variance_ratio_.sum()
    print(
        f"    [Fold {fold_idx}] PCA explained variance: {expl_var:.4f} ({expl_var*100:.2f}%)"
    )

    x_train_clean_pca = pca.transform(x_train_clean_scaled)
    x_train_aug_pca = pca.transform(x_train_aug_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    x_train_combined = np.vstack([x_train_clean_pca, x_train_aug_pca])
    y_train_combined = np.vstack([y_train_scaled, y_train_scaled])

    return (
        x_train_combined,
        y_train_combined,
        x_test_pca,
        y_test_scaled,
        x_scaler,
        y_scaler,
        pca,
    )


def _evaluate_fold(
    model,
    x_test_pca,
    y_test_scaled,
    y_test,
    x_test_clean,
    y_scaler,
    pca,
    x_scaler,
    n_params,
    device,
):
    """Run evaluation on the held-out fold and return metric dicts.
    model: the trained UnifiedModel to evaluate
    x_test_pca: (n_test, n_pca) PCA-transformed test curves
    y_test_scaled: (n_test, n_params) scaled test parameters
    y_test: (n_test, n_params) true test parameters (unscaled)
    x_test_clean: (n_test, n_timepoints) true test curves (unscaled)
    y_scaler: fitted Scaler for parameters (to inverse transform predictions)
    pca: fitted PCA object (to inverse transform predicted curves)
    x_scaler: fitted Scaler for curves (to inverse transform predicted curves)
    n_params: number of parameters (for metric computation)
    device: torch.device to run on
    Returns:
    - char_metrics: dict of characterization metrics (RMSE, RRMSE, MAE, R2)
        comparing predicted parameters to y_test
    - gen_metrics: dict of generation metrics (RMSE, RRMSE, MAE, R2)
        comparing reconstructed curves to x_test_clean
    """
    model.eval()
    with torch.no_grad():
        x_test_t = torch.FloatTensor(x_test_pca).to(device)
        y_test_t = torch.FloatTensor(y_test_scaled).to(device)

        pred_params_sc = model.regressor(x_test_t).cpu().numpy()
        pred_curves_pca = model.generator(y_test_t).cpu().numpy()

        pred_params = y_scaler.inverse_transform(pred_params_sc)
        pred_curves = x_scaler.inverse_transform(
            pca.inverse_transform(pred_curves_pca)
        )

    char_metrics = {
        "RMSE": np.mean(
            [get_rmse(y_test[:, i], pred_params[:, i]) for i in range(n_params)]
        ),
        "RRMSE": np.mean(
            [
                get_rrmse(y_test[:, i], pred_params[:, i])
                for i in range(n_params)
            ]
        ),
        "MAE": np.mean(
            [get_mae(y_test[:, i], pred_params[:, i]) for i in range(n_params)]
        ),
        "R2": np.mean(
            [
                get_r_squared(y_test[:, i], pred_params[:, i])
                for i in range(n_params)
            ]
        ),
    }
    gen_metrics = {
        "RMSE": get_rmse(x_test_clean.ravel(), pred_curves.ravel()),
        "RRMSE": get_rrmse(x_test_clean.ravel(), pred_curves.ravel()),
        "MAE": get_mae(x_test_clean.ravel(), pred_curves.ravel()),
        "R2": get_r_squared(x_test_clean.ravel(), pred_curves.ravel()),
    }
    return char_metrics, gen_metrics


def main():
    """Entry point: load config, run K-Fold training, and save the best model.
    Steps:
    1. Parse command-line arguments for config path.
    2. Load the YAML configuration and determine compute device.
    3. Load the raw data (curves and parameters).
    4. For each fold in K-Fold cross-validation:
       a. Preprocess the fold's data (augmentation, scaling, PCA).
       b. Instantiate the UnifiedModel and optimizer.
       c. Train the model on the training fold.
       d. Evaluate on the test fold and record metrics.
       e. Save the model checkpoint if it has the best characterization R2 so far.
    5. After all folds, print aggregate statistics across folds.
    """
    parser = argparse.ArgumentParser(
        description="ASTRAI unified model training"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = create_experiment_dir()
    save_code(exp_dir)
    save_config(exp_dir, config_path=args.config)
    print(f"Experiment directory: {exp_dir}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    n_days = data_cfg["n_days"]
    n_params = data_cfg["n_params"]
    samples_per_day = data_cfg.get("samples_per_day", 4)
    n_pca = model_cfg["pca_components"]
    noise_std = cfg["augmentation"]["noise_std"]

    print(f"Loading data on {device}...")
    x_raw, y_raw = load_data(None, cfg)

    kf = KFold(
        n_splits=train_cfg["n_splits"],
        shuffle=True,
        random_state=train_cfg["random_seed"],
    )

    history_char = {"RMSE": [], "RRMSE": [], "MAE": [], "R2": []}
    history_gen = {"RMSE": [], "RRMSE": [], "MAE": [], "R2": []}
    best_global_r2 = -np.inf

    print(f"Starting Training Char + Gen with PCA ({n_pca} components)...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_raw), 1):
        start_time = time.time()

        x_train_clean, x_test_clean = x_raw[train_idx], x_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        (
            x_train_combined,
            y_train_combined,
            x_test_pca,
            y_test_scaled,
            x_scaler,
            y_scaler,
            pca,
        ) = _preprocess_fold(
            x_train_clean,
            x_test_clean,
            y_train,
            y_test,
            n_pca,
            noise_std,
            n_days,
            samples_per_day,
            fold_idx,
        )

        train_ds = TensorDataset(
            torch.FloatTensor(x_train_combined),
            torch.FloatTensor(y_train_combined),
        )
        train_loader = DataLoader(
            train_ds, batch_size=train_cfg["batch_size"], shuffle=True
        )

        regressor = SplitMLPRegressor(
            input_dim=n_pca,
            width=model_cfg["width"],
            num_params=n_params,
            depth=model_cfg["depth"],
            dropout=model_cfg["dropout"],
        )
        generator = MLPWithResiduals(
            input_dim=n_params,
            width=model_cfg["width"],
            out_dim=n_pca,
            depth=model_cfg["depth"],
            dropout=model_cfg["dropout"],
        )
        model = UnifiedModel(regressor, generator).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg["learning_rate"]
        )
        criterion = torch.nn.MSELoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

        model.fit(
            train_loader,
            optimizer,
            criterion,
            criterion,
            device,
            epochs=train_cfg["epochs"],
            alpha_char=cfg["loss"]["alpha_char"],
            alpha_gen=cfg["loss"]["alpha_gen"],
            scheduler=scheduler,
        )

        char_m, gen_m = _evaluate_fold(
            model,
            x_test_pca,
            y_test_scaled,
            y_test,
            x_test_clean,
            y_scaler,
            pca,
            x_scaler,
            n_params,
            device,
        )

        for key, values in history_char.items():
            values.append(char_m[key])
            history_gen[key].append(gen_m[key])

        elapsed = time.time() - start_time
        print(
            f"Fold {fold_idx} | {elapsed:.0f}s |",
            f"Char R2: {char_m['R2']:.4f} | Gen R2: {gen_m['R2']:.4f}",
        )

        if char_m["R2"] > best_global_r2:
            best_global_r2 = char_m["R2"]
            save_model_checkpoint(
                exp_dir, cfg["checkpoint"], model, x_scaler, y_scaler, pca
            )

    print("\n" + "=" * 50)
    print("FINAL PERFORMANCE REPORT (Un-scaled metrics)")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    print_final_stats("CHARACTERIZATION", history_char)
    print_final_stats("GENERATION", history_gen)
    print("=" * 50)


if __name__ == "__main__":
    main()
