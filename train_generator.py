"""
train_generator.py - K-Fold training of the generation branch (params -> curves).

Loads pre-computed preprocessing artifacts from ``preprocess.py`` and trains
the MLPWithResiduals with its own hyperparameters.

Usage::

    python train_generator.py
    python train_generator.py --config configs/default_split.yaml --prep preprocessed
"""
import os
import argparse
import shutil
import numpy as np
import torch
import time
import joblib
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from astrai.models import MLPWithResiduals
from astrai.metrics import get_rmse, get_mae, get_r_squared, get_rrmse
from train import print_final_stats
from log_experiments import create_experiment_dir, save_code, save_config


def run_generator_training(cfg, prep_dir="preprocessed", exp_dir=None,
                           config_path="configs/default_split.yaml"):
    """Train the generator and save checkpoints to exp_dir.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    prep_dir : str
        Directory with preprocess.py output.
    exp_dir : str or None
        Experiment directory. Created automatically if None.
    config_path : str
        Path to the config file (for saving into experiment dir).

    Returns
    -------
    str
        The experiment directory used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if exp_dir is None:
        exp_dir = create_experiment_dir(base_dir="experiments/generator")
        save_code(exp_dir)
        save_config(exp_dir, config_path=config_path)
    print(f"Generator experiment directory: {exp_dir}")

    n_params = cfg["data"]["n_params"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]

    gen_cfg = cfg["generator"]
    width = gen_cfg["model"]["width"]
    depth = gen_cfg["model"]["depth"]
    dropout = gen_cfg["model"]["dropout"]
    batch_size = gen_cfg["training"]["batch_size"]
    epochs = gen_cfg["training"]["epochs"]
    lr = gen_cfg["training"]["learning_rate"]

    history = {"RMSE": [], "RRMSE": [], "MAE": [], "R2": []}
    best_r2 = -np.inf

    print(f"Starting Generator Training (MLPWithResiduals) with PCA ({n_pca} components)...")

    for fold_idx in range(1, n_splits + 1):
        start_time = time.time()
        fold_dir = os.path.join(prep_dir, f"fold_{fold_idx}")

        X_train_clean_pca = np.load(os.path.join(fold_dir, "X_train_clean_pca.npy"))
        X_train_aug_pca = np.load(os.path.join(fold_dir, "X_train_aug_pca.npy"))
        X_test_pca = np.load(os.path.join(fold_dir, "X_test_pca.npy"))
        y_train_scaled = np.load(os.path.join(fold_dir, "y_train_scaled.npy"))
        y_test_scaled = np.load(os.path.join(fold_dir, "y_test_scaled.npy"))
        X_test_clean = np.load(os.path.join(fold_dir, "X_test_clean.npy"))

        print(f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}")

        pca_train_combined = np.vstack([X_train_clean_pca, X_train_aug_pca])
        y_train_combined = np.vstack([y_train_scaled, y_train_scaled])

        train_ds = TensorDataset(torch.FloatTensor(y_train_combined),
                                 torch.FloatTensor(pca_train_combined))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = MLPWithResiduals(input_dim=n_params, width=width,
                                 out_dim=n_pca, depth=depth,
                                 dropout=dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_params, batch_curves in train_loader:
                batch_params = batch_params.to(device)
                batch_curves = batch_curves.to(device)
                optimizer.zero_grad()
                pred = model(batch_params)
                loss = criterion(pred, batch_curves)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(train_loader)
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.6f} | LR: {cur_lr:.2e}")

        # Evaluation (inverse-transform with global scaler/PCA)
        x_scaler = joblib.load(os.path.join(prep_dir, "x_scaler.pkl"))
        pca = joblib.load(os.path.join(prep_dir, "pca.pkl"))

        model.eval()
        with torch.no_grad():
            y_test_t = torch.FloatTensor(y_test_scaled).to(device)
            pred_curves_pca = model(y_test_t).cpu().numpy()

            pred_curves_scaled = pca.inverse_transform(pred_curves_pca)
            pred_curves = x_scaler.inverse_transform(pred_curves_scaled)
            true_curves = X_test_clean

        rmse = get_rmse(true_curves.ravel(), pred_curves.ravel())
        rrmse = get_rrmse(true_curves.ravel(), pred_curves.ravel())
        mae = get_mae(true_curves.ravel(), pred_curves.ravel())
        r2 = get_r_squared(true_curves.ravel(), pred_curves.ravel())

        history["RMSE"].append(rmse)
        history["RRMSE"].append(rrmse)
        history["MAE"].append(mae)
        history["R2"].append(r2)

        elapsed = time.time() - start_time
        print(f"Fold {fold_idx} | {elapsed:.0f}s | R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            ckpt = gen_cfg["checkpoint"]
            torch.save(model.state_dict(), os.path.join(exp_dir, ckpt["model"]))
            # Copy global scaler/PCA into experiment dir
            shutil.copy2(os.path.join(prep_dir, "x_scaler.pkl"),
                         os.path.join(exp_dir, ckpt["x_scaler"]))
            shutil.copy2(os.path.join(prep_dir, "y_scaler.pkl"),
                         os.path.join(exp_dir, ckpt["y_scaler"]))
            shutil.copy2(os.path.join(prep_dir, "pca.pkl"),
                         os.path.join(exp_dir, ckpt["pca"]))

    print("\n" + "=" * 50)
    print("GENERATOR - FINAL PERFORMANCE REPORT")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    print_final_stats("GENERATION", history)
    print("=" * 50)

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="ASTRAI generator training")
    parser.add_argument("--config", default="configs/default_split.yaml")
    parser.add_argument("--prep", default="preprocessed",
                        help="Directory with preprocess.py output")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_generator_training(cfg, prep_dir=args.prep, config_path=args.config)


if __name__ == "__main__":
    main()
