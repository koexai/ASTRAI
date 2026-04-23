"""
train_characterizer.py - K-Fold training of the characterization branch (curves -> params).

Loads pre-computed preprocessing artifacts from ``preprocess.py`` and trains
the SplitMLPRegressor with its own hyperparameters.

Usage::

    python train_characterizer.py
    python train_characterizer.py --config configs/default_split.yaml --prep preprocessed
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

from astrai.models import SplitMLPRegressor
from astrai.metrics import get_rmse, get_mae, get_r_squared, get_rrmse
from train import print_final_stats
from log_experiments import create_experiment_dir, save_code, save_config


def run_characterizer_training(cfg, prep_dir="preprocessed", exp_dir=None,
                               config_path="configs/default_split.yaml"):
    """Train the characterizer and save checkpoints to exp_dir.

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
        exp_dir = create_experiment_dir(base_dir="experiments/characterizer")
        save_code(exp_dir)
        save_config(exp_dir, config_path=config_path)
    print(f"Characterizer experiment directory: {exp_dir}")

    n_params = cfg["data"]["n_params"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]

    char_cfg = cfg["characterizer"]
    width = char_cfg["model"]["width"]
    depth = char_cfg["model"]["depth"]
    dropout = char_cfg["model"]["dropout"]
    batch_size = char_cfg["training"]["batch_size"]
    epochs = char_cfg["training"]["epochs"]
    lr = char_cfg["training"]["learning_rate"]

    history = {"RMSE": [], "RRMSE": [], "MAE": [], "R2": []}
    best_r2 = -np.inf

    print(f"Starting Characterizer Training (SplitMLP) with PCA ({n_pca} components)...")

    for fold_idx in range(1, n_splits + 1):
        start_time = time.time()
        fold_dir = os.path.join(prep_dir, f"fold_{fold_idx}")

        X_train_clean_pca = np.load(os.path.join(fold_dir, "X_train_clean_pca.npy"))
        X_train_aug_pca = np.load(os.path.join(fold_dir, "X_train_aug_pca.npy"))
        X_test_pca = np.load(os.path.join(fold_dir, "X_test_pca.npy"))
        y_train_scaled = np.load(os.path.join(fold_dir, "y_train_scaled.npy"))
        y_test = np.load(os.path.join(fold_dir, "y_test.npy"))

        print(f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}")

        X_train_combined = np.vstack([X_train_clean_pca, X_train_aug_pca])
        y_train_combined = np.vstack([y_train_scaled, y_train_scaled])

        train_ds = TensorDataset(torch.FloatTensor(X_train_combined),
                                 torch.FloatTensor(y_train_combined))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = SplitMLPRegressor(input_dim=n_pca, width=width,
                                  num_params=n_params, depth=depth,
                                  dropout=dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(train_loader)
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.6f} | LR: {cur_lr:.2e}")

        # Evaluation (inverse-transform with global y_scaler)
        y_scaler = joblib.load(os.path.join(prep_dir, "y_scaler.pkl"))

        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test_pca).to(device)
            pred_sc = model(X_test_t).cpu().numpy()
            pred_params = y_scaler.inverse_transform(pred_sc)
            true_params = y_test

        rmse = np.mean([get_rmse(true_params[:, i], pred_params[:, i]) for i in range(n_params)])
        rrmse = np.mean([get_rrmse(true_params[:, i], pred_params[:, i]) for i in range(n_params)])
        mae = np.mean([get_mae(true_params[:, i], pred_params[:, i]) for i in range(n_params)])
        r2 = np.mean([get_r_squared(true_params[:, i], pred_params[:, i]) for i in range(n_params)])

        history["RMSE"].append(rmse)
        history["RRMSE"].append(rrmse)
        history["MAE"].append(mae)
        history["R2"].append(r2)

        elapsed = time.time() - start_time
        print(f"Fold {fold_idx} | {elapsed:.0f}s | R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            ckpt = char_cfg["checkpoint"]
            torch.save(model.state_dict(), os.path.join(exp_dir, ckpt["model"]))
            # Copy global scaler/PCA into experiment dir
            shutil.copy2(os.path.join(prep_dir, "x_scaler.pkl"),
                         os.path.join(exp_dir, ckpt["x_scaler"]))
            shutil.copy2(os.path.join(prep_dir, "y_scaler.pkl"),
                         os.path.join(exp_dir, ckpt["y_scaler"]))
            shutil.copy2(os.path.join(prep_dir, "pca.pkl"),
                         os.path.join(exp_dir, ckpt["pca"]))

    print("\n" + "=" * 50)
    print("CHARACTERIZER - FINAL PERFORMANCE REPORT")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    print_final_stats("CHARACTERIZATION", history)
    print("=" * 50)

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="ASTRAI characterizer training")
    parser.add_argument("--config", default="configs/default_split.yaml")
    parser.add_argument("--prep", default="preprocessed",
                        help="Directory with preprocess.py output")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_characterizer_training(cfg, prep_dir=args.prep, config_path=args.config)


if __name__ == "__main__":
    main()
