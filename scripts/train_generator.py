"""
train_generator.py - K-Fold training of the generation branch (params -> curves).

Loads pre-computed preprocessing artifacts from ``preprocess.py`` and trains
the MLPWithResiduals with its own hyperparameters.

Usage::

    python train_generator.py
    python train_generator.py --config configs/default_split.yaml --prep preprocessed
"""
import argparse
import os
import time

import joblib
import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from train import print_final_stats
from models.split_mlp import MLPWithResiduals
from utils.array_dtypes import load_model_array
from utils.metrics import get_rmse, get_mae, get_r_squared, get_rrmse
from utils.checkpoints import copy_preprocessing_artifacts
from utils.fold_selection import resolve_fold_indices
from utils.log_experiments import create_experiment_dir, save_code, save_config
from utils.reproducibility import (
    build_training_seed_plan,
    configure_torch_determinism,
    make_torch_generator,
    seed_data_loader_worker,
)


def _load_fold_data(fold_dir):
    """Load preprocessed arrays for a single generator fold.
    Expects the following files in fold_dir:
    - x_train_clean_pca.npy
    - x_train_aug_pca.npy
    - y_train_scaled.npy
    - y_test_scaled.npy
    - x_test_clean.npy
    Returns:
    - x_train_clean_pca: (n_train_clean, n_pca)
    - x_train_aug_pca: (n_train_aug, n_pca)
    - y_train_scaled: (n_train, n_curves)
    - y_test_scaled: (n_test, n_curves)
    - x_test_clean: (n_test, n_params)
    """
    x_train_clean_pca = load_model_array(
        os.path.join(fold_dir, "x_train_clean_pca.npy")
    )
    x_train_aug_pca = load_model_array(
        os.path.join(fold_dir, "x_train_aug_pca.npy")
    )
    y_train_scaled = load_model_array(
        os.path.join(fold_dir, "y_train_scaled.npy")
    )
    y_test_scaled = load_model_array(
        os.path.join(fold_dir, "y_test_scaled.npy")
    )
    x_test_clean = load_model_array(
        os.path.join(fold_dir, "x_test_clean.npy")
    )
    return (
        x_train_clean_pca,
        x_train_aug_pca,
        y_train_scaled,
        y_test_scaled,
        x_test_clean,
    )


def _train_model(
    model, train_loader, optimizer, criterion, scheduler, epochs, device
):
    """Run the training loop for a single fold.
    model: the PyTorch model to train
    train_loader: DataLoader for the training data
    optimizer: the optimizer to use (e.g. Adam)
    criterion: the loss function (e.g. MSELoss)
    scheduler: learning rate scheduler (e.g. CosineAnnealingLR)
    epochs: number of training epochs
    device: torch.device to run on (e.g. 'cuda' or 'cpu')
    Prints training loss every 10 epochs.
    """
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
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg:.6f} | LR: {cur_lr:.2e}"
            )


def _evaluate_generator(model, y_test_scaled, x_test_clean, prep_dir, device):
    """Evaluate generator on the test fold, return metrics dict.
    Loads the x_scaler and pca from prep_dir to inverse transform predictions.
    model: the trained generator model
    y_test_scaled: (n_test, n_params) scaled test parameters
    x_test_clean: (n_test, n_timepoints) true test curves
    prep_dir: directory containing preprocessing artifacts
    device: torch.device to run on
    Returns a dict of metrics (RMSE, RRMSE, MAE, R2)
    comparing the reconstructed curves to x_test_clean."""
    x_scaler = joblib.load(os.path.join(prep_dir, "x_scaler.pkl"))
    pca = joblib.load(os.path.join(prep_dir, "pca.pkl"))

    model.eval()
    with torch.no_grad():
        y_test_t = torch.FloatTensor(y_test_scaled).to(device)
        pred_curves_pca = model(y_test_t).cpu().numpy()
        pred_curves = x_scaler.inverse_transform(
            pca.inverse_transform(pred_curves_pca)
        )

    return {
        "RMSE": get_rmse(x_test_clean.ravel(), pred_curves.ravel()),
        "RRMSE": get_rrmse(x_test_clean.ravel(), pred_curves.ravel()),
        "MAE": get_mae(x_test_clean.ravel(), pred_curves.ravel()),
        "R2": get_r_squared(x_test_clean.ravel(), pred_curves.ravel()),
    }


def run_generator_training(
    cfg,
    prep_dir="preprocessed",
    exp_dir=None,
    config_path="configs/default_split.yaml",
):
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
    n_params = cfg["data"]["n_params"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]
    base_seed = cfg["preprocessing"]["random_seed"]
    gen_cfg = cfg["generator"]

    held_out_fold = gen_cfg["training"].get("held_out_fold")
    fold_indices = resolve_fold_indices(held_out_fold, n_splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if exp_dir is None:
        exp_dir = create_experiment_dir(base_dir="experiments/generator")
        save_code(exp_dir)
        save_config(exp_dir, config_path=config_path)
    print(f"Generator experiment directory: {exp_dir}")

    history = {"RMSE": [], "RRMSE": [], "MAE": [], "R2": []}
    best_r2 = -np.inf

    print(
        f"Starting Generator Training (MLPWithResiduals) with PCA ({n_pca} components)..."
    )

    if held_out_fold is None:
        print(f"Training all {n_splits} folds.")
    else:
        print(f"Training split with held-out fold {held_out_fold}.")

    for fold_idx in fold_indices:
        start_time = time.time()
        fold_dir = os.path.join(prep_dir, f"fold_{fold_idx}")

        (
            x_train_clean_pca,
            x_train_aug_pca,
            y_train_scaled,
            y_test_scaled,
            x_test_clean,
        ) = _load_fold_data(fold_dir)
        print(f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}")

        pca_train_combined = np.vstack([x_train_clean_pca, x_train_aug_pca])
        y_train_combined = np.vstack([y_train_scaled, y_train_scaled])

        seed_plan = build_training_seed_plan(
            base_seed,
            "generator",
            fold_idx,
        )
        configure_torch_determinism(seed_plan["model"])
        print(
            f"    [Fold {fold_idx}] Reproducibility seeds: "
            f"model={seed_plan['model']}, "
            f"data_loader={seed_plan['data_loader']}"
        )

        train_ds = TensorDataset(
            torch.FloatTensor(y_train_combined),
            torch.FloatTensor(pca_train_combined),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=gen_cfg["training"]["batch_size"],
            shuffle=True,
            generator=make_torch_generator(seed_plan["data_loader"]),
            worker_init_fn=seed_data_loader_worker,
        )

        model = MLPWithResiduals(
            input_dim=n_params,
            width=gen_cfg["model"]["width"],
            out_dim=n_pca,
            depth=gen_cfg["model"]["depth"],
            dropout=gen_cfg["model"]["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=gen_cfg["training"]["learning_rate"]
        )
        criterion = torch.nn.MSELoss()
        scheduler = CosineAnnealingLR(
            optimizer, T_max=gen_cfg["training"]["epochs"]
        )

        _train_model(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            gen_cfg["training"]["epochs"],
            device,
        )

        metrics = _evaluate_generator(
            model, y_test_scaled, x_test_clean, prep_dir, device
        )

        for key, values in history.items():
            values.append(metrics[key])

        elapsed = time.time() - start_time
        print(f"Fold {fold_idx} | {elapsed:.0f}s | R2: {metrics['R2']:.4f}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            torch.save(
                model.state_dict(),
                os.path.join(exp_dir, gen_cfg["checkpoint"]["model"]),
            )
            copy_preprocessing_artifacts(
                prep_dir, exp_dir, gen_cfg["checkpoint"]
            )

    print("\n" + "=" * 50)
    print("GENERATOR - FINAL PERFORMANCE REPORT")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    if held_out_fold is None:
        print_final_stats("GENERATION", history)
    else:
        print(f"\n--- GENERATION (Held-out fold {held_out_fold}) ---")
        for key, values in history.items():
            print(f"  {key}: {values[0]:.4f}")
    print("=" * 50)

    return exp_dir


def main():
    """Entry point for generator training.
    Parses CLI args, loads config, and runs training."""
    parser = argparse.ArgumentParser(description="ASTRAI generator training")
    parser.add_argument("--config", default="configs/default_split.yaml")
    parser.add_argument(
        "--prep",
        default="preprocessed",
        help="Directory with preprocess.py output",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_generator_training(cfg, prep_dir=args.prep, config_path=args.config)


if __name__ == "__main__":
    main()
