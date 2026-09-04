"""
train_characterizer.py - K-Fold training of the characterization branch (curves -> params).

Loads pre-computed preprocessing artifacts from ``preprocess.py`` and trains
the SplitMLPRegressor with its own hyperparameters.

Usage::

    python train_characterizer.py
    python train_characterizer.py --config configs/default_split.yaml --prep preprocessed
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
from models.split_mlp import SplitMLPRegressor
from utils.array_dtypes import load_model_array
from utils.metrics import (
    METRIC_NAMES,
    compute_parameter_metrics,
)
from utils.parameter_validation import validate_parameter_names
from utils.checkpoints import copy_preprocessing_artifacts
from utils.fold_selection import resolve_fold_indices
from utils.log_experiments import create_experiment_dir, save_code, save_config


def _load_fold_data(fold_dir):
    """Load preprocessed arrays for a single fold.
    Expects the following files in fold_dir:
    - x_train_clean_pca.npy
    - x_train_aug_pca.npy
    - x_test_pca.npy
    - y_train_scaled.npy
    - y_test.npy
    Returns:
    - x_train_clean_pca: (n_train_clean, n_pca)
    - x_train_aug_pca: (n_train_aug, n_pca)
    - x_test_pca: (n_test, n_pca)
    - y_train_scaled: (n_train, n_params)
    - y_test: (n_test, n_params)"""
    x_train_clean_pca = load_model_array(
        os.path.join(fold_dir, "x_train_clean_pca.npy")
    )
    x_train_aug_pca = load_model_array(
        os.path.join(fold_dir, "x_train_aug_pca.npy")
    )
    x_test_pca = load_model_array(os.path.join(fold_dir, "x_test_pca.npy"))
    y_train_scaled = load_model_array(
        os.path.join(fold_dir, "y_train_scaled.npy")
    )
    y_test = load_model_array(os.path.join(fold_dir, "y_test.npy"))
    return (
        x_train_clean_pca,
        x_train_aug_pca,
        x_test_pca,
        y_train_scaled,
        y_test,
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
    Prints training loss every 10 epochs."""
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
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg:.6f} | LR: {cur_lr:.2e}"
            )


def _evaluate_characterizer(
    model, x_test_pca, y_test, param_names, prep_dir, device
):
    """Evaluate the characterizer and return aggregate and named metrics.

    Loads the y_scaler from prep_dir to inverse transform predictions.
    model: the trained characterizer model
    x_test_pca: (n_test, n_pca) PCA-transformed test curves
    y_test: (n_test, n_params) true parameters for the test set
    param_names: ordered names of the predicted parameters
    prep_dir: directory containing preprocessing artifacts (expects y_scaler.pkl)
    device: torch.device to run on
    Returns aggregate and per-parameter RMSE, RRMSE, MAE and R2 values.
    """
    y_scaler = joblib.load(os.path.join(prep_dir, "y_scaler.pkl"))

    model.eval()
    with torch.no_grad():
        x_test_t = torch.FloatTensor(x_test_pca).to(device)
        pred_sc = model(x_test_t).cpu().numpy()
        pred_params = y_scaler.inverse_transform(pred_sc)

    return compute_parameter_metrics(y_test, pred_params, param_names)


def _initialise_parameter_history(param_names):
    """Create an empty metric history for every configured parameter."""
    return {
        name: {metric_name: [] for metric_name in METRIC_NAMES}
        for name in param_names
    }


def _record_parameter_metrics(history, per_parameter):
    """Append one fold's named metrics to the per-parameter history."""
    for name, metric_history in history.items():
        for metric_name in METRIC_NAMES:
            metric_history[metric_name].append(
                per_parameter[name][metric_name]
            )


def _print_parameter_metrics(per_parameter):
    """Print one fold's metrics in configured parameter order."""
    print("    Per-parameter metrics:")
    for name, metrics in per_parameter.items():
        values = " | ".join(
            f"{metric_name}={metrics[metric_name]:.6f}"
            for metric_name in METRIC_NAMES
        )
        print(f"      {name}: {values}")


def _print_parameter_final_stats(history, held_out_fold):
    """Print per-parameter values for one fold or statistics across folds."""
    if held_out_fold is None:
        n_folds = len(next(iter(history.values()))[METRIC_NAMES[0]])
        print(f"\n--- PER-PARAMETER CHARACTERIZATION ({n_folds}-Fold Mean) ---")
        for name, metric_history in history.items():
            print(f"  {name}:")
            for metric_name in METRIC_NAMES:
                values = metric_history[metric_name]
                print(
                    f"    {metric_name}: {np.mean(values):.4f}  "
                    f"(+/- {np.std(values):.4f})"
                )
        return

    print(
        f"\n--- PER-PARAMETER CHARACTERIZATION "
        f"(Held-out fold {held_out_fold}) ---"
    )
    for name, metric_history in history.items():
        values = " | ".join(
            f"{metric_name}={metric_history[metric_name][0]:.6f}"
            for metric_name in METRIC_NAMES
        )
        print(f"  {name}: {values}")


def run_characterizer_training(
    cfg,
    prep_dir="preprocessed",
    exp_dir=None,
    config_path="configs/default_split.yaml",
):
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
    data_cfg = cfg["data"]
    n_params = data_cfg["n_params"]
    param_names = validate_parameter_names(
        n_params,
        data_cfg.get("param_names"),
    )
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]
    char_cfg = cfg["characterizer"]

    held_out_fold = char_cfg["training"].get("held_out_fold")
    fold_indices = resolve_fold_indices(held_out_fold, n_splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if exp_dir is None:
        exp_dir = create_experiment_dir(base_dir="experiments/characterizer")
        save_code(exp_dir)
        save_config(exp_dir, config_path=config_path)
    print(f"Characterizer experiment directory: {exp_dir}")

    history = {metric_name: [] for metric_name in METRIC_NAMES}
    parameter_history = _initialise_parameter_history(param_names)
    best_r2 = -np.inf

    print(
        f"Starting Characterizer Training (SplitMLP) with PCA ({n_pca} components)..."
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
            x_test_pca,
            y_train_scaled,
            y_test,
        ) = _load_fold_data(fold_dir)
        print(f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}")

        x_train_combined = np.vstack([x_train_clean_pca, x_train_aug_pca])
        y_train_combined = np.vstack([y_train_scaled, y_train_scaled])

        train_ds = TensorDataset(
            torch.FloatTensor(x_train_combined),
            torch.FloatTensor(y_train_combined),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=char_cfg["training"]["batch_size"],
            shuffle=True,
        )

        model = SplitMLPRegressor(
            input_dim=n_pca,
            width=char_cfg["model"]["width"],
            num_params=n_params,
            depth=char_cfg["model"]["depth"],
            dropout=char_cfg["model"]["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=char_cfg["training"]["learning_rate"]
        )
        criterion = torch.nn.MSELoss()
        scheduler = CosineAnnealingLR(
            optimizer, T_max=char_cfg["training"]["epochs"]
        )

        _train_model(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            char_cfg["training"]["epochs"],
            device,
        )

        evaluation = _evaluate_characterizer(
            model, x_test_pca, y_test, param_names, prep_dir, device
        )
        metrics = evaluation["aggregate"]

        for key, values in history.items():
            values.append(metrics[key])
        _record_parameter_metrics(
            parameter_history,
            evaluation["per_parameter"],
        )
        _print_parameter_metrics(evaluation["per_parameter"])

        elapsed = time.time() - start_time
        print(f"Fold {fold_idx} | {elapsed:.0f}s | R2: {metrics['R2']:.4f}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            torch.save(
                model.state_dict(),
                os.path.join(exp_dir, char_cfg["checkpoint"]["model"]),
            )
            copy_preprocessing_artifacts(
                prep_dir, exp_dir, char_cfg["checkpoint"]
            )

    print("\n" + "=" * 50)
    print("CHARACTERIZER - FINAL PERFORMANCE REPORT")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    if held_out_fold is None:
        print_final_stats("CHARACTERIZATION", history)
    else:
        print(
            f"\n--- CHARACTERIZATION "
            f"(Held-out fold {held_out_fold}) ---"
        )
        for key, values in history.items():
            print(f"  {key}: {values[0]:.4f}")
    _print_parameter_final_stats(parameter_history, held_out_fold)
    print("=" * 50)

    return exp_dir


def main():
    """Entry point for characterizer training.
    Parses CLI args, loads config, and runs training."""
    parser = argparse.ArgumentParser(
        description="ASTRAI characterizer training"
    )
    parser.add_argument("--config", default="configs/default_split.yaml")
    parser.add_argument(
        "--prep",
        default="preprocessed",
        help="Directory with preprocess.py output",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_characterizer_training(cfg, prep_dir=args.prep, config_path=args.config)


if __name__ == "__main__":
    main()
