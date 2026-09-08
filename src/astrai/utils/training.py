"""Shared technical contracts for ASTRAI training workflows."""

from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from astrai.utils.array_dtypes import load_model_array
from astrai.utils.metrics import METRIC_NAMES
from astrai.utils.reproducibility import (
    make_torch_generator,
    seed_data_loader_worker,
)


def select_training_device():
    """Select the historical CUDA-or-CPU training device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fold_arrays(fold_dir, filenames):
    """Load named preprocessing arrays using the model dtype contract."""
    directory = Path(fold_dir)
    return tuple(load_model_array(directory / filename) for filename in filenames)


def combine_clean_and_augmented(clean_inputs, augmented_inputs, targets):
    """Pair clean and augmented inputs with duplicated training targets."""
    return (
        np.vstack([clean_inputs, augmented_inputs]),
        np.vstack([targets, targets]),
    )


def create_metric_history():
    """Return an empty history in the canonical metric order."""
    return {metric_name: [] for metric_name in METRIC_NAMES}


def record_metric_values(history, metrics):
    """Append one fold's scalar metrics to an existing history."""
    for metric_name, values in history.items():
        values.append(metrics[metric_name])


def build_training_loader(inputs, targets, batch_size, seed):
    """Create the deterministic DataLoader shared by training stages."""
    dataset = TensorDataset(
        torch.FloatTensor(inputs),
        torch.FloatTensor(targets),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=make_torch_generator(seed),
        worker_init_fn=seed_data_loader_worker,
    )


def build_training_components(model, training_cfg):
    """Create the existing Adam, MSE and cosine-scheduler combination."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
    )
    criterion = torch.nn.MSELoss()
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_cfg["epochs"],
    )
    return optimizer, criterion, scheduler


def train_supervised_model(
    model,
    train_loader,
    optimizer,
    criterion,
    scheduler,
    epochs,
    device,
):
    """Run the common single-input, single-target training loop."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            average_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {average_loss:.6f} "
                f"| LR: {current_lr:.2e}"
            )


def print_metric_history(name, history, held_out_fold=None):
    """Print one held-out result or aggregate statistics across folds."""
    if held_out_fold is not None:
        print(f"\n--- {name} (Held-out fold {held_out_fold}) ---")
        for metric_name, values in history.items():
            print(f"  {metric_name}: {values[0]:.4f}")
        return

    n_folds = len(next(iter(history.values())))
    print(f"\n--- {name} ({n_folds}-Fold Mean) ---")
    for metric_name, values in history.items():
        print(
            f"  {metric_name}: {np.mean(values):.4f}  "
            f"(+/- {np.std(values):.4f})"
        )
