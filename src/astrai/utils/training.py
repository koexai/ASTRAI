"""Shared technical contracts for ASTRAI training workflows."""

import copy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from astrai.utils.array_dtypes import load_index_array, load_model_array
from astrai.utils.metrics import METRIC_NAMES, get_metric_function
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


def load_outer_fold_indices(fold_dir):
    """Load global development and test indices from preprocessing."""
    directory = Path(fold_dir)
    try:
        return (
            load_index_array(directory / "train_idx.npy"),
            load_index_array(directory / "test_idx.npy"),
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Validation-based training requires preprocessing fold files "
            "train_idx.npy and test_idx.npy. Regenerate legacy artefacts "
            "with the current preprocessing command."
        ) from exc


def load_preprocessing_source_array(preprocessing_dir, filename):
    """Load a root source array required for exact validation evaluation."""
    try:
        return load_model_array(Path(preprocessing_dir) / filename)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Validation-based training requires the preprocessing source "
            f"array {filename}. Regenerate legacy artefacts with the current "
            "preprocessing command."
        ) from exc


def partition_precomputed_development_data(
    clean_inputs,
    augmented_inputs,
    targets,
    validation_fraction,
    validation_seed,
    minimum_validation_samples=1,
):
    """Create clean validation data and paired clean/augmented training data."""
    sample_count = len(clean_inputs)
    if len(augmented_inputs) != sample_count or len(targets) != sample_count:
        raise ValueError(
            "Clean inputs, augmented inputs and targets must contain the "
            "same development samples."
        )
    training_local, validation_local = split_development_indices(
        sample_count,
        validation_fraction,
        validation_seed,
        minimum_validation_samples,
    )
    training_inputs, training_targets = combine_clean_and_augmented(
        clean_inputs[training_local],
        augmented_inputs[training_local],
        targets[training_local],
    )
    return {
        "training_inputs": training_inputs,
        "training_targets": training_targets,
        "validation_inputs": clean_inputs[validation_local],
        "validation_targets": targets[validation_local],
        "training_local_indices": training_local,
        "validation_local_indices": validation_local,
    }


def combine_clean_and_augmented(clean_inputs, augmented_inputs, targets):
    """Pair clean and augmented inputs with duplicated training targets."""
    return (
        np.vstack([clean_inputs, augmented_inputs]),
        np.vstack([targets, targets]),
    )


def resolve_test_fold(training_cfg):
    """Return the configured outer test fold with legacy alias support."""
    has_test_fold = "test_fold" in training_cfg
    has_legacy_fold = "held_out_fold" in training_cfg
    test_fold = training_cfg.get("test_fold")
    legacy_fold = training_cfg.get("held_out_fold")
    if has_test_fold and has_legacy_fold and test_fold != legacy_fold:
        raise ValueError(
            "training.test_fold and legacy training.held_out_fold must "
            "match when both are configured."
        )
    return test_fold if has_test_fold else legacy_fold


def resolve_training_control(training_cfg, metric_path):
    """Validate and normalise validation, selection and stopping settings."""
    validation_fraction = training_cfg.get("validation_fraction", 0.1)
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, Real)
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("training.validation_fraction must be between 0 and 1.")

    selection_cfg = training_cfg.get("checkpoint_selection", {})
    if not isinstance(selection_cfg, dict):
        raise TypeError("training.checkpoint_selection must be a mapping.")
    metric = selection_cfg.get("metric", "R2")
    _, mode = get_metric_function(metric)

    stopping_cfg = training_cfg.get("early_stopping", {})
    if not isinstance(stopping_cfg, dict):
        raise TypeError("training.early_stopping must be a mapping.")
    enabled = stopping_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("training.early_stopping.enabled must be a boolean.")
    patience = stopping_cfg.get("patience", 100)
    if isinstance(patience, bool) or not isinstance(patience, Integral):
        raise TypeError("training.early_stopping.patience must be an integer.")
    if patience < 1:
        raise ValueError("training.early_stopping.patience must be at least 1.")
    min_delta = stopping_cfg.get("min_delta", 0.0)
    if (
        isinstance(min_delta, bool)
        or not isinstance(min_delta, Real)
        or float(min_delta) < 0.0
    ):
        raise ValueError(
            "training.early_stopping.min_delta must be a non-negative number."
        )

    return {
        "validation_fraction": float(validation_fraction),
        "selection_policy": {
            "dataset": "validation",
            "metric": metric,
            "metric_path": metric_path,
            "mode": mode,
        },
        "early_stopping": {
            "enabled": enabled,
            "patience": int(patience),
            "min_delta": float(min_delta),
        },
    }


def split_development_indices(
    n_samples,
    validation_fraction,
    seed,
    minimum_validation_samples=1,
):
    """Split one outer-fold development pool into train and validation rows."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        raise TypeError("Development sample count must be an integer.")
    if n_samples < 2:
        raise ValueError(
            "At least two development samples are required for validation."
        )
    validation_size = max(
        int(minimum_validation_samples),
        int(np.ceil(n_samples * validation_fraction)),
    )
    if validation_size >= n_samples:
        raise ValueError(
            "validation_fraction leaves no samples for fold training."
        )
    permutation = np.random.default_rng(seed).permutation(n_samples)
    validation_indices = np.sort(permutation[:validation_size])
    training_indices = np.sort(permutation[validation_size:])
    return training_indices, validation_indices


@dataclass
class ValidationSelectionTracker:
    """Track strict best-checkpoint selection and independent patience state."""

    metric: str
    mode: str
    early_stopping_enabled: bool = False
    patience: int = 100
    min_delta: float = 0.0

    def __post_init__(self):
        self.selected_epoch = None
        self.selected_score = None
        self.selected_state = None
        self.early_stopping_reference_score = None
        self.epochs_without_significant_improvement = 0
        self.trace = []

    def _strictly_better(self, score, reference):
        return score > reference if self.mode == "max" else score < reference

    def _significantly_better(self, score, reference):
        if self.mode == "max":
            return score > reference + self.min_delta
        return score < reference - self.min_delta

    def observe(self, epoch, score, model, training_loss, learning_rate):
        """Observe one epoch and return whether patience is exhausted."""
        score = float(score)
        if not np.isfinite(score):
            raise ValueError(
                f"Validation {self.metric} is not finite at epoch {epoch}."
            )

        selected = self.selected_score is None or self._strictly_better(
            score,
            self.selected_score,
        )
        if selected:
            self.selected_epoch = int(epoch)
            self.selected_score = score
            self.selected_state = {
                name: (
                    value.detach().cpu().clone()
                    if isinstance(value, torch.Tensor)
                    else copy.deepcopy(value)
                )
                for name, value in model.state_dict().items()
            }

        significant = (
            self.early_stopping_reference_score is None
            or self._significantly_better(
                score,
                self.early_stopping_reference_score,
            )
        )
        if significant:
            self.early_stopping_reference_score = score
            self.epochs_without_significant_improvement = 0
        else:
            self.epochs_without_significant_improvement += 1

        self.trace.append(
            {
                "epoch": int(epoch),
                "training_loss": float(training_loss),
                "validation_selection_score": score,
                "learning_rate": float(learning_rate),
                "selected_checkpoint": selected,
            }
        )
        return (
            self.early_stopping_enabled
            and self.epochs_without_significant_improvement >= self.patience
        )

    def restore_selected_checkpoint(self, model):
        """Restore the strictly best validation epoch into ``model``."""
        if self.selected_state is None:
            raise RuntimeError("No validation epoch was observed.")
        model.load_state_dict(self.selected_state)

    def result(self, epochs_completed, stopped_early):
        """Return serialisable training-control results for one fold."""
        return {
            "selected_epoch": self.selected_epoch,
            "selected_validation_score": self.selected_score,
            "epochs_completed": int(epochs_completed),
            "stopped_early": bool(stopped_early),
            "trace": self.trace,
        }


def build_validation_selection_tracker(training_control):
    """Build a tracker from a normalised training-control contract."""
    policy = training_control["selection_policy"]
    stopping = training_control["early_stopping"]
    return ValidationSelectionTracker(
        metric=policy["metric"],
        mode=policy["mode"],
        early_stopping_enabled=stopping["enabled"],
        patience=stopping["patience"],
        min_delta=stopping["min_delta"],
    )


def assert_selected_validation_score(
    selected_score,
    complete_validation_metrics,
    metric_path,
    *,
    rtol=1e-6,
    atol=1e-7,
):
    """Ensure lightweight selection and complete validation agree."""
    value = complete_validation_metrics
    for component in metric_path.split("."):
        value = value[component]
    complete_score = float(value)
    if not np.isclose(
        selected_score,
        complete_score,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        raise ValueError(
            "The selected validation score does not match the same metric "
            "from complete validation evaluation: "
            f"{selected_score} != {complete_score}."
        )
    return complete_score


def is_strictly_better_score(candidate, reference, mode):
    """Return whether a score strictly improves on an existing score."""
    if reference is None:
        return True
    return candidate > reference if mode == "max" else candidate < reference


def create_metric_history():
    """Return an empty history in the canonical metric order."""
    return {metric_name: [] for metric_name in METRIC_NAMES}


def record_metric_values(history, metrics):
    """Append one fold's scalar metrics to an existing history."""
    for metric_name, values in history.items():
        values.append(metrics[metric_name])


def build_training_loader(
    inputs,
    targets,
    batch_size,
    seed,
    *,
    reconstruction_targets=None,
):
    """Create the deterministic DataLoader shared by training stages.

    Unified training supplies explicit clean reconstruction targets as a
    third tensor. Split training retains the existing two-tensor batches.
    """
    tensors = [torch.FloatTensor(inputs), torch.FloatTensor(targets)]
    if reconstruction_targets is not None:
        tensors.append(torch.FloatTensor(reconstruction_targets))
    dataset = TensorDataset(*tensors)
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
    validation_score_fn=None,
    selection_tracker=None,
):
    """Train a supervised model and optionally restore its best epoch."""
    epochs_completed = 0
    stopped_early = False
    for epoch in range(epochs):
        model.train()
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
        epochs_completed = epoch + 1
        average_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        if validation_score_fn is not None:
            if selection_tracker is None:
                raise ValueError(
                    "selection_tracker is required with validation_score_fn."
                )
            validation_score = validation_score_fn(model)
            should_stop = selection_tracker.observe(
                epoch + 1,
                validation_score,
                model,
                average_loss,
                current_lr,
            )
        else:
            should_stop = False
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {average_loss:.6f} "
                f"| LR: {current_lr:.2e}"
            )
        if should_stop:
            stopped_early = True
            print(
                f"Early stopping after epoch {epoch + 1}; restoring the "
                f"best validation epoch {selection_tracker.selected_epoch}."
            )
            break

    if selection_tracker is None:
        return None
    selection_tracker.restore_selected_checkpoint(model)
    return selection_tracker.result(epochs_completed, stopped_early)


def print_metric_history(name, history, test_fold=None):
    """Print one test-fold result or aggregate statistics across folds."""
    if test_fold is not None:
        print(f"\n--- {name} (Test fold {test_fold}) ---")
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
