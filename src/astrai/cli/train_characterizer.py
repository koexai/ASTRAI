"""
train_characterizer.py - K-Fold training of the characterization branch (curves -> params).

Loads pre-computed preprocessing artifacts from ``preprocess.py`` and trains
the SplitMLPRegressor with its own hyperparameters.

Usage::

    astrai train-characterizer
    astrai train-characterizer --config configs/default_split.yaml --prep preprocessed
"""
import argparse
import os
import time

import joblib
import numpy as np
import torch

from astrai.models.factories import build_characterizer
from astrai.utils.metrics import (
    METRIC_NAMES,
    compute_parameter_metrics,
)
from astrai.utils.parameter_validation import validate_parameter_names
from astrai.utils.checkpoints import (
    save_split_checkpoint,
)
from astrai.utils.configuration import load_config
from astrai.utils.fold_selection import resolve_fold_indices
from astrai.utils.log_experiments import ExperimentRun, summarise_metric_history
from astrai.utils.reproducibility import (
    build_training_seed_plan,
    configure_torch_determinism,
)
from astrai.utils.training import (
    build_training_components,
    build_training_loader,
    combine_clean_and_augmented,
    create_metric_history,
    load_fold_arrays,
    print_metric_history,
    record_metric_values,
    select_training_device,
    train_supervised_model,
)
from astrai.paths import resolve_config_path, resolve_user_path


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
    return load_fold_arrays(
        fold_dir,
        (
            "x_train_clean_pca.npy",
            "x_train_aug_pca.npy",
            "x_test_pca.npy",
            "y_train_scaled.npy",
            "y_test.npy",
        ),
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


def _summarise_parameter_history(history):
    """Return serialisable metric summaries in parameter order."""
    return {
        name: summarise_metric_history(metric_history)
        for name, metric_history in history.items()
    }


def run_characterizer_training(
    cfg,
    prep_dir="preprocessed",
    exp_dir=None,
    config_path=None,
    pipeline_run_id=None,
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
    config_path : str or None
        Source config path to record. The effective config is always saved.
    pipeline_run_id : str or None
        Identifier shared with a generator run from the same split pipeline.

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
    base_seed = cfg["preprocessing"]["random_seed"]
    char_cfg = cfg["characterizer"]

    held_out_fold = char_cfg["training"].get("held_out_fold")
    fold_indices = resolve_fold_indices(held_out_fold, n_splits)
    prep_dir = resolve_user_path(prep_dir)

    device = select_training_device()

    experiment = ExperimentRun.start(
        stage="characterizer",
        config=cfg,
        config_path=config_path,
        exp_dir=exp_dir,
        base_dir="experiments/characterizer",
        preprocessing_dir=prep_dir,
        pipeline_run_id=pipeline_run_id,
        folds=fold_indices,
        base_seed=base_seed,
        device=device,
        checkpoint_metric="aggregate.R2",
    )
    exp_dir = str(experiment.directory)
    print(f"Characterizer experiment directory: {exp_dir}")

    history = create_metric_history()
    parameter_history = _initialise_parameter_history(param_names)
    best_r2 = -np.inf

    try:
        print(
            "Starting Characterizer Training (SplitMLP) with PCA "
            f"({n_pca} components)..."
        )

        if held_out_fold is None:
            print(f"Training all {n_splits} folds.")
        else:
            print(f"Training split with held-out fold {held_out_fold}.")

        for fold_idx in fold_indices:
            start_time = time.time()
            fold_dir = prep_dir / f"fold_{fold_idx}"

            (
                x_train_clean_pca,
                x_train_aug_pca,
                x_test_pca,
                y_train_scaled,
                y_test,
            ) = _load_fold_data(fold_dir)
            print(
                f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}"
            )

            x_train_combined, y_train_combined = (
                combine_clean_and_augmented(
                    x_train_clean_pca,
                    x_train_aug_pca,
                    y_train_scaled,
                )
            )

            seed_plan = build_training_seed_plan(
                base_seed,
                "characterizer",
                fold_idx,
            )
            configure_torch_determinism(seed_plan["model"])
            experiment.record_execution_environment(device=device)
            print(
                f"    [Fold {fold_idx}] Reproducibility seeds: "
                f"model={seed_plan['model']}, "
                f"data_loader={seed_plan['data_loader']}"
            )

            train_loader = build_training_loader(
                x_train_combined,
                y_train_combined,
                batch_size=char_cfg["training"]["batch_size"],
                seed=seed_plan["data_loader"],
            )

            model = build_characterizer(cfg).to(device)

            optimizer, criterion, scheduler = build_training_components(
                model,
                char_cfg["training"],
            )

            train_supervised_model(
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

            record_metric_values(history, metrics)
            _record_parameter_metrics(
                parameter_history,
                evaluation["per_parameter"],
            )
            _print_parameter_metrics(evaluation["per_parameter"])

            elapsed = time.time() - start_time
            print(
                f"Fold {fold_idx} | {elapsed:.0f}s | "
                f"R2: {metrics['R2']:.4f}"
            )
            experiment.record_fold(
                fold_idx,
                evaluation,
                seed_plan,
                elapsed,
            )

            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                checkpoint_paths = save_split_checkpoint(
                    exp_dir,
                    char_cfg["checkpoint"],
                    model,
                    prep_dir,
                )
                experiment.record_checkpoint(
                    fold_idx,
                    best_r2,
                    checkpoint_paths,
                )

        print("\n" + "=" * 50)
        print("CHARACTERIZER - FINAL PERFORMANCE REPORT")
        print(f"PCA Components: {n_pca}")
        print("=" * 50)
        print_metric_history(
            "CHARACTERIZATION",
            history,
            held_out_fold=held_out_fold,
        )
        _print_parameter_final_stats(parameter_history, held_out_fold)
        print("=" * 50)

        experiment.complete(
            {
                "aggregate": summarise_metric_history(history),
                "per_parameter": _summarise_parameter_history(
                    parameter_history
                ),
            }
        )
    except BaseException as exc:
        experiment.fail(exc)
        raise

    return exp_dir


def main(argv=None):
    """Entry point for characterizer training.
    Parses CLI args, loads config, and runs training."""
    parser = argparse.ArgumentParser(
        prog="astrai train-characterizer",
        description="ASTRAI characterizer training"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML (default: packaged default_split.yaml)",
    )
    parser.add_argument(
        "--prep",
        default="preprocessed",
        help="Directory with preprocess.py output",
    )
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config, "default_split.yaml")

    cfg = load_config(config_path)

    run_characterizer_training(
        cfg,
        prep_dir=args.prep,
        config_path=str(config_path),
    )


if __name__ == "__main__":
    main()
