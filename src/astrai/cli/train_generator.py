"""
train_generator.py - K-Fold training of the generation branch (params -> curves).

Loads pre-computed preprocessing artifacts from ``preprocess.py`` and trains
the MLPWithResiduals with its own hyperparameters.

Usage::

    astrai train-generator
    astrai train-generator --config configs/default_split.yaml --prep preprocessed
"""
import argparse
import os
import time

import joblib
import numpy as np
import torch

from astrai.models.factories import build_generator
from astrai.utils.metrics import get_rmse, get_mae, get_r_squared, get_rrmse
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
    return load_fold_arrays(
        fold_dir,
        (
            "x_train_clean_pca.npy",
            "x_train_aug_pca.npy",
            "y_train_scaled.npy",
            "y_test_scaled.npy",
            "x_test_clean.npy",
        ),
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
    config_path=None,
    pipeline_run_id=None,
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
    config_path : str or None
        Source config path to record. The effective config is always saved.
    pipeline_run_id : str or None
        Identifier shared with a characterizer run from the same pipeline.

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
    prep_dir = resolve_user_path(prep_dir)

    device = select_training_device()

    experiment = ExperimentRun.start(
        stage="generator",
        config=cfg,
        config_path=config_path,
        exp_dir=exp_dir,
        base_dir="experiments/generator",
        preprocessing_dir=prep_dir,
        pipeline_run_id=pipeline_run_id,
        folds=fold_indices,
        base_seed=base_seed,
        device=device,
    )
    exp_dir = str(experiment.directory)
    print(f"Generator experiment directory: {exp_dir}")

    history = create_metric_history()
    best_r2 = -np.inf

    try:
        print(
            "Starting Generator Training (MLPWithResiduals) with PCA "
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
                y_train_scaled,
                y_test_scaled,
                x_test_clean,
            ) = _load_fold_data(fold_dir)
            print(
                f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}"
            )

            pca_train_combined, y_train_combined = (
                combine_clean_and_augmented(
                    x_train_clean_pca,
                    x_train_aug_pca,
                    y_train_scaled,
                )
            )

            seed_plan = build_training_seed_plan(
                base_seed,
                "generator",
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
                y_train_combined,
                pca_train_combined,
                batch_size=gen_cfg["training"]["batch_size"],
                seed=seed_plan["data_loader"],
            )

            model = build_generator(cfg).to(device)

            optimizer, criterion, scheduler = build_training_components(
                model,
                gen_cfg["training"],
            )

            train_supervised_model(
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

            record_metric_values(history, metrics)

            elapsed = time.time() - start_time
            print(
                f"Fold {fold_idx} | {elapsed:.0f}s | "
                f"R2: {metrics['R2']:.4f}"
            )
            experiment.record_fold(
                fold_idx,
                metrics,
                seed_plan,
                elapsed,
            )

            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                checkpoint_paths = save_split_checkpoint(
                    exp_dir,
                    gen_cfg["checkpoint"],
                    model,
                    prep_dir,
                )
                experiment.record_checkpoint(
                    fold_idx,
                    best_r2,
                    checkpoint_paths,
                )

        print("\n" + "=" * 50)
        print("GENERATOR - FINAL PERFORMANCE REPORT")
        print(f"PCA Components: {n_pca}")
        print("=" * 50)
        print_metric_history(
            "GENERATION",
            history,
            held_out_fold=held_out_fold,
        )
        print("=" * 50)

        experiment.complete(summarise_metric_history(history))
    except BaseException as exc:
        experiment.fail(exc)
        raise

    return exp_dir


def main(argv=None):
    """Entry point for generator training.
    Parses CLI args, loads config, and runs training."""
    parser = argparse.ArgumentParser(
        prog="astrai train-generator",
        description="ASTRAI generator training",
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

    run_generator_training(
        cfg,
        prep_dir=args.prep,
        config_path=str(config_path),
    )


if __name__ == "__main__":
    main()
