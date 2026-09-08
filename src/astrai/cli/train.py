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

    astrai train                              # uses configs/default.yaml
    astrai train --config configs/4par.yaml   # 4-parameter synthetic dataset
"""
import argparse
import time

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from astrai.models.factories import build_unified_model
from astrai.utils.metrics import get_rmse, get_mae, get_r_squared, get_rrmse
from astrai.utils.checkpoints import (
    checkpoint_artefact_paths,
    load_config,
    load_data,
    save_model_checkpoint,
)
from astrai.utils.augmentation import apply_lsst_pipeline
from astrai.utils.log_experiments import ExperimentRun, summarise_metric_history
from astrai.utils.reproducibility import (
    build_training_seed_plan,
    build_unified_preprocessing_seed_plan,
    configure_torch_determinism,
    make_numpy_rng,
)
from astrai.utils.training import (
    build_training_components,
    build_training_loader,
    create_metric_history,
    print_metric_history,
    record_metric_values,
    select_training_device,
)
from astrai.paths import resolve_config_path


def print_final_stats(name, history):
    """Compatibility wrapper for aggregate fold reporting."""
    print_metric_history(name, history)


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
    augmentation_seed,
    pca_seed,
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
    augmentation_seed : int
        Seed for the fold-local augmentation stream.
    pca_seed : int
        Seed passed to the fold-local PCA.
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
        rng=make_numpy_rng(augmentation_seed),
    )

    x_scaler = StandardScaler()
    x_scaler.fit(x_train_clean)
    x_train_clean_scaled = x_scaler.transform(x_train_clean)
    x_train_aug_scaled = x_scaler.transform(x_train_aug)
    x_test_scaled = x_scaler.transform(x_test_clean)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    pca = PCA(n_components=n_pca, random_state=pca_seed)
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


def _execute_unified_training(cfg, experiment, device):
    """Run unified cross-validation inside an initialised experiment.

    Steps:
    1. Load the raw data (curves and parameters).
    2. For each fold in K-Fold cross-validation:
       a. Preprocess the fold's data (augmentation, scaling, PCA).
       b. Instantiate the UnifiedModel and optimizer.
       c. Train the model on the training fold.
       d. Evaluate on the test fold and record metrics.
       e. Save the model checkpoint if it has the best characterization R2 so far.
    3. Report and persist aggregate statistics across folds.
    """
    exp_dir = str(experiment.directory)
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

    history_char = create_metric_history()
    history_gen = create_metric_history()
    best_global_r2 = -np.inf

    print(f"Starting Training Char + Gen with PCA ({n_pca} components)...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_raw), 1):
        start_time = time.time()

        preprocessing_seed_plan = build_unified_preprocessing_seed_plan(
            train_cfg["random_seed"],
            fold_idx,
        )
        training_seed_plan = build_training_seed_plan(
            train_cfg["random_seed"],
            "unified",
            fold_idx,
        )
        print(
            f"    [Fold {fold_idx}] Reproducibility seeds: "
            f"augmentation={preprocessing_seed_plan['augmentation']}, "
            f"pca={preprocessing_seed_plan['pca']}, "
            f"model={training_seed_plan['model']}, "
            f"data_loader={training_seed_plan['data_loader']}"
        )

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
            preprocessing_seed_plan["augmentation"],
            preprocessing_seed_plan["pca"],
        )

        configure_torch_determinism(training_seed_plan["model"])
        experiment.record_execution_environment(device=device)

        train_loader = build_training_loader(
            x_train_combined,
            y_train_combined,
            batch_size=train_cfg["batch_size"],
            seed=training_seed_plan["data_loader"],
        )

        model = build_unified_model(cfg).to(device)

        optimizer, criterion, scheduler = build_training_components(
            model,
            train_cfg,
        )

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

        record_metric_values(history_char, char_m)
        record_metric_values(history_gen, gen_m)

        elapsed = time.time() - start_time
        print(
            f"Fold {fold_idx} | {elapsed:.0f}s |",
            f"Char R2: {char_m['R2']:.4f} | Gen R2: {gen_m['R2']:.4f}",
        )
        experiment.record_fold(
            fold_idx,
            {
                "characterization": char_m,
                "generation": gen_m,
            },
            {
                "k_fold": train_cfg["random_seed"],
                "preprocessing": preprocessing_seed_plan,
                "training": training_seed_plan,
            },
            elapsed,
        )

        if char_m["R2"] > best_global_r2:
            best_global_r2 = char_m["R2"]
            save_model_checkpoint(
                exp_dir, cfg["checkpoint"], model, x_scaler, y_scaler, pca
            )
            experiment.record_checkpoint(
                fold_idx,
                best_global_r2,
                checkpoint_artefact_paths(exp_dir, cfg["checkpoint"]),
            )

    print("\n" + "=" * 50)
    print("FINAL PERFORMANCE REPORT (Un-scaled metrics)")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    print_metric_history("CHARACTERIZATION", history_char)
    print_metric_history("GENERATION", history_gen)
    print("=" * 50)

    experiment.complete(
        {
            "characterization": summarise_metric_history(history_char),
            "generation": summarise_metric_history(history_gen),
        }
    )


def run_unified_training(
    cfg,
    exp_dir=None,
    config_path=None,
):
    """Train the unified model and return its isolated experiment directory."""
    train_cfg = cfg["training"]
    device = select_training_device()
    experiment = ExperimentRun.start(
        stage="unified",
        config=cfg,
        config_path=config_path,
        exp_dir=exp_dir,
        base_dir="experiments",
        folds=range(1, train_cfg["n_splits"] + 1),
        base_seed=train_cfg["random_seed"],
        device=device,
        checkpoint_metric="characterization.R2",
    )
    try:
        _execute_unified_training(cfg, experiment, device)
    except BaseException as exc:
        experiment.fail(exc)
        raise
    return str(experiment.directory)


def main(argv=None):
    """Parse CLI arguments and run unified model training."""
    parser = argparse.ArgumentParser(
        prog="astrai train",
        description="ASTRAI unified model training"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML (default: packaged default.yaml)",
    )
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config, "default.yaml")
    cfg = load_config(config_path)
    run_unified_training(cfg, config_path=str(config_path))


if __name__ == "__main__":
    main()
