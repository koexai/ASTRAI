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
   e. Select an epoch on validation and evaluate once on the test fold.
4. Report test statistics and persist the validation-selected checkpoint.

Usage::

    astrai train                              # uses configs/default.yaml
    astrai train --config configs/4par.yaml   # 4-parameter synthetic dataset
"""
import argparse
import time

import numpy as np
import torch

from astrai.models.factories import build_unified_model
from astrai.utils.metrics import (
    METRIC_NAMES,
    compute_selection_metric,
    compute_target_metrics,
    get_rmse,
    get_mae,
    get_r_squared,
    get_rrmse,
)
from astrai.utils.checkpoints import (
    checkpoint_artefact_paths,
    load_config,
    save_model_checkpoint,
)
from astrai.utils.data import load_raw_data
from astrai.utils.parameter_validation import validate_parameter_names
from astrai.utils.target_transformations import (
    physical_to_transformed,
    scaled_to_transformed,
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
    assert_selected_validation_score,
    build_training_components,
    build_training_loader,
    build_validation_selection_tracker,
    create_metric_history,
    is_strictly_better_score,
    print_metric_history,
    record_metric_values,
    resolve_training_control,
    select_training_device,
)
from astrai.utils.partitions import (
    Partition, holdout_partition, outer_partitions, partition_seed, resolve_validation_fraction, selection_plan,
)
from astrai.utils.preprocessing import fit_preprocessing, dataset_identity, validate_pca_size
from astrai.paths import resolve_config_path


def print_final_stats(name, history):
    """Compatibility wrapper for aggregate fold reporting."""
    print_metric_history(name, history)


def _preprocess_fold(
    x_train_clean,
    x_validation_clean,
    x_test_clean,
    y_train,
    y_validation,
    y_test,
    n_pca,
    noise_std,
    n_days,
    samples_per_day,
    fold_idx,
    augmentation_seed,
    pca_seed,
    bundle=None,
):
    """Augment, scale, and PCA-transform a single fold's data.
    Parameters
    ----------
    x_train_clean : np.ndarray
        Clean training curves, shape (n_train_clean, n_timepoints).
    x_validation_clean : np.ndarray
        Clean validation curves used for model selection.
    x_test_clean : np.ndarray
        Clean test curves reserved for final performance estimation.
    y_train : np.ndarray
        Training parameters, shape (n_train, n_params).
    y_validation : np.ndarray
        Validation parameters in transformed target space.
    y_test : np.ndarray
        Test parameters in transformed target space.
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
    x_train_clean_targets : np.ndarray
        PCA-transformed clean curve targets, duplicated to align with the
        clean and augmented Characterizer inputs.
    x_validation_pca, y_validation_scaled : np.ndarray
        Model-ready validation inputs and parameters.
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

    if bundle is None:
        # Compatibility for direct callers: local row identity, same fit boundary.
        curves = np.vstack([x_train_clean, x_validation_clean, x_test_clean])
        parameters = np.vstack([y_train, y_validation, y_test])
        n_train, n_val = len(x_train_clean), len(x_validation_clean)
        local = Partition(len(curves), tuple(range(n_train + n_val)),
                          tuple(range(n_train)), tuple(range(n_train, n_train + n_val)),
                          tuple(range(n_train + n_val, len(curves))),
                          outer_fold=fold_idx)
        bundle = fit_preprocessing(curves, parameters, local, n_pca, pca_seed)
    x_scaler, y_scaler, pca = bundle.x_scaler, bundle.y_scaler, bundle.pca
    x_train_clean_pca = bundle.transform_curves(x_train_clean)
    x_train_aug_pca = bundle.transform_curves(x_train_aug)
    x_validation_pca = bundle.transform_curves(x_validation_clean)
    x_test_pca = bundle.transform_curves(x_test_clean)
    y_train_scaled = bundle.transform_parameters(y_train)
    y_validation_scaled = bundle.transform_parameters(y_validation)
    y_test_scaled = bundle.transform_parameters(y_test)

    x_train_combined = np.vstack([x_train_clean_pca, x_train_aug_pca])
    y_train_combined = np.vstack([y_train_scaled, y_train_scaled])
    x_train_clean_targets = np.vstack(
        [x_train_clean_pca, x_train_clean_pca]
    )

    return (
        x_train_combined,
        y_train_combined,
        x_train_clean_targets,
        x_validation_pca,
        y_validation_scaled,
        x_test_pca,
        y_test_scaled,
        x_scaler,
        y_scaler,
        pca,
    )


def _evaluate_fold(
    model,
    inputs_pca,
    parameters_scaled,
    parameters_transformed,
    target_curves,
    y_scaler,
    pca,
    x_scaler,
    n_params,
    device,
    cfg=None,
):
    """Run complete evaluation on an explicit validation or test dataset.
    model: the restored UnifiedModel to evaluate
    inputs_pca: PCA-transformed validation or test curves
    parameters_scaled: scaled validation or test parameters
    parameters_transformed: true parameters in transformed target space
    target_curves: true curves in original light-curve space
    y_scaler: fitted Scaler for parameters (to inverse transform predictions)
    pca: fitted PCA object (to inverse transform predicted curves)
    x_scaler: fitted Scaler for curves (to inverse transform predicted curves)
    n_params: number of parameters (for metric computation)
    device: torch.device to run on
    Returns:
    - char_metrics: dict of characterization metrics (RMSE, RRMSE, MAE, R2)
        comparing predicted parameters to transformed targets
    - gen_metrics: dict of generation metrics (RMSE, RRMSE, MAE, R2)
        comparing reconstructed curves to original target curves
    """
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.FloatTensor(inputs_pca).to(device)
        parameters_tensor = torch.FloatTensor(parameters_scaled).to(device)

        pred_params_sc = model.regressor(inputs_tensor).cpu().numpy()
        pred_curves_pca = model.generator(parameters_tensor).cpu().numpy()

        pred_params_transformed = scaled_to_transformed(
            pred_params_sc,
            y_scaler,
        )
        pred_curves = x_scaler.inverse_transform(
            pca.inverse_transform(pred_curves_pca)
        )

    param_names = validate_parameter_names(
        n_params,
        None if cfg is None else cfg["data"].get("param_names"),
    )
    char_metrics = compute_target_metrics(
        parameters_transformed,
        pred_params_transformed,
        param_names,
        cfg,
    )
    gen_metrics = {
        "RMSE": get_rmse(target_curves.ravel(), pred_curves.ravel()),
        "RRMSE": get_rrmse(target_curves.ravel(), pred_curves.ravel()),
        "MAE": get_mae(target_curves.ravel(), pred_curves.ravel()),
        "R2": get_r_squared(target_curves.ravel(), pred_curves.ravel()),
    }
    return char_metrics, gen_metrics


def _predict_unified_characterisation_transformed(
    model,
    inputs_pca,
    y_scaler,
    device,
):
    """Return unified regressor predictions in transformed target space."""
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.FloatTensor(inputs_pca).to(device)
        predictions_scaled = model.regressor(inputs_tensor).cpu().numpy()
    return scaled_to_transformed(predictions_scaled, y_scaler)


def _initialise_parameter_history(param_names):
    return {
        name: {metric: [] for metric in METRIC_NAMES}
        for name in param_names
    }


def _record_parameter_metrics(history, per_parameter):
    for name, metric_history in history.items():
        for metric in METRIC_NAMES:
            metric_history[metric].append(per_parameter[name][metric])


def _print_parameter_history(history, space):
    """Print cross-validation parameter metrics in one explicit space."""
    print(f"\n--- CHARACTERIZATION ({space} space, per parameter) ---")
    for name, metric_history in history.items():
        print(f"  {name}:")
        for metric, values in metric_history.items():
            print(
                f"    {metric}: {np.mean(values):.4f}  "
                f"(+/- {np.std(values):.4f})"
            )


def _summarise_parameter_history(history):
    return {
        name: summarise_metric_history(metric_history)
        for name, metric_history in history.items()
    }


def _execute_unified_training(cfg, experiment, device):
    """Run unified cross-validation inside an initialised experiment.

    Steps:
    1. Load the raw data (curves and parameters).
    2. For each fold in K-Fold cross-validation:
       a. Preprocess the fold's data (augmentation, scaling, PCA).
       b. Instantiate the UnifiedModel and optimizer.
       c. Train the model on the training fold.
       d. Select and restore the best validation epoch.
       e. Evaluate complete validation and final test metrics.
       f. Select the global checkpoint using validation only.
    3. Report and persist aggregate test statistics across folds.
    """
    exp_dir = str(experiment.directory)
    print(f"Experiment directory: {exp_dir}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    training_control = resolve_training_control(
        train_cfg,
        "characterization.transformed.aggregate.R2",
    )
    training_control["selection_policy"]["metric_path"] = (
        "characterization.transformed.aggregate."
        + training_control["selection_policy"]["metric"]
    )
    training_control["validation_fraction"] = resolve_validation_fraction(cfg)
    n_days = data_cfg["n_days"]
    n_params = data_cfg["n_params"]
    param_names = validate_parameter_names(
        n_params,
        data_cfg.get("param_names"),
    )
    samples_per_day = data_cfg.get("samples_per_day", 4)
    n_pca = model_cfg["pca_components"]
    noise_std = cfg["augmentation"]["noise_std"]

    print(f"Loading data on {device}...")
    x_raw, y_physical = load_raw_data(None, cfg)
    if y_physical is None:
        raise ValueError("Unified training requires physical parameter labels")
    y_transformed = physical_to_transformed(y_physical, cfg)

    data_id = dataset_identity(x_raw, y_transformed)
    experiment.save_preprocessing_source(x_raw, y_physical, data_id)
    partitions = [holdout_partition(
        len(x_raw), development, test, training_control["validation_fraction"],
        partition_seed(train_cfg["random_seed"], fold), fold)
        for fold, (development, test) in enumerate(outer_partitions(
            len(x_raw), train_cfg["n_splits"], train_cfg["random_seed"]), 1)]
    for partition in partitions:
        validate_pca_size(n_pca, len(partition.training), n_days)

    selections = selection_plan(partitions, cfg.get("partitioning", {}).get("selection_folds"),
                                train_cfg["random_seed"])
    for folds in selections.values():
        for partition in folds:
            validate_pca_size(n_pca, len(partition.training), n_days)
    experiment.save_partition_plan(partitions, selections)

    history_char = create_metric_history()
    history_gen = create_metric_history()
    transformed_parameter_history = _initialise_parameter_history(param_names)
    physical_parameter_history = _initialise_parameter_history(param_names)
    globally_selected_validation_score = None

    print(f"Starting Training Char + Gen with PCA ({n_pca} components)...")
    print(
        "Checkpoint selection: validation "
        f"{training_control['selection_policy']['metric']} on the "
        "characterisation branch in transformed aggregate target space "
        f"({training_control['selection_policy']['mode']}); test data is "
        "reserved for final estimation."
    )
    print(
        "Early stopping: "
        + (
            "enabled "
            f"(patience={training_control['early_stopping']['patience']}, "
            f"min_delta={training_control['early_stopping']['min_delta']})"
            if training_control["early_stopping"]["enabled"]
            else "disabled; all maximum epochs will run"
        )
        + "."
    )

    for sample_partition in partitions:
        fold_idx = sample_partition.outer_fold
        test_idx = sample_partition.indices("test")
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
        training_seed_plan["validation_split"] = partition_seed(train_cfg["random_seed"], fold_idx)
        training_global_indices = sample_partition.indices("training")
        validation_global_indices = sample_partition.indices("validation")
        bundle = fit_preprocessing(x_raw, y_transformed, sample_partition, n_pca,
                                   preprocessing_seed_plan["pca"], cfg, data_id=data_id)
        print(
            f"    [Fold {fold_idx}] Reproducibility seeds: "
            f"augmentation={preprocessing_seed_plan['augmentation']}, "
            f"pca={preprocessing_seed_plan['pca']}, "
            f"model={training_seed_plan['model']}, "
            f"data_loader={training_seed_plan['data_loader']}"
        )

        x_train_clean = x_raw[training_global_indices]
        x_validation_clean = x_raw[validation_global_indices]
        x_test_clean = x_raw[test_idx]
        y_train = y_transformed[training_global_indices]
        y_validation = y_transformed[validation_global_indices]
        y_test = y_transformed[test_idx]

        (
            x_train_combined,
            y_train_combined,
            x_train_clean_targets,
            x_validation_pca,
            y_validation_scaled,
            x_test_pca,
            y_test_scaled,
            x_scaler,
            y_scaler,
            pca,
        ) = _preprocess_fold(
            x_train_clean,
            x_validation_clean,
            x_test_clean,
            y_train,
            y_validation,
            y_test,
            n_pca,
            noise_std,
            n_days,
            samples_per_day,
            fold_idx,
            preprocessing_seed_plan["augmentation"],
            preprocessing_seed_plan["pca"],
            bundle=bundle,
        )

        configure_torch_determinism(training_seed_plan["model"])
        experiment.record_execution_environment(device=device)

        train_loader = build_training_loader(
            x_train_combined,
            y_train_combined,
            batch_size=train_cfg["batch_size"],
            seed=training_seed_plan["data_loader"],
            reconstruction_targets=x_train_clean_targets,
        )

        model = build_unified_model(cfg).to(device)

        optimizer, criterion, scheduler = build_training_components(
            model,
            train_cfg,
        )

        selection_tracker = build_validation_selection_tracker(
            training_control
        )

        def validation_score_fn(current_model):
            predictions = _predict_unified_characterisation_transformed(
                current_model,
                x_validation_pca,
                y_scaler,
                device,
            )
            return compute_selection_metric(
                y_validation,
                predictions,
                training_control["selection_policy"]["metric"],
                n_columns=n_params,
            )

        training_result = model.fit(
            train_loader,
            optimizer,
            criterion,
            criterion,
            device,
            epochs=train_cfg["epochs"],
            alpha_char=cfg["loss"]["alpha_char"],
            alpha_gen=cfg["loss"]["alpha_gen"],
            scheduler=scheduler,
            validation_score_fn=validation_score_fn,
            selection_tracker=selection_tracker,
        )

        validation_char_metrics, validation_gen_metrics = _evaluate_fold(
            model,
            x_validation_pca,
            y_validation_scaled,
            y_validation,
            x_validation_clean,
            y_scaler,
            pca,
            x_scaler,
            n_params,
            device,
            cfg,
        )
        validation_metrics = {
            "characterization": validation_char_metrics,
            "generation": validation_gen_metrics,
        }
        selected_validation_score = assert_selected_validation_score(
            training_result["selected_validation_score"],
            validation_metrics,
            training_control["selection_policy"]["metric_path"],
        )

        test_char_metrics, test_gen_metrics = _evaluate_fold(
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
            cfg,
        )

        transformed_char_metrics = test_char_metrics["transformed"][
            "aggregate"
        ]

        record_metric_values(history_char, transformed_char_metrics)
        record_metric_values(history_gen, test_gen_metrics)
        _record_parameter_metrics(
            transformed_parameter_history,
            test_char_metrics["transformed"]["per_parameter"],
        )
        _record_parameter_metrics(
            physical_parameter_history,
            test_char_metrics["physical"]["per_parameter"],
        )

        elapsed = time.time() - start_time
        print(
            f"Fold {fold_idx} | {elapsed:.0f}s |",
            f"selected validation "
            f"{training_control['selection_policy']['metric']}: "
            f"{selected_validation_score:.4f} | test Char R2 "
            "(transformed): "
            f"{transformed_char_metrics['R2']:.4f} | "
            f"test Gen R2: {test_gen_metrics['R2']:.4f}",
        )
        index_files = experiment.save_fold_indices(
            fold_idx,
            training_global_indices,
            validation_global_indices,
            test_idx,
        )
        trace_path = experiment.save_training_trace(
            fold_idx,
            training_result["trace"],
        )
        experiment.record_fold(
            fold_idx,
            {
                "validation": validation_metrics,
                "test": {
                    "characterization": test_char_metrics,
                    "generation": test_gen_metrics,
                },
            },
            {
                "k_fold": train_cfg["random_seed"],
                "preprocessing": preprocessing_seed_plan,
                "training": training_seed_plan,
            },
            elapsed,
            selection={
                "fold_selected_epoch": training_result["selected_epoch"],
                "fold_selected_validation_score": selected_validation_score,
            },
            training={
                "maximum_epochs": train_cfg["epochs"],
                "validation_fraction": training_control[
                    "validation_fraction"
                ],
                "epochs_completed": training_result["epochs_completed"],
                "stopped_early": training_result["stopped_early"],
                "early_stopping": training_control["early_stopping"],
            },
            index_files=index_files,
            training_trace=trace_path,
            preprocessing=bundle.manifest,
        )

        if is_strictly_better_score(
            selected_validation_score,
            globally_selected_validation_score,
            training_control["selection_policy"]["mode"],
        ):
            globally_selected_validation_score = selected_validation_score
            save_model_checkpoint(
                exp_dir, cfg["checkpoint"], model, x_scaler, y_scaler, pca
            )
            experiment.record_checkpoint(
                fold_idx,
                selected_validation_score,
                checkpoint_artefact_paths(exp_dir, cfg["checkpoint"]),
                selected_epoch=training_result["selected_epoch"],
            )

    print("\n" + "=" * 50)
    print("FINAL TEST PERFORMANCE REPORT")
    print(f"PCA Components: {n_pca}")
    print("=" * 50)
    print_metric_history(
        "TEST CHARACTERIZATION (transformed space)",
        history_char,
    )
    print_metric_history("TEST GENERATION", history_gen)
    _print_parameter_history(transformed_parameter_history, "transformed")
    _print_parameter_history(physical_parameter_history, "physical")
    print("=" * 50)

    experiment.complete(
        {
            "characterization": {
                "transformed": {
                    "aggregate": summarise_metric_history(history_char),
                    "per_parameter": _summarise_parameter_history(
                        transformed_parameter_history
                    ),
                },
                "physical": {
                    "per_parameter": _summarise_parameter_history(
                        physical_parameter_history
                    ),
                },
            },
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
    training_control = resolve_training_control(
        train_cfg,
        "characterization.transformed.aggregate.R2",
    )
    training_control["selection_policy"]["metric_path"] = (
        "characterization.transformed.aggregate."
        + training_control["selection_policy"]["metric"]
    )
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
        checkpoint_selection_policy=training_control["selection_policy"],
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
