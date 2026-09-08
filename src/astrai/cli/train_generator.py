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
import torch

from astrai.models.factories import build_generator
from astrai.utils.metrics import (
    compute_selection_metric,
    get_rmse,
    get_mae,
    get_r_squared,
    get_rrmse,
)
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
    build_validation_selection_tracker,
    assert_selected_validation_score,
    create_metric_history,
    is_strictly_better_score,
    load_fold_arrays,
    load_outer_fold_indices,
    load_preprocessing_source_array,
    partition_precomputed_development_data,
    print_metric_history,
    record_metric_values,
    resolve_test_fold,
    resolve_training_control,
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


def _predict_generator_curves(
    model,
    parameters_scaled,
    x_scaler,
    pca,
    device,
):
    """Return generated light curves in their original data space."""
    model.eval()
    with torch.no_grad():
        parameters_tensor = torch.FloatTensor(parameters_scaled).to(device)
        predictions_pca = model(parameters_tensor).cpu().numpy()
    return x_scaler.inverse_transform(pca.inverse_transform(predictions_pca))


def _evaluate_generator(
    model,
    parameters_scaled,
    target_curves,
    prep_dir,
    device,
    x_scaler=None,
    pca=None,
):
    """Evaluate complete generation metrics on an explicit dataset.

    ``parameters_scaled`` and ``target_curves`` may represent validation or
    test data. Scaler and PCA instances are loaded from ``prep_dir`` only when
    the caller has not supplied the already loaded instances.
    """
    if x_scaler is None:
        x_scaler = joblib.load(os.path.join(prep_dir, "x_scaler.pkl"))
    if pca is None:
        pca = joblib.load(os.path.join(prep_dir, "pca.pkl"))
    pred_curves = _predict_generator_curves(
        model,
        parameters_scaled,
        x_scaler,
        pca,
        device,
    )

    return {
        "RMSE": get_rmse(target_curves.ravel(), pred_curves.ravel()),
        "RRMSE": get_rrmse(target_curves.ravel(), pred_curves.ravel()),
        "MAE": get_mae(target_curves.ravel(), pred_curves.ravel()),
        "R2": get_r_squared(target_curves.ravel(), pred_curves.ravel()),
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
        Directory with preprocessing output.
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
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]
    base_seed = cfg["preprocessing"]["random_seed"]
    gen_cfg = cfg["generator"]
    training_cfg = gen_cfg["training"]
    training_control = resolve_training_control(training_cfg, "R2")
    training_control["selection_policy"]["metric_path"] = training_control[
        "selection_policy"
    ]["metric"]

    test_fold = resolve_test_fold(training_cfg)
    fold_indices = resolve_fold_indices(test_fold, n_splits)
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
        checkpoint_selection_policy=training_control["selection_policy"],
    )
    exp_dir = str(experiment.directory)
    print(f"Generator experiment directory: {exp_dir}")

    history = create_metric_history()
    globally_selected_validation_score = None
    x_raw = None
    x_scaler = None
    pca = None

    try:
        print(
            "Starting Generator Training (MLPWithResiduals) with PCA "
            f"({n_pca} components)..."
        )
        print(
            "Checkpoint selection: validation "
            f"{training_control['selection_policy']['metric']} in flattened "
            "light-curve space "
            f"({training_control['selection_policy']['mode']}); test data "
            "is reserved for final estimation."
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

        if test_fold is None:
            print(f"Training all {n_splits} folds.")
        else:
            print(f"Training split with test fold {test_fold}.")

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
            if x_raw is None:
                x_raw = load_preprocessing_source_array(
                    prep_dir,
                    "x_raw.npy",
                )
                x_scaler = joblib.load(prep_dir / "x_scaler.pkl")
                pca = joblib.load(prep_dir / "pca.pkl")
            development_global_indices, test_global_indices = (
                load_outer_fold_indices(fold_dir)
            )
            print(
                f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}"
            )

            seed_plan = build_training_seed_plan(
                base_seed,
                "generator",
                fold_idx,
            )
            partition = partition_precomputed_development_data(
                x_train_clean_pca,
                x_train_aug_pca,
                y_train_scaled,
                training_control["validation_fraction"],
                seed_plan["validation_split"],
                minimum_validation_samples=(
                    2
                    if training_control["selection_policy"]["metric"] == "R2"
                    else 1
                ),
            )
            training_global_indices = development_global_indices[
                partition["training_local_indices"]
            ]
            validation_global_indices = development_global_indices[
                partition["validation_local_indices"]
            ]
            validation_target_curves = x_raw[validation_global_indices]
            configure_torch_determinism(seed_plan["model"])
            experiment.record_execution_environment(device=device)
            print(
                f"    [Fold {fold_idx}] Reproducibility seeds: "
                f"model={seed_plan['model']}, "
                f"data_loader={seed_plan['data_loader']}"
            )

            train_loader = build_training_loader(
                partition["training_targets"],
                partition["training_inputs"],
                batch_size=training_cfg["batch_size"],
                seed=seed_plan["data_loader"],
            )

            model = build_generator(cfg).to(device)

            optimizer, criterion, scheduler = build_training_components(
                model,
                training_cfg,
            )

            selection_tracker = build_validation_selection_tracker(
                training_control
            )

            def validation_score_fn(current_model):
                predictions = _predict_generator_curves(
                    current_model,
                    partition["validation_targets"],
                    x_scaler,
                    pca,
                    device,
                )
                return compute_selection_metric(
                    validation_target_curves,
                    predictions,
                    training_control["selection_policy"]["metric"],
                )

            training_result = train_supervised_model(
                model,
                train_loader,
                optimizer,
                criterion,
                scheduler,
                training_cfg["epochs"],
                device,
                validation_score_fn=validation_score_fn,
                selection_tracker=selection_tracker,
            )

            validation_metrics = _evaluate_generator(
                model,
                partition["validation_targets"],
                validation_target_curves,
                prep_dir,
                device,
                x_scaler=x_scaler,
                pca=pca,
            )
            selected_validation_score = assert_selected_validation_score(
                training_result["selected_validation_score"],
                validation_metrics,
                training_control["selection_policy"]["metric_path"],
            )
            test_metrics = _evaluate_generator(
                model,
                y_test_scaled,
                x_test_clean,
                prep_dir,
                device,
                x_scaler=x_scaler,
                pca=pca,
            )

            record_metric_values(history, test_metrics)

            elapsed = time.time() - start_time
            print(
                f"Fold {fold_idx} | {elapsed:.0f}s | "
                f"selected validation "
                f"{training_control['selection_policy']['metric']}: "
                f"{selected_validation_score:.4f} | test R2: "
                f"{test_metrics['R2']:.4f}"
            )
            index_files = experiment.save_fold_indices(
                fold_idx,
                training_global_indices,
                validation_global_indices,
                test_global_indices,
            )
            trace_path = experiment.save_training_trace(
                fold_idx,
                training_result["trace"],
            )
            experiment.record_fold(
                fold_idx,
                {
                    "validation": validation_metrics,
                    "test": test_metrics,
                },
                seed_plan,
                elapsed,
                selection={
                    "fold_selected_epoch": training_result["selected_epoch"],
                    "fold_selected_validation_score": (
                        selected_validation_score
                    ),
                },
                training={
                    "maximum_epochs": training_cfg["epochs"],
                    "validation_fraction": training_control[
                        "validation_fraction"
                    ],
                    "epochs_completed": training_result["epochs_completed"],
                    "stopped_early": training_result["stopped_early"],
                    "early_stopping": training_control["early_stopping"],
                },
                index_files=index_files,
                training_trace=trace_path,
            )

            if is_strictly_better_score(
                selected_validation_score,
                globally_selected_validation_score,
                training_control["selection_policy"]["mode"],
            ):
                globally_selected_validation_score = selected_validation_score
                checkpoint_paths = save_split_checkpoint(
                    exp_dir,
                    gen_cfg["checkpoint"],
                    model,
                    prep_dir,
                )
                experiment.record_checkpoint(
                    fold_idx,
                    selected_validation_score,
                    checkpoint_paths,
                    selected_epoch=training_result["selected_epoch"],
                )

        print("\n" + "=" * 50)
        print("GENERATOR - FINAL TEST PERFORMANCE REPORT")
        print(f"PCA Components: {n_pca}")
        print("=" * 50)
        print_metric_history(
            "TEST GENERATION",
            history,
            test_fold=test_fold,
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
