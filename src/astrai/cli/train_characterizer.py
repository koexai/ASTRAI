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
    compute_selection_metric,
    compute_target_metrics,
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
from astrai.utils.array_dtypes import as_model_array
from astrai.utils.target_transformations import (
    load_fold_target_array,
    scaled_to_transformed,
)
from astrai.paths import resolve_config_path, resolve_user_path


def _load_fold_data(fold_dir):
    """Load preprocessed arrays for a single fold.
    Expects the following files in fold_dir:
    - x_train_clean_pca.npy
    - x_train_aug_pca.npy
    - x_test_pca.npy
    - y_train_scaled.npy
    - y_test_transformed.npy (or y_test.npy for metadata-free legacy runs)
    Returns:
    - x_train_clean_pca: (n_train_clean, n_pca)
    - x_train_aug_pca: (n_train_aug, n_pca)
    - x_test_pca: (n_test, n_pca)
    - y_train_scaled: (n_train, n_params)
    - y_test: (n_test, n_params)"""
    common = load_fold_arrays(
        fold_dir,
        (
            "x_train_clean_pca.npy",
            "x_train_aug_pca.npy",
            "x_test_pca.npy",
            "y_train_scaled.npy",
        ),
    )
    y_test_transformed = as_model_array(
        load_fold_target_array(
            fold_dir,
            "y_test_transformed.npy",
            legacy_name="y_test.npy",
        )
    )
    return (*common, y_test_transformed)


def _predict_characterizer_transformed(model, inputs_pca, y_scaler, device):
    """Return characterizer predictions in canonical transformed space."""
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.FloatTensor(inputs_pca).to(device)
        predictions_scaled = model(inputs_tensor).cpu().numpy()
    return scaled_to_transformed(predictions_scaled, y_scaler)


def _evaluate_characterizer(
    model,
    inputs_pca,
    targets_transformed,
    param_names,
    prep_dir,
    device,
    cfg,
    y_scaler=None,
):
    """Evaluate complete characterisation metrics on an explicit dataset.

    ``inputs_pca`` and ``targets_transformed`` may represent validation or
    test data. Aggregate metrics stay in transformed space, while physical
    metrics remain per parameter. The scaler is loaded from ``prep_dir`` only
    when a caller has not supplied the already loaded instance.
    """
    if y_scaler is None:
        y_scaler = joblib.load(os.path.join(prep_dir, "y_scaler.pkl"))
    pred_transformed = _predict_characterizer_transformed(
        model,
        inputs_pca,
        y_scaler,
        device,
    )

    return compute_target_metrics(
        targets_transformed,
        pred_transformed,
        param_names,
        cfg,
    )


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


def _print_parameter_metrics(per_parameter, space):
    """Print one fold's metrics in configured parameter order."""
    print(f"    Per-parameter metrics ({space} space):")
    for name, metrics in per_parameter.items():
        values = " | ".join(
            f"{metric_name}={metrics[metric_name]:.6f}"
            for metric_name in METRIC_NAMES
        )
        print(f"      {name}: {values}")


def _print_parameter_final_stats(history, test_fold):
    """Print per-parameter values for one fold or statistics across folds."""
    if test_fold is None:
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
        f"(Test fold {test_fold}) ---"
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
        Directory with preprocessing output.
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
    training_cfg = char_cfg["training"]
    training_control = resolve_training_control(
        training_cfg,
        "transformed.aggregate.R2",
    )
    training_control["selection_policy"]["metric_path"] = (
        "transformed.aggregate."
        + training_control["selection_policy"]["metric"]
    )

    test_fold = resolve_test_fold(training_cfg)
    fold_indices = resolve_fold_indices(test_fold, n_splits)
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
        checkpoint_selection_policy=training_control["selection_policy"],
    )
    exp_dir = str(experiment.directory)
    print(f"Characterizer experiment directory: {exp_dir}")

    history = create_metric_history()
    transformed_parameter_history = _initialise_parameter_history(param_names)
    physical_parameter_history = _initialise_parameter_history(param_names)
    globally_selected_validation_score = None
    y_transformed = None
    y_scaler = None

    try:
        print(
            "Starting Characterizer Training (SplitMLP) with PCA "
            f"({n_pca} components)..."
        )
        print(
            "Checkpoint selection: validation "
            f"{training_control['selection_policy']['metric']} "
            "in transformed aggregate target space "
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
                x_test_pca,
                y_train_scaled,
                y_test_transformed,
            ) = _load_fold_data(fold_dir)
            if y_transformed is None:
                y_transformed = load_preprocessing_source_array(
                    prep_dir,
                    "y_transformed.npy",
                )
                y_scaler = joblib.load(prep_dir / "y_scaler.pkl")
            development_global_indices, test_global_indices = (
                load_outer_fold_indices(fold_dir)
            )
            print(
                f"    [Fold {fold_idx}] Loaded preprocessing from {fold_dir}"
            )

            seed_plan = build_training_seed_plan(
                base_seed,
                "characterizer",
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
            validation_targets_transformed = y_transformed[
                validation_global_indices
            ]
            configure_torch_determinism(seed_plan["model"])
            experiment.record_execution_environment(device=device)
            print(
                f"    [Fold {fold_idx}] Reproducibility seeds: "
                f"model={seed_plan['model']}, "
                f"data_loader={seed_plan['data_loader']}"
            )

            train_loader = build_training_loader(
                partition["training_inputs"],
                partition["training_targets"],
                batch_size=training_cfg["batch_size"],
                seed=seed_plan["data_loader"],
            )

            model = build_characterizer(cfg).to(device)

            optimizer, criterion, scheduler = build_training_components(
                model,
                training_cfg,
            )

            selection_tracker = build_validation_selection_tracker(
                training_control
            )

            def validation_score_fn(current_model):
                predictions = _predict_characterizer_transformed(
                    current_model,
                    partition["validation_inputs"],
                    y_scaler,
                    device,
                )
                return compute_selection_metric(
                    validation_targets_transformed,
                    predictions,
                    training_control["selection_policy"]["metric"],
                    n_columns=n_params,
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

            validation_evaluation = _evaluate_characterizer(
                model,
                partition["validation_inputs"],
                validation_targets_transformed,
                param_names,
                prep_dir,
                device,
                cfg,
                y_scaler=y_scaler,
            )
            selected_validation_score = assert_selected_validation_score(
                training_result["selected_validation_score"],
                validation_evaluation,
                training_control["selection_policy"]["metric_path"],
            )

            test_evaluation = _evaluate_characterizer(
                model,
                x_test_pca,
                y_test_transformed,
                param_names,
                prep_dir,
                device,
                cfg,
                y_scaler=y_scaler,
            )
            test_metrics = test_evaluation["transformed"]["aggregate"]

            record_metric_values(history, test_metrics)
            _record_parameter_metrics(
                transformed_parameter_history,
                test_evaluation["transformed"]["per_parameter"],
            )
            _record_parameter_metrics(
                physical_parameter_history,
                test_evaluation["physical"]["per_parameter"],
            )
            _print_parameter_metrics(
                test_evaluation["transformed"]["per_parameter"],
                "transformed",
            )
            _print_parameter_metrics(
                test_evaluation["physical"]["per_parameter"],
                "physical",
            )

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
                    "validation": validation_evaluation,
                    "test": test_evaluation,
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
                    char_cfg["checkpoint"],
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
        print("CHARACTERIZER - FINAL TEST PERFORMANCE REPORT")
        print(f"PCA Components: {n_pca}")
        print("=" * 50)
        print_metric_history(
            "TEST CHARACTERIZATION",
            history,
            test_fold=test_fold,
        )
        print("\nTest metrics in transformed target space:")
        _print_parameter_final_stats(
            transformed_parameter_history,
            test_fold,
        )
        print("\nTest metrics in physical target space:")
        _print_parameter_final_stats(
            physical_parameter_history,
            test_fold,
        )
        print("=" * 50)

        experiment.complete(
            {
                "transformed": {
                    "aggregate": summarise_metric_history(history),
                    "per_parameter": _summarise_parameter_history(
                        transformed_parameter_history
                    ),
                },
                "physical": {
                    "per_parameter": _summarise_parameter_history(
                        physical_parameter_history
                    ),
                },
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
