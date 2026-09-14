"""
preprocess.py - Fit and persist shared preprocessing artifacts (scalers, PCA, augmented data).

Partitions original samples first, then fits clean-only scaler/PCA bundles on
each effective training subset. Both model stages consume the same partitions.

Usage::

    astrai preprocess
    astrai preprocess --config configs/default_split.yaml
    astrai preprocess --config configs/default_split.yaml --out path/to/run
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from astrai.utils.augmentation import apply_lsst_pipeline
from astrai.utils.array_dtypes import (
    INDEX_ARRAY_DTYPE,
    MODEL_ARRAY_DTYPE,
    as_index_array,
    as_model_array,
)
from astrai.utils.configuration import load_config
from astrai.utils.data import load_raw_data
from astrai.utils.log_experiments import save_code
from astrai.utils.reproducibility import (
    build_pool_preprocessing_seed_plan,
    make_numpy_rng,
)
from astrai.utils.runtime_environment import capture_runtime_environment
from astrai.utils.target_transformations import (
    PREPROCESSING_ARTEFACT_SCHEMA_VERSION,
    physical_to_transformed,
    target_transform_contract,
)
from astrai.utils.partitions import (
    holdout_partition, outer_partitions, partition_seed,
    resolve_validation_fraction, selection_plan,
)
from astrai.utils.preprocessing import (
    dataset_identity, fit_preprocessing, validate_pca_size,
)
from astrai.paths import (
    resolve_config_path,
    source_checkout_root,
    source_snapshot_root,
)


_SOURCE_CHECKOUT_ROOT = source_checkout_root()
_REPOSITORY_ROOT = _SOURCE_CHECKOUT_ROOT or source_snapshot_root()
_DEFAULT_RUNS_DIR = Path("preprocessed")
_CONFIG_SNAPSHOT_NAME = "config.yaml"
_METADATA_NAME = "metadata.yaml"
_ARTEFACT_SCHEMA_VERSION = PREPROCESSING_ARTEFACT_SCHEMA_VERSION


def _utc_now():
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _config_run_name(config_path):
    """Return a filesystem-safe run name derived from a config path."""
    if config_path is None:
        return "config"

    name = Path(config_path).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-_")
    return name or "config"


def _create_run_directory(config_path, out_dir=None, now=None):
    """Create and return a new preprocessing run directory.

    An explicit destination is used exactly as supplied. Otherwise a
    timestamped directory is created below ``preprocessed``. Existing
    destinations are rejected so that prior artefacts cannot be overwritten.
    """
    if out_dir is None:
        timestamp = (now or _utc_now()).strftime("%Y%m%d_%H%M%S")
        destination = (
            _DEFAULT_RUNS_DIR
            / f"{timestamp}_{_config_run_name(config_path)}"
        ).expanduser().resolve()
    else:
        destination = Path(out_dir).expanduser().resolve()

    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Preprocessing output directory already exists: {destination}. "
            "Choose a new --out directory or omit --out to create a "
            "timestamped run."
        ) from exc

    return destination


def _run_git_command(repository_root, *args):
    """Return stripped Git output, or ``None`` outside a Git worktree."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_metadata(repository_root):
    """Describe the source revision used for a preprocessing run."""
    if (
        _SOURCE_CHECKOUT_ROOT is None
        and Path(repository_root).resolve() == Path(_REPOSITORY_ROOT).resolve()
    ):
        return {
            "commit": None,
            "branch": None,
            "working_tree_dirty": None,
        }
    commit = _run_git_command(repository_root, "rev-parse", "HEAD")
    if commit is None:
        return {
            "commit": None,
            "branch": None,
            "working_tree_dirty": None,
        }

    branch = _run_git_command(
        repository_root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
    )
    status = _run_git_command(repository_root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": None if status is None else bool(status),
    }


def _write_metadata(run_dir, metadata):
    """Write preprocessing run metadata in a human-readable format."""
    metadata_path = Path(run_dir) / _METADATA_NAME
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)


def _array_artefact_metadata(run_dir):
    """Describe every persisted NumPy array without loading it into memory."""
    run_dir = Path(run_dir)
    artefacts = {}
    for path in sorted(run_dir.rglob("*.npy")):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        artefacts[path.relative_to(run_dir).as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dtype": array.dtype.name,
            "shape": [int(size) for size in array.shape],
        }
        del array
    return artefacts


def _save_model_array(path, values):
    """Persist model-ready values using the shared ``float32`` contract."""
    np.save(path, as_model_array(values))


def _save_index_array(path, values):
    """Persist fold indices using the shared ``int64`` contract."""
    np.save(path, as_index_array(values))


def _save_config_snapshot(run_dir, cfg, config_path=None):
    """Save the exact source config, or serialise a programmatic config."""
    destination = Path(run_dir) / _CONFIG_SNAPSHOT_NAME
    if config_path is not None:
        shutil.copy2(Path(config_path).expanduser(), destination)
        return

    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)


def _initial_metadata(cfg, started_at, repository_root):
    """Build the initial metadata record for a preprocessing run."""
    preprocessing_cfg = cfg["preprocessing"]
    n_splits = preprocessing_cfg["n_splits"]
    seed_plan = build_pool_preprocessing_seed_plan(
        preprocessing_cfg["random_seed"],
        n_splits,
    )
    return {
        "preprocessing_artefact_schema_version": _ARTEFACT_SCHEMA_VERSION,
        "run": {
            "status": "running",
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": None,
        },
        "config": {
            "snapshot": _CONFIG_SNAPSHOT_NAME,
        },
        "preprocessing": {
            "random_seed": preprocessing_cfg["random_seed"],
            "n_splits": n_splits,
            "folds": list(range(1, n_splits + 1)),
            "seed_plan": seed_plan,
        },
        "array_dtypes": {
            "model": MODEL_ARRAY_DTYPE.name,
            "indices": INDEX_ARRAY_DTYPE.name,
        },
        "target_transform": target_transform_contract(cfg),
        "array_artefacts": {},
        "git": _git_metadata(repository_root),
        "environment": capture_runtime_environment(),
    }


def _process_fold(fold_idx, x_raw, y_physical, y_transformed, partition,
                  bundle, n_days, noise_std, samples_per_day, out_dir,
                  augmentation_seed):
    """Materialise clean/augmented training and clean validation/test views."""
    fold_dir = Path(out_dir) / f"fold_{fold_idx}"
    bundle.save(fold_dir)
    for name, indices in (("train_idx", partition.pool),
                          ("training_idx", partition.training),
                          ("validation_idx", partition.validation),
                          ("test_idx", partition.test)):
        _save_index_array(fold_dir / f"{name}.npy", np.asarray(indices, dtype=np.int64))
    train = partition.indices("training")
    validation = partition.indices("validation")
    test = partition.indices("test")
    augmented, _ = apply_lsst_pipeline(
        x_raw[train], n_days, noise_std, samples_per_day=samples_per_day,
        rng=make_numpy_rng(augmentation_seed),
    )
    arrays = {
        "x_train_clean_pca.npy": bundle.transform_curves(x_raw[train]),
        "x_train_aug_pca.npy": bundle.transform_curves(augmented),
        "y_train_scaled.npy": bundle.transform_parameters(y_transformed[train]),
        "x_validation_pca.npy": bundle.transform_curves(x_raw[validation]),
        "y_validation_scaled.npy": bundle.transform_parameters(y_transformed[validation]),
        "x_test_pca.npy": bundle.transform_curves(x_raw[test]),
        "y_test_scaled.npy": bundle.transform_parameters(y_transformed[test]),
        "y_test_transformed.npy": y_transformed[test],
        "y_test_physical.npy": y_physical[test],
        "x_test_clean.npy": x_raw[test],
    }
    for filename, values in arrays.items():
        _save_model_array(fold_dir / filename, values)
    return {"bundle": f"fold_{fold_idx}/bundle.yaml",
            "bundle_id": bundle.manifest["bundle_id"]}


def _generate_preprocessing_artefacts(cfg, out_dir):
    """Build shared partitions before any learned preprocessing is fitted."""
    n_days = cfg["data"]["n_days"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]
    base_seed = cfg["preprocessing"]["random_seed"]
    seed_plan = build_pool_preprocessing_seed_plan(base_seed, n_splits)
    fraction = resolve_validation_fraction(cfg)
    x_raw, y_physical = load_raw_data(None, cfg)
    if y_physical is None:
        raise ValueError("Preprocessing requires physical parameter labels")
    x_raw, y_physical = as_model_array(x_raw), as_model_array(y_physical)
    y_transformed = as_model_array(physical_to_transformed(y_physical, cfg))
    data_id = dataset_identity(x_raw, y_transformed)
    partitions = [holdout_partition(len(x_raw), development, test, fraction,
                                   partition_seed(base_seed, fold), fold)
                  for fold, (development, test) in enumerate(
                      outer_partitions(len(x_raw), n_splits, base_seed), 1)]
    for partition in partitions:
        validate_pca_size(n_pca, len(partition.training), n_days)
    # Optional future selection topology: indices only, never nested training.
    k_select = cfg.get("partitioning", {}).get("selection_folds")
    selections = selection_plan(partitions, k_select, base_seed)
    for folds in selections.values():
        for partition in folds:
            validate_pca_size(n_pca, len(partition.training), n_days)
    for name, values in (("x_raw", x_raw), ("y_physical", y_physical),
                         ("y_transformed", y_transformed)):
        _save_model_array(Path(out_dir) / f"{name}.npy", values)
    plan = {"schema_version": 1, "dataset_id": data_id,
            "protocol": "holdout_validation", "validation_fraction": fraction,
            "outer_folds": n_splits, "base_seed": base_seed,
            "selection_folds": k_select,
            "holdout": [p.record() for p in partitions],
            "selection": {key: [p.record() for p in folds]
                          for key, folds in selections.items()}}
    (Path(out_dir) / "partitions.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    bundles = {}
    for partition in partitions:
        fold = partition.outer_fold
        bundle = fit_preprocessing(
            x_raw, y_transformed, partition, n_pca,
            seed_plan["pca"][f"fold_{fold}"],
            cfg, data_id=data_id)
        bundles[f"fold_{fold}"] = _process_fold(
            fold, x_raw, y_physical, y_transformed, partition, bundle,
            n_days, cfg["augmentation"]["noise_std"],
            cfg["data"].get("samples_per_day", 4), out_dir,
            seed_plan["augmentation"][f"fold_{fold}"])
    return {"dataset": {"id": data_id, "sample_identity": "canonical_row_index",
                        "curves": "x_raw.npy", "physical_parameters": "y_physical.npy",
                        "transformed_parameters": "y_transformed.npy"},
            "partition_plan": "partitions.yaml", "bundles": bundles,
            "view_configuration": {"noise_std": cfg["augmentation"]["noise_std"],
                                   "samples_per_day": cfg["data"].get("samples_per_day", 4)},
            "fit_policy": "clean_original_samples", "protocol": "holdout_validation"}


def run_preprocessing(cfg, out_dir=None, config_path=None):
    """Run preprocessing in a new, self-contained artefact directory.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    out_dir : str or pathlib.Path or None
        Exact destination for the new run. A timestamped directory below
        ``preprocessed`` is created when omitted.
    config_path : str or pathlib.Path or None
        Source YAML file to preserve verbatim. When omitted, ``cfg`` is
        serialised to the run directory.

    Returns
    -------
    str
        Path to the completed preprocessing run directory.
    """
    if config_path is not None and not Path(config_path).expanduser().is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    started_at = _utc_now()
    metadata = _initial_metadata(cfg, started_at, _REPOSITORY_ROOT)
    run_dir = _create_run_directory(config_path, out_dir, now=started_at)
    _write_metadata(run_dir, metadata)

    try:
        _save_config_snapshot(run_dir, cfg, config_path=config_path)
        save_code(run_dir, folder=_REPOSITORY_ROOT)
        generated = _generate_preprocessing_artefacts(cfg, run_dir)
        if isinstance(generated, dict):
            metadata.update(generated)
        array_artefacts = _array_artefact_metadata(run_dir)
    except BaseException as exc:
        metadata["run"].update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now().isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        _write_metadata(run_dir, metadata)
        raise

    metadata["run"].update(
        {
            "status": "completed",
            "completed_at_utc": _utc_now().isoformat(),
        }
    )
    metadata["array_artefacts"] = array_artefacts
    _write_metadata(run_dir, metadata)

    run_path = str(run_dir)
    print(f"\nPreprocessing complete. Artefacts in: {run_path}")
    print("Use this directory for subsequent training:")
    print(f"  --prep {run_path}")
    return run_path


def main(argv=None):
    """Main function to run preprocessing.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    out_dir : str
        Output directory for preprocessing artifacts.
    """
    parser = argparse.ArgumentParser(
        prog="astrai preprocess",
        description="ASTRAI shared preprocessing",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML (default: packaged default_split.yaml)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Exact output directory for this preprocessing run. If omitted, "
            "a timestamped directory is created under preprocessed/."
        ),
    )
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config, "default_split.yaml")

    cfg = load_config(config_path)

    run_preprocessing(
        cfg,
        out_dir=args.out,
        config_path=str(config_path),
    )


if __name__ == "__main__":
    main()
