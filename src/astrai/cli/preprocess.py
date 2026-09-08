"""
preprocess.py - Fit and persist shared preprocessing artifacts (scalers, PCA, augmented data).

Fits PCA and StandardScalers once on the full dataset, then for each K-Fold
split saves the pre-transformed arrays so that training scripts can load them
directly without recomputing.

Usage::

    astrai preprocess
    astrai preprocess --config configs/default_split.yaml
    astrai preprocess --config configs/default_split.yaml --out path/to/run
"""
import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
    build_preprocessing_seed_plan,
    make_numpy_rng,
)
from astrai.utils.runtime_environment import capture_runtime_environment
from astrai.utils.target_transformations import (
    PREPROCESSING_ARTEFACT_SCHEMA_VERSION,
    physical_to_transformed,
    target_transform_contract,
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
    seed_plan = build_preprocessing_seed_plan(
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


def _fit_global_artifacts(
    x_raw,
    y_transformed,
    n_pca,
    out_dir,
    random_state,
):
    """Fit scalers and PCA on full dataset and save to out_dir.

    Parameters
    ----------
    x_raw : np.ndarray
        Raw input curves, shape (n_samples, n_days).
    y_transformed : np.ndarray
        Parameters in canonical model target space.
    n_pca : int
        Number of PCA components to keep.
    out_dir : str
        Output directory to save fitted artifacts.
    random_state : int
        Explicit PCA seed.
    """
    print("Fitting scalers and PCA on full dataset...")
    x_scaler = StandardScaler()
    x_scaler.fit(x_raw)

    y_scaler = StandardScaler()
    y_scaler.fit(y_transformed)

    pca = PCA(n_components=n_pca, random_state=random_state)
    pca.fit(x_scaler.transform(x_raw))
    explained_var = pca.explained_variance_ratio_.sum()
    print(
        f"PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)"
    )

    joblib.dump(x_scaler, os.path.join(out_dir, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(out_dir, "y_scaler.pkl"))
    joblib.dump(pca, os.path.join(out_dir, "pca.pkl"))

    return x_scaler, y_scaler, pca


def _process_fold(
    fold_idx,
    x_raw,
    y_physical,
    y_transformed,
    train_idx,
    test_idx,
    x_scaler,
    y_scaler,
    pca,
    n_days,
    noise_std,
    samples_per_day,
    out_dir,
    augmentation_seed,
):
    """Augment, transform, and save a single fold's data.

    Parameters
    ----------
    fold_idx : int
        Index of the current fold (1-based).
        x_raw : np.ndarray
        Raw input curves, shape (n_samples, n_days).
        y_physical : np.ndarray
        Parameters in physical space, shape (n_samples, n_params).
        y_transformed : np.ndarray
        Parameters in canonical model target space.
        train_idx : np.ndarray
        Indices for training samples in this fold.
        test_idx : np.ndarray
        Indices for test samples in this fold.
        x_scaler : StandardScaler
        Fitted scaler for input curves.
        y_scaler : StandardScaler
        Fitted scaler for parameters.
        pca : PCA
        Fitted PCA for input curves.
        n_days : int
        Number of days in the input curves.
        noise_std : float
        Standard deviation of Gaussian noise for augmentation.
        samples_per_day : int
        Number of augmented samples to generate per day.
        out_dir : str
        Base output directory for this fold's artifacts.
    augmentation_seed : int
        Seed for this fold's independent augmentation stream.
    """
    fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    x_train_clean = x_raw[train_idx]
    x_test_clean = x_raw[test_idx]
    y_train_transformed = y_transformed[train_idx]
    y_test_transformed = y_transformed[test_idx]
    y_test_physical = y_physical[test_idx]

    _save_index_array(os.path.join(fold_dir, "train_idx.npy"), train_idx)
    _save_index_array(os.path.join(fold_dir, "test_idx.npy"), test_idx)

    print("  Applying LSST augmentation...")
    x_train_aug, _ = apply_lsst_pipeline(
        x_train_clean,
        n_days,
        noise_std,
        samples_per_day=samples_per_day,
        rng=make_numpy_rng(augmentation_seed),
    )

    x_train_clean_pca = pca.transform(x_scaler.transform(x_train_clean))
    x_train_aug_pca = pca.transform(x_scaler.transform(x_train_aug))
    x_test_pca = pca.transform(x_scaler.transform(x_test_clean))

    y_train_scaled = y_scaler.transform(y_train_transformed)
    y_test_scaled = y_scaler.transform(y_test_transformed)

    model_arrays = {
        "x_train_clean_pca.npy": x_train_clean_pca,
        "x_train_aug_pca.npy": x_train_aug_pca,
        "x_test_pca.npy": x_test_pca,
        "y_train_scaled.npy": y_train_scaled,
        "y_test_scaled.npy": y_test_scaled,
        "y_test_transformed.npy": y_test_transformed,
        "y_test_physical.npy": y_test_physical,
        "x_test_clean.npy": x_test_clean,
    }
    for filename, values in model_arrays.items():
        _save_model_array(os.path.join(fold_dir, filename), values)

    print(f"  Saved to {fold_dir}")


def _generate_preprocessing_artefacts(cfg, out_dir):
    """Generate the numerical artefacts inside an existing run directory."""
    n_days = cfg["data"]["n_days"]
    samples_per_day = cfg["data"].get("samples_per_day", 4)
    noise_std = cfg["augmentation"]["noise_std"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]
    seed_plan = build_preprocessing_seed_plan(
        cfg["preprocessing"]["random_seed"],
        n_splits,
    )

    print("Loading data...")
    x_raw, y_physical = load_raw_data(None, cfg)
    if y_physical is None:
        raise ValueError("Preprocessing requires physical parameter labels")
    y_transformed = physical_to_transformed(y_physical, cfg)

    x_scaler, y_scaler, pca = _fit_global_artifacts(
        x_raw,
        y_transformed,
        n_pca,
        out_dir,
        random_state=seed_plan["pca"],
    )

    _save_model_array(os.path.join(out_dir, "x_raw.npy"), x_raw)
    _save_model_array(
        os.path.join(out_dir, "y_physical.npy"),
        y_physical,
    )
    _save_model_array(
        os.path.join(out_dir, "y_transformed.npy"),
        y_transformed,
    )

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed_plan["k_fold"],
    )

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_raw), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        _process_fold(
            fold_idx,
            x_raw,
            y_physical,
            y_transformed,
            train_idx,
            test_idx,
            x_scaler,
            y_scaler,
            pca,
            n_days,
            noise_std,
            samples_per_day,
            out_dir,
            seed_plan["augmentation"][f"fold_{fold_idx}"],
        )


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
        _generate_preprocessing_artefacts(cfg, run_dir)
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
