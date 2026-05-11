"""
preprocess.py - Fit and persist shared preprocessing artifacts (scalers, PCA, augmented data).

Fits PCA and StandardScalers once on the full dataset, then for each K-Fold
split saves the pre-transformed arrays so that training scripts can load them
directly without recomputing.

Usage::

    python preprocess.py
    python preprocess.py --config configs/default_split.yaml --out preprocessed
"""
import os
import argparse
import numpy as np
import joblib
import yaml
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.checkpoints import load_data
from utils.augmentation import apply_lsst_pipeline


def _fit_global_artifacts(x_raw, y_raw, n_pca, out_dir):
    """Fit scalers and PCA on full dataset and save to out_dir.

    Parameters
    ----------
    x_raw : np.ndarray
        Raw input curves, shape (n_samples, n_days).
    y_raw : np.ndarray
        Raw parameters, shape (n_samples, n_params).
    n_pca : int
        Number of PCA components to keep.
    out_dir : str
        Output directory to save fitted artifacts.
    """
    print("Fitting scalers and PCA on full dataset...")
    x_scaler = StandardScaler()
    x_scaler.fit(x_raw)

    y_scaler = StandardScaler()
    y_scaler.fit(y_raw)

    pca = PCA(n_components=n_pca)
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
    y_raw,
    train_idx,
    test_idx,
    x_scaler,
    y_scaler,
    pca,
    n_days,
    noise_std,
    samples_per_day,
    out_dir,
):
    """Augment, transform, and save a single fold's data.

    Parameters
    ----------
    fold_idx : int
        Index of the current fold (1-based).
        x_raw : np.ndarray
        Raw input curves, shape (n_samples, n_days).
        y_raw : np.ndarray
        Raw parameters, shape (n_samples, n_params).
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
    """
    fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    x_train_clean = x_raw[train_idx]
    x_test_clean = x_raw[test_idx]
    y_train = y_raw[train_idx]
    y_test = y_raw[test_idx]

    np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(fold_dir, "test_idx.npy"), test_idx)

    print("  Applying LSST augmentation...")
    x_train_aug = apply_lsst_pipeline(
        x_train_clean, n_days, noise_std, samples_per_day=samples_per_day
    )

    x_train_clean_pca = pca.transform(x_scaler.transform(x_train_clean))
    x_train_aug_pca = pca.transform(x_scaler.transform(x_train_aug))
    x_test_pca = pca.transform(x_scaler.transform(x_test_clean))

    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    np.save(os.path.join(fold_dir, "x_train_clean_pca.npy"), x_train_clean_pca)
    np.save(os.path.join(fold_dir, "x_train_aug_pca.npy"), x_train_aug_pca)
    np.save(os.path.join(fold_dir, "x_test_pca.npy"), x_test_pca)
    np.save(os.path.join(fold_dir, "y_train_scaled.npy"), y_train_scaled)
    np.save(os.path.join(fold_dir, "y_test_scaled.npy"), y_test_scaled)
    np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
    np.save(os.path.join(fold_dir, "x_test_clean.npy"), x_test_clean)

    print(f"  Saved to {fold_dir}")


def run_preprocessing(cfg, out_dir="preprocessed"):
    """Run the full preprocessing pipeline and save artifacts to out_dir.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    out_dir : str
        Output directory for preprocessing artifacts.
    """
    n_days = cfg["data"]["n_days"]
    samples_per_day = cfg["data"].get("samples_per_day", 4)
    noise_std = cfg["augmentation"]["noise_std"]
    n_pca = cfg["preprocessing"]["pca_components"]
    n_splits = cfg["preprocessing"]["n_splits"]
    seed = cfg["preprocessing"]["random_seed"]

    print("Loading data...")
    x_raw, y_raw = load_data(None, cfg)
    y_raw = np.log1p(y_raw)

    os.makedirs(out_dir, exist_ok=True)

    x_scaler, y_scaler, pca = _fit_global_artifacts(
        x_raw, y_raw, n_pca, out_dir
    )

    np.save(os.path.join(out_dir, "x_raw.npy"), x_raw)
    np.save(os.path.join(out_dir, "y_raw.npy"), y_raw)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_raw), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        _process_fold(
            fold_idx,
            x_raw,
            y_raw,
            train_idx,
            test_idx,
            x_scaler,
            y_scaler,
            pca,
            n_days,
            noise_std,
            samples_per_day,
            out_dir,
        )

    print(f"\nPreprocessing complete. Artifacts in: {out_dir}")


def main():
    """Main function to run preprocessing.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    out_dir : str
        Output directory for preprocessing artifacts.
    """
    parser = argparse.ArgumentParser(description="ASTRAI shared preprocessing")
    parser.add_argument("--config", default="configs/default_split.yaml")
    parser.add_argument(
        "--out",
        default="preprocessed",
        help="Output directory for preprocessing artifacts",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_preprocessing(cfg, out_dir=args.out)


if __name__ == "__main__":
    main()
