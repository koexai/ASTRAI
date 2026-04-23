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

from train import load_data
from astrai.augmentation import apply_lsst_pipeline


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
    X_raw, y_raw = load_data(cfg)
    y_raw = np.log1p(y_raw)

    os.makedirs(out_dir, exist_ok=True)

    # --- Fit PCA and scalers ONCE on the full dataset ---
    print("Fitting scalers and PCA on full dataset...")
    x_scaler = StandardScaler()
    x_scaler.fit(X_raw)
    X_all_scaled = x_scaler.transform(X_raw)

    y_scaler = StandardScaler()
    y_scaler.fit(y_raw)

    pca = PCA(n_components=n_pca)
    pca.fit(X_all_scaled)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")

    # Save global artifacts
    joblib.dump(x_scaler, os.path.join(out_dir, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(out_dir, "y_scaler.pkl"))
    joblib.dump(pca, os.path.join(out_dir, "pca.pkl"))

    # Save raw (log-transformed) data
    np.save(os.path.join(out_dir, "X_raw.npy"), X_raw)
    np.save(os.path.join(out_dir, "y_raw.npy"), y_raw)

    # --- Per-fold: augment and transform ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_raw), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        fold_dir = os.path.join(out_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train_clean = X_raw[train_idx]
        X_test_clean = X_raw[test_idx]
        y_train = y_raw[train_idx]
        y_test = y_raw[test_idx]

        # Save fold indices for reproducibility
        np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
        np.save(os.path.join(fold_dir, "test_idx.npy"), test_idx)

        # LSST augmentation on training set
        print("  Applying LSST augmentation...")
        X_train_aug = apply_lsst_pipeline(X_train_clean, n_days, noise_std,
                                          samples_per_day=samples_per_day)

        # Transform with global scaler + PCA
        X_train_clean_pca = pca.transform(x_scaler.transform(X_train_clean))
        X_train_aug_pca = pca.transform(x_scaler.transform(X_train_aug))
        X_test_pca = pca.transform(x_scaler.transform(X_test_clean))

        y_train_scaled = y_scaler.transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # Save pre-transformed arrays
        np.save(os.path.join(fold_dir, "X_train_clean_pca.npy"), X_train_clean_pca)
        np.save(os.path.join(fold_dir, "X_train_aug_pca.npy"), X_train_aug_pca)
        np.save(os.path.join(fold_dir, "X_test_pca.npy"), X_test_pca)
        np.save(os.path.join(fold_dir, "y_train_scaled.npy"), y_train_scaled)
        np.save(os.path.join(fold_dir, "y_test_scaled.npy"), y_test_scaled)
        np.save(os.path.join(fold_dir, "y_test.npy"), y_test)
        np.save(os.path.join(fold_dir, "X_test_clean.npy"), X_test_clean)

        print(f"  Saved to {fold_dir}")

    print(f"\nPreprocessing complete. Artifacts in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="ASTRAI shared preprocessing")
    parser.add_argument("--config", default="configs/default_split.yaml")
    parser.add_argument("--out", default="preprocessed",
                        help="Output directory for preprocessing artifacts")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_preprocessing(cfg, out_dir=args.out)


if __name__ == "__main__":
    main()
