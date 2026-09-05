"""Deterministic random-number handling for ASTRAI workflows.

The project uses one configured base seed and derives independent seeds for
each stochastic stage.  Stable numeric namespaces keep a fold reproducible
regardless of which other folds or training stages run before it.
"""
import os
import random
from numbers import Integral

import numpy as np
import torch


SEED_SCHEME_VERSION = 1
_MAX_SEED = int(np.iinfo(np.uint32).max)

_PCA_NAMESPACE = 1
_AUGMENTATION_NAMESPACE = 2
_MODEL_NAMESPACE = 3
_DATA_LOADER_NAMESPACE = 4
_DIAGNOSTIC_NAMESPACE = 5

_TRAINING_STAGE_NAMESPACES = {
    "characterizer": 10,
    "generator": 11,
    "unified": 12,
}


def validate_seed(seed, name="random seed"):
    """Return *seed* as an integer after validating the shared contract."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError(f"{name} must be an integer, got {seed!r}.")

    seed = int(seed)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(
            f"{name} must be between 0 and {_MAX_SEED}, got {seed}."
        )
    return seed


def derive_seed(base_seed, *components):
    """Derive a stable uint32 seed from a base seed and numeric components."""
    entropy = [validate_seed(base_seed, "base seed")]
    entropy.extend(
        validate_seed(component, "seed namespace component")
        for component in components
    )
    sequence = np.random.SeedSequence(entropy)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def build_preprocessing_seed_plan(base_seed, n_splits):
    """Return deterministic seeds for K-fold preprocessing stages."""
    base_seed = validate_seed(base_seed, "preprocessing.random_seed")
    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(
        n_splits, Integral
    ):
        raise TypeError(
            "preprocessing.n_splits must be an integer, "
            f"got {n_splits!r}."
        )
    if n_splits < 2:
        raise ValueError(
            f"preprocessing.n_splits must be at least 2, got {n_splits}."
        )

    return {
        "scheme_version": SEED_SCHEME_VERSION,
        "base_seed": base_seed,
        "k_fold": base_seed,
        "pca": derive_seed(base_seed, _PCA_NAMESPACE),
        "augmentation": {
            f"fold_{fold_idx}": derive_seed(
                base_seed,
                _AUGMENTATION_NAMESPACE,
                fold_idx,
            )
            for fold_idx in range(1, int(n_splits) + 1)
        },
    }


def build_training_seed_plan(base_seed, stage, fold_idx):
    """Return independent model and DataLoader seeds for one training fold."""
    base_seed = validate_seed(base_seed, "training base seed")
    if stage not in _TRAINING_STAGE_NAMESPACES:
        choices = ", ".join(sorted(_TRAINING_STAGE_NAMESPACES))
        raise ValueError(
            f"Unknown training stage {stage!r}; expected one of: {choices}."
        )
    fold_idx = validate_seed(fold_idx, "fold index")
    if fold_idx == 0:
        raise ValueError("fold index must be at least 1.")

    stage_namespace = _TRAINING_STAGE_NAMESPACES[stage]
    return {
        "scheme_version": SEED_SCHEME_VERSION,
        "base_seed": base_seed,
        "stage": stage,
        "fold": fold_idx,
        "model": derive_seed(
            base_seed,
            stage_namespace,
            fold_idx,
            _MODEL_NAMESPACE,
        ),
        "data_loader": derive_seed(
            base_seed,
            stage_namespace,
            fold_idx,
            _DATA_LOADER_NAMESPACE,
        ),
    }


def build_unified_preprocessing_seed_plan(base_seed, fold_idx):
    """Return fold-local PCA and augmentation seeds for legacy training."""
    base_seed = validate_seed(base_seed, "training.random_seed")
    fold_idx = validate_seed(fold_idx, "fold index")
    if fold_idx == 0:
        raise ValueError("fold index must be at least 1.")
    return {
        "pca": derive_seed(base_seed, _PCA_NAMESPACE, fold_idx),
        "augmentation": derive_seed(
            base_seed,
            _AUGMENTATION_NAMESPACE,
            fold_idx,
        ),
    }


def make_numpy_rng(seed):
    """Create a local NumPy generator from a validated seed."""
    return np.random.default_rng(validate_seed(seed))


def derive_diagnostic_seed(base_seed, sample_index):
    """Derive a sample-local seed for reconstruction diagnostics."""
    return derive_seed(
        base_seed,
        _DIAGNOSTIC_NAMESPACE,
        validate_seed(sample_index, "sample index"),
    )


def configure_torch_determinism(seed):
    """Seed training libraries and request deterministic PyTorch algorithms."""
    seed = validate_seed(seed, "PyTorch seed")

    # Required by deterministic CUDA matrix multiplication on supported CUDA
    # versions.  ``setdefault`` respects an explicit user configuration.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_torch_generator(seed):
    """Create an explicitly seeded CPU generator for a DataLoader."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(validate_seed(seed, "DataLoader seed"))
    return generator


def seed_data_loader_worker(_worker_id):
    """Seed Python and NumPy from the worker seed assigned by PyTorch."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
