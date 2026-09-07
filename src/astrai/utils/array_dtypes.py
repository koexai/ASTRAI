"""Shared dtype contract for persisted and model-ready NumPy arrays."""

from pathlib import Path

import numpy as np


MODEL_ARRAY_DTYPE = np.dtype(np.float32)
INDEX_ARRAY_DTYPE = np.dtype(np.int64)


def _as_real_numeric_array(values, *, name):
    """Return ``values`` as an array after validating its numeric dtype."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise TypeError(
            f"{name} must contain real numeric values, got {array.dtype}."
        )
    return array


def as_model_array(values):
    """Return a real numeric array in the model dtype (``float32``)."""
    array = _as_real_numeric_array(values, name="Model array")
    return array.astype(MODEL_ARRAY_DTYPE, copy=False)


def as_index_array(values):
    """Return an integer array in the persisted index dtype (``int64``)."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"Index array must contain integers, got {array.dtype}.")
    return array.astype(INDEX_ARRAY_DTYPE, copy=False)


def load_model_array(path):
    """Load a model array and normalise legacy numeric dtypes to ``float32``."""
    return as_model_array(np.load(Path(path), allow_pickle=False))
