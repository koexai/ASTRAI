"""
astrai.metrics - Regression evaluation metrics for model assessment.

All functions operate on NumPy arrays and follow the convention
``(y_true, y_pred)`` for argument ordering.
"""
import numpy as np


def get_mse(y, y_pred):
    """Compute Mean Squared Error between ground-truth and predictions."""
    return np.mean((y - y_pred) ** 2)


def get_rmse(y, y_pred):
    """Compute Root Mean Squared Error (square root of MSE)."""
    return np.sqrt(get_mse(y, y_pred))


def get_mae(y, y_pred):
    """Compute Mean Absolute Error between ground-truth and predictions."""
    return np.mean(np.abs(y - y_pred))


def get_r_squared(y, y_pred):
    """Compute coefficient of determination (R^2).

    Returns 1.0 for perfect predictions and can be negative when the
    model performs worse than predicting the mean.
    """
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def get_rrmse(y, y_pred):
    """Compute Relative RMSE (RMSE normalized by the mean absolute value of y).

    Falls back to raw RMSE when the mean of |y| is zero to avoid
    division-by-zero.
    """
    rmse = get_rmse(y, y_pred)
    mean_y = np.mean(np.abs(y))
    return rmse / mean_y if mean_y != 0 else rmse
