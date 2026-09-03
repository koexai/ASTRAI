"""
astrai.metrics - Regression evaluation metrics for model assessment.

All functions operate on NumPy arrays and follow the convention
``(y_true, y_pred)`` for argument ordering.
"""
import numpy as np

from utils.parameter_validation import validate_parameter_names


METRIC_NAMES = ("RMSE", "RRMSE", "MAE", "R2")


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


def compute_parameter_metrics(true, pred, param_names):
    """Compute named metrics for every parameter and their aggregate means.

    The aggregate values preserve the existing characterizer convention: each
    metric is computed independently for every output column and then averaged
    without weighting across parameters.

    Parameters
    ----------
    true : numpy.ndarray
        Ground-truth values with shape ``(n_samples, n_params)``.
    pred : numpy.ndarray
        Predicted values with the same shape as *true*.
    param_names : sequence of str
        Ordered names associated with the parameter columns.

    Returns
    -------
    dict
        ``aggregate`` contains one scalar per metric. ``per_parameter`` maps
        each configured parameter name to its RMSE, RRMSE, MAE and R2 values.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)

    if true.ndim != 2 or pred.ndim != 2:
        raise ValueError(
            "Parameter targets and predictions must be two-dimensional"
        )
    if true.shape != pred.shape:
        raise ValueError(
            "Parameter targets and predictions must have the same shape: "
            f"{true.shape} != {pred.shape}"
        )
    if true.shape[0] == 0:
        raise ValueError("Parameter metrics require at least one sample")

    names = validate_parameter_names(true.shape[1], param_names)
    metric_functions = {
        "RMSE": get_rmse,
        "RRMSE": get_rrmse,
        "MAE": get_mae,
        "R2": get_r_squared,
    }
    per_parameter = {}

    for column, name in enumerate(names):
        per_parameter[name] = {
            metric_name: float(
                metric_function(true[:, column], pred[:, column])
            )
            for metric_name, metric_function in metric_functions.items()
        }

    aggregate = {
        metric_name: float(
            np.mean(
                [
                    parameter_metrics[metric_name]
                    for parameter_metrics in per_parameter.values()
                ]
            )
        )
        for metric_name in METRIC_NAMES
    }
    return {
        "aggregate": aggregate,
        "per_parameter": per_parameter,
    }


def compute_metrics(true, pred, n_cols=None):
    """Compute regression metrics between ground truth and predictions.

    When ``n_cols`` is provided, metrics are computed per-column and then
    averaged (used for characterization with multiple physical parameters).
    Otherwise, arrays are flattened before computing (used for generation).

    Parameters
    ----------
    true : numpy.ndarray
        Ground-truth values.
    pred : numpy.ndarray
        Predicted values (same shape as *true*).
    n_cols : int, optional
        Number of columns for per-column averaging. If ``None``, arrays
        are flattened.

    Returns
    -------
    dict
        Dictionary with keys ``"R2"``, ``"RMSE"``, ``"RRMSE"``, ``"MAE"``.
        Values are scalars when n_cols is None, tuples (mean, std) when
        n_cols is provided.
    """
    if n_cols is not None:
        rmse_vals = [get_rmse(true[:, i], pred[:, i]) for i in range(n_cols)]
        rrmse_vals = [get_rrmse(true[:, i], pred[:, i]) for i in range(n_cols)]
        mae_vals = [get_mae(true[:, i], pred[:, i]) for i in range(n_cols)]
        r2_vals = [get_r_squared(true[:, i], pred[:, i]) for i in range(n_cols)]
        return {
            "R2": (np.mean(r2_vals), np.std(r2_vals)),
            "RMSE": (np.mean(rmse_vals), np.std(rmse_vals)),
            "RRMSE": (np.mean(rrmse_vals), np.std(rrmse_vals)),
            "MAE": (np.mean(mae_vals), np.std(mae_vals)),
        }
    rmse = get_rmse(true.ravel(), pred.ravel())
    rrmse = get_rrmse(true.ravel(), pred.ravel())
    mae = get_mae(true.ravel(), pred.ravel())
    r2 = get_r_squared(true.ravel(), pred.ravel())
    return {"R2": r2, "RMSE": rmse, "RRMSE": rrmse, "MAE": mae}
