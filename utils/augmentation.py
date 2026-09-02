"""
astrai.augmentation - Data augmentation pipeline for light-curve training.

Combines additive Gaussian noise with LSST-realistic cadence degradation
(sun masking + cloud masking) followed by linear interpolation to fill
masked epochs.  This encourages the network to generalize across
observational conditions rather than overfitting to uniform-cadence data.
"""
import numpy as np
from utils import lsst


def add_gaussian_noise_slow(x, noise_std):
    """Add i.i.d. Gaussian noise to each element (fully random, slower).

    Parameters
    ----------
    x : numpy.ndarray
        Input batch of shape ``(n_samples, series_length)``.
    noise_std : float
        Standard deviation of the additive noise.

    Returns
    -------
    numpy.ndarray
        Noisy copy of *x* with the same shape.
    """
    noise = np.random.randn(*x.shape)
    return x + noise_std * noise


def add_gaussian_noise(x, noise_std):
    """Add Gaussian noise using a tiled pseudo-random vector (fast variant).

    Generates a single random vector of length ``n_samples`` and tiles it
    across the series dimension.  This is ~2x faster than full-random
    sampling for large batches while still providing sufficient
    perturbation for regularization purposes.

    Parameters
    ----------
    x : numpy.ndarray
        Input batch of shape ``(n_samples, series_length)``.
    noise_std : float
        Standard deviation of the additive noise.

    Returns
    -------
    numpy.ndarray
        Noisy copy of *x* with the same shape.
    """
    fast_pseudo_rands = np.tile(np.random.randn((len(x))), len(x[0]))
    return x + noise_std * fast_pseudo_rands.reshape(*x.shape)


def add_exp_gaussian_log_noise(x, sigma=1.0, eps=1e-12, random_state=None):
    """
    Apply exp, add Gaussian noise proportional to sqrt(value),
    then take log and return.

    Parameters
    ----------
    x : array-like
        Input values (log-scale).
    sigma : float
        Noise scale factor.
    eps : float
        Small value to avoid log(0).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    noisy_x : np.ndarray
        Noisy values in log-scale.
    """

    x = np.asarray(x, dtype=float)

    if random_state is not None:
        np.random.seed(random_state)

    # Go to linear space
    y = np.exp(x)

    # Standard deviation proportional to sqrt(y)
    std = sigma * np.sqrt(y)

    # Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=std, size=y.shape)
    y_noisy = y + noise

    # Avoid negative / zero values
    y_noisy = np.maximum(y_noisy, eps)

    # Back to log space
    return np.log(y_noisy), np.log(y + std) - np.log(y - std)


def apply_lsst_pipeline(
    curves_batch,
    n_days,
    noise_std,
    samples_per_day=None,
):
    """Apply noise and LSST cadence degradation to light curves.

    For each curve:
    1. Add noise to the full-cadence curve.
    2. Generate stochastic sun and cloud masks.
    3. Retain the epochs that survive both masks.
    4. Interpolate the retained samples onto the original grid.

    Parameters
    ----------
    curves_batch : numpy.ndarray
        Clean light curves with shape ``(n_samples, n_days)``.
    n_days : int
        Number of time steps per curve.
    noise_std : float
        Standard deviation passed to the existing noise function.
    samples_per_day : int, optional
        Digital samples per day. Defaults to
        ``lsst.DIG_SAMPLES_X_DAY``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The augmented curves and a boolean mask identifying the samples
        retained before interpolation. Both arrays have the same shape as
        ``curves_batch``.
    """
    if samples_per_day is None:
        samples_per_day = lsst.DIG_SAMPLES_X_DAY

    augmented = curves_batch.copy()

    augmented = add_gaussian_noise(augmented, noise_std)
    retained_mask = np.zeros_like(augmented, dtype=bool)

    calendar = np.arange(n_days) / samples_per_day

    for i, _ in enumerate(augmented):
        sun_mask = lsst.sun_masking_np(calendar)
        cloud_mask = lsst.random_cloud_masking(np.ones_like(calendar))
        combined_mask = (1 - sun_mask) * (1 - cloud_mask)

        curve = augmented[i]
        valid_idx = np.where(combined_mask == 1)[0]
        retained_mask[i, valid_idx] = True

        if len(valid_idx) < 2:
            continue

        valid_vals = curve[valid_idx]
        augmented[i] = np.interp(np.arange(n_days), valid_idx, valid_vals)

    return augmented, retained_mask
