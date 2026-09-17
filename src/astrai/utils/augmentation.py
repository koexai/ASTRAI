"""
astrai.augmentation - Data augmentation pipeline for light-curve training.

Combines the historical additive noise with phenomenological missing-data
masks and interpolation in log10 bolometric luminosity. No survey cadence
or detector geometry is simulated.
"""
from numbers import Real

import numpy as np
from astrai.utils.masking import (
    MaskingConfig, build_time_axis, generate_masking, interpolate_observations,
)


_MAX_POSITIVE_NOISE_DRAWS = 128


def _noise_inputs(log10_luminosity, rng):
    """Validate the new kernels without coercing non-real data or global RNGs."""
    values = np.asarray(log10_luminosity)
    if values.ndim not in (1, 2) or values.dtype.kind not in "iuf":
        raise ValueError("log10_luminosity must be a real numeric 1D or 2D array")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("log10_luminosity must contain finite float64 values")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None")
    return values


def _noise_scalar(value, name, *, non_negative=True):
    """Require finite real scalars, excluding booleans and array parameters."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(value) or (non_negative and value < 0):
        constraint = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{name} must be {constraint}")
    return value


def add_iid_gaussian_noise_in_log10_luminosity(
    log10_luminosity, *, sigma_dex, rng=None
):
    """Perturb bolometric log10 luminosities independently with Gaussian noise.

    This empirical augmentation baseline softens idealised semi-analytical
    curves and supports noise diagnostics; it is not a calibrated detector
    simulation. For each element, ``x_new = x + sigma_dex * Z``, with
    independent standard normal ``Z``. In linear luminosity this is
    multiplicative lognormal noise: the median luminosity is preserved, while
    its mean is multiplied by ``exp((ln(10) * sigma_dex)**2 / 2)``.

    Parameters
    ----------
    log10_luminosity : array-like
        Finite real 1D curve or 2D batch, in log10(L_bol / [erg s^-1]).
    sigma_dex : float
        Finite, non-negative standard deviation in dex, independent of
        luminosity. Zero returns an identical copy without consuming the RNG.
    rng : numpy.random.Generator, optional
        Explicit local generator. If omitted, a local entropy-seeded generator
        is created when needed; NumPy's global random state is never used.

    Returns
    -------
    numpy.ndarray
        Perturbed log10 luminosities, float64, with the input shape. The input
        is not modified. An empty curve or batch returns an empty copy.

    Raises
    ------
    ValueError
        If data or parameters are invalid, or the result is not finite.
    TypeError
        If rng is neither a Generator nor None.
    """
    values = _noise_inputs(log10_luminosity, rng)
    sigma_dex = _noise_scalar(sigma_dex, "sigma_dex")
    if sigma_dex == 0 or values.size == 0:
        return values
    if rng is None:
        rng = np.random.default_rng()
    with np.errstate(over="ignore", invalid="ignore"):
        result = values + sigma_dex * rng.standard_normal(values.shape)
    if not np.isfinite(result).all():
        raise ValueError("Perturbed log10 luminosity is not finite")
    return result


def add_heteroscedastic_noise_in_normalised_luminosity(
    log10_luminosity, *, a, b, x_ref=42.0, rng=None
):
    """Perturb normalised bolometric luminosity with positive Gaussian draws.

    An effective model for perturbing idealised semi-analytical curves, not
    a calibrated LSST detector model. Set ``f = 10**(x - x_ref)`` and propose
    ``f_new = f + sqrt(a*f + b)*Z`` independently at every element. Resample
    only non-positive proposals, always using the original f and variance.
    Return ``x_ref + log10(f_new)``.

    The accepted distribution is N(f, a*f+b) conditional on f_new > 0.
    Consequently a*f+b is the variance BEFORE truncation, not the output
    variance. Truncation raises the mean, negligibly at high signal-to-noise
    but substantially for faint values. For b > 0 and f tending to zero,
    the mean tends to sqrt(b)*sqrt(2/pi). No clipping floor or detection mask
    is applied. Input x=0 is a numeric log10 value, not a missing-data marker.

    Parameters
    ----------
    log10_luminosity : array-like
        Finite real 1D curve or 2D batch, in log10(L_bol / [erg s^-1]).
    a, b : float
        Finite, non-negative coefficients of the dimensionless normalised
        variance. a*f is source-dependent; b is constant. Their numerical
        values depend on x_ref. The nominal relative error is sqrt(a*f+b)/f.
        Both zero return an identical copy without consuming the RNG.
    x_ref : float, default 42.0
        Finite normalisation convention, NOT a physically calibrated value.
        It defines L_ref = 10**x_ref erg/s, shared by all curves. Changing it
        requires rescaling a and b to preserve a given physical perturbation:
        if f_new_units = c*f, use a_new = c*a and b_new = c**2*b.
    rng : numpy.random.Generator, optional
        Local generator; an entropy-seeded local generator is used if omitted.
        Identical inputs, order, parameters and initial RNG state repeat the
        output. Different batch partitioning need not repeat individual draws
        because resampling changes RNG consumption. Global state is untouched.

    Returns
    -------
    numpy.ndarray
        Perturbed log10 luminosities, float64, with the input shape and no
        in-place changes. An empty curve or batch returns an empty copy.

    Raises
    ------
    ValueError
        For invalid inputs, non-finite arithmetic, or normalisation/variance
        underflow to zero. No silent numerical clipping is performed.
    TypeError
        If rng is neither a Generator nor None.
    RuntimeError
        If positive sampling exceeds the internal limit of 128 draws per
        element. No alternative distribution is substituted on failure.
    """
    values = _noise_inputs(log10_luminosity, rng)
    a = _noise_scalar(a, "a")
    b = _noise_scalar(b, "b")
    x_ref = _noise_scalar(x_ref, "x_ref", non_negative=False)
    if (a == 0 and b == 0) or values.size == 0:
        return values
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalised = np.power(10.0, values - x_ref).ravel()
        variance = a * normalised + b
    if not np.isfinite(normalised).all() or np.any(normalised <= 0):
        raise ValueError("Normalised luminosity must be finite and strictly positive")
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Noise variance must be finite and strictly positive")
    scale = np.sqrt(variance)
    if rng is None:
        rng = np.random.default_rng()
    result = np.empty_like(normalised)
    pending = np.arange(normalised.size)
    # Positivity is equivalent to Z > Z_min = -f/scale. Since Z_min < 0,
    # acceptance exceeds 1/2, so rejection sampling needs <2 draws on average.
    # Check the actual proposal too, to exclude cancellation to zero.
    for _ in range(_MAX_POSITIVE_NOISE_DRAWS):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            proposals = normalised[pending] + scale[pending] * rng.standard_normal(pending.size)
        if not np.isfinite(proposals).all():
            raise ValueError("Perturbed normalised luminosity is not finite")
        accepted = proposals > 0
        result[pending[accepted]] = proposals[accepted]
        pending = pending[~accepted]
        if pending.size == 0:
            break
    else:
        raise RuntimeError("Positive noise sampling exceeded 128 draws per element")
    with np.errstate(over="ignore", invalid="ignore"):
        result = x_ref + np.log10(result.reshape(values.shape))
    if not np.isfinite(result).all():
        raise ValueError("Perturbed log10 luminosity is not finite")
    return result


def add_gaussian_noise_slow(x, noise_std, rng=None):
    """Add i.i.d. Gaussian noise to each element (fully random, slower).

    Deprecated: retained unchanged for historical reproduction and backwards
    compatibility. New direct callers should use the explicit luminosity
    kernels above. No runtime deprecation warning is emitted.

    Parameters
    ----------
    x : numpy.ndarray
        Input batch of shape ``(n_samples, series_length)``.
    noise_std : float
        Standard deviation of the additive noise.
    rng : numpy.random.Generator, optional
        Random-number generator. A local entropy-seeded generator is created
        when omitted.

    Returns
    -------
    numpy.ndarray
        Noisy copy of *x* with the same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(x.shape)
    return x + noise_std * noise


def add_gaussian_noise(x, noise_std, rng=None):
    """Add Gaussian noise using a tiled pseudo-random vector (fast variant).

    Deprecated: retained unchanged for historical reproduction and backwards
    compatibility, including its use by the current augmentation pipeline.
    New direct callers should use the explicit luminosity kernels above.
    No runtime deprecation warning is emitted.

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
    rng : numpy.random.Generator, optional
        Random-number generator. A local entropy-seeded generator is created
        when omitted.

    Returns
    -------
    numpy.ndarray
        Noisy copy of *x* with the same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    fast_pseudo_rands = np.tile(rng.standard_normal(len(x)), len(x[0]))
    return x + noise_std * fast_pseudo_rands.reshape(*x.shape)


def add_exp_gaussian_log_noise(
    x,
    sigma=1.0,
    eps=1e-12,
    random_state=None,
    rng=None,
):
    """
    Apply exp, add Gaussian noise proportional to sqrt(value),
    then take log and return.

    Deprecated: retained unchanged for historical reproduction and backwards
    compatibility, including its natural-log convention and two-array return.
    New direct callers should use the explicit luminosity kernels above.
    No runtime deprecation warning is emitted.

    Parameters
    ----------
    x : array-like
        Input values (log-scale).
    sigma : float
        Noise scale factor.
    eps : float
        Small value to avoid log(0).
    random_state : int or None
        Seed used to create a local generator for backwards compatibility.
    rng : numpy.random.Generator or None
        Explicit random-number generator. Cannot be combined with
        ``random_state``.

    Returns
    -------
    noisy_x : np.ndarray
        Noisy values in log-scale.
    """

    x = np.asarray(x, dtype=float)

    if random_state is not None and rng is not None:
        raise ValueError("random_state and rng cannot be supplied together.")
    if rng is None:
        rng = (
            np.random.default_rng()
            if random_state is None
            else np.random.RandomState(random_state)
        )

    # Go to linear space
    y = np.exp(x)

    # Standard deviation proportional to sqrt(y)
    std = sigma * np.sqrt(y)

    # Add Gaussian noise
    noise = rng.normal(loc=0.0, scale=std, size=y.shape)
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
    rng=None,
    *,
    masking_config=None,
):
    """Perturb curves and interpolate phenomenologically retained observations.

    The historical function name and (curves, boolean retained_mask) return
    contract remain supported. ``n_days`` is the number of time samples, not
    the duration. Masking is specified by ``MaskingConfig`` or a parameter
    mapping; omitted parameters use its documented defaults. The legacy noise
    path and shared local RNG are unchanged. Outputs are converted to float32
    by model/preprocessing callers as before.

    Zero observations raise an error identifying the batch row; one produces
    a constant curve. Two or more use linear interpolation in log10 luminosity
    with constant edges. Hidden values never fill gaps and masks are not redrawn.
    """
    raw = np.asarray(curves_batch)
    if raw.ndim != 2 or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("Curves must be a finite real two-dimensional array")
    calendar = build_time_axis(n_days, samples_per_day)
    if raw.shape[1] != len(calendar):
        raise ValueError("n_days must match the number of curve samples")
    config = MaskingConfig.from_mapping(masking_config)
    rng = np.random.default_rng() if rng is None else rng
    if len(raw) == 0:
        return raw.astype(np.float64, copy=True), np.zeros(raw.shape, dtype=bool)
    augmented = add_gaussian_noise(raw.astype(np.float64, copy=True), noise_std, rng=rng)
    retained_mask = np.zeros(raw.shape, dtype=bool)
    for row, curve in enumerate(augmented):
        mask = generate_masking(calendar, config, rng)["retained_mask"]
        if not mask.any():
            raise ValueError(f"No observations retained for curve {row}; the mask is not resampled")
        retained_mask[row] = mask
        augmented[row] = interpolate_observations(curve, mask, calendar)
    return augmented, retained_mask
