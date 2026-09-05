"""
astrai.lsst - Simulation of LSST-like observing conditions.

This module models the key observational constraints of the Vera C. Rubin
Observatory (LSST) that affect ground-based time-domain surveys:

* **Solar contamination** -- daylight hours and sun-elevation masking.
* **Lunar contamination** -- phase-dependent moon brightness and proximity.
* **Weather** -- stochastic consecutive-cloudy-night masking.
* **Cadence** -- non-uniform temporal sampling derived from the observable
  sky-area budget.

The functions are vectorized with NumPy and designed to be called
per-light-curve during data augmentation (see ``astrai.augmentation``).

References
----------
LSST Science Book, v2.0 (arXiv:0912.0201)
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Astronomical / site constants
# ---------------------------------------------------------------------------
ROT_PER_DAY = np.pi / 180.0  # radians per degree
DAY_PER_ROT = 180.0 / np.pi  # degrees per radian

SUN_PERIOD = 365.25  # Earth orbital period [days]
MOON_PERIOD = 29.53  # Synodic lunar period  [days]

CONSECUTIVE_CLOUDY_DAYS = 2.3  # Mean length of a cloudy spell [days]
CLOUDY_PROB = 20  # Fraction of time lost to clouds [%]
EEPS = 23.44  # Earth axial tilt [degrees]
LATITUDE_LSST = 30  # Cerro Pachon latitude [degrees]
PHI = LATITUDE_LSST * ROT_PER_DAY  # Site latitude [radians]

GOOD_MOON_ANGLE = 33  # Min angular distance from the moon [deg]
GOOD_SUN_ANGLE = 64  # Min sun depression angle for obs [deg]
MOON_MASK_MIN_ANGLE = 6.5  # Inner moon exclusion zone [deg]
MOON_MASK_MAX_ANGLE = 20  # Outer moon exclusion zone [deg]


DIG_SAMPLES_X_DAY = 1  # Digital time-steps per day
AVG_SAMPLING_RATE = 4
MAX_OBS_LENGTH = 420  # Maximum observation baseline [days]

N_SAMPLES = MAX_OBS_LENGTH * DIG_SAMPLES_X_DAY + 1  # Total time-grid size


def _local_rng(rng):
    """Return an explicit generator without touching NumPy's global state."""
    return np.random.default_rng() if rng is None else rng


def daylight_hours_np(days, day0=None, rng=None):
    """Compute daylight duration for a given array of Julian-like days.

    Uses the classical sunrise-equation approximation with the Earth's
    axial tilt (obliquity) and the LSST site latitude.

    Parameters
    ----------
    days : array_like
        Day indices (integer or float).
    day0 : float, optional
        Phase offset for the solar cycle. Randomised if not provided.
    rng : numpy.random.Generator, optional
        Generator used when ``day0`` is omitted.

    Returns
    -------
    numpy.ndarray
        Daylight hours for each entry in *days*.
    """
    days = np.asarray(days, dtype=np.float64)

    if day0 is None:
        day0 = _local_rng(rng).random() * SUN_PERIOD

    # Solar declination (radians)
    delta = EEPS * np.sin(2.0 * np.pi * (days - day0) / SUN_PERIOD)
    delta *= ROT_PER_DAY

    # Hour angle (degrees)
    cos_omega0 = -np.tan(PHI) * np.tan(delta)
    cos_omega0 = np.clip(cos_omega0, -1.0, 1.0)
    omega0 = np.arccos(cos_omega0) * DAY_PER_ROT
    # Daylight hours
    return 2.0 * omega0 / 15.0


def moon_luminosity_np(days, day0=None, rng=None):
    """Compute fractional moon illumination weighted by angular proximity.

    The illuminated fraction follows a cosine model of the synodic period.
    A geometric mask zeroes out epochs when the moon is far from the
    pointing direction.

    Parameters
    ----------
    days : array_like
        Day indices.
    day0 : float, optional
        Lunar phase offset. Randomised if not provided.
    rng : numpy.random.Generator, optional
        Generator used when ``day0`` is omitted.

    Returns
    -------
    numpy.ndarray
        Effective moon luminosity contribution (0 = no contamination).
    """
    days = np.asarray(days, dtype=np.float64)

    if day0 is None:
        day0 = _local_rng(rng).random() * MOON_PERIOD

    # Phase angle [0, 2π]
    phase = 2.0 * np.pi * np.mod((days - day0) / MOON_PERIOD, 1.0)

    # Illuminated fraction
    illuminated_fraction = 0.5 * (1.0 + np.cos(phase))
    moon_presence = np.cos(phase) > np.cos(np.pi / 180 * (90 - GOOD_MOON_ANGLE))

    return illuminated_fraction * moon_presence


def sun_masking_np(days, day0=None, elev=None, rng=None):
    """Return a boolean mask where True indicates solar contamination.

    Parameters
    ----------
    days : array_like
        Day indices.
    day0 : float, optional
        Solar phase offset.
    elev : float, optional
        Target elevation above horizon [degrees].
    rng : numpy.random.Generator, optional
        Generator used when ``day0`` or ``elev`` is omitted.

    Returns
    -------
    numpy.ndarray of bool
        True where observations are blocked by sunlight.
    """
    days = np.asarray(days, dtype=np.float64)

    if day0 is None or elev is None:
        rng = _local_rng(rng)
    if day0 is None:
        day0 = rng.random() * SUN_PERIOD
    if elev is None:
        elev = rng.random() * 60

    # Phase angle [0, 2π]
    phase = 2.0 * np.pi * np.mod((days - day0) / SUN_PERIOD, 1.0)
    sun_presence = np.cos(phase) > np.cos(
        np.pi / 180 * (90 - GOOD_SUN_ANGLE + elev**2 / 60)
    )

    return sun_presence


def random_cloud_masking(arr, percentage=CLOUDY_PROB, seed=None, rng=None):
    """Apply stochastic consecutive-night cloud masking.

    Randomly selects block-start indices and zeros out contiguous spans
    of ``CONSECUTIVE_CLOUDY_DAYS * DIG_SAMPLES_X_DAY`` time-steps to
    simulate multi-night weather losses.

    Parameters
    ----------
    arr : array_like
        Input array (e.g. an observability indicator).
    percentage : int
        Approximate fraction of time-steps to mask [%].
    seed : int, optional
        Seed used to create a local generator for backwards compatibility.
    rng : numpy.random.Generator, optional
        Explicit generator. Cannot be combined with ``seed``.

    Returns
    -------
    numpy.ndarray
        Copy of *arr* with cloudy epochs set to zero.
    """
    arr = np.asarray(arr, dtype=np.float64)

    masked_arr = arr.copy()
    if seed is not None and rng is not None:
        raise ValueError("seed and rng cannot be supplied together.")
    if rng is None:
        rng = np.random.default_rng(seed)

    consecutive_cloudy_samples = CONSECUTIVE_CLOUDY_DAYS * DIG_SAMPLES_X_DAY

    n_total = masked_arr.size
    n_mask = int(
        np.round(n_total * percentage / consecutive_cloudy_samples / 100.0)
    )

    indices = rng.choice(n_total, n_mask, replace=False)

    for shift in range(int(consecutive_cloudy_samples)):
        masked_arr.flat[indices - shift] = 0

    return masked_arr


def intersections_monotone(f, g):
    """Find interpolated intersection indices of monotone *f* with values *g*.

    Given a monotonically increasing array *f* and a set of target values *g*,
    returns the (fractional) indices at which *f* crosses each value in *g*
    via linear interpolation.

    Parameters
    ----------
    f : array_like
        Monotonically increasing 1-D array.
    g : array_like
        Target values to locate in *f*.

    Returns
    -------
    numpy.ndarray
        Fractional indices into *f*.
    """
    f = np.asarray(f)
    g = np.asarray(g)
    # indices where each g would be inserted to keep f sorted
    idx = np.searchsorted(f, g)
    # clamp to valid range
    idx = np.clip(idx, 1, len(f) - 1)
    x0 = idx - 1
    x1 = idx
    f0 = f[idx - 1]
    f1 = f[idx]
    # linear interpolation — handle degenerate plateaus (f1 == f0)
    # where the cumulative budget is flat (no observing time available)
    df = f1 - f0
    safe = df != 0
    result = np.where(
        safe, x0 + (g - f0) * (x1 - x0) / np.where(safe, df, 1.0), x0
    )
    return result


def get_samples(calendar, combo):
    """Derive non-uniform LSST-like sampling epochs from the sky-area budget.

    Converts the cumulative observable-area function into approximately
    uniformly-spaced observation epochs in *observation space*, yielding
    denser sampling during better observing conditions.

    Parameters
    ----------
    calendar : array_like
        Day grid over the full baseline.

    Returns
    -------
    numpy.ndarray
        Fractional day-grid indices of the selected observations.
    """

    f = np.cumsum(combo)

    n_points = (calendar[-1] - calendar[0]) / AVG_SAMPLING_RATE
    g = np.arange(n_points) * f[-1] / n_points
    samples = intersections_monotone(f, g)
    return np.unique(samples)


def keep_only_samples_from_lc(lc, sampling):
    """Retain only the sampled epochs of a light curve, NaN-filling the rest.

    Parameters
    ----------
    lc : array_like
        Full-cadence light curve.
    sampling : array_like of int
        Indices of epochs to keep (from ``get_samples``).

    Returns
    -------
    numpy.ndarray
        Light curve with unobserved epochs set to NaN.
    """
    lc = np.asarray(lc, dtype=float)
    out = np.full_like(lc, np.nan, dtype=float)

    out.flat[sampling] = lc.flat[sampling]
    return out


def get_masks(rng=None):
    """Generate the LSST masking components and their combination."""
    rng = _local_rng(rng)
    cal = np.arange(N_SAMPLES) / DIG_SAMPLES_X_DAY
    dh = daylight_hours_np(cal, rng=rng)
    ml = moon_luminosity_np(cal, rng=rng)
    sm = sun_masking_np(cal, rng=rng)
    cm = 1 - random_cloud_masking(np.ones_like(cal), rng=rng)
    cb = (1 - cm) * ((1 - sm) * (24 - dh - ml * 4))
    return cal, cb, dh, ml, sm, cm


def _setup_axis(ax, label):
    """Configure a demo plot axis with right-side ticks."""
    ax.set_ylabel(label, rotation=0, ha="right", va="center")
    ax.tick_params(axis="y", labelleft=False, labelright=True)
    ax.yaxis.tick_right()
    ax.grid(True)


def _demo_plot(seed=42):
    """Plot all LSST masking components and the resulting sampling.
    Demonstrates the interplay of the different masking factors
    and how they combine to produce the final sampling pattern.
    The top panels show the individual contributions of daylight,
    moon luminosity, sun masking, and cloud masking.
    The bottom panels show the combined mask and the resulting sampling epochs.
    """
    (
        calendar,
        combo,
        daylight_hours,
        moon_luminosity,
        sun_masking,
        cloud_masking,
    ) = get_masks(rng=np.random.default_rng(seed))
    sampling = np.int32(get_samples(calendar, combo))

    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(5, 7))

    axes[0].plot(calendar, daylight_hours)
    _setup_axis(axes[0], "DayLight\nhours")

    axes[1].plot(calendar, moon_luminosity)
    _setup_axis(axes[1], "Moon\nmask")

    axes[2].plot(calendar, sun_masking)
    _setup_axis(axes[2], "Sun\nmask")

    axes[3].plot(calendar, cloud_masking)
    _setup_axis(axes[3], "Weather\nmask")

    axes[4].plot(calendar, combo)
    _setup_axis(axes[4], "Combined\nmask")

    axes[5].plot(sampling / DIG_SAMPLES_X_DAY, np.ones_like(sampling), "|")
    _setup_axis(axes[5], "Sampling")
    axes[5].set_ylim(0.5, 1.5)

    axes[6].plot(
        (sampling[:-1] + sampling[1:]) / 2 / DIG_SAMPLES_X_DAY,
        (sampling[1:] - sampling[:-1]) / DIG_SAMPLES_X_DAY,
        "_",
    )
    _setup_axis(axes[6], "Sampling\nlag ")
    axes[6].set_xlabel("Days")
    axes[6].set_yscale("log")

    fig.suptitle("Non-uniform sampling simulation LSST")
    plt.show()


if __name__ == "__main__":
    _demo_plot()
