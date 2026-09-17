"""Phenomenological masking diagnostics and historical LSST helper APIs.

The canonical pipeline and demo use astrai.utils.masking. The astronomical
profile helpers and cumulative-budget sampler below are retained only as
historical APIs; they are not an observing-geometry or survey simulation.
"""
import numpy as np
import matplotlib.pyplot as plt

from astrai.utils.masking import (
    MaskingConfig, build_time_axis, generate_masking,
)

# ---------------------------------------------------------------------------
# Astronomical / site constants
# ---------------------------------------------------------------------------
ROT_PER_DAY = np.pi / 180.0  # radians per degree
DAY_PER_ROT = 180.0 / np.pi  # degrees per radian

SUN_PERIOD = 365.25  # Earth orbital period [days]
MOON_PERIOD = 29.53  # Synodic lunar period  [days]

CONSECUTIVE_CLOUDY_DAYS = 2.3  # Mean length of a cloudy spell [days]
CLOUDY_PROB = 30  # Fraction of time lost to clouds [%]
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


def random_cloud_masking(arr, percentage=30, seed=None, rng=None, *, samples_per_day=1):
    """Return a copy with cloudy samples zeroed (one means clear for unit input).

    Weather is defined in days, with stationary exponential episodes of mean
    2.3 days and the requested nominal occupancy. No index wrapping occurs.
    The historical seed argument remains supported with a local RNG.
    """
    if seed is not None and rng is not None:
        raise ValueError("seed and rng cannot be supplied together.")
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Cloud masking expects a non-empty one-dimensional array")
    config = MaskingConfig(cloudy_fraction=percentage / 100)
    times = build_time_axis(len(values), samples_per_day)
    local = np.random.default_rng(seed) if rng is None else rng
    components = generate_masking(times, config, local)
    return np.where(components["cloud_blocked"], 0.0, values)


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


def get_masks(rng=None, *, n_samples=N_SAMPLES, samples_per_day=1, masking_config=None):
    """Return canonical profiles using the historical six-array tuple layout.

    The combined array is availability in equivalent hours; solar/cloud arrays
    indicate blocked samples. Use generate_masking for the actual retained mask.
    """
    parts = generate_masking(build_time_axis(n_samples, samples_per_day), masking_config, rng)
    return (parts["time_days"], 24 * parts["availability"], parts["daylight_hours"],
            parts["moon_contribution"], parts["sun_blocked"], parts["cloud_blocked"])


def plot_masking_components(parts):
    """Show the same realised mask used for sampling, not a second RNG draw."""
    time = parts["time_days"]
    mask = parts["retained_mask"]
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(8, 10))
    for ax, key, label in zip(axes[:4],
            ("daylight_hours", "moon_contribution", "sun_blocked", "cloud_blocked"),
            ("Daylight (h)", "Moon contribution", "Sun blocked", "Cloud blocked")):
        ax.plot(time, parts[key])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[4].plot(time, parts["availability"], label="Availability q(t)")
    axes[4].step(time, mask.astype(float), where="post", alpha=0.4, label="Retained mask")
    axes[4].set_ylabel("Combined")
    axes[4].legend(loc="upper right")
    observed = time[mask]
    axes[5].plot(observed, np.ones(len(observed)), "|")
    axes[5].set_ylabel("Sampling")
    if len(observed) >= 2:
        axes[6].plot((observed[:-1] + observed[1:]) / 2, np.diff(observed), ".")
        axes[6].set_yscale("log")
    axes[6].set_ylabel("Gap (days)")
    axes[6].set_xlabel("Days from first clean sample")
    fig.suptitle("Phenomenological bolometric observation masking")
    fig.tight_layout()
    return fig


def _demo_plot(seed=42, *, n_samples=N_SAMPLES, samples_per_day=1, masking_config=None):
    parts = generate_masking(build_time_axis(n_samples, samples_per_day),
                             masking_config, np.random.default_rng(seed))
    return plot_masking_components(parts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot phenomenological masking components")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--samples-per-day", type=float, default=1)
    parser.add_argument("--output", help="Save a figure instead of opening a window")
    args = parser.parse_args()
    figure = _demo_plot(args.seed, n_samples=args.n_samples, samples_per_day=args.samples_per_day)
    if args.output:
        figure.savefig(args.output)
    else:
        plt.show()
