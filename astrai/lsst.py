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
ROT_PER_DAY = np.pi / 180.0        # radians per degree
DAY_PER_ROT = 180.0 / np.pi        # degrees per radian

SUN_PERIOD = 365.25                 # Earth orbital period [days]
MOON_PERIOD = 29.53                 # Synodic lunar period  [days]

CONSECUTIVE_CLOUDY_DAYS = 2.3       # Mean length of a cloudy spell [days]
CLOUDY_PROB = 20                    # Fraction of time lost to clouds [%]
CLOUD_SEED = 42                     # Default RNG seed for reproducibility

EEPS = 23.44                        # Earth axial tilt [degrees]
LATITUDE_LSST = 30                  # Cerro Pachon latitude [degrees]
PHI = LATITUDE_LSST * ROT_PER_DAY  # Site latitude [radians]

GOOD_MOON_ANGLE = 33                # Min angular distance from the moon [deg]
GOOD_SUN_ANGLE = 64                 # Min sun depression angle for obs [deg]
MOON_MASK_MIN_ANGLE = 6.5           # Inner moon exclusion zone [deg]
MOON_MASK_MAX_ANGLE = 20            # Outer moon exclusion zone [deg]


DIG_SAMPLES_X_DAY = 1               # Digital time-steps per day
AVG_SAMPLING_RATE = 4
MAX_OBS_LENGTH = 420                # Maximum observation baseline [days]

N_SAMPLES = MAX_OBS_LENGTH * DIG_SAMPLES_X_DAY + 1  # Total time-grid size


def daylight_hours_np(days, day0=None):
    """Compute daylight duration for a given array of Julian-like days.

    Uses the classical sunrise-equation approximation with the Earth's
    axial tilt (obliquity) and the LSST site latitude.

    Parameters
    ----------
    days : array_like
        Day indices (integer or float).
    day0 : float, optional
        Phase offset for the solar cycle. Randomized if not provided.

    Returns
    -------
    numpy.ndarray
        Daylight hours for each entry in *days*.
    """
    days = np.asarray(days, dtype=np.float64)

    if day0 is None:
        day0 = np.random.rand(1) * SUN_PERIOD
    
    # Solar declination (radians)
    delta = EEPS * np.sin(2.0 * np.pi * (days - day0) / SUN_PERIOD)
    delta *= ROT_PER_DAY
    
    # Hour angle (degrees)
    cos_omega0 = -np.tan(PHI) * np.tan(delta)
    cos_omega0 = np.clip(cos_omega0, -1.0, 1.0)
    omega0 = np.arccos(cos_omega0) * DAY_PER_ROT
    # Daylight hours
    return 2.0 * omega0 / 15.0


def moon_luminosity_np(days, day0=None):
    """Compute fractional moon illumination weighted by angular proximity.

    The illuminated fraction follows a cosine model of the synodic period.
    A geometric mask zeroes out epochs when the moon is far from the
    pointing direction.

    Parameters
    ----------
    days : array_like
        Day indices.
    day0 : float, optional
        Lunar phase offset. Randomized if not provided.

    Returns
    -------
    numpy.ndarray
        Effective moon luminosity contribution (0 = no contamination).
    """
    days = np.asarray(days, dtype=np.float64)

    if day0 is None:
        day0 = np.random.rand(1) * MOON_PERIOD
    
    # Phase angle [0, 2π]
    phase = 2.0 * np.pi * np.mod((days - day0) / MOON_PERIOD, 1.0)
    
    # Illuminated fraction
    illuminated_fraction = 0.5 * (1.0 + np.cos(phase))
    moon_presence = np.cos(phase) > np.cos(np.pi / 180 * (90 - GOOD_MOON_ANGLE))

    return illuminated_fraction * moon_presence


def sun_masking_np(days, day0=None, elev=None):
    """Return a boolean mask where True indicates solar contamination.

    Parameters
    ----------
    days : array_like
        Day indices.
    day0 : float, optional
        Solar phase offset.
    elev : float, optional
        Target elevation above horizon [degrees].

    Returns
    -------
    numpy.ndarray of bool
        True where observations are blocked by sunlight.
    """
    days = np.asarray(days, dtype=np.float64)

    if day0 is None:
        day0 = np.random.rand(1) * SUN_PERIOD
    if elev is None:
        elev = np.random.rand(1) * 60
    
    # Phase angle [0, 2π]
    phase = 2.0 * np.pi * np.mod((days - day0) / SUN_PERIOD, 1.0)
    sun_presence = np.cos(phase) > np.cos(np.pi / 180 * (90 - GOOD_SUN_ANGLE + elev**2 / 60))

    return sun_presence


def random_cloud_masking(arr, percentage=CLOUDY_PROB, seed=None):
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
        RNG seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Copy of *arr* with cloudy epochs set to zero.
    """
    arr = np.asarray(arr, dtype=np.float64)

    if seed is None:
        seed = np.random.randint(100000, size=1)
    masked_arr = arr.copy()

    rng = np.random.RandomState(seed)

    consecutive_cloudy_samples = CONSECUTIVE_CLOUDY_DAYS * DIG_SAMPLES_X_DAY

    n_total = masked_arr.size
    n_mask = int(np.round(n_total * percentage / consecutive_cloudy_samples / 100.0))

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
    result = np.where(safe, x0 + (g - f0) * (x1 - x0) / np.where(safe, df, 1.0), x0)
    return result


def get_samples(calendar,combo):
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


def get_masks():
    calendar = np.arange(N_SAMPLES) / DIG_SAMPLES_X_DAY
    daylight_hours = daylight_hours_np(calendar)
    moon_luminosity = moon_luminosity_np(calendar)
    sun_masking = sun_masking_np(calendar)
    cloud_masking = 1-random_cloud_masking(np.ones_like(calendar))
    combo = (1-cloud_masking)*((1-sun_masking)*(24-daylight_hours-moon_luminosity*4))
    return calendar, combo, daylight_hours, moon_luminosity, sun_masking, cloud_masking


if __name__ == "__main__":
    
    calendar, combo, daylight_hours, moon_luminosity, sun_masking, cloud_masking = get_masks()
    
    sampling = np.int32(get_samples(calendar,combo))
    
    fig, axes = plt.subplots(
        7, 1,
        sharex=True,
        figsize=(5, 7),
        #constrained_layout=True
    )
    ax_id = 0
    
    # --- Night hours ---
    axes[ax_id].plot(calendar, daylight_hours)
    axes[ax_id].set_ylabel("DayLight\nhours", rotation=0, ha='right', va='center')
    axes[ax_id].tick_params(axis='y', labelleft=False, labelright=True)
    axes[ax_id].yaxis.tick_right()
    axes[ax_id].grid(True)
    ax_id += 1
    
    # --- Moon light masking ---
    axes[ax_id].plot(calendar, moon_luminosity)
    axes[ax_id].set_ylabel("Moon\nmask", rotation=0, ha='right', va='center')
    axes[ax_id].tick_params(axis='y', labelleft=False, labelright=True)
    axes[ax_id].yaxis.tick_right()
    axes[ax_id].grid(True)
    ax_id += 1
    
    # --- Sun masking ---
    axes[ax_id].plot(calendar, sun_masking)
    axes[ax_id].set_ylabel("Sun\nmask", rotation=0, ha='right', va='center')
    axes[ax_id].tick_params(axis='y', labelleft=False, labelright=True)
    axes[ax_id].yaxis.tick_right()
    axes[ax_id].grid(True)
    ax_id += 1
    
    # --- Cloud masking ---
    axes[ax_id].plot(calendar, cloud_masking)
    axes[ax_id].set_ylabel("Weather\nmask", rotation=0, ha='right', va='center')
    axes[ax_id].tick_params(axis='y', labelleft=False, labelright=True)
    axes[ax_id].yaxis.tick_right()
    axes[ax_id].grid(True)
    ax_id += 1
    
    # --- Combined Masking ---
    axes[ax_id].plot(calendar, combo)
    axes[ax_id].set_ylabel("Combined\nmask", rotation=0, ha='right', va='center')
    axes[ax_id].tick_params(axis='y', labelleft=False, labelright=True)
    axes[ax_id].yaxis.tick_right()
    axes[ax_id].grid(True)
    ax_id += 1
    
    # --- Sampling instants ---
    axes[ax_id].plot(
        sampling / DIG_SAMPLES_X_DAY,
        np.ones_like(sampling),
        "|"
    )
    axes[ax_id].set_ylabel("Sampling", rotation=0, ha='right', va='center')
    axes[ax_id].tick_params(axis='y', labelleft=False, labelright=True)
    axes[ax_id].yaxis.tick_right()
    axes[ax_id].set_ylim(0.5, 1.5)
    axes[ax_id].grid(True)
    ax_id += 1
    
    # --- Sampling lag ---
    axes[ax_id].plot(
        (sampling[:-1] + sampling[1:]) / 2 / DIG_SAMPLES_X_DAY,
        (sampling[1:] - sampling[:-1]) / DIG_SAMPLES_X_DAY,
        "_"
    )
    axes[ax_id].set_ylabel("Sampling\nlag ", rotation=0, ha='right', va='center')
    axes[ax_id].tick_params(axis='y', labelleft=False, labelright=True)
    axes[ax_id].yaxis.tick_right()
    axes[ax_id].set_xlabel("Days")
    axes[ax_id].grid(True)
    axes[ax_id].set_yscale('log')
    ax_id += 1
    
    fig.suptitle("Non-uniform sampling simulation LSST")
    
    plt.show()
