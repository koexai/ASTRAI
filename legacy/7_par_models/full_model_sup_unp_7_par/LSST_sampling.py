import numpy as np
import matplotlib.pyplot as plt

# Constants
ROT_PER_DAY = np.pi / 180.0
DAY_PER_ROT = 180.0 / np.pi

SUN_PERIOD = 365.25
MOON_PERIOD = 29.53

CONSECUTIVE_CLOUDY_DAYS = 2.3 # average duration of covering
CLOUDY_PROB = 20 #percentage
CLOUD_SEED = 42

EEPS = 23.44 #earth axis inclination epsilon
LATITUDE_LSST = 30
PHI = LATITUDE_LSST * ROT_PER_DAY

GOOD_MOON_ANGLE = 33
GOOD_SUN_ANGLE = 64
MOON_MASK_MIN_ANGLE = 6.5
MOON_MASK_MAX_ANGLE = 20

AVG_SAMPLING_RATE = 4
DIG_SAMPLES_X_DAY = 4
MAX_OBS_LENGTH = 400

N_SAMPLES = MAX_OBS_LENGTH * DIG_SAMPLES_X_DAY + 1


def daylight_hours_np(days, day0=None):
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
    days = np.asarray(days, dtype=np.float64)

    if day0 is None:
        day0 = np.random.rand(1) * MOON_PERIOD

    # Phase angle [0, 2π]
    phase = 2.0 * np.pi * np.mod((days - day0) / MOON_PERIOD, 1.0)

    # Illuminated fraction
    illuminated_fraction = 0.5 * (1.0 + np.cos(phase))

    moon_presence = np.cos(phase)>np.cos(np.pi/180*(90-GOOD_MOON_ANGLE))

    return illuminated_fraction * moon_presence


def sun_masking_np(days, day0=None, elev=None):
    days = np.asarray(days, dtype=np.float64)

    if day0 is None:
        day0 = np.random.rand(1) * SUN_PERIOD
    if elev is None:
        elev = np.random.rand(1) * 60

    # Phase angle [0, 2π]
    phase = 2.0 * np.pi * np.mod((days - day0) / SUN_PERIOD, 1.0)

    sun_presence = np.cos(phase)>np.cos(np.pi/180*(90-GOOD_SUN_ANGLE+elev**2/60))

    return sun_presence



def random_cloud_masking(arr, percentage=CLOUDY_PROB, seed=None):
    arr = np.asarray(arr, dtype=np.float64)

    if seed is None:
        seed = np.random.randint(100000, size=1)
    masked_arr = arr.copy()

    # FIX 6: Use local RandomState instead of resetting the global seed
    rng = np.random.RandomState(seed)

    consecutive_cloudy_samples = CONSECUTIVE_CLOUDY_DAYS * DIG_SAMPLES_X_DAY

    n_total = masked_arr.size
    n_mask = int(np.round(n_total * percentage / consecutive_cloudy_samples / 100.0))

    indices = rng.choice(n_total, n_mask, replace=False)

    for shift in range(int(consecutive_cloudy_samples)):
        masked_arr.flat[indices-shift] = 0 #np.nan

    return masked_arr


def good_area_x_time(days):
    nigth_time = 24-daylight_hours_np(days)
    moon_area = MOON_MASK_MIN_ANGLE**2 + moon_luminosity_np(days)*(MOON_MASK_MAX_ANGLE)**2
    nigth_time = nigth_time*np.cos(np.pi*days)**2
    clear_nigth_time = random_cloud_masking(nigth_time)
    return clear_nigth_time * (GOOD_MOON_ANGLE**2 - moon_area) * (1-sun_masking_np(days))


def intersections_monotone(f, g):
    f = np.asarray(f)
    g = np.asarray(g)
    # indices where each g would be inserted to keep f sorted
    idx = np.searchsorted(f, g)
    # clamp to valid range
    idx = np.clip(idx, 1, len(f)-1)
    x0 = idx-1
    x1 = idx
    f0 = f[idx-1]
    f1 = f[idx]
    # linear interpolation
    return x0 + (g - f0) * (x1 - x0) / (f1 - f0)


def get_samples(calendar):
    f = np.cumsum(good_area_x_time(calendar))
    n_points = (calendar[-1]-calendar[0])/AVG_SAMPLING_RATE
    g = np.arange(n_points)*f[-1]/n_points
    return intersections_monotone(f, g)


def keep_only_samples_from_lc(lc, sampling):
    lc = np.asarray(lc, dtype=float)
    out = np.full_like(lc, np.nan, dtype=float)

    out.flat[sampling] = lc.flat[sampling]
    return out



if __name__ == "__main__":

    calendar = np.arange(N_SAMPLES) / DIG_SAMPLES_X_DAY
    sampling = np.int32(get_samples(calendar))

    fig, axes = plt.subplots(
        7, 1,
        sharex=True,
        figsize=(7, 10),
        #constrained_layout=True
    )
    ax_id = 0

    # --- Night hours ---
    axes[ax_id].plot(calendar, daylight_hours_np(calendar))
    axes[ax_id].set_ylabel("DayLight\nhours")
    axes[ax_id].grid(True)
    ax_id += 1

    # --- Moon light masking ---
    axes[ax_id].plot(calendar, moon_luminosity_np(calendar))
    axes[ax_id].set_ylabel("Moon\nmasking")
    axes[ax_id].grid(True)
    ax_id += 1

    # --- Sun masking ---
    axes[ax_id].plot(calendar, sun_masking_np(calendar))
    axes[ax_id].set_ylabel("Sun\nmasking")
    axes[ax_id].grid(True)
    ax_id += 1

    # --- Cloud masking ---
    axes[ax_id].plot(
        calendar, 1-random_cloud_masking(np.ones_like(calendar))
    )
    axes[ax_id].set_ylabel("Weather\nmasking")
    axes[ax_id].grid(True)
    ax_id += 1

    # --- Combined Masking ---
    axes[ax_id].plot(
        calendar,
        random_cloud_masking((1-sun_masking_np(calendar))*(24-daylight_hours_np(calendar)-moon_luminosity_np(calendar)*4))
    )
    axes[ax_id].set_ylabel("Combined\nmask")
    axes[ax_id].grid(True)
    ax_id += 1

    # --- Sampling instants ---
    axes[ax_id].plot(
        sampling / DIG_SAMPLES_X_DAY,
        np.ones_like(sampling),
        "|"
    )
    axes[ax_id].set_ylabel("Sampling")
    axes[ax_id].set_ylim(0.5, 1.5)
    axes[ax_id].grid(True)
    ax_id += 1

    # --- Sampling lag ---
    axes[ax_id].plot(
        (sampling[:-1] + sampling[1:]) / 2 / DIG_SAMPLES_X_DAY,
        (sampling[1:] - sampling[:-1]) / DIG_SAMPLES_X_DAY,
        "_"
    )
    axes[ax_id].set_ylabel("Sampling lag")
    axes[ax_id].set_xlabel("Days")
    axes[ax_id].grid(True)
    axes[ax_id].set_yscale('log')
    ax_id += 1

    fig.suptitle("Non-uniform sampling simulation LLST")

    plt.show()
