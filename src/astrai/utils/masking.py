"""Phenomenological missing-observation masks on a physical time axis.

These proxies degrade ideal bolometric curves; they are not an LSST survey
or detector simulation. Realisation draws depend on the physical horizon,
not the number of candidate samples. Evaluation consumes no random numbers.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np


MASKING_RECIPE = "phenomenological_masking_v1"
DEFAULT_SAMPLES_PER_DAY = 1


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class MaskingConfig:
    annual_period_days: float = 365.25
    daylight_mean_hours: float = 12.0
    daylight_amplitude_hours: float = 2.0
    seasonal_gap_min_days: float = 0.0
    seasonal_gap_max_days: float = 90.0
    moon_period_days: float = 29.53
    moon_active_days: float = 29.53 * 57 / 180
    moon_loss_hours: float = 4.0
    cloudy_fraction: float = 0.30
    mean_cloudy_days: float = 2.3
    threshold_interval_days: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _number(value, f"augmentation.masking.{name}"))
        for name in ("annual_period_days", "moon_period_days", "mean_cloudy_days", "threshold_interval_days"):
            if getattr(self, name) <= 0:
                raise ValueError(f"augmentation.masking.{name} must be positive")
        if not 0 <= self.daylight_amplitude_hours <= self.daylight_mean_hours <= 24:
            raise ValueError("Daylight amplitude and mean must describe non-negative hours")
        if self.daylight_mean_hours + self.daylight_amplitude_hours > 24:
            raise ValueError("Daylight hours cannot exceed 24")
        if not 0 <= self.seasonal_gap_min_days <= self.seasonal_gap_max_days <= self.annual_period_days:
            raise ValueError("Seasonal gap bounds must lie within the annual period")
        if not 0 <= self.moon_active_days <= self.moon_period_days:
            raise ValueError("Moon active duration must lie within the lunar period")
        if not 0 <= self.moon_loss_hours <= 24 - self.daylight_mean_hours - self.daylight_amplitude_hours:
            raise ValueError("Moon loss must not exceed the minimum available night hours")
        if not 0 <= self.cloudy_fraction <= 1:
            raise ValueError("cloudy_fraction must lie between zero and one")

    @classmethod
    def from_mapping(cls, values=None):
        if isinstance(values, cls):
            return values
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("augmentation.masking must be a mapping")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown masking parameters: {', '.join(sorted(map(str, unknown)))}")
        return cls(**values)

    def record(self):
        return {"recipe": MASKING_RECIPE, "parameters": asdict(self),
                "time_unit": "day", "time_origin": "first_clean_sample",
                "cloud_duration_distribution": "exponential_stationary",
                "threshold_distribution": "uniform_per_physical_interval",
                "interpolation": {"space": "log10_luminosity", "method": "linear",
                                  "edges": "constant", "zero_observations": "error",
                                  "one_observation": "constant", "resample_mask": False}}


def resolve_masking_config(cfg):
    return MaskingConfig.from_mapping(cfg.get("augmentation", {}).get("masking"))


def resolve_samples_per_day(cfg):
    value = _number(cfg.get("data", {}).get("samples_per_day", DEFAULT_SAMPLES_PER_DAY),
                    "data.samples_per_day")
    if value <= 0:
        raise ValueError("data.samples_per_day must be positive")
    return value


def build_time_axis(n_samples, samples_per_day=None):
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("Number of time samples must be a positive integer")
    rate = resolve_samples_per_day({"data": {"samples_per_day":
        DEFAULT_SAMPLES_PER_DAY if samples_per_day is None else samples_per_day}})
    times = np.arange(n_samples, dtype=np.float64) / rate
    if not np.isfinite(times).all():
        raise ValueError("Time axis is not representable")
    return times


def view_configuration(cfg):
    """Effective identity of precomputed views, including masking defaults."""
    return {"noise_std": cfg.get("augmentation", {}).get("noise_std"),
            "samples_per_day": resolve_samples_per_day(cfg),
            "masking": resolve_masking_config(cfg).record()}


@dataclass(frozen=True)
class MaskingRealisation:
    config: MaskingConfig
    horizon_days: float
    daylight_phase: float
    seasonal_centre: float
    seasonal_duration: float
    moon_phase: float
    initially_cloudy: bool
    cloud_transitions: np.ndarray
    threshold_offset: float
    thresholds: np.ndarray

    def evaluate(self, times):
        """Evaluate the same latent path on any increasing grid in [0, T]."""
        raw = np.asarray(times)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
            raise ValueError("Times must be a non-empty real one-dimensional array")
        t = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(t).all() or t[0] < 0 or t[-1] > self.horizon_days or np.any(np.diff(t) <= 0):
            raise ValueError("Times must increase within the realisation's physical horizon")
        c = self.config
        daylight = c.daylight_mean_hours + c.daylight_amplitude_hours * np.sin(
            2 * np.pi * (t - self.daylight_phase) / c.annual_period_days)
        seasonal_position = np.mod(t - self.seasonal_centre + self.seasonal_duration / 2,
                                   c.annual_period_days)
        sun_blocked = seasonal_position < self.seasonal_duration
        lunar_distance = np.mod(t - self.moon_phase + c.moon_period_days / 2,
                                c.moon_period_days) - c.moon_period_days / 2
        moon = (0.5 * (1 + np.cos(2 * np.pi * lunar_distance / c.moon_period_days))
                * (np.abs(lunar_distance) < c.moon_active_days / 2))
        cloud_blocked = np.logical_xor(self.initially_cloudy,
            np.searchsorted(self.cloud_transitions, t, side="right") % 2 == 1)
        availability = (24 - daylight - c.moon_loss_hours * moon) / 24
        availability *= ~sun_blocked & ~cloud_blocked
        # The validated parameter bounds guarantee [0, 1]; round-off only.
        availability = np.clip(availability, 0.0, 1.0)
        indices = np.floor((t + self.threshold_offset) / c.threshold_interval_days).astype(np.int64)
        threshold = self.thresholds[indices]
        return {"time_days": t, "daylight_hours": daylight, "moon_contribution": moon,
                "sun_blocked": sun_blocked, "cloud_blocked": cloud_blocked,
                "availability": availability, "threshold": threshold,
                "retained_mask": threshold < availability}


def generate_masking_realisation(horizon_days, config=None, rng=None):
    """Draw one path, with stationary exponential clear/cloudy episodes.

    Reproducibility is scoped to the same horizon, configuration and initial
    RNG state. No invariance to batch regrouping or horizon changes is claimed.
    """
    horizon = _number(horizon_days, "horizon_days")
    if horizon < 0:
        raise ValueError("horizon_days must be non-negative")
    c = MaskingConfig.from_mapping(config)
    rng = np.random.default_rng() if rng is None else rng
    daylight_phase = rng.uniform(0, c.annual_period_days)
    seasonal_centre = rng.uniform(0, c.annual_period_days)
    duration = rng.uniform(c.seasonal_gap_min_days, c.seasonal_gap_max_days)
    moon_phase = rng.uniform(0, c.moon_period_days)
    initially_cloudy = bool(rng.random() < c.cloudy_fraction)
    transitions = []
    if 0 < c.cloudy_fraction < 1:
        clear_mean = c.mean_cloudy_days * (1 - c.cloudy_fraction) / c.cloudy_fraction
        cloudy, current = initially_cloudy, 0.0
        while current < horizon:
            following = current + rng.exponential(c.mean_cloudy_days if cloudy else clear_mean)
            if not np.isfinite(following) or following <= current:
                raise ValueError("Cloud episode duration is not representable")
            if following > horizon:
                break
            transitions.append(following)
            current, cloudy = following, not cloudy
    offset = rng.uniform(0, c.threshold_interval_days)
    count = int(np.floor((horizon + offset) / c.threshold_interval_days)) + 1
    thresholds = rng.random(count)
    transitions = np.asarray(transitions, dtype=np.float64)
    transitions.setflags(write=False)
    thresholds.setflags(write=False)
    return MaskingRealisation(c, horizon, daylight_phase, seasonal_centre, duration,
                              moon_phase, initially_cloudy, transitions, offset, thresholds)


def generate_masking(times, config=None, rng=None):
    """Generate all components for one candidate grid, without selecting a cadence."""
    t = np.asarray(times)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("Times must be a non-empty one-dimensional array")
    return generate_masking_realisation(t[-1], config, rng).evaluate(t)


def interpolate_observations(values, retained_mask, times):
    """Fill from observed log10 luminosities only, with constant edge values."""
    values, mask, t = np.asarray(values), np.asarray(retained_mask), np.asarray(times)
    if values.ndim != 1 or mask.dtype != np.bool_ or values.shape != mask.shape or t.shape != values.shape:
        raise ValueError("Values, boolean mask and times must be aligned one-dimensional arrays")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Interpolation times must be finite and increasing")
    if not mask.any():
        raise ValueError("No observations retained; the mask is not resampled")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed luminosities must be finite")
    return np.interp(t, t[mask], values[mask])
