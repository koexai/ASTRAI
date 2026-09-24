"""Explicit noise recipes shared by augmentation and its provenance.

The IID log10 recipe is an empirical operational baseline. Neither modern
recipe is a calibrated observation model or a scientifically canonical choice.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np


NOISE_RECIPE_VERSION = 1
_PARAMETERS = {
    "iid_log10": (("sigma_dex",), {}),
    "heteroscedastic_normalised": (("a", "b"), {"x_ref": 42.0}),
    "legacy_iid_gaussian": (("noise_std",), {}),
    "legacy_tiled_gaussian": (("noise_std",), {}),
    "legacy_exp_sqrt_gaussian": (("sigma",), {"eps": 1e-12}),
}


@dataclass(frozen=True)
class NoiseConfig:
    """An immutable, validated recipe; parameters have a canonical order."""

    model: str
    parameters: tuple

    def __post_init__(self):
        if not isinstance(self.model, str) or self.model not in _PARAMETERS:
            raise ValueError(f"Unknown augmentation.noise.model: {self.model!r}")
        supplied = dict(self.parameters)
        if len(supplied) != len(self.parameters):
            raise ValueError("Duplicate augmentation.noise parameters")
        required, defaults = _PARAMETERS[self.model]
        unknown = set(supplied) - set(required) - set(defaults)
        missing = set(required) - set(supplied)
        if unknown or missing:
            raise ValueError(
                f"Invalid augmentation.noise parameters for {self.model}: "
                f"missing={sorted(missing)}, unknown={sorted(map(str, unknown))}"
            )
        resolved = {**defaults, **supplied}
        for name, value in resolved.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"augmentation.noise.{name} must be a finite real scalar")
            try:
                value = float(value)
            except (ValueError, OverflowError) as error:
                raise ValueError(f"augmentation.noise.{name} must be finite") from error
            if not np.isfinite(value) or (name != "x_ref" and value < 0):
                raise ValueError(f"augmentation.noise.{name} must be finite"
                                 + (" and non-negative" if name != "x_ref" else ""))
            if name == "eps" and value == 0:
                raise ValueError("augmentation.noise.eps must be strictly positive")
            resolved[name] = value
        object.__setattr__(self, "parameters", tuple(sorted(resolved.items())))

    @classmethod
    def from_mapping(cls, values):
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping) or "model" not in values:
            raise ValueError("augmentation.noise must be a mapping with an explicit model")
        return cls(values["model"], tuple((key, value) for key, value in values.items()
                                         if key != "model"))

    @property
    def modern(self):
        return not self.model.startswith("legacy_")

    def record(self):
        """Record numerical semantics, including the temporary integration rule."""
        record = {
            "recipe_version": NOISE_RECIPE_VERSION,
            "family": "modern" if self.modern else "legacy",
            "model": self.model,
            "parameters": dict(self.parameters),
            "input_output_space": "log10_bolometric_luminosity",
            "zero_treatment": "ordinary_numeric_value",
        }
        if self.model == "heteroscedastic_normalised":
            record.update(
                proposal_variance="a*f+b; f=10**(x-x_ref)",
                positivity="resample_non_positive_proposals",
                maximum_positive_draws=128,
                zero_treatment="temporary_exact_zero_exclusion_at_noise_stage_v1",
                zero_treatment_scientifically_validated=False,
                excluded_zeros_consume_noise_rng=False,
            )
        elif self.model == "iid_log10":
            record["interpretation"] = "empirical_operational_baseline"
        elif self.model == "legacy_exp_sqrt_gaussian":
            record.update(internal_transform="natural_exp_and_log",
                          positivity="clip_to_eps")
        return record


def resolve_noise_config(cfg):
    """Require the final explicit contract, without interpreting old amplitudes."""
    augmentation = cfg.get("augmentation")
    if not isinstance(augmentation, Mapping):
        raise ValueError("An explicit augmentation.noise configuration is required")
    if "noise_std" in augmentation:
        raise ValueError("augmentation.noise_std is no longer supported; use augmentation.noise")
    return NoiseConfig.from_mapping(augmentation.get("noise"))
