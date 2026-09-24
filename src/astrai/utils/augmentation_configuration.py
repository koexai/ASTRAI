"""Effective identity shared by generated views and all their consumers."""
from astrai.utils.masking import resolve_masking_config, resolve_samples_per_day
from astrai.utils.noise_configuration import resolve_noise_config
from astrai.utils.reproducibility import augmentation_rng_policy


def view_configuration(cfg):
    noise = resolve_noise_config(cfg)
    return {
        "noise": noise.record(),
        "rng": augmentation_rng_policy(modern=noise.modern),
        "samples_per_day": resolve_samples_per_day(cfg),
        "masking": resolve_masking_config(cfg).record(),
    }
