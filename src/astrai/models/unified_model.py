"""Historical checkpoint container for characterisation and generation.

The registered submodule names are preserved for existing state dictionaries.
New models are trained independently through the split pipeline.
"""
from torch import nn


class UnifiedModel(nn.Module):
    """Wrapper holding both models
    - Regressor (curves -> params)
    - Generator (params -> curves).

    Parameters
    ----------
    regressor : nn.Module
        Characterization network (e.g. ``SplitMLPRegressor``).
    generator : nn.Module
        Generative decoder (e.g. ``MLPWithResiduals``).
    """

    def __init__(self, regressor, generator):
        super().__init__()
        self.regressor = regressor
        self.generator = generator

    def fit(self, *args, **kwargs):
        """Reject shared optimisation while preserving checkpoint loading."""
        from astrai.cli.train import RETIREMENT_MESSAGE

        raise RuntimeError(RETIREMENT_MESSAGE)
