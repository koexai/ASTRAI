"""
astrai - Astrophysical parameter estimation via deep learning.

This package provides neural network architectures and utilities for
light-curve characterization (curves -> physical parameters) and
generation (physical parameters -> curves), with support for LSST-like
observational constraints and data augmentation.

Subpackages
-----------
models
    Neural network architectures (SplitMLP, MLPWithResiduals, UnifiedModel).

Modules
-------
metrics
    Regression evaluation metrics (RMSE, MAE, R2, RRMSE).
lsst
    LSST observing-condition simulation (sun masking, cloud masking,
    moon luminosity, cadence sampling).
augmentation
    Data augmentation pipeline combining Gaussian noise injection
    and LSST-realistic cadence degradation.
"""
