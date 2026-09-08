"""Canonical constructors for ASTRAI model architectures."""

from astrai.models.residual_blocks import MLPWithResiduals
from astrai.models.split_mlp import SplitMLPRegressor
from astrai.models.unified_model import UnifiedModel


def build_characterizer(cfg):
    """Build the characterizer described by a split-pipeline config."""
    model_cfg = cfg["characterizer"]["model"]
    return SplitMLPRegressor(
        input_dim=cfg["preprocessing"]["pca_components"],
        width=model_cfg["width"],
        num_params=cfg["data"]["n_params"],
        depth=model_cfg["depth"],
        dropout=model_cfg["dropout"],
    )


def build_generator(cfg):
    """Build the generator described by a split-pipeline config."""
    model_cfg = cfg["generator"]["model"]
    return MLPWithResiduals(
        input_dim=cfg["data"]["n_params"],
        width=model_cfg["width"],
        out_dim=cfg["preprocessing"]["pca_components"],
        depth=model_cfg["depth"],
        dropout=model_cfg["dropout"],
    )


def build_unified_model(cfg):
    """Build the jointly trained model described by a unified config."""
    model_cfg = cfg["model"]
    n_pca = model_cfg["pca_components"]
    n_params = cfg["data"]["n_params"]

    regressor = SplitMLPRegressor(
        input_dim=n_pca,
        width=model_cfg["width"],
        num_params=n_params,
        depth=model_cfg["depth"],
        dropout=model_cfg["dropout"],
    )
    generator = MLPWithResiduals(
        input_dim=n_params,
        width=model_cfg["width"],
        out_dim=n_pca,
        depth=model_cfg["depth"],
        dropout=model_cfg["dropout"],
    )
    return UnifiedModel(regressor, generator)
