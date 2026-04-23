"""
astrai.models - Neural network architectures for light-curve analysis.

Exports
-------
ResidualBlock
    Single pre-activation residual block (BN -> LeakyReLU -> Dropout -> Linear).
MLPWithResiduals
    Feed-forward MLP with stacked residual blocks.
SplitMLPRegressor
    Multi-output regressor composed of N independent MLPWithResiduals sub-networks,
    one per target physical parameter.
UnifiedModel
    Bi-directional wrapper coupling a characterization regressor
    (curves -> params) with a generative decoder (params -> curves).
"""
from .residual_blocks import ResidualBlock, MLPWithResiduals
from .split_mlp import SplitMLPRegressor
from .unified_model import UnifiedModel
