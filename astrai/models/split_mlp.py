"""
astrai.models.split_mlp - Parameter-wise split MLP regressor.

Instead of a single shared network for all target parameters, this module
instantiates one independent ``MLPWithResiduals`` sub-network per output
dimension.  This avoids negative transfer between heterogeneous physical
quantities (e.g. mass vs. nickel fraction) at the cost of a linear
increase in parameter count.
"""
import torch
import torch.nn as nn
from .residual_blocks import MLPWithResiduals


class SplitMLPRegressor(nn.Module):
    """Multi-output regressor with one independent MLP per target parameter.

    Each sub-network maps the shared PCA-compressed input to a single
    scalar prediction.  Outputs are concatenated along ``dim=1``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the (PCA-compressed) input features.
    width : int
        Hidden-layer width for every sub-network.
    num_params : int
        Number of target physical parameters (= number of sub-networks).
    depth : int
        Residual-block depth for every sub-network.
    dropout : float
        Dropout probability forwarded to each sub-network.
    """

    def __init__(self, input_dim, width, num_params, depth, dropout):
        super().__init__()
        self.nets = nn.ModuleList([
            MLPWithResiduals(input_dim, width, 1, depth, dropout)
            for _ in range(num_params)
        ])

    def forward(self, x):
        """Run each sub-network and concatenate scalar outputs."""
        outputs = [net(x) for net in self.nets]
        return torch.cat(outputs, dim=1)
