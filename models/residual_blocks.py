"""
models.residual_blocks - Pre-activation residual blocks and residual MLP.

Implements the core building blocks used by all network architectures in
ASTRAI.  The design follows the *pre-activation* residual pattern
(BN -> activation -> dropout -> linear + skip) which empirically
stabilizes training for narrow, deep regression networks.
"""
from torch import nn


class ResidualBlock(nn.Module):
    """Single pre-activation residual block.

    Architecture::

        input -+-> BatchNorm -> LeakyReLU -> [Dropout] -> Linear -> (+) -> output
               |                                                     ^
               +-----------------------------------------------------+

    Parameters
    ----------
    in_features : int
        Dimensionality of both input and output (identity skip-connection).
    dropout : float
        Dropout probability. Set to 0 to disable.
    """

    def __init__(self, in_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.leaky = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout_val = dropout
        if self.dropout_val > 0:
            self.drop = nn.Dropout(p=self.dropout_val)

    def forward(self, x):
        """Apply BN -> LeakyReLU -> [Dropout] -> Linear, then add residual."""
        residual = x
        x = self.bn1(x)
        x = self.leaky(x)
        if self.dropout_val > 0:
            x = self.drop(x)
        x = self.fc1(x)
        x += residual
        return x


class MLPWithResiduals(nn.Module):
    """Feed-forward MLP with stacked pre-activation residual blocks.

    Structure: ``Linear(in -> width) -> [ResidualBlock] * depth -> Linear(width -> out)``.

    Parameters
    ----------
    input_dim : int
        Input feature dimensionality.
    width : int
        Hidden-layer width (constant across all residual blocks).
    out_dim : int
        Output dimensionality.
    depth : int
        Number of stacked ``ResidualBlock`` layers.
    dropout : float
        Dropout probability forwarded to each ``ResidualBlock``.
    """

    def __init__(self, input_dim, width, out_dim, depth, dropout):
        super().__init__()
        layers = [nn.Linear(input_dim, width)]
        for _ in range(depth):
            layers.append(ResidualBlock(width, dropout))
        layers.append(nn.Linear(width, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the full residual MLP stack."""
        return self.network(x)
