"""Lightweight source-only feature canonicalization modules."""

import torch
from torch import Tensor, nn


class CanonicalizationAdapter(nn.Module):
    """Residual feature adapter used to remove normal source-only shifts.

    The second projection is zero initialized and ``alpha`` starts at zero, so
    the module is an exact identity at initialization.  The ``1 + alpha``
    parameterization keeps gradients flowing into the zero-initialized second
    projection on the first update.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_ratio: float = 0.25,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not 0 < bottleneck_ratio <= 1:
            raise ValueError("bottleneck_ratio must be in (0, 1]")
        hidden_dim = max(1, int(round(dim * bottleneck_ratio)))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] <= 0 or features.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError(
                f"expected the last feature dimension to be {self.norm.normalized_shape[0]}, "
                f"got {features.shape[-1]}"
            )
        residual = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(features))))
        return features + (1.0 + self.alpha) * residual


class FeatureGroupAdapters(nn.ModuleList):
    """A named ModuleList for checkpoint readability and type checking."""

    pass
