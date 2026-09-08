"""Training losses used by Dinomaly2."""

from .combined_loss import CombinedDinomalyLoss
from .robust_sinkhorn import RobustSinkhornLoss
from .source_dg_loss import (
    anomaly_margin_loss,
    cosine_consistency,
    identity_loss,
    map_consistency,
    preservation_loss,
)

__all__ = [
    "CombinedDinomalyLoss",
    "RobustSinkhornLoss",
    "anomaly_margin_loss",
    "cosine_consistency",
    "identity_loss",
    "map_consistency",
    "preservation_loss",
]
