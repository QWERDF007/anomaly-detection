"""Composition of the original Dinomaly2 loss and local robust OT."""

from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from .robust_sinkhorn import RobustSinkhornLoss


class CombinedDinomalyLoss(nn.Module):
    """Add a warm-started robust OT auxiliary loss to an existing base loss."""

    def __init__(
        self,
        robust_ot: RobustSinkhornLoss,
        lambda_ot: float = 0.05,
        warmup_iters: int = 2000,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if lambda_ot < 0:
            raise ValueError("lambda_ot must be non-negative")
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        self.robust_ot = robust_ot
        self.lambda_ot = float(lambda_ot)
        self.warmup_iters = int(warmup_iters)
        self.enabled = bool(enabled)

    def weight_at(self, iteration: int) -> float:
        if not self.enabled or self.lambda_ot == 0:
            return 0.0
        if self.warmup_iters == 0:
            return self.lambda_ot
        progress = min(max(float(iteration), 0.0) / self.warmup_iters, 1.0)
        return self.lambda_ot * progress

    def forward(
        self,
        loss_cos: Tensor,
        encoder_features: Sequence[Tensor],
        decoder_features: Sequence[Tensor],
        iteration: int,
    ) -> Dict[str, Tensor]:
        weight = self.weight_at(iteration)
        if not self.enabled or self.lambda_ot == 0:
            zero = loss_cos.new_zeros((), dtype=torch.float32)
            return {
                "loss_total": loss_cos,
                "loss_cos": loss_cos,
                "loss_ot": zero,
                "lambda_ot": zero,
                "transport_mass": zero,
                "layer_loss": zero.new_empty((0,)),
            }

        ot_result = self.robust_ot(encoder_features, decoder_features)
        weight_tensor = ot_result["loss_ot"].new_tensor(weight)
        return {
            "loss_total": loss_cos + weight_tensor * ot_result["loss_ot"],
            "loss_cos": loss_cos,
            "loss_ot": ot_result["loss_ot"],
            "lambda_ot": weight_tensor,
            "transport_mass": ot_result["transport_mass"],
            "layer_loss": ot_result["layer_loss"],
        }
