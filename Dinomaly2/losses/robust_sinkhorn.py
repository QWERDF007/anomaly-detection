"""Local robust Sinkhorn loss for Dinomaly2 feature reconstruction.

The transport plan is solved in FP32 and detached before the transport cost is
evaluated.  Detaching the plan follows the envelope gradient of the optimized
OT objective and prevents gradients through the iterative solver from making
training unnecessarily expensive or unstable.
"""

from contextlib import nullcontext
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RobustSinkhornLoss(nn.Module):
    """Compute a local balanced or unbalanced entropic OT feature loss.

    Args:
        window_size: Spatial side length of every local feature window.
        stride: Window stride. Values smaller than ``window_size`` create
            overlapping windows.
        epsilon: Entropic regularization strength.
        rho: Marginal relaxation strength for unbalanced OT. Larger values
            approach balanced OT.
        iterations: Number of Sinkhorn scaling iterations.
        position_weight: Weight of the normalized within-window position cost.
        ot_type: Either ``"balanced"`` or ``"unbalanced"``.

    Inputs are equally-sized sequences of ``[B, C, H, W]`` tensors. The
    encoder inputs are always detached. The output dictionary contains the
    mean loss, one loss per feature layer, and the mean transported mass.
    """

    _SUPPORTED_TYPES = {"balanced", "unbalanced"}

    def __init__(
        self,
        window_size: int = 5,
        stride: int = 5,
        epsilon: float = 0.05,
        rho: float = 0.5,
        iterations: int = 20,
        position_weight: float = 0.0,
        ot_type: str = "unbalanced",
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0 or stride > window_size:
            raise ValueError("stride must be in [1, window_size]")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if rho <= 0:
            raise ValueError("rho must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if position_weight < 0:
            raise ValueError("position_weight must be non-negative")
        if ot_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                f"ot_type must be one of {sorted(self._SUPPORTED_TYPES)}, got {ot_type!r}"
            )

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.epsilon = float(epsilon)
        self.rho = float(rho)
        self.iterations = int(iterations)
        self.position_weight = float(position_weight)
        self.ot_type = ot_type

        self.register_buffer(
            "position_cost",
            self._make_position_cost(self.window_size),
            persistent=False,
        )

    @staticmethod
    def _make_position_cost(window_size: int) -> Tensor:
        if window_size == 1:
            return torch.zeros(1, 1, dtype=torch.float32)
        axis = torch.linspace(0.0, 1.0, steps=window_size, dtype=torch.float32)
        rows, cols = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((rows, cols), dim=-1).reshape(-1, 2)
        return torch.cdist(coordinates, coordinates, p=2) / (2.0 ** 0.5)

    def _padding_for_full_coverage(self, size: int) -> int:
        if size <= self.window_size:
            return self.window_size - size
        remainder = (size - self.window_size) % self.stride
        return 0 if remainder == 0 else self.stride - remainder

    def _extract_windows(self, feature: Tensor) -> Tuple[Tensor, Tensor]:
        """Return windows ``[B, L, T, C]`` and validity ``[B, L, T]``."""

        if feature.ndim != 4:
            raise ValueError(
                f"feature tensors must have shape [B, C, H, W], got {tuple(feature.shape)}"
            )
        batch, channels, height, width = feature.shape
        if height <= 0 or width <= 0:
            raise ValueError("feature maps must have non-empty spatial dimensions")

        pad_h = self._padding_for_full_coverage(height)
        pad_w = self._padding_for_full_coverage(width)
        padded = F.pad(feature, (0, pad_w, 0, pad_h))
        valid = feature.new_ones((batch, 1, height, width), dtype=torch.float32)
        valid = F.pad(valid, (0, pad_w, 0, pad_h))

        window_tokens = self.window_size * self.window_size
        windows = F.unfold(
            padded,
            kernel_size=self.window_size,
            stride=self.stride,
        )
        windows = windows.transpose(1, 2).reshape(
            batch, -1, channels, window_tokens
        )
        windows = windows.permute(0, 1, 3, 2).contiguous()

        valid_windows = F.unfold(
            valid,
            kernel_size=self.window_size,
            stride=self.stride,
        )
        valid_windows = valid_windows.transpose(1, 2).contiguous() > 0.5
        return windows, valid_windows

    def _sinkhorn_plan(self, cost: Tensor, valid: Tensor) -> Tensor:
        """Solve the regularized transport plan in the log domain.

        ``cost`` is ``[N, T, T]`` and ``valid`` is ``[N, T]``. Padding
        tokens receive exactly zero marginal mass.
        """

        marginal = valid.to(dtype=cost.dtype)
        marginal = marginal / marginal.sum(dim=-1, keepdim=True).clamp_min(1.0)
        negative_infinity = torch.full_like(marginal, -torch.inf)
        log_marginal = torch.where(valid, marginal.clamp_min(1e-32).log(), negative_infinity)

        # The reference measure a x b keeps the transported mass independent
        # of the number of tokens when all pairwise costs are zero.
        log_kernel = (
            log_marginal.unsqueeze(-1)
            + log_marginal.unsqueeze(-2)
            - cost / self.epsilon
        )
        log_u = torch.zeros_like(log_marginal)
        log_v = torch.zeros_like(log_marginal)
        tau = 1.0
        if self.ot_type == "unbalanced":
            tau = self.rho / (self.rho + self.epsilon)

        for _ in range(self.iterations):
            row_normalizer = torch.logsumexp(
                log_kernel + log_v.unsqueeze(-2), dim=-1
            )
            next_log_u = tau * (log_marginal - row_normalizer)
            log_u = torch.where(valid, next_log_u, negative_infinity)

            column_normalizer = torch.logsumexp(
                log_kernel + log_u.unsqueeze(-1), dim=-2
            )
            next_log_v = tau * (log_marginal - column_normalizer)
            log_v = torch.where(valid, next_log_v, negative_infinity)

        log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
        plan = log_plan.exp()
        valid_pairs = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        return torch.where(valid_pairs, plan, torch.zeros_like(plan))

    def _layer_loss(self, encoder: Tensor, decoder: Tensor) -> Tuple[Tensor, Tensor]:
        if encoder.shape != decoder.shape:
            raise ValueError(
                "encoder and decoder feature shapes must match, got "
                f"{tuple(encoder.shape)} and {tuple(decoder.shape)}"
            )

        encoder_windows, valid = self._extract_windows(encoder.detach().float())
        decoder_windows, decoder_valid = self._extract_windows(decoder.float())
        if not torch.equal(valid, decoder_valid):
            raise RuntimeError("encoder and decoder window masks unexpectedly differ")

        encoder_windows = F.normalize(encoder_windows, p=2, dim=-1, eps=1e-8)
        decoder_windows = F.normalize(decoder_windows, p=2, dim=-1, eps=1e-8)
        feature_cost = 1.0 - torch.einsum(
            "bltc,blsc->blts", encoder_windows, decoder_windows
        )
        # Numerical round-off can make cosine distance slightly negative.
        cost = feature_cost.clamp_min(0.0)
        if self.position_weight > 0:
            cost = cost + self.position_weight * self.position_cost.to(cost).view(
                1, 1, *self.position_cost.shape
            )

        tokens = self.window_size * self.window_size
        flat_cost = cost.reshape(-1, tokens, tokens)
        flat_valid = valid.reshape(-1, tokens)
        with torch.no_grad():
            plan = self._sinkhorn_plan(flat_cost.detach(), flat_valid)

        # The detached optimal plan gives the stable envelope gradient dL/dC=P.
        window_loss = (plan * flat_cost).sum(dim=(-2, -1))
        transport_mass = plan.sum(dim=(-2, -1))
        # Boundary windows can contain fewer real tokens after coverage padding.
        # Weight them by valid area so a small corner window is not as influential
        # as a full window.
        window_weight = flat_valid.sum(dim=-1).to(dtype=window_loss.dtype)
        weight_sum = window_weight.sum().clamp_min(1.0)
        return (
            (window_loss * window_weight).sum() / weight_sum,
            (transport_mass * window_weight).sum() / weight_sum,
        )

    def forward(
        self,
        encoder_features: Sequence[Tensor],
        decoder_features: Sequence[Tensor],
    ) -> Dict[str, Tensor]:
        if len(encoder_features) == 0:
            raise ValueError("at least one feature layer is required")
        if len(encoder_features) != len(decoder_features):
            raise ValueError(
                "encoder_features and decoder_features must contain the same number of layers"
            )

        device_type = decoder_features[0].device.type
        autocast_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            layer_results: List[Tensor] = []
            layer_masses: List[Tensor] = []
            for encoder, decoder in zip(encoder_features, decoder_features):
                layer_loss, layer_mass = self._layer_loss(encoder, decoder)
                layer_results.append(layer_loss)
                layer_masses.append(layer_mass)

            per_layer = torch.stack(layer_results)
            per_layer_mass = torch.stack(layer_masses)
            return {
                "loss_ot": per_layer.mean(),
                "layer_loss": per_layer,
                "transport_mass": per_layer_mass.mean(),
                "layer_transport_mass": per_layer_mass,
            }
