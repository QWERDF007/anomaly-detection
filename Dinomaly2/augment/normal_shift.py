"""Differentiable normal-shift generator operating on normalized image tensors."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TVF


@dataclass
class NormalShiftBatch:
    images: Tensor
    theta: Tensor
    family: List[str]

    def inverse_warp(self, feature: Tensor, padding_mode: str = 'reflection') -> Tensor:
        """Map a shifted tensor back to anchor coordinates."""

        if feature.shape[0] != self.theta.shape[0]:
            raise ValueError('feature batch and transform batch must match')
        inverse = torch.linalg.inv(self.theta)
        inverse = inverse[:, :2]
        grid = F.affine_grid(inverse, feature.shape, align_corners=False)
        return F.grid_sample(
            feature,
            grid,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=False,
        )


class NormalShiftGenerator:
    """Generate bounded photometric, imaging, illumination and geometric shifts."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        geometry_probability: float = 0.30,
        photometric_probability: float = 0.80,
        degradation_probability: float = 0.40,
        illumination_probability: float = 0.40,
        max_ops_per_view: int = 3,
        translate_fraction: float = 0.03,
        rotation_degrees: float = 2.0,
        scale_range: Tuple[float, float] = (0.97, 1.03),
        noise_std: float = 0.02,
        blur_sigma: float = 1.2,
    ) -> None:
        for name, value in (
            ('geometry_probability', geometry_probability),
            ('photometric_probability', photometric_probability),
            ('degradation_probability', degradation_probability),
            ('illumination_probability', illumination_probability),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        if max_ops_per_view <= 0:
            raise ValueError('max_ops_per_view must be positive')
        if translate_fraction < 0 or rotation_degrees < 0:
            raise ValueError('geometry magnitudes must be non-negative')
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError('scale_range must be positive and ordered')
        if noise_std < 0 or blur_sigma < 0:
            raise ValueError('noise_std and blur_sigma must be non-negative')
        self.geometry_probability = float(geometry_probability)
        self.photometric_probability = float(photometric_probability)
        self.degradation_probability = float(degradation_probability)
        self.illumination_probability = float(illumination_probability)
        self.max_ops_per_view = int(max_ops_per_view)
        self.translate_fraction = float(translate_fraction)
        self.rotation_degrees = float(rotation_degrees)
        self.scale_range = tuple(float(v) for v in scale_range)
        self.noise_std = float(noise_std)
        self.blur_sigma = float(blur_sigma)
        self.mean = None
        self.std = None

    def _normalization(self, device: torch.device, dtype: torch.dtype):
        mean = torch.tensor(self.MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        return mean, std

    @staticmethod
    def _rand_uniform(
        batch: int,
        low: float,
        high: float,
        device: torch.device,
        dtype: torch.dtype,
        generator=None,
    ) -> Tensor:
        return torch.empty(batch, device=device, dtype=dtype).uniform_(
            low, high, generator=generator
        )

    def _affine(self, images: Tensor, generator=None) -> Tuple[Tensor, Tensor]:
        batch, _, height, width = images.shape
        device, dtype = images.device, images.dtype
        use = torch.rand(batch, device=device, generator=generator) < self.geometry_probability
        angles = self._rand_uniform(batch, -self.rotation_degrees, self.rotation_degrees, device, dtype, generator)
        scales = self._rand_uniform(batch, self.scale_range[0], self.scale_range[1], device, dtype, generator)
        tx = self._rand_uniform(batch, -self.translate_fraction, self.translate_fraction, device, dtype, generator)
        ty = self._rand_uniform(batch, -self.translate_fraction, self.translate_fraction, device, dtype, generator)
        angles = torch.where(use, angles, torch.zeros_like(angles))
        scales = torch.where(use, scales, torch.ones_like(scales))
        tx = torch.where(use, tx, torch.zeros_like(tx))
        ty = torch.where(use, ty, torch.zeros_like(ty))
        radians = angles * torch.pi / 180.0
        cos, sin = torch.cos(radians), torch.sin(radians)
        theta = torch.zeros(batch, 3, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = cos / scales
        theta[:, 0, 1] = sin / scales
        theta[:, 1, 0] = -sin / scales
        theta[:, 1, 1] = cos / scales
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        theta[:, 2, 2] = 1.0
        grid = F.affine_grid(theta[:, :2], images.shape, align_corners=False)
        shifted = F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        return shifted, theta

    def __call__(self, images: Tensor, generator=None, force_category: Optional[str] = None) -> NormalShiftBatch:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f'expected normalized images [B, 3, H, W], got {tuple(images.shape)}')
        device, dtype = images.device, images.dtype
        batch = images.shape[0]
        mean, std = self._normalization(device, dtype)
        shifted = images * std + mean
        
        # If geometry is specifically forced or sampled
        if force_category == 'geometry':
            old_geo_prob = self.geometry_probability
            self.geometry_probability = 1.0
            shifted, theta = self._affine(shifted, generator=generator)
            self.geometry_probability = old_geo_prob
        else:
            shifted, theta = self._affine(shifted, generator=generator)
            
        families: List[str] = []

        for index in range(batch):
            image = shifted[index:index + 1]
            applied: List[str] = []
            if force_category in ('photometric', 'degradation', 'illumination'):
                candidates = [force_category]
            else:
                candidates = ['photometric', 'degradation', 'illumination']
                # Keep the default view bounded and avoid composing many strong ops.
                candidates = [candidates[i] for i in torch.randperm(len(candidates), device=device, generator=generator).tolist()]
                
            for candidate in candidates[:self.max_ops_per_view]:
                if force_category is None:
                    probability = {
                        'photometric': self.photometric_probability,
                        'degradation': self.degradation_probability,
                        'illumination': self.illumination_probability,
                    }[candidate]
                    if torch.rand((), device=device, generator=generator).item() >= probability:
                        continue
                if candidate == 'photometric':
                    brightness = self._rand_uniform(1, 0.75, 1.25, device, dtype, generator).view(1, 1, 1, 1)
                    contrast = self._rand_uniform(1, 0.75, 1.25, device, dtype, generator).view(1, 1, 1, 1)
                    gamma = self._rand_uniform(1, 0.80, 1.25, device, dtype, generator).view(1, 1, 1, 1)
                    gains = self._rand_uniform(3, 0.85, 1.15, device, dtype, generator).view(1, 3, 1, 1)
                    image = ((image - 0.5) * contrast + 0.5) * brightness
                    image = image.clamp(0.0, 1.0).pow(gamma)
                    image = image * gains
                elif candidate == 'degradation':
                    if self.noise_std > 0:
                        sigma = self._rand_uniform(1, 0.0, self.noise_std, device, dtype, generator).view(1, 1, 1, 1)
                        image = image + torch.randn(image.shape, device=device, dtype=dtype, generator=generator) * sigma
                    if self.blur_sigma > 0 and torch.rand((), device=device, generator=generator).item() < 0.5:
                        # Average pooling is a bounded, dependency-free blur approximation.
                        image = F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
                else:
                    height, width = image.shape[-2:]
                    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
                    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
                    gradient = 1.0 + self._rand_uniform(1, -0.20, 0.20, device, dtype, generator).view(1, 1, 1, 1) * (0.5 * xx + 0.5 * yy)
                    vignette = 1.0 + self._rand_uniform(1, -0.20, 0.20, device, dtype, generator).view(1, 1, 1, 1) * (xx.square() + yy.square())
                    image = image * gradient * vignette
                applied.append(candidate)
            shifted[index:index + 1] = image.clamp(0.0, 1.0)
            families.append('+'.join(applied) if applied else 'geometry')

        shifted = (shifted.clamp(0.0, 1.0) - mean) / std
        return NormalShiftBatch(shifted, theta, families)
