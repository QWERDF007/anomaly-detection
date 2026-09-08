"""Small local pseudo-anomaly generator for adapter preservation training."""

from typing import Optional

import torch
from torch import Tensor


class PseudoAnomalyGenerator:
    def __init__(self, min_area_ratio: float = 1e-4, max_area_ratio: float = 0.02):
        if not 0 < min_area_ratio <= max_area_ratio:
            raise ValueError('area ratios must be positive and ordered')
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)

    def __call__(self, images: Tensor, generator=None):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('expected images with shape [B, 3, H, W]')
        batch, _, height, width = images.shape
        output = images.clone()
        masks = torch.zeros(batch, 1, height, width, device=images.device, dtype=images.dtype)
        for index in range(batch):
            flaw_type = torch.randint(0, 4, (), device=images.device, generator=generator).item()
            
            if flaw_type == 0:  # Scratch: Slender elongated defect
                length = int(torch.empty((), device=images.device).uniform_(15, min(80, height // 2), generator=generator).item())
                thickness = int(torch.empty((), device=images.device).uniform_(1, 4, generator=generator).item())
                horizontal = bool(torch.rand((), device=images.device, generator=generator).item() > 0.5)
                box_h, box_w = (thickness, length) if horizontal else (length, thickness)
            elif flaw_type == 1:  # Pinhole / Small Spot
                spot_size = int(torch.empty((), device=images.device).uniform_(2, 7, generator=generator).item())
                box_h, box_w = spot_size, spot_size
            elif flaw_type == 2:  # Texture / Noise perturbation patch
                ratio = float(torch.empty((), device=images.device, dtype=images.dtype).uniform_(
                    self.min_area_ratio, self.max_area_ratio, generator=generator
                ))
                area = max(1, int(round(ratio * height * width)))
                aspect = float(torch.empty((), device=images.device, dtype=images.dtype).uniform_(0.5, 2.0, generator=generator))
                box_h = max(1, min(height, int(round((area / max(aspect, 1e-6)) ** 0.5))))
                box_w = max(1, min(width, int(round(area / box_h))))
            else:  # General Blob Patch
                ratio = float(torch.empty((), device=images.device, dtype=images.dtype).uniform_(
                    self.min_area_ratio, self.max_area_ratio, generator=generator
                ))
                area = max(1, int(round(ratio * height * width)))
                aspect = float(torch.empty((), device=images.device, dtype=images.dtype).uniform_(0.25, 4.0, generator=generator))
                box_h = max(1, min(height, int(round((area / max(aspect, 1e-6)) ** 0.5))))
                box_w = max(1, min(width, int(round(area / box_h))))

            box_h = max(1, min(height - 1, box_h))
            box_w = max(1, min(width - 1, box_w))
            top = int(torch.randint(0, max(height - box_h + 1, 1), (), device=images.device, generator=generator))
            left = int(torch.randint(0, max(width - box_w + 1, 1), (), device=images.device, generator=generator))
            masks[index, :, top:top + box_h, left:left + box_w] = 1.0
            patch = output[index, :, top:top + box_h, left:left + box_w]
            
            if flaw_type == 2:  # High-frequency noise perturbation
                noise = torch.randn_like(patch) * 0.8
                output[index, :, top:top + box_h, left:left + box_w] = (patch + noise).clamp(-2.5, 2.5)
            else:
                color = torch.rand(3, 1, 1, device=images.device, dtype=images.dtype, generator=generator) * 2.0 - 1.0
                output[index, :, top:top + box_h, left:left + box_w] = 0.4 * patch + 0.6 * color
        return output, masks
