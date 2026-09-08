"""Losses for source-only normal-shift generalization."""

from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def cosine_consistency(
    anchor: Sequence[Tensor],
    shifted: Sequence[Tensor],
    inverse_warp=None,
) -> Tensor:
    """Point-wise cosine distance between canonical features on valid (non-padded) regions."""

    if len(anchor) != len(shifted) or not anchor:
        raise ValueError('anchor and shifted feature lists must be non-empty and equal-sized')
    losses = []
    for anchor_feature, shifted_feature in zip(anchor, shifted):
        aligned = inverse_warp(shifted_feature) if inverse_warp is not None else shifted_feature
        if anchor_feature.shape != aligned.shape:
            raise ValueError(
                f'feature shapes must match after alignment, got '
                f'{tuple(anchor_feature.shape)} and {tuple(aligned.shape)}'
            )
        cos_dist = 1.0 - F.cosine_similarity(anchor_feature, aligned, dim=1)
        if inverse_warp is not None:
            ones = torch.ones(
                anchor_feature.shape[0], 1, anchor_feature.shape[2], anchor_feature.shape[3],
                device=anchor_feature.device, dtype=anchor_feature.dtype
            )
            valid_mask = (inverse_warp(ones, padding_mode='zeros') > 0.95).squeeze(1)
            if valid_mask.any():
                cos_dist = cos_dist[valid_mask]
        losses.append(cos_dist.mean())
    return torch.stack(losses).mean()


def identity_loss(canonical: Sequence[Tensor], raw: Sequence[Tensor]) -> Tensor:
    """Keep the adapter close to the frozen source feature space."""

    return cosine_consistency(raw, canonical)


def map_consistency(anchor_map: Tensor, shifted_map: Tensor, inverse_warp=None) -> Tensor:
    """L1 equivariance loss for normal-view anomaly maps on valid regions."""

    aligned = inverse_warp(shifted_map) if inverse_warp is not None else shifted_map
    if anchor_map.shape != aligned.shape:
        raise ValueError(
            f'map shapes must match after alignment, got {tuple(anchor_map.shape)} '
            f'and {tuple(aligned.shape)}'
        )
    diff = (anchor_map - aligned).abs()
    if inverse_warp is not None:
        ones = torch.ones(
            anchor_map.shape[0], 1, anchor_map.shape[2], anchor_map.shape[3],
            device=anchor_map.device, dtype=anchor_map.dtype
        )
        valid_mask = inverse_warp(ones, padding_mode='zeros') > 0.95
        if valid_mask.any():
            diff = diff[valid_mask]
    return diff.mean()


def preservation_loss(
    anchor: Sequence[Tensor],
    pseudo_anomaly: Sequence[Tensor],
    masks: Tensor,
    margin: float = 0.10,
) -> Tensor:
    """Hinge loss that prevents adapters from removing local pseudo defects."""

    if margin < 0:
        raise ValueError('margin must be non-negative')
    if len(anchor) != len(pseudo_anomaly) or not anchor:
        raise ValueError('anchor and pseudo_anomaly feature lists must be equal-sized and non-empty')
    losses = []
    for anchor_feature, anomaly_feature in zip(anchor, pseudo_anomaly):
        if anchor_feature.shape != anomaly_feature.shape:
            raise ValueError('anchor and pseudo-anomaly feature shapes must match')
        feature_mask = F.interpolate(
            masks.float(),
            size=anchor_feature.shape[-2:],
            mode='nearest',
        )
        distance = 1.0 - F.cosine_similarity(anchor_feature, anomaly_feature, dim=1, eps=1e-8)
        selected = distance[feature_mask[:, 0] > 0.5]
        if selected.numel() == 0:
            continue
        losses.append(F.relu(float(margin) - selected.mean()))
    if not losses:
        return anchor[0].new_zeros(())
    return torch.stack(losses).mean()


def anomaly_margin_loss(
    normal_map: Tensor,
    anomaly_map: Tensor,
    masks: Tensor,
    margin: float = 0.05,
) -> Tensor:
    """Optional image-map margin for pseudo-anomaly score preservation."""

    if margin < 0:
        raise ValueError('margin must be non-negative')
    if normal_map.shape != anomaly_map.shape:
        raise ValueError('normal_map and anomaly_map must have the same shape')
    mask = F.interpolate(masks.float(), size=normal_map.shape[-2:], mode='nearest')[:, 0] > 0.5
    normal_values = normal_map[:, 0][mask]
    anomaly_values = anomaly_map[:, 0][mask]
    if normal_values.numel() == 0:
        return normal_map.new_zeros(())
    return F.relu(float(margin) + normal_values.mean() - anomaly_values.mean())
