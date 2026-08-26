#!/usr/bin/env python3
"""Track 3: Multi-Scale Guided Filter & Bilateral Sharpening Pyramid on 672 Predictions.

Comprehensive experimental script evaluating:
1. Single-Scale Guided Filter (Self-guided, Gray-guided, RGB-guided) with r in [2, 4, 8] and eps in [1e-4, 1e-3, 1e-2]
2. Multi-Scale Guided Filter Pyramid with multi-radius residual decomposition and detail boosting
3. Bilateral Edge-Preserving Filter with sigma_color in [5, 10, 20] and sigma_space in [3, 5, 10]
4. Bilateral Sharpening (Unsharp Masking) with sharpening factors beta in [0.5, 1.0, 1.5, 2.0]
5. Multi-Scale Bilateral Smoothing Pyramid with progressive octave decomposition
6. Cascaded Guided + Bilateral Hybrid Pipelines with boundary preservation and noise suppression
7. Quantitative metrics: P-AP, P-AUPRO, P-AUROC, P-F1, R-MissRate, R-FP-RegionCount, R-PixelCoverage, R-FPR, I-AUROC, I-AP, I-F1

Runs on GPU 4 / GPU 5 using cached score maps and ground truth masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from sklearn.metrics import auc
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(1, str(_UTILS_DIR))

from anomaly_evaluation import (
    safe_auroc,
    safe_ap,
    max_f1,
    training_image_score,
)

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030


# ============================================================================
# Fast GPU / CPU Guided Filter Implementations
# ============================================================================

def guided_filter_torch(
    p: torch.Tensor,
    I: torch.Tensor,
    r: int = 4,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Vectorized Guided Filter in PyTorch on CUDA/CPU.
    
    Args:
        p: Input anomaly score tensor (B, 1, H, W).
        I: Guidance image tensor (B, 1, H, W) for grayscale/self or (B, 3, H, W) for RGB.
        r: Filter radius. Window size = 2 * r + 1.
        eps: Regularization parameter.
    
    Returns:
        Filtered score map tensor (B, 1, H, W).
    """
    kernel_size = 2 * r + 1
    pad = r

    def box_filter(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)

    if I.shape[1] == 1:
        # Grayscale / Self Guidance
        mean_I = box_filter(I)
        mean_p = box_filter(p)
        corr_I = box_filter(I * I)
        corr_Ip = box_filter(I * p)

        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a)
        mean_b = box_filter(b)

        q = mean_a * I + mean_b
        return q
    else:
        # 3-Channel Color Guidance (RGB)
        B, C, H, W = I.shape
        mean_I = box_filter(I)  # (B, 3, H, W)
        mean_p = box_filter(p)  # (B, 1, H, W)
        corr_Ip = box_filter(I * p)  # (B, 3, H, W)
        cov_Ip = corr_Ip - mean_I * mean_p  # (B, 3, H, W)

        # Covariance matrix components
        var_I_rr = box_filter(I[:, 0:1] * I[:, 0:1]) - mean_I[:, 0:1] * mean_I[:, 0:1] + eps
        var_I_rg = box_filter(I[:, 0:1] * I[:, 1:2]) - mean_I[:, 0:1] * mean_I[:, 1:2]
        var_I_rb = box_filter(I[:, 0:1] * I[:, 2:3]) - mean_I[:, 0:1] * mean_I[:, 2:3]
        var_I_gg = box_filter(I[:, 1:2] * I[:, 1:2]) - mean_I[:, 1:2] * mean_I[:, 1:2] + eps
        var_I_gb = box_filter(I[:, 1:2] * I[:, 2:3]) - mean_I[:, 1:2] * mean_I[:, 2:3]
        var_I_bb = box_filter(I[:, 2:3] * I[:, 2:3]) - mean_I[:, 2:3] * mean_I[:, 2:3] + eps

        sigma = torch.zeros((B, H, W, 3, 3), device=p.device, dtype=p.dtype)
        sigma[..., 0, 0] = var_I_rr.squeeze(1)
        sigma[..., 0, 1] = var_I_rg.squeeze(1)
        sigma[..., 0, 2] = var_I_rb.squeeze(1)
        sigma[..., 1, 0] = var_I_rg.squeeze(1)
        sigma[..., 1, 1] = var_I_gg.squeeze(1)
        sigma[..., 1, 2] = var_I_gb.squeeze(1)
        sigma[..., 2, 0] = var_I_rb.squeeze(1)
        sigma[..., 2, 1] = var_I_gb.squeeze(1)
        sigma[..., 2, 2] = var_I_bb.squeeze(1)

        cov_Ip_perm = cov_Ip.permute(0, 2, 3, 1).unsqueeze(-1)  # (B, H, W, 3, 1)
        a = torch.linalg.solve(sigma, cov_Ip_perm).squeeze(-1).permute(0, 3, 1, 2)  # (B, 3, H, W)
        b = mean_p - (a * mean_I).sum(dim=1, keepdim=True)

        mean_a = box_filter(a)
        mean_b = box_filter(b)

        q = (mean_a * I).sum(dim=1, keepdim=True) + mean_b
        return q


def multiscale_guided_filter_torch(
    p: torch.Tensor,
    I: torch.Tensor,
    radii: Sequence[int] = (2, 4, 8),
    eps: float = 1e-3,
    detail_boost: Sequence[float] = (1.2, 1.0),
    fusion_mode: str = "residual",  # "residual" or "average"
) -> torch.Tensor:
    """Multi-Scale Guided Filter Pyramid on GPU.
    
    Decomposes score map into multi-scale residual bands or weighted multi-scale filters.
    """
    filtered_levels = [guided_filter_torch(p, I, r=r, eps=eps) for r in radii]
    
    if fusion_mode == "average":
        weights = torch.tensor([1.0 / len(radii)] * len(radii), device=p.device, dtype=p.dtype)
        res = sum(w * fl for w, fl in zip(weights, filtered_levels))
        return res
    
    elif fusion_mode == "residual":
        # Multi-scale detail boosting: Base is coarsest (largest radius)
        # Detail layers are differences between adjacent scales
        base = filtered_levels[-1]
        reconstructed = base.clone()
        for idx in range(len(radii) - 1):
            detail = filtered_levels[idx] - filtered_levels[idx + 1]
            boost = detail_boost[idx] if idx < len(detail_boost) else 1.0
            reconstructed = reconstructed + boost * detail
        return torch.clamp(reconstructed, min=0.0)
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")


# ============================================================================
# Bilateral Filtering & Sharpening Implementations
# ============================================================================

def apply_bilateral_filter_single(
    score_map: np.ndarray,
    d: int = 9,
    sigma_color: float = 10.0,
    sigma_space: float = 5.0,
    is_scaled_255: bool = True,
) -> np.ndarray:
    """OpenCV bilateral filter on a single 2D float32 score map."""
    if is_scaled_255:
        max_v = float(score_map.max())
        if max_v <= 1e-8:
            return score_map.copy()
        scale = 255.0 / max(max_v, 1e-6)
        scaled_map = (score_map * scale).astype(np.float32)
        filtered_scaled = cv2.bilateralFilter(
            scaled_map, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space)
        )
        return (filtered_scaled / scale).astype(np.float32)
    else:
        eff_sigma_color = float(sigma_color)
        if eff_sigma_color > 1.0:
            eff_sigma_color = eff_sigma_color / 255.0 * 0.1
        return cv2.bilateralFilter(
            score_map.astype(np.float32), d=d, sigmaColor=eff_sigma_color, sigmaSpace=float(sigma_space)
        )


def apply_bilateral_sharpening_single(
    score_map: np.ndarray,
    d: int = 9,
    sigma_color: float = 10.0,
    sigma_space: float = 5.0,
    beta: float = 1.0,
    is_scaled_255: bool = True,
) -> np.ndarray:
    """Bilateral Unsharp Masking / Sharpening.
    
    Formula: S_sharp = S + beta * (S - Bilateral(S, d, sigma_c, sigma_s))
    """
    smoothed = apply_bilateral_filter_single(
        score_map, d=d, sigma_color=sigma_color, sigma_space=sigma_space, is_scaled_255=is_scaled_255
    )
    detail = score_map - smoothed
    sharpened = score_map + beta * detail
    return np.maximum(sharpened, 0.0).astype(np.float32)


def apply_multiscale_bilateral_pyramid_single(
    score_map: np.ndarray,
    sigmas_color: Sequence[float] = (5.0, 10.0, 20.0),
    sigmas_space: Sequence[float] = (3.0, 5.0, 10.0),
    detail_gains: Sequence[float] = (1.5, 1.2),
    d_list: Sequence[int] = (5, 9, 15),
) -> np.ndarray:
    """Multi-Scale Bilateral Smoothing Pyramid decomposition and detail reconstruction."""
    levels = []
    curr = score_map
    for sc, ss, d in zip(sigmas_color, sigmas_space, d_list):
        filtered = apply_bilateral_filter_single(curr, d=d, sigma_color=sc, sigma_space=ss, is_scaled_255=True)
        levels.append(filtered)
        curr = filtered

    base = levels[-1]
    reconstructed = base.copy()
    for i in range(len(levels) - 1):
        detail = levels[i] - levels[i + 1]
        gain = detail_gains[i] if i < len(detail_gains) else 1.0
        reconstructed = reconstructed + gain * detail

    return np.maximum(reconstructed, 0.0).astype(np.float32)


# ============================================================================
# High-Speed Vectorized GPU Metrics Engine
# ============================================================================

def fast_region_detection_metrics(
    masks: np.ndarray,
    score_maps: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Fast Region detection metrics."""
    masks = np.asarray(masks, dtype=np.uint8)
    score_maps = np.asarray(score_maps, dtype=np.float32)

    image_miss_rates = []
    image_pixel_coverages = []
    image_fp_rates = []
    total_regions = 0
    total_fp_regions = 0

    for mask, score_map in zip(masks, score_maps):
        gt_labels = measure.label(mask.astype(bool))
        region_count = int(gt_labels.max())
        prediction = score_map >= threshold
        pred_labels = measure.label(prediction.astype(np.uint8))
        pred_count = int(pred_labels.max())

        detected = 0
        region_coverages = []
        for region_id in range(1, region_count + 1):
            region_mask = (gt_labels == region_id)
            covered_pixels = int(prediction[region_mask].sum())
            if covered_pixels:
                detected += 1
            region_coverages.append(covered_pixels / int(region_mask.sum()))

        missed = region_count - detected

        if pred_count > 0:
            if region_count > 0:
                overlapping_pred_ids = np.unique(pred_labels[gt_labels > 0])
                tp = int(np.count_nonzero(overlapping_pred_ids > 0))
            else:
                tp = 0
            fp = pred_count - tp
        else:
            tp = 0
            fp = 0

        total_fp_regions += fp
        if pred_count > 0:
            image_fp_rates.append(fp / pred_count)
        else:
            image_fp_rates.append(0.0)

        if region_count > 0:
            image_miss_rates.append(missed / region_count)
            image_pixel_coverages.append(float(np.mean(region_coverages)))
        total_regions += region_count

    return {
        "R-MissRate": float(np.mean(image_miss_rates)) if image_miss_rates else float("nan"),
        "R-PixelCoverage": float(np.mean(image_pixel_coverages)) if image_pixel_coverages else float("nan"),
        "R-FPR": float(np.mean(image_fp_rates)) if image_fp_rates else float("nan"),
        "R-FP-RegionCount": float(total_fp_regions),
        "R-GT-RegionCount": float(total_regions),
    }


class FastMetricEvaluator:
    """High-speed GPU batch evaluator for AUPRO, AUROC, AP, F1, and Region metrics."""

    def __init__(self, bad_gt_masks_256: np.ndarray, labels: np.ndarray, device: torch.device):
        self.bad_gt_masks = bad_gt_masks_256.astype(np.uint8)
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.device = device

        # Precompute flattened tensors on GPU for fast pixel-level evaluation
        pix_labels_np = self.bad_gt_masks.reshape(-1)
        self.t_pix_labels = torch.from_numpy(pix_labels_np).to(device=device, dtype=torch.float32)
        self.t_labels = torch.from_numpy(self.labels).to(device=device, dtype=torch.float32)

        # Precompute region connected components & background indices for fast GPU AUPRO
        self.region_indices_list = []
        self.bg_indices_list = []
        self.total_regions = 0

        for mask in self.bad_gt_masks:
            lbls = measure.label(mask)
            n_regs = int(lbls.max())
            self.total_regions += n_regs
            regs = [np.where(lbls == r_id) for r_id in range(1, n_regs + 1)]
            self.region_indices_list.append(regs)
            self.bg_indices_list.append(np.where(mask == 0))

    def evaluate_pixel_metrics_gpu(self, bad_scores_np: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute P-AUROC, P-AP, P-F1, and P-AUPRO completely on GPU."""
        t_scores_flat = torch.from_numpy(bad_scores_np.reshape(-1)).to(device=self.device, dtype=torch.float32)

        # 1. Sort scores on GPU
        sorted_scores, order = torch.sort(t_scores_flat, descending=True)
        sorted_labels = self.t_pix_labels[order]

        tp_cumsum = torch.cumsum(sorted_labels, dim=0)
        fp_cumsum = torch.cumsum(1.0 - sorted_labels, dim=0)
        total_tp = tp_cumsum[-1]
        total_fp = fp_cumsum[-1]

        recalls = tp_cumsum / (total_tp + 1e-12)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-12)
        fprs = fp_cumsum / (total_fp + 1e-12)

        # P-AUROC via trapezoidal integration
        fpr_diff = torch.cat([fprs[0:1], fprs[1:] - fprs[:-1]])
        tpr_avg = torch.cat([recalls[0:1] / 2.0, (recalls[1:] + recalls[:-1]) / 2.0])
        p_auroc = (fpr_diff * tpr_avg).sum().item()

        # P-AP
        recall_diff = torch.cat([recalls[0:1], recalls[1:] - recalls[:-1]])
        p_ap = (recall_diff * precisions).sum().item()

        # P-F1
        f1 = 2.0 * precisions * recalls / (precisions + recalls + 1e-7)
        p_f1 = f1.max().item()

        # 2. P-AUPRO on GPU
        min_val = float(sorted_scores[-1].item())
        max_val = float(sorted_scores[0].item())
        delta = (max_val - min_val) / 200.0
        if delta <= 0.0:
            p_aupro = 0.0
        else:
            thresholds = np.arange(min_val, max_val, delta, dtype=np.float32)
            t_ths = torch.from_numpy(thresholds).to(self.device)
            t_scores = torch.from_numpy(bad_scores_np).to(self.device)

            bg_vals_all = []
            pro_sums = torch.zeros(len(thresholds), device=self.device, dtype=torch.float64)
            total_bg_count = 0

            for i, (regs, bg_idx) in enumerate(zip(self.region_indices_list, self.bg_indices_list)):
                s_img = t_scores[i]
                bg_vals = s_img[bg_idx[0], bg_idx[1]]
                bg_vals_all.append(bg_vals)
                total_bg_count += bg_vals.numel()

                for reg in regs:
                    reg_vals = s_img[reg[0], reg[1]].sort().values
                    area = reg_vals.numel()
                    hits = area - torch.searchsorted(reg_vals, t_ths, right=True)
                    pro_sums += hits.double() / area

            all_bg_vals = torch.cat(bg_vals_all).sort().values
            fp_counts = all_bg_vals.numel() - torch.searchsorted(all_bg_vals, t_ths, right=True)
            pro_fprs = (fp_counts.double() / total_bg_count).cpu().numpy()
            pros = (pro_sums / self.total_regions).cpu().numpy()

            valid = pro_fprs < 0.3
            if not np.any(valid):
                p_aupro = float("nan")
            else:
                pro_fprs = pro_fprs[valid]
                pros = pros[valid]
                max_fpr = float(pro_fprs.max())
                p_aupro = float(auc(pro_fprs / max_fpr, pros)) if max_fpr > 0 else float("nan")

        return p_auroc, p_ap, p_f1, p_aupro

    def evaluate_all(
        self,
        all_scores_256: np.ndarray,
    ) -> Dict[str, float]:
        """Compute full suite of image, pixel, and region metrics rapidly."""
        # 1. Fast GPU image scores (mean of highest 1% pixels per image)
        t_all = torch.from_numpy(all_scores_256).to(device=self.device, dtype=torch.float32)
        B, H, W = t_all.shape
        top_k = max(1, int(H * W * 0.01))
        t_top = t_all.reshape(B, -1).topk(top_k, dim=-1).values.mean(dim=-1)
        image_scores = t_top.cpu().numpy()

        i_auroc = safe_auroc(self.labels, image_scores)
        i_ap = safe_ap(self.labels, image_scores)
        i_f1 = max_f1(self.labels, image_scores)

        # 2. Pixel metrics
        bad_scores_256 = all_scores_256[self.labels == 1]
        p_auroc, p_ap, p_f1, p_aupro = self.evaluate_pixel_metrics_gpu(bad_scores_256)

        # 3. Region metrics
        reg = fast_region_detection_metrics(self.bad_gt_masks, bad_scores_256, GOOD_THRESHOLD)

        return {
            "P-AP": p_ap,
            "P-AUPRO": p_aupro,
            "P-AUROC": p_auroc,
            "P-F1": p_f1,
            "R-MissRate": reg["R-MissRate"],
            "R-FP-RegionCount": reg["R-FP-RegionCount"],
            "R-PixelCoverage": reg["R-PixelCoverage"],
            "R-FPR": reg["R-FPR"],
            "I-AUROC": i_auroc,
            "I-AP": i_ap,
            "I-F1": i_f1,
        }


# ============================================================================
# Cached Dataset Loading
# ============================================================================

def load_evaluation_records(
    root: Path,
    data_root: Path,
    gt_dir: Path,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load and prepare all 680 test images, resized score maps, GT masks, and guidance images."""
    cache_pkl = root / "preds" / "cached_eval_records.pkl"
    print(f"Loading predictions cache from {cache_pkl}...")
    t0 = time.time()
    import pickle
    with open(cache_pkl, "rb") as f:
        raw_records = pickle.load(f)
    print(f"Loaded {len(raw_records)} records in {time.time() - t0:.2f}s.")

    score_maps_256 = []
    gt_masks_256 = []
    labels = []
    gray_guides_256 = []
    rgb_guides_256 = []

    print("Pre-processing guidance images and resized score maps...")
    t1 = time.time()
    for rec in tqdm(raw_records, desc="Prepping images", unit="img"):
        is_bad = (rec["dataset_label"] != "good")
        labels.append(1 if is_bad else 0)

        # 256x256 score map
        sm = cv2.resize(rec["score_map"], target_size, interpolation=cv2.INTER_LINEAR)
        score_maps_256.append(sm)

        # GT Mask
        if is_bad:
            gt_masks_256.append(rec["gt_mask_256"])
        
        # Load raw guidance image
        img_rel = Path(rec["image_relative"])
        img_full_path = data_root / img_rel
        if img_full_path.is_file():
            raw_img = cv2.imread(str(img_full_path))
            raw_img_256 = cv2.resize(raw_img, target_size, interpolation=cv2.INTER_LINEAR)
            gray_256 = cv2.cvtColor(raw_img_256, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            rgb_256 = cv2.cvtColor(raw_img_256, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            gray_256 = sm.copy()
            rgb_256 = np.stack([sm, sm, sm], axis=-1)

        gray_guides_256.append(gray_256)
        rgb_guides_256.append(rgb_256)

    score_maps_256 = np.stack(score_maps_256).astype(np.float32)
    gt_masks_256 = np.stack(gt_masks_256).astype(np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    gray_guides_256 = np.stack(gray_guides_256).astype(np.float32)
    rgb_guides_256 = np.stack(rgb_guides_256).astype(np.float32)

    print(f"Pre-processing complete in {time.time() - t1:.2f}s.")
    return score_maps_256, gt_masks_256, labels, gray_guides_256, rgb_guides_256, raw_records


# ============================================================================
# Main Experiment Execution Matrix
# ============================================================================

def build_experiment_matrix() -> List[Dict[str, Any]]:
    """Build comprehensive experimental configuration matrix."""
    matrix = []

    # -------------------------------------------------------------
    # 0. Baselines
    # -------------------------------------------------------------
    matrix.append({
        "name": "00_Raw_Base_672_Baseline",
        "category": "0_Baseline",
        "type": "baseline",
    })

    # -------------------------------------------------------------
    # 1. Single-Scale Guided Filter Sweep (Self, Gray, RGB)
    # Radii in [2, 4, 8], eps in [1e-4, 1e-3, 1e-2]
    # -------------------------------------------------------------
    radii = [2, 4, 8]
    eps_list = [1e-4, 1e-3, 1e-2]

    # 1A. Self-Guided Filter (I = S)
    for r in radii:
        for eps in eps_list:
            matrix.append({
                "name": f"10_GF_Self (r={r}, eps={eps:.0e})",
                "category": "1A_Guided_Self",
                "type": "guided_self",
                "r": r,
                "eps": eps,
            })

    # 1B. Grayscale-Guided Filter (I = Gray)
    for r in radii:
        for eps in eps_list:
            matrix.append({
                "name": f"11_GF_Gray (r={r}, eps={eps:.0e})",
                "category": "1B_Guided_Gray",
                "type": "guided_gray",
                "r": r,
                "eps": eps,
            })

    # 1C. RGB-Guided Filter (I = RGB)
    for r in radii:
        for eps in eps_list:
            matrix.append({
                "name": f"12_GF_RGB (r={r}, eps={eps:.0e})",
                "category": "1C_Guided_RGB",
                "type": "guided_rgb",
                "r": r,
                "eps": eps,
            })

    # -------------------------------------------------------------
    # 2. Multi-Scale Guided Filter Pyramid
    # -------------------------------------------------------------
    # 2A. Multi-scale Residual Detail Boosting
    for eps in [1e-4, 1e-3, 1e-2]:
        for boost in [(1.0, 1.0), (1.2, 1.0), (1.5, 1.2), (2.0, 1.5)]:
            matrix.append({
                "name": f"20_MS_GF_Pyramid (radii=[2,4,8], eps={eps:.0e}, boost={boost[0]}/{boost[1]})",
                "category": "2A_MultiScale_GF_Pyramid",
                "type": "multiscale_gf",
                "radii": (2, 4, 8),
                "eps": eps,
                "detail_boost": boost,
                "fusion_mode": "residual",
                "guide_mode": "self",
            })
            matrix.append({
                "name": f"21_MS_GF_Gray_Pyramid (radii=[2,4,8], eps={eps:.0e}, boost={boost[0]}/{boost[1]})",
                "category": "2A_MultiScale_GF_Pyramid",
                "type": "multiscale_gf",
                "radii": (2, 4, 8),
                "eps": eps,
                "detail_boost": boost,
                "fusion_mode": "residual",
                "guide_mode": "gray",
            })

    # 2B. Multi-scale Averaging
    for eps in [1e-4, 1e-3]:
        matrix.append({
            "name": f"22_MS_GF_Average (radii=[2,4,8], eps={eps:.0e})",
            "category": "2B_MultiScale_GF_Avg",
            "type": "multiscale_gf",
            "radii": (2, 4, 8),
            "eps": eps,
            "fusion_mode": "average",
            "guide_mode": "self",
        })

    # -------------------------------------------------------------
    # 3. Bilateral Edge-Preserving Filter Smoothing
    # sigma_color in [5, 10, 20], sigma_space in [3, 5, 10], d in [5, 9, 15]
    # -------------------------------------------------------------
    for sc in [5, 10, 20]:
        for ss in [3, 5, 10]:
            for d in [5, 9]:
                matrix.append({
                    "name": f"30_Bilateral_Smooth (d={d}, sc={sc}, ss={ss})",
                    "category": "3_Bilateral_Smooth",
                    "type": "bilateral_smooth",
                    "d": d,
                    "sigma_color": float(sc),
                    "sigma_space": float(ss),
                })

    # -------------------------------------------------------------
    # 4. Bilateral Sharpening (Unsharp Masking)
    # sigma_color in [5, 10, 20], beta in [0.5, 1.0, 1.5, 2.0]
    # -------------------------------------------------------------
    for sc in [5, 10, 20]:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            matrix.append({
                "name": f"40_Bilateral_Sharp (sc={sc}, ss=5, beta={beta})",
                "category": "4_Bilateral_Sharpening",
                "type": "bilateral_sharp",
                "d": 9,
                "sigma_color": float(sc),
                "sigma_space": 5.0,
                "beta": beta,
            })

    # -------------------------------------------------------------
    # 5. Multi-Scale Bilateral Smoothing Pyramid
    # -------------------------------------------------------------
    for gains in [(1.0, 1.0), (1.5, 1.2), (2.0, 1.5)]:
        matrix.append({
            "name": f"50_MS_Bilateral_Pyramid (sc=[5,10,20], gains={gains[0]}/{gains[1]})",
            "category": "5_MS_Bilateral_Pyramid",
            "type": "multiscale_bilateral",
            "sigmas_color": (5.0, 10.0, 20.0),
            "sigmas_space": (3.0, 5.0, 10.0),
            "detail_gains": gains,
            "d_list": (5, 9, 15),
        })

    # -------------------------------------------------------------
    # 6. Cascaded Guided + Bilateral Hybrids & Advanced Fusion
    # -------------------------------------------------------------
    # 6A. Guided Filter -> Bilateral Sharpening
    for r in [2, 4]:
        for eps in [1e-4, 1e-3]:
            for sc in [5, 10, 20]:
                for beta in [0.5, 1.0]:
                    matrix.append({
                        "name": f"60_GF(r={r},eps={eps:.0e}) -> Bilateral_Sharp(sc={sc},beta={beta})",
                        "category": "6A_Cascaded_Hybrid",
                        "type": "cascaded_gf_bilateral",
                        "r": r,
                        "eps": eps,
                        "d": 9,
                        "sigma_color": float(sc),
                        "sigma_space": 5.0,
                        "beta": beta,
                        "guide_mode": "self",
                    })

    # 6B. Multi-Scale Guided Pyramid -> Bilateral Sharpening
    for beta in [0.5, 1.0, 1.5]:
        matrix.append({
            "name": f"61_MS_GF_Pyramid -> Bilateral_Sharp(sc=10, beta={beta})",
            "category": "6B_Pyramid_Hybrid",
            "type": "ms_gf_bilateral_sharp",
            "radii": (2, 4, 8),
            "eps": 1e-3,
            "detail_boost": (1.2, 1.0),
            "d": 9,
            "sigma_color": 10.0,
            "sigma_space": 5.0,
            "beta": beta,
        })

    # 6C. Guided Filter + Adaptive Noise Floor Subtraction + Bilateral Sharpening
    for p_floor in [20, 30, 40]:
        matrix.append({
            "name": f"62_GF(r=2,eps=1e-3) + BG_Floor(p={p_floor}%) + Bilateral_Sharp(sc=10,beta=1.0)",
            "category": "6C_Floor_Hybrid",
            "type": "gf_floor_bilateral",
            "r": 2,
            "eps": 1e-3,
            "p_floor": float(p_floor),
            "d": 9,
            "sigma_color": 10.0,
            "sigma_space": 5.0,
            "beta": 1.0,
        })

    return matrix


def run_filter_experiment(
    config: Dict[str, Any],
    raw_scores_256: np.ndarray,
    gray_guides_256: np.ndarray,
    rgb_guides_256: np.ndarray,
    evaluator: FastMetricEvaluator,
    device: torch.device,
) -> Dict[str, Any]:
    """Execute filtering and compute all metrics for a configuration."""
    t0 = time.time()
    cfg_type = config["type"]
    N, H, W = raw_scores_256.shape

    # -------------------------------------------------------------
    # 0. Baseline
    # -------------------------------------------------------------
    if cfg_type == "baseline":
        filtered_scores = raw_scores_256.copy()

    # -------------------------------------------------------------
    # 1. Guided Filter (PyTorch CUDA Accelerated)
    # -------------------------------------------------------------
    elif cfg_type == "guided_self":
        t_p = torch.from_numpy(raw_scores_256).unsqueeze(1).to(device)
        t_out = guided_filter_torch(t_p, t_p, r=config["r"], eps=config["eps"])
        filtered_scores = t_out.squeeze(1).cpu().numpy()

    elif cfg_type == "guided_gray":
        t_p = torch.from_numpy(raw_scores_256).unsqueeze(1).to(device)
        t_I = torch.from_numpy(gray_guides_256).unsqueeze(1).to(device)
        t_out = guided_filter_torch(t_p, t_I, r=config["r"], eps=config["eps"])
        filtered_scores = t_out.squeeze(1).cpu().numpy()

    elif cfg_type == "guided_rgb":
        t_p = torch.from_numpy(raw_scores_256).unsqueeze(1).to(device)
        t_I = torch.from_numpy(rgb_guides_256).permute(0, 3, 1, 2).to(device)
        t_out = guided_filter_torch(t_p, t_I, r=config["r"], eps=config["eps"])
        filtered_scores = t_out.squeeze(1).cpu().numpy()

    # -------------------------------------------------------------
    # 2. Multi-Scale Guided Filter Pyramid
    # -------------------------------------------------------------
    elif cfg_type == "multiscale_gf":
        t_p = torch.from_numpy(raw_scores_256).unsqueeze(1).to(device)
        if config.get("guide_mode") == "gray":
            t_I = torch.from_numpy(gray_guides_256).unsqueeze(1).to(device)
        else:
            t_I = t_p
        t_out = multiscale_guided_filter_torch(
            t_p,
            t_I,
            radii=config["radii"],
            eps=config["eps"],
            detail_boost=config.get("detail_boost", (1.0, 1.0)),
            fusion_mode=config.get("fusion_mode", "residual"),
        )
        filtered_scores = t_out.squeeze(1).cpu().numpy()

    # -------------------------------------------------------------
    # 3. Bilateral Filter Smoothing
    # -------------------------------------------------------------
    elif cfg_type == "bilateral_smooth":
        d = config["d"]
        sc = config["sigma_color"]
        ss = config["sigma_space"]
        filtered_scores = np.empty_like(raw_scores_256)
        for i in range(N):
            filtered_scores[i] = apply_bilateral_filter_single(
                raw_scores_256[i], d=d, sigma_color=sc, sigma_space=ss, is_scaled_255=True
            )

    # -------------------------------------------------------------
    # 4. Bilateral Sharpening
    # -------------------------------------------------------------
    elif cfg_type == "bilateral_sharp":
        d = config["d"]
        sc = config["sigma_color"]
        ss = config["sigma_space"]
        beta = config["beta"]
        filtered_scores = np.empty_like(raw_scores_256)
        for i in range(N):
            filtered_scores[i] = apply_bilateral_sharpening_single(
                raw_scores_256[i], d=d, sigma_color=sc, sigma_space=ss, beta=beta, is_scaled_255=True
            )

    # -------------------------------------------------------------
    # 5. Multi-Scale Bilateral Pyramid
    # -------------------------------------------------------------
    elif cfg_type == "multiscale_bilateral":
        filtered_scores = np.empty_like(raw_scores_256)
        for i in range(N):
            filtered_scores[i] = apply_multiscale_bilateral_pyramid_single(
                raw_scores_256[i],
                sigmas_color=config["sigmas_color"],
                sigmas_space=config["sigmas_space"],
                detail_gains=config["detail_gains"],
                d_list=config["d_list"],
            )

    # -------------------------------------------------------------
    # 6. Cascaded Guided + Bilateral Hybrids
    # -------------------------------------------------------------
    elif cfg_type == "cascaded_gf_bilateral":
        t_p = torch.from_numpy(raw_scores_256).unsqueeze(1).to(device)
        t_out = guided_filter_torch(t_p, t_p, r=config["r"], eps=config["eps"])
        gf_np = t_out.squeeze(1).cpu().numpy()

        d = config["d"]
        sc = config["sigma_color"]
        ss = config["sigma_space"]
        beta = config["beta"]
        filtered_scores = np.empty_like(raw_scores_256)
        for i in range(N):
            filtered_scores[i] = apply_bilateral_sharpening_single(
                gf_np[i], d=d, sigma_color=sc, sigma_space=ss, beta=beta, is_scaled_255=True
            )

    elif cfg_type == "ms_gf_bilateral_sharp":
        t_p = torch.from_numpy(raw_scores_256).unsqueeze(1).to(device)
        t_out = multiscale_guided_filter_torch(
            t_p,
            t_p,
            radii=config["radii"],
            eps=config["eps"],
            detail_boost=config.get("detail_boost", (1.2, 1.0)),
            fusion_mode="residual",
        )
        ms_gf_np = t_out.squeeze(1).cpu().numpy()

        d = config["d"]
        sc = config["sigma_color"]
        ss = config["sigma_space"]
        beta = config["beta"]
        filtered_scores = np.empty_like(raw_scores_256)
        for i in range(N):
            filtered_scores[i] = apply_bilateral_sharpening_single(
                ms_gf_np[i], d=d, sigma_color=sc, sigma_space=ss, beta=beta, is_scaled_255=True
            )

    elif cfg_type == "gf_floor_bilateral":
        t_p = torch.from_numpy(raw_scores_256).unsqueeze(1).to(device)
        t_out = guided_filter_torch(t_p, t_p, r=config["r"], eps=config["eps"])
        gf_np = t_out.squeeze(1).cpu().numpy()

        p_floor = config["p_floor"]
        d = config["d"]
        sc = config["sigma_color"]
        ss = config["sigma_space"]
        beta = config["beta"]

        filtered_scores = np.empty_like(raw_scores_256)
        for i in range(N):
            curr = gf_np[i]
            bg_floor = float(np.percentile(curr, p_floor))
            cleaned = np.maximum(curr - bg_floor, 0.0)
            filtered_scores[i] = apply_bilateral_sharpening_single(
                cleaned, d=d, sigma_color=sc, sigma_space=ss, beta=beta, is_scaled_255=True
            )

    else:
        raise ValueError(f"Unknown config type: {cfg_type}")

    # Compute comprehensive evaluation metrics completely on GPU
    metrics = evaluator.evaluate_all(filtered_scores)
    elapsed = time.time() - t0

    result = {
        "Config_Name": config["name"],
        "Category": config["category"],
        "Type": cfg_type,
        "P-AP": metrics["P-AP"],
        "P-AUPRO": metrics["P-AUPRO"],
        "P-AUROC": metrics["P-AUROC"],
        "P-F1": metrics["P-F1"],
        "R-MissRate": metrics["R-MissRate"],
        "R-FP-RegionCount": int(metrics["R-FP-RegionCount"]),
        "R-PixelCoverage": metrics["R-PixelCoverage"],
        "R-FPR": metrics["R-FPR"],
        "I-AUROC": metrics["I-AUROC"],
        "I-AP": metrics["I-AP"],
        "I-F1": metrics["I-F1"],
        "Elapsed_s": round(elapsed, 2),
    }
    return result


# ============================================================================
# Main CLI & Report Generation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Track 3: Multi-Scale Guided Filter & Bilateral Sharpening Pyramid on 672 Predictions"
    )
    parser.add_argument("--gpu", type=int, default=4, help="GPU device ID (default: 4)")
    parser.add_argument(
        "--root",
        default="/data/wt/two_stages/base_672_15k",
        help="Path to two-stage base_672_15k root directory",
    )
    parser.add_argument(
        "--data_root",
        default="/data/wt/ramdisk/leishi_026/test",
        help="Path to raw test dataset directory",
    )
    parser.add_argument(
        "--ground_truth",
        default="/data/wt/ramdisk/leishi_026/ground_truth",
        help="Path to ground truth directory",
    )
    parser.add_argument(
        "--output_csv",
        default="/data/wt/anomaly-detection/experiments/08_multiscale_guided_bilateral_results.csv",
        help="Output CSV path for quantitative comparison table",
    )
    parser.add_argument(
        "--output_json",
        default="/data/wt/anomaly-detection/experiments/08_multiscale_guided_bilateral_results.json",
        help="Output JSON path for detailed results",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"=== Track 3: Multi-Scale Guided Filter & Bilateral Sharpening ===")
    print(f"Device: {device}")
    print(f"Prediction Root: {args.root}")
    print(f"Data Root: {args.data_root}")
    print(f"Ground Truth: {args.ground_truth}")

    # 1. Load data
    score_maps_256, gt_masks_256, labels, gray_guides_256, rgb_guides_256, raw_records = (
        load_evaluation_records(Path(args.root), Path(args.data_root), Path(args.ground_truth))
    )

    # 2. Initialize Evaluator
    evaluator = FastMetricEvaluator(gt_masks_256, labels, device=device)

    # 3. Build Experiment Matrix
    all_experiments = build_experiment_matrix()
    print(f"\nTotal experimental configurations to evaluate: {len(all_experiments)}")

    # 4. Run Experiments
    results = []
    print("\n" + "=" * 130)
    header = (
        f"{'Category':<24} {'Config Name':<46} {'P-AP':>8} {'P-AUPRO':>8} "
        f"{'P-AUROC':>8} {'P-F1':>8} {'R-Miss%':>8} {'R-FP-Count':>11} {'I-AUROC':>8} {'Time':>6}"
    )
    print(header)
    print("=" * 130)

    for exp in all_experiments:
        res = run_filter_experiment(
            config=exp,
            raw_scores_256=score_maps_256,
            gray_guides_256=gray_guides_256,
            rgb_guides_256=rgb_guides_256,
            evaluator=evaluator,
            device=device,
        )
        results.append(res)

        row_str = (
            f"{res['Category']:<24} {res['Config_Name']:<46} "
            f"{res['P-AP']:>8.4f} {res['P-AUPRO']:>8.4f} {res['P-AUROC']:>8.4f} {res['P-F1']:>8.4f} "
            f"{res['R-MissRate']*100:>7.2f}% {res['R-FP-RegionCount']:>11d} "
            f"{res['I-AUROC']:>8.4f} {res['Elapsed_s']:>5.1f}s"
        )
        print(row_str, flush=True)

    print("=" * 130)

    # 5. Save Results to CSV & JSON
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    out_json = Path(args.output_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to CSV: {out_csv}")
    print(f"All results saved to JSON: {out_json}")


if __name__ == "__main__":
    main()
