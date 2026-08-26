#!/usr/bin/env python3
"""Track H: Edge Curvature & Gradient Saliency Attenuation on 672 Predictions.

This experiment implements and systematically evaluates:
1. Workpiece Edge & Chamfer Curvature-Adaptive Attenuation:
   - Isophote Curvature kappa_iso = - (Ixx * Iy^2 - 2 * Ixy * Ix * Iy + Iyy * Ix^2) / (Ix^2 + Iy^2)^(3/2)
   - Hessian Determinant / Cornerness kappa_hess = det(H) - 0.04 * trace(H)^2
   - Laplacian of Gaussian / Mean Curvature Delta I = Ixx + Iyy
2. Sobel Gradient Saliency Modulation:
   - Linear, Exponential, Sigmoidal, and Power-Law Attenuation profiles
   - Normalized Gradient Energy Saliency
3. Chamfer Edge-Band Distance Transform & Multi-Radius Proximity Bands (r in [1, 2, 3, 5, 8])
4. Specular Highlight & Reflection Saliency Attenuation:
   - Joint Luminance x Gradient x Curvature Saliency: S_spec = I_norm * G_norm * (1 + beta * kappa_norm)
   - Brightness Contrast Gating (avoids suppressing dark pits / scratches)
5. Defect-Preserving Directional Gradient Coherence Gating (DP-GCG):
   - Coherence C_grad = |<grad I, n_edge>|
   - Distinguishes directional edge reflections (coherent) from transverse defect cracks (incoherent)
6. Connected-Component Chamfer Reflection Blob Attenuation
7. End-to-End Pipeline Synergy:
   - Integration with 2nd-Stage Feature-Bank Hard Trigger (d_ano <= 0.15), Good Suppression, Morphological Opening, and BG Floor Subtraction
8. Full 680-image quantitative benchmark:
   - I-AUROC, I-AP, P-AUROC, P-AP, P-AUPRO, R-MissRate, R-FP-RegionCount, P-F1, I-F1, R-PixelCoverage, R-FPR

Usage:
    python experiments/17_edge_saliency_attenuation.py --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DINOMALY_DIR = _PROJECT_ROOT / "Dinomaly2"
_UTILS_DIR = _PROJECT_ROOT / "utils"

if str(_DINOMALY_DIR) not in sys.path:
    sys.path.insert(0, str(_DINOMALY_DIR))
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from anomaly_evaluation import (
    safe_auroc,
    safe_ap,
    max_f1,
    training_image_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("track_h_edge_saliency")

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030


# ============================================================================
# 1. GPU-Accelerated Gradient & Curvature Operators
# ============================================================================

class EdgeCurvatureGradientEngine:
    """Vectorized GPU engine for Sobel gradients, Curvature, and Edge Saliency."""

    def __init__(self, device: torch.device):
        self.device = device

        # Sobel Filters for 1st derivatives
        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
        ky = torch.tensor([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
        self.kx = kx
        self.ky = ky

    def compute_gradients(self, gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Ix, Iy, and Gradient Magnitude on GPU (B, 1, H, W)."""
        Ix = F.conv2d(gray, self.kx, padding=1)
        Iy = F.conv2d(gray, self.ky, padding=1)
        grad_mag = torch.sqrt(Ix * Ix + Iy * Iy + 1e-12)
        return Ix, Iy, grad_mag

    def compute_isophote_curvature(self, gray: torch.Tensor) -> torch.Tensor:
        """Compute Isophote Curvature kappa_iso on GPU (B, 1, H, W).
        kappa = - (Ixx * Iy^2 - 2 * Ixy * Ix * Iy + Iyy * Ix^2) / (Ix^2 + Iy^2 + eps)^(3/2)
        """
        Ix = F.conv2d(gray, self.kx, padding=1)
        Iy = F.conv2d(gray, self.ky, padding=1)
        Ixx = F.conv2d(Ix, self.kx, padding=1)
        Iyy = F.conv2d(Iy, self.ky, padding=1)
        Ixy = F.conv2d(Ix, self.ky, padding=1)

        num = -(Ixx * (Iy * Iy) - 2.0 * Ixy * Ix * Iy + Iyy * (Ix * Ix))
        den = torch.pow(Ix * Ix + Iy * Iy + 1e-6, 1.5)
        curv = torch.abs(num / den)
        return curv

    def compute_hessian_cornerness(self, gray: torch.Tensor) -> torch.Tensor:
        """Compute Hessian Determinant / Cornerness on GPU (B, 1, H, W).
        det(H) = Ixx * Iyy - Ixy^2
        """
        Ix = F.conv2d(gray, self.kx, padding=1)
        Iy = F.conv2d(gray, self.ky, padding=1)
        Ixx = F.conv2d(Ix, self.kx, padding=1)
        Iyy = F.conv2d(Iy, self.ky, padding=1)
        Ixy = F.conv2d(Ix, self.ky, padding=1)

        det_H = Ixx * Iyy - Ixy * Ixy
        trace_H = Ixx + Iyy
        cornerness = F.relu(det_H - 0.04 * (trace_H * trace_H))
        return cornerness

    def compute_laplacian(self, gray: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian / Mean Curvature on GPU (B, 1, H, W)."""
        Ix = F.conv2d(gray, self.kx, padding=1)
        Iy = F.conv2d(gray, self.ky, padding=1)
        Ixx = F.conv2d(Ix, self.kx, padding=1)
        Iyy = F.conv2d(Iy, self.ky, padding=1)
        lap = torch.abs(Ixx + Iyy)
        return lap


# ============================================================================
# 2. Fast GPU Metric Evaluator
# ============================================================================

def fast_region_detection_metrics(
    masks: np.ndarray,
    score_maps: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute region detection metrics across 610 anomaly test images."""
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

        pix_labels_np = self.bad_gt_masks.reshape(-1)
        self.t_pix_labels = torch.from_numpy(pix_labels_np).to(device=device, dtype=torch.float32)
        self.t_labels = torch.from_numpy(self.labels).to(device=device, dtype=torch.float32)

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

        sorted_scores, order = torch.sort(t_scores_flat, descending=True)
        sorted_labels = self.t_pix_labels[order]

        tp_cumsum = torch.cumsum(sorted_labels, dim=0)
        fp_cumsum = torch.cumsum(1.0 - sorted_labels, dim=0)
        total_tp = tp_cumsum[-1]
        total_fp = fp_cumsum[-1]

        recalls = tp_cumsum / (total_tp + 1e-12)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-12)
        fprs = fp_cumsum / (total_fp + 1e-12)

        # P-AUROC
        fpr_diff = torch.cat([fprs[0:1], fprs[1:] - fprs[:-1]])
        tpr_avg = torch.cat([recalls[0:1] / 2.0, (recalls[1:] + recalls[:-1]) / 2.0])
        p_auroc = (fpr_diff * tpr_avg).sum().item()

        # P-AP
        recall_diff = torch.cat([recalls[0:1], recalls[1:] - recalls[:-1]])
        p_ap = (recall_diff * precisions).sum().item()

        # P-F1
        f1 = 2.0 * precisions * recalls / (precisions + recalls + 1e-7)
        p_f1 = f1.max().item()

        # P-AUPRO on GPU
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

    def evaluate_all(self, all_scores_256: np.ndarray) -> Dict[str, float]:
        """Compute full suite of metrics across all 680 test images."""
        t_all = torch.from_numpy(all_scores_256).to(device=self.device, dtype=torch.float32)
        B, H, W = t_all.shape
        top_k = max(1, int(H * W * 0.01))
        t_top = t_all.reshape(B, -1).topk(top_k, dim=-1).values.mean(dim=-1)
        image_scores = t_top.cpu().numpy()

        i_auroc = safe_auroc(self.labels, image_scores)
        i_ap = safe_ap(self.labels, image_scores)
        i_f1 = max_f1(self.labels, image_scores)

        bad_scores_256 = all_scores_256[self.labels == 1]
        p_auroc, p_ap, p_f1, p_aupro = self.evaluate_pixel_metrics_gpu(bad_scores_256)

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
# 3. Data Loading & Feature-Bank Information
# ============================================================================

def load_data_and_features(
    root: Path,
    data_root: Path,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load cached test records, resized score maps, GT masks, and guidance images."""
    cache_pkl = root / "preds" / "cached_eval_records.pkl"
    LOGGER.info(f"Loading predictions cache from {cache_pkl}...")
    t0 = time.time()
    with open(cache_pkl, "rb") as f:
        raw_records = pickle.load(f)
    LOGGER.info(f"Loaded {len(raw_records)} records in {time.time() - t0:.2f}s.")

    score_maps_256 = []
    gt_masks_256 = []
    labels = []
    gray_images_256 = []
    rgb_images_256 = []

    for rec in tqdm(raw_records, desc="Prepping images", unit="img"):
        is_bad = (rec["dataset_label"] != "good")
        labels.append(1 if is_bad else 0)

        sm = cv2.resize(rec["score_map"], target_size, interpolation=cv2.INTER_LINEAR)
        score_maps_256.append(sm)

        if is_bad:
            gt_masks_256.append(rec["gt_mask_256"])

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

        gray_images_256.append(gray_256)
        rgb_images_256.append(rgb_256)

    score_maps_256 = np.stack(score_maps_256).astype(np.float32)
    gt_masks_256 = np.stack(gt_masks_256).astype(np.uint8)
    labels = np.array(labels, dtype=np.uint8)
    gray_images_256 = np.stack(gray_images_256).astype(np.float32)
    rgb_images_256 = np.stack(rgb_images_256).astype(np.float32)

    return score_maps_256, gt_masks_256, labels, gray_images_256, rgb_images_256, raw_records


# ============================================================================
# 4. Attenuation Model Implementations (Vectorized on GPU)
# ============================================================================

def apply_edge_saliency_pipeline_gpu(
    score_maps_gpu: torch.Tensor,       # (B, 1, H, W)
    gray_gpu: torch.Tensor,             # (B, 1, H, W)
    engine: EdgeCurvatureGradientEngine,
    config: Dict[str, Any],
    two_stage_details: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """Apply edge curvature & gradient saliency attenuation pipeline on GPU."""
    cfg_type = config.get("type", "baseline")
    if cfg_type == "baseline":
        return score_maps_gpu.squeeze(1).cpu().numpy()

    B, C, H, W = score_maps_gpu.shape
    smap = score_maps_gpu.clone()

    # 1. Compute Gradients & Saliency
    Ix, Iy, grad_mag = engine.compute_gradients(gray_gpu)

    # Normalize gradient magnitude per image (robust percentile)
    p95_grad = torch.quantile(grad_mag.view(B, -1), 0.95, dim=-1, keepdim=True).view(B, 1, 1, 1) + 1e-6
    grad_norm = torch.clamp(grad_mag / p95_grad, 0.0, 1.0)

    # Optional multi-scale gradient smoothing
    sigma_g = config.get("grad_smooth_sigma", 0.0)
    if sigma_g > 0.0:
        ks = int(2 * round(3 * sigma_g) + 1)
        pad = ks // 2
        grad_norm = F.avg_pool2d(grad_norm, kernel_size=ks, stride=1, padding=pad)

    # 2. Compute Curvature
    curv_type = config.get("curv_type", "isophote")
    if curv_type == "isophote":
        curv_raw = engine.compute_isophote_curvature(gray_gpu)
    elif curv_type == "hessian":
        curv_raw = engine.compute_hessian_cornerness(gray_gpu)
    elif curv_type == "laplacian":
        curv_raw = engine.compute_laplacian(gray_gpu)
    else:
        curv_raw = torch.zeros_like(grad_norm)

    p95_curv = torch.quantile(curv_raw.view(B, -1), 0.95, dim=-1, keepdim=True).view(B, 1, 1, 1) + 1e-6
    curv_norm = torch.clamp(curv_raw / p95_curv, 0.0, 1.0)

    # 3. Workpiece Edge Band / Distance Masking
    edge_th = config.get("edge_th", 0.25)
    edge_mask = (grad_norm >= edge_th).float()

    # Dilation for chamfer band
    band_r = config.get("edge_band_radius", 0)
    if band_r > 0:
        ks = 2 * band_r + 1
        edge_band = F.max_pool2d(edge_mask, kernel_size=ks, stride=1, padding=band_r)
    else:
        edge_band = edge_mask

    # 4. Attenuation Factor Construction
    mode = config.get("mode", "linear")
    alpha_g = config.get("alpha_g", 0.0)
    alpha_c = config.get("alpha_c", 0.0)
    alpha_s = config.get("alpha_s", 0.0)
    gamma = config.get("gamma", 1.0)
    min_decay = config.get("min_decay", 0.1)

    # 4A. Gradient Attenuation Map A_G
    if mode == "linear":
        A_G = 1.0 - alpha_g * torch.pow(grad_norm, gamma)
    elif mode == "exp":
        A_G = torch.exp(-alpha_g * torch.pow(grad_norm, gamma))
    elif mode == "sigmoid":
        mu_g = config.get("sigmoid_mu", 0.5)
        tau_g = config.get("sigmoid_tau", 0.15)
        A_G = 1.0 - alpha_g * torch.sigmoid((grad_norm - mu_g) / tau_g)
    else:
        A_G = torch.ones_like(grad_norm)

    # 4B. Curvature Attenuation Map A_C
    if alpha_c > 0.0:
        A_C = 1.0 - alpha_c * torch.pow(curv_norm, gamma)
    else:
        A_C = torch.ones_like(curv_norm)

    # 4C. Specular Reflection Saliency A_S (Luminance * Gradient * Curvature)
    if alpha_s > 0.0:
        bright_gate = torch.sigmoid((gray_gpu - 0.60) / 0.10)
        spec_saliency = bright_gate * grad_norm * (1.0 + curv_norm)
        p95_spec = torch.quantile(spec_saliency.view(B, -1), 0.95, dim=-1, keepdim=True).view(B, 1, 1, 1) + 1e-6
        spec_norm = torch.clamp(spec_saliency / p95_spec, 0.0, 1.0)
        A_S = 1.0 - alpha_s * spec_norm
    else:
        A_S = torch.ones_like(grad_norm)

    # 4D. Defect-Preserving Gradient Coherence Gating (DP-GCG)
    use_coherence_gating = config.get("use_coherence_gating", False)
    if use_coherence_gating:
        Ix_smooth = F.avg_pool2d(Ix, kernel_size=5, stride=1, padding=2)
        Iy_smooth = F.avg_pool2d(Iy, kernel_size=5, stride=1, padding=2)
        norm_smooth = torch.sqrt(Ix_smooth * Ix_smooth + Iy_smooth * Iy_smooth + 1e-12)
        nx = Ix_smooth / norm_smooth
        ny = Iy_smooth / norm_smooth

        norm_local = torch.sqrt(Ix * Ix + Iy * Iy + 1e-12)
        gx = Ix / norm_local
        gy = Iy / norm_local

        coherence = torch.abs(gx * nx + gy * ny)
        coh_th = config.get("coh_threshold", 0.6)
        coh_weight = torch.clamp((coherence - coh_th) / (1.0 - coh_th + 1e-6), 0.0, 1.0)
    else:
        coh_weight = torch.ones_like(grad_norm)

    # Combine attenuation factors
    atten_raw = A_G * A_C * A_S
    atten_raw = torch.clamp(atten_raw, min=min_decay, max=1.0)

    # Apply edge band gating and coherence gating
    if config.get("restrict_to_edge_band", True):
        atten_final = 1.0 - edge_band * coh_weight * (1.0 - atten_raw)
    else:
        atten_final = 1.0 - coh_weight * (1.0 - atten_raw)

    atten_final = torch.clamp(atten_final, min=min_decay, max=1.0)

    # Apply Attenuation to Score Map
    smap = smap * atten_final

    # 5. Two-Stage Feature-Bank Refinement
    use_two_stage = config.get("use_two_stage", False)
    if use_two_stage and two_stage_details is not None:
        smap_np = smap.squeeze(1).cpu().numpy()
        hard_t = config.get("hard_ano_th", 0.15)
        supp_ratio = config.get("supp_ratio", 0.1)

        for i, det in enumerate(two_stage_details):
            raw_s = float(det.get("raw_score", 0.0))
            if GOOD_THRESHOLD <= raw_s <= ANOMALY_THRESHOLD:
                cur_sm = smap_np[i]
                for r in det.get("regions", []):
                    da = float(r.get("anomaly_distance", 1.0))
                    dg = float(r.get("good_distance", 1.0))
                    bbox = [int(v * 256 / 672) for v in r.get("bbox_original", [0, 0, 672, 672])]
                    r0, c0, r1, c1 = max(0, bbox[0]), max(0, bbox[1]), min(256, bbox[2]), min(256, bbox[3])
                    sub_s = cur_sm[r0:r1, c0:c1]
                    if sub_s.size == 0:
                        continue
                    max_s = float(np.max(sub_s))
                    if da <= hard_t:
                        w = (sub_s / max_s) if max_s > 1e-8 else 1.0
                        cur_sm[r0:r1, c0:c1] = np.clip(sub_s + 0.008 * w, 0.0, None)
                    elif dg < da:
                        conf = (da - dg) / (da + dg + 1e-8)
                        decay = max(0.0, 1.0 - conf * (1.0 - supp_ratio))
                        cur_sm[r0:r1, c0:c1] = sub_s * decay
                    else:
                        margin = (dg - da) / (da + dg + 1e-8)
                        w = (sub_s / max_s) if max_s > 1e-8 else 1.0
                        cur_sm[r0:r1, c0:c1] = np.clip(sub_s + 0.008 * margin * w, 0.0, None)
                smap_np[i] = cur_sm
        smap = torch.from_numpy(smap_np).unsqueeze(1).to(device=score_maps_gpu.device, dtype=torch.float32)

    # 6. Morphological Opening & Adaptive Noise Floor Subtraction
    k_open = config.get("k_open", 0)
    p_floor = config.get("p_floor", 0.0)

    if k_open > 0 or p_floor > 0.0:
        smap_np = smap.squeeze(1).cpu().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)) if k_open > 0 else None
        for i in range(B):
            m = smap_np[i]
            if kernel is not None:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            if p_floor > 0.0:
                bg = float(np.percentile(m, p_floor))
                m = np.maximum(m - bg, 0.0)
            smap_np[i] = m
        return smap_np

    return smap.squeeze(1).cpu().numpy()


# ============================================================================
# 5. Comprehensive Experiment Matrix Builder
# ============================================================================

def build_experiment_matrix() -> List[Dict[str, Any]]:
    """Construct systematic evaluation configurations across all dimensions of Track H."""
    matrix = []

    # 0. Baseline
    matrix.append({
        "name": "00_Raw_Base_672_Baseline",
        "category": "0_Baseline",
        "type": "baseline",
    })

    # 1. Sobel Gradient Saliency Attenuation (GSA) Sweep
    for mode in ["linear", "exp", "sigmoid"]:
        for alpha in [0.2, 0.4, 0.6, 0.8]:
            matrix.append({
                "name": f"10_GSA_{mode.capitalize()} (alpha={alpha:.1f})",
                "category": "1_Gradient_Saliency",
                "type": "attenuation",
                "mode": mode,
                "alpha_g": alpha,
                "alpha_c": 0.0,
                "alpha_s": 0.0,
                "edge_th": 0.25,
                "edge_band_radius": 2,
                "restrict_to_edge_band": True,
            })

    # Power law decay
    for gamma in [0.5, 1.5, 2.0]:
        for alpha in [0.4, 0.6]:
            matrix.append({
                "name": f"11_GSA_Power (gamma={gamma:.1f}, alpha={alpha:.1f})",
                "category": "1_Gradient_Saliency",
                "type": "attenuation",
                "mode": "linear",
                "gamma": gamma,
                "alpha_g": alpha,
                "alpha_c": 0.0,
                "alpha_s": 0.0,
                "edge_th": 0.25,
                "edge_band_radius": 2,
                "restrict_to_edge_band": True,
            })

    # 2. Curvature-Adaptive Attenuation (CAA) Sweep
    for curv_type in ["isophote", "hessian", "laplacian"]:
        for alpha_c in [0.2, 0.4, 0.6, 0.8]:
            matrix.append({
                "name": f"20_CAA_{curv_type.capitalize()} (alpha_c={alpha_c:.1f})",
                "category": "2_Curvature_Adaptive",
                "type": "attenuation",
                "curv_type": curv_type,
                "alpha_g": 0.0,
                "alpha_c": alpha_c,
                "alpha_s": 0.0,
                "edge_th": 0.25,
                "edge_band_radius": 2,
                "restrict_to_edge_band": True,
            })

    # 3. Chamfer Edge-Band Distance Sweep
    for band_r in [1, 2, 3, 5, 8]:
        for edge_th in [0.15, 0.25, 0.35]:
            matrix.append({
                "name": f"30_EdgeBand (r={band_r}, th={edge_th:.2f})",
                "category": "3_Edge_Band_Geometry",
                "type": "attenuation",
                "alpha_g": 0.5,
                "alpha_c": 0.3,
                "alpha_s": 0.0,
                "edge_th": edge_th,
                "edge_band_radius": band_r,
                "restrict_to_edge_band": True,
            })

    # 4. Specular Highlight & Reflection Saliency (RSA)
    for alpha_s in [0.3, 0.5, 0.7]:
        for alpha_g in [0.3, 0.5]:
            matrix.append({
                "name": f"40_RSA_Reflection (alpha_s={alpha_s:.1f}, alpha_g={alpha_g:.1f})",
                "category": "4_Reflection_Saliency",
                "type": "attenuation",
                "alpha_g": alpha_g,
                "alpha_c": 0.3,
                "alpha_s": alpha_s,
                "edge_th": 0.25,
                "edge_band_radius": 2,
                "restrict_to_edge_band": True,
            })

    # 5. Defect-Preserving Gradient Coherence Gating (DP-GCG)
    for coh_th in [0.4, 0.6, 0.8]:
        for alpha_g in [0.4, 0.6]:
            matrix.append({
                "name": f"50_DP_GCG_Coherence (coh_th={coh_th:.1f}, alpha_g={alpha_g:.1f})",
                "category": "5_Defect_Preserving_Gating",
                "type": "attenuation",
                "alpha_g": alpha_g,
                "alpha_c": 0.4,
                "alpha_s": 0.3,
                "use_coherence_gating": True,
                "coh_threshold": coh_th,
                "edge_th": 0.25,
                "edge_band_radius": 2,
                "restrict_to_edge_band": True,
            })

    # 6. Combined Multi-Factor Hybrid Attenuation
    hybrid_configs = [
        {"name": "60_Hybrid_Balanced", "alpha_g": 0.4, "alpha_c": 0.3, "alpha_s": 0.3, "coh_th": 0.6, "band_r": 2},
        {"name": "61_Hybrid_Curv_Dominant", "alpha_g": 0.3, "alpha_c": 0.6, "alpha_s": 0.2, "coh_th": 0.6, "band_r": 2},
        {"name": "62_Hybrid_Grad_Dominant", "alpha_g": 0.6, "alpha_c": 0.2, "alpha_s": 0.3, "coh_th": 0.6, "band_r": 3},
        {"name": "63_Hybrid_High_Suppression", "alpha_g": 0.7, "alpha_c": 0.5, "alpha_s": 0.5, "coh_th": 0.5, "band_r": 3},
        {"name": "64_Hybrid_Conservative", "alpha_g": 0.3, "alpha_c": 0.2, "alpha_s": 0.2, "coh_th": 0.7, "band_r": 1},
    ]
    for hc in hybrid_configs:
        matrix.append({
            "name": hc["name"],
            "category": "6_Hybrid_MultiFactor",
            "type": "attenuation",
            "alpha_g": hc["alpha_g"],
            "alpha_c": hc["alpha_c"],
            "alpha_s": hc["alpha_s"],
            "use_coherence_gating": True,
            "coh_threshold": hc["coh_th"],
            "edge_th": 0.25,
            "edge_band_radius": hc["band_r"],
            "restrict_to_edge_band": True,
        })

    # 7. End-to-End Pipeline Synergy (+ Two-Stage + Opening + Floor)
    for base_hyb in ["60_Hybrid_Balanced", "61_Hybrid_Curv_Dominant", "63_Hybrid_High_Suppression"]:
        hyb_ref = next(c for c in hybrid_configs if c["name"] == base_hyb)
        for k_op in [0, 3]:
            for p_fl in [0.0, 15.0, 20.0, 25.0]:
                for use_ts in [True, False]:
                    ts_tag = "+TwoStage" if use_ts else "+Direct"
                    matrix.append({
                        "name": f"70_Synergy_{base_hyb}{ts_tag}_k{k_op}_p{int(p_fl)}",
                        "category": "7_Full_Pipeline_Synergy",
                        "type": "attenuation",
                        "alpha_g": hyb_ref["alpha_g"],
                        "alpha_c": hyb_ref["alpha_c"],
                        "alpha_s": hyb_ref["alpha_s"],
                        "use_coherence_gating": True,
                        "coh_threshold": hyb_ref["coh_th"],
                        "edge_th": 0.25,
                        "edge_band_radius": hyb_ref["band_r"],
                        "restrict_to_edge_band": True,
                        "use_two_stage": use_ts,
                        "k_open": k_op,
                        "p_floor": p_fl,
                    })

    return matrix


# ============================================================================
# 6. Main Execution & Benchmark Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Track H: Edge Curvature & Gradient Saliency Attenuation Benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--output-csv", type=str, default="experiments/17_edge_saliency_attenuation_results.csv", help="Path to output CSV")
    parser.add_argument("--output-json", type=str, default="experiments/17_edge_saliency_attenuation_results.json", help="Path to output JSON")
    args = parser.parse_args()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    device = torch.device(device_str)
    LOGGER.info(f"Running Track H Benchmark on device: {device}")

    # Paths
    root_15k = Path("/data/wt/two_stages/base_672_15k")
    data_root = Path("/data/wt/ramdisk/leishi_026/test")

    # Load data
    score_maps_256, gt_masks_256, labels, gray_images_256, rgb_images_256, raw_records = load_data_and_features(
        root=root_15k,
        data_root=data_root,
        target_size=(256, 256),
    )

    LOGGER.info(f"Loaded {len(raw_records)} images: {np.sum(labels == 0)} Normal, {np.sum(labels == 1)} Anomaly.")

    # Initialize GPU Engines
    engine = EdgeCurvatureGradientEngine(device=device)
    evaluator = FastMetricEvaluator(bad_gt_masks_256=gt_masks_256, labels=labels, device=device)

    # Convert test batch to GPU tensors
    score_maps_gpu = torch.from_numpy(score_maps_256).unsqueeze(1).to(device=device, dtype=torch.float32)
    gray_gpu = torch.from_numpy(gray_images_256).unsqueeze(1).to(device=device, dtype=torch.float32)

    # Build experiment matrix
    matrix = build_experiment_matrix()
    LOGGER.info(f"Generated {len(matrix)} experimental configurations.")

    # Extract details for two-stage fusion
    two_stage_details = [rec.get("detail", {}) for rec in raw_records]
    if not two_stage_details or not any(two_stage_details):
        details_dir = root_15k / "preds" / "details"
        two_stage_details = []
        for rec in raw_records:
            d_path = details_dir / Path(rec["image_relative"]).with_suffix(".json")
            if d_path.is_file():
                try:
                    two_stage_details.append(json.loads(d_path.read_text()))
                except Exception:
                    two_stage_details.append({})
            else:
                two_stage_details.append({})

    # Run Benchmark
    results = []
    print("\n" + "=" * 145, flush=True)
    header = (
        f"{'Config_Name':<56} | {'Category':<24} | {'P-AP':<7} | {'P-AUPRO':<7} | "
        f"{'P-AUROC':<7} | {'P-F1':<7} | {'Miss%':<6} | {'FP-Reg':<7} | {'I-AUROC':<7} | {'I-AP':<7} | {'Time':<5}"
    )
    print(header, flush=True)
    print("=" * 145, flush=True)

    for i, cfg in enumerate(matrix):
        t0 = time.time()
        filtered_scores_np = apply_edge_saliency_pipeline_gpu(
            score_maps_gpu=score_maps_gpu,
            gray_gpu=gray_gpu,
            engine=engine,
            config=cfg,
            two_stage_details=two_stage_details,
        )

        metrics = evaluator.evaluate_all(filtered_scores_np)
        elapsed = time.time() - t0

        res = {
            "Config_Name": cfg["name"],
            "Category": cfg["category"],
            "Type": cfg.get("type", "unknown"),
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
        results.append(res)

        row = (
            f"{res['Config_Name']:<56} | {res['Category']:<24} | "
            f"{res['P-AP']:<7.4f} | {res['P-AUPRO']:<7.4f} | {res['P-AUROC']:<7.4f} | {res['P-F1']:<7.4f} | "
            f"{res['R-MissRate']*100:<5.2f}% | {res['R-FP-RegionCount']:<7d} | {res['I-AUROC']:<7.4f} | {res['I-AP']:<7.4f} | {res['Elapsed_s']:<4.2f}s"
        )
        print(row, flush=True)

    print("=" * 145, flush=True)

    # Save to CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    LOGGER.info(f"Saved results to CSV: {out_csv}")

    # Save to JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "title": "Track H: Edge Curvature & Gradient Saliency Attenuation Quantitative Results",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_images": len(raw_records),
        "experiments_count": len(results),
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    LOGGER.info(f"Saved results to JSON: {out_json}")


if __name__ == "__main__":
    main()
