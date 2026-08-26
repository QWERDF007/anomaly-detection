#!/usr/bin/env python3
"""Experiment 11: Dynamic Adaptive Double Thresholding & Continuous Soft-Margin Modulation (Track B).

This script explores:
1. Dynamic Adaptive Double Thresholds [T_good(mu, sigma), T_ano(mu, sigma)]:
   - Image-specific robust background statistics (mu, sigma, trimmed/quantile).
   - Statistical Z-score formulation: T_good = mu + k_g * sigma, T_ano = mu + k_a * sigma.
   - Base-anchored statistical perturbation: T_good = T_g0 + alpha_g * (mu - mu0) + beta_g * (sigma - sigma0).
   - Percentile/quantile adaptive bounds: T_good = P_qg, T_ano = P_qa.
   - Dynamic ambiguity bandwidth scaling: W(mu, sigma) = (T_ano - T_good) / 2.
2. Continuous Soft Modulation Functions:
   - Smooth continuous Tanh, Sigmoid, Algebraic Soft-Sign (p=2, 4), Gaussian exponential gating.
   - Continuous smooth Hard-Anomaly transition avoiding step-function cliffing.
   - Continuous asymmetric temperature scaling and power-law shaping.
3. Continuous Spatial Soft Weighting:
   - Sigmoid spatial gating and Cosine falloff at component boundaries.
   - Elimination of isolated noise spikes and boundary artifacts.
4. Comprehensive 680-Image Full Benchmark:
   - Evaluates I-AUROC, I-AP, P-AUROC, P-AP, P-AUPRO, R-MissRate, R-FP-RegionCount.

Usage:
    python experiments/11_adaptive_threshold_soft_modulation.py --gpu 2
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from sklearn.metrics import auc
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DINOMALY_DIR = _PROJECT_ROOT / "Dinomaly2"
_UTILS_DIR = _PROJECT_ROOT / "utils"

for d in [_DINOMALY_DIR, _UTILS_DIR]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from anomaly_evaluation import (
    safe_auroc,
    safe_ap,
    max_f1,
    training_image_score,
)
from dinomaly_two_stage import (
    load_feature_library,
    select_patch_positions,
    linear_score_to_feature,
    l2_normalize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("trackB_adaptive_soft_modulation")

# Standard reference constants
DEFAULT_GOOD_TH = 0.014
DEFAULT_ANO_TH = 0.030
DEFAULT_BANDWIDTH = (DEFAULT_ANO_TH - DEFAULT_GOOD_TH) / 2.0  # 0.008
REF_MU = 0.007236
REF_SIGMA = 0.003824


# =========================================================================
# 1. Dynamic Adaptive Double Threshold Formulations
# =========================================================================

def compute_dynamic_thresholds(
    smap: np.ndarray,
    method: str = "static",
    k_g: float = 2.0,
    k_a: float = 5.5,
    alpha_g: float = 1.0,
    beta_g: float = 1.0,
    alpha_a: float = 1.0,
    beta_a: float = 1.0,
    q_g: float = 90.0,
    q_a: float = 99.0,
    base_tg: float = DEFAULT_GOOD_TH,
    base_ta: float = DEFAULT_ANO_TH,
    min_tg: float = 0.008,
    max_tg: float = 0.022,
    min_ta: float = 0.020,
    max_ta: float = 0.050,
) -> Tuple[float, float, float]:
    """Compute image-level dynamic double thresholds [T_good, T_ano] and dynamic bandwidth.
    
    Returns:
        (T_good, T_ano, bandwidth)
    """
    if method == "static":
        tg = base_tg
        ta = base_ta
        bw = (ta - tg) / 2.0
        return tg, ta, bw

    # Calculate robust background statistics (excluding extreme outliers)
    p95_val = float(np.percentile(smap, 95.0))
    bg_pixels = smap[smap <= p95_val]
    if bg_pixels.size > 10:
        mu = float(np.mean(bg_pixels))
        sigma = float(np.std(bg_pixels))
    else:
        mu = float(np.mean(smap))
        sigma = float(np.std(smap))

    if method == "z_score":
        # T_good = mu + k_g * sigma, T_ano = mu + k_a * sigma
        tg = np.clip(mu + k_g * sigma, min_tg, max_tg)
        ta = np.clip(mu + k_a * sigma, max(tg + 0.004, min_ta), max_ta)
        bw = (ta - tg) / 2.0
        return float(tg), float(ta), float(bw)

    elif method == "base_perturbation":
        # T_good = base_tg + alpha_g * (mu - mu0) + beta_g * (sigma - sigma0)
        d_mu = mu - REF_MU
        d_sig = sigma - REF_SIGMA
        tg = np.clip(base_tg + alpha_g * d_mu + beta_g * d_sig, min_tg, max_tg)
        ta = np.clip(base_ta + alpha_a * d_mu + beta_a * d_sig, max(tg + 0.004, min_ta), max_ta)
        bw = (ta - tg) / 2.0
        return float(tg), float(ta), float(bw)

    elif method == "quantile":
        # Quantile-based adaptive bounds
        tg = np.clip(float(np.percentile(smap, q_g)), min_tg, max_tg)
        ta = np.clip(float(np.percentile(smap, q_a)), max(tg + 0.004, min_ta), max_ta)
        bw = (ta - tg) / 2.0
        return float(tg), float(ta), float(bw)

    else:
        raise ValueError(f"Unknown dynamic threshold method: {method}")


# =========================================================================
# 2. Continuous Soft Modulation Functions
# =========================================================================

def continuous_soft_margin_modulation(
    dg: float,
    da: float,
    func_type: str = "tanh",
    tau_g: float = 0.20,
    tau_a: float = 0.10,
    scale_g: float = 0.5,
    scale_a: float = 2.0,
    gamma_g: float = 1.0,
    gamma_a: float = 1.0,
    p_alg: float = 2.0,
    bandwidth: float = DEFAULT_BANDWIDTH,
    hard_th: float = 0.15,
    hard_tau: float = 0.02,
    enable_soft_hard_trigger: bool = True,
) -> Tuple[float, float, str]:
    """Calculate signed score adjustment via continuous soft modulation functions.
    
    Args:
        dg: Distance to Good feature bank.
        da: Distance to Anomaly feature bank.
        func_type: Modulation function ('linear', 'tanh', 'sigmoid', 'algebraic', 'gaussian').
        tau_g, tau_a: Good and Anomaly temperature/smoothness parameters.
        scale_g, scale_a: Good and Anomaly offset multipliers.
        gamma_g, gamma_a: Power shaping exponents.
        p_alg: Algebraic function exponent (e.g. 2.0 or 4.0).
        bandwidth: Dynamic or fixed adjustment bandwidth.
        hard_th: Anomaly distance threshold for hard trigger.
        hard_tau: Smooth transition temperature for soft hard-trigger blend.
        enable_soft_hard_trigger: If True, uses smooth Sigmoidal blend instead of step switch.
    
    Returns:
        signed_offset, confidence, decision_type
    """
    # 1. Soft / Continuous Hard Anomaly Blending
    # B_hard = 1 / (1 + exp((da - hard_th) / hard_tau))
    if hard_th > 0:
        if enable_soft_hard_trigger:
            hard_gate = 1.0 / (1.0 + np.exp(np.clip((da - hard_th) / max(hard_tau, 1e-6), -20.0, 20.0)))
        else:
            hard_gate = 1.0 if da <= hard_th else 0.0
    else:
        hard_gate = 0.0

    # 2. Margin computation: M in [-1, 1]
    denom = da + dg + 1e-8
    M = (da - dg) / denom  # M > 0 -> closer to good, M < 0 -> closer to anomaly

    # 3. Continuous Modulation Base Calculation
    if func_type == "linear":
        if M >= 0:
            conf = M ** gamma_g
            off = - bandwidth * scale_g * conf
            dec = "good"
        else:
            conf = (-M) ** gamma_a
            off = bandwidth * scale_a * conf
            dec = "anomaly"

    elif func_type == "tanh":
        if M >= 0:
            raw = np.tanh(M / max(tau_g, 1e-6))
            conf = float(raw ** gamma_g)
            off = - bandwidth * scale_g * conf
            dec = "good"
        else:
            raw = np.tanh((-M) / max(tau_a, 1e-6))
            conf = float(raw ** gamma_a)
            off = bandwidth * scale_a * conf
            dec = "anomaly"

    elif func_type == "sigmoid":
        if M >= 0:
            # Shifted smooth Sigmoid with f(0) = 0, f(1) ~ 1
            raw = (2.0 / (1.0 + np.exp(- M / max(tau_g, 1e-6)))) - 1.0
            conf = float(max(0.0, raw) ** gamma_g)
            off = - bandwidth * scale_g * conf
            dec = "good"
        else:
            raw = (2.0 / (1.0 + np.exp(- (-M) / max(tau_a, 1e-6)))) - 1.0
            conf = float(max(0.0, raw) ** gamma_a)
            off = bandwidth * scale_a * conf
            dec = "anomaly"

    elif func_type == "algebraic":
        # Algebraic soft-sign: x / (1 + x^p)^(1/p)
        if M >= 0:
            x = M / max(tau_g, 1e-6)
            raw = x / ((1.0 + x ** p_alg) ** (1.0 / p_alg))
            conf = float(max(0.0, raw) ** gamma_g)
            off = - bandwidth * scale_g * conf
            dec = "good"
        else:
            x = (-M) / max(tau_a, 1e-6)
            raw = x / ((1.0 + x ** p_alg) ** (1.0 / p_alg))
            conf = float(max(0.0, raw) ** gamma_a)
            off = bandwidth * scale_a * conf
            dec = "anomaly"

    elif func_type == "gaussian":
        # Gaussian exponential soft gating: 1 - exp(-(x/tau)^2)
        if M >= 0:
            x = M / max(tau_g, 1e-6)
            raw = 1.0 - np.exp(- (x ** 2))
            conf = float(max(0.0, raw) ** gamma_g)
            off = - bandwidth * scale_g * conf
            dec = "good"
        else:
            x = (-M) / max(tau_a, 1e-6)
            raw = 1.0 - np.exp(- (x ** 2))
            conf = float(max(0.0, raw) ** gamma_a)
            off = bandwidth * scale_a * conf
            dec = "anomaly"

    else:
        raise ValueError(f"Unknown modulation func_type: {func_type}")

    # 4. Continuous Blend with Hard-Anomaly Gate
    hard_offset = bandwidth * scale_a
    total_offset = (1.0 - hard_gate) * off + hard_gate * hard_offset
    total_conf = (1.0 - hard_gate) * conf + hard_gate * 1.0
    final_dec = "hard_anomaly" if hard_gate > 0.5 else dec

    return float(total_offset), float(total_conf), final_dec


# =========================================================================
# 3. Continuous Spatial Soft Weighting Functions
# =========================================================================

def compute_continuous_spatial_weights(
    patch_scores: np.ndarray,
    t_good: float,
    spatial_mode: str = "linear",
    tau_spatial: float = 0.003,
) -> np.ndarray:
    """Calculate continuous smooth spatial weights within a candidate ROI.
    
    Args:
        patch_scores: 2D array of anomaly scores within ROI.
        t_good: Dynamic good threshold.
        spatial_mode: 'linear', 'sigmoid', 'cosine', 'prominence'.
        tau_spatial: Spatial sigmoid temperature.
    
    Returns:
        2D weight array in [0, 1].
    """
    max_s = float(np.max(patch_scores)) if patch_scores.size > 0 else 1.0
    if max_s <= 1e-8:
        return np.ones_like(patch_scores, dtype=np.float32)

    if spatial_mode == "linear":
        return (patch_scores / max_s).astype(np.float32)

    elif spatial_mode == "sigmoid":
        # Continuous sigmoid activation centered at T_good
        w = 1.0 / (1.0 + np.exp(- (patch_scores - t_good) / max(tau_spatial, 1e-6)))
        return w.astype(np.float32)

    elif spatial_mode == "cosine":
        # Cosine smooth falloff: sin^2(pi/2 * normalized_score)
        norm_s = np.clip((patch_scores - t_good) / max(max_s - t_good, 1e-6), 0.0, 1.0)
        w = np.sin(np.pi * 0.5 * norm_s) ** 2
        return w.astype(np.float32)

    elif spatial_mode == "prominence":
        # Prominence above dynamic threshold
        norm_s = np.maximum(patch_scores - t_good, 0.0) / max(max_s - t_good, 1e-6)
        return norm_s.astype(np.float32)

    else:
        raise ValueError(f"Unknown spatial_mode: {spatial_mode}")


# =========================================================================
# 4. Ultra-Fast Precomputed Dataset & Vectorized PRO Engine
# =========================================================================

class FastTrackBDataset:
    """Precomputed dataset for Track B with fast vectorized metrics."""

    def __init__(self, records: List[Dict[str, Any]], device: torch.device, target_size: Tuple[int, int] = (256, 256)):
        self.records = records
        self.device = device
        self.target_size = target_size
        self.num_records = len(records)

        self.labels = np.array([1 if r["dataset_label"] != "good" else 0 for r in records], dtype=np.uint8)
        self.static_raw_scores = np.array([r["raw_score"] for r in records], dtype=np.float32)

        self.bad_indices = [i for i, r in enumerate(records) if r["dataset_label"] != "good"]
        self.bad_idx_to_pos = {idx: pos for pos, idx in enumerate(self.bad_indices)}

        LOGGER.info("Pre-resizing base score maps for %d bad images to %s...", len(self.bad_indices), target_size)
        self.base_bad_overlays_256 = np.stack([
            cv2.resize(records[idx]["score_map"], target_size, interpolation=cv2.INTER_LINEAR)
            for idx in self.bad_indices
        ]).astype(np.float32)

        self.bad_gt_masks_256 = np.stack([
            records[idx]["gt_mask_256"] for idx in self.bad_indices
        ]).astype(np.uint8)

        self.pix_labels_gpu = torch.from_numpy(self.bad_gt_masks_256.reshape(-1)).to(self.device)

        # Precompute GT PRO metadata
        self.pro_thresholds = np.linspace(0.0, 0.25, 100, dtype=np.float32)
        self.pro_gt_meta = []
        self.inverse_masks = []
        self.total_regions = 0
        total_inv_count = 0

        self.gt_region_meta = []
        for mask in self.bad_gt_masks_256:
            inv = ~mask.astype(bool)
            self.inverse_masks.append(inv)
            total_inv_count += int(inv.sum())

            lbls = measure.label(mask)
            areas = np.bincount(lbls.reshape(-1))[1:].astype(np.float64)
            reg_masks = [lbls == rid for rid in range(1, len(areas) + 1)]
            self.pro_gt_meta.append((lbls, areas, reg_masks))
            self.total_regions += len(areas)

            rc = int(lbls.max())
            self.gt_region_meta.append((rc, reg_masks))

        self.inverse_count = total_inv_count
        LOGGER.info("Precomputed %d GT masks, %d defect regions for fast vector PRO.", len(self.bad_indices), self.total_regions)

        # Precomputed components for all candidate regions across all 680 images
        self.candidate_cache: List[Dict[str, Any]] = []

    def precompute_all_candidate_regions(
        self,
        good_lib: Any,
        anomaly_lib: Any,
        min_thresh: float = 0.008,
        min_area_pct: float = 0.005,
        top_k: int = 3,
        query_patches: int = 3,
    ):
        """Precompute candidate ROIs and patch FAISS distances down to min_thresh=0.008."""
        LOGGER.info("Precomputing all candidate ROIs & FAISS distances across 680 images (min_thresh=%.4f)...", min_thresh)
        t0 = time.time()
        self.candidate_cache = []

        for idx, rec in enumerate(self.records):
            smap = rec["score_map"]
            feat = rec["feature"]
            h, w = smap.shape[:2]
            fh, fw = feat.shape[-2:]

            binary = (smap >= min_thresh).astype(np.uint8)
            min_area = max(1, int(round(min_area_pct / 100.0 * smap.size)))
            cnt, lbls, stats, cents = cv2.connectedComponentsWithStats(binary, 8)

            img_comps = []
            for comp_id in range(1, cnt):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue

                x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                y = int(stats[comp_id, cv2.CC_STAT_TOP])
                cw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                ch = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                local_mask = (lbls[y:y+ch, x:x+cw] == comp_id)
                local_scores = smap[y:y+ch, x:x+cw]
                peak_score = float(local_scores[local_mask].max()) if local_mask.any() else 0.0
                mean_score = float(local_scores[local_mask].mean()) if local_mask.any() else 0.0

                mask_feat = np.zeros((fh, fw), dtype=bool)
                r0 = int(np.clip(np.floor(y * fh / h), 0, fh - 1))
                r1 = int(np.clip(np.ceil((y + ch) * fh / h), r0 + 1, fh))
                c0 = int(np.clip(np.floor(x * fw / w), 0, fw - 1))
                c1 = int(np.clip(np.ceil((x + cw) * fw / w), c0 + 1, fw))

                grid_r = (np.arange(r0, r1, dtype=np.float64) + 0.5) * float(h) / float(fh) - y
                grid_c = (np.arange(c0, c1, dtype=np.float64) + 0.5) * float(w) / float(fw) - x
                grid_r_idx = np.clip(np.floor(grid_r).astype(np.int64), 0, ch - 1)
                grid_c_idx = np.clip(np.floor(grid_c).astype(np.int64), 0, cw - 1)

                sub = local_mask[grid_r_idx[:, None], grid_c_idx[None, :]]
                if sub.any():
                    mask_feat[r0:r1, c0:c1] = sub
                else:
                    cx, cy = cents[comp_id]
                    cr = int(np.clip(np.floor(cy * fh / h), 0, fh - 1))
                    cc = int(np.clip(np.floor(cx * fw / w), 0, fw - 1))
                    mask_feat[cr, cc] = True

                score_feat = linear_score_to_feature(smap, (fh, fw))
                pos = select_patch_positions(score_feat, mask_feat, 0.5)[:query_patches]
                if pos.shape[0] == 0:
                    continue

                patches = []
                for r, c in pos:
                    p_vec = l2_normalize(feat[:, int(r), int(c)])
                    D_g, _ = good_lib.index.search(p_vec[None, :], top_k)
                    D_a, _ = anomaly_lib.index.search(p_vec[None, :], top_k)
                    dg = float(np.mean(np.sqrt(np.maximum(D_g[0], 0.0))))
                    da = float(np.mean(np.sqrt(np.maximum(D_a[0], 0.0))))
                    patches.append({"dg": dg, "da": da})

                img_comps.append({
                    "x": x, "y": y, "w": cw, "h": ch,
                    "local_mask": local_mask,
                    "peak_score": peak_score,
                    "mean_score": mean_score,
                    "patches": patches,
                })

            is_bad = (rec["dataset_label"] != "good")
            bad_pos = self.bad_idx_to_pos.get(idx, -1)

            self.candidate_cache.append({
                "record_idx": idx,
                "is_bad": is_bad,
                "bad_pos": bad_pos,
                "comps": img_comps,
            })

        LOGGER.info("Precomputed candidate cache for all %d images in %.2fs.", len(self.candidate_cache), time.time() - t0)

    def compute_fast_aupro(self, amaps: np.ndarray) -> float:
        """Vectorized PRO computation across all bad images."""
        thresholds = self.pro_thresholds
        pro_sums = np.zeros_like(thresholds, dtype=np.float64)
        fp_counts = np.zeros_like(thresholds, dtype=np.int64)

        for pos in range(len(self.bad_indices)):
            amap = amaps[pos]
            inv_mask = self.inverse_masks[pos]
            out_vals = np.sort(amap[inv_mask])
            fp_counts += out_vals.size - np.searchsorted(out_vals, thresholds, side="right")

            lbls, areas, reg_masks = self.pro_gt_meta[pos]
            for rid_idx, area in enumerate(areas):
                reg_vals = np.sort(amap[reg_masks[rid_idx]])
                hits = reg_vals.size - np.searchsorted(reg_vals, thresholds, side="right")
                pro_sums += hits / area

        pros = pro_sums / max(self.total_regions, 1)
        fprs = fp_counts / max(self.inverse_count, 1)
        valid = fprs < 0.3
        if not np.any(valid):
            return float("nan")
        fprs = fprs[valid]
        pros = pros[valid]
        max_fpr = float(fprs.max())
        if max_fpr <= 0.0:
            return float("nan")
        return float(auc(fprs / max_fpr, pros))


# =========================================================================
# 5. Core Track B Benchmark Evaluator
# =========================================================================

def evaluate_track_b_configuration(
    dataset: FastTrackBDataset,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute complete 680-image evaluation for a given Track B configuration."""
    t0 = time.time()

    # Dynamic double threshold parameters
    dyn_method = config.get("dyn_method", "static")
    k_g = config.get("k_g", 2.0)
    k_a = config.get("k_a", 5.5)
    alpha_g = config.get("alpha_g", 1.0)
    beta_g = config.get("beta_g", 1.0)
    alpha_a = config.get("alpha_a", 1.0)
    beta_a = config.get("beta_a", 1.0)
    q_g = config.get("q_g", 90.0)
    q_a = config.get("q_a", 99.0)

    # Continuous soft modulation parameters
    func_type = config.get("func_type", "tanh")
    tau_g = config.get("tau_g", 0.20)
    tau_a = config.get("tau_a", 0.10)
    scale_g = config.get("scale_g", 0.5)
    scale_a = config.get("scale_a", 2.0)
    gamma_g = config.get("gamma_g", 1.0)
    gamma_a = config.get("gamma_a", 1.0)
    p_alg = config.get("p_alg", 2.0)
    hard_th = config.get("hard_th", 0.15)
    hard_tau = config.get("hard_tau", 0.02)
    soft_hard_blend = config.get("soft_hard_blend", True)

    # Continuous spatial weighting parameters
    spatial_mode = config.get("spatial_mode", "linear")
    tau_spatial = config.get("tau_spatial", 0.003)

    # Post-processing parameters
    bg_floor = config.get("bg_floor", 0.0)
    morph_k = config.get("morph_k", 0)

    adj_scores = np.zeros(dataset.num_records, dtype=np.float32)
    bad_overlays_256 = np.zeros_like(dataset.base_bad_overlays_256)

    # Iterate over all 680 images
    for i, item in enumerate(dataset.candidate_cache):
        rec = dataset.records[i]
        orig_smap = rec["score_map"]
        raw_s = rec["raw_score"]

        # 1. Dynamic Double Threshold Computation
        t_good, t_ano, bandwidth = compute_dynamic_thresholds(
            orig_smap,
            method=dyn_method,
            k_g=k_g, k_a=k_a,
            alpha_g=alpha_g, beta_g=beta_g,
            alpha_a=alpha_a, beta_a=beta_a,
            q_g=q_g, q_a=q_a,
        )

        smap = orig_smap.copy()

        # Check if raw image score falls within the dynamic ambiguity window [T_good, T_ano]
        is_ambiguous = (t_good <= raw_s <= t_ano)

        if is_ambiguous and len(item["comps"]) > 0:
            for comp in item["comps"]:
                # Only process components whose peak score exceeds t_good
                if comp["peak_score"] < t_good:
                    continue

                candidates = []
                for p in comp["patches"]:
                    dg, da = p["dg"], p["da"]
                    off, conf, dec = continuous_soft_margin_modulation(
                        dg=dg, da=da,
                        func_type=func_type,
                        tau_g=tau_g, tau_a=tau_a,
                        scale_g=scale_g, scale_a=scale_a,
                        gamma_g=gamma_g, gamma_a=gamma_a,
                        p_alg=p_alg,
                        bandwidth=bandwidth,
                        hard_th=hard_th,
                        hard_tau=hard_tau,
                        enable_soft_hard_trigger=soft_hard_blend,
                    )
                    candidates.append((off, da))

                if not candidates:
                    continue

                best_off, best_da = max(candidates, key=lambda c: (c[0], -c[1]))
                x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
                local_mask = comp["local_mask"]
                local_patch = smap[y:y+h, x:x+w]
                reg_scores = local_patch[local_mask]

                if reg_scores.size > 0:
                    weights = compute_continuous_spatial_weights(
                        reg_scores,
                        t_good=t_good,
                        spatial_mode=spatial_mode,
                        tau_spatial=tau_spatial,
                    )
                    local_patch[local_mask] = np.clip(reg_scores + best_off * weights, 0.0, None)

        # 2. Post-processing
        if morph_k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
            smap = cv2.morphologyEx(smap, cv2.MORPH_OPEN, kernel)
        if bg_floor > 0.0:
            p_val = float(np.percentile(smap, bg_floor))
            smap = np.maximum(smap - p_val, 0.0)

        # 3. Image score calculation
        adj_scores[i] = float(training_image_score(smap))

        # 4. Save bad overlays for pixel / PRO evaluation
        if rec["dataset_label"] != "good":
            pos = dataset.bad_idx_to_pos[i]
            bad_overlays_256[pos] = cv2.resize(
                smap, dataset.target_size, interpolation=cv2.INTER_LINEAR
            )

    # Vectorized fast PRO evaluation
    p_aupro = dataset.compute_fast_aupro(bad_overlays_256)

    # Fast region metrics evaluation
    eval_threshold = DEFAULT_GOOD_TH  # canonical evaluation threshold (0.014)
    image_miss_rates = []
    image_coverages = []
    total_fp = 0

    for pos, (rc, reg_masks) in enumerate(dataset.gt_region_meta):
        sm = bad_overlays_256[pos]
        pred = (sm >= eval_threshold).astype(np.uint8)
        pred_lbl = measure.label(pred)
        pc = int(pred_lbl.max())

        det = 0
        covs = []
        for rmask in reg_masks:
            covered = int(pred[rmask].sum())
            if covered > 0:
                det += 1
            covs.append(covered / max(int(rmask.sum()), 1))

        if rc > 0:
            image_miss_rates.append((rc - det) / rc)
            image_coverages.append(float(np.mean(covs)))

        if pc > 0:
            gt_m = dataset.bad_gt_masks_256[pos]
            tp = int(np.count_nonzero(np.unique(pred_lbl[gt_m > 0]) > 0)) if rc > 0 else 0
            total_fp += (pc - tp)

    # Classification metrics
    labels = dataset.labels
    i_auroc = safe_auroc(labels, adj_scores)
    i_ap = safe_ap(labels, adj_scores)
    i_f1 = max_f1(labels, adj_scores)

    # GPU-accelerated pixel AUROC and AP
    pix_scores_gpu = torch.from_numpy(bad_overlays_256.reshape(-1)).to(dataset.device)
    p_auroc = float(binary_auroc(pix_scores_gpu, dataset.pix_labels_gpu))
    p_ap = float(binary_average_precision(pix_scores_gpu, dataset.pix_labels_gpu))

    r_miss = float(np.mean(image_miss_rates)) if image_miss_rates else float("nan")
    r_cov = float(np.mean(image_coverages)) if image_coverages else float("nan")
    elapsed = time.time() - t0

    return {
        "Config_Name": config.get("name", "unnamed"),
        "Category": config.get("category", "General"),
        "dyn_method": dyn_method,
        "func_type": func_type,
        "tau_g": tau_g,
        "tau_a": tau_a,
        "scale_g": scale_g,
        "scale_a": scale_a,
        "gamma_g": gamma_g,
        "gamma_a": gamma_a,
        "spatial_mode": spatial_mode,
        "hard_th": hard_th,
        "soft_hard_blend": soft_hard_blend,
        "bg_floor": bg_floor,
        "morph_k": morph_k,
        "I-AUROC": round(i_auroc, 6),
        "I-AP": round(i_ap, 6),
        "I-F1": round(i_f1, 6),
        "P-AUROC": round(p_auroc, 6),
        "P-AP": round(p_ap, 6),
        "P-AUPRO": round(p_aupro, 6),
        "R-MissRate": round(r_miss, 6),
        "R-FP-RegionCount": int(total_fp),
        "R-PixelCoverage": round(r_cov, 6),
        "Elapsed_s": round(elapsed, 2),
    }


# =========================================================================
# 6. Comprehensive Parameter Matrix
# =========================================================================

def build_track_b_experiment_matrix() -> List[Dict[str, Any]]:
    """Construct complete experimental matrix covering all aspects of Track B."""
    experiments = []

    # ---------------------------------------------------------------------
    # Group 0: Standard Baselines
    # ---------------------------------------------------------------------
    experiments.append({
        "name": "00_Raw_SingleStage_Baseline",
        "category": "0_Baseline",
        "dyn_method": "static",
        "func_type": "linear",
        "scale_g": 0.0,
        "scale_a": 0.0,
        "hard_th": 0.0,
    })
    experiments.append({
        "name": "01_Static_Linear_TwoStage_Baseline",
        "category": "0_Baseline",
        "dyn_method": "static",
        "func_type": "linear",
        "scale_g": 1.0,
        "scale_a": 1.0,
        "hard_th": 0.0,
    })
    experiments.append({
        "name": "02_Static_Hard_Trigger_Baseline (da<=0.15)",
        "category": "0_Baseline",
        "dyn_method": "static",
        "func_type": "linear",
        "scale_g": 1.0,
        "scale_a": 1.0,
        "hard_th": 0.15,
        "soft_hard_blend": False,
    })

    # ---------------------------------------------------------------------
    # Group 1: Dynamic Adaptive Double Thresholding Exploration
    # ---------------------------------------------------------------------
    # 1.1 Statistical Z-Score Thresholds: T_good = mu + k_g * sigma, T_ano = mu + k_a * sigma
    for kg in [1.5, 2.0, 2.5, 3.0]:
        for ka in [4.5, 5.5, 6.5]:
            experiments.append({
                "name": f"10_DynThresh_ZScore (kg={kg:.1f}, ka={ka:.1f})",
                "category": "1_Dynamic_Threshold",
                "dyn_method": "z_score",
                "k_g": kg,
                "k_a": ka,
                "func_type": "tanh",
                "tau_g": 0.20,
                "tau_a": 0.10,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "hard_th": 0.15,
                "soft_hard_blend": True,
            })

    # 1.2 Base-Anchored Statistical Perturbation Model
    for ag in [0.5, 1.0, 1.5]:
        for bg in [0.5, 1.0, 1.5]:
            experiments.append({
                "name": f"11_DynThresh_Perturb (alpha={ag:.1f}, beta={bg:.1f})",
                "category": "1_Dynamic_Threshold",
                "dyn_method": "base_perturbation",
                "alpha_g": ag,
                "beta_g": bg,
                "alpha_a": ag,
                "beta_a": bg,
                "func_type": "tanh",
                "tau_g": 0.20,
                "tau_a": 0.10,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "hard_th": 0.15,
                "soft_hard_blend": True,
            })

    # 1.3 Quantile-Based Adaptive Bounds
    for qg in [85.0, 90.0, 95.0]:
        for qa in [98.0, 99.0, 99.5]:
            experiments.append({
                "name": f"12_DynThresh_Quantile (qg={qg:.1f}%, qa={qa:.1f}%)",
                "category": "1_Dynamic_Threshold",
                "dyn_method": "quantile",
                "q_g": qg,
                "q_a": qa,
                "func_type": "tanh",
                "tau_g": 0.20,
                "tau_a": 0.10,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "hard_th": 0.15,
                "soft_hard_blend": True,
            })

    # ---------------------------------------------------------------------
    # Group 2: Continuous Soft Modulation Function Architectures
    # ---------------------------------------------------------------------
    # 2.1 Continuous Hyperbolic Tangent (Tanh) vs Temperature
    for tg in [0.05, 0.10, 0.20, 0.30]:
        for ta in [0.02, 0.05, 0.10, 0.15]:
            experiments.append({
                "name": f"20_SoftMod_Tanh (tau_g={tg:.2f}, tau_a={ta:.2f})",
                "category": "2_Soft_Modulation_Functions",
                "dyn_method": "base_perturbation",
                "func_type": "tanh",
                "tau_g": tg,
                "tau_a": ta,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "hard_th": 0.15,
                "soft_hard_blend": True,
            })

    # 2.2 Continuous Shifted Sigmoid
    for tg in [0.05, 0.10, 0.20]:
        for ta in [0.03, 0.05, 0.10]:
            experiments.append({
                "name": f"21_SoftMod_Sigmoid (tau_g={tg:.2f}, tau_a={ta:.2f})",
                "category": "2_Soft_Modulation_Functions",
                "dyn_method": "base_perturbation",
                "func_type": "sigmoid",
                "tau_g": tg,
                "tau_a": ta,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "hard_th": 0.15,
                "soft_hard_blend": True,
            })

    # 2.3 Continuous Smooth Algebraic Soft-Sign (p=2.0 and p=4.0)
    for p in [2.0, 4.0]:
        for ta in [0.05, 0.10]:
            experiments.append({
                "name": f"22_SoftMod_Algebraic (p={p:.1f}, tau_g=0.20, tau_a={ta:.2f})",
                "category": "2_Soft_Modulation_Functions",
                "dyn_method": "base_perturbation",
                "func_type": "algebraic",
                "p_alg": p,
                "tau_g": 0.20,
                "tau_a": ta,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "hard_th": 0.15,
                "soft_hard_blend": True,
            })

    # 2.4 Continuous Gaussian / Exponential Soft Gating
    for tg in [0.10, 0.20]:
        for ta in [0.05, 0.10]:
            experiments.append({
                "name": f"23_SoftMod_GaussianExp (tau_g={tg:.2f}, tau_a={ta:.2f})",
                "category": "2_Soft_Modulation_Functions",
                "dyn_method": "base_perturbation",
                "func_type": "gaussian",
                "tau_g": tg,
                "tau_a": ta,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "hard_th": 0.15,
                "soft_hard_blend": True,
            })

    # 2.5 Soft vs Hard Anomaly Trigger Comparison
    experiments.append({
        "name": "24_Trigger_Step_HardCutoff (Step Transition at da=0.15)",
        "category": "2_Soft_Modulation_Functions",
        "dyn_method": "base_perturbation",
        "func_type": "tanh",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "soft_hard_blend": False,
    })
    for h_tau in [0.01, 0.02, 0.04, 0.08]:
        experiments.append({
            "name": f"25_Trigger_SmoothSigmoid_Blend (hard_tau={h_tau:.2f})",
            "category": "2_Soft_Modulation_Functions",
            "dyn_method": "base_perturbation",
            "func_type": "tanh",
            "tau_g": 0.20,
            "tau_a": 0.10,
            "scale_g": 0.5,
            "scale_a": 2.0,
            "hard_th": 0.15,
            "hard_tau": h_tau,
            "soft_hard_blend": True,
        })

    # ---------------------------------------------------------------------
    # Group 3: Continuous Spatial Soft Weighting
    # ---------------------------------------------------------------------
    for sp_mode in ["linear", "sigmoid", "cosine", "prominence"]:
        experiments.append({
            "name": f"30_SpatialWeighting_{sp_mode.capitalize()}",
            "category": "3_Spatial_Soft_Weighting",
            "dyn_method": "base_perturbation",
            "func_type": "tanh",
            "tau_g": 0.20,
            "tau_a": 0.10,
            "scale_g": 0.5,
            "scale_a": 2.0,
            "hard_th": 0.15,
            "soft_hard_blend": True,
            "spatial_mode": sp_mode,
            "tau_spatial": 0.003,
        })

    # Spatial sigmoid temperature sweep
    for sp_tau in [0.001, 0.003, 0.006, 0.010]:
        experiments.append({
            "name": f"31_SpatialSigmoid_Tau (tau_sp={sp_tau:.3f})",
            "category": "3_Spatial_Soft_Weighting",
            "dyn_method": "base_perturbation",
            "func_type": "tanh",
            "tau_g": 0.20,
            "tau_a": 0.10,
            "scale_g": 0.5,
            "scale_a": 2.0,
            "hard_th": 0.15,
            "soft_hard_blend": True,
            "spatial_mode": "sigmoid",
            "tau_spatial": sp_tau,
        })

    # ---------------------------------------------------------------------
    # Group 4: Synergistic Integrated Pipelines (Track B SOTA)
    # ---------------------------------------------------------------------
    experiments.append({
        "name": "40_TrackB_Core (DynPerturb + SoftTanh + SmoothHard + CosineSpatial)",
        "category": "4_Synergistic_Pipeline",
        "dyn_method": "base_perturbation",
        "func_type": "tanh",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "soft_hard_blend": True,
        "spatial_mode": "cosine",
    })

    experiments.append({
        "name": "41_TrackB_ZScoreCore (DynZScore + SoftTanh + SmoothHard + CosineSpatial)",
        "category": "4_Synergistic_Pipeline",
        "dyn_method": "z_score",
        "k_g": 2.0,
        "k_a": 5.5,
        "func_type": "tanh",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "soft_hard_blend": True,
        "spatial_mode": "cosine",
    })

    experiments.append({
        "name": "42_TrackB_Plus_MorphOpening (k=3)",
        "category": "4_Synergistic_Pipeline",
        "dyn_method": "base_perturbation",
        "func_type": "tanh",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "soft_hard_blend": True,
        "spatial_mode": "cosine",
        "morph_k": 3,
    })

    experiments.append({
        "name": "43_TrackB_Plus_BGFloor (p=15%)",
        "category": "4_Synergistic_Pipeline",
        "dyn_method": "base_perturbation",
        "func_type": "tanh",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "soft_hard_blend": True,
        "spatial_mode": "cosine",
        "bg_floor": 15.0,
    })

    experiments.append({
        "name": "44_TrackB_SOTA_Full_Pipeline (DynPerturb + SoftTanh + SmoothHard + CosineSpatial + BGFloor15% + Morph3)",
        "category": "4_Synergistic_Pipeline",
        "dyn_method": "base_perturbation",
        "func_type": "tanh",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "soft_hard_blend": True,
        "spatial_mode": "cosine",
        "bg_floor": 15.0,
        "morph_k": 3,
    })

    experiments.append({
        "name": "45_TrackB_UltraClean_LowFP (DynPerturb + SoftTanh + SmoothHard + CosineSpatial + BGFloor20% + Morph3)",
        "category": "4_Synergistic_Pipeline",
        "dyn_method": "base_perturbation",
        "func_type": "tanh",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "scale_g": 0.8,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "soft_hard_blend": True,
        "spatial_mode": "cosine",
        "bg_floor": 20.0,
        "morph_k": 3,
    })

    return experiments


# =========================================================================
# 7. Main Execution Routine
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Track B: Dynamic Adaptive Double Thresholding & Continuous Soft Modulation")
    parser.add_argument("--gpu", type=int, default=2, help="GPU device index (default: 2)")
    parser.add_argument("--root", default="/data/wt/two_stages/base_672_15k", help="Base 672 feature bank root")
    parser.add_argument("--output_csv", default="/data/wt/two_stages/base_672_15k/trackB_adaptive_soft_modulation_results.csv", help="Output CSV path")
    parser.add_argument("--output_json", default="/data/wt/two_stages/base_672_15k/trackB_adaptive_soft_modulation_summary.json", help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("=" * 125, flush=True)
    print(" TRACK B: DYNAMIC ADAPTIVE DOUBLE THRESHOLDING & CONTINUOUS SOFT MODULATION (672 MODEL)", flush=True)
    print("=" * 125, flush=True)
    print(f"Using Compute Device: {device}", flush=True)

    root = Path(args.root)
    good_lib = load_feature_library(root / "good", device, True)
    anomaly_lib = load_feature_library(root / "anomaly", device, True)
    print(f"Loaded Feature Banks: Good={good_lib.index.ntotal} vectors, Anomaly={anomaly_lib.index.ntotal} vectors", flush=True)

    cache_pkl = root / "preds" / "cached_eval_records.pkl"
    t0 = time.time()
    with open(cache_pkl, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} test images in {time.time() - t0:.2f}s", flush=True)

    dataset = FastTrackBDataset(records, device=device, target_size=(256, 256))
    dataset.precompute_all_candidate_regions(
        good_lib, anomaly_lib,
        min_thresh=0.008,
        min_area_pct=0.005,
        top_k=3,
        query_patches=3,
    )

    experiments = build_track_b_experiment_matrix()
    print(f"\nTotal experimental configurations to evaluate: {len(experiments)}", flush=True)

    print("\n" + "=" * 135, flush=True)
    header = (
        f"{'Category':<24} {'Config Name':<46} {'I-AUROC':>8} {'I-AP':>8} "
        f"{'P-AUROC':>8} {'P-AP':>8} {'P-AUPRO':>8} {'Miss%':>7} {'FP-Count':>9} {'Time':>6}"
    )
    print(header, flush=True)
    print("=" * 135, flush=True)

    results: List[Dict[str, Any]] = []
    for exp in experiments:
        res = evaluate_track_b_configuration(dataset, exp)
        results.append(res)
        row_str = (
            f"{res['Category']:<24} {res['Config_Name']:<46} "
            f"{res['I-AUROC']:>8.4f} {res['I-AP']:>8.4f} {res['P-AUROC']:>8.4f} "
            f"{res['P-AP']:>8.4f} {res['P-AUPRO']:>8.4f} {res['R-MissRate']*100:>6.2f}% "
            f"{res['R-FP-RegionCount']:>9d} {res['Elapsed_s']:>5.2f}s"
        )
        print(row_str, flush=True)

    print("=" * 135, flush=True)

    # Save to CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[OK] Quantitative results saved to: {out_csv}", flush=True)

    # Save summary JSON
    out_json = Path(args.output_json)
    summary_data = {
        "title": "Track B: Dynamic Adaptive Double Thresholding & Continuous Soft-Margin Modulation Quantitative Results",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_images": len(records),
        "feature_banks": {
            "good_vectors": int(good_lib.index.ntotal),
            "anomaly_vectors": int(anomaly_lib.index.ntotal),
        },
        "experiments_count": len(results),
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Summary JSON saved to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
