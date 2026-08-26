#!/usr/bin/env python3
"""Experiment 16: GMM & KDE Log-Likelihood Ratio Scoring (Track G).

This experiment implements, systematically evaluates, and benchmarks:
1. Log-Likelihood Ratio Scoring Framework:
   Score' = log P(x | Anomaly) - log P(x | Good)
   - Models probability density functions of Good and Anomaly feature banks.
   - Converts Log-Likelihood Ratio (LLR) to bounded, temperature-scaled score offsets.
2. Density Estimation Paradigms:
   - von Mises-Fisher (vMF) / Cosine Kernel Density Estimation (KDE) directly on 768-D unit hypersphere.
   - Top-K Local vMF KDE with nearest-neighbor manifold truncation.
   - PCA Subspace Dimension Reduction + Gaussian Mixture Models (GMM) with component and covariance sweeps.
   - PCA Subspace + Euclidean Gaussian KDE with multi-scale bandwidths.
   - Adaptive Bandwidth KDE based on k-NN distance.
3. Scoring Mapping & Asymmetric Temperature Calibration:
   - Tanh / Sigmoid temperature scaling: M = tanh(beta * LLR).
   - Asymmetric Good vs Anomaly scaling and deadband thresholding.
   - Hybrid Fusion of Log-Likelihood Ratio and Distance Margin.
   - Hard Anomaly Priority Triggering.
4. Comprehensive 680-image evaluation:
   - I-AUROC, I-AP, I-F1
   - P-AUROC, P-AP, P-F1, P-AUPRO
   - R-MissRate, R-FP-RegionCount, R-PixelCoverage, R-FPR
   - End-to-end integration with morphological opening and background floor subtraction.

Usage:
    python experiments/16_gmm_kde_likelihood_scoring.py --gpu 0
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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

# Setup path imports
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
    safe_aupro,
    pixel_f1_score_and_threshold,
    training_image_score,
)
from dinomaly_two_stage import (
    select_patch_positions,
    linear_score_to_feature,
    l2_normalize,
)
from eval_track3_adaptive_geometry import (
    extract_candidate_regions_track3,
    fast_region_detection_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("track_g_gmm_kde")

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030
DEFAULT_BANDWIDTH = (ANOMALY_THRESHOLD - GOOD_THRESHOLD) / 2.0  # 0.008


# =========================================================================
# 1. Density Estimation & Likelihood Ratio Estimator Classes
# =========================================================================

class LikelihoodRatioScorer:
    """Base class for Good vs Anomaly density estimation and LLR scoring."""

    def compute_llr_and_dists(
        self,
        query_vecs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LLR, good distances, anomaly distances for query vectors.

        Args:
            query_vecs: (M, 768) float32 L2-normalized tensor on device.

        Returns:
            (llr, min_dist_good, min_dist_anomaly)
        """
        raise NotImplementedError


class VMFKernelDensityScorer(LikelihoodRatioScorer):
    """GPU-accelerated von Mises-Fisher (vMF) / Cosine Kernel Density Estimator.

    Computes density on the unit hypersphere:
    P(x | D) proportional to (1/N) * sum_i exp((x^T x_i - 1) / sigma^2)
    log P(x | D) = logsumexp((x^T X^T - 1) / sigma^2) - log(N)
    """

    def __init__(
        self,
        good_vecs: np.ndarray,
        anomaly_vecs: np.ndarray,
        device: torch.device,
        sigma_good: float = 0.15,
        sigma_anomaly: float = 0.15,
        top_k: Optional[int] = None,
        prior_ratio: float = 1.0,
    ):
        self.device = device
        self.sigma_g = sigma_good
        self.sigma_a = sigma_anomaly
        self.top_k = top_k
        self.log_prior = float(np.log(max(prior_ratio, 1e-8)))

        self.good_vecs = torch.tensor(good_vecs, dtype=torch.float32, device=device)
        self.anomaly_vecs = torch.tensor(anomaly_vecs, dtype=torch.float32, device=device)
        self.n_g = self.good_vecs.shape[0]
        self.n_a = self.anomaly_vecs.shape[0]
        self.log_ng = float(np.log(self.n_g))
        self.log_na = float(np.log(self.n_a))

    def compute_llr_and_dists(
        self,
        query_vecs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # query_vecs: (M, 768)
        sim_g = torch.matmul(query_vecs, self.good_vecs.T)  # (M, N_g) in [-1, 1]
        sim_a = torch.matmul(query_vecs, self.anomaly_vecs.T)  # (M, N_a) in [-1, 1]

        # Squared Euclidean distances: ||x - y||^2 = 2 - 2 * sim
        dist_sq_g = 2.0 - 2.0 * sim_g
        dist_sq_a = 2.0 - 2.0 * sim_a
        min_dist_g = torch.min(dist_sq_g, dim=-1).values
        min_dist_a = torch.min(dist_sq_a, dim=-1).values

        # Kernel exponent: - dist_sq / (2 * sigma^2)
        inv_2s2_g = 1.0 / (2.0 * self.sigma_g ** 2)
        inv_2s2_a = 1.0 / (2.0 * self.sigma_a ** 2)

        exp_g = -dist_sq_g * inv_2s2_g
        exp_a = -dist_sq_a * inv_2s2_a

        if self.top_k is not None and self.top_k > 0:
            k_g = min(self.top_k, self.n_g)
            k_a = min(self.top_k, self.n_a)
            top_exp_g, _ = torch.topk(exp_g, k=k_g, dim=-1)
            top_exp_a, _ = torch.topk(exp_a, k=k_a, dim=-1)
            log_p_g = torch.logsumexp(top_exp_g, dim=-1) - np.log(k_g)
            log_p_a = torch.logsumexp(top_exp_a, dim=-1) - np.log(k_a)
        else:
            log_p_g = torch.logsumexp(exp_g, dim=-1) - self.log_ng
            log_p_a = torch.logsumexp(exp_a, dim=-1) - self.log_na

        llr = log_p_a - log_p_g + self.log_prior
        return llr, min_dist_g, min_dist_a


class PCAGaussianMixtureScorer(LikelihoodRatioScorer):
    """PCA Dimension Reduction + Gaussian Mixture Models (GMM) Scorer."""

    def __init__(
        self,
        good_vecs: np.ndarray,
        anomaly_vecs: np.ndarray,
        device: torch.device,
        n_components: int = 4,
        n_pca_components: int = 32,
        covariance_type: str = "diag",
        reg_covar: float = 1e-3,
        prior_ratio: float = 1.0,
    ):
        self.device = device
        self.n_pca = n_pca_components
        self.log_prior = float(np.log(max(prior_ratio, 1e-8)))

        # Fit PCA on combined representation
        all_vecs = np.concatenate([good_vecs, anomaly_vecs], axis=0)
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        self.pca.fit(all_vecs)

        g_pca = self.pca.transform(good_vecs)
        a_pca = self.pca.transform(anomaly_vecs)

        self.gmm_good = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=42,
        ).fit(g_pca)

        self.gmm_anomaly = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=42,
        ).fit(a_pca)

        self.good_vecs_t = torch.tensor(good_vecs, dtype=torch.float32, device=device)
        self.anomaly_vecs_t = torch.tensor(anomaly_vecs, dtype=torch.float32, device=device)

    def compute_llr_and_dists(
        self,
        query_vecs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute nearest neighbor Euclidean distances
        sim_g = torch.matmul(query_vecs, self.good_vecs_t.T)
        sim_a = torch.matmul(query_vecs, self.anomaly_vecs_t.T)
        dist_sq_g = 2.0 - 2.0 * sim_g
        dist_sq_a = 2.0 - 2.0 * sim_a
        min_dist_g = torch.min(dist_sq_g, dim=-1).values
        min_dist_a = torch.min(dist_sq_a, dim=-1).values

        # PCA projection & GMM score samples on CPU
        q_np = query_vecs.detach().cpu().numpy()
        q_pca = self.pca.transform(q_np)

        log_p_g = self.gmm_good.score_samples(q_pca)
        log_p_a = self.gmm_anomaly.score_samples(q_pca)

        llr_np = log_p_a - log_p_g + self.log_prior
        llr = torch.tensor(llr_np, dtype=torch.float32, device=self.device)
        return llr, min_dist_g, min_dist_a


class PCAKernelDensityScorer(LikelihoodRatioScorer):
    """PCA Dimension Reduction + Euclidean Gaussian KDE Scorer."""

    def __init__(
        self,
        good_vecs: np.ndarray,
        anomaly_vecs: np.ndarray,
        device: torch.device,
        bandwidth: float = 0.2,
        n_pca_components: int = 32,
        prior_ratio: float = 1.0,
    ):
        self.device = device
        self.n_pca = n_pca_components
        self.log_prior = float(np.log(max(prior_ratio, 1e-8)))

        all_vecs = np.concatenate([good_vecs, anomaly_vecs], axis=0)
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        self.pca.fit(all_vecs)

        g_pca = self.pca.transform(good_vecs)
        a_pca = self.pca.transform(anomaly_vecs)

        self.kde_good = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(g_pca)
        self.kde_anomaly = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(a_pca)

        self.good_vecs_t = torch.tensor(good_vecs, dtype=torch.float32, device=device)
        self.anomaly_vecs_t = torch.tensor(anomaly_vecs, dtype=torch.float32, device=device)

    def compute_llr_and_dists(
        self,
        query_vecs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sim_g = torch.matmul(query_vecs, self.good_vecs_t.T)
        sim_a = torch.matmul(query_vecs, self.anomaly_vecs_t.T)
        dist_sq_g = 2.0 - 2.0 * sim_g
        dist_sq_a = 2.0 - 2.0 * sim_a
        min_dist_g = torch.min(dist_sq_g, dim=-1).values
        min_dist_a = torch.min(dist_sq_a, dim=-1).values

        q_np = query_vecs.detach().cpu().numpy()
        q_pca = self.pca.transform(q_np)

        log_p_g = self.kde_good.score_samples(q_pca)
        log_p_a = self.kde_anomaly.score_samples(q_pca)

        llr_np = log_p_a - log_p_g + self.log_prior
        llr = torch.tensor(llr_np, dtype=torch.float32, device=self.device)
        return llr, min_dist_g, min_dist_a


# =========================================================================
# 2. Offset Mapping Functions from LLR
# =========================================================================

def map_llr_to_offset(
    llr: float,
    da: float,
    dg: float,
    mapping_mode: str = "tanh",
    beta: float = 0.5,
    beta_g: float = 0.5,
    beta_a: float = 0.5,
    scale_g: float = 1.0,
    scale_a: float = 1.0,
    deadband: float = 0.0,
    bandwidth: float = DEFAULT_BANDWIDTH,
    hard_anomaly_th: Optional[float] = 0.15,
    hybrid_weight_llr: Optional[float] = None,
) -> float:
    """Map Log-Likelihood Ratio into signed score adjustment offset."""
    # 1. Hard Anomaly Trigger Priority
    if hard_anomaly_th is not None and da <= float(hard_anomaly_th):
        return 0.020

    # 2. Hybrid LLR + Distance Margin Fusion
    if hybrid_weight_llr is not None:
        denom = dg + da + 1e-8
        dist_margin = (dg - da) / denom
        llr_margin = float(np.tanh(beta * llr))
        w = float(hybrid_weight_llr)
        combined_margin = w * llr_margin + (1.0 - w) * dist_margin
        return float(np.clip(combined_margin * bandwidth, -bandwidth, bandwidth))

    # 3. Deadband filtering
    if abs(llr) < deadband:
        return 0.0

    # 4. Mapping Functions
    if mapping_mode == "tanh":
        margin = float(np.tanh(beta * llr))
        return float(margin * bandwidth)

    elif mapping_mode == "asymmetric_tanh":
        if llr >= 0:
            effective_llr = llr - deadband
            margin = float(np.tanh(beta_a * effective_llr))
            return float(margin * scale_a * bandwidth)
        else:
            effective_llr = llr + deadband
            margin = float(np.tanh(beta_g * effective_llr))
            return float(margin * scale_g * bandwidth)

    elif mapping_mode == "sigmoid":
        margin = float(2.0 / (1.0 + np.exp(-np.clip(beta * llr, -20.0, 20.0))) - 1.0)
        return float(margin * bandwidth)

    elif mapping_mode == "linear_clip":
        margin = float(np.clip(beta * llr, -1.0, 1.0))
        return float(margin * bandwidth)

    else:
        margin = float(np.tanh(beta * llr))
        return float(margin * bandwidth)


# =========================================================================
# 3. Pre-Extraction & Fast Evaluation Pipeline
# =========================================================================

def preprocess_and_preextract_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pre-extract all candidate ROI features and non-middle base overlays once."""
    target_metric_size = (256, 256)
    preprocessed = []
    total_patches = 0

    for rec in records:
        is_bad = (rec["dataset_label"] != "good")
        raw_score = rec["raw_score"]
        score_map = rec["score_map"]
        feature = rec["feature"]
        base_overlay_256 = (
            cv2.resize(score_map, target_metric_size, interpolation=cv2.INTER_LINEAR)
            if is_bad else None
        )

        item = {
            "image_relative": rec["image_relative"],
            "dataset_label": rec["dataset_label"],
            "is_bad": is_bad,
            "raw_score": raw_score,
            "gt_mask_256": rec["gt_mask_256"],
            "score_map": score_map,
            "base_overlay_256": base_overlay_256,
            "is_middle": (GOOD_THRESHOLD <= raw_score <= ANOMALY_THRESHOLD),
            "rois": [],
        }

        if item["is_middle"]:
            components, _ = extract_candidate_regions_track3(
                score_map=score_map,
                good_threshold=GOOD_THRESHOLD,
                anomaly_threshold=ANOMALY_THRESHOLD,
                min_area_pct=0.0,
            )
            height, width = score_map.shape[:2]
            feature_shape = feature.shape[-2:]
            feature_height, feature_width = feature_shape
            score_feature = linear_score_to_feature(score_map, feature_shape)

            for comp in components:
                x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
                local_mask = comp["local_mask"]

                mask_feature = np.zeros(feature_shape, dtype=bool)
                r_start = int(np.clip(np.floor(y * feature_height / height), 0, feature_height - 1))
                r_end = int(np.clip(np.ceil((y + h) * feature_height / height), r_start + 1, feature_height))
                c_start = int(np.clip(np.floor(x * feature_width / width), 0, feature_width - 1))
                c_end = int(np.clip(np.ceil((x + w) * feature_width / width), c_start + 1, feature_width))

                grid_r = (np.arange(r_start, r_end, dtype=np.float64) + 0.5) * float(height) / float(feature_height) - y
                grid_c = (np.arange(c_start, c_end, dtype=np.float64) + 0.5) * float(width) / float(feature_width) - x
                grid_r_idx = np.clip(np.floor(grid_r).astype(np.int64), 0, h - 1)
                grid_c_idx = np.clip(np.floor(grid_c).astype(np.int64), 0, w - 1)

                sub_cells = local_mask[grid_r_idx[:, None], grid_c_idx[None, :]]
                if sub_cells.any():
                    mask_feature[r_start:r_end, c_start:c_end] = sub_cells
                else:
                    cx, cy = comp["centroid"]
                    cr = int(np.clip(np.floor(cy * feature_height / height), 0, feature_height - 1))
                    cc = int(np.clip(np.floor(cx * feature_width / width), 0, feature_width - 1))
                    mask_feature[cr, cc] = True

                positions = select_patch_positions(score_feature, mask_feature, 0.5)
                if positions.shape[0] == 0:
                    continue
                positions = positions[:3]

                p_vecs = [l2_normalize(feature[:, int(r), int(c)]) for r, c in positions]
                total_patches += len(p_vecs)

                local_patch = score_map[y:y+h, x:x+w]
                reg_scores = local_patch[local_mask]
                max_s = float(np.max(reg_scores)) if reg_scores.size else 1.0
                weight = (reg_scores / max_s) if max_s > 1e-8 else np.ones_like(reg_scores)

                item["rois"].append({
                    "x": x, "y": y, "w": w, "h": h,
                    "local_mask": local_mask,
                    "weight": weight,
                    "patch_vectors": np.array(p_vecs, dtype=np.float32),
                })

        preprocessed.append(item)

    LOGGER.info(f"Preprocessed {len(preprocessed)} samples, extracted {total_patches} ROI patch vectors.")
    return preprocessed


def evaluate_llr_configuration_fast(
    dataset: List[Dict[str, Any]],
    scorer: Optional[LikelihoodRatioScorer],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Fast evaluation of one experimental configuration on pre-extracted dataset."""
    t0 = time.time()

    mapping_mode = str(config.get("mapping_mode", "tanh"))
    beta = float(config.get("beta", 0.5))
    beta_g = float(config.get("beta_g", beta))
    beta_a = float(config.get("beta_a", beta))
    scale_g = float(config.get("scale_g", 1.0))
    scale_a = float(config.get("scale_a", 1.0))
    deadband = float(config.get("deadband", 0.0))
    hard_anomaly_th = config.get("hard_anomaly_th", 0.15)
    hybrid_weight_llr = config.get("hybrid_weight_llr", None)

    # Post-processing flags
    k_open = int(config.get("k_open", 0))
    p_floor = float(config.get("p_floor", 0.0))
    min_area = int(config.get("min_area", 0))
    has_postproc = (k_open > 0 or p_floor > 0 or min_area > 0)

    target_metric_size = (256, 256)
    adj_scores = []
    labels = []
    bad_overlays_256 = []
    bad_gt_masks_256 = []
    total_rois = 0
    middle_image_count = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)) if k_open > 0 else None

    for item in dataset:
        is_bad = item["is_bad"]
        labels.append(1 if is_bad else 0)
        raw_score = item["raw_score"]
        score_map = item["score_map"]

        if not item["is_middle"] or scorer is None or len(item["rois"]) == 0:
            if not has_postproc:
                adj_scores.append(raw_score)
                if is_bad:
                    bad_overlays_256.append(item["base_overlay_256"])
                    bad_gt_masks_256.append(item["gt_mask_256"])
            else:
                smap = score_map.copy()
                if k_open > 0:
                    smap = cv2.morphologyEx(smap, cv2.MORPH_OPEN, kernel)
                if p_floor > 0:
                    bg = float(np.percentile(smap, p_floor))
                    smap = np.maximum(smap - bg, 0.0)
                if min_area > 0:
                    bin_mask = smap >= GOOD_THRESHOLD
                    lbl = measure.label(bin_mask)
                    for i in range(1, lbl.max() + 1):
                        if (lbl == i).sum() < min_area:
                            smap[lbl == i] = np.clip(smap[lbl == i], 0.0, GOOD_THRESHOLD * 0.8)
                adj_s = float(training_image_score(smap))
                adj_scores.append(adj_s)
                if is_bad:
                    bad_overlays_256.append(cv2.resize(smap, target_metric_size, interpolation=cv2.INTER_LINEAR))
                    bad_gt_masks_256.append(item["gt_mask_256"])
            continue

        middle_image_count += 1
        overlay = score_map.copy()

        for roi in item["rois"]:
            p_vecs = roi["patch_vectors"]
            p_vecs_t = torch.tensor(p_vecs, dtype=torch.float32, device=device)
            llrs, d_goods, d_anoms = scorer.compute_llr_and_dists(p_vecs_t)
            llrs = llrs.tolist()
            d_goods = d_goods.tolist()
            d_anoms = d_anoms.tolist()

            patch_offsets = []
            for idx in range(len(p_vecs)):
                off = map_llr_to_offset(
                    llr=llrs[idx],
                    da=d_anoms[idx],
                    dg=d_goods[idx],
                    mapping_mode=mapping_mode,
                    beta=beta,
                    beta_g=beta_g,
                    beta_a=beta_a,
                    scale_g=scale_g,
                    scale_a=scale_a,
                    deadband=deadband,
                    bandwidth=DEFAULT_BANDWIDTH,
                    hard_anomaly_th=hard_anomaly_th,
                    hybrid_weight_llr=hybrid_weight_llr,
                )
                patch_offsets.append((off, d_anoms[idx]))

            best_off = max(patch_offsets, key=lambda x: (x[0], -x[1]))[0]
            total_rois += 1

            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            local_mask = roi["local_mask"]
            weight = roi["weight"]
            local_patch = overlay[y:y+h, x:x+w]
            reg_scores = local_patch[local_mask]
            local_patch[local_mask] = np.clip(reg_scores + best_off * weight, 0.0, None)

        if has_postproc:
            if k_open > 0:
                overlay = cv2.morphologyEx(overlay, cv2.MORPH_OPEN, kernel)
            if p_floor > 0:
                bg = float(np.percentile(overlay, p_floor))
                overlay = np.maximum(overlay - bg, 0.0)
            if min_area > 0:
                bin_mask = overlay >= GOOD_THRESHOLD
                lbl = measure.label(bin_mask)
                for i in range(1, lbl.max() + 1):
                    if (lbl == i).sum() < min_area:
                        overlay[lbl == i] = np.clip(overlay[lbl == i], 0.0, GOOD_THRESHOLD * 0.8)

        adj_s = float(training_image_score(overlay))
        adj_scores.append(adj_s)
        if is_bad:
            bad_overlays_256.append(cv2.resize(overlay, target_metric_size, interpolation=cv2.INTER_LINEAR))
            bad_gt_masks_256.append(item["gt_mask_256"])

    elapsed = time.time() - t0

    labels = np.array(labels, dtype=np.uint8)
    adj_scores = np.array(adj_scores, dtype=np.float32)
    i_auroc = safe_auroc(labels, adj_scores)
    i_ap = safe_ap(labels, adj_scores)
    i_f1 = max_f1(labels, adj_scores)

    bad_overlays_256 = np.stack(bad_overlays_256).astype(np.float32)
    bad_gt_masks_256 = np.stack(bad_gt_masks_256).astype(np.uint8)
    pix_labels = bad_gt_masks_256.reshape(-1)
    pix_scores = bad_overlays_256.reshape(-1)

    p_auroc = safe_auroc(pix_labels, pix_scores)
    p_ap = safe_ap(pix_labels, pix_scores)
    p_f1, _ = pixel_f1_score_and_threshold(bad_gt_masks_256, bad_overlays_256)
    p_aupro = safe_aupro(bad_gt_masks_256, bad_overlays_256, show_progress=False)
    reg_metrics = fast_region_detection_metrics(bad_gt_masks_256, bad_overlays_256, GOOD_THRESHOLD)

    return {
        "Config_Name": config.get("name", "Unnamed_Config"),
        "Category": config.get("category", "General"),
        "Method": config.get("method", "Baseline"),
        "Param_Detail": config.get("param_detail", ""),
        "Total_ROIs": total_rois,
        "ROIs_per_Mid_Img": round(total_rois / max(middle_image_count, 1), 1),
        "R-FP-RegionCount": int(reg_metrics["R-FP-RegionCount"]),
        "R-MissRate": reg_metrics["R-MissRate"],
        "R-PixelCoverage": reg_metrics["R-PixelCoverage"],
        "R-FPR": reg_metrics["R-FPR"],
        "P-AUROC": p_auroc,
        "P-AP": p_ap,
        "P-F1": p_f1,
        "P-AUPRO": p_aupro,
        "I-AUROC": i_auroc,
        "I-AP": i_ap,
        "I-F1": i_f1,
        "Elapsed_s": round(elapsed, 2),
    }


# =========================================================================
# 4. Experiment Matrix Construction
# =========================================================================

def build_track_g_experiment_matrix() -> List[Dict[str, Any]]:
    """Build complete matrix of Track G experiments."""
    matrix = []

    # -------------------------------------------------------------
    # Group 0: Baselines
    # -------------------------------------------------------------
    matrix.append({
        "name": "0.0 Baseline (Raw Stage-1 15k Model)",
        "category": "0_Baselines",
        "method": "Raw_Stage1",
        "param_detail": "No 2nd Stage",
        "scorer_type": "none",
    })
    matrix.append({
        "name": "0.1 Standard 2-Stage (Distance Margin)",
        "category": "0_Baselines",
        "method": "Distance_Margin",
        "param_detail": "Margin = (dg-da)/(dg+da)",
        "scorer_type": "vmf_kde",
        "sigma": 0.15,
        "hybrid_weight_llr": 0.0,
        "hard_anomaly_th": 0.15,
    })

    # -------------------------------------------------------------
    # Group 1: Spherical / vMF Kernel Density Estimation (Bandwidth Sweep)
    # -------------------------------------------------------------
    for sigma in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
        matrix.append({
            "name": f"1. vMF KDE (sigma={sigma:.2f}, beta=0.5)",
            "category": "1_vMF_KDE_Bandwidth_Sweep",
            "method": "vMF_KDE",
            "param_detail": f"sigma={sigma:.2f}, beta=0.5, Tanh",
            "scorer_type": "vmf_kde",
            "sigma": sigma,
            "beta": 0.5,
            "mapping_mode": "tanh",
            "hard_anomaly_th": 0.15,
        })

    # -------------------------------------------------------------
    # Group 2: Top-K Local vMF KDE (Local Manifold Density)
    # -------------------------------------------------------------
    for k in [3, 5, 10, 20, 50]:
        matrix.append({
            "name": f"2. Local Top-{k} vMF KDE (sigma=0.15, beta=0.5)",
            "category": "2_Local_TopK_KDE",
            "method": "Local_vMF_KDE",
            "param_detail": f"Top-k={k}, sigma=0.15, beta=0.5",
            "scorer_type": "vmf_kde",
            "sigma": 0.15,
            "top_k": k,
            "beta": 0.5,
            "mapping_mode": "tanh",
            "hard_anomaly_th": 0.15,
        })

    # -------------------------------------------------------------
    # Group 3: PCA Dimension Reduction + Gaussian Mixture Model (GMM)
    # -------------------------------------------------------------
    for n_comp in [1, 2, 4, 8]:
        for cov_type in ["diag", "full"]:
            matrix.append({
                "name": f"3. PCA(32)+GMM({n_comp} comp, {cov_type})",
                "category": "3_PCA_GMM",
                "method": "PCA_GMM",
                "param_detail": f"d=32, K={n_comp}, cov={cov_type}",
                "scorer_type": "pca_gmm",
                "n_pca": 32,
                "n_components": n_comp,
                "covariance_type": cov_type,
                "beta": 0.2,
                "mapping_mode": "tanh",
                "hard_anomaly_th": 0.15,
            })

    for n_pca in [16, 64, 128]:
        matrix.append({
            "name": f"3. PCA({n_pca})+GMM(4 comp, diag)",
            "category": "3_PCA_GMM",
            "method": "PCA_GMM",
            "param_detail": f"d={n_pca}, K=4, cov=diag",
            "scorer_type": "pca_gmm",
            "n_pca": n_pca,
            "n_components": 4,
            "covariance_type": "diag",
            "beta": 0.2,
            "mapping_mode": "tanh",
            "hard_anomaly_th": 0.15,
        })

    # -------------------------------------------------------------
    # Group 4: PCA Dimension Reduction + Gaussian KDE
    # -------------------------------------------------------------
    for bw in [0.1, 0.2, 0.5, 1.0]:
        matrix.append({
            "name": f"4. PCA(32)+KDE(bw={bw:.1f})",
            "category": "4_PCA_KDE",
            "method": "PCA_KDE",
            "param_detail": f"d=32, bw={bw:.1f}, Gaussian",
            "scorer_type": "pca_kde",
            "n_pca": 32,
            "bandwidth": bw,
            "beta": 0.2,
            "mapping_mode": "tanh",
            "hard_anomaly_th": 0.15,
        })

    # -------------------------------------------------------------
    # Group 5: Temperature & Asymmetric Mapping Calibration
    # -------------------------------------------------------------
    for beta_val in [0.1, 0.2, 0.5, 1.0, 2.0]:
        matrix.append({
            "name": f"5. vMF KDE (sigma=0.15, beta={beta_val:.1f})",
            "category": "5_Mapping_Calibration",
            "method": "vMF_KDE",
            "param_detail": f"sigma=0.15, beta={beta_val:.1f}, Tanh",
            "scorer_type": "vmf_kde",
            "sigma": 0.15,
            "beta": beta_val,
            "mapping_mode": "tanh",
            "hard_anomaly_th": 0.15,
        })

    matrix.append({
        "name": "5. Asymmetric vMF KDE (beta_a=1.0, beta_g=0.3, deadband=1.0)",
        "category": "5_Mapping_Calibration",
        "method": "vMF_KDE_Asym",
        "param_detail": "sigma=0.15, beta_a=1.0, beta_g=0.3, deadband=1.0",
        "scorer_type": "vmf_kde",
        "sigma": 0.15,
        "beta_a": 1.0,
        "beta_g": 0.3,
        "scale_a": 1.0,
        "scale_g": 0.8,
        "deadband": 1.0,
        "mapping_mode": "asymmetric_tanh",
        "hard_anomaly_th": 0.15,
    })

    # -------------------------------------------------------------
    # Group 6: Hybrid Likelihood-Distance Fusion
    # -------------------------------------------------------------
    for w_llr in [0.2, 0.5, 0.8]:
        matrix.append({
            "name": f"6. Hybrid Fusion (w_LLR={w_llr:.1f}, w_Dist={1.0-w_llr:.1f})",
            "category": "6_Hybrid_Fusion",
            "method": "Hybrid_LLR_Dist",
            "param_detail": f"w_llr={w_llr:.1f}, sigma=0.15, beta=0.5",
            "scorer_type": "vmf_kde",
            "sigma": 0.15,
            "beta": 0.5,
            "hybrid_weight_llr": w_llr,
            "hard_anomaly_th": 0.15,
        })

    # -------------------------------------------------------------
    # Group 7: Unified High-Precision Pipeline Integration
    # -------------------------------------------------------------
    matrix.append({
        "name": "7.0 vMF KDE + MorphOpening(k=3)",
        "category": "7_Unified_Pipeline",
        "method": "vMF_KDE_Morph",
        "param_detail": "sigma=0.15, beta=0.5, k=3",
        "scorer_type": "vmf_kde",
        "sigma": 0.15,
        "beta": 0.5,
        "hard_anomaly_th": 0.15,
        "k_open": 3,
    })
    matrix.append({
        "name": "7.1 vMF KDE + FloorSub(p=20%)",
        "category": "7_Unified_Pipeline",
        "method": "vMF_KDE_Floor",
        "param_detail": "sigma=0.15, beta=0.5, p=20%",
        "scorer_type": "vmf_kde",
        "sigma": 0.15,
        "beta": 0.5,
        "hard_anomaly_th": 0.15,
        "p_floor": 20.0,
    })
    matrix.append({
        "name": "7.2 vMF KDE + Morph(k=3) + Floor(p=20%)",
        "category": "7_Unified_Pipeline",
        "method": "vMF_KDE_Morph_Floor",
        "param_detail": "sigma=0.15, beta=0.5, k=3, p=20%",
        "scorer_type": "vmf_kde",
        "sigma": 0.15,
        "beta": 0.5,
        "hard_anomaly_th": 0.15,
        "k_open": 3,
        "p_floor": 20.0,
    })
    matrix.append({
        "name": "7.3 Unified Optimal: vMF KDE + Morph(k=3) + Floor(p=20%) + AreaFilter(30)",
        "category": "7_Unified_Pipeline",
        "method": "Unified_Optimal_LLR",
        "param_detail": "sigma=0.15, beta=0.5, k=3, p=20%, area=30",
        "scorer_type": "vmf_kde",
        "sigma": 0.15,
        "beta": 0.5,
        "hard_anomaly_th": 0.15,
        "k_open": 3,
        "p_floor": 20.0,
        "min_area": 30,
    })
    matrix.append({
        "name": "7.4 Unified Optimal: Hybrid(w=0.5) + Morph(k=3) + Floor(p=20%) + AreaFilter(30)",
        "category": "7_Unified_Pipeline",
        "method": "Unified_Optimal_Hybrid",
        "param_detail": "Hybrid w=0.5, k=3, p=20%, area=30",
        "scorer_type": "vmf_kde",
        "sigma": 0.15,
        "beta": 0.5,
        "hybrid_weight_llr": 0.5,
        "hard_anomaly_th": 0.15,
        "k_open": 3,
        "p_floor": 20.0,
        "min_area": 30,
    })

    return matrix


# =========================================================================
# 5. Main Execution & Reporting
# =========================================================================

def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print beautifully formatted ASCII markdown results table."""
    col_w = {
        "Config_Name": 54,
        "I-AUROC": 7,
        "I-AP": 7,
        "P-AUROC": 7,
        "P-AP": 7,
        "P-AUPRO": 7,
        "Miss%": 6,
        "FP Regions": 10,
        "Time(s)": 7,
    }

    header = (
        f"{'Configuration':<{col_w['Config_Name']}} | "
        f"{'I-AUROC':<{col_w['I-AUROC']}} | "
        f"{'I-AP':<{col_w['I-AP']}} | "
        f"{'P-AUROC':<{col_w['P-AUROC']}} | "
        f"{'P-AP':<{col_w['P-AP']}} | "
        f"{'P-AUPRO':<{col_w['P-AUPRO']}} | "
        f"{'Miss%':<{col_w['Miss%']}} | "
        f"{'FP Regions':<{col_w['FP Regions']}} | "
        f"{'Time(s)':<{col_w['Time(s)']}}"
    )
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print("TRACK G: GMM & KDE LOG-LIKELIHOOD RATIO SCORING RESULTS ON 680 TEST SAMPLES")
    print("=" * len(header))
    print(header)
    print(divider)

    current_cat = None
    for r in results:
        cat = r.get("Category", "")
        if cat != current_cat:
            current_cat = cat
            print(f"--- [{current_cat}] " + "-" * (len(header) - len(current_cat) - 8))

        name = r["Config_Name"][:col_w["Config_Name"]]
        i_auroc = f"{r['I-AUROC']:.4f}"
        i_ap = f"{r['I-AP']:.4f}"
        p_auroc = f"{r['P-AUROC']:.4f}"
        p_ap = f"{r['P-AP']:.4f}"
        p_aupro = f"{r['P-AUPRO']:.4f}"
        miss_pct = f"{r['R-MissRate']*100:.2f}%"
        fp_cnt = f"{r['R-FP-RegionCount']}"
        el_s = f"{r['Elapsed_s']:.2f}"

        line = (
            f"{name:<{col_w['Config_Name']}} | "
            f"{i_auroc:<{col_w['I-AUROC']}} | "
            f"{i_ap:<{col_w['I-AP']}} | "
            f"{p_auroc:<{col_w['P-AUROC']}} | "
            f"{p_ap:<{col_w['P-AP']}} | "
            f"{p_aupro:<{col_w['P-AUPRO']}} | "
            f"{miss_pct:<{col_w['Miss%']}} | "
            f"{fp_cnt:<{col_w['FP Regions']}} | "
            f"{el_s:<{col_w['Time(s)']}}"
        )
        print(line)

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Track G: GMM & KDE Likelihood Ratio Scoring")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU device index")
    parser.add_argument("--root_dir", type=str, default="/data/wt/two_stages/base_672_15k", help="Feature bank & preds root")
    parser.add_argument("--output_csv", type=str, default="experiments/16_gmm_kde_likelihood_scoring_results.csv")
    parser.add_argument("--output_json", type=str, default="experiments/16_gmm_kde_likelihood_scoring_results.json")
    parser.add_argument("--quick", action="store_true", help="Run a fast subset for sanity testing")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Track G evaluation on {device}...")

    root = Path(args.root_dir)
    cache_pkl = root / "preds" / "cached_eval_records.pkl"
    if not cache_pkl.is_file():
        raise FileNotFoundError(f"Cache file not found at {cache_pkl}")

    print(f"Loading 680 pre-cached evaluation records from {cache_pkl}...")
    t0 = time.time()
    with open(cache_pkl, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} test images in {time.time() - t0:.2f}s.")

    # Pre-extract ROI data
    print("Pre-processing and pre-extracting candidate ROI features...")
    dataset = preprocess_and_preextract_dataset(records)

    # Load Good and Anomaly feature banks
    p_good = root / "good" / "vectors.npy"
    p_anom = root / "anomaly" / "vectors.npy"
    print(f"Loading feature banks: Good from {p_good}, Anomaly from {p_anom}...")
    good_vecs = np.load(p_good).astype(np.float32)
    anomaly_vecs = np.load(p_anom).astype(np.float32)
    print(f"Loaded Good Bank: {good_vecs.shape}, Anomaly Bank: {anomaly_vecs.shape}.")

    # Build experiment matrix
    configs = build_track_g_experiment_matrix()
    if args.quick:
        configs = configs[:6]
    print(f"Total configurations to evaluate: {len(configs)}")

    # Pre-instantiate scorers
    scorer_cache: Dict[str, LikelihoodRatioScorer] = {}

    def get_scorer(cfg: Dict[str, Any]) -> Optional[LikelihoodRatioScorer]:
        stype = cfg.get("scorer_type", "none")
        if stype == "none":
            return None
        elif stype == "vmf_kde":
            sigma = float(cfg.get("sigma", 0.15))
            top_k = cfg.get("top_k", None)
            key = f"vmf_{sigma}_{top_k}"
            if key not in scorer_cache:
                scorer_cache[key] = VMFKernelDensityScorer(
                    good_vecs=good_vecs,
                    anomaly_vecs=anomaly_vecs,
                    device=device,
                    sigma_good=sigma,
                    sigma_anomaly=sigma,
                    top_k=top_k,
                )
            return scorer_cache[key]
        elif stype == "pca_gmm":
            n_pca = int(cfg.get("n_pca", 32))
            n_comp = int(cfg.get("n_components", 4))
            cov = str(cfg.get("covariance_type", "diag"))
            key = f"pca_gmm_{n_pca}_{n_comp}_{cov}"
            if key not in scorer_cache:
                scorer_cache[key] = PCAGaussianMixtureScorer(
                    good_vecs=good_vecs,
                    anomaly_vecs=anomaly_vecs,
                    device=device,
                    n_components=n_comp,
                    n_pca_components=n_pca,
                    covariance_type=cov,
                )
            return scorer_cache[key]
        elif stype == "pca_kde":
            n_pca = int(cfg.get("n_pca", 32))
            bw = float(cfg.get("bandwidth", 0.2))
            key = f"pca_kde_{n_pca}_{bw}"
            if key not in scorer_cache:
                scorer_cache[key] = PCAKernelDensityScorer(
                    good_vecs=good_vecs,
                    anomaly_vecs=anomaly_vecs,
                    device=device,
                    bandwidth=bw,
                    n_pca_components=n_pca,
                )
            return scorer_cache[key]
        return None

    results = []
    print("\nStarting systematic evaluation over all 680 images...")
    t_start = time.time()
    for idx, cfg in enumerate(configs):
        scorer = get_scorer(cfg)
        res = evaluate_llr_configuration_fast(dataset, scorer, cfg, device)
        results.append(res)
        print(
            f"[{idx+1:02d}/{len(configs):02d}] {res['Config_Name']:<52} | "
            f"I-AUC: {res['I-AUROC']:.4f} | I-AP: {res['I-AP']:.4f} | "
            f"P-AUC: {res['P-AUROC']:.4f} | P-AP: {res['P-AP']:.4f} | "
            f"P-PRO: {res['P-AUPRO']:.4f} | Miss: {res['R-MissRate']*100:.2f}% | "
            f"FP: {res['R-FP-RegionCount']:<6d} | {res['Elapsed_s']:.2f}s",
            flush=True,
        )
    print(f"\nAll {len(configs)} configurations evaluated in {time.time() - t_start:.2f}s.")

    # Print markdown table
    print_results_table(results)

    # Save outputs
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved full JSON results to {out_json}")

    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved CSV results to {out_csv}")


if __name__ == "__main__":
    main()
