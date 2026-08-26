#!/usr/bin/env python3
"""Experiment 14: Multi-Metric Manifold Distance & Cosine Alignment (Track E).

Systematic exploration of multi-metric distance representations and manifold alignment:
1. Metric Group A: L2 Normalized Euclidean Variants (L2 Squared vs L2 Norm / Chord).
2. Metric Group B: Cosine Inner Product & Spherical Geodesic / Angular Distance (Riemannian S^{D-1} Manifold).
3. Metric Group C: L1 Manhattan Distance & Robust Minkowski Lp Metrics (p=0.5, 1.0, 1.5, inf).
4. Metric Group D: Mahalanobis Covariance Scaling & Inverse Variance Regularization (Empirical normal variance, shrinkage).
5. Metric Group E: Manifold Feature Whitening (ZCA/Standardization) + Hypersphere Normalization.
6. Metric Group F: Multi-Metric Ensembling & Manifold Density Alignment (Angular+L1, Angular+Maha, Density-Calibrated Margin).
7. Decision Modes: Pure Margin vs Calibrated Hard Anomaly Trigger vs Asymmetric Confidence Scaling.
8. End-to-End Synergistic Pipeline: Metric Alignment + Morphological Opening + Background Floor Subtraction.

Evaluation on full 680 test images (70 Good + 610 Bad):
- Image Metrics: I-AUROC, I-AP, I-F1
- Pixel Metrics: P-AUROC, P-AP, P-F1, P-AUPRO
- Region Metrics: R-MissRate, R-FP-RegionCount, R-PixelCoverage, R-FPR

Usage:
    python experiments/14_metric_manifold_alignment.py --gpu 0
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
from skimage import measure
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

# Setup Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DINOMALY_DIR = _PROJECT_ROOT / "Dinomaly2"
_UTILS_DIR = _PROJECT_ROOT / "utils"

for p in [_DINOMALY_DIR, _UTILS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from anomaly_evaluation import (
    safe_auroc,
    safe_ap,
    max_f1,
    pixel_f1_score_and_threshold,
)
from dinomaly_two_stage import (
    select_patch_positions,
    linear_score_to_feature,
    l2_normalize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("track_e_manifold_metric")

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030
DEFAULT_BANDWIDTH = (ANOMALY_THRESHOLD - GOOD_THRESHOLD) / 2.0  # 0.008


def fast_training_image_score(arr: np.ndarray) -> float:
    """Fast top-1% image score using np.partition in O(N)."""
    flat = arr.reshape(-1)
    k = max(1, int(flat.size * 0.01))
    return float(np.partition(flat, -k)[-k:].mean())


# =========================================================================
# 1. GPU Distance & Manifold Metric Engine
# =========================================================================

class ManifoldMetricEngine:
    """GPU-accelerated multi-metric distance and manifold alignment engine."""

    def __init__(self, root: Path, device: torch.device):
        self.device = device
        self.root = root

        # Load Good and Anomaly bank vectors
        p_good = root / "good" / "vectors.npy"
        p_anomaly = root / "anomaly" / "vectors.npy"

        raw_good = np.load(p_good)
        raw_anomaly = np.load(p_anomaly)

        self.good_vecs = torch.tensor(raw_good, dtype=torch.float32, device=device)  # (N_g, 768)
        self.anomaly_vecs = torch.tensor(raw_anomaly, dtype=torch.float32, device=device)  # (N_a, 768)

        # Normalize to ensure unit hypersphere S^{D-1}
        self.good_vecs = torch.nn.functional.normalize(self.good_vecs, p=2, dim=-1)
        self.anomaly_vecs = torch.nn.functional.normalize(self.anomaly_vecs, p=2, dim=-1)

        # Compute empirical statistics of the normal feature manifold
        self.mu_good = self.good_vecs.mean(dim=0)  # (768,)
        self.var_good = self.good_vecs.var(dim=0, unbiased=False)  # (768,)
        self.mean_var_good = self.var_good.mean().item()

        # Precompute intra-bank neighbor radii for density calibration
        with torch.no_grad():
            cos_gg = torch.mm(self.good_vecs, self.good_vecs.T)
            cos_gg.fill_diagonal_(-1.0)
            max_cos_g, _ = cos_gg.max(dim=-1)
            self.good_density_radius = torch.sqrt(torch.clamp(2.0 - 2.0 * max_cos_g, min=1e-7)).mean().item()

            cos_aa = torch.mm(self.anomaly_vecs, self.anomaly_vecs.T)
            cos_aa.fill_diagonal_(-1.0)
            max_cos_a, _ = cos_aa.max(dim=-1)
            self.anomaly_density_radius = torch.sqrt(torch.clamp(2.0 - 2.0 * max_cos_a, min=1e-7)).mean().item()

        LOGGER.info(
            "ManifoldMetricEngine initialized on %s: Good Bank=%d, Anomaly Bank=%d. "
            "Normal Var Mean=%.6f, Good Density Radius=%.4f, Anomaly Density Radius=%.4f",
            device, len(self.good_vecs), len(self.anomaly_vecs),
            self.mean_var_good, self.good_density_radius, self.anomaly_density_radius
        )

    def compute_distances(
        self,
        q_vecs: torch.Tensor,
        metric_name: str,
        knn_k: int = 1,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute nearest or top-k mean distances to Good and Anomaly banks."""
        q = torch.nn.functional.normalize(q_vecs, p=2, dim=-1)

        # 1. Cosine & Angular / Spherical Manifold Metrics
        if metric_name == "cosine":
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_g = 1.0 - s_g
            d_a = 1.0 - s_a

        elif metric_name == "l2_squared":
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_g = torch.clamp(2.0 - 2.0 * s_g, min=0.0)
            d_a = torch.clamp(2.0 - 2.0 * s_a, min=0.0)

        elif metric_name == "l2_norm":
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_g = torch.sqrt(torch.clamp(2.0 - 2.0 * s_g, min=0.0))
            d_a = torch.sqrt(torch.clamp(2.0 - 2.0 * s_a, min=0.0))

        elif metric_name == "spherical_angular":
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_g = torch.acos(torch.clamp(s_g, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi
            d_a = torch.acos(torch.clamp(s_a, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi

        elif metric_name == "cayley_hyperbolic":
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_g = (1.0 - s_g) / (1.0 + s_g + 1e-4)
            d_a = (1.0 - s_a) / (1.0 + s_a + 1e-4)

        elif metric_name == "power_cosine":
            gamma = float(kwargs.get("gamma", 1.5))
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_g = torch.clamp(1.0 - s_g, min=0.0) ** gamma
            d_a = torch.clamp(1.0 - s_a, min=0.0) ** gamma

        # 2. L1 Manhattan & Minkowski Lp Metrics
        elif metric_name == "l1_manhattan":
            d_g = torch.cdist(q, self.good_vecs, p=1) / 768.0
            d_a = torch.cdist(q, self.anomaly_vecs, p=1) / 768.0

        elif metric_name == "minkowski_l05":
            chunk_size = 2000
            d_g_list, d_a_list = [], []
            for i in range(0, len(q), chunk_size):
                sub_q = q[i:i+chunk_size]
                diff_g = torch.abs(sub_q.unsqueeze(1) - self.good_vecs.unsqueeze(0))
                sub_dg = (torch.sqrt(diff_g).sum(dim=-1) / 768.0) ** 2
                diff_a = torch.abs(sub_q.unsqueeze(1) - self.anomaly_vecs.unsqueeze(0))
                sub_da = (torch.sqrt(diff_a).sum(dim=-1) / 768.0) ** 2
                d_g_list.append(sub_dg)
                d_a_list.append(sub_da)
            d_g = torch.cat(d_g_list, dim=0)
            d_a = torch.cat(d_a_list, dim=0)

        elif metric_name == "minkowski_l15":
            d_g = (torch.cdist(q, self.good_vecs, p=1.5) ** 1.5 / 768.0) ** (1.0 / 1.5)
            d_a = (torch.cdist(q, self.anomaly_vecs, p=1.5) ** 1.5 / 768.0) ** (1.0 / 1.5)

        elif metric_name == "chebyshev_linf":
            d_g = torch.cdist(q, self.good_vecs, p=float("inf"))
            d_a = torch.cdist(q, self.anomaly_vecs, p=float("inf"))

        # 3. Mahalanobis Covariance Scaling & Inverse Variance Weights
        elif metric_name.startswith("mahalanobis_diag"):
            eps_cov = float(kwargs.get("eps_cov", 1e-3))
            shrinkage = float(kwargs.get("shrinkage", 0.0))

            if shrinkage > 0:
                eff_var = (1.0 - shrinkage) * self.var_good + shrinkage * self.mean_var_good
            else:
                eff_var = self.var_good

            w = 1.0 / (eff_var + eps_cov)
            w = w / w.mean()
            w_sqrt = torch.sqrt(w)

            q_w = q * w_sqrt
            good_w = self.good_vecs * w_sqrt
            anomaly_w = self.anomaly_vecs * w_sqrt

            mode = kwargs.get("maha_mode", "root")
            if mode == "squared":
                d_g = torch.cdist(q_w, good_w, p=2) ** 2 / 768.0
                d_a = torch.cdist(q_w, anomaly_w, p=2) ** 2 / 768.0
            else:
                d_g = torch.cdist(q_w, good_w, p=2) / np.sqrt(768.0)
                d_a = torch.cdist(q_w, anomaly_w, p=2) / np.sqrt(768.0)

        # 4. Manifold Feature Whitening (ZCA / Hyperspherical Normalization)
        elif metric_name.startswith("whitened_"):
            sub_metric = metric_name.replace("whitened_", "")
            eps_cov = float(kwargs.get("eps_cov", 1e-3))
            scale = 1.0 / torch.sqrt(self.var_good + eps_cov)

            q_white = (q - self.mu_good) * scale
            g_white = (self.good_vecs - self.mu_good) * scale
            a_white = (self.anomaly_vecs - self.mu_good) * scale

            q_white = torch.nn.functional.normalize(q_white, p=2, dim=-1)
            g_white = torch.nn.functional.normalize(g_white, p=2, dim=-1)
            a_white = torch.nn.functional.normalize(a_white, p=2, dim=-1)

            if sub_metric == "angular":
                s_g = torch.mm(q_white, g_white.T)
                s_a = torch.mm(q_white, a_white.T)
                d_g = torch.acos(torch.clamp(s_g, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi
                d_a = torch.acos(torch.clamp(s_a, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi
            elif sub_metric == "cosine":
                s_g = torch.mm(q_white, g_white.T)
                s_a = torch.mm(q_white, a_white.T)
                d_g = 1.0 - s_g
                d_a = 1.0 - s_a
            elif sub_metric == "l2_norm":
                s_g = torch.mm(q_white, g_white.T)
                s_a = torch.mm(q_white, a_white.T)
                d_g = torch.sqrt(torch.clamp(2.0 - 2.0 * s_g, min=0.0))
                d_a = torch.sqrt(torch.clamp(2.0 - 2.0 * s_a, min=0.0))
            elif sub_metric == "l1":
                d_g = torch.cdist(q_white, g_white, p=1) / 768.0
                d_a = torch.cdist(q_white, a_white, p=1) / 768.0
            else:
                raise ValueError(f"Unknown whitened sub-metric: {sub_metric}")

        # 5. Multi-Metric Hybrid Ensembles
        elif metric_name == "hybrid_angular_l1":
            beta = float(kwargs.get("beta", 0.3))
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_ang_g = torch.acos(torch.clamp(s_g, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi
            d_ang_a = torch.acos(torch.clamp(s_a, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi

            d_l1_g = torch.cdist(q, self.good_vecs, p=1) / 768.0
            d_l1_a = torch.cdist(q, self.anomaly_vecs, p=1) / 768.0

            d_g = (1.0 - beta) * d_ang_g + beta * (d_l1_g / 0.15)
            d_a = (1.0 - beta) * d_ang_a + beta * (d_l1_a / 0.15)

        elif metric_name == "hybrid_angular_mahalanobis":
            beta = float(kwargs.get("beta", 0.3))
            eps_cov = float(kwargs.get("eps_cov", 1e-3))
            s_g = torch.mm(q, self.good_vecs.T)
            s_a = torch.mm(q, self.anomaly_vecs.T)
            d_ang_g = torch.acos(torch.clamp(s_g, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi
            d_ang_a = torch.acos(torch.clamp(s_a, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi

            w = 1.0 / (self.var_good + eps_cov)
            w = w / w.mean()
            w_sqrt = torch.sqrt(w)
            d_maha_g = torch.cdist(q * w_sqrt, self.good_vecs * w_sqrt, p=2) / np.sqrt(768.0)
            d_maha_a = torch.cdist(q * w_sqrt, self.anomaly_vecs * w_sqrt, p=2) / np.sqrt(768.0)

            d_g = (1.0 - beta) * d_ang_g + beta * d_maha_g
            d_a = (1.0 - beta) * d_ang_a + beta * d_maha_a

        elif metric_name == "hybrid_whitened_l1":
            beta = float(kwargs.get("beta", 0.3))
            eps_cov = float(kwargs.get("eps_cov", 1e-3))
            scale = 1.0 / torch.sqrt(self.var_good + eps_cov)
            q_white = torch.nn.functional.normalize((q - self.mu_good) * scale, p=2, dim=-1)
            g_white = torch.nn.functional.normalize((self.good_vecs - self.mu_good) * scale, p=2, dim=-1)
            a_white = torch.nn.functional.normalize((self.anomaly_vecs - self.mu_good) * scale, p=2, dim=-1)

            s_g = torch.mm(q_white, g_white.T)
            s_a = torch.mm(q_white, a_white.T)
            d_ang_g = torch.acos(torch.clamp(s_g, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi
            d_ang_a = torch.acos(torch.clamp(s_a, -1.0 + 1e-7, 1.0 - 1e-7)) / np.pi

            d_l1_g = torch.cdist(q, self.good_vecs, p=1) / 768.0
            d_l1_a = torch.cdist(q, self.anomaly_vecs, p=1) / 768.0

            d_g = (1.0 - beta) * d_ang_g + beta * (d_l1_g / 0.15)
            d_a = (1.0 - beta) * d_ang_a + beta * (d_l1_a / 0.15)

        else:
            raise ValueError(f"Unknown metric name: {metric_name}")

        # Top-K Nearest Neighbor Aggregation
        if knn_k > 1:
            val_g, _ = torch.topk(d_g, k=knn_k, dim=-1, largest=False)
            val_a, _ = torch.topk(d_a, k=knn_k, dim=-1, largest=False)
            final_dg = val_g.mean(dim=-1)
            final_da = val_a.mean(dim=-1)
        else:
            final_dg, _ = torch.min(d_g, dim=-1)
            final_da, _ = torch.min(d_a, dim=-1)

        # Density Calibration if requested
        if kwargs.get("density_calibrated", False):
            final_dg = final_dg / max(self.good_density_radius, 1e-6)
            final_da = final_da / max(self.anomaly_density_radius, 1e-6)

        return final_dg, final_da


# =========================================================================
# 2. Fast Precomputation & Dataset Cache
# =========================================================================

class FastBenchmarkDataset:
    """Pre-loads 680 evaluation images and precomputes candidate ROIs & patch vectors."""

    def __init__(self, records: List[Dict[str, Any]], target_size: Tuple[int, int] = (256, 256)):
        self.records = records
        self.target_size = target_size
        self.num_records = len(records)

        self.labels = np.array([1 if r["dataset_label"] != "good" else 0 for r in records], dtype=np.uint8)
        self.static_raw_scores = np.array([r["raw_score"] for r in records], dtype=np.float32)

        self.bad_indices = [i for i, r in enumerate(records) if r["dataset_label"] != "good"]
        self.bad_idx_to_pos = {idx: pos for pos, idx in enumerate(self.bad_indices)}

        LOGGER.info("Pre-resizing base score maps for %d bad images...", len(self.bad_indices))
        self.base_bad_overlays_256 = np.stack([
            cv2.resize(records[idx]["score_map"], target_size, interpolation=cv2.INTER_LINEAR)
            for idx in self.bad_indices
        ]).astype(np.float32)

        self.bad_gt_masks_256 = np.stack([
            records[idx]["gt_mask_256"] for idx in self.bad_indices
        ]).astype(np.uint8)

        # Precompute GT PRO metadata for all bad images
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
        LOGGER.info("Precomputed %d GT masks, %d defect regions.", len(self.bad_indices), self.total_regions)

        self.middle_data: List[Dict[str, Any]] = []
        self.middle_bad_positions: List[int] = []
        self.static_bad_positions: List[int] = []
        self.all_patch_vectors: Optional[torch.Tensor] = None

    def precompute_middle_components(
        self,
        device: torch.device,
        min_area_pct: float = 0.0,
        query_patches: int = 3,
    ):
        """Extract and cache candidate ROIs and patch vectors for middle-band images."""
        LOGGER.info("Precomputing candidate ROIs and patch vectors for middle-band images...")
        self.middle_data = []
        collected_vectors = []
        patch_global_idx = 0

        for idx, rec in enumerate(self.records):
            s = rec["raw_score"]
            if not (GOOD_THRESHOLD <= s <= ANOMALY_THRESHOLD):
                continue

            smap = rec["score_map"]
            feat = rec["feature"]
            h, w = smap.shape[:2]
            fh, fw = feat.shape[-2:]

            binary = (smap >= GOOD_THRESHOLD).astype(np.uint8)
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

                patch_indices = []
                for r, c in pos:
                    p_vec = l2_normalize(feat[:, int(r), int(c)])
                    collected_vectors.append(p_vec)
                    patch_indices.append(patch_global_idx)
                    patch_global_idx += 1

                img_comps.append({
                    "x": x, "y": y, "w": cw, "h": ch,
                    "local_mask": local_mask,
                    "patch_indices": patch_indices,
                })

            is_bad = (rec["dataset_label"] != "good")
            bad_pos = self.bad_idx_to_pos.get(idx, -1)

            self.middle_data.append({
                "record_idx": idx,
                "is_bad": is_bad,
                "bad_pos": bad_pos,
                "comps": img_comps,
            })

        middle_bad_set = {m["bad_pos"] for m in self.middle_data if m["is_bad"]}
        self.middle_bad_positions = sorted(list(middle_bad_set))
        self.static_bad_positions = [p for p in range(len(self.bad_indices)) if p not in middle_bad_set]

        self.all_patch_vectors = torch.tensor(np.array(collected_vectors), dtype=torch.float32, device=device)
        LOGGER.info(
            "Precomputed %d total query patch vectors across %d middle-band images (%d bad middle, %d static bad).",
            len(self.all_patch_vectors), len(self.middle_data), len(self.middle_bad_positions), len(self.static_bad_positions)
        )

        # Precompute static PRO values for static bad images
        self.static_pro_sums = np.zeros_like(self.pro_thresholds, dtype=np.float64)
        self.static_fp_counts = np.zeros_like(self.pro_thresholds, dtype=np.int64)

        for pos in self.static_bad_positions:
            amap = self.base_bad_overlays_256[pos]
            inv_mask = self.inverse_masks[pos]
            out_vals = np.sort(amap[inv_mask])
            self.static_fp_counts += out_vals.size - np.searchsorted(out_vals, self.pro_thresholds, side="right")
            lbls, areas, reg_masks = self.pro_gt_meta[pos]
            for rid, area in enumerate(areas, start=1):
                rv = np.sort(amap[reg_masks[rid - 1]])
                hits = rv.size - np.searchsorted(rv, self.pro_thresholds, side="right")
                self.static_pro_sums += hits / area

        # Precompute static region detection statistics for static bad images
        self.static_region_stats = []
        for pos in self.static_bad_positions:
            gt_count, reg_masks = self.gt_region_meta[pos]
            amap = self.base_bad_overlays_256[pos]
            gt_mask = self.bad_gt_masks_256[pos]

            pred_binary = (amap >= GOOD_THRESHOLD).astype(np.uint8)
            cnt, pred_lbls, stats, _ = cv2.connectedComponentsWithStats(pred_binary, 8)
            pred_count = cnt - 1

            detected = 0
            covs = []
            for rmask in reg_masks:
                cov_px = int(pred_binary[rmask].sum())
                if cov_px > 0:
                    detected += 1
                covs.append(cov_px / int(rmask.sum()))

            missed = gt_count - detected
            if pred_count > 0:
                if gt_count > 0:
                    overlap = np.unique(pred_lbls[gt_mask > 0])
                    tp = int(np.count_nonzero(overlap > 0))
                else:
                    tp = 0
                fp = pred_count - tp
            else:
                tp = 0
                fp = 0

            self.static_region_stats.append({
                "fp": fp,
                "fp_rate": fp / pred_count if pred_count > 0 else 0.0,
                "miss_rate": missed / gt_count if gt_count > 0 else 0.0,
                "coverage": float(np.mean(covs)) if covs else 0.0,
            })


# =========================================================================
# 3. Fast Evaluation Routine
# =========================================================================

def evaluate_manifold_configuration(
    dataset: FastBenchmarkDataset,
    engine: ManifoldMetricEngine,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate one multi-metric / manifold configuration across all 680 images."""
    metric_name = config.get("metric_name", "l2_squared")
    knn_k = int(config.get("knn_k", 1))
    decision_mode = config.get("decision_mode", "pure_margin")
    hard_anomaly_th = config.get("hard_anomaly_th", None)
    scale_g = float(config.get("scale_g", 1.0))
    scale_a = float(config.get("scale_a", 1.0))
    tau_g = float(config.get("tau_g", 0.10))
    tau_a = float(config.get("tau_a", 0.05))
    k_open = int(config.get("k_open", 0))
    p_floor = float(config.get("p_floor", 0.0))
    min_area = int(config.get("min_area", 0))

    t0 = time.time()

    # 1. Batch compute distances for all query patches on GPU
    with torch.no_grad():
        all_dg, all_da = engine.compute_distances(
            dataset.all_patch_vectors,
            metric_name=metric_name,
            knn_k=knn_k,
            **config.get("metric_kwargs", {}),
        )
        all_dg_np = all_dg.cpu().numpy()
        all_da_np = all_da.cpu().numpy()

    # 2. Adjust middle images
    adj_scores = dataset.static_raw_scores.copy()
    bad_overlays_256 = dataset.base_bad_overlays_256.copy()

    for m in dataset.middle_data:
        idx = m["record_idx"]
        rec = dataset.records[idx]
        orig_smap = rec["score_map"]
        overlay = orig_smap.copy()

        for comp in m["comps"]:
            x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
            local_mask = comp["local_mask"]
            patch_indices = comp["patch_indices"]

            best_signed_off = -1.0
            min_anomaly_d = float("inf")

            for pidx in patch_indices:
                dg = float(all_dg_np[pidx])
                da = float(all_da_np[pidx])

                # Decision Modes
                if decision_mode == "hard_trigger" and hard_anomaly_th is not None and da <= hard_anomaly_th:
                    signed_off = DEFAULT_BANDWIDTH * scale_a
                elif decision_mode == "asymmetric_tanh":
                    if hard_anomaly_th is not None and da <= hard_anomaly_th:
                        signed_off = DEFAULT_BANDWIDTH * scale_a
                    else:
                        denom = da + dg + 1e-8
                        rel_diff = (da - dg) / denom
                        if rel_diff > 0:
                            conf = float(np.tanh(rel_diff / tau_g))
                            signed_off = - DEFAULT_BANDWIDTH * scale_g * conf
                        else:
                            conf = float(np.tanh((-rel_diff) / tau_a))
                            signed_off = DEFAULT_BANDWIDTH * scale_a * conf
                else:
                    # Pure Margin
                    denom = da + dg + 1e-8
                    rel_diff = (da - dg) / denom
                    if rel_diff > 0:
                        signed_off = - DEFAULT_BANDWIDTH * scale_g * rel_diff
                    else:
                        signed_off = DEFAULT_BANDWIDTH * scale_a * (-rel_diff)

                if signed_off > best_signed_off or (signed_off == best_signed_off and da < min_anomaly_d):
                    best_signed_off = signed_off
                    min_anomaly_d = da

            if best_signed_off == -1.0 and len(patch_indices) == 0:
                best_signed_off = 0.0

            local_patch = overlay[y:y+h, x:x+w]
            reg_scores = local_patch[local_mask]
            max_s = float(np.max(reg_scores)) if reg_scores.size else 1.0
            weight = (reg_scores / max_s) if max_s > 1e-8 else 1.0
            local_patch[local_mask] = np.clip(reg_scores + best_signed_off * weight, 0.0, None)

        # Morphological opening
        if k_open > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
            overlay = cv2.morphologyEx(overlay, cv2.MORPH_OPEN, kernel)

        # Background floor subtraction
        if p_floor > 0.0:
            bg = float(np.percentile(overlay, p_floor))
            overlay = np.maximum(overlay - bg, 0.0)

        # Connected component area filter
        if min_area > 0:
            binary = overlay >= GOOD_THRESHOLD
            lbl = measure.label(binary)
            for cid in range(1, lbl.max() + 1):
                if (lbl == cid).sum() < min_area:
                    overlay[lbl == cid] = np.clip(overlay[lbl == cid], 0.0, GOOD_THRESHOLD * 0.8)

        adj_score = fast_training_image_score(overlay) if overlay.size else rec["raw_score"]
        adj_scores[idx] = adj_score

        if m["is_bad"]:
            bad_pos = m["bad_pos"]
            bad_overlays_256[bad_pos] = cv2.resize(overlay, dataset.target_size, interpolation=cv2.INTER_LINEAR)

    # 3. Calculate Image Metrics
    i_auroc = safe_auroc(dataset.labels, adj_scores)
    i_ap = safe_ap(dataset.labels, adj_scores)
    i_f1 = max_f1(dataset.labels, adj_scores)

    # 4. Calculate Pixel Metrics
    pix_labels = dataset.bad_gt_masks_256.reshape(-1)
    pix_scores = bad_overlays_256.reshape(-1)
    p_auroc = safe_auroc(pix_labels, pix_scores)
    p_ap = safe_ap(pix_labels, pix_scores)
    p_f1, _ = pixel_f1_score_and_threshold(dataset.bad_gt_masks_256, bad_overlays_256)

    # 5. Fast Vectorized PRO Calculation
    pro_sums = dataset.static_pro_sums.copy()
    fp_counts = dataset.static_fp_counts.copy()

    for pos in dataset.middle_bad_positions:
        amap = bad_overlays_256[pos]
        inv_mask = dataset.inverse_masks[pos]
        out_vals = np.sort(amap[inv_mask])
        fp_counts += out_vals.size - np.searchsorted(out_vals, dataset.pro_thresholds, side="right")
        lbls, areas, reg_masks = dataset.pro_gt_meta[pos]
        for rid, area in enumerate(areas, start=1):
            rv = np.sort(amap[reg_masks[rid - 1]])
            hits = rv.size - np.searchsorted(rv, dataset.pro_thresholds, side="right")
            pro_sums += hits / area

    pros = pro_sums / dataset.total_regions
    fprs = fp_counts / dataset.inverse_count
    valid = fprs < 0.3
    if np.any(valid):
        fprs_v = fprs[valid]
        pros_v = pros[valid]
        max_fpr = float(fprs_v.max())
        p_aupro = float(auc(fprs_v / max_fpr, pros_v)) if max_fpr > 0 else 0.0
    else:
        p_aupro = 0.0

    # 6. Fast Region Detection Metrics (Static Bad precomputed + Dynamic Middle Bad)
    image_miss_rates = [s["miss_rate"] for s in dataset.static_region_stats]
    image_fp_rates = [s["fp_rate"] for s in dataset.static_region_stats]
    image_coverages = [s["coverage"] for s in dataset.static_region_stats]
    total_fp_regions = sum(s["fp"] for s in dataset.static_region_stats)
    th = GOOD_THRESHOLD

    for pos in dataset.middle_bad_positions:
        gt_count, reg_masks = dataset.gt_region_meta[pos]
        score_map = bad_overlays_256[pos]
        gt_mask = dataset.bad_gt_masks_256[pos]

        pred_binary = (score_map >= th).astype(np.uint8)
        cnt, pred_lbls, stats, _ = cv2.connectedComponentsWithStats(pred_binary, 8)
        pred_count = cnt - 1

        detected = 0
        covs = []
        for rmask in reg_masks:
            cov_px = int(pred_binary[rmask].sum())
            if cov_px > 0:
                detected += 1
            covs.append(cov_px / int(rmask.sum()))

        missed = gt_count - detected

        if pred_count > 0:
            if gt_count > 0:
                overlap = np.unique(pred_lbls[gt_mask > 0])
                tp = int(np.count_nonzero(overlap > 0))
            else:
                tp = 0
            fp = pred_count - tp
        else:
            tp = 0
            fp = 0

        total_fp_regions += fp
        image_fp_rates.append(fp / pred_count if pred_count > 0 else 0.0)
        if gt_count > 0:
            image_miss_rates.append(missed / gt_count)
            image_coverages.append(float(np.mean(covs)))

    r_miss = float(np.mean(image_miss_rates)) if image_miss_rates else 0.0
    r_fp_count = float(total_fp_regions)
    r_cov = float(np.mean(image_coverages)) if image_coverages else 0.0
    r_fpr = float(np.mean(image_fp_rates)) if image_fp_rates else 0.0

    elapsed = time.time() - t0

    return {
        "I-AUROC": i_auroc,
        "I-AP": i_ap,
        "I-F1": i_f1,
        "P-AUROC": p_auroc,
        "P-AP": p_ap,
        "P-F1": p_f1,
        "P-AUPRO": p_aupro,
        "R-MissRate": r_miss,
        "R-FP-RegionCount": r_fp_count,
        "R-PixelCoverage": r_cov,
        "R-FPR": r_fpr,
        "Elapsed": elapsed,
    }


# =========================================================================
# 4. Main Experiment Runner & Benchmark Suites
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Track E: Multi-Metric Manifold Distance & Cosine Alignment")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (0-7)")
    parser.add_argument("--root", type=str, default="/data/wt/two_stages/base_672_15k", help="Feature bank root")
    parser.add_argument("--output_csv", type=str, default="experiments/14_metric_manifold_alignment_results.csv")
    parser.add_argument("--output_json", type=str, default="experiments/14_metric_manifold_alignment_results.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Running Track E experiments on device: %s", device)

    root_path = Path(args.root)
    cache_pkl = root_path / "preds" / "cached_eval_records.pkl"

    LOGGER.info("Loading cached eval records from %s...", cache_pkl)
    with open(cache_pkl, "rb") as f:
        records = pickle.load(f)

    # Initialize Fast Benchmark Dataset
    dataset = FastBenchmarkDataset(records)
    dataset.precompute_middle_components(device=device, min_area_pct=0.0, query_patches=3)

    # Initialize Manifold Metric Engine
    engine = ManifoldMetricEngine(root_path, device=device)

    all_results = []

    def run_eval(suite_name: str, config_name: str, config: Dict[str, Any]):
        res = evaluate_manifold_configuration(dataset, engine, config)
        res_record = {
            "Suite": suite_name,
            "Config": config_name,
            **res,
        }
        all_results.append(res_record)
        print(
            f"[{suite_name:<18}] {config_name:<46} | "
            f"I-AUROC: {res['I-AUROC']:.4f} | I-AP: {res['I-AP']:.4f} | "
            f"P-AUROC: {res['P-AUROC']:.4f} | P-AP: {res['P-AP']:.4f} | "
            f"P-AUPRO: {res['P-AUPRO']:.4f} | Miss: {res['R-MissRate']*100:.2f}% | "
            f"FP Reg: {res['R-FP-RegionCount']:<6.0f} ({res['Elapsed']:.2f}s)",
            flush=True,
        )

    print("\n" + "=" * 145, flush=True)
    header = f"{'Suite':<20} {'Configuration':<46} | {'I-AUROC':<7} | {'I-AP':<7} | {'P-AUROC':<7} | {'P-AP':<7} | {'P-AUPRO':<7} | {'Miss%':<6} | {'FP Reg':<6}"
    print(header, flush=True)
    print("=" * 145, flush=True)

    # -------------------------------------------------------------
    # Suite 0: Baseline Verification
    # -------------------------------------------------------------
    run_eval("0_Baseline", "00_Raw_Dinomaly_Baseline", {
        "metric_name": "l2_squared",
        "scale_g": 0.0, "scale_a": 0.0,
    })
    run_eval("0_Baseline", "01_Standard_TwoStage_L2Sq_PureMargin", {
        "metric_name": "l2_squared",
        "decision_mode": "pure_margin",
        "scale_g": 1.0, "scale_a": 1.0,
    })
    run_eval("0_Baseline", "02_Standard_TwoStage_L2Sq_HardTrigger_0.15", {
        "metric_name": "l2_squared",
        "decision_mode": "hard_trigger",
        "hard_anomaly_th": 0.15,
        "scale_g": 1.0, "scale_a": 1.0,
    })

    # -------------------------------------------------------------
    # Suite 1: Pure Metric Geometry (Pure Margin, No Heuristic)
    # -------------------------------------------------------------
    metrics_suite1 = [
        ("01_L2_Squared", "l2_squared", {}),
        ("02_L2_Norm_Chord", "l2_norm", {}),
        ("03_Cosine_Distance", "cosine", {}),
        ("04_Spherical_Angular_Geodesic", "spherical_angular", {}),
        ("05_Cayley_Hyperbolic", "cayley_hyperbolic", {}),
        ("06_Power_Cosine_gamma1.5", "power_cosine", {"gamma": 1.5}),
        ("07_Power_Cosine_gamma2.0", "power_cosine", {"gamma": 2.0}),
        ("08_L1_Manhattan", "l1_manhattan", {}),
        ("09_Minkowski_L0.5", "minkowski_l05", {}),
        ("10_Minkowski_L1.5", "minkowski_l15", {}),
        ("11_Chebyshev_Linf", "chebyshev_linf", {}),
    ]
    for cname, mname, mkwargs in metrics_suite1:
        run_eval("1_Metric_Geometry", cname, {
            "metric_name": mname,
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": mkwargs,
        })

    # -------------------------------------------------------------
    # Suite 2: Mahalanobis Covariance Scaling & Regularization
    # -------------------------------------------------------------
    for eps in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        run_eval("2_Mahalanobis", f"Maha_Diag_eps_{eps:g}", {
            "metric_name": "mahalanobis_diag",
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"eps_cov": eps, "maha_mode": "root"},
        })
    run_eval("2_Mahalanobis", "Maha_Diag_eps_1e-3_Squared", {
        "metric_name": "mahalanobis_diag",
        "decision_mode": "pure_margin",
        "scale_g": 1.0, "scale_a": 1.0,
        "metric_kwargs": {"eps_cov": 1e-3, "maha_mode": "squared"},
    })
    for alpha in [0.02, 0.05, 0.10, 0.20, 0.50]:
        run_eval("2_Mahalanobis", f"Maha_Shrinkage_alpha_{alpha:.2f}", {
            "metric_name": "mahalanobis_diag",
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"eps_cov": 1e-3, "shrinkage": alpha, "maha_mode": "root"},
        })

    # -------------------------------------------------------------
    # Suite 3: Manifold Feature Whitening (ZCA/Standardization)
    # -------------------------------------------------------------
    whitened_variants = [
        ("Whitened_Angular_eps_1e-3", "whitened_angular", {"eps_cov": 1e-3}),
        ("Whitened_Angular_eps_5e-3", "whitened_angular", {"eps_cov": 5e-3}),
        ("Whitened_Angular_eps_1e-2", "whitened_angular", {"eps_cov": 1e-2}),
        ("Whitened_Cosine_eps_1e-3", "whitened_cosine", {"eps_cov": 1e-3}),
        ("Whitened_L2_Norm_eps_1e-3", "whitened_l2_norm", {"eps_cov": 1e-3}),
        ("Whitened_L1_eps_1e-3", "whitened_l1", {"eps_cov": 1e-3}),
    ]
    for cname, mname, mkwargs in whitened_variants:
        run_eval("3_Whitening", cname, {
            "metric_name": mname,
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": mkwargs,
        })

    # -------------------------------------------------------------
    # Suite 4: Multi-Metric Ensembles & Manifold Density Alignment
    # -------------------------------------------------------------
    for beta in [0.1, 0.2, 0.3, 0.5, 0.7]:
        run_eval("4_Ensemble_Hybrid", f"Hybrid_Angular_L1_beta_{beta:.1f}", {
            "metric_name": "hybrid_angular_l1",
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"beta": beta},
        })
    for beta in [0.1, 0.2, 0.3, 0.5, 0.7]:
        run_eval("4_Ensemble_Hybrid", f"Hybrid_Angular_Maha_beta_{beta:.1f}", {
            "metric_name": "hybrid_angular_mahalanobis",
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"beta": beta, "eps_cov": 1e-3},
        })
    for beta in [0.1, 0.2, 0.3, 0.5]:
        run_eval("4_Ensemble_Hybrid", f"Hybrid_Whitened_L1_beta_{beta:.1f}", {
            "metric_name": "hybrid_whitened_l1",
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"beta": beta, "eps_cov": 1e-3},
        })
    for mname in ["spherical_angular", "l2_norm", "mahalanobis_diag", "whitened_angular"]:
        run_eval("4_Ensemble_Hybrid", f"Density_Calibrated_{mname}", {
            "metric_name": mname,
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"density_calibrated": True, "eps_cov": 1e-3},
        })

    # -------------------------------------------------------------
    # Suite 5: KNN Aggregation (k in [1, 3, 5])
    # -------------------------------------------------------------
    for k in [1, 3, 5]:
        run_eval("5_KNN_Aggregation", f"Angular_KNN_k{k}", {
            "metric_name": "spherical_angular",
            "knn_k": k,
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
        })
        run_eval("5_KNN_Aggregation", f"Mahalanobis_KNN_k{k}", {
            "metric_name": "mahalanobis_diag",
            "knn_k": k,
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"eps_cov": 1e-3},
        })
        run_eval("5_KNN_Aggregation", f"Whitened_Angular_KNN_k{k}", {
            "metric_name": "whitened_angular",
            "knn_k": k,
            "decision_mode": "pure_margin",
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": {"eps_cov": 1e-3},
        })

    # -------------------------------------------------------------
    # Suite 6: Decision Mode Synergy & Calibrated Hard Anomaly Trigger
    # -------------------------------------------------------------
    hard_th_sweeps = [
        ("Angular_Hard_0.12", "spherical_angular", 0.12, {}),
        ("Angular_Hard_0.14", "spherical_angular", 0.14, {}),
        ("Angular_Hard_0.16", "spherical_angular", 0.16, {}),
        ("Angular_Hard_0.18", "spherical_angular", 0.18, {}),
        ("Whitened_Angular_Hard_0.14", "whitened_angular", 0.14, {"eps_cov": 1e-3}),
        ("Whitened_Angular_Hard_0.16", "whitened_angular", 0.16, {"eps_cov": 1e-3}),
        ("Whitened_Angular_Hard_0.18", "whitened_angular", 0.18, {"eps_cov": 1e-3}),
        ("Maha_Hard_0.15", "mahalanobis_diag", 0.15, {"eps_cov": 1e-3}),
        ("Maha_Hard_0.18", "mahalanobis_diag", 0.18, {"eps_cov": 1e-3}),
        ("Hybrid_Hard_0.14", "hybrid_angular_l1", 0.14, {"beta": 0.2}),
        ("Hybrid_Hard_0.16", "hybrid_angular_l1", 0.16, {"beta": 0.2}),
    ]
    for cname, mname, hth, mkwargs in hard_th_sweeps:
        run_eval("6_Decision_Synergy", cname, {
            "metric_name": mname,
            "knn_k": 3,
            "decision_mode": "hard_trigger",
            "hard_anomaly_th": hth,
            "scale_g": 1.0, "scale_a": 1.0,
            "metric_kwargs": mkwargs,
        })
        run_eval("6_Decision_Synergy", f"{cname}_AsymTanh", {
            "metric_name": mname,
            "knn_k": 3,
            "decision_mode": "asymmetric_tanh",
            "hard_anomaly_th": hth,
            "scale_g": 1.0, "scale_a": 1.0,
            "tau_g": 0.10, "tau_a": 0.05,
            "metric_kwargs": mkwargs,
        })

    # -------------------------------------------------------------
    # Suite 7: Full End-to-End Pipeline (Metric + Morphology + BG Floor + Area Filter)
    # -------------------------------------------------------------
    top_candidates = [
        ("Angular_k3_Hard0.16", "spherical_angular", 0.16, {}),
        ("Whitened_Angular_k3_Hard0.16", "whitened_angular", 0.16, {"eps_cov": 1e-3}),
        ("Hybrid_AngL1_k3_Hard0.16", "hybrid_angular_l1", 0.16, {"beta": 0.2}),
        ("Mahalanobis_k3_Hard0.18", "mahalanobis_diag", 0.18, {"eps_cov": 1e-3}),
    ]

    for cname, mname, hth, mkwargs in top_candidates:
        run_eval("7_End_to_End", f"{cname}_MorphOpen3", {
            "metric_name": mname,
            "knn_k": 3,
            "decision_mode": "hard_trigger",
            "hard_anomaly_th": hth,
            "k_open": 3,
            "metric_kwargs": mkwargs,
        })
        run_eval("7_End_to_End", f"{cname}_Morph3_Floor20", {
            "metric_name": mname,
            "knn_k": 3,
            "decision_mode": "hard_trigger",
            "hard_anomaly_th": hth,
            "k_open": 3,
            "p_floor": 20.0,
            "metric_kwargs": mkwargs,
        })
        run_eval("7_End_to_End", f"{cname}_FullPipeline_Morph3_Floor20_Area30", {
            "metric_name": mname,
            "knn_k": 3,
            "decision_mode": "hard_trigger",
            "hard_anomaly_th": hth,
            "k_open": 3,
            "p_floor": 20.0,
            "min_area": 30,
            "metric_kwargs": mkwargs,
        })

    print("=" * 145, flush=True)

    # Save results to CSV and JSON
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        LOGGER.info("Saved %d experimental results to %s", len(all_results), out_csv)

        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        LOGGER.info("Saved JSON results to %s", out_json)


if __name__ == "__main__":
    main()
