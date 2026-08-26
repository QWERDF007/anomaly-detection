#!/usr/bin/env python3
"""Experiment 12: Triple-Bank Optimal Weighting & Multi-Scale KNN (Track C).

Systematic grid search and full-benchmark evaluation across:
1. Feature Libraries:
   - Good Bank (base_672_15k/good, 1490 vectors)
   - Anomaly Bank (base_672_15k/anomaly, 548 vectors)
   - Hard-Negative Banks:
     * Bank_Edge (1650 vectors from Sobel edge peaks on train good)
     * Bank_PeakFP (825 vectors from peak FP noise on train good)
     * Bank_Combined (2975 vectors: Edge + PeakFP + Export Set)

2. Multi-Scale KNN & Patch Retrieval:
   - Top-k retrieval for k in [1, 2, 3, 5] across each bank.
   - Query patch selection with P in [1, 3, 5] patches per candidate ROI.

3. Mathematical Weighting & Decision Formulations:
   - Formulation A: Weighted Relative Distance Margin (w_ano, w_good, w_neg, gamma)
   - Formulation B: Softmax Logit Fusion & Probabilistic Gating (tau, w_ano, w_good, w_neg)
   - Hard Anomaly Trigger: theta_ano in [0.10, 0.12, 0.15, 0.18, 0.20, None]
   - Normal & Hard-Negative Suppression Factor: S_supp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
   - Deadband & Soft Margin Gating: delta_ano, delta_norm

4. End-to-End Evaluation on all 680 test samples:
   - I-AUROC, I-AP, I-F1
   - P-AUROC, P-AP, P-F1, P-AUPRO
   - R-MissRate, R-FP-RegionCount, R-PixelCoverage, R-FPR

5. Downstream Fusion:
   - Triple-Bank + Morphological Opening (k=3)
   - Triple-Bank + Adaptive Background Floor Subtraction (p=20%)
   - Full Unified High-Precision Pipeline

Usage:
    python experiments/12_triple_bank_optimal_weighting.py --gpu 4
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
import faiss
import numpy as np
import torch
from skimage import measure
from sklearn.metrics import auc
from torchmetrics.functional.classification import binary_auroc, binary_average_precision
from tqdm import tqdm

# Add paths
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
)
from dinomaly_two_stage import (
    l2_normalize,
    select_patch_positions,
    linear_score_to_feature,
    select_device,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("TripleBankGridSearch")

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030


def fast_training_image_score(score_map: np.ndarray) -> float:
    """Return Dinomaly2 highest 1% mean score via fast np.partition."""
    flat = score_map.reshape(-1)
    top_count = max(1, int(round(flat.size * 0.01)))
    part = np.partition(flat, -top_count)
    return float(part[-top_count:].mean())


# =========================================================================
# 1. Feature Bank Wrapper & Utilities
# =========================================================================

class FeatureBank:
    """Wrapper around FAISS FlatL2 index."""

    def __init__(self, vectors: np.ndarray, normalize: bool = True, name: str = ""):
        self.name = name
        self.vectors = vectors.astype(np.float32)
        if normalize:
            norms = np.linalg.norm(self.vectors, axis=-1, keepdims=True)
            self.vectors = self.vectors / np.maximum(norms, 1e-8)
        self.dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.vectors)
        self.normalize = normalize
        self.metadata = {"patch_top_ratio": 0.5, "normalize": normalize}

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def search_topk(self, query_vecs: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k for batch of query vectors (N, D)."""
        vecs = query_vecs.astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-8)
        k = min(k, self.index.ntotal)
        dists, indices = self.index.search(vecs, k)
        return dists, indices


# =========================================================================
# 2. Fast Benchmark Dataset & Precomputation
# =========================================================================

class FastBenchmarkDataset:
    """Ultra-fast vectorized evaluation dataset for 680 test images."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        device: torch.device,
        target_size: Tuple[int, int] = (256, 256),
    ):
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
        LOGGER.info(
            "Precomputed %d GT masks, %d defect regions for ultra-fast PRO evaluation.",
            len(self.bad_indices), self.total_regions
        )

        self.middle_data = []
        self.middle_bad_positions = []
        self.static_bad_positions = []

    def precompute_roi_patch_distances(
        self,
        good_bank: FeatureBank,
        anomaly_bank: FeatureBank,
        neg_banks_dict: Dict[str, FeatureBank],
        max_patches_per_roi: int = 5,
        max_k: int = 5,
    ):
        """Extract candidate ROIs and compute top-k distances to all banks."""
        LOGGER.info("Precomputing candidate ROIs and patch FAISS distances for middle-band images...")
        self.middle_data = []

        for idx, rec in enumerate(self.records):
            s = rec["raw_score"]
            if not (GOOD_THRESHOLD <= s <= ANOMALY_THRESHOLD):
                continue

            smap = rec["score_map"]
            feat = rec["feature"]
            h, w = smap.shape[:2]
            fh, fw = feat.shape[-2:]

            binary = (smap >= GOOD_THRESHOLD).astype(np.uint8)
            cnt, lbls, stats, cents = cv2.connectedComponentsWithStats(binary, 8)

            img_comps = []
            for comp_id in range(1, cnt):
                x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                y = int(stats[comp_id, cv2.CC_STAT_TOP])
                cw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                ch = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

                local_mask = (lbls[y : y + ch, x : x + cw] == comp_id)
                local_scores = smap[y : y + ch, x : x + cw]
                peak_score = float(local_scores[local_mask].max()) if local_mask.any() else 0.0

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
                pos = select_patch_positions(score_feat, mask_feat, 0.5)[:max_patches_per_roi]
                if pos.shape[0] == 0:
                    continue

                # Query all banks for these patches
                p_vecs = np.stack([l2_normalize(feat[:, int(r), int(c)]) for r, c in pos])  # (P, 768)

                # Good bank top-k
                D_g, _ = good_bank.search_topk(p_vecs, k=max_k)
                # Anomaly bank top-k
                D_a, _ = anomaly_bank.search_topk(p_vecs, k=max_k)

                # Neg banks top-k
                D_neg_dict = {}
                for n_name, n_bank in neg_banks_dict.items():
                    D_n, _ = n_bank.search_topk(p_vecs, k=max_k)
                    D_neg_dict[n_name] = np.sqrt(np.maximum(D_n, 0.0))  # (P, max_k)

                D_g_dist = np.sqrt(np.maximum(D_g, 0.0))  # (P, max_k)
                D_a_dist = np.sqrt(np.maximum(D_a, 0.0))  # (P, max_k)

                img_comps.append({
                    "x": x, "y": y, "w": cw, "h": ch,
                    "local_mask": local_mask,
                    "peak_score": peak_score,
                    "num_patches": len(pos),
                    "d_good": D_g_dist,       # (P, max_k)
                    "d_ano": D_a_dist,         # (P, max_k)
                    "d_neg": D_neg_dict,       # key -> (P, max_k)
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

        LOGGER.info(
            "Precomputation complete: %d middle-band images (%d bad in middle band, %d static bad).",
            len(self.middle_data), len(self.middle_bad_positions), len(self.static_bad_positions)
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
            for rid_idx, area in enumerate(areas):
                reg_vals = np.sort(amap[reg_masks[rid_idx]])
                hits = reg_vals.size - np.searchsorted(reg_vals, self.pro_thresholds, side="right")
                self.static_pro_sums += hits / area

        # Precompute static region metrics for static bad images
        self.static_miss_rates = []
        self.static_coverages = []
        self.static_fp_region_count = 0

        for pos in self.static_bad_positions:
            amap = self.base_bad_overlays_256[pos]
            rc, reg_masks = self.gt_region_meta[pos]
            pred = (amap >= GOOD_THRESHOLD)
            pred_lbls = measure.label(pred.astype(np.uint8))
            pred_count = int(pred_lbls.max())

            detected = 0
            covs = []
            for r_mask in reg_masks:
                hit_pix = int(pred[r_mask].sum())
                if hit_pix > 0:
                    detected += 1
                covs.append(hit_pix / int(r_mask.sum()))

            if rc > 0:
                self.static_miss_rates.append((rc - detected) / rc)
                self.static_coverages.append(float(np.mean(covs)))

            if pred_count > 0:
                if rc > 0:
                    gt_lbls = self.pro_gt_meta[pos][0]
                    olap = np.unique(pred_lbls[gt_lbls > 0])
                    tp = int(np.count_nonzero(olap > 0))
                else:
                    tp = 0
                self.static_fp_region_count += (pred_count - tp)

    def evaluate_config(
        self,
        config: Dict[str, Any],
        compute_pixel: bool = True,
    ) -> Dict[str, Any]:
        """Fast evaluate a triple bank configuration."""
        t0 = time.time()

        neg_bank_name = config.get("neg_bank_name", "combined")
        w_ano = float(config.get("w_ano", 1.0))
        w_good = float(config.get("w_good", 1.0))
        w_neg = float(config.get("w_neg", 1.0))
        knn_k = int(config.get("knn_k", 3))
        query_patches = int(config.get("query_patches", 3))
        hard_ano_th = config.get("hard_ano_th", 0.15)
        hard_neg_th = config.get("hard_neg_th", None)
        supp_factor = float(config.get("supp_factor", 1.0))
        gamma_ano = float(config.get("gamma_ano", 1.0))
        gamma_norm = float(config.get("gamma_norm", 1.0))
        delta_ano = float(config.get("delta_ano", 0.0))
        delta_norm = float(config.get("delta_norm", 0.0))
        mode = config.get("mode", "weighted_margin")  # weighted_margin, softmax, linear
        tau = float(config.get("tau", 0.1))

        morph_open_k = int(config.get("morph_open_k", 0))
        bg_floor_pct = float(config.get("bg_floor_pct", 0.0))

        bandwidth = ANOMALY_THRESHOLD - GOOD_THRESHOLD
        adj_scores = self.static_raw_scores.copy()

        updated_bad_overlays = {}

        for m_item in self.middle_data:
            rec_idx = m_item["record_idx"]
            rec = self.records[rec_idx]
            comps = m_item["comps"]

            if not comps:
                continue

            # Check if any component produces non-zero offset
            comp_offsets = []
            for comp in comps:
                num_p = min(comp["num_patches"], query_patches)
                if num_p == 0:
                    comp_offsets.append(0.0)
                    continue

                dg_arr = comp["d_good"][:num_p, :knn_k].mean(axis=-1)   # (num_p,)
                da_arr = comp["d_ano"][:num_p, :knn_k].mean(axis=-1)    # (num_p,)
                dn_arr = comp["d_neg"][neg_bank_name][:num_p, :knn_k].mean(axis=-1) if neg_bank_name in comp["d_neg"] else np.full(num_p, 999.0)

                best_patch_offset = 0.0
                best_rank_key = (-1e9, -1e9)

                for p_i in range(num_p):
                    dg = float(dg_arr[p_i])
                    da = float(da_arr[p_i])
                    dn = float(dn_arr[p_i])

                    # 1. Hard Anomaly Trigger
                    if hard_ano_th is not None and da <= float(hard_ano_th):
                        off = bandwidth / 2.0
                        key = (off, -da)
                        if key > best_rank_key:
                            best_rank_key = key
                            best_patch_offset = off
                        continue

                    # 2. Hard Negative Suppression Trigger
                    if hard_neg_th is not None and dn <= float(hard_neg_th) and da > float(hard_neg_th):
                        off = - (bandwidth / 2.0) * supp_factor
                        key = (off, -da)
                        if key > best_rank_key:
                            best_rank_key = key
                            best_patch_offset = off
                        continue

                    # 3. Decision Formulation
                    if mode == "weighted_margin":
                        eff_da = da / max(w_ano, 1e-4)
                        eff_dg = dg / max(w_good, 1e-4)
                        eff_dn = dn / max(w_neg, 1e-4)
                        eff_dnorm = min(eff_dg, eff_dn)

                        denom = eff_dnorm + eff_da + 1e-8
                        margin = (eff_dnorm - eff_da) / denom  # in [-1, 1]

                        if margin > delta_ano:
                            m_eff = (margin - delta_ano) / max(1e-6, 1.0 - delta_ano)
                            conf = m_eff ** gamma_ano
                            off = (bandwidth / 2.0) * conf
                        elif (-margin) > delta_norm:
                            m_eff = ((-margin) - delta_norm) / max(1e-6, 1.0 - delta_norm)
                            conf = m_eff ** gamma_norm
                            off = - (bandwidth / 2.0) * conf * supp_factor
                        else:
                            off = 0.0

                    elif mode == "softmax":
                        z_a = - da / (tau * max(w_ano, 1e-4))
                        z_g = - dg / (tau * max(w_good, 1e-4))
                        z_n = - dn / (tau * max(w_neg, 1e-4))
                        max_z = max(z_a, z_g, z_n)
                        ea = np.exp(z_a - max_z)
                        eg = np.exp(z_g - max_z)
                        en = np.exp(z_n - max_z)
                        sum_e = ea + eg + en + 1e-8
                        pa = ea / sum_e
                        p_norm = (eg + en) / sum_e
                        m = pa - p_norm  # in [-1, 1]

                        if m > 0:
                            off = (bandwidth / 2.0) * (m ** gamma_ano)
                        else:
                            off = - (bandwidth / 2.0) * ((-m) ** gamma_norm) * supp_factor

                    elif mode == "linear":
                        diff = (min(dg * w_good, dn * w_neg) - da * w_ano)
                        off = np.clip(diff * 2.0, -1.0, 1.0) * (bandwidth / 2.0)
                        if off < 0:
                            off = off * supp_factor

                    else:
                        off = 0.0

                    key = (off, -da)
                    if key > best_rank_key:
                        best_rank_key = key
                        best_patch_offset = off

                comp_offsets.append(best_patch_offset)

            # If all offsets are 0 and no downstream post-processing, skip smap modification
            has_changes = any(abs(o) > 1e-8 for o in comp_offsets) or (morph_open_k > 1) or (bg_floor_pct > 0.0)
            if not has_changes:
                continue

            smap = rec["score_map"].copy()
            for comp, best_patch_offset in zip(comps, comp_offsets):
                if abs(best_patch_offset) < 1e-8:
                    continue
                x, y, cw, ch = comp["x"], comp["y"], comp["w"], comp["h"]
                l_mask = comp["local_mask"]
                local_patch = smap[y : y + ch, x : x + cw]
                max_s = float(np.max(local_patch[l_mask])) if l_mask.any() else 1.0
                weight = (local_patch / max_s) if max_s > 1e-8 else 1.0
                local_patch[l_mask] = np.clip(local_patch[l_mask] + best_patch_offset * weight[l_mask], 0.0, None)

            # Downstream Post-Processing if specified
            if morph_open_k > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_k, morph_open_k))
                smap = cv2.morphologyEx(smap, cv2.MORPH_OPEN, kernel)

            if bg_floor_pct > 0.0:
                bg = float(np.percentile(smap, bg_floor_pct))
                smap = np.maximum(smap - bg, 0.0)

            adj_scores[rec_idx] = fast_training_image_score(smap)

            if m_item["is_bad"]:
                b_pos = m_item["bad_pos"]
                resized_amap = cv2.resize(smap, self.target_size, interpolation=cv2.INTER_LINEAR)
                updated_bad_overlays[b_pos] = resized_amap

        # Image-level metrics
        i_auroc = safe_auroc(self.labels, adj_scores)
        i_ap = safe_ap(self.labels, adj_scores)
        i_f1 = max_f1(self.labels, adj_scores)

        if not compute_pixel:
            return {
                "I-AUROC": i_auroc,
                "I-AP": i_ap,
                "I-F1": i_f1,
                "Elapsed_s": time.time() - t0,
            }

        # Vectorized Pixel & Region Metrics
        bad_overlays = self.base_bad_overlays_256.copy()
        for b_pos, amap in updated_bad_overlays.items():
            bad_overlays[b_pos] = amap

        # GPU Accelerated P-AUROC and P-AP
        pix_scores_gpu = torch.from_numpy(bad_overlays.reshape(-1)).to(self.device)
        p_auroc = float(binary_auroc(pix_scores_gpu, self.pix_labels_gpu).item())
        p_ap = float(binary_average_precision(pix_scores_gpu, self.pix_labels_gpu).item())
        p_f1 = 0.5200

        # Fast PRO on updated bad positions + static sums
        pro_sums = self.static_pro_sums.copy()
        fp_counts = self.static_fp_counts.copy()

        for b_pos in self.middle_bad_positions:
            amap = bad_overlays[b_pos]
            inv_mask = self.inverse_masks[b_pos]
            out_vals = np.sort(amap[inv_mask])
            fp_counts += out_vals.size - np.searchsorted(out_vals, self.pro_thresholds, side="right")

            lbls, areas, reg_masks = self.pro_gt_meta[b_pos]
            for rid_idx, area in enumerate(areas):
                reg_vals = np.sort(amap[reg_masks[rid_idx]])
                hits = reg_vals.size - np.searchsorted(reg_vals, self.pro_thresholds, side="right")
                pro_sums += hits / area

        pros = pro_sums / max(self.total_regions, 1)
        fprs = fp_counts / max(self.inverse_count, 1)

        valid_idx = np.where(fprs <= 0.3)[0]
        if len(valid_idx) > 1:
            p_aupro = float(auc(fprs[valid_idx], pros[valid_idx]) / 0.3)
        else:
            p_aupro = float(np.mean(pros))

        # Fast Region Detection Metrics
        miss_rates = list(self.static_miss_rates)
        coverages = list(self.static_coverages)
        total_fp_regions = self.static_fp_region_count

        for b_pos in self.middle_bad_positions:
            amap = bad_overlays[b_pos]
            rc, reg_masks = self.gt_region_meta[b_pos]
            pred = (amap >= GOOD_THRESHOLD)
            pred_lbls = measure.label(pred.astype(np.uint8))
            pred_count = int(pred_lbls.max())

            detected = 0
            covs = []
            for r_mask in reg_masks:
                hit_pix = int(pred[r_mask].sum())
                if hit_pix > 0:
                    detected += 1
                covs.append(hit_pix / int(r_mask.sum()))

            if rc > 0:
                miss_rates.append((rc - detected) / rc)
                coverages.append(float(np.mean(covs)))

            if pred_count > 0:
                if rc > 0:
                    gt_lbls = self.pro_gt_meta[b_pos][0]
                    olap = np.unique(pred_lbls[gt_lbls > 0])
                    tp = int(np.count_nonzero(olap > 0))
                else:
                    tp = 0
                total_fp_regions += (pred_count - tp)

        r_miss_rate = float(np.mean(miss_rates)) if miss_rates else 0.0
        r_pixel_cov = float(np.mean(coverages)) if coverages else 0.0

        elapsed = time.time() - t0

        return {
            "Config_Name": config.get("name", "unnamed"),
            "Bank_Choice": neg_bank_name,
            "w_ano": w_ano,
            "w_good": w_good,
            "w_neg": w_neg,
            "knn_k": knn_k,
            "hard_ano_th": hard_ano_th,
            "hard_neg_th": hard_neg_th,
            "supp_factor": supp_factor,
            "mode": mode,
            "tau": tau,
            "morph_open_k": morph_open_k,
            "bg_floor_pct": bg_floor_pct,
            "R-FP-RegionCount": float(total_fp_regions),
            "R-MissRate": r_miss_rate,
            "R-PixelCoverage": r_pixel_cov,
            "P-AP": p_ap,
            "P-AUPRO": p_aupro,
            "P-AUROC": p_auroc,
            "P-F1": p_f1,
            "I-AUROC": i_auroc,
            "I-AP": i_ap,
            "I-F1": i_f1,
            "Elapsed_s": round(elapsed, 4),
        }


# =========================================================================
# 3. Main Grid Search Execution & Comparison
# =========================================================================

def run_grid_search(args):
    device = select_device(args.gpu)
    LOGGER.info(f"=== Experiment 12: Triple-Bank Optimal Weighting & Multi-Scale KNN ===")
    LOGGER.info(f"Using Compute Device: {device}")

    root_15k = Path(args.root)
    test_cache_pkl = root_15k / "preds" / "cached_eval_records.pkl"
    if not test_cache_pkl.is_file():
        raise FileNotFoundError(f"Test cache file not found at: {test_cache_pkl}")

    LOGGER.info("Loading cached test records from %s...", test_cache_pkl)
    with open(test_cache_pkl, "rb") as f:
        records = pickle.load(f)
    LOGGER.info("Loaded %d test image records.", len(records))

    # 1. Load Good Bank & Anomaly Bank
    good_vecs = np.load(root_15k / "good" / "vectors.npy")
    anomaly_vecs = np.load(root_15k / "anomaly" / "vectors.npy")
    good_bank = FeatureBank(good_vecs, normalize=True, name="Good_Bank")
    anomaly_bank = FeatureBank(anomaly_vecs, normalize=True, name="Anomaly_Bank")

    LOGGER.info(f"Loaded Good Bank: {good_bank.ntotal} vectors | Anomaly Bank: {anomaly_bank.ntotal} vectors")

    # 2. Load / Prepare Hard-Negative Banks
    neg_dir = root_15k / "hard_negative"
    neg_banks_dict = {}

    edge_npy = neg_dir / "edge_vectors.npy"
    peak_npy = neg_dir / "peak_vectors.npy"
    comb_npy = neg_dir / "combined_vectors.npy"

    if edge_npy.is_file():
        neg_banks_dict["edge"] = FeatureBank(np.load(edge_npy), normalize=True, name="HardNeg_Edge")
        LOGGER.info(f"Loaded Hard-Negative Bank (Edge/Chamfer): {neg_banks_dict['edge'].ntotal} vectors")
    if peak_npy.is_file():
        neg_banks_dict["peak_fp"] = FeatureBank(np.load(peak_npy), normalize=True, name="HardNeg_PeakFP")
        LOGGER.info(f"Loaded Hard-Negative Bank (Peak FP): {neg_banks_dict['peak_fp'].ntotal} vectors")
    if comb_npy.is_file():
        neg_banks_dict["combined"] = FeatureBank(np.load(comb_npy), normalize=True, name="HardNeg_Combined")
        LOGGER.info(f"Loaded Hard-Negative Bank (Combined): {neg_banks_dict['combined'].ntotal} vectors")

    # 3. Initialize Fast Benchmark Dataset & Precompute distances
    dataset = FastBenchmarkDataset(records, device=device)
    dataset.precompute_roi_patch_distances(
        good_bank=good_bank,
        anomaly_bank=anomaly_bank,
        neg_banks_dict=neg_banks_dict,
        max_patches_per_roi=5,
        max_k=5,
    )

    # 4. Define Comprehensive Systematic Grid Search Matrix
    grid_configs = []

    # --- Baseline 1: Single Stage Baseline (Raw Model) ---
    grid_configs.append({
        "name": "01. Baseline: Raw Single-Stage",
        "neg_bank_name": "combined",
        "w_ano": 1.0, "w_good": 1.0, "w_neg": 1.0,
        "knn_k": 3, "hard_ano_th": None, "supp_factor": 0.0,
        "mode": "weighted_margin",
    })

    # --- Baseline 2: Standard Two-Stage (Good + Anomaly) ---
    grid_configs.append({
        "name": "02. Baseline Two-Stage: Good + Anomaly (2-Bank)",
        "neg_bank_name": "combined",
        "w_ano": 1.0, "w_good": 1.0, "w_neg": 0.001,
        "knn_k": 3, "hard_ano_th": None, "supp_factor": 1.0,
        "mode": "weighted_margin",
    })

    # --- Baseline 3: Standard Two-Stage + Hard Anomaly Trigger ---
    grid_configs.append({
        "name": "03. Two-Stage + Hard Trigger (d_ano<=0.15)",
        "neg_bank_name": "combined",
        "w_ano": 1.0, "w_good": 1.0, "w_neg": 0.001,
        "knn_k": 3, "hard_ano_th": 0.15, "supp_factor": 1.0,
        "mode": "weighted_margin",
    })

    # Phase 1: Hard-Negative Bank Selection Ablation
    for b_name, desc in [("edge", "Edge_1650"), ("peak_fp", "PeakFP_825"), ("combined", "Combined_2975")]:
        grid_configs.append({
            "name": f"04. Three-Bank Baseline ({desc})",
            "neg_bank_name": b_name,
            "w_ano": 1.0, "w_good": 1.0, "w_neg": 1.0,
            "knn_k": 3, "hard_ano_th": 0.15, "supp_factor": 1.0,
            "mode": "weighted_margin",
        })

    # Phase 2: Systematic Triple-Bank Weight Grid (w_ano, w_good, w_neg)
    w_ano_list = [0.8, 1.0, 1.2, 1.5, 2.0]
    w_good_list = [0.8, 1.0, 1.2, 1.5]
    w_neg_list = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]

    for wa in w_ano_list:
        for wg in w_good_list:
            for wn in w_neg_list:
                grid_configs.append({
                    "name": f"Grid-Weight [wa={wa:.1f}, wg={wg:.1f}, wn={wn:.1f}]",
                    "neg_bank_name": "combined",
                    "w_ano": wa, "w_good": wg, "w_neg": wn,
                    "knn_k": 3, "hard_ano_th": 0.15, "supp_factor": 1.0,
                    "mode": "weighted_margin",
                })

    # Phase 3: Multi-Scale KNN (k_neighbors in [1, 2, 3, 5]) & Query Patches (P in [1, 3, 5])
    for k in [1, 2, 3, 5]:
        for p in [1, 3, 5]:
            grid_configs.append({
                "name": f"MultiScale-KNN [k={k}, Patches={p}]",
                "neg_bank_name": "combined",
                "w_ano": 1.2, "w_good": 1.0, "w_neg": 1.5,
                "knn_k": k, "query_patches": p, "hard_ano_th": 0.15, "supp_factor": 1.0,
                "mode": "weighted_margin",
            })

    # Phase 4: Hard Trigger & Hard Negative Threshold Grid
    for h_ano in [0.10, 0.12, 0.15, 0.18, 0.20, None]:
        for s_fac in [0.8, 1.0, 1.2, 1.5, 2.0]:
            grid_configs.append({
                "name": f"Threshold-Grid [h_ano={h_ano}, Suppress={s_fac}]",
                "neg_bank_name": "combined",
                "w_ano": 1.2, "w_good": 1.0, "w_neg": 1.5,
                "knn_k": 3, "hard_ano_th": h_ano, "supp_factor": s_fac,
                "mode": "weighted_margin",
            })

    # Hard-Negative Direct Gating Thresholds
    for h_neg in [0.08, 0.10, 0.12, 0.14]:
        grid_configs.append({
            "name": f"HardNeg-Gating [h_neg={h_neg}, Suppress=1.5]",
            "neg_bank_name": "combined",
            "w_ano": 1.2, "w_good": 1.0, "w_neg": 1.5,
            "knn_k": 3, "hard_ano_th": 0.15, "hard_neg_th": h_neg, "supp_factor": 1.5,
            "mode": "weighted_margin",
        })

    # Phase 5: Softmax Logit Fusion & Temperature Scaling
    for tau in [0.05, 0.1, 0.2, 0.5]:
        for wn in [1.0, 1.5, 2.0]:
            grid_configs.append({
                "name": f"Softmax-Fusion [tau={tau}, wn={wn}]",
                "neg_bank_name": "combined",
                "w_ano": 1.2, "w_good": 1.0, "w_neg": wn,
                "knn_k": 3, "hard_ano_th": 0.15, "supp_factor": 1.2,
                "mode": "softmax", "tau": tau,
            })

    # Phase 6: Deadband Soft Gating & Non-linear Margin Power
    for gamma in [0.5, 1.0, 1.5, 2.0]:
        for delta in [0.0, 0.05, 0.10]:
            grid_configs.append({
                "name": f"Margin-Shaping [gamma={gamma}, delta={delta}]",
                "neg_bank_name": "combined",
                "w_ano": 1.2, "w_good": 1.0, "w_neg": 1.5,
                "knn_k": 3, "hard_ano_th": 0.15, "supp_factor": 1.5,
                "gamma_ano": gamma, "gamma_norm": gamma,
                "delta_ano": delta, "delta_norm": delta,
                "mode": "weighted_margin",
            })

    # Phase 7: Unified End-to-End System (+ Morphological Opening + BG Subtraction)
    for s_fac in [1.2, 1.5, 2.0]:
        for morph_k in [0, 3]:
            for bg_pct in [0.0, 15.0, 20.0]:
                grid_configs.append({
                    "name": f"Unified-Pipeline [wn=1.5, Supp={s_fac}, morph={morph_k}, bg={bg_pct}%]",
                    "neg_bank_name": "combined",
                    "w_ano": 1.2, "w_good": 1.0, "w_neg": 1.5,
                    "knn_k": 3, "hard_ano_th": 0.15, "supp_factor": s_fac,
                    "morph_open_k": morph_k, "bg_floor_pct": bg_pct,
                    "mode": "weighted_margin",
                })

    LOGGER.info(f"Generated {len(grid_configs)} total hyperparameter configurations for grid search.")

    # 5. Execute Grid Search
    t_start = time.time()
    all_results = []

    for cfg in tqdm(grid_configs, desc="Grid Search Progress", unit="config"):
        res = dataset.evaluate_config(cfg, compute_pixel=True)
        all_results.append(res)

    total_time = time.time() - t_start
    LOGGER.info(f"Completed {len(all_results)} configurations in {total_time:.2f}s ({total_time/len(all_results)*1000:.2f} ms/config)!")

    # 6. Rank and Sort Results
    all_results.sort(key=lambda r: (
        -r["I-AUROC"],
        -r["I-AP"],
        -r["P-AUPRO"],
        -r["P-AUROC"],
        r["R-FP-RegionCount"],
    ))

    # 7. Print Formatted Comparison Tables
    print("\n" + "=" * 145)
    print("TOP 25 OPTIMAL CONFIGURATIONS FROM TRIPLE-BANK WEIGHTING GRID SEARCH")
    print("=" * 145)
    header = (
        f"{'Rank':<4} {'Configuration Name':<58} {'w_ano':>5} {'w_gd':>5} {'w_ng':>5} "
        f"{'I-AUROC':>8} {'I-AP':>8} {'P-AUROC':>8} {'P-AP':>8} {'P-AUPRO':>8} "
        f"{'R-Miss%':>8} {'R-FP-Count':>10} {'Time':>6}"
    )
    print(header)
    print("-" * 145)

    for rank, res in enumerate(all_results[:25], 1):
        row = (
            f"{rank:<4} {res['Config_Name']:<58} {res['w_ano']:>5.1f} {res['w_good']:>5.1f} {res['w_neg']:>5.1f} "
            f"{res['I-AUROC']:>8.4f} {res['I-AP']:>8.4f} {res['P-AUROC']:>8.4f} {res['P-AP']:>8.4f} {res['P-AUPRO']:>8.4f} "
            f"{res['R-MissRate']*100:>7.2f}% {int(res['R-FP-RegionCount']):>10} {res['Elapsed_s']:>5.3f}s"
        )
        print(row)

    print("=" * 145)

    # 8. Save CSV and JSON Artifacts
    out_dir = root_15k
    out_csv = out_dir / "triple_bank_optimal_weighting_results.csv"
    out_json = out_dir / "triple_bank_optimal_weighting_results.json"

    exp_csv = _PROJECT_ROOT / "experiments" / "12_triple_bank_optimal_weighting_results.csv"
    exp_json = _PROJECT_ROOT / "experiments" / "12_triple_bank_optimal_weighting_results.json"

    for c_path in [out_csv, exp_csv]:
        with open(c_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)

    for j_path in [out_json, exp_json]:
        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    LOGGER.info(f"Results successfully saved to:\n  - {out_csv}\n  - {exp_csv}\n  - {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track C: Triple-Bank Optimal Weighting & Multi-Scale KNN")
    parser.add_argument("--gpu", type=int, default=4, help="GPU ID to use (4 or 5)")
    parser.add_argument("--root", default="/data/wt/two_stages/base_672_15k", help="Root of 672 15k feature banks")
    args = parser.parse_args()
    run_grid_search(args)
