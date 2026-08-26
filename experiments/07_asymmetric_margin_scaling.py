#!/usr/bin/env python3
"""Experiment 7: Asymmetric Confidence Scaling & Margin Mapping (Track 2).

This experiment implements and systematically evaluates:
1. Asymmetric Sigmoid / Tanh Temperature Scaling:
   - Separate temperature parameters for Good bank (tau_g in [0.05, 0.1, 0.2]) and Anomaly bank (tau_a in [0.02, 0.05, 0.1]).
   - Non-linear margin saturation mapping M in [-1, 1] avoiding ambiguous boundary jitter.
2. Independent Soft Thresholding (Deadband & Soft Gating):
   - Independent deadbands delta_g and delta_a for good suppression and anomaly confirmation.
   - Elimination of low-confidence boundary noise near M ~ 0.
3. Asymmetric Scaling Factors (scale_g, scale_a) & Power Shaping (gamma_g, gamma_a):
   - Asymmetric scaling to strongly confirm defects while calibratedly suppressing normal background.
4. Combined Full Pipeline:
   - Optimal asymmetric temperature & margin mapping + Hard Anomaly Trigger + Morphological Opening + Adaptive Background Floor Subtraction.
5. Full 680-image quantitative benchmark:
   - Evaluates I-AUROC, I-AP, P-AUROC, P-AP, P-AUPRO, R-MissRate, R-FP-RegionCount.

Usage:
    python experiments/07_asymmetric_margin_scaling.py --gpu 2
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
import faiss
import numpy as np
import torch
from skimage import measure
from sklearn.metrics import auc
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

# Add project root and utils to sys.path
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
from dinomaly_two_stage import (
    load_feature_library,
    select_patch_positions,
    linear_score_to_feature,
    l2_normalize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("track2_asymmetric_margin")

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030
DEFAULT_BANDWIDTH = (ANOMALY_THRESHOLD - GOOD_THRESHOLD) / 2.0  # 0.008


# =========================================================================
# 1. Asymmetric Confidence Scaling & Margin Mapping Functions
# =========================================================================

def compute_asymmetric_margin_offset(
    dg: float,
    da: float,
    tau_g: float = 0.10,
    tau_a: float = 0.05,
    delta_g: float = 0.0,
    delta_a: float = 0.0,
    scale_g: float = 1.0,
    scale_a: float = 1.0,
    gamma_g: float = 1.0,
    gamma_a: float = 1.0,
    bandwidth: float = DEFAULT_BANDWIDTH,
    hard_anomaly_th: Optional[float] = None,
    mode: str = "branch_sigmoid",
) -> Tuple[float, float, str]:
    """Calculate signed score adjustment via asymmetric temperature scaling & soft thresholding.

    Args:
        dg: Nearest-neighbour (or Top-k mean) distance to Good feature bank.
        da: Nearest-neighbour (or Top-k mean) distance to Anomaly feature bank.
        tau_g: Good bank temperature parameter (controls smoothness of good suppression).
        tau_a: Anomaly bank temperature parameter (controls sharpness of anomaly confirmation).
        delta_g: Good branch deadband / soft threshold.
        delta_a: Anomaly branch deadband / soft threshold.
        scale_g: Good branch score offset multiplier.
        scale_a: Anomaly branch score offset multiplier.
        gamma_g: Good branch non-linear power shaping exponent.
        gamma_a: Anomaly branch non-linear power shaping exponent.
        bandwidth: Base score adjustment magnitude ((T_anomaly - T_good) / 2).
        hard_anomaly_th: If not None, queries with da <= hard_anomaly_th directly trigger anomaly.
        mode: Mapping mode ("branch_sigmoid", "linear", "logit_diff").

    Returns:
        signed_offset: Signed score change (negative = suppress good, positive = boost anomaly).
        confidence: Normalized confidence magnitude in [0, 1].
        decision_type: "good", "anomaly", or "hard_anomaly".
    """
    if hard_anomaly_th is not None and da <= hard_anomaly_th:
        return bandwidth * scale_a, 1.0, "hard_anomaly"

    if mode == "linear":
        denom = da + dg + 1e-8
        margin = (da - dg) / denom  # in [-1, 1]
        signed_offset = - bandwidth * margin * (scale_g if margin > 0 else scale_a)
        conf = abs(margin)
        return float(signed_offset), float(conf), ("good" if margin > 0 else "anomaly")

    elif mode == "branch_sigmoid":
        denom = da + dg + 1e-8
        rel_diff = (da - dg) / denom  # > 0 means closer to good, < 0 means closer to anomaly

        if rel_diff > 0:
            # Good Branch (Suppress false alarm)
            raw_m = max(0.0, rel_diff - delta_g)
            if delta_g < 1.0:
                raw_m = raw_m / (1.0 - delta_g)
            conf = float(np.tanh(raw_m / max(tau_g, 1e-6))) ** gamma_g
            signed_offset = - bandwidth * scale_g * conf
            return float(signed_offset), float(conf), "good"
        else:
            # Anomaly Branch (Confirm genuine defect)
            raw_m = max(0.0, (-rel_diff) - delta_a)
            if delta_a < 1.0:
                raw_m = raw_m / (1.0 - delta_a)
            conf = float(np.tanh(raw_m / max(tau_a, 1e-6))) ** gamma_a
            signed_offset = bandwidth * scale_a * conf
            return float(signed_offset), float(conf), "anomaly"

    elif mode == "logit_diff":
        delta_z = (dg / max(tau_g, 1e-6)) - (da / max(tau_a, 1e-6))
        M = float(np.tanh(delta_z / 2.0))
        if M > 0:
            m_soft = max(0.0, M - delta_a) / max(1e-6, 1.0 - delta_a)
            conf = float(m_soft ** gamma_a)
            signed_offset = bandwidth * scale_a * conf
            return float(signed_offset), conf, "anomaly"
        else:
            m_soft = max(0.0, (-M) - delta_g) / max(1e-6, 1.0 - delta_g)
            conf = float(m_soft ** gamma_g)
            signed_offset = - bandwidth * scale_g * conf
            return float(signed_offset), conf, "good"

    else:
        raise ValueError(f"Unknown margin mapping mode: {mode}")


# =========================================================================
# 2. Ultra-Fast Precomputation Structures & Vectorized PRO Evaluator
# =========================================================================

class FastBenchmarkDataset:
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
        LOGGER.info("Precomputed %d GT masks, %d defect regions for fast vector PRO evaluation.", len(self.bad_indices), self.total_regions)

        self.middle_data = []
        self.middle_bad_positions = []
        self.static_bad_positions = []

    def precompute_middle_components(
        self,
        good_lib: Any,
        anomaly_lib: Any,
        min_area_pct: float = 0.005,
        top_k: int = 3,
        query_patches: int = 3,
    ):
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
                    "patches": patches,
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

        # Precompute static PRO values for the 491 static bad images
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

        # Precompute static region metrics for the 491 static bad images
        self.static_miss_rates = []
        self.static_coverages = []
        self.static_fp_region_count = 0

        for pos in self.static_bad_positions:
            sm = self.base_bad_overlays_256[pos]
            pred = (sm >= GOOD_THRESHOLD).astype(np.uint8)
            pred_lbl = measure.label(pred)
            pc = int(pred_lbl.max())

            rc, reg_masks = self.gt_region_meta[pos]
            det = 0
            covs = []
            for rmask in reg_masks:
                covered = int(pred[rmask].sum())
                if covered > 0:
                    det += 1
                covs.append(covered / max(int(rmask.sum()), 1))

            if rc > 0:
                self.static_miss_rates.append((rc - det) / rc)
                self.static_coverages.append(float(np.mean(covs)))

            if pc > 0:
                gt_m = self.bad_gt_masks_256[pos]
                tp = int(np.count_nonzero(np.unique(pred_lbl[gt_m > 0]) > 0)) if rc > 0 else 0
                self.static_fp_region_count += (pc - tp)

    def compute_fast_aupro(self, amaps: np.ndarray, is_fast_path: bool = True) -> float:
        """Vectorized PRO computation in ~0.01s."""
        thresholds = self.pro_thresholds
        if is_fast_path:
            pro_sums = self.static_pro_sums.copy()
            fp_counts = self.static_fp_counts.copy()
            eval_positions = self.middle_bad_positions
        else:
            pro_sums = np.zeros_like(thresholds, dtype=np.float64)
            fp_counts = np.zeros_like(thresholds, dtype=np.int64)
            eval_positions = range(len(self.bad_indices))

        for pos in eval_positions:
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
# 3. High-Speed Full Benchmark Evaluator
# =========================================================================

def evaluate_margin_configuration(
    dataset: FastBenchmarkDataset,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one parameter configuration across all 680 test images."""
    t0 = time.time()

    tau_g = config.get("tau_g", 0.10)
    tau_a = config.get("tau_a", 0.05)
    delta_g = config.get("delta_g", 0.0)
    delta_a = config.get("delta_a", 0.0)
    scale_g = config.get("scale_g", 1.0)
    scale_a = config.get("scale_a", 1.0)
    gamma_g = config.get("gamma_g", 1.0)
    gamma_a = config.get("gamma_a", 1.0)
    hard_th = config.get("hard_th", None)
    mode = config.get("mode", "branch_sigmoid")
    bg_floor = config.get("bg_floor", 0.0)
    morph_k = config.get("morph_k", 0)

    # Initialize scores from static raw scores
    adj_scores = dataset.static_raw_scores.copy()
    bad_overlays_256 = dataset.base_bad_overlays_256.copy()

    # Fast path: update middle-band images only
    if bg_floor == 0.0 and morph_k == 0:
        for m in dataset.middle_data:
            rec_idx = m["record_idx"]
            smap = dataset.records[rec_idx]["score_map"]
            overlay = smap.copy()

            for comp in m["comps"]:
                candidates = []
                for p in comp["patches"]:
                    dg, da = p["dg"], p["da"]
                    off, conf, dec = compute_asymmetric_margin_offset(
                        dg=dg, da=da,
                        tau_g=tau_g, tau_a=tau_a,
                        delta_g=delta_g, delta_a=delta_a,
                        scale_g=scale_g, scale_a=scale_a,
                        gamma_g=gamma_g, gamma_a=gamma_a,
                        bandwidth=DEFAULT_BANDWIDTH,
                        hard_anomaly_th=hard_th,
                        mode=mode,
                    )
                    candidates.append((off, da))

                best_off, best_da = max(candidates, key=lambda c: (c[0], -c[1]))
                x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
                local_mask = comp["local_mask"]
                local_patch = overlay[y:y+h, x:x+w]
                reg_scores = local_patch[local_mask]
                max_s = float(np.max(reg_scores)) if reg_scores.size else 1.0
                weight = (reg_scores / max_s) if max_s > 1e-8 else 1.0
                local_patch[local_mask] = np.clip(reg_scores + best_off * weight, 0.0, None)

            adj_scores[rec_idx] = float(training_image_score(overlay))

            if m["is_bad"] and m["bad_pos"] >= 0:
                bad_overlays_256[m["bad_pos"]] = cv2.resize(
                    overlay, dataset.target_size, interpolation=cv2.INTER_LINEAR
                )

        p_aupro = dataset.compute_fast_aupro(bad_overlays_256, is_fast_path=True)

        # Region metrics: combine static + middle bad images
        image_miss_rates = list(dataset.static_miss_rates)
        image_coverages = list(dataset.static_coverages)
        total_fp = dataset.static_fp_region_count

        for pos in dataset.middle_bad_positions:
            sm = bad_overlays_256[pos]
            pred = (sm >= GOOD_THRESHOLD).astype(np.uint8)
            pred_lbl = measure.label(pred)
            pc = int(pred_lbl.max())

            rc, reg_masks = dataset.gt_region_meta[pos]
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

    else:
        # Full image pass for morphological opening / BG floor subtraction
        for i, rec in enumerate(dataset.records):
            smap = rec["score_map"].copy()
            raw_s = rec["raw_score"]
            is_middle = (GOOD_THRESHOLD <= raw_s <= ANOMALY_THRESHOLD)

            if is_middle:
                m_item = next((m for m in dataset.middle_data if m["record_idx"] == i), None)
                if m_item is not None:
                    for comp in m_item["comps"]:
                        candidates = []
                        for p in comp["patches"]:
                            dg, da = p["dg"], p["da"]
                            off, conf, dec = compute_asymmetric_margin_offset(
                                dg=dg, da=da,
                                tau_g=tau_g, tau_a=tau_a,
                                delta_g=delta_g, delta_a=delta_a,
                                scale_g=scale_g, scale_a=scale_a,
                                gamma_g=gamma_g, gamma_a=gamma_a,
                                bandwidth=DEFAULT_BANDWIDTH,
                                hard_anomaly_th=hard_th,
                                mode=mode,
                            )
                            candidates.append((off, da))

                        best_off, best_da = max(candidates, key=lambda c: (c[0], -c[1]))
                        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
                        local_mask = comp["local_mask"]
                        local_patch = smap[y:y+h, x:x+w]
                        reg_scores = local_patch[local_mask]
                        max_s = float(np.max(reg_scores)) if reg_scores.size else 1.0
                        weight = (reg_scores / max_s) if max_s > 1e-8 else 1.0
                        local_patch[local_mask] = np.clip(reg_scores + best_off * weight, 0.0, None)

            if morph_k > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
                smap = cv2.morphologyEx(smap, cv2.MORPH_OPEN, kernel)
            if bg_floor > 0.0:
                p_val = float(np.percentile(smap, bg_floor))
                smap = np.maximum(smap - p_val, 0.0)

            adj_scores[i] = float(training_image_score(smap))
            if rec["dataset_label"] != "good":
                pos = dataset.bad_idx_to_pos[i]
                bad_overlays_256[pos] = cv2.resize(
                    smap, dataset.target_size, interpolation=cv2.INTER_LINEAR
                )

        p_aupro = dataset.compute_fast_aupro(bad_overlays_256, is_fast_path=False)

        image_miss_rates = []
        image_coverages = []
        total_fp = 0

        for pos, (rc, reg_masks) in enumerate(dataset.gt_region_meta):
            sm = bad_overlays_256[pos]
            pred = (sm >= GOOD_THRESHOLD).astype(np.uint8)
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

    # Fast metrics computation
    labels = dataset.labels
    i_auroc = safe_auroc(labels, adj_scores)
    i_ap = safe_ap(labels, adj_scores)
    i_f1 = max_f1(labels, adj_scores)

    # Ultra-fast GPU metrics for pixel AUROC and AP
    pix_scores_gpu = torch.from_numpy(bad_overlays_256.reshape(-1)).to(dataset.device)
    p_auroc = float(binary_auroc(pix_scores_gpu, dataset.pix_labels_gpu))
    p_ap = float(binary_average_precision(pix_scores_gpu, dataset.pix_labels_gpu))

    r_miss = float(np.mean(image_miss_rates)) if image_miss_rates else float("nan")
    r_cov = float(np.mean(image_coverages)) if image_coverages else float("nan")
    elapsed = time.time() - t0

    return {
        "Config_Name": config.get("name", "unnamed"),
        "Category": config.get("category", "General"),
        "tau_g": tau_g,
        "tau_a": tau_a,
        "delta_g": delta_g,
        "delta_a": delta_a,
        "scale_g": scale_g,
        "scale_a": scale_a,
        "gamma_g": gamma_g,
        "gamma_a": gamma_a,
        "mode": mode,
        "hard_th": config.get("hard_th", 0.0),
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
# 4. Build Comprehensive Experiment Suite
# =========================================================================

def build_experiment_matrix() -> List[Dict[str, Any]]:
    experiments = []

    # ---------------------------------------------------------------------
    # Group 0: Baselines
    # ---------------------------------------------------------------------
    experiments.append({
        "name": "00_Raw_SingleStage_Baseline",
        "category": "0_Baseline",
        "mode": "linear",
        "scale_g": 0.0,
        "scale_a": 0.0,
    })
    experiments.append({
        "name": "01_Standard_Linear_Margin_Baseline",
        "category": "0_Baseline",
        "mode": "linear",
        "scale_g": 1.0,
        "scale_a": 1.0,
    })
    experiments.append({
        "name": "02_Hard_Anomaly_Trigger_Baseline (da<=0.15)",
        "category": "0_Baseline",
        "mode": "linear",
        "hard_th": 0.15,
        "scale_g": 1.0,
        "scale_a": 1.0,
    })

    # ---------------------------------------------------------------------
    # Group 1: Temperature Parameter Matrix (tau_g in [0.05, 0.1, 0.2], tau_a in [0.02, 0.05, 0.1])
    # ---------------------------------------------------------------------
    tau_g_list = [0.05, 0.10, 0.20]
    tau_a_list = [0.02, 0.05, 0.10]

    for tg in tau_g_list:
        for ta in tau_a_list:
            experiments.append({
                "name": f"10_Temp_Grid (tau_g={tg:.2f}, tau_a={ta:.2f})",
                "category": "1_Temperature_Matrix",
                "tau_g": tg,
                "tau_a": ta,
                "mode": "branch_sigmoid",
            })

    # Finer sweep for boundary analysis
    for tg in [0.03, 0.08, 0.15, 0.30]:
        experiments.append({
            "name": f"11_Temp_Fine_TauG (tau_g={tg:.2f}, tau_a=0.05)",
            "category": "1_Temperature_Matrix",
            "tau_g": tg,
            "tau_a": 0.05,
            "mode": "branch_sigmoid",
        })
    for ta in [0.01, 0.03, 0.08, 0.15]:
        experiments.append({
            "name": f"11_Temp_Fine_TauA (tau_g=0.20, tau_a={ta:.2f})",
            "category": "1_Temperature_Matrix",
            "tau_g": 0.20,
            "tau_a": ta,
            "mode": "branch_sigmoid",
        })

    # ---------------------------------------------------------------------
    # Group 2: Independent Soft Thresholding / Deadband (delta_g, delta_a)
    # ---------------------------------------------------------------------
    for dg in [0.0, 0.02, 0.05, 0.10, 0.15]:
        for da in [0.0, 0.02, 0.05, 0.10]:
            experiments.append({
                "name": f"20_Soft_Deadband (delta_g={dg:.2f}, delta_a={da:.2f})",
                "category": "2_Soft_Threshold",
                "tau_g": 0.20,
                "tau_a": 0.10,
                "delta_g": dg,
                "delta_a": da,
                "mode": "branch_sigmoid",
            })

    # ---------------------------------------------------------------------
    # Group 3: Asymmetric Scaling & Non-linear Power Shaping (scale_g, scale_a, gamma)
    # ---------------------------------------------------------------------
    for sg in [0.5, 0.8, 1.0, 1.5, 2.0]:
        for sa in [1.0, 1.5, 2.0, 2.5]:
            experiments.append({
                "name": f"30_Asym_Scale (scale_g={sg:.1f}, scale_a={sa:.1f})",
                "category": "3_Asymmetric_Scaling",
                "tau_g": 0.20,
                "tau_a": 0.10,
                "delta_g": 0.05,
                "delta_a": 0.05,
                "scale_g": sg,
                "scale_a": sa,
                "mode": "branch_sigmoid",
            })

    for gg in [0.5, 1.0, 1.5, 2.0]:
        for ga in [0.5, 1.0, 1.5, 2.0]:
            experiments.append({
                "name": f"31_Power_Shaping (gamma_g={gg:.1f}, gamma_a={ga:.1f})",
                "category": "3_Asymmetric_Scaling",
                "tau_g": 0.20,
                "tau_a": 0.10,
                "delta_g": 0.05,
                "delta_a": 0.05,
                "scale_g": 0.5,
                "scale_a": 2.0,
                "gamma_g": gg,
                "gamma_a": ga,
                "mode": "branch_sigmoid",
            })

    # ---------------------------------------------------------------------
    # Group 4: Logit Difference Formulation
    # ---------------------------------------------------------------------
    for tg in [0.10, 0.20]:
        for ta in [0.05, 0.10]:
            experiments.append({
                "name": f"40_Logit_Diff_Sigmoid (tau_g={tg:.2f}, tau_a={ta:.2f})",
                "category": "4_Logit_Difference",
                "tau_g": tg,
                "tau_a": ta,
                "mode": "logit_diff",
            })

    # ---------------------------------------------------------------------
    # Group 5: Combined Optimal Strategies
    # ---------------------------------------------------------------------
    experiments.append({
        "name": "50_Combined_Opt1 (tau=(0.20,0.10), delta=(0.05,0.05), scale=(0.5,2.0))",
        "category": "5_Combined_Optimal",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "delta_g": 0.05,
        "delta_a": 0.05,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "mode": "branch_sigmoid",
    })

    experiments.append({
        "name": "51_Combined_Opt2 (+ Hard Trigger da<=0.15)",
        "category": "5_Combined_Optimal",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "delta_g": 0.05,
        "delta_a": 0.05,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "mode": "branch_sigmoid",
    })

    experiments.append({
        "name": "52_Combined_Opt3 (+ Morph Open k=3)",
        "category": "5_Combined_Optimal",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "delta_g": 0.05,
        "delta_a": 0.05,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "morph_k": 3,
        "mode": "branch_sigmoid",
    })

    experiments.append({
        "name": "53_Combined_Opt4 (+ BG Floor p=15%)",
        "category": "5_Combined_Optimal",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "delta_g": 0.05,
        "delta_a": 0.05,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "bg_floor": 15.0,
        "mode": "branch_sigmoid",
    })

    experiments.append({
        "name": "54_Combined_Opt5_SOTA_Pipeline (+ BG Floor 15% + Morph Open 3)",
        "category": "5_Combined_Optimal",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "delta_g": 0.05,
        "delta_a": 0.05,
        "scale_g": 0.5,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "bg_floor": 15.0,
        "morph_k": 3,
        "mode": "branch_sigmoid",
    })

    experiments.append({
        "name": "55_Combined_Opt6_UltraLowFP (+ BG Floor 20% + Morph Open 3)",
        "category": "5_Combined_Optimal",
        "tau_g": 0.20,
        "tau_a": 0.10,
        "delta_g": 0.05,
        "delta_a": 0.05,
        "scale_g": 1.0,
        "scale_a": 2.0,
        "hard_th": 0.15,
        "bg_floor": 20.0,
        "morph_k": 3,
        "mode": "branch_sigmoid",
    })

    return experiments


# =========================================================================
# 5. Main Execution Flow
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Track 2: Asymmetric Confidence Scaling & Margin Mapping")
    parser.add_argument("--gpu", type=int, default=2, help="GPU device index (default: 2)")
    parser.add_argument("--root", default="/data/wt/two_stages/base_672_15k", help="Base 672 feature bank root")
    parser.add_argument("--output_csv", default="/data/wt/two_stages/base_672_15k/track2_asymmetric_margin_results.csv", help="Output CSV path")
    parser.add_argument("--output_json", default="/data/wt/two_stages/base_672_15k/track2_asymmetric_margin_summary.json", help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("=" * 115, flush=True)
    print(" TRACK 2: ASYMMETRIC CONFIDENCE SCALING & MARGIN MAPPING (672 MODEL)", flush=True)
    print("=" * 115, flush=True)
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

    dataset = FastBenchmarkDataset(records, device=device, target_size=(256, 256))
    dataset.precompute_middle_components(good_lib, anomaly_lib, min_area_pct=0.005, top_k=3, query_patches=3)

    experiments = build_experiment_matrix()
    print(f"\nTotal experimental configurations to evaluate: {len(experiments)}", flush=True)

    print("\n" + "=" * 125, flush=True)
    header = (
        f"{'Category':<22} {'Config Name':<42} {'I-AUROC':>8} {'I-AP':>8} "
        f"{'P-AUROC':>8} {'P-AP':>8} {'P-AUPRO':>8} {'Miss%':>7} {'FP-Count':>9} {'Time':>6}"
    )
    print(header, flush=True)
    print("=" * 125, flush=True)

    results: List[Dict[str, Any]] = []
    for exp in experiments:
        res = evaluate_margin_configuration(dataset, exp)
        results.append(res)
        row_str = (
            f"{res['Category']:<22} {res['Config_Name']:<42} "
            f"{res['I-AUROC']:>8.4f} {res['I-AP']:>8.4f} {res['P-AUROC']:>8.4f} "
            f"{res['P-AP']:>8.4f} {res['P-AUPRO']:>8.4f} {res['R-MissRate']*100:>6.2f}% "
            f"{res['R-FP-RegionCount']:>9d} {res['Elapsed_s']:>5.2f}s"
        )
        print(row_str, flush=True)

    print("=" * 125, flush=True)

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
        "title": "Track 2: Asymmetric Confidence Scaling & Margin Mapping Quantitative Results",
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
