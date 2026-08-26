#!/usr/bin/env python3
"""Track 2: Hard-Negative Bank & Edge Noise Suppression.

This script implements:
1. Extraction of Hard-Negative (high-scoring normal edge & chamfer) features from good training/bank data.
2. Construction of FAISS Hard-Negative index.
3. Three-bank decision mechanism (Good Bank, Hard-Negative Bank, Anomaly Bank):
   - Hard Anomaly Trigger: d_ano <= 0.15 -> Anomaly.
   - Normal/Hard-Negative Suppression: d_neg < d_ano or d_good < d_ano -> Strongly suppress background score.
   - Moderate Anomaly: d_ano < min(d_good, d_neg) -> positive offset.
4. Comprehensive full-set evaluation on 680 test images for all metrics:
   - R-FP-RegionCount, R-MissRate, R-PixelCoverage, P-AP, P-AUPRO, P-AUROC, I-AUROC, I-AP, I-F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from skimage import measure
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(1, str(_UTILS_DIR))

from anomaly_evaluation import (
    safe_auroc,
    safe_ap,
    max_f1,
    safe_aupro,
    pixel_f1_score_and_threshold,
    training_image_score,
)
from dinomaly_two_stage import (
    add_model_arguments,
    build_transform,
    calculate_distance_offset,
    connected_components,
    dilate_mask,
    infer_image,
    iter_image_paths,
    linear_patch_geometry,
    linear_score_to_feature,
    load_dinomaly_model,
    load_feature_library,
    load_mask,
    l2_normalize,
    mask_bbox,
    search_library,
    search_library_topk,
    select_device,
    select_patch_positions,
)
from utils import refine_anomaly_map_guided

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030


def fast_region_detection_metrics(
    masks: np.ndarray,
    score_maps: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Optimized region detection metrics mathematically identical to anomaly_evaluation."""
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
            region_mask = gt_labels == region_id
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


def load_all_cached_test_data(
    root: Path,
    data_root: Path,
    gt_dir: Path,
) -> List[Dict[str, Any]]:
    """Load all 680 test images, score maps, GT masks, and features into memory."""
    cache_pkl = root / "preds" / "cached_eval_records.pkl"
    if cache_pkl.is_file():
        print(f"Loading cached records from {cache_pkl}...")
        t0 = time.time()
        import pickle

        with open(cache_pkl, "rb") as f:
            records = pickle.load(f)
        print(f"Loaded {len(records)} test images in {time.time() - t0:.2f}s.")
        return records

    print("Loading test data and cached score maps / features from individual files...")
    t0 = time.time()
    with open(root / "preds" / "run.json") as f:
        run_data = json.load(f)

    records = []
    target_metric_size = (256, 256)

    for r in tqdm(run_data["results"], desc="Loading cache", unit="image"):
        rel = Path(r["image_relative"])
        is_bad = r["dataset_label"] != "good"
        score_path = root / "preds" / "score_maps" / rel.with_suffix(".npy")
        feat_path = root / "preds" / "features" / rel.with_suffix(".npy")

        score_map = np.load(score_path)
        feature = np.load(feat_path)

        gt_mask_256 = None
        if is_bad:
            gt_path = None
            for sfx in [".png", ".jpg", ".jpeg", ".npy", ".tif", ".json"]:
                cand = gt_dir / rel.with_suffix(sfx)
                if cand.is_file():
                    gt_path = cand
                    break
            if gt_path is not None:
                gt_full = load_mask(gt_path, score_map.shape[:2])
                gt_mask_256 = cv2.resize(gt_full.astype(np.uint8), target_metric_size, interpolation=cv2.INTER_NEAREST)
                gt_mask_256 = (gt_mask_256 > 0).astype(np.uint8)
            else:
                gt_mask_256 = np.zeros(target_metric_size, dtype=np.uint8)

        records.append({
            "image_relative": str(rel),
            "dataset_label": r["dataset_label"],
            "raw_score": float(r["raw_score"]),
            "initial_label": r["initial_label"],
            "score_map": score_map,
            "feature": feature,
            "original_shape": score_map.shape[:2],
            "gt_mask_256": gt_mask_256,
        })

    print(f"Loaded {len(records)} test images in {time.time() - t0:.2f}s.")
    return records


class FeatureBank:
    """Wrapper around FAISS FlatL2 index."""

    def __init__(self, vectors: np.ndarray, normalize: bool = True):
        self.vectors = vectors.astype(np.float32)
        if normalize:
            norms = np.linalg.norm(self.vectors, axis=-1, keepdims=True)
            self.vectors = self.vectors / np.maximum(norms, 1e-8)
        self.dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.vectors)
        self.normalize = normalize
        self.metadata = {"patch_top_ratio": 0.5, "normalize": normalize}

    def search(self, query_vec: np.ndarray, k: int = 1) -> Tuple[float, int]:
        vec = query_vec.reshape(1, -1).astype(np.float32)
        if self.normalize:
            vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
        dists, indices = self.index.search(vec, k)
        return float(dists[0][0]), int(indices[0][0])

    def search_topk(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[float, int]]:
        vec = query_vec.reshape(1, -1).astype(np.float32)
        if self.normalize:
            vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
        k = min(k, self.index.ntotal)
        dists, indices = self.index.search(vec, k)
        return [(float(d), int(i)) for d, i in zip(dists[0], indices[0])]


def infer_all_images(
    image_paths: List[Path],
    model: torch.nn.Module,
    transform: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run model on images once and cache (image_path, score_map, feat_chw, pil_img)."""
    results = []
    print(f"Running inference on {len(image_paths)} images to cache features...")
    t0 = time.time()
    for img_path in tqdm(image_paths, desc="Inference", unit="image"):
        score_map, feat_chw = infer_image(model, img_path, transform, device)
        results.append({
            "image_path": img_path,
            "score_map": score_map,
            "feat_chw": feat_chw,
        })
    print(f"Cached features for {len(results)} images in {time.time() - t0:.2f}s.")
    return results


def extract_hard_negatives_from_cache(
    cached_items: List[Dict[str, Any]],
    strategy: str = "threshold",
    score_threshold: float = 0.014,
    top_k_per_image: int = 10,
    edge_only: bool = False,
    sobel_threshold: float = 40.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract hard negative patch feature vectors from pre-inferred cache."""
    vectors = []
    metadata_list = []

    for item in cached_items:
        img_path = item["image_path"]
        score_map = item["score_map"]
        feat_chw = item["feat_chw"]
        H_f, W_f = feat_chw.shape[-2:]

        score_feat = linear_score_to_feature(score_map, (H_f, W_f))

        edge_feat_mask = np.ones((H_f, W_f), dtype=bool)
        if edge_only:
            pil_img = Image.open(img_path).convert("L")
            gray = np.array(pil_img)
            gray_resized = cv2.resize(gray, (W_f, H_f))
            sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            edge_feat_mask = grad_mag >= sobel_threshold

        candidate_patches = []
        for r in range(H_f):
            for c in range(W_f):
                s = score_feat[r, c]
                is_edge = bool(edge_feat_mask[r, c])
                if edge_only and not is_edge:
                    continue
                candidate_patches.append((s, r, c, is_edge))

        candidate_patches.sort(key=lambda x: -x[0])

        selected = []
        if strategy == "threshold":
            selected = [p for p in candidate_patches if p[0] >= score_threshold]
            if top_k_per_image > 0 and len(selected) > top_k_per_image:
                selected = selected[:top_k_per_image]
        elif strategy == "top_k":
            selected = candidate_patches[:top_k_per_image]
        elif strategy == "edge_peak":
            edge_candidates = [p for p in candidate_patches if p[3]]
            selected = edge_candidates[:top_k_per_image]

        for s, r, c, is_edge in selected:
            vec = feat_chw[:, r, c]
            norm_vec = l2_normalize(vec)
            vectors.append(norm_vec)
            metadata_list.append({
                "image": str(img_path.name),
                "row": r,
                "col": c,
                "score": float(s),
                "is_edge": is_edge,
            })

    if not vectors:
        raise RuntimeError(f"No vectors extracted with strategy={strategy}")
    return np.stack(vectors).astype(np.float32), metadata_list


def extract_candidate_regions(
    score_map: np.ndarray,
    good_threshold: float = GOOD_THRESHOLD,
    anomaly_threshold: float = ANOMALY_THRESHOLD,
    min_area_pct: float = 0.0,
    morph_open_k: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Extract candidate ROI connected components."""
    binary = (score_map >= good_threshold).astype(np.uint8)

    if morph_open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_k, morph_open_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    min_area = 1
    if min_area_pct > 0.0:
        min_area = max(1, int(round(min_area_pct / 100.0 * score_map.size)))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    components = []

    for comp_id in range(1, count):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

        local_labels = labels[y : y + h, x : x + w]
        local_mask = local_labels == comp_id

        local_scores = score_map[y : y + h, x : x + w]
        peak_score = float(local_scores[local_mask].max()) if local_mask.any() else 0.0
        cx, cy = centroids[comp_id]
        components.append({
            "component_id": int(comp_id),
            "area": area,
            "bbox": (x, y, x + w, y + h),
            "local_mask": local_mask,
            "centroid": (float(cx), float(cy)),
            "peak_score": peak_score,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    candidate_mask = np.zeros(score_map.shape, dtype=np.uint8)
    for c in components:
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        candidate_mask[y : y + h, x : x + w][c["local_mask"]] = 1

    return components, candidate_mask


def run_three_bank_prediction(
    score_map: np.ndarray,
    feature: np.ndarray,
    good_bank: Any,
    neg_bank: Optional[Any],
    anomaly_bank: Any,
    components: List[Dict[str, Any]],
    candidate_mask: np.ndarray,
    knn_k: int = 3,
    query_patches: int = 3,
    hard_ano_threshold: Optional[float] = 0.15,
    suppression_factor: float = 1.0,
    guided_filter: bool = False,
    img_bgr: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, int]:
    """Execute Three-Bank Decision Mechanism on one image."""
    raw_score = float(training_image_score(score_map))

    if raw_score < GOOD_THRESHOLD:
        return raw_score, score_map.copy(), 0
    if raw_score > ANOMALY_THRESHOLD:
        return raw_score, score_map.copy(), 0

    if not components:
        return raw_score, score_map.copy(), 0

    height, width = score_map.shape[:2]
    feature_shape = feature.shape[-2:]
    feature_height, feature_width = feature_shape
    regions = []

    bandwidth = ANOMALY_THRESHOLD - GOOD_THRESHOLD

    for comp in components:
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        local_mask = comp["local_mask"]

        # Map ROI to feature cells
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

        score_feature = linear_score_to_feature(score_map, feature_shape)
        patch_ratio = float(good_bank.metadata.get("patch_top_ratio", 0.5))
        positions = select_patch_positions(score_feature, mask_feature, patch_ratio)
        if positions.shape[0] == 0:
            continue
        if query_patches > 0:
            positions = positions[:query_patches]

        patch_candidates = []
        for r, c in positions:
            p_vec = feature[:, int(r), int(c)]
            p_vec = l2_normalize(p_vec)

            # Query Good Bank
            if knn_k > 1:
                g_m = search_library_topk(good_bank, p_vec, top_k=knn_k) if hasattr(good_bank, "index") else good_bank.search_topk(p_vec, k=knn_k)
                g_dist = float(np.mean([m[0] for m in g_m])) if g_m else 1.0
            else:
                g_dist, _ = search_library(good_bank, p_vec) if hasattr(good_bank, "index") else good_bank.search(p_vec)

            # Query Anomaly Bank
            if knn_k > 1:
                a_m = search_library_topk(anomaly_bank, p_vec, top_k=knn_k) if hasattr(anomaly_bank, "index") else anomaly_bank.search_topk(p_vec, k=knn_k)
                a_dist = float(np.mean([m[0] for m in a_m])) if a_m else 1.0
            else:
                a_dist, _ = search_library(anomaly_bank, p_vec) if hasattr(anomaly_bank, "index") else anomaly_bank.search(p_vec)

            # Query Hard-Negative Bank if available
            if neg_bank is not None:
                if knn_k > 1:
                    n_m = search_library_topk(neg_bank, p_vec, top_k=knn_k) if hasattr(neg_bank, "index") else neg_bank.search_topk(p_vec, k=knn_k)
                    n_dist = float(np.mean([m[0] for m in n_m])) if n_m else 1.0
                else:
                    n_dist, _ = search_library(neg_bank, p_vec) if hasattr(neg_bank, "index") else neg_bank.search(p_vec)
            else:
                n_dist = float("inf")

            # --- Three-Bank Decision Mechanism ---
            # 1. Hard Anomaly Trigger: d_ano <= 0.15 -> judge anomaly directly
            if hard_ano_threshold is not None and a_dist <= float(hard_ano_threshold):
                offset = (bandwidth / 2.0)
                patch_candidates.append({
                    "d_good": g_dist,
                    "d_neg": n_dist,
                    "d_ano": a_dist,
                    "signed_offset": float(offset),
                    "decision": "hard_anomaly",
                    "row": int(r),
                    "col": int(c),
                })
                continue

            # 2. Triple-bank comparison: d_normal = min(d_good, d_neg)
            d_norm = min(g_dist, n_dist)

            if d_norm < a_dist:
                # Hits Good or Hard-Negative bank -> Strong suppression
                denom = d_norm + a_dist + 1e-8
                margin = (a_dist - d_norm) / denom
                offset = -(bandwidth / 2.0) * margin * suppression_factor
                hit_bank = "hard_neg" if n_dist < g_dist else "good"
                patch_candidates.append({
                    "d_good": g_dist,
                    "d_neg": n_dist,
                    "d_ano": a_dist,
                    "signed_offset": float(offset),
                    "decision": hit_bank,
                    "row": int(r),
                    "col": int(c),
                })
            else:
                # Closer to Anomaly bank (moderate anomaly)
                denom = d_norm + a_dist + 1e-8
                margin = (d_norm - a_dist) / denom
                offset = (bandwidth / 2.0) * margin
                patch_candidates.append({
                    "d_good": g_dist,
                    "d_neg": n_dist,
                    "d_ano": a_dist,
                    "signed_offset": float(offset),
                    "decision": "anomaly",
                    "row": int(r),
                    "col": int(c),
                })

        if not patch_candidates:
            continue

        best = max(
            patch_candidates,
            key=lambda p: (
                float(p["signed_offset"]),
                -float(p["d_ano"]),
            ),
        )

        regions.append({
            "component_id": comp["component_id"],
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "local_mask": local_mask,
            "signed_offset": float(best["signed_offset"]),
            "decision": best["decision"],
        })

    overlay = score_map.copy()
    for reg in regions:
        signed_off = reg["signed_offset"]
        x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
        local_mask = reg["local_mask"]
        local_patch = overlay[y : y + h, x : x + w]
        max_s = float(np.max(local_patch[local_mask])) if local_mask.any() else 1.0
        weight = (local_patch / max_s) if max_s > 1e-8 else 1.0
        local_patch[local_mask] = np.clip(local_patch[local_mask] + signed_off * weight[local_mask], 0.0, None)

    if guided_filter and img_bgr is not None:
        overlay = refine_anomaly_map_guided(img_bgr, overlay, radius=4, eps=1e-3)

    adj_score = float(training_image_score(overlay)) if overlay.size else raw_score
    return adj_score, overlay, len(regions)


def evaluate_records(
    records: List[Dict[str, Any]],
    good_bank: Any,
    neg_bank: Optional[Any],
    anomaly_bank: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate full set of 680 records with specified config."""
    t0 = time.time()
    adj_scores = []
    labels = []
    bad_overlays_256 = []
    bad_gt_masks_256 = []
    total_rois = 0
    target_metric_size = (256, 256)

    min_area_pct = config.get("min_area_pct", 0.0)
    morph_open_k = config.get("morph_open_k", 0)
    hard_ano_th = config.get("hard_ano_threshold", 0.15)
    suppress_fac = config.get("suppression_factor", 1.0)
    knn_k = config.get("knn_k", 3)
    query_patches = config.get("query_patches", 3)
    use_guided = config.get("guided_filter", False)

    for rec in records:
        score_map = rec["score_map"]
        feature = rec["feature"]
        is_bad = rec["dataset_label"] != "good"
        labels.append(1 if is_bad else 0)

        raw_score = rec["raw_score"]
        if raw_score < GOOD_THRESHOLD or raw_score > ANOMALY_THRESHOLD:
            adj_scores.append(raw_score)
            if is_bad:
                bad_overlays_256.append(cv2.resize(score_map, target_metric_size, interpolation=cv2.INTER_LINEAR))
                bad_gt_masks_256.append(rec["gt_mask_256"])
            continue

        components, candidate_mask = extract_candidate_regions(
            score_map=score_map,
            good_threshold=GOOD_THRESHOLD,
            anomaly_threshold=ANOMALY_THRESHOLD,
            min_area_pct=min_area_pct,
            morph_open_k=morph_open_k,
        )

        adj_score, overlay, roi_count = run_three_bank_prediction(
            score_map=score_map,
            feature=feature,
            good_bank=good_bank,
            neg_bank=neg_bank,
            anomaly_bank=anomaly_bank,
            components=components,
            candidate_mask=candidate_mask,
            knn_k=knn_k,
            query_patches=query_patches,
            hard_ano_threshold=hard_ano_th,
            suppression_factor=suppress_fac,
            guided_filter=use_guided,
        )

        total_rois += roi_count
        adj_scores.append(adj_score)
        if is_bad:
            bad_overlays_256.append(cv2.resize(overlay, target_metric_size, interpolation=cv2.INTER_LINEAR))
            bad_gt_masks_256.append(rec["gt_mask_256"])

    elapsed = time.time() - t0

    # Image metrics
    labels = np.array(labels, dtype=np.uint8)
    adj_scores = np.array(adj_scores, dtype=np.float32)
    i_auroc = safe_auroc(labels, adj_scores)
    i_ap = safe_ap(labels, adj_scores)
    i_f1 = max_f1(labels, adj_scores)

    # Pixel & Region metrics
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
        "Config_Name": config.get("name", "unnamed"),
        "Bank_Config": config.get("bank_desc", ""),
        "Total_ROIs": total_rois,
        "R-FP-RegionCount": reg_metrics["R-FP-RegionCount"],
        "R-MissRate": reg_metrics["R-MissRate"],
        "R-PixelCoverage": reg_metrics["R-PixelCoverage"],
        "R-FPR": reg_metrics["R-FPR"],
        "P-AP": p_ap,
        "P-AUPRO": p_aupro,
        "P-AUROC": p_auroc,
        "P-F1": p_f1,
        "I-AUROC": i_auroc,
        "I-AP": i_ap,
        "I-F1": i_f1,
        "Elapsed_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2, help="GPU ID to use (2 or 3)")
    parser.add_argument("--root", default="/data/wt/two_stages/base_patch_672")
    parser.add_argument("--model_path", default="/data/wt/trainlogs/leishi_026/Dinomaly/default/20260805233425/model.pth")
    parser.add_argument("--data_root", default="/data/wt/ramdisk/leishi_026/test")
    parser.add_argument("--train_good_root", default="/data/wt/ramdisk/leishi_026/train/good")
    parser.add_argument("--export_images_root", default="/data/wt/ramdisk/test_export_03/images")
    parser.add_argument("--ground_truth", default="/data/wt/ramdisk/leishi_026/ground_truth")
    parser.add_argument("--output_dir", default="/data/wt/two_stages/track2_hard_negative_results")
    args = parser.parse_args()

    device = select_device(args.gpu)
    print(f"=== Track 2: Hard-Negative Bank & Edge Noise Suppression ===")
    print(f"Running on Device: {device}")

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline Good and Anomaly Libraries
    good_lib = load_feature_library(root / "good", device, True)
    anomaly_lib = load_feature_library(root / "anomaly", device, True)
    print(f"Base Good Bank: {good_lib.index.ntotal} vectors | Anomaly Bank: {anomaly_lib.index.ntotal} vectors")

    # 2. Load 680 test cached records
    records = load_all_cached_test_data(root, Path(args.data_root), Path(args.ground_truth))

    # 3. Build Model & Transform for feature extraction
    sub_parser = argparse.ArgumentParser()
    add_model_arguments(sub_parser)
    model_args = sub_parser.parse_args([
        "--model", args.model_path,
        "--backbone", "dinov2reg_vit_base_14",
        "--image_size", "672",
        "--crop_size", "672",
        "--gpu", str(args.gpu),
    ])
    model = load_dinomaly_model(model_args, device)
    transform = build_transform(model_args)

    # 4. Extract Hard Negative Feature Banks
    # Pre-infer training images once
    train_good_paths = sorted(iter_image_paths(Path(args.train_good_root)))
    cached_train_good = infer_all_images(train_good_paths, model, transform, device)

    # Bank A1: Threshold score >= 0.014 on Train Good (top 20 per image)
    vecs_a1, meta_a1 = extract_hard_negatives_from_cache(
        cached_train_good,
        strategy="threshold", score_threshold=0.014, top_k_per_image=20, edge_only=False
    )
    bank_a1 = FeatureBank(vecs_a1, normalize=True)
    print(f"Bank A1 (Train Good Score>=0.014, top 20): {bank_a1.index.ntotal} vectors")

    # Bank A2: Edge/Chamfer Specific on Train Good (Sobel edge + high score)
    vecs_a2, meta_a2 = extract_hard_negatives_from_cache(
        cached_train_good,
        strategy="edge_peak", top_k_per_image=10, edge_only=True, sobel_threshold=40.0
    )
    bank_a2 = FeatureBank(vecs_a2, normalize=True)
    print(f"Bank A2 (Train Good Edge/Chamfer Sobel, top 10): {bank_a2.index.ntotal} vectors")

    # Bank A3: Top-5 Peak False Positive patches per train good image
    vecs_a3, meta_a3 = extract_hard_negatives_from_cache(
        cached_train_good,
        strategy="top_k", top_k_per_image=5, edge_only=False
    )
    bank_a3 = FeatureBank(vecs_a3, normalize=True)
    print(f"Bank A3 (Train Good Top-5 Peaks): {bank_a3.index.ntotal} vectors")

    # Bank A4: All patches score >= 0.014 (Dense coverage)
    vecs_a4, meta_a4 = extract_hard_negatives_from_cache(
        cached_train_good,
        strategy="threshold", score_threshold=0.014, top_k_per_image=0, edge_only=False
    )
    bank_a4 = FeatureBank(vecs_a4, normalize=True)
    print(f"Bank A4 (Train Good All Score>=0.014): {bank_a4.index.ntotal} vectors")

    # Bank B: Bank Building images (test_export_03)
    export_paths = sorted(iter_image_paths(Path(args.export_images_root)))
    cached_export = infer_all_images(export_paths, model, transform, device)
    vecs_b, meta_b = extract_hard_negatives_from_cache(
        cached_export,
        strategy="threshold", score_threshold=0.014, top_k_per_image=10, edge_only=False
    )
    bank_b = FeatureBank(vecs_b, normalize=True)
    print(f"Bank B (Export Images Score>=0.014): {bank_b.index.ntotal} vectors")

    # Bank C: Combined Train Good + Export Hard Negatives
    vecs_c = np.concatenate([vecs_a1, vecs_b], axis=0)
    bank_c = FeatureBank(vecs_c, normalize=True)
    print(f"Bank C (Combined Train A1 + Export B): {bank_c.index.ntotal} vectors")

    # 5. Define Experimental Matrix
    experiments = []

    # --- Baselines ---
    experiments.append({
        "name": "1. Baseline (Raw Single-Stage)",
        "bank_desc": "No Stage 2",
        "hard_ano_threshold": None,
        "suppression_factor": 0.0,
        "use_neg_bank": None,
    })
    experiments.append({
        "name": "2. Baseline Two-Stage (Good + Anomaly 2-Bank)",
        "bank_desc": "Good + Anomaly",
        "hard_ano_threshold": None,
        "suppression_factor": 1.0,
        "use_neg_bank": None,
    })
    experiments.append({
        "name": "3. Two-Stage + Hard Anomaly Trigger (d_ano <= 0.15)",
        "bank_desc": "Good + Anomaly + HardAno(0.15)",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.0,
        "use_neg_bank": None,
    })

    # --- Three-Bank Decisions with Hard-Negative Banks ---
    experiments.append({
        "name": "4. Three-Bank: + Bank A1 (Train Score>=0.014)",
        "bank_desc": f"Good + Anomaly + HardNeg_A1({bank_a1.index.ntotal})",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.0,
        "use_neg_bank": bank_a1,
    })
    experiments.append({
        "name": "5. Three-Bank: + Bank A2 (Train Edge/Chamfer Sobel)",
        "bank_desc": f"Good + Anomaly + HardNeg_A2({bank_a2.index.ntotal})",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.0,
        "use_neg_bank": bank_a2,
    })
    experiments.append({
        "name": "6. Three-Bank: + Bank A3 (Train Top-5 Peaks)",
        "bank_desc": f"Good + Anomaly + HardNeg_A3({bank_a3.index.ntotal})",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.0,
        "use_neg_bank": bank_a3,
    })
    experiments.append({
        "name": "7. Three-Bank: + Bank A4 (Train Dense Score>=0.014)",
        "bank_desc": f"Good + Anomaly + HardNeg_A4({bank_a4.index.ntotal})",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.0,
        "use_neg_bank": bank_a4,
    })
    experiments.append({
        "name": "8. Three-Bank: + Bank B (Export High-Score)",
        "bank_desc": f"Good + Anomaly + HardNeg_B({bank_b.index.ntotal})",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.0,
        "use_neg_bank": bank_b,
    })
    experiments.append({
        "name": "9. Three-Bank: + Bank C (Combined Train+Export)",
        "bank_desc": f"Good + Anomaly + HardNeg_C({bank_c.index.ntotal})",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.0,
        "use_neg_bank": bank_c,
    })

    # --- Suppression Factor Ablation (on best Hard-Negative banks) ---
    for s_fac in [1.2, 1.5, 2.0]:
        experiments.append({
            "name": f"10. Three-Bank A1 + SuppressFactor={s_fac}",
            "bank_desc": f"HardNeg_A1 + Suppress({s_fac})",
            "hard_ano_threshold": 0.15,
            "suppression_factor": s_fac,
            "use_neg_bank": bank_a1,
        })

    for s_fac in [1.2, 1.5, 2.0]:
        experiments.append({
            "name": f"11. Three-Bank C + SuppressFactor={s_fac}",
            "bank_desc": f"HardNeg_C + Suppress({s_fac})",
            "hard_ano_threshold": 0.15,
            "suppression_factor": s_fac,
            "use_neg_bank": bank_c,
        })

    # --- Combined with Morphological Optimization ---
    experiments.append({
        "name": "12. Three-Bank A1 (Suppress 1.5) + MorphOpen(3)",
        "bank_desc": "HardNeg_A1 + Suppress(1.5) + MorphOpen(3)",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.5,
        "morph_open_k": 3,
        "use_neg_bank": bank_a1,
    })
    experiments.append({
        "name": "13. Three-Bank C (Combined) + Suppress(1.5) + MorphOpen(3)",
        "bank_desc": "HardNeg_C + Suppress(1.5) + MorphOpen(3)",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 1.5,
        "morph_open_k": 3,
        "use_neg_bank": bank_c,
    })
    experiments.append({
        "name": "14. Three-Bank C (Combined) + Suppress(2.0) + MorphOpen(3)",
        "bank_desc": "HardNeg_C + Suppress(2.0) + MorphOpen(3)",
        "hard_ano_threshold": 0.15,
        "suppression_factor": 2.0,
        "morph_open_k": 3,
        "use_neg_bank": bank_c,
    })

    # 6. Execute Evaluation Loop
    print("\n" + "=" * 135)
    header = (
        f"{'Experiment Name':<52} {'R-FP-Count':>10} {'R-Miss%':>8} {'P-AP':>8} "
        f"{'P-AUPRO':>8} {'P-AUROC':>8} {'I-AUROC':>8} {'I-AP':>8} {'I-F1':>8} {'Time':>6}"
    )
    print(header)
    print("=" * 135)

    results = []
    for exp in experiments:
        neg_b = exp.get("use_neg_bank")
        res = evaluate_records(records, good_lib, neg_b, anomaly_lib, exp)
        results.append(res)

        row_str = (
            f"{res['Config_Name']:<52} {int(res['R-FP-RegionCount']):>10} "
            f"{res['R-MissRate']*100:>7.2f}% {res['P-AP']:>8.4f} {res['P-AUPRO']:>8.4f} "
            f"{res['P-AUROC']:>8.4f} {res['I-AUROC']:>8.4f} {res['I-AP']:>8.4f} "
            f"{res['I-F1']:>8.4f} {res['Elapsed_s']:>5.1f}s"
        )
        print(row_str, flush=True)

    print("=" * 135)

    # Save to CSV and JSON
    out_csv = out_dir / "track2_hard_negative_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    out_json = out_dir / "track2_hard_negative_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults successfully exported to:\n  - CSV: {out_csv}\n  - JSON: {out_json}")


if __name__ == "__main__":
    main()
