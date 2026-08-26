#!/usr/bin/env python3
"""Track 2: Local Adaptive ROI Segmentation & Morphology Optimization.

Evaluates:
1. Morphological filtering (Opening with k=3, 5, 7, ellipse/rect)
2. Global and local background statistical adaptive thresholding
3. Min area percentage sweep (0.000%, 0.003%, 0.005%, 0.008%, 0.010%, 0.015%)
4. Region saliency / peak score filtering
5. Non-candidate noise suppression in final overlay
6. Combined optimal strategies

Runs on GPU 2, 3 using cached score maps and features.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
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
    calculate_distance_offset,
    connected_components,
    dilate_mask,
    linear_patch_geometry,
    linear_score_to_feature,
    load_feature_library,
    load_mask,
    l2_normalize,
    mask_bbox,
    nearest_feature_cell_from_polygon,
    patch_center_mask_from_polygons,
    polygon_points_from_mask,
    search_library,
    search_library_topk,
    select_patch_positions,
    select_strongest_region,
)

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
            region_mask = (gt_labels == region_id)
            covered_pixels = int(prediction[region_mask].sum())
            if covered_pixels:
                detected += 1
            region_coverages.append(covered_pixels / int(region_mask.sum()))
        
        missed = region_count - detected
        
        # Fast true positive / false positive calculation:
        # A predicted component is true positive iff it overlaps ANY GT defect pixel.
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


def extract_candidate_regions_adaptive(
    score_map: np.ndarray,
    good_threshold: float,
    anomaly_threshold: float,
    adaptive_mode: Optional[str] = None,
    adaptive_k: float = 2.0,
    morph_open_k: Optional[int] = None,
    morph_close_k: Optional[int] = None,
    morph_shape: str = "ellipse",
    min_area_pct: float = 0.0,
    min_peak_offset: float = 0.0,
    max_regions: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Extract candidate ROI connected components with local adaptive threshold & morphology."""
    
    # 1. Base binary mask
    if adaptive_mode == "global_stat":
        bg_scores = score_map[score_map < anomaly_threshold]
        if bg_scores.size > 100:
            mu_bg = float(np.mean(bg_scores))
            std_bg = float(np.std(bg_scores))
            effective_th = max(good_threshold, mu_bg + adaptive_k * std_bg)
        else:
            effective_th = good_threshold
        binary = (score_map >= effective_th).astype(np.uint8)
        
    elif adaptive_mode == "local_stat":
        local_mean = cv2.GaussianBlur(score_map, (31, 31), 5.0)
        local_sq_mean = cv2.GaussianBlur(score_map * score_map, (31, 31), 5.0)
        local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
        local_std = np.sqrt(local_var)
        local_th = np.maximum(good_threshold, local_mean + adaptive_k * local_std)
        binary = (score_map >= local_th).astype(np.uint8)
        
    elif adaptive_mode == "percentile":
        bg_scores = score_map[score_map < anomaly_threshold]
        if bg_scores.size > 100:
            p_val = float(np.percentile(bg_scores, adaptive_k))
            effective_th = max(good_threshold, p_val)
        else:
            effective_th = good_threshold
        binary = (score_map >= effective_th).astype(np.uint8)
    else:
        binary = (score_map >= good_threshold).astype(np.uint8)

    # 2. Morphological operations
    if morph_shape == "rect":
        elem_shape = cv2.MORPH_RECT
    elif morph_shape == "cross":
        elem_shape = cv2.MORPH_CROSS
    else:
        elem_shape = cv2.MORPH_ELLIPSE

    if morph_open_k is not None and morph_open_k > 1:
        k = int(morph_open_k)
        kernel = cv2.getStructuringElement(elem_shape, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if morph_close_k is not None and morph_close_k > 1:
        k = int(morph_close_k)
        kernel = cv2.getStructuringElement(elem_shape, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. Connected components with min_area filter
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
        
        local_labels = labels[y:y+h, x:x+w]
        local_mask = (local_labels == comp_id)
        
        # 4. Saliency / peak score filter
        local_scores = score_map[y:y+h, x:x+w]
        peak_score = float(local_scores[local_mask].max()) if local_mask.any() else 0.0
        if min_peak_offset > 0.0:
            if peak_score < (good_threshold + min_peak_offset):
                continue
                
        cx, cy = centroids[comp_id]
        components.append({
            "component_id": int(comp_id),
            "area": area,
            "bbox": (x, y, x + w, y + h),
            "local_mask": local_mask,
            "centroid": (float(cx), float(cy)),
            "peak_score": peak_score,
            "x": x, "y": y, "w": w, "h": h,
        })

    components.sort(key=lambda c: (-c["area"], c["component_id"]))
    if max_regions > 0:
        components = components[:max_regions]

    # Build candidate mask
    candidate_mask = np.zeros(score_map.shape, dtype=np.uint8)
    for c in components:
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        candidate_mask[y:y+h, x:x+w][c["local_mask"]] = 1

    return components, candidate_mask


def run_single_image_prediction(
    score_map: np.ndarray,
    feature: np.ndarray,
    good_library: Any,
    anomaly_library: Any,
    original_shape: Tuple[int, int],
    components: List[Dict[str, Any]],
    candidate_mask: np.ndarray,
    knn_k: int = 3,
    query_patches: int = 3,
    suppress_non_candidate_noise: bool = False,
) -> Tuple[float, np.ndarray, int]:
    """Execute Stage 2 ROI FAISS query and return (adjusted_score, overlay_map, roi_count)."""
    
    raw_score = float(training_image_score(score_map))
    
    if raw_score < GOOD_THRESHOLD:
        return raw_score, score_map.copy(), 0
    if raw_score > ANOMALY_THRESHOLD:
        return raw_score, score_map.copy(), 0

    # Middle band
    if not components:
        overlay = score_map.copy()
        if suppress_non_candidate_noise:
            overlay = np.minimum(overlay, GOOD_THRESHOLD - 1e-4)
        return float(training_image_score(overlay)), overlay, 0

    height, width = score_map.shape[:2]
    feature_shape = feature.shape[-2:]
    feature_height, feature_width = feature_shape
    regions = []

    for comp in components:
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        local_mask = comp["local_mask"]
        
        # Map ROI to feature cell mask
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
        patch_ratio = float(good_library.metadata.get("patch_top_ratio", 0.5))
        positions = select_patch_positions(score_feature, mask_feature, patch_ratio)
        if positions.shape[0] == 0:
            continue
        if query_patches > 0:
            positions = positions[:query_patches]

        patch_candidates = []
        for r, c in positions:
            p_vec = feature[:, int(r), int(c)]
            if bool(good_library.metadata.get("normalize", True)):
                p_vec = l2_normalize(p_vec)
            
            if knn_k > 1:
                g_m = search_library_topk(good_library, p_vec, top_k=knn_k)
                a_m = search_library_topk(anomaly_library, p_vec, top_k=knn_k)
                g_dist = float(np.mean([m[0] for m in g_m])) if g_m else 1.0
                a_dist = float(np.mean([m[0] for m in a_m])) if a_m else 1.0
            else:
                g_dist, _ = search_library(good_library, p_vec)
                a_dist, _ = search_library(anomaly_library, p_vec)
                
            dec = calculate_distance_offset(
                g_dist, a_dist,
                offset_scale=1.0,
                max_offset=None,
                eps=1e-8,
                good_threshold=GOOD_THRESHOLD,
                anomaly_threshold=ANOMALY_THRESHOLD,
            )
            patch_candidates.append({
                "good_distance": g_dist,
                "anomaly_distance": a_dist,
                "row": int(r),
                "col": int(c),
                "decision": dec,
            })

        best = max(
            patch_candidates,
            key=lambda p: (
                float(p["decision"]["signed_offset"]),
                -float(p["anomaly_distance"]),
            ),
        )
        
        region_score = comp["peak_score"]
        regions.append({
            "component_id": comp["component_id"],
            "region_score": region_score,
            "x": x, "y": y, "w": w, "h": h,
            "local_mask": local_mask,
            "signed_offset": float(best["decision"]["signed_offset"]),
            "similar_library": best["decision"]["similar_library"],
        })

    overlay = score_map.copy()
    if suppress_non_candidate_noise:
        outside = (candidate_mask == 0) & (overlay >= GOOD_THRESHOLD)
        overlay[outside] = GOOD_THRESHOLD - 1e-4

    for reg in regions:
        signed_off = reg["signed_offset"]
        x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
        local_mask = reg["local_mask"]
        local_patch = overlay[y:y+h, x:x+w]
        local_patch[local_mask] = np.clip(local_patch[local_mask] + signed_off, 0.0, None)

    adj_score = float(training_image_score(overlay)) if overlay.size else raw_score
    return adj_score, overlay, len(regions)


def load_all_cached_data(
    root: Path,
    data_root: Path,
    gt_dir: Path,
) -> List[Dict[str, Any]]:
    """Load all 680 images, score maps, GT masks, and features into memory cache."""
    cache_pkl = root / "preds" / "cached_eval_records.pkl"
    if cache_pkl.is_file():
        print(f"Loading pre-cached records from {cache_pkl}...")
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
        is_bad = (r["dataset_label"] != "good")
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
    
    print(f"Loaded {len(records)} test images in {time.time() - t0:.2f}s. Saving cache to {cache_pkl}...")
    try:
        import pickle
        with open(cache_pkl, "wb") as f:
            pickle.dump(records, f)
    except Exception as e:
        print(f"Could not save cache pkl: {e}")
        
    return records


def evaluate_configuration(
    records: List[Dict[str, Any]],
    good_lib: Any,
    anomaly_lib: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one ROI extraction and morphology configuration across all 680 images."""
    
    t0 = time.time()
    adj_scores = []
    labels = []
    bad_overlays_256 = []
    bad_gt_masks_256 = []
    total_rois = 0
    middle_image_count = 0
    target_metric_size = (256, 256)
    
    for rec in records:
        score_map = rec["score_map"]
        feature = rec["feature"]
        is_bad = (rec["dataset_label"] != "good")
        labels.append(1 if is_bad else 0)
        
        raw_score = rec["raw_score"]
        if raw_score < GOOD_THRESHOLD or raw_score > ANOMALY_THRESHOLD:
            adj_scores.append(raw_score)
            if is_bad:
                bad_overlays_256.append(cv2.resize(score_map, target_metric_size, interpolation=cv2.INTER_LINEAR))
                bad_gt_masks_256.append(rec["gt_mask_256"])
            continue

        # Middle band image
        middle_image_count += 1
        components, candidate_mask = extract_candidate_regions_adaptive(
            score_map=score_map,
            good_threshold=GOOD_THRESHOLD,
            anomaly_threshold=ANOMALY_THRESHOLD,
            adaptive_mode=config.get("adaptive_mode"),
            adaptive_k=config.get("adaptive_k", 2.0),
            morph_open_k=config.get("morph_open_k"),
            morph_close_k=config.get("morph_close_k"),
            morph_shape=config.get("morph_shape", "ellipse"),
            min_area_pct=config.get("min_area_pct", 0.0),
            min_peak_offset=config.get("min_peak_offset", 0.0),
            max_regions=config.get("max_regions", 0),
        )
        
        adj_score, overlay, roi_count = run_single_image_prediction(
            score_map=score_map,
            feature=feature,
            good_library=good_lib,
            anomaly_library=anomaly_lib,
            original_shape=rec["original_shape"],
            components=components,
            candidate_mask=candidate_mask,
            knn_k=config.get("knn_k", 3),
            query_patches=config.get("query_patches", 3),
            suppress_non_candidate_noise=config.get("suppress_non_candidate_noise", False),
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
        "Category": config.get("category", "General"),
        "min_area_pct": config.get("min_area_pct", 0.0),
        "morph_open_k": config.get("morph_open_k", 0),
        "adaptive_mode": config.get("adaptive_mode", "none"),
        "adaptive_k": config.get("adaptive_k", 0.0),
        "min_peak_offset": config.get("min_peak_offset", 0.0),
        "noise_suppress": config.get("suppress_non_candidate_noise", False),
        "Total_ROIs": total_rois,
        "ROIs_per_Middle_Img": round(total_rois / max(middle_image_count, 1), 1),
        "R-FP-RegionCount": reg_metrics["R-FP-RegionCount"],
        "R-MissRate": reg_metrics["R-MissRate"],
        "R-PixelCoverage": reg_metrics["R-PixelCoverage"],
        "R-FPR": reg_metrics["R-FPR"],
        "P-AUROC": p_auroc,
        "P-AP": p_ap,
        "P-F1": p_f1,
        "P-AUPRO": p_aupro,
        "I-AUROC": i_auroc,
        "I-AP": i_ap,
        "Elapsed_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2, help="GPU to use (default: 2)")
    parser.add_argument("--root", default="/data/wt/two_stages/base_patch_1344")
    parser.add_argument("--data_root", default="/data/wt/ramdisk/leishi_026/test")
    parser.add_argument("--ground_truth", default="/data/wt/ramdisk/leishi_026/ground_truth")
    parser.add_argument("--output_csv", default="/data/wt/two_stages/base_patch_1344/track2_roi_morphology_results.csv")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = Path(args.root)
    good_lib = load_feature_library(root / "good", device, True)
    anomaly_lib = load_feature_library(root / "anomaly", device, True)
    print(f"Loaded libraries: good={good_lib.index.ntotal}, anomaly={anomaly_lib.index.ntotal}")

    records = load_all_cached_data(root, Path(args.data_root), Path(args.ground_truth))

    # Define comprehensive experimental matrix
    experiments = []

    # --- Baseline ---
    experiments.append({
        "name": "Baseline (Raw CCs, min_area=1)",
        "category": "Baseline",
        "min_area_pct": 0.0,
    })
    experiments.append({
        "name": "Baseline (Default min_area_pct=0.005%)",
        "category": "Baseline",
        "min_area_pct": 0.005,
    })

    # --- 1. min_area_pct Sweep ---
    for area_pct in [0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
        experiments.append({
            "name": f"Area Sweep: min_area_pct={area_pct:.3f}%",
            "category": "Area_Sweep",
            "min_area_pct": area_pct,
        })

    # --- 2. Morphological Opening Sweep (with area_pct=0.005%) ---
    for k in [3, 5, 7, 9]:
        for shape in ["ellipse", "rect"]:
            experiments.append({
                "name": f"Morph Opening: k={k} ({shape}) + min_area_pct=0.005%",
                "category": "Morphology",
                "min_area_pct": 0.005,
                "morph_open_k": k,
                "morph_shape": shape,
            })

    # --- 2b. Morphological Opening without area filter (pure morphology) ---
    for k in [3, 5, 7]:
        experiments.append({
            "name": f"Pure Morph Opening: k={k} (no area filter)",
            "category": "Morphology_Pure",
            "min_area_pct": 0.0,
            "morph_open_k": k,
        })

    # --- 2c. Morph Open + Close ---
    experiments.append({
        "name": "Morph Open(3) + Close(3) + min_area_pct=0.005%",
        "category": "Morphology",
        "min_area_pct": 0.005,
        "morph_open_k": 3,
        "morph_close_k": 3,
    })

    # --- 3. Background Statistical Adaptive Threshold ---
    for ak in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        experiments.append({
            "name": f"Global BG Stat: mean + {ak}*std + min_area_pct=0.005%",
            "category": "Global_Adaptive",
            "min_area_pct": 0.005,
            "adaptive_mode": "global_stat",
            "adaptive_k": ak,
        })

    # --- 3b. Local Gaussian Adaptive Contrast Threshold ---
    for ak in [1.0, 1.5, 2.0, 2.5]:
        experiments.append({
            "name": f"Local Gaussian Contrast: local_mean + {ak}*std + min_area_pct=0.005%",
            "category": "Local_Adaptive",
            "min_area_pct": 0.005,
            "adaptive_mode": "local_stat",
            "adaptive_k": ak,
        })

    # --- 4. Peak Saliency Filter Sweep ---
    for peak_off in [0.001, 0.002, 0.003, 0.005]:
        experiments.append({
            "name": f"Peak Saliency: min_peak >= th+{peak_off:.3f} + min_area_pct=0.005%",
            "category": "Peak_Saliency",
            "min_area_pct": 0.005,
            "min_peak_offset": peak_off,
        })

    # --- 5. Non-Candidate Background Noise Suppression ---
    experiments.append({
        "name": "Baseline (0.005%) + Clean Noise Suppression",
        "category": "Noise_Suppression",
        "min_area_pct": 0.005,
        "suppress_non_candidate_noise": True,
    })
    experiments.append({
        "name": "Morph Open(3) (0.005%) + Clean Noise Suppression",
        "category": "Noise_Suppression",
        "min_area_pct": 0.005,
        "morph_open_k": 3,
        "suppress_non_candidate_noise": True,
    })
    experiments.append({
        "name": "Morph Open(5) (0.005%) + Clean Noise Suppression",
        "category": "Noise_Suppression",
        "min_area_pct": 0.005,
        "morph_open_k": 5,
        "suppress_non_candidate_noise": True,
    })

    # --- 6. Combined Optimal Strategies ---
    experiments.append({
        "name": "Combined Opt 1: Morph Open(3) + Peak(0.002) + area=0.005%",
        "category": "Combined_Optimal",
        "min_area_pct": 0.005,
        "morph_open_k": 3,
        "min_peak_offset": 0.002,
    })
    experiments.append({
        "name": "Combined Opt 2: Morph Open(3) + Peak(0.002) + area=0.005% + Noise Suppress",
        "category": "Combined_Optimal",
        "min_area_pct": 0.005,
        "morph_open_k": 3,
        "min_peak_offset": 0.002,
        "suppress_non_candidate_noise": True,
    })
    experiments.append({
        "name": "Combined Opt 3: Global BG Stat(1.5) + Morph Open(3) + area=0.005% + Noise Suppress",
        "category": "Combined_Optimal",
        "min_area_pct": 0.005,
        "adaptive_mode": "global_stat",
        "adaptive_k": 1.5,
        "morph_open_k": 3,
        "suppress_non_candidate_noise": True,
    })
    experiments.append({
        "name": "Combined Opt 4: Morph Open(5) + Peak(0.002) + area=0.008% + Noise Suppress",
        "category": "Combined_Optimal",
        "min_area_pct": 0.008,
        "morph_open_k": 5,
        "min_peak_offset": 0.002,
        "suppress_non_candidate_noise": True,
    })
    experiments.append({
        "name": "Combined Opt 5: Morph Open(3) + area=0.008% + Noise Suppress",
        "category": "Combined_Optimal",
        "min_area_pct": 0.008,
        "morph_open_k": 3,
        "suppress_non_candidate_noise": True,
    })

    print(f"\nTotal experimental configurations to evaluate: {len(experiments)}")
    results = []

    print("\n" + "=" * 120)
    header = (
        f"{'Category':<16} {'Config Name':<42} {'ROIs':>6} {'R-FP-Count':>10} "
        f"{'R-Miss%':>8} {'P-AP':>8} {'P-AUROC':>8} {'P-AUPRO':>8} {'I-AUROC':>8} {'Time':>6}"
    )
    print(header)
    print("=" * 120)

    for exp in experiments:
        res = evaluate_configuration(records, good_lib, anomaly_lib, exp)
        results.append(res)
        
        row_str = (
            f"{res['Category']:<16} {res['Config_Name']:<42} {res['Total_ROIs']:>6} "
            f"{int(res['R-FP-RegionCount']):>10} {res['R-MissRate']*100:>7.2f}% "
            f"{res['P-AP']:>8.4f} {res['P-AUROC']:>8.4f} {res['P-AUPRO']:>8.4f} "
            f"{res['I-AUROC']:>8.4f} {res['Elapsed_s']:>5.1f}s"
        )
        print(row_str, flush=True)

    print("=" * 120)

    # Save results to CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll results saved to: {out_csv}")


if __name__ == "__main__":
    main()

