#!/usr/bin/env python3
"""Track 1: Spatial Coordinate Prior & Position-Weighted KNN Matching on 672 Resolution.

Implements and evaluates:
1. Spatial coordinate-aware distance metric:
   D_spatial(p, q) = D_feat(p, q) + lambda * ||(u_p, v_p) - (u_q, v_q)||^2
   where (u, v) in [0, 1]^2 are normalized feature-map grid coordinates:
   u = col / W_feat, v = row / H_feat (W_feat=48, H_feat=48).
2. KNN matching with spatial distance weighting:
   KNN selection on Good and Anomaly feature libraries with top-k retrieval (k in [1, 3, 5]).
3. Evaluation across all 680 test images on GPU 0/1:
   - I-AUROC, I-AP, I-F1
   - P-AUROC, P-AP, P-F1, P-AUPRO
   - R-MissRate, R-FP-RegionCount, R-PixelCoverage, R-FPR
4. Quantitative comparison tables across lambda in [0.0, 0.02, 0.05, 0.1, 0.2].
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from skimage import measure
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
_DINOMALY_DIR = Path(__file__).resolve().parent.parent / "Dinomaly2"
for d in [_UTILS_DIR, _DINOMALY_DIR]:
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
    calculate_distance_offset,
    linear_score_to_feature,
    load_feature_library,
    load_mask,
    l2_normalize,
    select_patch_positions,
)
from eval_track3_adaptive_geometry import (
    extract_candidate_regions_track3,
    fast_region_detection_metrics,
)

GOOD_THRESHOLD = 0.014
ANOMALY_THRESHOLD = 0.030


def load_cached_test_data(
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

    print("Loading test data from individual files...")
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

    print(f"Loaded {len(records)} test images in {time.time() - t0:.2f}s.")
    return records


class SpatialKNNFeatureBank:
    """GPU-accelerated spatial-coordinate-aware feature bank matching."""

    def __init__(self, root: Path, device: torch.device):
        self.device = device
        p_good = root / "good"
        p_anomaly = root / "anomaly"

        vecs_good = np.load(p_good / "vectors.npy")
        vecs_anomaly = np.load(p_anomaly / "vectors.npy")

        with open(p_good / "id_mapping.json") as f:
            good_meta = json.load(f)
        with open(p_anomaly / "id_mapping.json") as f:
            anomaly_meta = json.load(f)

        self.good_records = good_meta["records"]
        self.anomaly_records = anomaly_meta["records"]

        # Grid size (48, 48)
        grid_w, grid_h = 48.0, 48.0
        good_uv = [[r["patch_col"] / grid_w, r["patch_row"] / grid_h] for r in self.good_records]
        anomaly_uv = [[r["patch_col"] / grid_w, r["patch_row"] / grid_h] for r in self.anomaly_records]

        self.good_coords = torch.tensor(good_uv, dtype=torch.float32, device=device)
        self.anomaly_coords = torch.tensor(anomaly_uv, dtype=torch.float32, device=device)
        self.good_vecs = torch.tensor(vecs_good, dtype=torch.float32, device=device)
        self.anomaly_vecs = torch.tensor(vecs_anomaly, dtype=torch.float32, device=device)

        print(f"SpatialKNNFeatureBank initialized on {device}: Good={len(self.good_vecs)} Anomaly={len(self.anomaly_vecs)}")

    def query(
        self,
        patch_vectors: torch.Tensor,
        patch_coords: torch.Tensor,
        lam: float,
        knn_k: int = 1,
        dist_mode: str = "spatial_dist",
    ) -> Tuple[List[float], List[float]]:
        """Query spatial-weighted KNN against good and anomaly banks.
        
        Args:
            patch_vectors: (M, 768) float32 normalized vectors on device.
            patch_coords: (M, 2) float32 (u, v) in [0, 1]^2 on device.
            lam: Spatial distance penalty lambda.
            knn_k: Number of nearest neighbors.
            dist_mode: 'spatial_dist' or 'feat_dist_of_spatial_neighbors'.
        
        Returns:
            (good_distances, anomaly_distances) as lists of floats.
        """
        # 1. Feature squared Euclidean distances
        d_feat_g = torch.cdist(patch_vectors, self.good_vecs, p=2) ** 2  # (M, N_good)
        d_feat_a = torch.cdist(patch_vectors, self.anomaly_vecs, p=2) ** 2  # (M, N_anomaly)

        # 2. Spatial squared Euclidean distances: ||(u_p, v_p) - (u_q, v_q)||^2
        if lam > 0.0:
            d_pos_g = torch.cdist(patch_coords, self.good_coords, p=2) ** 2  # (M, N_good)
            d_pos_a = torch.cdist(patch_coords, self.anomaly_coords, p=2) ** 2  # (M, N_anomaly)
            d_spat_g = d_feat_g + lam * d_pos_g
            d_spat_a = d_feat_a + lam * d_pos_a
        else:
            d_spat_g = d_feat_g
            d_spat_a = d_feat_a

        # 3. Top-K KNN retrieval
        if knn_k > 1:
            topk_spat_g, idx_g = torch.topk(d_spat_g, k=knn_k, dim=-1, largest=False)
            topk_spat_a, idx_a = torch.topk(d_spat_a, k=knn_k, dim=-1, largest=False)
            if dist_mode == "feat_dist_of_spatial_neighbors":
                g_dists = torch.gather(d_feat_g, -1, idx_g).mean(dim=-1).tolist()
                a_dists = torch.gather(d_feat_a, -1, idx_a).mean(dim=-1).tolist()
            else:
                g_dists = topk_spat_g.mean(dim=-1).tolist()
                a_dists = topk_spat_a.mean(dim=-1).tolist()
        else:
            min_val_g, min_idx_g = torch.min(d_spat_g, dim=-1)
            min_val_a, min_idx_a = torch.min(d_spat_a, dim=-1)
            if dist_mode == "feat_dist_of_spatial_neighbors":
                g_dists = d_feat_g.gather(1, min_idx_g.unsqueeze(1)).squeeze(1).tolist()
                a_dists = d_feat_a.gather(1, min_idx_a.unsqueeze(1)).squeeze(1).tolist()
            else:
                g_dists = min_val_g.tolist()
                a_dists = min_val_a.tolist()

        return g_dists, a_dists


def evaluate_spatial_knn(
    records: List[Dict[str, Any]],
    bank: SpatialKNNFeatureBank,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one experimental configuration across all 680 test images."""
    lam = float(config.get("lambda", 0.0))
    knn_k = int(config.get("knn_k", 1))
    dist_mode = str(config.get("dist_mode", "spatial_dist"))
    hard_anomaly_direct = bool(config.get("hard_anomaly_direct", True))
    hard_anomaly_dist_th = float(config.get("hard_anomaly_dist_th", 0.15))
    query_patches = int(config.get("query_patches", 3))
    patch_ratio = float(config.get("patch_ratio", 0.5))
    min_area_pct = float(config.get("min_area_pct", 0.0))

    t0 = time.time()
    adj_scores = []
    labels = []
    bad_overlays_256 = []
    bad_gt_masks_256 = []
    total_rois = 0
    middle_image_count = 0
    target_metric_size = (256, 256)
    device = bank.device

    for rec in records:
        score_map = rec["score_map"]
        feature = rec["feature"]
        is_bad = (rec["dataset_label"] != "good")
        labels.append(1 if is_bad else 0)
        raw_score = rec["raw_score"]

        # Stage 1 classification gating
        if raw_score < GOOD_THRESHOLD or raw_score > ANOMALY_THRESHOLD:
            adj_scores.append(raw_score)
            if is_bad:
                bad_overlays_256.append(cv2.resize(score_map, target_metric_size, interpolation=cv2.INTER_LINEAR))
                bad_gt_masks_256.append(rec["gt_mask_256"])
            continue

        middle_image_count += 1
        components, candidate_mask = extract_candidate_regions_track3(
            score_map=score_map,
            good_threshold=GOOD_THRESHOLD,
            anomaly_threshold=ANOMALY_THRESHOLD,
            min_area_pct=min_area_pct,
        )

        height, width = score_map.shape[:2]
        feature_shape = feature.shape[-2:]
        feature_height, feature_width = feature_shape
        regions = []

        for comp in components:
            x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
            local_mask = comp["local_mask"]

            # Map ROI component to feature cells
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
            positions = select_patch_positions(score_feature, mask_feature, patch_ratio)
            if positions.shape[0] == 0:
                continue
            if query_patches > 0:
                positions = positions[:query_patches]

            # Batch query patch features and coords
            p_vecs = []
            p_coords = []
            for r, c in positions:
                p_vec = l2_normalize(feature[:, int(r), int(c)])
                p_vecs.append(p_vec)
                p_coords.append([float(c) / float(feature_width), float(r) / float(feature_height)])

            p_vecs_t = torch.tensor(np.array(p_vecs), dtype=torch.float32, device=device)
            p_coords_t = torch.tensor(np.array(p_coords), dtype=torch.float32, device=device)

            g_dists, a_dists = bank.query(
                p_vecs_t,
                p_coords_t,
                lam=lam,
                knn_k=knn_k,
                dist_mode=dist_mode,
            )

            patch_candidates = []
            for idx, (r, c) in enumerate(positions):
                g_dist = g_dists[idx]
                a_dist = a_dists[idx]

                if hard_anomaly_direct and a_dist <= hard_anomaly_dist_th:
                    signed_off = 0.020
                    sim_lib = "anomaly"
                    dec = {"signed_offset": signed_off, "similar_library": sim_lib}
                else:
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
            regions.append({
                "component_id": comp["component_id"],
                "x": x, "y": y, "w": w, "h": h,
                "local_mask": local_mask,
                "signed_offset": float(best["decision"]["signed_offset"]),
            })

        total_rois += len(regions)
        overlay = score_map.copy()
        for reg in regions:
            signed_off = reg["signed_offset"]
            x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
            local_mask = reg["local_mask"]
            local_patch = overlay[y:y+h, x:x+w]
            reg_scores = local_patch[local_mask]
            max_s = float(np.max(reg_scores)) if reg_scores.size else 1.0
            weight = (reg_scores / max_s) if max_s > 1e-8 else 1.0
            local_patch[local_mask] = np.clip(reg_scores + signed_off * weight, 0.0, None)

        adj_score = float(training_image_score(overlay)) if overlay.size else raw_score
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

    # Pixel and Region metrics
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
        "Config_Name": config.get("name", f"SpatialKNN_lambda_{lam}_k_{knn_k}"),
        "Category": config.get("category", "Spatial_Prior_KNN"),
        "Lambda": lam,
        "KNN_K": knn_k,
        "Dist_Mode": dist_mode,
        "Hard_Anomaly_Direct": hard_anomaly_direct,
        "Total_ROIs": total_rois,
        "ROIs_per_Middle_Img": round(total_rois / max(middle_image_count, 1), 1),
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


def build_spatial_knn_experiment_matrix() -> List[Dict[str, Any]]:
    """Construct complete matrix of spatial prior KNN experiments."""
    experiments = []

    # 1. Main Track 1 Requirement: lambda in [0.0, 0.02, 0.05, 0.1, 0.2] with KNN k=1 (Hard Anomaly Direct)
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2]:
        experiments.append({
            "name": f"Spatial_k1_lam_{lam:.2f} (Direct Hard Anomaly)",
            "category": "1_Spatial_Prior_k1",
            "lambda": lam,
            "knn_k": 1,
            "dist_mode": "spatial_dist",
            "hard_anomaly_direct": True,
        })

    # 2. Track 1 Extension: lambda in [0.0, 0.02, 0.05, 0.1, 0.2] with KNN k=3 (Hard Anomaly Direct)
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2]:
        experiments.append({
            "name": f"Spatial_k3_lam_{lam:.2f} (Direct Hard Anomaly)",
            "category": "2_Spatial_Prior_k3",
            "lambda": lam,
            "knn_k": 3,
            "dist_mode": "spatial_dist",
            "hard_anomaly_direct": True,
        })

    # 3. Pure Margin Mode (hard_anomaly_direct=False) across lambda in [0.0, 0.02, 0.05, 0.1, 0.2] with k=1
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2]:
        experiments.append({
            "name": f"Spatial_PureMargin_k1_lam_{lam:.2f}",
            "category": "3_Pure_Margin_k1",
            "lambda": lam,
            "knn_k": 1,
            "dist_mode": "spatial_dist",
            "hard_anomaly_direct": False,
        })

    # 4. Feature Distance of Spatial Neighbors (Rerank mode) across lambda in [0.0, 0.02, 0.05, 0.1, 0.2] with k=3
    for lam in [0.0, 0.02, 0.05, 0.1, 0.2]:
        experiments.append({
            "name": f"Spatial_RerankFeat_k3_lam_{lam:.2f}",
            "category": "4_Spatial_Rerank_k3",
            "lambda": lam,
            "knn_k": 3,
            "dist_mode": "feat_dist_of_spatial_neighbors",
            "hard_anomaly_direct": True,
        })

    return experiments


def main():
    parser = argparse.ArgumentParser(description="Track 1: Spatial Coordinate Prior & Position-Weighted KNN")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--root", default="/data/wt/two_stages/base_672_15k", help="Root feature library path")
    parser.add_argument("--data_root", default="/data/wt/ramdisk/leishi_026/test", help="Test image directory")
    parser.add_argument("--ground_truth", default="/data/wt/ramdisk/leishi_026/ground_truth", help="Ground truth mask directory")
    parser.add_argument("--output_csv", default="/data/wt/two_stages/base_672_15k/track1_spatial_prior_knn_results.csv", help="Output CSV path")
    parser.add_argument("--output_json", default="/data/wt/two_stages/base_672_15k/track1_spatial_prior_knn_results.json", help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Executing Track 1 Spatial Prior KNN Evaluation on {device}")
    print(f"Feature Bank Root: {args.root}")
    print(f"Test Data Root:    {args.data_root}")
    print(f"Ground Truth Dir:  {args.ground_truth}")

    root = Path(args.root)
    bank = SpatialKNNFeatureBank(root, device)
    records = load_cached_test_data(root, Path(args.data_root), Path(args.ground_truth))

    experiments = build_spatial_knn_experiment_matrix()
    print(f"\nTotal Track 1 experimental configurations to evaluate: {len(experiments)}")

    results = []

    print("\n" + "=" * 135)
    header = (
        f"{'Category':<22} {'Config Name':<42} {'Lambda':>6} {'k':>2} "
        f"{'FP-Count':>9} {'R-Miss%':>8} {'P-AP':>8} {'P-AUROC':>8} {'P-AUPRO':>8} {'I-AUROC':>8} {'I-AP':>8} {'Time':>6}"
    )
    print(header)
    print("=" * 135)

    for exp in experiments:
        res = evaluate_spatial_knn(records, bank, exp)
        results.append(res)

        row_str = (
            f"{res['Category']:<22} {res['Config_Name']:<42} {res['Lambda']:>6.2f} {res['KNN_K']:>2} "
            f"{int(res['R-FP-RegionCount']):>9} {res['R-MissRate']*100:>7.2f}% "
            f"{res['P-AP']:>8.4f} {res['P-AUROC']:>8.4f} {res['P-AUPRO']:>8.4f} "
            f"{res['I-AUROC']:>8.4f} {res['I-AP']:>8.4f} {res['Elapsed_s']:>5.1f}s"
        )
        print(row_str, flush=True)

    print("=" * 135)

    # Save to CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save to JSON
    out_json = Path(args.output_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll Track 1 results successfully saved to:\n  - CSV:  {out_csv}\n  - JSON: {out_json}")


if __name__ == "__main__":
    main()
