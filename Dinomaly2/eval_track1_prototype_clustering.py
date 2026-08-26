#!/usr/bin/env python3
"""Track 1: Prototype Clustering & Feature Bank Optimization.

Ultra-fast GPU-accelerated evaluation suite for:
1. Baseline (Full Uncompressed Bank)
2. K-Means Prototype Clustering (Centroid & Medoid)
3. K-Center Greedy Coreset Selection
4. Cross-Boundary Noise Trimming & Density Outlier Removal
5. Hybrid Cleaning + Prototype Compression
6. Parameter sweeps (K_good in [100, 300, 600, 1000], K_anomaly in [50, 100, 200, 400], k-NN in [1, 2, 3, 5, 7])

Outputs comprehensive quantitative comparison table and JSON summary.
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
from sklearn.cluster import KMeans
from sklearn.metrics import auc
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from anomaly_evaluation import max_f1, pixel_f1_score_and_threshold
from dinomaly_two_stage import calculate_distance_offset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("track1_prototype")


# =========================================================================
# 1. Clustering & Selection Algorithms
# =========================================================================

def l2_norm_vectors(vectors: np.ndarray) -> np.ndarray:
    """Ensure vectors are L2-normalized float32."""
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def kmeans_clustering(
    vectors: np.ndarray,
    n_clusters: int,
    use_medoids: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """K-Means clustering on L2-normalized vectors."""
    n_samples = vectors.shape[0]
    if n_clusters >= n_samples:
        return vectors.copy()

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=5,
        max_iter=300,
        random_state=random_state,
    )
    kmeans.fit(vectors)
    centers = kmeans.cluster_centers_

    if use_medoids:
        medoids = []
        for i in range(n_clusters):
            cluster_mask = (kmeans.labels_ == i)
            if not np.any(cluster_mask):
                medoids.append(centers[i])
                continue
            cluster_vecs = vectors[cluster_mask]
            dists = np.linalg.norm(cluster_vecs - centers[i : i + 1], axis=1)
            medoids.append(cluster_vecs[np.argmin(dists)])
        prototypes = np.array(medoids, dtype=np.float32)
    else:
        prototypes = centers

    return l2_norm_vectors(prototypes)


def kcenter_greedy_coreset(
    vectors: np.ndarray,
    target_size: int,
    random_state: int = 42,
) -> np.ndarray:
    """K-Center Greedy / Farthest Point Sampling."""
    n_samples = vectors.shape[0]
    if target_size >= n_samples:
        return vectors.copy()

    mean_vec = vectors.mean(axis=0, keepdims=True)
    dists_to_mean = np.linalg.norm(vectors - mean_vec, axis=1)
    init_idx = int(np.argmin(dists_to_mean))

    selected = [init_idx]
    min_distances = np.linalg.norm(vectors - vectors[init_idx : init_idx + 1], axis=1)

    for _ in range(1, target_size):
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)
        new_dists = np.linalg.norm(vectors - vectors[next_idx : next_idx + 1], axis=1)
        min_distances = np.minimum(min_distances, new_dists)

    return l2_norm_vectors(vectors[np.array(selected, dtype=np.int64)])


def remove_boundary_noise(
    good_vecs: np.ndarray,
    bad_vecs: np.ndarray,
    margin_ratio: float = 1.0,
    trim_isolated_percentile: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Clean boundary noise and isolated outliers between good and anomaly libraries."""
    g_idx = faiss.IndexFlatL2(768)
    g_idx.add(good_vecs)
    b_idx = faiss.IndexFlatL2(768)
    b_idx.add(bad_vecs)

    Dg_self, _ = g_idx.search(good_vecs, 2)
    d_g_self = np.sqrt(np.maximum(Dg_self[:, 1], 0.0))

    Db_self, _ = b_idx.search(bad_vecs, 2)
    d_b_self = np.sqrt(np.maximum(Db_self[:, 1], 0.0))

    Dg_cross, _ = b_idx.search(good_vecs, 1)
    d_g_cross = np.sqrt(np.maximum(Dg_cross[:, 0], 0.0))

    Db_cross, _ = g_idx.search(bad_vecs, 1)
    d_b_cross = np.sqrt(np.maximum(Db_cross[:, 0], 0.0))

    good_keep = (d_g_cross >= margin_ratio * d_g_self)
    bad_keep = (d_b_cross >= margin_ratio * d_b_self)

    if trim_isolated_percentile > 0.0:
        g_thresh = np.percentile(d_g_self, 100.0 - trim_isolated_percentile)
        b_thresh = np.percentile(d_b_self, 100.0 - trim_isolated_percentile)
        good_keep = good_keep & (d_g_self <= g_thresh)
        bad_keep = bad_keep & (d_b_self <= b_thresh)

    cleaned_good = good_vecs[good_keep]
    cleaned_bad = bad_vecs[bad_keep]

    stats = {
        "good_orig": len(good_vecs),
        "good_removed": int((~good_keep).sum()),
        "good_retained": len(cleaned_good),
        "bad_orig": len(bad_vecs),
        "bad_removed": int((~bad_keep).sum()),
        "bad_retained": len(cleaned_bad),
    }
    return cleaned_good, cleaned_bad, stats


# =========================================================================
# 2. Optimized Fast GPU Metric Evaluator
# =========================================================================

def gpu_auroc_and_ap(labels_t: torch.Tensor, scores_t: torch.Tensor) -> Tuple[float, float]:
    """Compute exact AUROC and AP on GPU in milliseconds."""
    desc_idx = torch.argsort(scores_t, descending=True)
    ordered_labels = labels_t[desc_idx]
    tps = torch.cumsum(ordered_labels, dim=0)
    fps = torch.cumsum(1.0 - ordered_labels, dim=0)
    total_pos = float(tps[-1].item())
    total_neg = float(fps[-1].item())
    
    if total_pos == 0 or total_neg == 0:
        return float("nan"), float("nan")

    tpr = tps / total_pos
    fpr = fps / total_neg
    auroc = torch.trapz(tpr, fpr).item()

    precision = tps / (tps + fps)
    ap = (precision * ordered_labels).sum().item() / total_pos
    return float(auroc), float(ap)


class UltraFastEvaluator:
    def __init__(self, records: List[Dict[str, Any]], device: str = "cuda:0", target_size: Tuple[int, int] = (256, 256)):
        self.device = torch.device(device)
        self.target_size = target_size
        self.num_records = len(records)
        
        # 1. Labels
        self.labels = np.array([1 if r["dataset_label"] != "good" else 0 for r in records], dtype=np.uint8)
        self.labels_t = torch.from_numpy(self.labels).float().to(self.device)
        
        # 2. Separate static non-middle vs dynamic middle records
        print("Pre-indexing test dataset into memory structures...")
        self.static_raw_scores = np.array([r["raw_score"] for r in records], dtype=np.float32)
        
        self.bad_indices = [i for i, r in enumerate(records) if r["dataset_label"] != "good"]
        self.bad_gt_masks_256 = np.stack([records[i]["gt_mask_256"] for i in self.bad_indices]).astype(np.uint8)
        self.pix_labels_t = torch.from_numpy(self.bad_gt_masks_256.reshape(-1)).float().to(self.device)
        
        # Precompute static base overlays for bad images (256x256)
        self.base_bad_overlays_256 = []
        self.bad_idx_to_pos = {idx: pos for pos, idx in enumerate(self.bad_indices)}
        
        for idx in self.bad_indices:
            r = records[idx]
            sm_256 = cv2.resize(r["score_map"], target_size, interpolation=cv2.INTER_LINEAR)
            self.base_bad_overlays_256.append(sm_256)
            
        self.base_bad_overlays_256 = np.stack(self.base_bad_overlays_256).astype(np.float32)

        # Precompute middle-band records
        self.middle_records_meta = []
        for i, r in enumerate(records):
            if not r["is_middle"] or not r["component_data"]:
                continue
            
            sm = r["score_map"]
            top_k_count = max(1, int(sm.size * 0.001))
            is_bad = (r["dataset_label"] != "good")
            bad_pos = self.bad_idx_to_pos.get(i, -1)
            
            candidate_mask = np.zeros(sm.shape, dtype=np.uint8)
            for c in r["component_data"]:
                candidate_mask[c["mask"]] = 1
            count, comp_labels, _, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
            
            comps_meta = []
            for c in r["component_data"]:
                cid = c["component_id"]
                if not 0 < cid < count:
                    continue
                reg_mask = (comp_labels == cid)
                pix_idx = np.flatnonzero(reg_mask)
                reg_scores = sm.ravel()[pix_idx]
                max_s = float(np.max(reg_scores)) if reg_scores.size else 1.0
                weights = (reg_scores / max_s) if max_s > 1e-8 else np.ones_like(reg_scores)
                
                # 256x256 indices
                reg_mask_256 = cv2.resize(reg_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST) > 0
                pix_idx_256 = np.flatnonzero(reg_mask_256)
                
                comps_meta.append({
                    "component_id": cid,
                    "patch_vectors": c["patch_vectors"],
                    "pixel_indices": pix_idx,
                    "orig_scores": reg_scores,
                    "weights": weights,
                    "pix_idx_256": pix_idx_256,
                })
                
            self.middle_records_meta.append({
                "record_index": i,
                "is_bad": is_bad,
                "bad_pos": bad_pos,
                "base_flat_score_map": sm.ravel().copy(),
                "top_k_count": top_k_count,
                "comps": comps_meta,
            })
            
        print(f"Pre-indexed {len(self.middle_records_meta)} middle-band images with active ROIs.")

        # Precompute Ground Truth PRO metadata
        self.pro_gt_meta = []
        self.inverse_masks = []
        self.total_regions = 0
        total_inv_count = 0
        
        for mask in self.bad_gt_masks_256:
            inv = ~mask.astype(bool)
            self.inverse_masks.append(inv)
            total_inv_count += int(inv.sum())
            
            lbls = measure.label(mask)
            areas = np.bincount(lbls.reshape(-1))[1:].astype(np.float64)
            reg_masks = [lbls == rid for rid in range(1, len(areas) + 1)]
            self.pro_gt_meta.append((lbls, areas, reg_masks))
            self.total_regions += len(areas)
            
        self.inverse_count = total_inv_count
        print(f"Precomputed {len(self.bad_indices)} GT masks, {self.total_regions} defect regions.")

    def compute_fast_aupro(self, amaps: np.ndarray, num_th: int = 100) -> float:
        """Fast vectorized PRO computation."""
        minimum = float(amaps.min())
        maximum = float(amaps.max())
        delta = (maximum - minimum) / num_th
        if delta <= 0.0:
            return 0.0
        thresholds = np.arange(minimum, maximum, delta, dtype=np.float32)
        if len(thresholds) == 0:
            return float("nan")

        pro_sums = np.zeros_like(thresholds, dtype=np.float64)
        fp_counts = np.zeros_like(thresholds, dtype=np.int64)

        for i, (lbls, areas, reg_masks) in enumerate(self.pro_gt_meta):
            amap = amaps[i]
            inv_mask = self.inverse_masks[i]
            
            out_vals = np.sort(amap[inv_mask])
            fp_counts += out_vals.size - np.searchsorted(out_vals, thresholds, side="right")
            
            for rid_idx, area in enumerate(areas):
                reg_vals = np.sort(amap[reg_masks[rid_idx]])
                hits = reg_vals.size - np.searchsorted(reg_vals, thresholds, side="right")
                pro_sums += hits / area

        pros = pro_sums / self.total_regions
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

    def evaluate(
        self,
        good_vecs: np.ndarray,
        bad_vecs: np.ndarray,
        config: Dict[str, Any],
        good_threshold: float = 0.014,
        anomaly_threshold: float = 0.030,
    ) -> Dict[str, Any]:
        t0 = time.time()
        
        # Build FAISS indexes
        g_idx = faiss.IndexFlatL2(768)
        g_idx.add(l2_norm_vectors(good_vecs))
        b_idx = faiss.IndexFlatL2(768)
        b_idx.add(l2_norm_vectors(bad_vecs))

        knn_k = config.get("knn_k", 3)
        query_patches = config.get("query_patches", 3)

        adj_scores = self.static_raw_scores.copy()
        bad_overlays_256 = self.base_bad_overlays_256.copy()

        for mid in self.middle_records_meta:
            rec_idx = mid["record_index"]
            is_bad = mid["is_bad"]
            bad_pos = mid["bad_pos"]
            
            flat_sm = mid["base_flat_score_map"].copy()
            
            if is_bad and bad_pos >= 0:
                sm_256_flat = bad_overlays_256[bad_pos].ravel()
            else:
                sm_256_flat = None

            for comp in mid["comps"]:
                patch_candidates = []
                p_vecs = comp["patch_vectors"][:query_patches]
                for row, col, p_vec in p_vecs:
                    p_v = p_vec[None, :].astype(np.float32)
                    Dg, _ = g_idx.search(p_v, min(knn_k, g_idx.ntotal))
                    Db, _ = b_idx.search(p_v, min(knn_k, b_idx.ntotal))

                    good_dist = float(np.mean(np.sqrt(np.maximum(Dg[0], 0.0))))
                    anomaly_dist = float(np.mean(np.sqrt(np.maximum(Db[0], 0.0))))

                    dec = calculate_distance_offset(
                        good_dist,
                        anomaly_dist,
                        offset_scale=1.0,
                        max_offset=None,
                        eps=1e-8,
                        good_threshold=good_threshold,
                        anomaly_threshold=anomaly_threshold,
                    )
                    patch_candidates.append({
                        "good_distance": good_dist,
                        "anomaly_distance": anomaly_dist,
                        "decision": dec,
                    })

                if not patch_candidates:
                    continue

                best = max(
                    patch_candidates,
                    key=lambda p: (
                        float(p["decision"]["signed_offset"]),
                        -float(p["anomaly_distance"]),
                    ),
                )
                signed_off = float(best["decision"]["signed_offset"])
                
                # In-place modify score map pixels
                pix_idx = comp["pixel_indices"]
                orig_s = comp["orig_scores"]
                w = comp["weights"]
                flat_sm[pix_idx] = np.clip(orig_s + signed_off * w, 0.0, None)
                
                # In-place modify 256x256 overlay pixels
                if sm_256_flat is not None:
                    pix_idx_256 = comp["pix_idx_256"]
                    sm_256_flat[pix_idx_256] = np.clip(sm_256_flat[pix_idx_256] + signed_off, 0.0, None)

            # Compute image score using fast np.partition
            top_k = mid["top_k_count"]
            adj_scores[rec_idx] = float(np.partition(flat_sm, -top_k)[-top_k:].mean())

        adj_scores_t = torch.tensor(adj_scores, dtype=torch.float32, device=self.device)
        pix_scores_t = torch.from_numpy(bad_overlays_256.reshape(-1)).float().to(self.device)

        # Fast GPU Image AUROC & AP
        i_auroc, i_ap = gpu_auroc_and_ap(self.labels_t, adj_scores_t)
        i_f1 = max_f1(self.labels, adj_scores)

        # Fast GPU Pixel AUROC & AP
        p_auroc, p_ap = gpu_auroc_and_ap(self.pix_labels_t, pix_scores_t)
        p_f1, _ = pixel_f1_score_and_threshold(self.bad_gt_masks_256, bad_overlays_256)
        p_aupro = self.compute_fast_aupro(bad_overlays_256, num_th=100)

        # Region metrics
        image_miss_rates = []
        image_coverages = []
        total_fp = 0
        for mask, sm in zip(self.bad_gt_masks_256, bad_overlays_256):
            gt_l = measure.label(mask.astype(bool))
            rc = int(gt_l.max())
            pred = sm >= good_threshold
            pred_l = measure.label(pred.astype(np.uint8))
            pc = int(pred_l.max())
            det = 0
            covs = []
            for rid in range(1, rc + 1):
                rmask = (gt_l == rid)
                cov = int(pred[rmask].sum())
                if cov > 0:
                    det += 1
                covs.append(cov / max(int(rmask.sum()), 1))
            if rc > 0:
                image_miss_rates.append((rc - det) / rc)
                image_coverages.append(float(np.mean(covs)))
            if pc > 0:
                if rc > 0:
                    tp = int(np.count_nonzero(np.unique(pred_l[gt_l > 0]) > 0))
                else:
                    tp = 0
                total_fp += (pc - tp)

        r_miss = float(np.mean(image_miss_rates)) if image_miss_rates else float("nan")
        r_cov = float(np.mean(image_coverages)) if image_coverages else float("nan")

        elapsed = time.time() - t0

        total_vecs = len(good_vecs) + len(bad_vecs)
        comp_ratio = (1490 + 548) / max(total_vecs, 1)
        mem_kb = (total_vecs * 768 * 4) / 1024

        return {
            "Config_Name": config.get("name", "unnamed"),
            "Category": config.get("category", "General"),
            "K_Good": len(good_vecs),
            "K_Anomaly": len(bad_vecs),
            "Total_Vectors": total_vecs,
            "Compression_Ratio": round(comp_ratio, 2),
            "Memory_KB": round(mem_kb, 1),
            "kNN_k": knn_k,
            "I-AUROC": round(i_auroc, 6),
            "I-AP": round(i_ap, 6),
            "I-F1": round(i_f1, 6),
            "P-AUROC": round(p_auroc, 6),
            "P-AP": round(p_ap, 6),
            "P-F1": round(p_f1, 6),
            "P-AUPRO": round(p_aupro, 6),
            "R-MissRate": round(r_miss, 6),
            "R-FP-RegionCount": int(total_fp),
            "R-PixelCoverage": round(r_cov, 6),
            "Elapsed_s": round(elapsed, 2),
        }


# =========================================================================
# 3. Main Experiment Suite
# =========================================================================

def run_track1_experiments(
    root: Path,
    output_csv: Path,
    output_json: Path,
    device: str = "cuda:0",
):
    print("=" * 105, flush=True)
    print(" TRACK 1: PROTOTYPE CLUSTERING & FEATURE BANK OPTIMIZATION (672 MODEL)", flush=True)
    print("=" * 105, flush=True)

    good_vecs = np.load(root / "good" / "vectors.npy")
    bad_vecs = np.load(root / "anomaly" / "vectors.npy")
    print(f"Original Feature Bank: Good={good_vecs.shape[0]} vecs, Anomaly={bad_vecs.shape[0]} vecs (Total={len(good_vecs)+len(bad_vecs)})", flush=True)

    cache_path = root / "preds" / "cached_eval_records.pkl"
    t0 = time.time()
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} test images in {time.time() - t0:.2f}s.", flush=True)

    evaluator = UltraFastEvaluator(records, device=device)
    results: List[Dict[str, Any]] = []

    def evaluate_and_log(name: str, category: str, g_vecs: np.ndarray, b_vecs: np.ndarray, knn_k: int = 3, query_patches: int = 3):
        cfg = {"name": name, "category": category, "knn_k": knn_k, "query_patches": query_patches}
        res = evaluator.evaluate(g_vecs, b_vecs, cfg)
        results.append(res)
        print(
            f"[{category:<15}] {name:<35} | G:{res['K_Good']:<4} A:{res['K_Anomaly']:<3} ({res['Compression_Ratio']}x) | "
            f"I-AUROC:{res['I-AUROC']:.4f} I-AP:{res['I-AP']:.4f} | "
            f"P-AUROC:{res['P-AUROC']:.4f} P-AP:{res['P-AP']:.4f} P-AUPRO:{res['P-AUPRO']:.4f} | "
            f"MissRate:{res['R-MissRate']*100:.2f}% FP-Reg:{res['R-FP-RegionCount']:<5} ({res['Elapsed_s']:.2f}s)",
            flush=True,
        )
        return res

    # ---------------------------------------------------------------------
    # 0. Baseline (Full Feature Bank)
    # ---------------------------------------------------------------------
    print("\n>>> 0. Baselines (Full Bank)", flush=True)
    evaluate_and_log("Full_Uncompressed_kNN3", "Baseline", good_vecs, bad_vecs, knn_k=3)
    evaluate_and_log("Full_Uncompressed_kNN1", "Baseline", good_vecs, bad_vecs, knn_k=1)
    evaluate_and_log("Full_Uncompressed_kNN5", "Baseline", good_vecs, bad_vecs, knn_k=5)

    # ---------------------------------------------------------------------
    # 1. K-Means Prototype Clustering (Centroid Prototypes)
    # Grid: K_good in [100, 300, 600, 1000], K_anomaly in [50, 100, 200, 400]
    # ---------------------------------------------------------------------
    print("\n>>> 1. K-Means Prototype Clustering (Centroid Prototypes)", flush=True)
    k_good_list = [100, 300, 600, 1000]
    k_bad_list = [50, 100, 200, 400]

    for kg in k_good_list:
        for kb in k_bad_list:
            g_proto = kmeans_clustering(good_vecs, kg, use_medoids=False, random_state=42)
            b_proto = kmeans_clustering(bad_vecs, kb, use_medoids=False, random_state=42)
            evaluate_and_log(f"KMeans_Centroid_G{kg}_A{kb}", "K-Means Centroid", g_proto, b_proto, knn_k=3)

    # ---------------------------------------------------------------------
    # 2. K-Means Medoid Prototypes (Real Exemplars)
    # ---------------------------------------------------------------------
    print("\n>>> 2. K-Means Medoid Prototypes (Real Exemplars)", flush=True)
    for kg in [300, 600, 1000]:
        for kb in [100, 200, 400]:
            g_med = kmeans_clustering(good_vecs, kg, use_medoids=True, random_state=42)
            b_med = kmeans_clustering(bad_vecs, kb, use_medoids=True, random_state=42)
            evaluate_and_log(f"KMeans_Medoid_G{kg}_A{kb}", "K-Means Medoid", g_med, b_med, knn_k=3)

    # ---------------------------------------------------------------------
    # 3. K-Center Greedy Coreset Selection
    # ---------------------------------------------------------------------
    print("\n>>> 3. K-Center Greedy Coreset Selection", flush=True)
    for kg in [100, 300, 600, 1000]:
        for kb in [50, 100, 200, 400]:
            g_core = kcenter_greedy_coreset(good_vecs, kg, random_state=42)
            b_core = kcenter_greedy_coreset(bad_vecs, kb, random_state=42)
            evaluate_and_log(f"Coreset_Greedy_G{kg}_A{kb}", "Coreset Greedy", g_core, b_core, knn_k=3)

    # ---------------------------------------------------------------------
    # 4. Boundary Margin Noise Cleaning & Density Outlier Trimming
    # ---------------------------------------------------------------------
    print("\n>>> 4. Boundary Margin Noise Cleaning & Outlier Trimming", flush=True)
    for margin in [1.0, 1.05, 1.10, 1.15, 1.20]:
        cg, cb, stats = remove_boundary_noise(good_vecs, bad_vecs, margin_ratio=margin, trim_isolated_percentile=0.0)
        evaluate_and_log(f"BoundaryClean_Margin_{margin:.2f}", "Boundary Clean", cg, cb, knn_k=3)

    for p_trim in [3.0, 5.0, 10.0]:
        cg, cb, stats = remove_boundary_noise(good_vecs, bad_vecs, margin_ratio=1.05, trim_isolated_percentile=p_trim)
        evaluate_and_log(f"BoundaryClean_M1.05_Trim_{p_trim:.0f}pct", "Density Trim", cg, cb, knn_k=3)

    # ---------------------------------------------------------------------
    # 5. Hybrid Optimization (Noise Cleaning + Prototype Clustering / Coreset)
    # ---------------------------------------------------------------------
    print("\n>>> 5. Hybrid Optimization (Noise Cleaning + Prototype Clustering / Coreset)", flush=True)
    clean_g, clean_b, _ = remove_boundary_noise(good_vecs, bad_vecs, margin_ratio=1.05, trim_isolated_percentile=5.0)

    for kg in [300, 600, 1000]:
        for kb in [100, 200, 400]:
            if kg < len(clean_g) and kb < len(clean_b):
                hg_km = kmeans_clustering(clean_g, kg, use_medoids=False, random_state=42)
                hb_km = kmeans_clustering(clean_b, kb, use_medoids=False, random_state=42)
                evaluate_and_log(f"Hybrid_Clean_KMeans_G{kg}_A{kb}", "Hybrid Clean+KM", hg_km, hb_km, knn_k=3)

                hg_co = kcenter_greedy_coreset(clean_g, kg, random_state=42)
                hb_co = kcenter_greedy_coreset(clean_b, kb, random_state=42)
                evaluate_and_log(f"Hybrid_Clean_Coreset_G{kg}_A{kb}", "Hybrid Clean+Core", hg_co, hb_co, knn_k=3)

    # ---------------------------------------------------------------------
    # 6. Retrieval k-NN Sweep on Champion Configurations
    # ---------------------------------------------------------------------
    print("\n>>> 6. Retrieval k-NN Sweep on Champion Prototypes", flush=True)
    for k_val in [1, 2, 3, 5, 7]:
        g_top = kmeans_clustering(clean_g, min(600, len(clean_g)), use_medoids=False, random_state=42)
        b_top = kmeans_clustering(clean_b, min(200, len(clean_b)), use_medoids=False, random_state=42)
        evaluate_and_log(f"Champion_CleanKM_G600_A200_kNN{k_val}", "kNN Sweep", g_top, b_top, knn_k=k_val)

    for k_val in [1, 2, 3, 5, 7]:
        g_top = kmeans_clustering(good_vecs, 600, use_medoids=False, random_state=42)
        b_top = kmeans_clustering(bad_vecs, 200, use_medoids=False, random_state=42)
        evaluate_and_log(f"Champion_RawKM_G600_A200_kNN{k_val}", "kNN Sweep", g_top, b_top, knn_k=k_val)

    # Save outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(results[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV results to {output_csv}", flush=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {output_json}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/wt/two_stages/base_patch_672")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_csv", default="/data/wt/two_stages/base_patch_672/track1_prototype_clustering_results.csv")
    parser.add_argument("--output_json", default="/data/wt/two_stages/base_patch_672/track1_prototype_clustering_results.json")
    args = parser.parse_args()

    run_track1_experiments(
        root=Path(args.root),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        device=args.device,
    )


if __name__ == "__main__":
    main()
