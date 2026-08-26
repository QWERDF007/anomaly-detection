"""Experiment 9: Unified High-Precision Two-Stage Pipeline Integration.
Combines:
1. 15k vit_base Deep Backbone (672)
2. Hard Anomaly Trigger (d_ano <= 0.15)
3. Confirmed Good Suppression
4. Adaptive Background Floor Subtraction (p=20%)
5. Morphological Opening (k=3)
6. Guided Filter Boundary Refinement

Usage:
    python experiments/09_unified_high_precision_pipeline.py
"""

import json
from pathlib import Path
import cv2
import numpy as np
from skimage import measure
import sys
sys.path.insert(0, '/data/wt/anomaly-detection/Dinomaly2')

from dinomaly_evaluation import (
    safe_auroc,
    safe_ap,
    safe_aupro,
    region_detection_metrics,
    training_image_score,
)
from dinomaly_two_stage import load_mask

def run_unified_evaluation():
    gt_root = Path('/data/wt/ramdisk/leishi_026/ground_truth')
    root_15k = Path('/data/wt/two_stages/base_672_15k')
    details_dir = root_15k / 'preds' / 'details'
    score_maps_dir = root_15k / 'preds' / 'score_maps'
    
    detail_files = sorted(list(details_dir.rglob('*.json')))
    samples = []
    for df in detail_files:
        d = json.loads(df.read_text())
        img_rel = Path(d['image_relative'])
        ds_label = str(d.get('dataset_label', ''))
        score_path = score_maps_dir / img_rel.with_suffix('.npy')
        if not score_path.is_file():
            continue
        score_map = np.load(score_path).astype(np.float32)
        raw_score = float(training_image_score(score_map))
        
        gt_mask = None
        if ds_label != 'good':
            for ext in ['.png', '.jpg', '.jpeg', '.npy', '.tif', '.json']:
                gp = gt_root / img_rel.with_suffix(ext)
                if gp.is_file():
                    gt_mask = load_mask(gp, score_map.shape[:2])
                    break
                
        samples.append({
            'score_map': score_map,
            'raw_score': raw_score,
            'gt_mask': gt_mask,
            'detail': d,
            'ds_label': ds_label,
        })

    print(f'Loaded {len(samples)} samples for Unified Pipeline Evaluation.')
    
    good_threshold = 0.014
    anomaly_threshold = 0.030
    
    def evaluate_pipeline(name, apply_fn, thresh=0.014):
        adj_scores = []
        labels = []
        eval_maps = []
        eval_masks = []
        
        for s in samples:
            smap = apply_fn(s['score_map'].copy(), s['raw_score'], s['detail'])
            adj_s = float(training_image_score(smap))
            is_bad = 1 if s['ds_label'] != 'good' else 0
            adj_scores.append(adj_s)
            labels.append(is_bad)
            if s['gt_mask'] is not None:
                eval_maps.append(cv2.resize(smap, (672, 672), interpolation=cv2.INTER_LINEAR))
                eval_masks.append(cv2.resize(s['gt_mask'].astype(np.uint8), (672, 672), interpolation=cv2.INTER_NEAREST))
                
        i_auroc = safe_auroc(labels, adj_scores)
        i_ap = safe_ap(labels, adj_scores)
        
        eval_maps_arr = np.stack(eval_maps)
        eval_masks_arr = np.stack(eval_masks)
        
        flat_maps = eval_maps_arr[:, ::2, ::2].flatten()
        flat_masks = eval_masks_arr[:, ::2, ::2].flatten()
        p_auroc = safe_auroc(flat_masks, flat_maps)
        p_ap = safe_ap(flat_masks, flat_maps)
        
        p_aupro = safe_aupro(eval_masks_arr[:, ::4, ::4], eval_maps_arr[:, ::4, ::4], show_progress=False)
        reg_metrics = region_detection_metrics(eval_masks_arr, eval_maps_arr, threshold=thresh)
        miss_rate = reg_metrics.get('R-MissRate', 0.0)
        fp_count = reg_metrics.get('R-FP-RegionCount', 0.0)
        
        print(f"{name:<48} | {i_auroc:<7.4f} | {i_ap:<7.4f} | {p_auroc:<7.4f} | {p_ap:<7.4f} | {p_aupro:<7.4f} | {miss_rate*100:<5.2f}% | {fp_count:<10.0f}", flush=True)

    print('============================================================================================================================', flush=True)
    header = f"{'Pipeline Configuration':<48} | {'I-AUROC':<7} | {'I-AP':<7} | {'P-AUROC':<7} | {'P-AP':<7} | {'P-AUPRO':<7} | {'Miss%':<6} | {'FP Regions':<10}"
    print(header, flush=True)
    print('============================================================================================================================', flush=True)
    
    # 1. Raw Baseline
    evaluate_pipeline('1. Raw 15k Model Baseline', lambda m, s, d: m)
    
    # 2. Hard Anomaly Trigger
    def hard_trigger_fn(smap, raw_s, detail):
        if good_threshold <= raw_s <= anomaly_threshold:
            for r in detail.get('regions', []):
                da = float(r.get('anomaly_distance', 1.0))
                dg = float(r.get('good_distance', 1.0))
                bbox = [int(v) for v in r.get('bbox_original', [0,0,smap.shape[0],smap.shape[1]])]
                r0, c0, r1, c1 = max(0, bbox[0]), max(0, bbox[1]), min(smap.shape[0], bbox[2]), min(smap.shape[1], bbox[3])
                sub_s = smap[r0:r1, c0:c1]
                if sub_s.size == 0:
                    continue
                max_s = float(np.max(sub_s))
                if da <= 0.15:
                    w = (sub_s / max_s) if max_s > 1e-8 else 1.0
                    smap[r0:r1, c0:c1] = np.clip(sub_s + 0.008 * w, 0.0, None)
                elif dg < da:
                    margin = (da - dg) / (da + dg + 1e-8)
                    w = (sub_s / max_s) if max_s > 1e-8 else 1.0
                    smap[r0:r1, c0:c1] = np.clip(sub_s - 0.004 * margin * w, 0.0, None)
                else:
                    margin = (dg - da) / (da + dg + 1e-8)
                    w = (sub_s / max_s) if max_s > 1e-8 else 1.0
                    smap[r0:r1, c0:c1] = np.clip(sub_s + 0.008 * margin * w, 0.0, None)
        return smap
    evaluate_pipeline('2. Two-Stage + Hard Trigger (d_ano<=0.15)', hard_trigger_fn)
    
    # 3. Two-Stage + Good Suppression
    def good_supp_fn(smap, raw_s, detail, supp_ratio=0.1):
        if good_threshold <= raw_s <= anomaly_threshold:
            for r in detail.get('regions', []):
                da = float(r.get('anomaly_distance', 1.0))
                dg = float(r.get('good_distance', 1.0))
                bbox = [int(v) for v in r.get('bbox_original', [0,0,smap.shape[0],smap.shape[1]])]
                r0, c0, r1, c1 = max(0, bbox[0]), max(0, bbox[1]), min(smap.shape[0], bbox[2]), min(smap.shape[1], bbox[3])
                sub_s = smap[r0:r1, c0:c1]
                if sub_s.size == 0:
                    continue
                max_s = float(np.max(sub_s))
                if da <= 0.15:
                    w = (sub_s / max_s) if max_s > 1e-8 else 1.0
                    smap[r0:r1, c0:c1] = np.clip(sub_s + 0.008 * w, 0.0, None)
                elif dg < da:
                    conf = (da - dg) / (da + dg + 1e-8)
                    decay = max(0.0, 1.0 - conf * (1.0 - supp_ratio))
                    smap[r0:r1, c0:c1] = sub_s * decay
                else:
                    margin = (dg - da) / (da + dg + 1e-8)
                    w = (sub_s / max_s) if max_s > 1e-8 else 1.0
                    smap[r0:r1, c0:c1] = np.clip(sub_s + 0.008 * margin * w, 0.0, None)
        return smap
    evaluate_pipeline('3. Hard Trigger + Good Suppression (ratio=0.1)', good_supp_fn)
    
    # 4. Unified Full Pipeline (+ Morph Opening + BG Floor Subtraction)
    def unified_full_fn(smap, raw_s, detail, p_floor=20, k_size=3):
        smap = good_supp_fn(smap, raw_s, detail, supp_ratio=0.1)
        # Morphological Opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        smap = cv2.morphologyEx(smap, cv2.MORPH_OPEN, kernel)
        # Background Floor Subtraction
        bg = float(np.percentile(smap, p_floor))
        smap = np.maximum(smap - bg, 0.0)
        return smap
    
    for p in [15, 20, 25]:
        evaluate_pipeline(f'4. Unified Full Pipeline (p={p}%, k=3)', lambda m, s, d, pf=p: unified_full_fn(m, s, d, p_floor=pf, k_size=3))

if __name__ == '__main__':
    run_unified_evaluation()
