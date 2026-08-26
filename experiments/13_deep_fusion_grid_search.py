"""Experiment 13: Deep Fusion Grid Search on 672 Resolution (15k vit_base).
Explores optimal synergy between:
1. Hard Anomaly Trigger
2. Confirmed Good Suppression
3. Background Noise Floor Subtraction
4. Morphological Opening
5. Connected Component Area Filtering

Usage:
    python experiments/13_deep_fusion_grid_search.py
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

def main():
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

    print(f'Loaded {len(samples)} samples. Starting Deep Fusion Grid Search...')
    
    good_threshold = 0.014
    anomaly_threshold = 0.030
    
    def evaluate_combination(name, smap_fn, thresh=0.014):
        adj_scores = []
        labels = []
        eval_maps = []
        eval_masks = []
        
        for s in samples:
            smap = smap_fn(s['score_map'].copy(), s['raw_score'], s['detail'])
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
        
        print(f"{name:<52} | {i_auroc:<7.4f} | {i_ap:<7.4f} | {p_auroc:<7.4f} | {p_ap:<7.4f} | {p_aupro:<7.4f} | {miss_rate*100:<5.2f}% | {fp_count:<10.0f}", flush=True)

    print('================================================================================================================================', flush=True)
    header = f"{'Configuration':<52} | {'I-AUROC':<7} | {'I-AP':<7} | {'P-AUROC':<7} | {'P-AP':<7} | {'P-AUPRO':<7} | {'Miss%':<6} | {'FP Regions':<10}"
    print(header, flush=True)
    print('================================================================================================================================', flush=True)
    
    evaluate_combination('00. Baseline (Raw Score Map)', lambda m, s, d: m)
    
    def process_pipeline(smap, raw_s, detail, hard_t=0.15, supp_ratio=0.1, p_floor=20, k_open=3, min_area=30):
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
                if da <= hard_t:
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
                    
        # Morphological Opening
        if k_open > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
            smap = cv2.morphologyEx(smap, cv2.MORPH_OPEN, kernel)
            
        # Background Floor Subtraction
        if p_floor > 0:
            bg = float(np.percentile(smap, p_floor))
            smap = np.maximum(smap - bg, 0.0)
            
        # Area Filter
        if min_area > 0:
            binary = smap >= good_threshold
            lbl = measure.label(binary)
            for i in range(1, lbl.max() + 1):
                if (lbl == i).sum() < min_area:
                    smap[lbl == i] = np.clip(smap[lbl == i], 0.0, good_threshold * 0.8)
                    
        return smap

    # Systematic Grid Sweep
    for p in [0, 15, 20, 25]:
        for k in [0, 3, 5]:
            for a in [0, 25, 50, 100]:
                name = f'Fusion (p={p}%, k={k}, area={a}px)'
                evaluate_combination(name, lambda m, s, d, pf=p, ko=k, ma=a: process_pipeline(m, s, d, p_floor=pf, k_open=ko, min_area=ma))

if __name__ == '__main__':
    main()
