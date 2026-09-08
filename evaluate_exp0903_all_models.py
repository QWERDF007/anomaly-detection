import os
import sys
import time
import json
import glob
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# Add subdirectories to sys.path
ROOT = Path("/data/wt/anomaly-detection").resolve()
sys.path.insert(0, str(ROOT / "Dinomaly2"))
sys.path.insert(0, str(ROOT / "patchcore-inspection" / "src"))
sys.path.insert(0, str(ROOT))

from dataset import CustomDataset, get_data_transforms
from evaluate_dg_benchmark import (
    load_model as load_dinomaly_model,
    ShiftTransformSuite,
    cal_anomaly_maps,
    get_gaussian_kernel,
)
import patchcore.patchcore
import patchcore.common

def get_latest_checkpoint(dir_path: Path):
    if not dir_path.is_dir():
        return None
    ckpts = list(dir_path.glob("**/model.pth"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]

def get_patchcore_params(dir_path: Path):
    if not dir_path.is_dir():
        return None
    pkls = list(dir_path.glob("**/patchcore_params.pkl"))
    if not pkls:
        # check if directly in dir
        p = dir_path / "patchcore_params.pkl"
        if p.is_file():
            return p
        return None
    pkls.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pkls[0]

def eval_dinomaly_variant(
    ckpt_path: Path,
    test_txt: Path,
    device: str = "cuda:0",
    image_size: int = 448,
    crop_size: int = 448,
    batch_size: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"Evaluating Dinomaly2 Checkpoint: {ckpt_path} on {test_txt.name} ...")
    model = load_dinomaly_model(str(ckpt_path), device=device)
    model.eval()

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    test_data = CustomDataset(root=str(test_txt), transform=data_transform, gt_transform=gt_transform, phase="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # 1. Measure Latency & FPS
    dummy = torch.randn(1, 3, image_size, crop_size, device=device)
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    with torch.no_grad():
        for _ in range(10):
            en, de = model(dummy)
            am, _ = cal_anomaly_maps(en, de, image_size)
            _ = gaussian_kernel(am)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            en, de = model(dummy)
            am, _ = cal_anomaly_maps(en, de, image_size)
            _ = gaussian_kernel(am)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    lat_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    fps = 1000.0 / lat_ms
    vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0

    # 2. In-Domain Clean Test Evaluation
    clean_scores = []
    clean_labels = []
    px_preds = []
    px_gts = []
    has_px = False

    with torch.no_grad():
        for img, gt, label, _ in test_loader:
            img = img.to(device)
            en, de = model(img)
            am, _ = cal_anomaly_maps(en, de, image_size)
            am = gaussian_kernel(am)

            # Top-1% spatial pooling
            flat_am = am.flatten(1)
            topk = max(1, int(flat_am.shape[1] * 0.01))
            sp_score = torch.sort(flat_am, dim=1, descending=True)[0][:, :topk].mean(dim=1)

            clean_scores.extend(sp_score.cpu().numpy().tolist())
            clean_labels.extend(label.numpy().tolist())

            if gt is not None and gt.sum() > 0:
                has_px = True
            if gt is not None:
                gt_down = F.interpolate(gt.float(), size=(112, 112), mode="nearest")
                am_down = F.interpolate(am, size=(112, 112), mode="bilinear", align_corners=False)
                px_preds.append(am_down.squeeze(1).cpu().numpy())
                px_gts.append(gt_down.squeeze(1).cpu().numpy())

    scores_arr = np.array(clean_scores, dtype=np.float64)
    labels_arr = np.array(clean_labels, dtype=np.int64)

    i_auroc = float(roc_auc_score(labels_arr, scores_arr))
    i_ap = float(average_precision_score(labels_arr, scores_arr))
    precs, recs, _ = precision_recall_curve(labels_arr, scores_arr)
    f1s = 2 * precs * recs / (precs + recs + 1e-8)
    i_f1 = float(np.nanmax(f1s))

    p_auroc, p_ap, p_f1 = None, None, None
    if has_px and len(px_preds) > 0:
        all_px_preds = np.concatenate(px_preds).ravel()
        all_px_gts = (np.concatenate(px_gts).ravel() > 0.5).astype(np.uint8)
        p_auroc = float(roc_auc_score(all_px_gts, all_px_preds))
        p_ap = float(average_precision_score(all_px_gts, all_px_preds))
        p_precs, p_recs, _ = precision_recall_curve(all_px_gts, all_px_preds)
        p_f1s = 2 * p_precs * p_recs / (p_precs + p_recs + 1e-8)
        p_f1 = float(np.nanmax(p_f1s))

    clean_norm_scores = scores_arr[labels_arr == 0]
    clean_bad_scores = scores_arr[labels_arr == 1]

    # Threshold calibration
    tau_prod = float(np.percentile(clean_norm_scores, 99.0)) if len(clean_norm_scores) > 0 else 0.0
    clean_fpr_prod = float(np.mean(clean_norm_scores >= tau_prod) * 100) if len(clean_norm_scores) > 0 else 0.0

    tau_rec100 = float(np.min(clean_bad_scores)) if len(clean_bad_scores) > 0 else 0.0
    clean_fpr_rec100 = float(np.mean(clean_norm_scores >= tau_rec100) * 100) if len(clean_norm_scores) > 0 else 0.0

    tau_rec99 = float(np.percentile(clean_bad_scores, 1.0)) if len(clean_bad_scores) > 0 else 0.0
    clean_fpr_rec99 = float(np.mean(clean_norm_scores >= tau_rec99) * 100) if len(clean_norm_scores) > 0 else 0.0

    # 3. Distribution Generalization (DG) Evaluation
    seen_shifts = {
        'Seen Photometric': ShiftTransformSuite.seen_photometric,
        'Seen Illumination': ShiftTransformSuite.seen_illumination,
        'Seen Degradation': ShiftTransformSuite.seen_degradation,
        'Seen Geometric': ShiftTransformSuite.seen_geometric,
    }
    unseen_shifts = {
        'Unseen WhiteBalance': ShiftTransformSuite.unseen_white_balance,
        'Unseen SensorPoisson': ShiftTransformSuite.unseen_sensor_poisson,
        'Unseen DefocusBlur': ShiftTransformSuite.unseen_defocus_blur,
        'Unseen SCurveTone': ShiftTransformSuite.unseen_s_curve_tone,
        'Unseen SpecularGlare': ShiftTransformSuite.unseen_specular_glare,
    }

    # Shift evaluation measures False Positive Rate on clean normal data under shift
    norm_indices = [i for i, l in enumerate(labels_arr) if l == 0]
    norm_subset = torch.utils.data.Subset(test_data, norm_indices)
    norm_loader = torch.utils.data.DataLoader(norm_subset, batch_size=16, shuffle=False, num_workers=2)

    def eval_shift_dict(shift_dict):
        metrics = {}
        prod_fprs = []
        rec100_fprs = []
        rec99_fprs = []
        rng = np.random.default_rng(seed)

        for name, fn in shift_dict.items():
            s_scores = []
            with torch.no_grad():
                for img, _, _, _ in norm_loader:
                    img = img.to(device)
                    shifted_img = fn(img, rng)
                    en, de = model(shifted_img)
                    am, _ = cal_anomaly_maps(en, de, image_size)
                    am = gaussian_kernel(am)
                    flat_am = am.flatten(1)
                    topk = max(1, int(flat_am.shape[1] * 0.01))
                    sp_score = torch.sort(flat_am, dim=1, descending=True)[0][:, :topk].mean(dim=1)
                    s_scores.extend(sp_score.cpu().numpy().tolist())

            s_norm = np.array(s_scores, dtype=np.float64)
            fpr_prod = float(np.mean(s_norm >= tau_prod) * 100) if len(s_norm) > 0 else 0.0
            fpr_rec100 = float(np.mean(s_norm >= tau_rec100) * 100) if len(s_norm) > 0 else 0.0
            fpr_rec99 = float(np.mean(s_norm >= tau_rec99) * 100) if len(s_norm) > 0 else 0.0

            metrics[name] = {
                'fpr_prod': fpr_prod,
                'fpr_rec100': fpr_rec100,
                'fpr_rec99': fpr_rec99,
                'mean_norm_score': float(np.mean(s_norm)) if len(s_norm) > 0 else 0.0,
            }
            prod_fprs.append(fpr_prod)
            rec100_fprs.append(fpr_rec100)
            rec99_fprs.append(fpr_rec99)

        summary = {
            'mean_fpr_prod': float(np.mean(prod_fprs)),
            'worst_fpr_prod': float(np.max(prod_fprs)),
            'mean_fpr_rec100': float(np.mean(rec100_fprs)),
            'mean_fpr_rec99': float(np.mean(rec99_fprs)),
        }
        return summary, metrics

    seen_summary, seen_details = eval_shift_dict(seen_shifts)
    unseen_summary, unseen_details = eval_shift_dict(unseen_shifts)

    return {
        'clean': {
            'i_auroc': i_auroc,
            'i_ap': i_ap,
            'i_f1': i_f1,
            'p_auroc': p_auroc,
            'p_ap': p_ap,
            'p_f1': p_f1,
            'clean_fpr_prod': clean_fpr_prod,
            'clean_fpr_rec100': clean_fpr_rec100,
            'clean_fpr_rec99': clean_fpr_rec99,
            'tau_prod': tau_prod,
            'tau_rec100': tau_rec100,
            'tau_rec99': tau_rec99,
            'mean_norm': float(np.mean(clean_norm_scores)),
            'mean_bad': float(np.mean(clean_bad_scores)),
        },
        'efficiency': {
            'latency_ms': lat_ms,
            'fps': fps,
            'vram_gb': vram_gb,
        },
        'seen_summary': seen_summary,
        'seen_details': seen_details,
        'unseen_summary': unseen_summary,
        'unseen_details': unseen_details,
    }

def eval_patchcore_variant(
    pkl_path: Path,
    test_txt: Path,
    device: str = "cuda:0",
    image_size: int = 448,
    seed: int = 42,
) -> Dict[str, Any]:
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(
        load_path=str(pkl_path.parent),
        device=device,
        prepend=pkl_path.name[:-len("patchcore_params.pkl")],
        nn_method=patchcore.common.FaissNN(on_gpu=True, device_id=0),
    )

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_data = CustomDataset(root=str(test_txt), transform=data_transform, gt_transform=gt_transform, phase="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4)

    # 1. Measure Latency & FPS
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    with torch.no_grad():
        for _ in range(5):
            _ = model._predict(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(25):
            _ = model._predict(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    lat_ms = (time.perf_counter() - t0) * 1000.0 / 25.0
    fps = 1000.0 / lat_ms
    vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else 0.0

    # 2. In-Domain Clean Test Evaluation
    clean_scores = []
    clean_labels = []
    px_preds = []
    px_gts = []
    has_px = False

    with torch.no_grad():
        for img, gt, label, _ in test_loader:
            scores, masks = model._predict(img.to(device))
            clean_scores.extend(scores.tolist() if isinstance(scores, np.ndarray) else scores)
            clean_labels.extend(label.numpy().tolist())

            if gt is not None and gt.sum() > 0:
                has_px = True
            if gt is not None:
                gt_down = F.interpolate(gt.float(), size=(112, 112), mode="nearest")
                masks_t = torch.from_numpy(np.array(masks)).unsqueeze(1).float()
                am_down = F.interpolate(masks_t, size=(112, 112), mode="bilinear", align_corners=False)
                px_preds.append(am_down.squeeze(1).numpy())
                px_gts.append(gt_down.squeeze(1).numpy())

    scores_arr = np.array(clean_scores, dtype=np.float64)
    labels_arr = np.array(clean_labels, dtype=np.int64)

    i_auroc = float(roc_auc_score(labels_arr, scores_arr))
    i_ap = float(average_precision_score(labels_arr, scores_arr))
    precs, recs, _ = precision_recall_curve(labels_arr, scores_arr)
    f1s = 2 * precs * recs / (precs + recs + 1e-8)
    i_f1 = float(np.nanmax(f1s))

    p_auroc, p_ap, p_f1 = None, None, None
    if has_px and len(px_preds) > 0:
        all_px_preds = np.concatenate(px_preds).ravel()
        all_px_gts = (np.concatenate(px_gts).ravel() > 0.5).astype(np.uint8)
        p_auroc = float(roc_auc_score(all_px_gts, all_px_preds))
        p_ap = float(average_precision_score(all_px_gts, all_px_preds))
        p_precs, p_recs, _ = precision_recall_curve(all_px_gts, all_px_preds)
        p_f1s = 2 * p_precs * p_recs / (p_precs + p_recs + 1e-8)
        p_f1 = float(np.nanmax(p_f1s))

    clean_norm_scores = scores_arr[labels_arr == 0]
    clean_bad_scores = scores_arr[labels_arr == 1]

    tau_prod = float(np.percentile(clean_norm_scores, 99.0)) if len(clean_norm_scores) > 0 else 0.0
    clean_fpr_prod = float(np.mean(clean_norm_scores >= tau_prod) * 100) if len(clean_norm_scores) > 0 else 0.0

    tau_rec100 = float(np.min(clean_bad_scores)) if len(clean_bad_scores) > 0 else 0.0
    clean_fpr_rec100 = float(np.mean(clean_norm_scores >= tau_rec100) * 100) if len(clean_norm_scores) > 0 else 0.0

    tau_rec99 = float(np.percentile(clean_bad_scores, 1.0)) if len(clean_bad_scores) > 0 else 0.0
    clean_fpr_rec99 = float(np.mean(clean_norm_scores >= tau_rec99) * 100) if len(clean_norm_scores) > 0 else 0.0

    # 3. Distribution Generalization (DG) Evaluation
    seen_shifts = {
        'Seen Photometric': ShiftTransformSuite.seen_photometric,
        'Seen Illumination': ShiftTransformSuite.seen_illumination,
        'Seen Degradation': ShiftTransformSuite.seen_degradation,
        'Seen Geometric': ShiftTransformSuite.seen_geometric,
    }
    unseen_shifts = {
        'Unseen WhiteBalance': ShiftTransformSuite.unseen_white_balance,
        'Unseen SensorPoisson': ShiftTransformSuite.unseen_sensor_poisson,
        'Unseen DefocusBlur': ShiftTransformSuite.unseen_defocus_blur,
        'Unseen SCurveTone': ShiftTransformSuite.unseen_s_curve_tone,
        'Unseen SpecularGlare': ShiftTransformSuite.unseen_specular_glare,
    }

    # Shift evaluation measures False Positive Rate on clean normal data under shift
    norm_indices = [i for i, l in enumerate(labels_arr) if l == 0]
    norm_subset = torch.utils.data.Subset(test_data, norm_indices)
    norm_loader = torch.utils.data.DataLoader(norm_subset, batch_size=8, shuffle=False, num_workers=2)

    def eval_patchcore_shift_dict(shift_dict):
        metrics = {}
        prod_fprs = []
        rec100_fprs = []
        rec99_fprs = []
        rng = np.random.default_rng(seed)

        for name, fn in shift_dict.items():
            s_scores = []
            with torch.no_grad():
                for img, _, _, _ in norm_loader:
                    img = img.to(device)
                    shifted_img = fn(img, rng)
                    scores, _ = model._predict(shifted_img)
                    s_scores.extend(scores.tolist() if isinstance(scores, np.ndarray) else scores)

            s_norm = np.array(s_scores, dtype=np.float64)
            fpr_prod = float(np.mean(s_norm >= tau_prod) * 100) if len(s_norm) > 0 else 0.0
            fpr_rec100 = float(np.mean(s_norm >= tau_rec100) * 100) if len(s_norm) > 0 else 0.0
            fpr_rec99 = float(np.mean(s_norm >= tau_rec99) * 100) if len(s_norm) > 0 else 0.0

            metrics[name] = {
                'fpr_prod': fpr_prod,
                'fpr_rec100': fpr_rec100,
                'fpr_rec99': fpr_rec99,
                'mean_norm_score': float(np.mean(s_norm)) if len(s_norm) > 0 else 0.0,
            }
            prod_fprs.append(fpr_prod)
            rec100_fprs.append(fpr_rec100)
            rec99_fprs.append(fpr_rec99)

        summary = {
            'mean_fpr_prod': float(np.mean(prod_fprs)),
            'worst_fpr_prod': float(np.max(prod_fprs)),
            'mean_fpr_rec100': float(np.mean(rec100_fprs)),
            'mean_fpr_rec99': float(np.mean(rec99_fprs)),
        }
        return summary, metrics

    seen_summary, seen_details = eval_patchcore_shift_dict(seen_shifts)
    unseen_summary, unseen_details = eval_patchcore_shift_dict(unseen_shifts)

    return {
        'clean': {
            'i_auroc': i_auroc,
            'i_ap': i_ap,
            'i_f1': i_f1,
            'p_auroc': p_auroc,
            'p_ap': p_ap,
            'p_f1': p_f1,
            'clean_fpr_prod': clean_fpr_prod,
            'clean_fpr_rec100': clean_fpr_rec100,
            'clean_fpr_rec99': clean_fpr_rec99,
            'tau_prod': tau_prod,
            'tau_rec100': tau_rec100,
            'tau_rec99': tau_rec99,
            'mean_norm': float(np.mean(clean_norm_scores)),
            'mean_bad': float(np.mean(clean_bad_scores)),
        },
        'efficiency': {
            'latency_ms': lat_ms,
            'fps': fps,
            'vram_gb': vram_gb,
        },
        'seen_summary': seen_summary,
        'seen_details': seen_details,
        'unseen_summary': unseen_summary,
        'unseen_details': unseen_details,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'VC', '中石'])
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    exp_dir = Path("/data/wt/exp0903")
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    
    all_models = [
        # VC
        ("VC", "PatchCore (Coreset 10%)", "patchcore", exp_dir / "VC_patchcore_n200", exp_dir / "data_splits" / "VC_test.txt"),
        ("VC", "Dinomaly2 Base (Baseline)", "dinomaly", exp_dir / "VC_dinomaly2_base", exp_dir / "data_splits" / "VC_test.txt"),
        ("VC", "Source-DG (Adapter)", "dinomaly", exp_dir / "VC_source_dg_adapter", exp_dir / "data_splits" / "VC_test.txt"),
        ("VC", "Source-DG (Full + UOT)", "dinomaly", exp_dir / "VC_source_dg_full_uot", exp_dir / "data_splits" / "VC_test.txt"),
        # 中石
        ("中石", "PatchCore (Coreset 10%)", "patchcore", exp_dir / "中石_patchcore_n56", exp_dir / "data_splits" / "中石_test.txt"),
        ("中石", "Dinomaly2 Base (Baseline)", "dinomaly", exp_dir / "中石_dinomaly2_base", exp_dir / "data_splits" / "中石_test.txt"),
        ("中石", "Source-DG (Adapter)", "dinomaly", exp_dir / "中石_source_dg_adapter", exp_dir / "data_splits" / "中石_test.txt"),
        ("中石", "Source-DG (Full + UOT)", "dinomaly", exp_dir / "中石_source_dg_full_uot", exp_dir / "data_splits" / "中石_test.txt"),
    ]

    if args.dataset != 'all':
        models_to_eval = [m for m in all_models if m[0] == args.dataset]
    else:
        models_to_eval = all_models

    all_results = {}

    for ds_name, model_name, m_type, m_dir, test_txt in models_to_eval:
        key = f"{ds_name}_{model_name}"
        print(f"\n=======================================================")
        print(f"EVALUATING: [{ds_name}] {model_name}")
        print(f"=======================================================")

        if m_type == "patchcore":
            pkl_p = get_patchcore_params(m_dir)
            if not pkl_p:
                print(f"[ERROR] Could not find patchcore_params.pkl in {m_dir}")
                continue
            res = eval_patchcore_variant(pkl_p, test_txt, device=device)
        else:
            ckpt_p = get_latest_checkpoint(m_dir)
            if not ckpt_p:
                print(f"[ERROR] Could not find model.pth in {m_dir}")
                continue
            res = eval_dinomaly_variant(ckpt_p, test_txt, device=device)

        all_results[key] = {
            'dataset': ds_name,
            'model': model_name,
            'type': m_type,
            'model_dir': str(m_dir),
            'metrics': res,
        }

        # Quick print summary
        cl = res['clean']
        eff = res['efficiency']
        sn = res['seen_summary']
        un = res['unseen_summary']
        print(f"  Clean I-AUROC: {cl['i_auroc']:.4f} | I-F1: {cl['i_f1']:.4f} | P-AUROC: {cl['p_auroc'] if cl['p_auroc'] is not None else 0:.4f}")
        print(f"  Clean FPR@Q99: {cl['clean_fpr_prod']:.2f}% | FPR@Rec100: {cl['clean_fpr_rec100']:.2f}%")
        print(f"  Seen Mean FPR: {sn['mean_fpr_prod']:.2f}% | Worst: {sn['worst_fpr_prod']:.2f}%")
        print(f"  Unseen Mean FPR: {un['mean_fpr_prod']:.2f}% | Worst: {un['worst_fpr_prod']:.2f}%")
        print(f"  Efficiency: {eff['latency_ms']:.1f} ms ({eff['fps']:.1f} FPS) | VRAM: {eff['vram_gb']:.2f} GB")

    out_json = exp_dir / f"benchmark_results_{args.dataset}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved benchmark results to: {out_json}")

    # Merge into main benchmark_results.json
    main_json = exp_dir / "benchmark_results.json"
    merged = {}
    if main_json.is_file():
        try:
            with open(main_json, "r", encoding="utf-8") as f:
                merged = json.load(f)
        except:
            merged = {}
    merged.update(all_results)
    with open(main_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Merged into: {main_json}")

if __name__ == '__main__':
    main()
