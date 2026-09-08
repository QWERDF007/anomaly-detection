#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive Multi-GPU Evaluation & Threshold Sweep for 透气膜 Dataset."""

import os, sys, glob, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import faiss
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
import concurrent.futures

ROOT = Path("/data/wt/anomaly-detection")
DINOMALY2_DIR = ROOT / "Dinomaly2"
if str(DINOMALY2_DIR) not in sys.path:
    sys.path.insert(0, str(DINOMALY2_DIR))

PATCHCORE_DIR = ROOT / "patchcore-inspection"
if str(PATCHCORE_DIR) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_DIR))
    sys.path.insert(0, str(PATCHCORE_DIR / "src"))

import patchcore.patchcore, patchcore.common
if "utils" in sys.modules and not hasattr(sys.modules["utils"], "cal_anomaly_maps"):
    del sys.modules["utils"]
import utils
from utils import cal_anomaly_maps, get_gaussian_kernel
from models import vit_encoder
from models.uad import Dinomaly
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from functools import partial

outs_dir = Path("/data/wt/report/0827-透气膜")
splits_dir = outs_dir / "data_splits"
bank_data_path = Path("/data/wt/ramdisk/透气膜/建库数据")
test_full_txt = splits_dir / "test_full.txt"

test_lines = [l.strip() for l in test_full_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
test_paths = [Path(l.split()[0]) for l in test_lines]
y_true = np.array([int(l.split()[1]) for l in test_lines], dtype=int)
print(f"Loaded Full Test Set: {len(test_paths)} images (OK={int((y_true==0).sum())}, NG={int((y_true==1).sum())})")

# 1. Parse LabelMe JSONs in Bank
def parse_labelme_bank(bank_dir):
    json_files = sorted(bank_dir.glob("*.json"))
    bank_items = []
    for jf in json_files:
        stem = jf.stem
        img_cands = [bank_dir / f"{stem}{ext}" for ext in [".png", ".jpg", ".bmp", ".jpeg", ".PNG", ".JPG"]]
        img_p = next((p for p in img_cands if p.is_file()), None)
        if not img_p: continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        polys = {"ad": [], "good": []}
        for s in data.get("shapes", []):
            lbl = s.get("label", "").lower()
            k = "ad" if lbl in ["ad", "ng", "anomaly", "defect", "abnormal"] else ("good" if lbl in ["good", "ok", "normal"] else None)
            if not k: continue
            pts = s.get("points", [])
            if len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                poly = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
            elif len(pts) >= 3:
                poly = [(int(p[0]), int(p[1])) for p in pts]
            else: continue
            polys[k].append(poly)
        bank_items.append((img_p, polys))
    return bank_items

bank_items = parse_labelme_bank(bank_data_path)
print(f"Parsed {len(bank_items)} labeled images in feature bank.")

tasks = []
sizes = [224, 448, 672]
ns = [20, 50, 100, 150]

for s in sizes:
    for n in ns:
        din_cands = sorted(list(outs_dir.glob(f"dinomaly2_n{n}_s{s}_seed2024/*/model.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
        pat_cands = sorted(list(outs_dir.glob(f"patchcore_n{n}_s{s}_seed2024/*/*patchcore_params.pkl")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not din_cands or not pat_cands:
            print(f"[warn] Missing candidates for N={n} Size={s}")
            continue
        din_model = din_cands[0]
        pat_pkl = pat_cands[0]
        out_e2e = outs_dir / f"e2e_out_n{n}_s{s}"
        save_bank = outs_dir / f"dinomaly2_n{n}_s{s}_seed2024" / "feature_bank.npz"
        tasks.append((n, s, din_model, pat_pkl, out_e2e, save_bank))

print(f"Total valid tasks to evaluate: {len(tasks)}")

def run_single_eval(args_tuple):
    idx, (n, s, din_model_path, pat_pkl_path, out_e2e, save_bank_path) = args_tuple
    gpu_id = idx % 8
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
    print(f"[{device}] Starting Task N={n} Size={s}...")

    # 1. Load Dinomaly2 Model
    ckpt = torch.load(str(din_model_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt: ckpt = ckpt["model"]

    backbone = "dinov2reg_vit_small_14"
    embed_dim, num_heads = 384, 6
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=0.4)),
        nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=0.4), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=0.4)),
    ])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=partial(LinearAttention2, eps=1e-8))
        for _ in range(8)
    ])
    encoder = vit_encoder.load(backbone)
    din_model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers, remove_class_token=False, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder, context_aware_recenter=1)
    din_model.load_state_dict(ckpt, strict=True)
    din_model.to(device).eval()

    din_transform = transforms.Compose([
        transforms.Resize((s, s)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, channels=1).to(device)

    # 2. Extract ROI Feature Bank
    ab_feats_list = []
    nor_feats_list = []
    with torch.no_grad():
        for img_p, polys in bank_items:
            img = Image.open(img_p).convert("RGB")
            orig_w, orig_h = img.size
            t = din_transform(img).unsqueeze(0).to(device)
            en_o, _ = din_model(t)
            feat = en_o[-1][0].permute(1, 2, 0).float().cpu().numpy()
            Hf, Wf, _ = feat.shape

            for key, plist in polys.items():
                if not plist: continue
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for poly in plist:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                mask_r = cv2.resize(mask, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
                idx_y, idx_x = np.where(mask_r > 0)
                if len(idx_y) > 0:
                    vecs = feat[idx_y, idx_x]
                    if key == "ad": ab_feats_list.append(vecs)
                    else: nor_feats_list.append(vecs)

    ab_feats = np.concatenate(ab_feats_list, axis=0) if ab_feats_list else np.empty((0, embed_dim), dtype=np.float32)
    nor_feats = np.concatenate(nor_feats_list, axis=0) if nor_feats_list else np.empty((0, embed_dim), dtype=np.float32)
    if len(ab_feats) > 0: faiss.normalize_L2(ab_feats)
    if len(nor_feats) > 0: faiss.normalize_L2(nor_feats)

    ab_idx = faiss.IndexFlatIP(embed_dim)
    nor_idx = faiss.IndexFlatIP(embed_dim)
    if len(ab_feats) > 0: ab_idx.add(np.ascontiguousarray(ab_feats, dtype=np.float32))
    if len(nor_feats) > 0: nor_idx.add(np.ascontiguousarray(nor_feats, dtype=np.float32))

    # Save bank npz
    save_bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_bank_path, ab_features=ab_feats, nor_features=nor_feats)

    # 3. Evaluate Dinomaly2 & Two-Stage E2E
    batch_size = 2 if s >= 672 else 4
    din_scores_all = []
    e2e_scores_all = []
    k_top = max(1, int(s * s * 0.01))

    low_thr, high_thr = 0.020, 0.045
    t_start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(test_paths), batch_size):
            b_paths = test_paths[i:i + batch_size]
            imgs = [din_transform(Image.open(p).convert("RGB")) for p in b_paths]
            b_t = torch.stack(imgs).to(device)
            en_o, de_o = din_model(b_t)
            amaps, _ = cal_anomaly_maps(en_o, de_o, s)
            amaps = gaussian_kernel(amaps)

            for j in range(len(b_paths)):
                amap = amaps[j, 0].float().cpu().numpy()
                raw_s = float(np.sort(amap.flatten())[-k_top:].mean())
                din_scores_all.append(raw_s)

                feat = en_o[-1][j].permute(1, 2, 0).float().cpu().numpy()
                Hf, Wf, _ = feat.shape
                amap_r = cv2.resize(amap, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                unc_mask = (amap_r > low_thr) & (amap_r < high_thr)
                if np.any(unc_mask) and ab_idx.ntotal > 0 and nor_idx.ntotal > 0:
                    unc_idx = np.where(unc_mask)
                    unc_feats = np.ascontiguousarray(feat[unc_idx], dtype=np.float32)
                    faiss.normalize_L2(unc_feats)
                    ab_ip, _ = ab_idx.search(unc_feats, 1)
                    nor_ip, _ = nor_idx.search(unc_feats, 1)
                    is_ab = (1.0 - ab_ip[:, 0]) < (1.0 - nor_ip[:, 0])
                    amap_r[unc_idx] = np.where(is_ab, 1.5 * high_thr, low_thr * 0.5)
                final_amap = cv2.resize(amap_r, (s, s), interpolation=cv2.INTER_LINEAR)
                cor_s = float(np.sort(final_amap.flatten())[-k_top:].mean())
                e2e_scores_all.append(cor_s)

    e2e_sec = time.perf_counter() - t_start
    fps = len(test_paths) / e2e_sec

    # 4. Evaluate PatchCore
    pat_model = patchcore.patchcore.PatchCore(device)
    pat_model.load_from_path(
        load_path=str(pat_pkl_path.parent),
        device=device,
        prepend=pat_pkl_path.name[:-len("patchcore_params.pkl")],
        nn_method=patchcore.common.FaissNN(on_gpu=True, num_workers=4)
    )
    pat_transform = transforms.Compose([
        transforms.Resize(s, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(s),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pat_scores_all = []
    with torch.no_grad():
        for p in test_paths:
            img = Image.open(p).convert("RGB")
            t = pat_transform(img).unsqueeze(0).to(device)
            sc, _ = pat_model.predict(t)
            pat_scores_all.append(float(sc[0]))

    # Calculate metrics
    din_scores = np.array(din_scores_all, dtype=np.float32)
    e2e_scores = np.array(e2e_scores_all, dtype=np.float32)
    pat_scores = np.array(pat_scores_all, dtype=np.float32)

    def calc_model_metrics(scores):
        auc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))
        p_arr, r_arr, t_arr = precision_recall_curve(y_true, scores)
        f1_arr = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
        b_idx = np.argmax(f1_arr)
        opt_f1 = float(f1_arr[b_idx])
        opt_th = float(t_arr[min(b_idx, len(t_arr) - 1)])
        preds = (scores >= opt_th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        return {
            "auc": auc, "ap": ap, "f1": opt_f1, "th": opt_th,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        }

    m_din = calc_model_metrics(din_scores)
    m_e2e = calc_model_metrics(e2e_scores)
    m_pat = calc_model_metrics(pat_scores)

    # 5. Experimental Multi-Threshold Sweep
    th_grid = np.linspace(float(e2e_scores.min()), float(e2e_scores.max()), 100)
    sweep_list = []
    for th in th_grid:
        p_bin = (e2e_scores >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, p_bin).ravel()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        sweep_list.append({"threshold": float(th), "f1": float(f1), "precision": float(prec), "recall": float(rec), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})

    out_e2e.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        "image_path": [str(p) for p in test_paths],
        "true_label": ["good" if y == 0 else "anomaly" for y in y_true],
        "raw_score": din_scores,
        "final_score": e2e_scores,
        "patchcore_score": pat_scores,
        "decision": ["anomaly" if sc >= m_e2e["th"] else "normal" for sc in e2e_scores]
    })
    df_out.to_csv(out_e2e / "e2e_results.csv", index=False)
    (out_e2e / "threshold_sweep.json").write_text(json.dumps(sweep_list, indent=2), encoding="utf-8")

    print(f"[{device}] Task N={n} Size={s} DONE: E2E AUROC={m_e2e["auc"]:.4f}, F1={m_e2e["f1"]:.4f} (th={m_e2e["th"]:.4f}), Din F1={m_din["f1"]:.4f}, Pat F1={m_pat["f1"]:.4f}")

    return {
        "n": n, "size": s,
        "din_auc": m_din["auc"], "pat_auc": m_pat["auc"], "e2e_auc": m_e2e["auc"],
        "din_ap": m_din["ap"], "pat_ap": m_pat["ap"], "e2e_ap": m_e2e["ap"],
        "din_f1": m_din["f1"], "pat_f1": m_pat["f1"], "e2e_f1": m_e2e["f1"],
        "e2e_opt_th": m_e2e["th"], "din_opt_th": m_din["th"], "pat_opt_th": m_pat["th"],
        "din_tp": m_din["tp"], "din_fn": m_din["fn"], "din_fp": m_din["fp"], "din_tn": m_din["tn"],
        "pat_tp": m_pat["tp"], "pat_fn": m_pat["fn"], "pat_fp": m_pat["fp"], "pat_tn": m_pat["tn"],
        "e2e_tp": m_e2e["tp"], "e2e_fn": m_e2e["fn"], "e2e_fp": m_e2e["fp"], "e2e_tn": m_e2e["tn"],
        "e2e_sec": e2e_sec, "fps": fps
    }

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    summary_results = list(ex.map(run_single_eval, enumerate(tasks)))

summary_results = sorted(summary_results, key=lambda x: (x["size"], x["n"]))
(outs_dir / "final_multisize_summary.json").write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
print(f"Saved final_multisize_summary.json with {len(summary_results)} tasks.")
