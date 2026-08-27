#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate PatchCore, Dinomaly2, and Two-Stage E2E on the Full 1733-Image Test Set (1680 OK + 53 NG)."""

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
from utils import cal_anomaly_maps, get_gaussian_kernel
from models import vit_encoder
from models.uad import Dinomaly
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from functools import partial

outs_dir = Path("/data/wt/report/0826")
test_txt_path = Path("/data/wt/outs/data_splits/test_50_seed2024.txt")
bank_data_path = Path("/data/wt/ramdisk/铜色异常检测6相机_建库数据2")

test_lines = [l.strip() for l in test_txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
test_paths = [Path(l.split()[0]) for l in test_lines]
y_true = np.array([int(l.split()[1]) for l in test_lines], dtype=int)
print(f"Loaded Unified Full Test Set: {len(test_paths)} images (OK={int((y_true==0).sum())}, NG={int((y_true==1).sum())})")

tasks = []
sizes = [224, 448, 672]
ns = [50, 100, 200, 400]

for s in sizes:
    for n in ns:
        din_candidates = list(outs_dir.glob(f"dinomaly2_n{n}_s{s}_seed2024/*/model.pth"))
        pat_candidates = list(outs_dir.glob(f"patchcore_n{n}_s{s}_seed2024/*/*patchcore_params.pkl"))
        if not din_candidates or not pat_candidates:
            print(f"[warn] Missing candidates for N={n} Size={s}")
            continue
        din_model = din_candidates[0]
        pat_pkl = pat_candidates[0]
        out_e2e = outs_dir / f"e2e_out_n{n}_s{s}"
        save_bank = outs_dir / f"dinomaly2_n{n}_s{s}_seed2024" / "feature_bank.npz"
        tasks.append((n, s, din_model, pat_pkl, out_e2e, save_bank))

print(f"Total valid tasks to evaluate on Full 1733: {len(tasks)}")

def run_single_eval(args_tuple):
    idx, (n, s, din_model_path, pat_pkl_path, out_e2e, save_bank_path) = args_tuple
    gpu_id = idx % 8
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
    print(f"[{device}] Starting Task N={n} Size={s} on Full 1733 dataset...")

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

    # Load authentic feature bank
    bank_data = np.load(str(save_bank_path), allow_pickle=True)
    ab_feats = bank_data.get("ab_features", bank_data.get("anomaly_features"))
    nor_feats = bank_data.get("nor_features", bank_data.get("good_features"))
    
    ab_idx = faiss.IndexFlatIP(embed_dim)
    nor_idx = faiss.IndexFlatIP(embed_dim)
    if ab_feats is not None and len(ab_feats) > 0:
        ab_idx.add(np.ascontiguousarray(ab_feats, dtype=np.float32))
    if nor_feats is not None and len(nor_feats) > 0:
        nor_idx.add(np.ascontiguousarray(nor_feats, dtype=np.float32))

    # Evaluate Dinomaly2 & Two-Stage E2E
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

    # 2. Evaluate PatchCore on Full 1733
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

    # Save to e2e_results.csv in output dir
    out_e2e.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        "image_path": [str(p) for p in test_paths],
        "true_label": ["good" if y == 0 else "anomaly" for y in y_true],
        "raw_score": din_scores,
        "final_score": e2e_scores,
        "patchcore_score": pat_scores,
        "decision": ["anomaly" if s >= m_e2e["th"] else "normal" for s in e2e_scores]
    })
    df_out.to_csv(out_e2e / "e2e_results.csv", index=False)

    print(f"[{device}] Task N={n} Size={s} DONE: E2E AUROC={m_e2e["auc"]:.4f}, F1={m_e2e["f1"]:.4f}, Din F1={m_din["f1"]:.4f}, Pat F1={m_pat["f1"]:.4f}")

    return {
        "n": n, "size": s,
        "din_auc": m_din["auc"], "pat_auc": m_pat["auc"], "e2e_auc": m_e2e["auc"],
        "din_ap": m_din["ap"], "pat_ap": m_pat["ap"], "e2e_ap": m_e2e["ap"],
        "din_f1": m_din["f1"], "pat_f1": m_pat["f1"], "e2e_f1": m_e2e["f1"],
        "din_tp": m_din["tp"], "din_fn": m_din["fn"], "din_fp": m_din["fp"], "din_tn": m_din["tn"],
        "pat_tp": m_pat["tp"], "pat_fn": m_pat["fn"], "pat_fp": m_pat["fp"], "pat_tn": m_pat["tn"],
        "e2e_tp": m_e2e["tp"], "e2e_fn": m_e2e["fn"], "e2e_fp": m_e2e["fp"], "e2e_tn": m_e2e["tn"],
        "e2e_sec": e2e_sec, "fps": fps
    }

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    summary_results = list(ex.map(run_single_eval, enumerate(tasks)))

# Sort by size and n
summary_results = sorted(summary_results, key=lambda x: (x["size"], x["n"]))
Path("/data/wt/report/0826/final_multisize_summary.json").write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
print("Saved final_multisize_summary.json on Full 1733 images.")
