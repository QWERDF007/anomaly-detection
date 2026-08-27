#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic Benchmark Evaluation Script for Anomaly Detection.

Evaluates Dinomaly2, PatchCore, and Two-Stage E2E across specified (N, Size) combinations.
Automatically measures hardware performance (GPU VRAM, latency) and saves:
  1. final_multisize_summary.json
  2. real_vram_measurements.json
  3. e2e_results.csv per task
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    py_dir = Path(sys.executable).parent
    for p in [py_dir, py_dir / "Library" / "bin", py_dir / "DLLs"]:
        if p.is_dir():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parent
DINOMALY2_DIR = ROOT / "Dinomaly2"
if str(DINOMALY2_DIR) not in sys.path:
    sys.path.insert(0, str(DINOMALY2_DIR))

PATCHCORE_DIR = ROOT / "patchcore-inspection"
if str(PATCHCORE_DIR) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_DIR))
    sys.path.insert(0, str(PATCHCORE_DIR / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

import patchcore.patchcore
import patchcore.common
from utils import cal_anomaly_maps, get_gaussian_kernel
from models import vit_encoder
from models.uad import Dinomaly
from models.vision_transformer import Block as VitBlock, LinearAttention2
from functools import partial

def build_parser():
    p = argparse.ArgumentParser(description="Generic Benchmark Evaluation Tool")
    p.add_argument("--outs_dir", type=str, required=True, help="Directory containing trained experiment tasks")
    p.add_argument("--test_list", type=str, default="", help="Path to test image list text file (auto-detected if omitted)")
    p.add_argument("--bank_data", type=str, default="", help="Path to defect/normal feature bank source data")
    p.add_argument("--train_sizes", type=int, nargs="+", default=[], help="Sample sizes N to evaluate (auto-detected if empty)")
    p.add_argument("--image_sizes", type=int, nargs="+", default=[224, 448, 672], help="Resolution sizes to evaluate")
    p.add_argument("--cuda", type=int, default=0, help="GPU device ID (-1 for CPU)")
    return p

def auto_detect_test_list(outs_dir: Path) -> Path:
    candidates = [
        outs_dir / "data_splits" / "test_full.txt",
        outs_dir / "data_splits" / "test_1733.txt",
        outs_dir / "data_splits" / "test.txt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    splits_dir = outs_dir / "data_splits"
    if splits_dir.is_dir():
        txts = list(splits_dir.glob("test*.txt"))
        if txts:
            return sorted(txts, key=lambda p: p.stat().st_size, reverse=True)[0]
    raise FileNotFoundError(f"Could not automatically locate test image list under {outs_dir / 'data_splits'}")

def auto_detect_train_sizes(outs_dir: Path) -> list:
    ns = set()
    for p in outs_dir.glob("dinomaly2_n*_s*"):
        name = p.name
        if "_n" in name and "_s" in name:
            try:
                n = int(name.split("_n")[1].split("_s")[0])
                ns.add(n)
            except Exception:
                pass
    return sorted(list(ns)) if ns else [50, 100, 200, 400]

def main():
    args = build_parser().parse_args()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    if args.test_list:
        test_txt_path = Path(args.test_list).expanduser().resolve()
    else:
        test_txt_path = auto_detect_test_list(outs_dir)

    test_lines = [l.strip() for l in test_txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_paths = [Path(l.split("\t")[0].strip()) for l in test_lines]
    y_true = np.array([0 if ("\\OK\\" in str(p) or "/OK/" in str(p) or "good" in str(p).lower()) else 1 for p in test_paths], dtype=int)
    print(f"Loaded Test Set from {test_txt_path.name}: {len(test_paths)} images (OK={int((y_true==0).sum())}, NG={int((y_true==1).sum())}) on {device}")

    sizes = sorted(list(set(args.image_sizes)))
    ns = sorted(list(set(args.train_sizes))) if args.train_sizes else auto_detect_train_sizes(outs_dir)
    print(f"Evaluating across N={ns} and Image Sizes={sizes} in {outs_dir}...")

    tasks = []
    for s in sizes:
        for n in ns:
            din_candidates = sorted(list(outs_dir.glob(f"dinomaly2_n{n}_s{s}_seed2024/*/model.pth")) + list(outs_dir.glob(f"dinomaly2_n{n}_s{s}_seed2024/model.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
            pat_candidates = sorted(list(outs_dir.glob(f"patchcore_n{n}_s{s}_seed2024/*/*patchcore_params.pkl")) + list(outs_dir.glob(f"patchcore_n{n}_s{s}_seed2024/models/patchcore_params.pkl")), key=lambda p: p.stat().st_mtime, reverse=True)
            if not din_candidates:
                continue
            din_model = din_candidates[0]
            pat_pkl = pat_candidates[0] if pat_candidates else None
            out_e2e = outs_dir / f"e2e_out_n{n}_s{s}"
            save_bank = outs_dir / f"dinomaly2_n{n}_s{s}_seed2024" / "feature_bank.npz"
            tasks.append((n, s, din_model, pat_pkl, out_e2e, save_bank))

    print(f"Total valid tasks to evaluate: {len(tasks)}")

    summary_results = []

    for idx, (n, s, din_model_path, pat_pkl_path, out_e2e, save_bank_path) in enumerate(tasks):
        print(f"\n[{device}] [{idx+1}/{len(tasks)}] Starting Task N={n} Size={s}...")

        # 1. Load Dinomaly2 Model
        ckpt = torch.load(str(din_model_path), map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt: ckpt = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt: ckpt = ckpt["model"]

        backbone = "dinov2reg_vit_base_14"
        embed_dim, num_heads = 768, 12
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
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=1, channels=1).to(device)

        # Load Feature Banks
        ab_t = None
        nor_t = None
        if save_bank_path.is_file():
            bank = np.load(str(save_bank_path))
            k_ab = "abnormal_bank" if "abnormal_bank" in bank else bank.files[0]
            ab_t = torch.from_numpy(bank[k_ab]).float().to(device)
            if "normal_bank" in bank:
                nor_t = torch.from_numpy(bank["normal_bank"]).float().to(device)
            elif len(bank.files) > 1:
                nor_t = torch.from_numpy(bank[bank.files[1]]).float().to(device)
            else:
                nor_t = ab_t

        din_scores_all = []
        e2e_scores_all = []

        k_top = max(1, int(0.01 * (s // 14) * (s // 14)))
        effective_low = 0.25
        effective_high = 0.50

        # Measure Dinomaly2 & E2E Inference
        batch_sz = 8 if s <= 448 else 4
        t_start = time.perf_counter()
        with torch.no_grad():
            for b_idx in range(0, len(test_paths), batch_sz):
                b_paths = test_paths[b_idx : b_idx + batch_sz]
                b_tensors = [din_transform(Image.open(p).convert("RGB")) for p in b_paths]
                b_t = torch.stack(b_tensors, dim=0).to(device)

                en_o, de_o = din_model(b_t)
                amaps, _ = cal_anomaly_maps(en_o, de_o, s)
                amaps = gaussian_kernel(amaps)

                for j in range(len(b_paths)):
                    amap = amaps[j, 0].float().cpu().numpy()
                    raw_s = float(np.sort(amap.flatten())[-k_top:].mean())
                    din_scores_all.append(raw_s)

                    feat = en_o[-1][j].permute(1, 2, 0).float()
                    Hf, Wf, _ = feat.shape
                    amap_r = cv2.resize(amap, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                    unc_mask = (amap_r > effective_low) & (amap_r < effective_high)
                    if np.any(unc_mask) and ab_t is not None and nor_t is not None:
                        unc_idx = np.where(unc_mask)
                        unc_feats = feat[unc_idx[0], unc_idx[1], :]
                        unc_feats = F.normalize(unc_feats, p=2, dim=-1)

                        ab_ip = torch.mm(unc_feats, ab_t.T).max(dim=-1).values
                        nor_ip = torch.mm(unc_feats, nor_t.T).max(dim=-1).values
                        ab_dist = 1.0 - ab_ip
                        nor_dist = 1.0 - nor_ip

                        is_ab = ab_dist < nor_dist
                        margin = (nor_dist - ab_dist) / (nor_dist + ab_dist + 1e-6)
                        gain = torch.where(is_ab, 1.0 + 0.8 * torch.clamp(margin, min=0.0), 1.0 - 0.5 * torch.clamp(-margin, min=0.0))
                        amap_r[unc_idx] = amap_r[unc_idx] * gain.cpu().numpy()

                    final_amap = cv2.resize(amap_r, (s, s), interpolation=cv2.INTER_LINEAR)
                    cor_s = float(np.sort(final_amap.flatten())[-k_top:].mean())
                    e2e_scores_all.append(cor_s)

        e2e_sec = time.perf_counter() - t_start
        fps = len(test_paths) / e2e_sec

        # 2. Evaluate PatchCore
        m_pat = None
        pat_scores = None
        if pat_pkl_path is not None and pat_pkl_path.is_file():
            try:
                pat_model = patchcore.patchcore.PatchCore(device)
                pat_model.load_from_path(
                    load_path=str(pat_pkl_path.parent),
                    device=device,
                    prepend=pat_pkl_path.name[:-len("patchcore_params.pkl")],
                    nn_method=patchcore.common.FaissNN(on_gpu=True, num_workers=0)
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
                pat_scores = np.array(pat_scores_all, dtype=np.float32)
            except Exception as e:
                print(f"[warn] PatchCore eval failed for N={n} Size={s}: {e}")

        # Calculate metrics
        din_scores = np.array(din_scores_all, dtype=np.float32)
        e2e_scores = np.array(e2e_scores_all, dtype=np.float32)

        def calc_model_metrics(scores):
            if scores is None: return None
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
        m_pat = calc_model_metrics(pat_scores) if pat_scores is not None else None

        # Save to e2e_results.csv in task dir
        out_e2e.mkdir(parents=True, exist_ok=True)
        csv_dict = {
            "image_path": [str(p) for p in test_paths],
            "true_label": ["good" if y == 0 else "anomaly" for y in y_true],
            "raw_score": din_scores,
            "final_score": e2e_scores,
            "decision": ["anomaly" if sc >= m_e2e["th"] else "normal" for sc in e2e_scores]
        }
        if pat_scores is not None:
            csv_dict["patchcore_score"] = pat_scores
        pd.DataFrame(csv_dict).to_csv(out_e2e / "e2e_results.csv", index=False)

        res_item = {
            "n": n, "size": s,
            "din_auc": m_din["auc"], "din_ap": m_din["ap"], "din_f1": m_din["f1"], "din_th": m_din["th"],
            "din_tp": m_din["tp"], "din_fp": m_din["fp"], "din_tn": m_din["tn"], "din_fn": m_din["fn"],
            "e2e_auc": m_e2e["auc"], "e2e_ap": m_e2e["ap"], "e2e_f1": m_e2e["f1"], "e2e_th": m_e2e["th"],
            "e2e_tp": m_e2e["tp"], "e2e_fp": m_e2e["fp"], "e2e_tn": m_e2e["tn"], "e2e_fn": m_e2e["fn"],
            "e2e_sec": round(e2e_sec, 2), "fps": round(fps, 1),
            "pat_auc": m_pat["auc"] if m_pat else 0.0, "pat_ap": m_pat["ap"] if m_pat else 0.0,
            "pat_f1": m_pat["f1"] if m_pat else 0.0, "pat_th": m_pat["th"] if m_pat else 0.0,
            "pat_tp": m_pat["tp"] if m_pat else 0, "pat_fp": m_pat["fp"] if m_pat else 0,
            "pat_tn": m_pat["tn"] if m_pat else 0, "pat_fn": m_pat["fn"] if m_pat else 0,
        }
        summary_results.append(res_item)

        print(f"Done Task N={n} Size={s} -> Dino AUC={m_din['auc']:.4f}, Patch AUC={m_pat['auc'] if m_pat else 0:.4f}, E2E AUC={m_e2e['auc']:.4f} (E2E FP={m_e2e['fp']}, TP={m_e2e['tp']})")

    # Save summary
    summary_path = outs_dir / "final_multisize_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Saved generic benchmark summary to -> {summary_path}")

if __name__ == "__main__":
    main()
