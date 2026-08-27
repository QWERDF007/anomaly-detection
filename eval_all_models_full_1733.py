import os, sys, glob, time, json, argparse
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

ROOT = Path(__file__).resolve().parent
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

def build_parser():
    p = argparse.ArgumentParser(description="Evaluate on Full 1733-Image Test Set")
    p.add_argument("--outs_dir", type=str, default="F:/tmp/0826")
    p.add_argument("--test_list", type=str, default="F:/tmp/outs/data_splits/test_50_seed2024.txt")
    p.add_argument("--bank_data", type=str, default=r"F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据2")
    p.add_argument("--cuda", type=int, default=0)
    return p

def main():
    args = build_parser().parse_args()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    test_txt_path = Path(args.test_list).expanduser().resolve()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    test_lines = [l.strip() for l in test_txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_paths = [Path(l.split()[0]) for l in test_lines]
    y_true = np.array([int(l.split()[1]) for l in test_lines], dtype=int)
    print(f"Loaded Unified Full Test Set: {len(test_paths)} images (OK={int((y_true==0).sum())}, NG={int((y_true==1).sum())}) on {device}")

    tasks = []
    sizes = [224, 448, 672]
    ns = [50, 100, 200, 400]

    for s in sizes:
        for n in ns:
            din_candidates = sorted(list(outs_dir.glob(f"dinomaly2_n{n}_s{s}_seed2024/*/model.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
            pat_candidates = sorted(list(outs_dir.glob(f"patchcore_n{n}_s{s}_seed2024/*/*patchcore_params.pkl")), key=lambda p: p.stat().st_mtime, reverse=True)
            if not din_candidates:
                print(f"[warn] Missing Dinomaly2 model for N={n} Size={s}")
                continue
            din_model = din_candidates[0]
            pat_pkl = pat_candidates[0] if pat_candidates else None
            out_e2e = outs_dir / f"e2e_out_n{n}_s{s}"
            save_bank = outs_dir / f"dinomaly2_n{n}_s{s}_seed2024" / "feature_bank.npz"
            tasks.append((n, s, din_model, pat_pkl, out_e2e, save_bank))

    print(f"Total valid tasks to evaluate on Full 1733: {len(tasks)}")

    summary_results = []
    for idx, (n, s, din_model_path, pat_pkl_path, out_e2e, save_bank_path) in enumerate(tasks):
        print(f"\n[{device}] [{idx+1}/{len(tasks)}] Starting Task N={n} Size={s} on Full 1733 dataset...")

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, channels=1).to(device)

        # Load authentic feature bank
        ab_idx = faiss.IndexFlatIP(embed_dim)
        nor_idx = faiss.IndexFlatIP(embed_dim)
        if save_bank_path.is_file():
            bank_data = np.load(str(save_bank_path), allow_pickle=True)
            ab_feats = bank_data.get("ab_features", bank_data.get("anomaly_features"))
            nor_feats = bank_data.get("nor_features", bank_data.get("good_features"))
            if ab_feats is not None and len(ab_feats) > 0:
                ab_norm = np.ascontiguousarray(ab_feats, dtype=np.float32)
                faiss.normalize_L2(ab_norm)
                ab_idx.add(ab_norm)
            if nor_feats is not None and len(nor_feats) > 0:
                nor_norm = np.ascontiguousarray(nor_feats, dtype=np.float32)
                faiss.normalize_L2(nor_norm)
                nor_idx.add(nor_norm)

        # Evaluate Dinomaly2 & Two-Stage E2E
        batch_size = 2 if s >= 672 else 4
        din_scores_all = []
        e2e_scores_all = []
        k_top = max(1, int(s * s * 0.01))

        if s == 224: effective_low, effective_high = 0.015, 0.038
        elif s == 448: effective_low, effective_high = 0.020, 0.052
        else: effective_low, effective_high = 0.025, 0.072

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
                    unc_mask = (amap_r > effective_low) & (amap_r < effective_high)
                    if np.any(unc_mask) and ab_idx.ntotal > 0 and nor_idx.ntotal > 0:
                        unc_idx = np.where(unc_mask)
                        unc_feats = np.ascontiguousarray(feat[unc_idx], dtype=np.float32)
                        faiss.normalize_L2(unc_feats)
                        ab_ip, _ = ab_idx.search(unc_feats, 1)
                        nor_ip, _ = nor_idx.search(unc_feats, 1)
                        ab_dist = 1.0 - ab_ip[:, 0]
                        nor_dist = 1.0 - nor_ip[:, 0]
                        is_ab = ab_dist < nor_dist
                        margin = (nor_dist - ab_dist) / (nor_dist + ab_dist + 1e-6)
                        gain = np.where(is_ab, 1.0 + 0.8 * np.maximum(0.0, margin), 1.0 - 0.5 * np.maximum(0.0, -margin))
                        amap_r[unc_idx] = amap_r[unc_idx] * gain

                    final_amap = cv2.resize(amap_r, (s, s), interpolation=cv2.INTER_LINEAR)
                    cor_s = float(np.sort(final_amap.flatten())[-k_top:].mean())
                    e2e_scores_all.append(cor_s)

        e2e_sec = time.perf_counter() - t_start
        fps = len(test_paths) / e2e_sec

        # 2. Evaluate PatchCore on Full 1733
        m_pat = None
        pat_scores = None
        if pat_pkl_path is not None and pat_pkl_path.is_file():
            try:
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

        # Save to e2e_results.csv in output dir
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

        pat_str = f"Pat AUROC={m_pat['auc']:.4f}, F1={m_pat['f1']:.4f}" if m_pat else "Pat=OOM"
        print(f"[{device}] Task N={n} Size={s} DONE: E2E AUROC={m_e2e['auc']:.4f}, F1={m_e2e['f1']:.4f}, Din F1={m_din['f1']:.4f}, {pat_str}")

        row_res = {
            "n": n, "size": s,
            "din_auc": m_din["auc"], "pat_auc": m_pat["auc"] if m_pat else None, "e2e_auc": m_e2e["auc"],
            "din_ap": m_din["ap"], "pat_ap": m_pat["ap"] if m_pat else None, "e2e_ap": m_e2e["ap"],
            "din_f1": m_din["f1"], "pat_f1": m_pat["f1"] if m_pat else None, "e2e_f1": m_e2e["f1"],
            "din_tp": m_din["tp"], "din_fn": m_din["fn"], "din_fp": m_din["fp"], "din_tn": m_din["tn"],
            "pat_tp": m_pat["tp"] if m_pat else None, "pat_fn": m_pat["fn"] if m_pat else None, "pat_fp": m_pat["fp"] if m_pat else None, "pat_tn": m_pat["tn"] if m_pat else None,
            "e2e_tp": m_e2e["tp"], "e2e_fn": m_e2e["fn"], "e2e_fp": m_e2e["fp"], "e2e_tn": m_e2e["tn"],
            "e2e_sec": e2e_sec, "fps": fps
        }
        summary_results.append(row_res)

    summary_results = sorted(summary_results, key=lambda x: (x["size"], x["n"]))
    (outs_dir / "final_multisize_summary.json").write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    print("\nSaved final_multisize_summary.json on Full 1733 images.")

    # Regenerate reports and plots
    print("\n=== Generating Final Multisize Reports ===")
    import subprocess
    cmd_rep = [sys.executable, str(ROOT / "generate_final_report_multisize.py"), "--outs_dir", str(outs_dir)]
    subprocess.run(cmd_rep, cwd=str(ROOT))

    print("\n=== Generating Real-Data Visualizations ===")
    cmd_plot = [sys.executable, str(ROOT / "plot_evaluation_charts.py"), "--outs_dir", str(outs_dir), "--chart_dir", str(outs_dir / "charts"), "--full_benchmark"]
    subprocess.run(cmd_plot, cwd=str(ROOT))
    print("\n=== All Evaluation and Plotting Successfully Finished! ===")

if __name__ == "__main__":
    main()
