#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

ROOT = Path("/data/wt/anomaly-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Dinomaly2") not in sys.path:
    sys.path.insert(0, str(ROOT / "Dinomaly2"))

import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from functools import partial

from models import vit_encoder
from models.uad import Dinomaly
from models.vision_transformer import Block as VitBlock, LinearAttention2
from utils import cal_anomaly_maps, get_gaussian_kernel

IMAGE_SIZE = 448
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0")

tasks = [
    {
        "name": "透气膜",
        "model_dir": Path("/data/wt/report_base/透气膜/dinomaly2_n150_s448_seed2024"),
        "test_list": Path("/data/wt/report_base/透气膜/data_splits/test_full.txt"),
        "out_dir": Path("/data/wt/tmp/heatmaps/透气膜"),
    },
    {
        "name": "铜色异常检测4相机",
        "model_dir": Path("/data/wt/report_base/铜色异常检测4相机/dinomaly2_n200_s448_seed2024"),
        "test_list": Path("/data/wt/report_base/铜色异常检测4相机/data_splits/test_full.txt"),
        "out_dir": Path("/data/wt/tmp/heatmaps/铜色异常检测4相机"),
    },
    {
        "name": "铜色异常检测6相机",
        "model_dir": Path("/data/wt/report_base/铜色异常检测6相机/dinomaly2_n200_s448_seed2024"),
        "test_list": Path("/data/wt/report_base/铜色异常检测6相机/data_splits/test_full.txt"),
        "out_dir": Path("/data/wt/tmp/heatmaps/铜色异常检测6相机"),
    },
]

def load_base_model(model_path, device):
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    embed_dim = 768
    num_heads = 12
    backbone = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(backbone)
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=0.4)),
        nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=0.4), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=0.4)),
    ])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=partial(LinearAttention2, eps=1e-8))
        for _ in range(8)
    ])
    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers, remove_class_token=False, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder, context_aware_recenter=1)
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()
    return model

def create_overlay(orig_bgr, norm_map):
    # norm_map in [0, 1]
    heat_u8 = (np.clip(norm_map, 0.0, 1.0) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.5, heat_color, 0.5, 0)
    return heat_color, overlay

def add_side_by_side_header(panel_bgr, score, global_min, global_max, loc_min, loc_max):
    h, w, c = panel_bgr.shape
    banner_h = 70
    banner = np.full((banner_h, w, 3), (25, 25, 25), dtype=np.uint8)

    # Line 1: Anomaly Score & Scale bounds (compact font 0.55)
    score_text = f"Anomaly Score: {score:.4f}  |  Global Scale: [{global_min:.4f}, {global_max:.4f}]  |  Local Range: [{loc_min:.4f}, {loc_max:.4f}]"
    cv2.putText(banner, score_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 235, 255), 1, cv2.LINE_AA)

    # Line 2: 3 Column Headings centered over each 448px section
    cv2.putText(banner, "[ Original Image ]", (135, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(banner, "[ Global-Norm Heatmap ]", (550, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, "[ Local-Norm Heatmap ]", (1000, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (130, 255, 170), 1, cv2.LINE_AA)

    # Fine separator line
    cv2.line(banner, (0, 68), (w, 68), (60, 60, 60), 1)

    return np.vstack([banner, panel_bgr])

def process_dataset(task_cfg):
    name = task_cfg["name"]
    model_dir = task_cfg["model_dir"]
    test_list_p = task_cfg["test_list"]
    out_dir = task_cfg["out_dir"]

    print("\n" + "=" * 80)
    print(f"=== Processing Heatmaps for: {name} ===")
    print(f"Model Dir: {model_dir}")
    print(f"Test List: {test_list_p}")
    print(f"Output:    {out_dir}")
    print("=" * 80, flush=True)

    # 1. Output directories split cleanly by OK and NG
    ok_global_dir = out_dir / "OK" / "global_norm"
    ok_local_dir = out_dir / "OK" / "local_norm"
    ok_side_dir = out_dir / "OK" / "side_by_side"

    ng_global_dir = out_dir / "NG" / "global_norm"
    ng_local_dir = out_dir / "NG" / "local_norm"
    ng_side_dir = out_dir / "NG" / "side_by_side"

    for d in [ok_global_dir, ok_local_dir, ok_side_dir, ng_global_dir, ng_local_dir, ng_side_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Find model and bank
    model_cands = sorted(list(model_dir.glob("*/model.pth")) + list(model_dir.glob("model.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_cands:
        print(f"[Error] No model.pth found in {model_dir}")
        return
    model_path = model_cands[0]

    bank_path = model_dir / "feature_bank.npz"
    ab_t, nor_t = None, None
    if bank_path.is_file():
        bank = np.load(str(bank_path))
        if "ab_features" in bank and "nor_features" in bank:
            ab_t = torch.from_numpy(bank["ab_features"]).float().to(DEVICE)
            nor_t = torch.from_numpy(bank["nor_features"]).float().to(DEVICE)
            print(f"Loaded Feature Bank: ab_features={ab_t.shape}, nor_features={nor_t.shape}")

    # 3. Load test list
    test_lines = [l.strip() for l in test_list_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_items = []
    for line in test_lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            test_items.append((Path(parts[0]), int(parts[1])))
        else:
            test_items.append((Path(parts[0]), 0))
    print(f"Loaded {len(test_items)} test images.")

    # 4. Load Model
    print(f"Loading Base model from {model_path} onto {DEVICE}...")
    model = load_base_model(model_path, DEVICE)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, channels=1).to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 5. PASS 1: Extract all raw anomaly maps across the dataset
    print(f"Pass 1: Extracting raw anomaly maps for all {len(test_items)} images...")
    raw_maps = []
    image_scores = []
    orig_images = []
    
    k_top = max(1, int(0.01 * (IMAGE_SIZE // 14) * (IMAGE_SIZE // 14)))
    effective_low = 0.25
    effective_high = 0.50
    batch_sz = 8

    t0 = time.perf_counter()
    with torch.no_grad():
        for b_idx in range(0, len(test_items), batch_sz):
            batch = test_items[b_idx : b_idx + batch_sz]
            b_paths = [p for p, _ in batch]
            
            b_imgs = [Image.open(p).convert("RGB") for p in b_paths]
            b_tensors = [transform(img) for img in b_imgs]
            b_t = torch.stack(b_tensors, dim=0).to(DEVICE)

            en_o, de_o = model(b_t)
            amaps, _ = cal_anomaly_maps(en_o, de_o, IMAGE_SIZE)
            amaps = gaussian_kernel(amaps)

            for j in range(len(batch)):
                amap = amaps[j, 0].float().cpu().numpy()
                feat = en_o[-1][j].permute(1, 2, 0).float()
                Hf, Wf, _ = feat.shape
                amap_r = cv2.resize(amap, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                
                # Two-stage correction
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

                final_amap = cv2.resize(amap_r, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                img_score = float(np.sort(final_amap.flatten())[-k_top:].mean())

                raw_maps.append(final_amap)
                image_scores.append(img_score)
                orig_images.append(b_imgs[j])

    elapsed_p1 = time.perf_counter() - t0
    print(f"Pass 1 finished in {elapsed_p1:.2f}s ({len(raw_maps)/(elapsed_p1+1e-6):.1f} FPS).")

    # 6. Global Normalization Bounds
    all_pixels = np.concatenate([m.flatten() for m in raw_maps])
    global_min = float(all_pixels.min())
    global_max = float(all_pixels.max())
    p01 = float(np.percentile(all_pixels, 0.1))
    p99_9 = float(np.percentile(all_pixels, 99.9))
    p99 = float(np.percentile(all_pixels, 99.0))
    p50 = float(np.percentile(all_pixels, 50.0))

    print("\n" + "-" * 60)
    print(f"[{name}] Global Anomaly Pixel Score Statistics:")
    print(f"  Absolute Global Min: {global_min:.6f}")
    print(f"  Absolute Global Max: {global_max:.6f}")
    print(f"  P0.1:  {p01:.6f} | P50 (Median): {p50:.6f} | P99: {p99:.6f} | P99.9: {p99_9:.6f}")
    print("-" * 60 + "\n")

    # Save stats JSON
    stats = {
        "dataset": name,
        "model": str(model_path),
        "total_images": len(test_items),
        "global_min": global_min,
        "global_max": global_max,
        "p01": p01,
        "p50": p50,
        "p99": p99,
        "p99_9": p99_9,
        "global_norm_formula": f"norm = np.clip((raw_score - {global_min:.6f}) / ({global_max - global_min:.6f}), 0.0, 1.0)"
    }
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7. PASS 2: Render Global-Norm, Local-Norm and Side-by-Side Heatmaps
    print(f"Pass 2: Rendering and saving normalized heatmaps to {out_dir}...")
    for idx, ((img_path, label), raw_map, score, orig_pil) in enumerate(zip(test_items, raw_maps, image_scores, orig_images)):
        # Resize orig to 448x448
        orig_resized = orig_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        orig_bgr = cv2.cvtColor(np.array(orig_resized), cv2.COLOR_RGB2BGR)

        # 1) Global Normalized Map
        glob_norm = (raw_map - global_min) / (global_max - global_min + 1e-8)
        glob_norm = np.clip(glob_norm, 0.0, 1.0)
        glob_heat, glob_overlay = create_overlay(orig_bgr, glob_norm)

        # 2) Local Normalized Map (per image min/max)
        loc_min, loc_max = raw_map.min(), raw_map.max()
        loc_norm = (raw_map - loc_min) / (loc_max - loc_min + 1e-8)
        loc_norm = np.clip(loc_norm, 0.0, 1.0)
        loc_heat, loc_overlay = create_overlay(orig_bgr, loc_norm)

        # Determine OK / NG folder and prefix
        if label == 1:
            lbl_tag = "NG"
            target_global_dir = ng_global_dir
            target_local_dir = ng_local_dir
            target_side_dir = ng_side_dir
        else:
            lbl_tag = "OK"
            target_global_dir = ok_global_dir
            target_local_dir = ok_local_dir
            target_side_dir = ok_side_dir

        # Filename: purely original image name
        fname = f"{img_path.stem}.jpg"

        cv2.imwrite(str(target_global_dir / fname), glob_overlay)
        cv2.imwrite(str(target_local_dir / fname), loc_overlay)

        # 3) Side-by-Side Comparison Panel with concise, smaller-font header
        panel = np.hstack([orig_bgr, glob_overlay, loc_overlay])
        titled_panel = add_side_by_side_header(panel, score, global_min, global_max, loc_min, loc_max)
        cv2.imwrite(str(target_side_dir / fname), titled_panel)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_items):
            print(f"  [{idx+1}/{len(test_items)}] Exported ({lbl_tag}): {fname}")

    print(f"\n[SUCCESS] Finished {name}! Heatmaps saved to:\n  - OK: {out_dir / 'OK'}\n  - NG: {out_dir / 'NG'}\n", flush=True)

def main():
    print("=" * 80)
    print("=== GLOBAL & LOCAL HEATMAP GENERATION (ViT-Base 448 N=200/150) ===")
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print("=" * 80)

    for task in tasks:
        process_dataset(task)

    print("\n" + "=" * 80)
    print("ALL 3 INDUSTRIAL DATASETS HEATMAPS GENERATED SUCCESSFULLY!")
    print(f"Output Root: /data/wt/tmp/heatmaps/")
    print("=" * 80)

if __name__ == "__main__":
    main()
