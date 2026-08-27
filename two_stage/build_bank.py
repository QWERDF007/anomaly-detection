#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build feature bank (Patch-level ROI from LabelMe polygons) based on authentic Dinomaly2_two_lib.

Usage:
  python two_stage/build_bank.py \
    --model /path/to/model.pth \
    --data_dir /path/to/bank_data \
    --save_bank /path/to/feature_bank.npz \
    --image_size 448 --cuda 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Allow import of Dinomaly2 modules
ROOT = Path(__file__).resolve().parents[1]
DINOMALY2 = ROOT / "Dinomaly2"
if str(DINOMALY2) not in sys.path:
    sys.path.insert(0, str(DINOMALY2))

from models import vit_encoder
from models.uad import Dinomaly
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from functools import partial
import torch.nn as nn
from utils import cal_anomaly_maps, get_gaussian_kernel

IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve_model(path_str: str) -> Path:
    candidates = glob.glob(path_str, recursive=True)
    if candidates:
        candidates = sorted([Path(p) for p in candidates if Path(p).is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    p = Path(path_str).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        found = sorted(list(p.rglob("model.pth")), key=lambda x: x.stat().st_mtime, reverse=True)
        if found:
            return found[0]
    raise FileNotFoundError(f"Model not found: {path_str}")


def parse_labelme_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = {"ad": [], "good": []}
    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower()
        key = "ad" if label in ["ad", "ng", "anomaly", "defect", "abnormal"] else ("good" if label in ["good", "ok", "normal"] else None)
        if not key:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        if len(points) == 2:  # rectangle
            (x1, y1), (x2, y2) = points
            poly = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
        else:
            poly = [(int(p[0]), int(p[1])) for p in points]
        polygons[key].append(poly)

    return polygons


def extract_roi_patch_features(feat_map: np.ndarray, polygon_pts: List[Tuple[int, int]], orig_W: int, orig_H: int):
    Hf, Wf, C = feat_map.shape
    scale_x = Wf / orig_W
    scale_y = Hf / orig_H

    feat_poly = np.array([(round(x * scale_x), round(y * scale_y)) for (x, y) in polygon_pts], dtype=np.int32)
    mask = np.zeros((Hf, Wf), dtype=np.uint8)
    cv2.fillPoly(mask, [feat_poly], 1)
    ys, xs = np.where(mask == 1)

    if len(ys) == 0:
        cx = int(np.mean([p[0] for p in feat_poly]))
        cy = int(np.mean([p[1] for p in feat_poly]))
        cx = max(0, min(Wf - 1, cx))
        cy = max(0, min(Hf - 1, cy))
        return feat_map[cy : cy + 1, cx : cx + 1, :].reshape(1, -1)

    return feat_map[ys, xs, :]


def build_parser():
    p = argparse.ArgumentParser(description="Build Dinomaly2 feature bank (Patch-level ROI from LabelMe polygons)")
    p.add_argument("--model", type=str, required=True, help="Dinomaly2 model.pth path")
    p.add_argument("--data_dir", type=str, required=True, help="Bank data directory")
    p.add_argument("--save_bank", type=str, required=True, help="Output npz path")
    p.add_argument("--save_dir", type=str, default=None, help="Optional save dir")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--backbone", type=str, default="dinov2reg_vit_base_14")
    p.add_argument("--cuda", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    model_path = resolve_model(args.model)
    data_dir = Path(args.data_dir).expanduser().resolve()
    save_bank = Path(args.save_bank).expanduser().resolve()
    save_bank.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")
    print(f"[build_bank] device={device}, model={model_path}, image_size={args.image_size}")

    # Build model
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    backbone = args.backbone
    if "bottleneck.0.0.weight" in ckpt:
        in_dim = ckpt["bottleneck.0.0.weight"].shape[1]
        if in_dim == 384 and "small" not in backbone:
            backbone = "dinov2reg_vit_small_14"
        elif in_dim == 768 and "base" not in backbone:
            backbone = "dinov2reg_vit_base_14"

    encoder = vit_encoder.load(backbone)
    if "small" in backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    else:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]

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
    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers, remove_class_token=False, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder, context_aware_recenter=1)
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, channels=1).to(device)

    # Collect Images & JSONs
    ab_feats_list = []
    nor_feats_list = []

    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [data_dir]

    with torch.no_grad():
        for sdir in subdirs:
            is_ng_dir = sdir.name.lower() in ["ng", "anomaly", "defect", "abnormal"]
            img_files = sorted([p for p in sdir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

            for img_p in img_files:
                json_p = img_p.with_suffix(".json")
                try:
                    img = Image.open(img_p).convert("RGB")
                except Exception as e:
                    print(f"[warn] Cannot read image {img_p}: {e}")
                    continue

                orig_W, orig_H = img.size
                img_t = transform(img).unsqueeze(0).to(device)
                en, de = model(img_t)
                feat = en[-1][0].permute(1, 2, 0).cpu().numpy()  # (Hf, Wf, C)
                Hf, Wf, C = feat.shape

                if json_p.is_file():
                    polys = parse_labelme_json(json_p)
                    for poly in polys["ad"]:
                        feats = extract_roi_patch_features(feat, poly, orig_W, orig_H)
                        if len(feats) > 0:
                            ab_feats_list.append(feats)
                    for poly in polys["good"]:
                        feats = extract_roi_patch_features(feat, poly, orig_W, orig_H)
                        if len(feats) > 0:
                            nor_feats_list.append(feats)
                else:
                    if is_ng_dir:
                        amap, _ = cal_anomaly_maps(en, de, (args.image_size, args.image_size))
                        amap = gaussian_kernel(amap)[0, 0].cpu().numpy()
                        amap_small = cv2.resize(amap, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                        thr = np.percentile(amap_small, 80)
                        idx = np.where(amap_small >= thr)
                        if len(idx[0]) > 0:
                            ab_feats_list.append(feat[idx])
                    else:
                        nor_feats_list.append(feat.reshape(-1, C))

    ab_feats = np.ascontiguousarray(np.vstack(ab_feats_list), dtype=np.float32) if ab_feats_list else np.zeros((0, embed_dim), dtype=np.float32)
    nor_feats = np.ascontiguousarray(np.vstack(nor_feats_list), dtype=np.float32) if nor_feats_list else np.zeros((0, embed_dim), dtype=np.float32)

    faiss.normalize_L2(ab_feats)
    faiss.normalize_L2(nor_feats)

    print(f"[build_bank] Final Bank built: {ab_feats.shape[0]} NG patch vectors, {nor_feats.shape[0]} OK patch vectors.")

    np.savez_compressed(
        str(save_bank),
        ab_features=ab_feats,
        nor_features=nor_feats,
        good_features=nor_feats,
        anomaly_features=ab_feats,
        image_size=np.array(args.image_size),
        backbone=np.array(backbone),
    )
    print(f"[build_bank] saved -> {save_bank} ({save_bank.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
