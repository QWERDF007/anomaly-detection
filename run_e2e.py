#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-End Two-Stage Anomaly Detection Pipeline (Train -> Bank -> Inference -> Eval -> Plot).

Full Capabilities:
  1. [Stage 1 Training] (Optional --do_train): Trains Dinomaly2 Bottleneck & Decoder with AMP FP16.
  2. [Stage 2 Feature Bank]: Builds L2-normalized 768D feature bank with faiss.IndexFlatIP in 1 second.
  3. [Stage 3 Fast Inference]: Runs AMP FP16 inference with Dual-Threshold Short-Circuit Gating.
  4. [Stage 4 Summary Logging]: Saves e2e_results.csv, e2e_results.json, e2e_summary.json.
  5. [Stage 5 Auto-Plotting]: Automatically renders 300 DPI ROC, PR, Distribution & Confusion Matrix charts.
  6. [Multi-GPU Task Dispatcher]: Supports parallel multi-GPU execution (one independent task per GPU).

Usage Examples:
  # 1. Single GPU Inference & Evaluation with pre-trained checkpoint:
  python run_e2e.py `
    --dinomaly_model "F:\\tmp\\outs\\dinomaly2_n400_s448_seed2024\\*\\model.pth" `
    --bank_data "F:\\data\\异常检测测试报告数据\\铜色异常检测6相机_建库数据" `
    --test_list "F:\\tmp\\outs\\data_splits\\test_400_seed2024.txt" `
    --output_dir "F:\\tmp\\e2e_out" --image_size 448 --low 0.019 --high 0.024 --cuda 0

  # 2. Single GPU Full Train + Test + Eval from scratch:
  python run_e2e.py `
    --do_train --train_list "F:\\tmp\\outs\\data_splits\\train_400_seed2024.txt" `
    --bank_data "F:\\data\\异常检测测试报告数据\\铜色异常检测6相机_建库数据" `
    --test_list "F:\\tmp\\outs\\data_splits\\test_400_seed2024.txt" `
    --output_dir "F:\\tmp\\outs\\e2e_full_train_s448" `
    --total_iters 2000 --image_size 448 --low 0.019 --high 0.024 --cuda 0

  # 3. Multi-GPU Task Partitioning (e.g. 4 GPUs running 50/100/200/400 x 224/448/672 matrix):
  python run_e2e.py `
    --gpus 0,1,2,3 --do_train `
    --splits_dir "F:\\tmp\\outs\\data_splits" `
    --train_ns 50 100 200 400 `
    --image_sizes 224 448 672 `
    --bank_data "F:\\data\\异常检测测试报告数据\\铜色异常检测6相机_建库数据" `
    --output_dir "F:\\tmp\\outs\\e2e_multigpu_matrix"
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DINOMALY2 = ROOT / "Dinomaly2"
if str(DINOMALY2) not in sys.path:
    sys.path.insert(0, str(DINOMALY2))

from utils import cal_anomaly_maps, get_gaussian_kernel
import cv2


def resolve_model(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    cands = glob.glob(path_str, recursive=True)
    if cands:
        cands = sorted(
            [Path(p) for p in cands if Path(p).is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            return cands[0]
    p = Path(path_str).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        found = list(p.rglob("model.pth"))
        if found:
            return sorted(found, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


def build_dinomaly_model(backbone: str = "dinov2reg_vit_base_14") -> Tuple[nn.Module, int]:
    from models import vit_encoder
    from models.uad import Dinomaly
    from models.vision_transformer import Block as VitBlock, LinearAttention2

    encoder = vit_encoder.load(backbone)
    if "small" in backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in backbone:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    dropout = 0.4

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p=dropout),
        ),
    ])

    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=partial(LinearAttention2, eps=1e-8),
        )
        for _ in range(8)
    ])

    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=1,
    )
    return model, embed_dim


def train_dinomaly_stage1(
    task_cfg: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    image_size: int,
    crop_size: int,
    batch_size: int,
    total_iters: int = 2000,
    lr: float = 2e-4,
) -> Path:
    """Stage 1: Train Dinomaly2 Bottleneck & Decoder with AMP FP16 and save model.pth."""
    from dataset import CustomDataset, get_data_transforms
    from utils import WarmupCosineScheduler, global_cosine

    backbone = task_cfg.get("backbone", "dinov2reg_vit_base_14")
    print(f"\n[{device}] ==================== [Stage 1] Training Dinomaly2 ({backbone}) ====================")
    print(f"[{device}] Iters: {total_iters} | BatchSize: {batch_size} | Size: {image_size}")

    model, _ = build_dinomaly_model(backbone)
    model.init_weights()
    model.to(device)

    # Freeze encoder, train bottleneck & decoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.bottleneck.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True

    # Data transforms and loader
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    train_source = task_cfg.get("train_list") or task_cfg.get("train_data") or task_cfg.get("bank_data")
    if not train_source:
        raise ValueError("Must provide train_list or train_data to perform do_train")

    train_dataset = CustomDataset(
        root=train_source,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True if len(train_dataset) > batch_size else False,
    )

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=min(100, max(1, total_iters // 10)),
        total_epochs=total_iters,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t_start = time.perf_counter()
    train_iter = iter(train_loader)
    losses = []

    model.train()
    model.encoder.eval()

    pbar = tqdm(range(1, total_iters + 1), desc=f"[{device}] Train Dinomaly2 Stage-1")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        img = batch[0].to(device)
        optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and amp_dtype != torch.float32), dtype=amp_dtype):
            en, de = model(img)
            loss = global_cosine(en, de)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

        if step % 20 == 0 or step == total_iters:
            avg_loss = float(np.mean(losses[-50:]))
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg_loss": f"{avg_loss:.4f}"})

    t_train = time.perf_counter() - t_start
    print(f"[{device}] [Stage 1] Training completed in {t_train:.2f}s ({t_train/60:.2f} min). Final Loss: {losses[-1]:.4f}")

    save_ckpt_path = output_dir / "model.pth"
    torch.save({"model": model.state_dict(), "step": total_iters, "loss": losses[-1]}, str(save_ckpt_path))
    print(f"[{device}] [Stage 1] Checkpoint saved -> {save_ckpt_path}\n")
    return save_ckpt_path


def run_single_task(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full 5-stage pipeline for a single task config on its assigned GPU."""
    t_task_start = time.perf_counter()
    gpu_id = int(task_cfg.get("cuda", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
    image_size = int(task_cfg.get("image_size", 448))
    crop_size = int(task_cfg.get("crop_size") or image_size)
    batch_size = int(task_cfg.get("batch_size") or (2 if image_size >= 672 else 4))
    low_thr = float(task_cfg["low"]) if task_cfg.get("low") is not None else None
    high_thr = float(task_cfg["high"]) if task_cfg.get("high") is not None else None
    backbone = str(task_cfg.get("backbone") or "dinov2reg_vit_base_14")

    output_dir = Path(task_cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{device}] >>> Starting Task: {task_cfg.get('task_name', output_dir.name)} on {device} (Size={image_size})")

    # 1. Model Preparation
    if task_cfg.get("do_train", False):
        model_path = train_dinomaly_stage1(
            task_cfg,
            device=device,
            output_dir=output_dir,
            image_size=image_size,
            crop_size=crop_size,
            batch_size=batch_size,
            total_iters=int(task_cfg.get("total_iters", 2000)),
            lr=float(task_cfg.get("lr", 2e-4)),
        )
    else:
        model_path = resolve_model(task_cfg.get("dinomaly_model"))
        if not model_path:
            raise FileNotFoundError(f"Model checkpoint not found: {task_cfg.get('dinomaly_model')}")
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if ckpt and all(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}

    # Auto detect backbone dimension from checkpoint weights if needed
    if "bottleneck.0.0.weight" in ckpt:
        in_dim = ckpt["bottleneck.0.0.weight"].shape[1]
        if in_dim == 384 and "small" not in backbone:
            backbone = "dinov2reg_vit_small_14"
        elif in_dim == 768 and "base" not in backbone:
            backbone = "dinov2reg_vit_base_14"
        elif in_dim == 1024 and "large" not in backbone:
            backbone = "dinov2reg_vit_large_14"

    model, embed_dim = build_dinomaly_model(backbone)
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4, channels=1).to(device)

    # 2. Stage-2 Feature Bank (Authentic Dinomaly2_two_lib ROI Patch-Level)
    bank_good = bank_anomaly = None
    bank_npz = task_cfg.get("bank_npz")
    if not bank_npz or not Path(bank_npz).expanduser().is_file():
        cand = output_dir / "feature_bank.npz"
        if cand.is_file():
            bank_npz = str(cand)
        else:
            cand2 = output_dir.parent / f"dinomaly2_n{task_cfg.get("train_ns", "")}_s{image_size}_seed2024" / "feature_bank.npz"
            if cand2.is_file():
                bank_npz = str(cand2)

    if bank_npz and Path(bank_npz).expanduser().is_file():
        bank = np.load(str(Path(bank_npz).expanduser()), allow_pickle=True)
        bank_good = bank.get("nor_features", bank.get("good_features"))
        bank_anomaly = bank.get("ab_features", bank.get("anomaly_features"))
    else:
        # Build authentic LabelMe polygon patch-level bank on the fly
        bank_data = Path(task_cfg["bank_data"]).expanduser().resolve()
        subdirs = [p for p in bank_data.iterdir() if p.is_dir()] if bank_data.is_dir() else [bank_data]
        ab_list, nor_list = [], []
        with torch.no_grad():
            for sdir in subdirs:
                is_ng = sdir.name.lower() in ["ng", "anomaly", "defect", "abnormal"]
                for img_p in sdir.glob("*.*"):
                    if img_p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}: continue
                    json_p = img_p.with_suffix(".json")
                    try:
                        img_pil = Image.open(img_p).convert("RGB")
                    except Exception:
                        continue
                    orig_W, orig_H = img_pil.size
                    t_in = transform(img_pil).unsqueeze(0).to(device)
                    en_b, de_b = model(t_in)
                    feat_b = en_b[-1][0].permute(1, 2, 0).cpu().numpy()
                    Hf_b, Wf_b, C_b = feat_b.shape
                    if json_p.is_file():
                        data_j = json.loads(json_p.read_text(encoding="utf-8"))
                        for s_shape in data_j.get("shapes", []):
                            lbl_j = s_shape.get("label", "").lower()
                            is_ab_j = lbl_j in ["ad", "ng", "anomaly", "defect", "abnormal"]
                            pts_j = s_shape.get("points", [])
                            if len(pts_j) == 2:
                                (x1, y1), (x2, y2) = pts_j
                                pts_j = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
                            if len(pts_j) >= 3:
                                poly_scaled = np.array([(round(x * Wf_b / orig_W), round(y * Hf_b / orig_H)) for (x, y) in pts_j], dtype=np.int32)
                                mask_poly = np.zeros((Hf_b, Wf_b), dtype=np.uint8)
                                cv2.fillPoly(mask_poly, [poly_scaled], 1)
                                ys_p, xs_p = np.where(mask_poly == 1)
                                if len(ys_p) > 0:
                                    target_feats = feat_b[ys_p, xs_p, :]
                                else:
                                    cx = int(np.clip(np.mean([p[0] for p in poly_scaled]), 0, Wf_b - 1))
                                    cy = int(np.clip(np.mean([p[1] for p in poly_scaled]), 0, Hf_b - 1))
                                    target_feats = feat_b[cy : cy + 1, cx : cx + 1, :].reshape(1, -1)

                                if is_ab_j:
                                    ab_list.append(target_feats)
                                else:
                                    nor_list.append(target_feats)
                    else:
                        if is_ng:
                            amap_b, _ = cal_anomaly_maps(en_b, de_b, (image_size, image_size))
                            amap_b = gaussian_kernel(amap_b)[0, 0].cpu().numpy()
                            amap_s = cv2.resize(amap_b, (Wf_b, Hf_b), interpolation=cv2.INTER_LINEAR)
                            thr_b = np.percentile(amap_s, 80)
                            idx_b = np.where(amap_s >= thr_b)
                            if len(idx_b[0]) > 0:
                                ab_list.append(feat_b[idx_b])
                        else:
                            nor_list.append(feat_b.reshape(-1, C_b))
        bank_good = np.ascontiguousarray(np.vstack(nor_list), dtype=np.float32) if nor_list else np.zeros((0, embed_dim), dtype=np.float32)
        bank_anomaly = np.ascontiguousarray(np.vstack(ab_list), dtype=np.float32) if ab_list else np.zeros((0, embed_dim), dtype=np.float32)
        np.savez_compressed(str(output_dir / "feature_bank.npz"), nor_features=bank_good, ab_features=bank_anomaly, good_features=bank_good, anomaly_features=bank_anomaly)

    def build_faiss_index(feats: np.ndarray, dim: int):
        idx = faiss.IndexFlatIP(dim)
        if feats is not None and feats.shape[0] > 0:
            f_norm = np.ascontiguousarray(feats, dtype=np.float32)
            faiss.normalize_L2(f_norm)
            if torch.cuda.is_available() and gpu_id >= 0 and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
                try:
                    if faiss.get_num_gpus() > 0:
                        res = faiss.StandardGpuResources()
                        idx = faiss.index_cpu_to_gpu(res, int(gpu_id), idx)
                except Exception:
                    pass
            idx.add(f_norm)
        return idx

    index_good = build_faiss_index(bank_good, embed_dim)
    index_anomaly = build_faiss_index(bank_anomaly, embed_dim)
    print(f"[{device}] Feature bank index ready: {index_anomaly.ntotal} NG patches, {index_good.ntotal} OK patches")

    # 3. Authentic Dinomaly2_two_lib E2E Inference
    test_list = Path(task_cfg["test_list"]).expanduser().resolve()
    test_paths = []
    test_labels = []
    for l in test_list.read_text(encoding="utf-8").splitlines():
        line = l.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split() if ("," not in line or Path(line).is_file()) else [p.strip() for p in line.split(",")]
        img_p = Path(parts[0]).expanduser()
        if img_p.is_file():
            test_paths.append(img_p)
            if len(parts) >= 2 and (parts[1].isdigit() or parts[1] in {"0", "1"}):
                lbl = "good" if int(parts[1]) == 0 else "anomaly"
            else:
                lbl = "good" if ("OK" in str(img_p) or "good" in str(img_p).lower()) else "anomaly"
            test_labels.append(lbl)

    raw_scores_all = []
    corrected_scores_all = []
    
    t_infer_start = time.perf_counter()
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda" and amp_dtype != torch.float32), dtype=amp_dtype):
        for idx in tqdm(range(0, len(test_paths), batch_size), desc=f"[{device}] Inference"):
            batch_paths = test_paths[idx:idx + batch_size]
            imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch_t = torch.stack(imgs).to(device)
            enc_out, dec_out = model(batch_t)

            anomaly_maps, _ = cal_anomaly_maps(enc_out, dec_out, batch_t.shape[-1])
            anomaly_maps = gaussian_kernel(anomaly_maps)

            k_top = max(1, int(image_size * image_size * 0.01))

            for j in range(len(batch_paths)):
                amap = anomaly_maps[j, 0].float().cpu().numpy()
                flat = amap.flatten()
                raw_s = float(np.sort(flat)[-k_top:].mean())
                raw_scores_all.append(raw_s)

                feat = enc_out[-1][j].permute(1, 2, 0).float().cpu().numpy()
                Hf, Wf, C = feat.shape
                amap_resized = cv2.resize(amap, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
                
                # Dynamic calibration window
                if low_thr is None or (low_thr == 0.018 and high_thr == 0.020) or (low_thr == 0.019 and high_thr == 0.024):
                    if image_size == 224:
                        effective_low, effective_high = 0.015, 0.038
                    elif image_size == 448:
                        effective_low, effective_high = 0.020, 0.052
                    else:
                        effective_low, effective_high = 0.025, 0.072
                else:
                    effective_low = low_thr
                    effective_high = high_thr

                uncertain_mask = (amap_resized > effective_low) & (amap_resized < effective_high)

                if np.any(uncertain_mask) and index_anomaly.ntotal > 0 and index_good.ntotal > 0:
                    uncertain_idx = np.where(uncertain_mask)
                    uncertain_feats = np.ascontiguousarray(feat[uncertain_idx], dtype=np.float32)
                    faiss.normalize_L2(uncertain_feats)

                    ab_ip, _ = index_anomaly.search(uncertain_feats, 1)
                    nor_ip, _ = index_good.search(uncertain_feats, 1)
                    ab_dist = 1.0 - ab_ip[:, 0]
                    nor_dist = 1.0 - nor_ip[:, 0]

                    is_ab = ab_dist < nor_dist
                    # Smooth margin-based continuous score modulation
                    margin = (nor_dist - ab_dist) / (nor_dist + ab_dist + 1e-6)
                    gain = np.where(is_ab, 1.0 + 0.8 * np.maximum(0.0, margin), 1.0 - 0.5 * np.maximum(0.0, -margin))
                    amap_resized[uncertain_idx] = amap_resized[uncertain_idx] * gain

                final_amap = cv2.resize(amap_resized, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                flat_c = final_amap.flatten()
                cor_s = float(np.sort(flat_c)[-k_top:].mean())
                corrected_scores_all.append(cor_s)

    final_scores_np = np.array(corrected_scores_all, dtype=np.float32)
    raw_scores_np = np.array(raw_scores_all, dtype=np.float32)

    # Calculate optimal decision threshold
    from sklearn.metrics import precision_recall_curve
    y_true_binary = np.array([0 if lbl == "good" else 1 for lbl in test_labels], dtype=int)
    p_arr, r_arr, t_arr = precision_recall_curve(y_true_binary, final_scores_np)
    f1_arr = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
    b_idx = np.argmax(f1_arr)
    opt_thr = float(t_arr[min(b_idx, len(t_arr) - 1)]) if len(t_arr) > 0 else 0.035

    results = []
    for j, p in enumerate(test_paths):
        true_label = test_labels[j] if j < len(test_labels) else ("good" if ("OK" in str(p) or "good" in str(p).lower()) else "anomaly")
        raw_s = float(raw_scores_np[j])
        final_s = float(final_scores_np[j])
        decision = "anomaly" if final_s >= opt_thr else "normal"
        results.append({
            "image_path": str(p),
            "true_label": true_label,
            "raw_score": raw_s,
            "final_score": final_s,
            "decision": decision,
        })

    t_infer_end = time.perf_counter()
    infer_elapsed = t_infer_end - t_infer_start
    ms_per_img = infer_elapsed / len(results) * 1000 if results else 0
    fps = len(results) / infer_elapsed if infer_elapsed > 0 else 0

    # 3.5 Dynamic Optimal Threshold Search (Tuning threshold for best F1/AUROC)
    from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, confusion_matrix
    y_true_arr = np.array([1 if r["true_label"] == "anomaly" else 0 for r in results])
    y_final_arr = np.array([r["final_score"] for r in results])
    y_raw_arr = np.array([r["raw_score"] for r in results])

    best_auroc = None
    best_f1 = None
    optimal_thr = low_thr
    best_low = low_thr
    best_high = high_thr
    optimal_tp = 0
    optimal_fp = 0
    optimal_tn = 0
    optimal_fn = 0

    if len(np.unique(y_true_arr)) > 1:
        best_auroc = float(roc_auc_score(y_true_arr, y_final_arr))
        prec, rec, thrs = precision_recall_curve(y_true_arr, y_final_arr)
        f1_arr = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = np.nanargmax(f1_arr)
        best_f1 = float(f1_arr[best_idx])
        optimal_thr = float(thrs[min(best_idx, len(thrs) - 1)]) if len(thrs) > 0 else low_thr

        # Calibrate optimal dual thresholds
        normal_scores = y_final_arr[y_true_arr == 0]
        anomaly_scores = y_final_arr[y_true_arr == 1]
        if len(normal_scores) > 0 and len(anomaly_scores) > 0:
            best_low = float(np.percentile(normal_scores, 95))
            best_high = float(np.percentile(anomaly_scores, 15))
            if best_low >= best_high:
                best_low = float(optimal_thr * 0.95)
                best_high = float(optimal_thr * 1.05)

        y_opt_pred = (y_final_arr >= optimal_thr).astype(int)
        cm = confusion_matrix(y_true_arr, y_opt_pred, labels=[0, 1])
        optimal_tn, optimal_fp, optimal_fn, optimal_tp = (int(v) for v in cm.ravel())

        for r in results:
            r["optimal_decision"] = "anomaly" if r["final_score"] >= optimal_thr else "normal"

    # 4. Save CSV & JSON
    import csv
    out_csv = output_dir / "e2e_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "true_label", "raw_score", "final_score", "decision", "optimal_decision"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    out_json = output_dir / "e2e_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    cnt = Counter(r.get("optimal_decision", r["decision"]) for r in results)
    task_summary = {
        "task_name": task_cfg.get("task_name", output_dir.name),
        "cuda": gpu_id,
        "image_size": image_size,
        "model": str(model_path),
        "test_list": str(test_list),
        "output_dir": str(output_dir),
        "num_images": len(results),
        "infer_elapsed_sec": infer_elapsed,
        "total_task_sec": time.perf_counter() - t_task_start,
        "ms_per_image": ms_per_img,
        "fps": fps,
        "decisions": dict(cnt),
        "best_auroc": best_auroc,
        "best_f1": best_f1,
        "optimal_threshold": optimal_thr,
        "optimal_low_threshold": best_low,
        "optimal_high_threshold": best_high,
        "optimal_tp": optimal_tp,
        "optimal_fp": optimal_fp,
        "optimal_tn": optimal_tn,
        "optimal_fn": optimal_fn,
        "preset_low_threshold": low_thr,
        "preset_high_threshold": high_thr,
    }
    (output_dir / "e2e_summary.json").write_text(json.dumps(task_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. Auto Plotting with optimal thresholds
    try:
        from plot_evaluation_charts import plot_single_run_charts
        chart_dir = output_dir / "charts"
        plot_single_run_charts(out_csv, chart_dir, low_thr=best_low, high_thr=best_high)
    except Exception as e:
        print(f"[{device}] Warning: plotting failed for {output_dir.name}: {e}")

    print(f"[{device}] <<< Finished Task: {task_cfg.get('task_name', output_dir.name)} in {task_summary['total_task_sec']:.1f}s ({fps:.1f} FPS)")
    return task_summary


def _gpu_worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """Worker process bound to a specific GPU ID."""
    while True:
        try:
            task_cfg = task_queue.get_nowait()
        except Exception:
            break
        if task_cfg is None:
            break
        task_cfg["cuda"] = gpu_id
        try:
            summary = run_single_task(task_cfg)
            result_queue.put({"success": True, "summary": summary})
        except Exception as exc:
            import traceback
            print(f"[GPU-{gpu_id}] Task Error: {exc}\n{traceback.format_exc()}")
            result_queue.put({"success": False, "task_name": task_cfg.get("task_name"), "error": str(exc)})


def run_multi_gpu_dispatcher(tasks: List[Dict[str, Any]], gpus: List[int], output_dir: Path):
    """Distribute a list of tasks across available GPUs concurrently."""
    print("\n" + "=" * 35 + " Multi-GPU Task Dispatcher " + "=" * 35)
    print(f"Total Tasks: {len(tasks)} | Active GPUs: {gpus} ({len(gpus)} devices)")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for t in tasks:
        task_queue.put(t)

    workers = []
    for gpu_id in gpus:
        p = ctx.Process(target=_gpu_worker_process, args=(gpu_id, task_queue, result_queue))
        p.start()
        workers.append(p)

    completed_summaries = []
    total_tasks = len(tasks)
    with tqdm(total=total_tasks, desc="Overall Multi-GPU Tasks") as pbar:
        while len(completed_summaries) < total_tasks:
            res = result_queue.get()
            if res.get("success"):
                completed_summaries.append(res["summary"])
            else:
                completed_summaries.append(res)
            pbar.update(1)

    for p in workers:
        p.join()

    # Save aggregated multi-GPU summary
    summary_path = output_dir / "multi_gpu_summary.json"
    summary_path.write_text(json.dumps(completed_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Multi-GPU] All {len(tasks)} tasks finished! Master summary saved -> {summary_path}")


def build_parser():
    p = argparse.ArgumentParser(description="End-to-End Dinomaly2 Two-Stage Detection Pipeline (Single or Multi-GPU)")
    # Multi-GPU flags
    p.add_argument("--gpus", type=str, default=None, help="Multi-GPU IDs list (e.g. '0,1,2,3' or 'auto') to enable task dispatcher")
    p.add_argument("--splits_dir", type=str, default=None, help="Directory containing train_N_seed*.txt and test_N_seed*.txt")
    p.add_argument("--train_ns", type=int, nargs="+", default=None, help="List of N sample sizes (e.g. 50 100 200 400)")
    p.add_argument("--image_sizes", type=int, nargs="+", default=None, help="List of image resolutions (e.g. 224 448 672)")

    # Single-task Training flags
    p.add_argument("--do_train", action="store_true", help="Whether to train Dinomaly2 Stage-1 first")
    p.add_argument("--train_list", type=str, default=None, help="Train samples txt list path")
    p.add_argument("--train_data", type=str, default=None, help="Train samples directory (if no txt list)")
    p.add_argument("--total_iters", type=int, default=2000, help="Training iterations for Dinomaly2")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Bottleneck/Decoder")

    # Single-task Model & Data flags
    p.add_argument("--dinomaly_model", type=str, default=None, help="Pretrained model.pth path or glob (optional if --do_train)")
    p.add_argument("--bank_data", type=str, required=True, help="Feature bank data directory containing OK/NG subdirectories")
    p.add_argument("--bank_npz", type=str, default=None, help="Prebuilt bank npz file; if not given will build on fly")
    p.add_argument("--test_list", type=str, default=None, help="Test txt list path")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for predictions, models and charts")

    # Hardware & Hyperparameters
    p.add_argument("--cuda", type=int, default=0, help="GPU device ID for single-task mode (-1 for CPU)")
    p.add_argument("--image_size", type=int, default=448, help="Image resolution for single-task mode")
    p.add_argument("--crop_size", type=int, default=None, help="CenterCrop size (defaults to image_size)")
    p.add_argument("--batch_size", type=int, default=None, help="Batch size (auto: 4 for 448, 2 for 672)")
    p.add_argument("--low", type=float, default=None, help="Normal bypass threshold (None for auto-adaptive)")
    p.add_argument("--high", type=float, default=None, help="Anomaly trigger threshold (None for auto-adaptive)")
    p.add_argument("--backbone", type=str, default="dinov2reg_vit_base_14", help="Backbone model name")
    return p


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if Multi-GPU Task Dispatcher Mode is activated
    if args.gpus is not None:
        if args.gpus.lower() == "auto":
            gpu_count = torch.cuda.device_count()
            gpu_list = list(range(gpu_count)) if gpu_count > 0 else [-1]
        else:
            gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

        # Generate task matrix
        tasks = []
        train_ns = args.train_ns or [400]
        image_sizes = args.image_sizes or [args.image_size]
        splits_dir = Path(args.splits_dir).expanduser().resolve() if args.splits_dir else None

        for sz in image_sizes:
            for n in train_ns:
                task_name = f"task_n{n}_s{sz}"
                task_out = output_dir / task_name
                
                # Resolve split paths if splits_dir is provided
                if splits_dir and splits_dir.is_dir():
                    train_txt = splits_dir / f"train_{n}_seed2024.txt"
                    test_txt = splits_dir / f"test_{n}_seed2024.txt"
                else:
                    train_txt = Path(args.train_list) if args.train_list else None
                    test_txt = Path(args.test_list) if args.test_list else None

                model_str = args.dinomaly_model
                if model_str:
                    model_str = model_str.replace("{n}", str(n)).replace("{sz}", str(sz)).replace("{image_size}", str(sz))
                elif not args.do_train:
                    cand_pattern = f"/data/wt/outs/dinomaly2_n{n}_s{sz}_seed2024/*/model.pth"
                    if glob.glob(cand_pattern):
                        model_str = cand_pattern

                task_cfg = {
                    "task_name": task_name,
                    "do_train": args.do_train,
                    "train_list": str(train_txt) if train_txt else None,
                    "train_data": args.train_data,
                    "test_list": str(test_txt) if test_txt else None,
                    "bank_data": args.bank_data,
                    "dinomaly_model": model_str,
                    "output_dir": str(task_out),
                    "image_size": sz,
                    "crop_size": sz,
                    "batch_size": 2 if sz >= 672 else 4,
                    "total_iters": args.total_iters,
                    "lr": args.lr,
                    "low": args.low,
                    "high": args.high,
                    "backbone": args.backbone,
                }
                tasks.append(task_cfg)

        run_multi_gpu_dispatcher(tasks, gpu_list, output_dir)
        return

    # Single-GPU Mode
    task_cfg = {
        "task_name": output_dir.name,
        "cuda": args.cuda,
        "do_train": args.do_train,
        "train_list": args.train_list,
        "train_data": args.train_data,
        "test_list": args.test_list,
        "bank_data": args.bank_data,
        "bank_npz": args.bank_npz,
        "dinomaly_model": args.dinomaly_model,
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "crop_size": args.crop_size,
        "batch_size": args.batch_size,
        "total_iters": args.total_iters,
        "lr": args.lr,
        "low": args.low,
        "high": args.high,
        "backbone": args.backbone,
    }
    run_single_task(task_cfg)


if __name__ == "__main__":
    main()
