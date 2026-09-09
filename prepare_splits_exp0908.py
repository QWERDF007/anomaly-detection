#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Proportional Stratified Data Splitter and Labelme Mask Renderer for Industrial Datasets.

Datasets:
  - shimo-all
  - 台达散热片栅格
  - 台达散热片正反面

Key Specifications:
  1. Labelme JSON Parsing:
     - No annotation file OR shapes == []  => Normal (良品, label=0)
     - Annotation file with shapes > 0     => Anomalous (异常, label=1)
  2. Proportional Sampling across all subdirectories:
     - For target N in [100, 200, 400], normal images are extracted from each subdirectory
       in strict proportion to that subdirectory's share of total normal images.
     - Largest remainder method (Hare-Niemeyer) ensures sum(n_i) == N exactly.
     - Strict nested subset guarantee: train_n100 ⊂ train_n200 ⊂ train_n400.
  3. Evaluation Set ("剩下的图用于评测"):
     - Unseen Normal: all normal images remaining after sampling max N (N=400).
     - Anomalous: all anomalous images across all subdirectories.
     - Test Full = Unseen Normal (label 0) + All Anomalous (label 1).
     - Clean In-Domain = All N=400 training normal images (label 0).
  4. Real Ground Truth Mask Rendering ("真实标注使用labelme的标注，矩形或者多边形"):
     - Renders rectangle, polygon, circle, line annotations into binary PNG masks (255 defect, 0 background).
     - Saved to /data/wt/exp0908/{dataset_name}/ground_truth/{sub_name}/{img.stem}_mask.png
"""
from __future__ import annotations

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def render_labelme_mask(json_path: Path, img_path: Path, out_mask_path: Path) -> Path:
    """Renders labelme JSON shapes (polygon, rectangle, etc.) into a binary mask PNG."""
    try:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as e:
        print(f"[WARN] Failed to read JSON {json_path}: {e}")
        data = {}

    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if not h or not w or h <= 0 or w <= 0:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            h, w = 1024, 1024

    mask = np.zeros((h, w), dtype=np.uint8)
    shapes = data.get("shapes", [])

    for s in shapes:
        st = (s.get("shape_type") or "polygon").lower()
        pts = s.get("points", [])
        if not pts:
            continue
        pts_arr = np.array(pts, dtype=np.float32)

        if st == "rectangle":
            if len(pts_arr) >= 2:
                x1, y1 = np.rint(pts_arr[0]).astype(np.int32)
                x2, y2 = np.rint(pts_arr[1]).astype(np.int32)
                xmin, xmax = max(0, min(x1, x2)), min(w - 1, max(x1, x2))
                ymin, ymax = max(0, min(y1, y2)), min(h - 1, max(y1, y2))
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
        elif st == "polygon":
            if len(pts_arr) >= 3:
                pts_clamped = np.copy(pts_arr)
                pts_clamped[:, 0] = np.clip(pts_clamped[:, 0], 0, w - 1)
                pts_clamped[:, 1] = np.clip(pts_clamped[:, 1], 0, h - 1)
                int_pts = np.rint(pts_clamped).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [int_pts], 255)
        elif st == "circle":
            if len(pts_arr) >= 2:
                center = tuple(np.rint(pts_arr[0]).astype(np.int32))
                radius = int(round(float(np.linalg.norm(pts_arr[1] - pts_arr[0]))))
                cv2.circle(mask, center, max(1, radius), 255, -1)
        elif st in {"line", "linestrip"}:
            if len(pts_arr) >= 2:
                int_pts = np.rint(pts_arr).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(mask, [int_pts], isClosed=False, color=255, thickness=5)
        elif st == "point":
            center = tuple(np.rint(pts_arr[0]).astype(np.int32))
            cv2.circle(mask, center, 5, 255, -1)
        else:
            if len(pts_arr) >= 3:
                int_pts = np.rint(pts_arr).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [int_pts], 255)

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask_path), mask)
    return out_mask_path


def allocate_proportional_quotas(sub_counts: Dict[str, int], total_n: int) -> Dict[str, int]:
    """Computes exact proportional integer quotas using largest remainder method."""
    total_k = sum(sub_counts.values())
    if total_k == 0:
        raise ValueError("Total normal count is 0, cannot allocate quotas.")
    if total_n > total_k:
        raise ValueError(f"Requested total N={total_n} exceeds available normal images {total_k}.")

    raw = {k: total_n * v / total_k for k, v in sub_counts.items()}
    floors = {k: math.floor(v) for k, v in raw.items()}
    deficit = total_n - sum(floors.values())
    remainders = sorted(raw.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    for k in remainders[:deficit]:
        floors[k] += 1
    return floors


def scan_subdirectory(sub_dir: Path) -> Tuple[List[Path], List[Tuple[Path, Path]]]:
    """Scans a subdirectory for normal and anomalous images."""
    img_dir = sub_dir / "images"
    ann_dir = sub_dir / "annotations"
    if not img_dir.is_dir():
        img_candidates = [p for p in sub_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        img_candidates = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    normal_imgs = []
    anomaly_items = []

    for img in sorted(img_candidates):
        json_cand = (ann_dir / (img.stem + ".json")) if ann_dir.is_dir() else (sub_dir / (img.stem + ".json"))
        if not json_cand.is_file():
            normal_imgs.append(img)
        else:
            try:
                with open(json_cand, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                shapes = data.get("shapes", [])
                if len(shapes) == 0:
                    normal_imgs.append(img)
                else:
                    anomaly_items.append((img, json_cand))
            except Exception as e:
                print(f"[WARN] Error reading {json_cand}: {e}, treating as normal.")
                normal_imgs.append(img)

    return normal_imgs, anomaly_items


def process_dataset(
    dataset_name: str,
    dataset_root: Path,
    out_dir: Path,
    train_sizes: List[int] = [100, 200, 400],
    seed: int = 2024,
    render_masks: bool = True
) -> dict:
    print("\n" + "=" * 80)
    print(f"=== Processing Dataset: {dataset_name} ===")
    print(f"Source Directory: {dataset_root}")
    print(f"Target Out Dir:   {out_dir}")
    print(f"Train Sizes N:    {train_sizes}")
    print("=" * 80, flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = out_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all subdirectories
    subdirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {dataset_root}")

    sub_normal_map: Dict[str, List[Path]] = {}
    sub_anomaly_map: Dict[str, List[Tuple[Path, Path]]] = {}
    sub_normal_counts: Dict[str, int] = {}

    for sub in subdirs:
        normals, anomalies = scan_subdirectory(sub)
        sub_normal_map[sub.name] = normals
        sub_anomaly_map[sub.name] = anomalies
        sub_normal_counts[sub.name] = len(normals)

    total_normal_all = sum(sub_normal_counts.values())
    total_anomaly_all = sum(len(v) for v in sub_anomaly_map.values())
    print(f"Discovered across {len(subdirs)} subdirectories: {total_normal_all} Normal, {total_anomaly_all} Anomalous.")

    for sub_name in sub_normal_counts:
        print(f"  Sub '{sub_name}': {sub_normal_counts[sub_name]} normal, {len(sub_anomaly_map[sub_name])} anomaly")

    sorted_train_sizes = sorted(train_sizes)
    max_train_n = sorted_train_sizes[-1]
    if max_train_n > total_normal_all:
        raise ValueError(f"Requested max N={max_train_n} exceeds total available normal images ({total_normal_all})")

    # 2. Compute proportional quotas for each N
    quotas_by_n: Dict[int, Dict[str, int]] = {}
    for n in sorted_train_sizes:
        quotas_by_n[n] = allocate_proportional_quotas(sub_normal_counts, n)
        print(f"\nQuota Allocation for N={n} (sum={sum(quotas_by_n[n].values())}):")
        for sub_name, q in quotas_by_n[n].items():
            pct = (q / n) * 100
            orig_pct = (sub_normal_counts[sub_name] / total_normal_all) * 100
            print(f"  - {sub_name:12s}: {q:4d} images ({pct:5.1f}% | original ratio: {orig_pct:5.1f}%)")

    # 3. Deterministic shuffling per subdirectory & nested slicing
    train_pools_by_n: Dict[int, List[Path]] = {n: [] for n in sorted_train_sizes}
    test_normal_pool: List[Path] = []

    for sub in subdirs:
        sub_name = sub.name
        normals = list(sub_normal_map[sub_name])
        rng = random.Random(seed + hash(sub_name) % 10000)
        rng.shuffle(normals)

        for idx in range(len(sorted_train_sizes) - 1):
            n_curr = sorted_train_sizes[idx]
            n_next = sorted_train_sizes[idx + 1]
            if quotas_by_n[n_curr][sub_name] > quotas_by_n[n_next][sub_name]:
                quotas_by_n[n_next][sub_name] = max(quotas_by_n[n_next][sub_name], quotas_by_n[n_curr][sub_name])

        sub_max_q = quotas_by_n[max_train_n][sub_name]
        for n in sorted_train_sizes:
            q = quotas_by_n[n][sub_name]
            train_pools_by_n[n].extend(normals[:q])

        remaining_normals = normals[sub_max_q:]
        test_normal_pool.extend(remaining_normals)

    # Verify train set sizes
    for n in sorted_train_sizes:
        assert len(train_pools_by_n[n]) == n, f"Error: Train N={n} has {len(train_pools_by_n[n])} images!"

    # Verify strict nested property
    for idx in range(len(sorted_train_sizes) - 1):
        n_curr = sorted_train_sizes[idx]
        n_next = sorted_train_sizes[idx + 1]
        set_curr = set(train_pools_by_n[n_curr])
        set_next = set(train_pools_by_n[n_next])
        assert set_curr.issubset(set_next), f"Error: train_n{n_curr} is not a strict subset of train_n{n_next}!"

    print(f"\n[OK] Strict nested subset verified: {' ⊂ '.join(f'train_n{n}' for n in sorted_train_sizes)}")
    print(f"Total Unseen Normal Images for Test: {len(test_normal_pool)} (all remaining from {total_normal_all} - {max_train_n})")

    # 4. Render Ground Truth Masks for Anomalies
    test_anomaly_items: List[Tuple[Path, Path]] = []
    for sub in subdirs:
        test_anomaly_items.extend(sub_anomaly_map[sub.name])

    print(f"\nRendering ground truth masks for {len(test_anomaly_items)} anomalous images...")
    if render_masks and test_anomaly_items:
        tasks = []
        for img, jf in test_anomaly_items:
            sub_name = img.parent.parent.name if img.parent.name == "images" else img.parent.name
            out_mask_path = gt_dir / sub_name / f"{img.stem}_mask.png"
            tasks.append((jf, img, out_mask_path))

        rendered_count = 0
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as executor:
            futures = [executor.submit(render_labelme_mask, jf, im, mp) for jf, im, mp in tasks]
            for f in futures:
                f.result()
                rendered_count += 1
                if rendered_count % 1000 == 0 or rendered_count == len(tasks):
                    print(f"  Rendered [{rendered_count}/{len(tasks)}] masks...", flush=True)

    # 5. Write Data Splits
    for n in sorted_train_sizes:
        train_file = splits_dir / f"train_n{n}.txt"
        with open(train_file, "w", encoding="utf-8") as fp:
            for p in train_pools_by_n[n]:
                fp.write(f"{p.resolve()}\n")
        print(f"Saved: {train_file} ({n} images)")

    test_file = splits_dir / "test_full.txt"
    test_items = [(p, 0) for p in test_normal_pool] + [(p, 1) for p, _ in test_anomaly_items]
    random.Random(seed).shuffle(test_items)
    with open(test_file, "w", encoding="utf-8") as fp:
        for p, lbl in test_items:
            fp.write(f"{p.resolve()}\t{lbl}\n")
    print(f"Saved: {test_file} ({len(test_items)} total images: {len(test_normal_pool)} normal, {len(test_anomaly_items)} anomaly)")

    clean_test_file = splits_dir / "test_clean_in_domain.txt"
    with open(clean_test_file, "w", encoding="utf-8") as fp:
        for p in train_pools_by_n[max_train_n]:
            fp.write(f"{p.resolve()}\t0\n")
    print(f"Saved: {clean_test_file} ({max_train_n} normal training images)")

    summary = {
        "dataset_name": dataset_name,
        "source_root": str(dataset_root),
        "total_images": total_normal_all + total_anomaly_all,
        "total_normal": total_normal_all,
        "total_anomaly": total_anomaly_all,
        "train_sizes": sorted_train_sizes,
        "subdirectories": {
            sub_name: {
                "original_normal": sub_normal_counts[sub_name],
                "original_anomaly": len(sub_anomaly_map[sub_name]),
                "quotas": {n: quotas_by_n[n][sub_name] for n in sorted_train_sizes}
            }
            for sub_name in sub_normal_counts
        },
        "test_breakdown": {
            "unseen_normal": len(test_normal_pool),
            "anomaly": len(test_anomaly_items),
            "total_test": len(test_items)
        }
    }
    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"Saved: {splits_dir / 'split_summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Proportional Stratified Data Splitter for Industrial Datasets")
    parser.add_argument("--data_dir", type=str, default="/data/wt/data", help="Root containing industrial datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["shimo-all", "台达散热片栅格", "台达散热片正反面"], help="Datasets to process")
    parser.add_argument("--out_base", type=str, default="/data/wt/exp0908", help="Output directory base")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[100, 200, 400], help="Target sample sizes N")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducible sampling")
    parser.add_argument("--skip_masks", action="store_true", help="Skip rendering binary ground truth masks")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_base = Path(args.out_base).expanduser().resolve()

    print("================================================================================")
    print("=== Industrial Dataset Proportional Stratified Splitting & Mask Rendering ===")
    print(f"Data Dir:     {data_dir}")
    print(f"Out Base:     {out_base}")
    print(f"Datasets:     {args.datasets}")
    print(f"Train Sizes:  {args.train_sizes}")
    print(f"Seed:         {args.seed}")
    print("================================================================================")

    all_summaries = {}
    for ds_name in args.datasets:
        ds_root = data_dir / ds_name
        if not ds_root.is_dir():
            print(f"[ERROR] Dataset directory not found: {ds_root}")
            continue
        out_ds_dir = out_base / ds_name
        summary = process_dataset(
            dataset_name=ds_name,
            dataset_root=ds_root,
            out_dir=out_ds_dir,
            train_sizes=args.train_sizes,
            seed=args.seed,
            render_masks=not args.skip_masks
        )
        all_summaries[ds_name] = summary

    print("\n" + "=" * 80)
    print("=== All Dataset Splits & Masks Generated Successfully! ===")
    for ds_name, sm in all_summaries.items():
        print(f"  [{ds_name}]: {sm['total_images']} imgs -> Train N={args.train_sizes}, Test={sm['test_breakdown']['total_test']} (Unseen Normal={sm['test_breakdown']['unseen_normal']}, Anomaly={sm['test_breakdown']['anomaly']})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
