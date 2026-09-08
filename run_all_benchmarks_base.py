#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/wt/anomaly-detection")
PYTHON = "/home/dell/miniconda3/envs/anomaly/bin/python"

datasets = [
    {
        "name": "透气膜",
        "dataset_root": "/data/wt/ramdisk/透气膜/透气膜",
        "bank_data": "/data/wt/ramdisk/透气膜/建库数据",
        "outs_dir": "/data/wt/report_base/透气膜",
        "train_sizes": ["20", "50", "100", "150"],
        "image_sizes": ["224", "448", "672"],
        "backbone": "dinov2reg_vit_base_14",
    },
    {
        "name": "铜色异常检测4相机",
        "dataset_root": "/data/wt/ramdisk/铜色异常检测4相机/铜色异常检测4相机",
        "bank_data": "/data/wt/ramdisk/铜色异常检测4相机/建库数据",
        "outs_dir": "/data/wt/report_base/铜色异常检测4相机",
        "train_sizes": ["20", "50", "100", "200"],
        "image_sizes": ["224", "448", "672"],
        "backbone": "dinov2reg_vit_base_14",
    },
    {
        "name": "铜色异常检测6相机",
        "dataset_root": "/data/wt/ramdisk/铜色异常检测6相机/铜色异常检测6相机",
        "bank_data": "/data/wt/ramdisk/铜色异常检测6相机/建库数据",
        "outs_dir": "/data/wt/report_base/铜色异常检测6相机",
        "train_sizes": ["50", "100", "200", "400"],
        "image_sizes": ["224", "448", "672"],
        "backbone": "dinov2reg_vit_base_14",
    }
]

for idx, ds in enumerate(datasets):
    print("\n" + "=" * 80)
    print(f"=== [{idx+1}/{len(datasets)}] Starting ViT-Base Pipeline for {ds['name']} ===")
    print("=" * 80, flush=True)

    cmd = [
        PYTHON, str(ROOT / "run_benchmark_pipeline.py"),
        "--dataset_root", ds["dataset_root"],
        "--bank_data", ds["bank_data"],
        "--outs_dir", ds["outs_dir"],
        "--backbone", ds["backbone"],
        "--train_sizes"] + ds["train_sizes"] + [
        "--image_sizes"] + ds["image_sizes"] + [
        "--gpus", "auto",
        "--max_iters", "2000",
        "--seed", "2024"
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd)
    elapsed = (time.perf_counter() - t0) / 60.0

    if res.returncode == 0:
        print(f"\n[SUCCESS] Completed {ds['name']} (ViT-Base) in {elapsed:.2f} mins! Outputs -> {ds['outs_dir']}\n", flush=True)
    else:
        print(f"\n[ERROR] Failed {ds['name']} with code {res.returncode}\n", flush=True)
        sys.exit(1)

print("\n" + "=" * 80)
print("ALL 3 INDUSTRIAL DATASETS BENCHMARK (ViT-Base) COMPLETED SUCCESSFULLY!")
print("=" * 80, flush=True)
