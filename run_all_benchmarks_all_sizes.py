#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/wt/anomaly-detection")
PYTHON = "/home/dell/miniconda3/envs/anomaly/bin/python"

backbones = [
    {"name": "small", "model": "dinov2reg_vit_small_14"},
    {"name": "base", "model": "dinov2reg_vit_base_14"},
    {"name": "large", "model": "dinov2reg_vit_large_14"},
]

dataset_configs = [
    {
        "name": "透气膜",
        "dataset_root": "/data/wt/ramdisk/透气膜/透气膜",
        "bank_data": "/data/wt/ramdisk/透气膜/建库数据",
        "train_sizes": ["20", "50", "100", "150"],
    },
    {
        "name": "铜色异常检测4相机",
        "dataset_root": "/data/wt/ramdisk/铜色异常检测4相机/铜色异常检测4相机",
        "bank_data": "/data/wt/ramdisk/铜色异常检测4相机/建库数据",
        "train_sizes": ["20", "50", "100", "200"],
    },
    {
        "name": "铜色异常检测6相机",
        "dataset_root": "/data/wt/ramdisk/铜色异常检测6相机/铜色异常检测6相机",
        "bank_data": "/data/wt/ramdisk/铜色异常检测6相机/建库数据",
        "train_sizes": ["50", "100", "200", "400"],
    },
]

image_sizes = ["224", "448", "672", "1024"]

total_experiments = len(backbones) * len(dataset_configs)
exp_idx = 0

for b_cfg in backbones:
    b_name = b_cfg["name"]
    b_model = b_cfg["model"]
    
    for ds_cfg in dataset_configs:
        exp_idx += 1
        ds_name = ds_cfg["name"]
        outs_dir = f"/data/wt/report/{b_name}/{ds_name}"
        
        print("\n" + "=" * 90)
        print(f"=== [{exp_idx}/{total_experiments}] Starting Backbone: {b_name} ({b_model}) | Dataset: {ds_name} ===")
        print(f"=== Output: {outs_dir} | Sizes: {image_sizes} ===")
        print("=" * 90, flush=True)

        cmd = [
            PYTHON, str(ROOT / "run_benchmark_pipeline.py"),
            "--dataset_root", ds_cfg["dataset_root"],
            "--bank_data", ds_cfg["bank_data"],
            "--outs_dir", outs_dir,
            "--backbone", b_model,
            "--train_sizes"] + ds_cfg["train_sizes"] + [
            "--image_sizes"] + image_sizes + [
            "--gpus", "auto",
            "--max_iters", "2000",
            "--seed", "2024"
        ]

        t0 = time.perf_counter()
        res = subprocess.run(cmd)
        elapsed = (time.perf_counter() - t0) / 60.0

        if res.returncode == 0:
            print(f"\n[SUCCESS] Completed {b_name}/{ds_name} in {elapsed:.2f} mins! Outputs -> {outs_dir}\n", flush=True)
        else:
            print(f"\n[ERROR] Failed {b_name}/{ds_name} with code {res.returncode}\n", flush=True)
            sys.exit(1)

print("\n" + "=" * 90)
print("ALL EXPERIMENTS ACROSS SMALL, BASE, LARGE WITH SIZES [224, 448, 672, 1024] COMPLETED SUCCESSFULLY!")
print("=" * 90, flush=True)
