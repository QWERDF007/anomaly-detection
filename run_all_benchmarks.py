#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Master Runner for All 3 Industrial Anomaly Detection Benchmarks
Uses the newly consolidated multi-GPU pipeline runner: run_benchmark_pipeline.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path("/data/wt/anomaly-detection")
PYTHON = sys.executable

DATASETS = [
    {
        "name": "透气膜",
        "dataset_root": "/data/wt/ramdisk/透气膜/透气膜",
        "bank_data": "/data/wt/ramdisk/透气膜/建库数据",
        "train_sizes": ["20", "50", "100", "150"],
        "image_sizes": ["224", "448", "672"],
    },
    {
        "name": "铜色异常检测4相机",
        "dataset_root": "/data/wt/ramdisk/铜色异常检测4相机/铜色异常检测4相机",
        "bank_data": "/data/wt/ramdisk/铜色异常检测4相机/建库数据",
        "train_sizes": ["20", "50", "100", "200"],
        "image_sizes": ["224", "448", "672"],
    },
    {
        "name": "铜色异常检测6相机",
        "dataset_root": "/data/wt/ramdisk/铜色异常检测6相机/铜色异常检测6相机",
        "bank_data": "/data/wt/ramdisk/铜色异常检测6相机/建库数据",
        "train_sizes": ["50", "100", "200", "400"],
        "image_sizes": ["224", "448", "672"],
    },
    {
        "name": "leishi_026",
        "dataset_root": "/data/wt/ramdisk/leishi_026",
        "bank_data": "",
        "train_sizes": ["20", "50", "100", "150"],
        "image_sizes": ["224", "448", "672"],
    },
]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master Runner for Industrial Datasets Benchmark")
    parser.add_argument("--dataset", type=str, default="all", help="Target dataset ('all', '透气膜', '铜色异常检测4相机', '铜色异常检测6相机', 'leishi_026')")
    parser.add_argument("--outs_base", type=str, default="/data/wt/exp0906", help="Base directory for experiment outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="Fixed batch size")
    parser.add_argument("--max_iters", type=str, nargs="+", default=["1000", "2000", "5000", "20000", "40000"], help="Max iters")
    parser.add_argument("--gpus", type=str, default="auto", help="GPUs to use")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", dest="dry_run", help="Dry run verification")
    args = parser.parse_args()

    active_datasets = [d for d in DATASETS if args.dataset == "all" or args.dataset in d["name"]]
    if not active_datasets:
        print(f"[ERROR] No matching dataset found for '{args.dataset}'. Available: {[d['name'] for d in DATASETS]}")
        sys.exit(1)

    print("=" * 80)
    print(f"Starting Full Benchmark Across {len(active_datasets)} Datasets:")
    for i, ds in enumerate(active_datasets):
        print(f" {i+1}. {ds['name']} (Bank: {'Yes' if ds['bank_data'] else 'No'})")
    print(f"Fixed Batch: {args.batch_size} | Max Iters: {args.max_iters} | GPUs: {args.gpus}")
    print("=" * 80)

    for idx, ds in enumerate(active_datasets):
        print("\n" + "=" * 80)
        print(f"=== [{idx+1}/{len(DATASETS)}] Starting Pipeline for {ds['name']} ===")
        print("=" * 80 + "\n")
        
        ds_outs_dir = os.path.join(args.outs_base, ds["name"])
        cmd = [
            PYTHON, str(ROOT / "run_benchmark_pipeline.py"),
            "--dataset_root", ds["dataset_root"],
            "--outs_dir", ds_outs_dir,
            "--batch_size", str(args.batch_size),
            "--train_sizes"] + ds["train_sizes"] + [
            "--image_sizes"] + ds["image_sizes"] + [
            "--gpus", args.gpus,
            "--max_iters"] + args.max_iters + [
            "--seed", "2024"
        ]
        if ds["bank_data"]:
            cmd.extend(["--bank_data", ds["bank_data"]])
        if args.dry_run:
            cmd.append("--dry-run")
        
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.perf_counter() - t0
        
        if proc.returncode != 0:
            print(f"\n[ERROR] Pipeline failed for {ds['name']} (returncode={proc.returncode}) after {elapsed/60:.2f} mins")
            sys.exit(proc.returncode)
        else:
            print(f"\n[SUCCESS] Completed {ds['name']} in {elapsed/60:.2f} mins! Outputs -> {ds_outs_dir}")

    print("\n" + "=" * 80)
    print("ALL INDUSTRIAL DATASETS BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.outs_base}")
    print("=" * 80)

if __name__ == "__main__":
    main()
