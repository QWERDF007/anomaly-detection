#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Master Benchmark Pipeline Runner for exp0908 Datasets.

Datasets:
  1. shimo-all
  2. 台达散热片栅格
  3. 台达散热片正反面

Specifications:
  - Train Sample Sizes N: [100, 200, 400]
  - Image Sizes S: [224, 448, 672]
  - Iterations: [1000, 2000, 5000, 20000] (Strictly excluding 40000)
  - Batch Size: 8
  - GPUs: 8x RTX 4090 (auto)
  - Outs Dir: /data/wt/exp0908/{dataset_name}
  - Splits Dir: /data/wt/exp0908/{dataset_name}/data_splits
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path("/data/wt/anomaly-detection")
ANOMALY_PYTHON = Path("/home/dell/miniconda3/envs/anomaly/bin/python")
PYTHON = str(ANOMALY_PYTHON) if ANOMALY_PYTHON.is_file() else sys.executable

DATASETS = [
    {
        "name": "shimo-all",
        "dataset_root": "/data/wt/data/shimo-all",
        "splits_dir": "/data/wt/exp0908/shimo-all/data_splits",
        "outs_dir": "/data/wt/exp0908/shimo-all",
    },
    {
        "name": "台达散热片栅格",
        "dataset_root": "/data/wt/data/台达散热片栅格",
        "splits_dir": "/data/wt/exp0908/台达散热片栅格/data_splits",
        "outs_dir": "/data/wt/exp0908/台达散热片栅格",
    },
    {
        "name": "台达散热片正反面",
        "dataset_root": "/data/wt/data/台达散热片正反面",
        "splits_dir": "/data/wt/exp0908/台达散热片正反面/data_splits",
        "outs_dir": "/data/wt/exp0908/台达散热片正反面",
    },
]

TRAIN_SIZES = ["100", "200", "400"]
IMAGE_SIZES = ["224", "448", "672"]
MAX_ITERS = ["1000", "2000", "5000", "20000"]
BATCH_SIZE = "8"
SEED = "2024"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master Runner for exp0908 Industrial Datasets")
    parser.add_argument("--dataset", type=str, default="all", help="Target dataset ('all', 'shimo-all', '台达散热片栅格', '台达散热片正反面')")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Dry run verification")
    parser.add_argument("--gpus", type=str, default="auto", help="GPUs to use")
    args = parser.parse_args()

    active_datasets = [d for d in DATASETS if args.dataset == "all" or args.dataset == d["name"]]
    if not active_datasets:
        print(f"[ERROR] Unknown dataset: {args.dataset}. Available: {[d['name'] for d in DATASETS]}")
        sys.exit(1)

    print("=" * 80)
    print("=== Master Pipeline Orchestrator for exp0908 ===")
    print(f"Datasets to run ({len(active_datasets)}): {[d['name'] for d in active_datasets]}")
    print(f"Train Sizes N:   {TRAIN_SIZES}")
    print(f"Image Sizes S:   {IMAGE_SIZES}")
    print(f"Max Iters:       {MAX_ITERS} (excluding 40000)")
    print(f"Batch Size:      {BATCH_SIZE}")
    print(f"GPUs:            {args.gpus}")
    print(f"Dry Run:         {args.dry_run}")
    print("=" * 80 + "\n", flush=True)

    t_global_start = time.perf_counter()
    results = {}

    for idx, ds in enumerate(active_datasets):
        ds_name = ds["name"]
        print("\n" + "#" * 80)
        print(f"### [{idx+1}/{len(active_datasets)}] Running Full Benchmark Pipeline: {ds_name} ###")
        print("#" * 80 + "\n", flush=True)

        cmd = [
            PYTHON, str(ROOT / "run_benchmark_pipeline.py"),
            "--dataset_root", ds["dataset_root"],
            "--outs_dir", ds["outs_dir"],
            "--splits_dir", ds["splits_dir"],
            "--train_sizes"] + TRAIN_SIZES + [
            "--image_sizes"] + IMAGE_SIZES + [
            "--max_iters"] + MAX_ITERS + [
            "--batch_size", BATCH_SIZE,
            "--gpus", args.gpus,
            "--seed", SEED
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            print(f"\n[ERROR] Pipeline failed for {ds_name} (exit code={proc.returncode}) after {elapsed/60:.2f} mins!")
            results[ds_name] = {"status": "FAILED", "elapsed_min": elapsed / 60.0}
            sys.exit(proc.returncode)
        else:
            print(f"\n[SUCCESS] Completed {ds_name} in {elapsed/60:.2f} mins!")
            results[ds_name] = {"status": "SUCCESS", "elapsed_min": elapsed / 60.0}

    total_elapsed = time.perf_counter() - t_global_start
    print("\n" + "=" * 80)
    print("=== All exp0908 Benchmark Pipelines Finished Successfully! ===")
    print(f"Total Time: {total_elapsed/60:.2f} minutes")
    for name, r in results.items():
        print(f"  - {name}: {r['status']} ({r['elapsed_min']:.2f} min)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
