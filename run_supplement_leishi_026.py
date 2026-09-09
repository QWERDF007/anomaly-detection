#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supplementation Runner for leishi_026 Dataset.

Runs the remaining training and unified evaluation tasks for leishi_026:
  - Dataset Root: /data/wt/ramdisk/leishi_026
  - Outs Dir:     /data/wt/exp0906/leishi_026
  - Splits Dir:   /data/wt/exp0906/leishi_026/data_splits
  - Train Sizes:  [20, 50, 100, 150]
  - Image Sizes:  [224, 448, 672]
  - Max Iters:    [1000, 2000, 5000, 20000] (Strictly excluding 40000)
  - Batch Size:   8
  - Seed:         2024
  - GPUs:         8x RTX 4090 (auto)

Can wait for both exp0908 and exp0909 to completely finish before starting.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path("/data/wt/anomaly-detection")
ANOMALY_PYTHON = Path("/home/dell/miniconda3/envs/anomaly/bin/python")
PYTHON = str(ANOMALY_PYTHON) if ANOMALY_PYTHON.is_file() else sys.executable

LEISHI_CONFIG = {
    "name": "leishi_026",
    "dataset_root": "/data/wt/ramdisk/leishi_026",
    "splits_dir": "/data/wt/exp0906/leishi_026/data_splits",
    "outs_dir": "/data/wt/exp0906/leishi_026",
    "train_sizes": ["20", "50", "100", "150"],
    "image_sizes": ["224", "448", "672"],
    "max_iters": ["1000", "2000", "5000", "20000"],
    "batch_size": "8",
    "seed": "2024",
}


def wait_for_prior_experiments():
    """Wait for all running exp0908 and exp0909 processes to finish."""
    print("=" * 80)
    print("[DAEMON] Waiting for exp0908 and exp0909 pipelines to complete...")
    print("=" * 80, flush=True)

    my_pid = os.getpid()
    while True:
        out = subprocess.run(
            ["pgrep", "-f", "run_all_exp0908.py|run_all_exp0909.py|run_benchmark_pipeline.py.*exp0908|run_benchmark_pipeline.py.*exp0909"],
            stdout=subprocess.PIPE, text=True
        ).stdout.strip()
        pids = [int(p) for p in out.split() if p.isdigit() and int(p) != my_pid]

        if not pids:
            print("[DAEMON] Both exp0908 and exp0909 have completed!", flush=True)
            time.sleep(10)
            break

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Active prior pipelines running (PIDs: {pids}). Sleeping 30s...", flush=True)
        time.sleep(30)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Supplementation Runner for leishi_026")
    parser.add_argument("--wait_for_all", "--wait", action="store_true", help="Wait for exp0908 and exp0909 to complete first")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Dry run verification")
    parser.add_argument("--gpus", type=str, default="auto", help="GPUs to use (default: auto)")
    args = parser.parse_args()

    if args.wait_for_all:
        wait_for_prior_experiments()

    cfg = LEISHI_CONFIG
    outs_dir = Path(cfg["outs_dir"])
    outs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("=== Supplementing leishi_026 Benchmark Pipeline ===")
    print(f"Dataset Root:    {cfg['dataset_root']}")
    print(f"Outs Dir:        {cfg['outs_dir']}")
    print(f"Splits Dir:      {cfg['splits_dir']}")
    print(f"Train Sizes N:   {cfg['train_sizes']}")
    print(f"Image Sizes S:   {cfg['image_sizes']}")
    print(f"Max Iters:       {cfg['max_iters']} (excluding 40000)")
    print(f"Batch Size:      {cfg['batch_size']}")
    print(f"Seed:            {cfg['seed']}")
    print(f"GPUs:            {args.gpus}")
    print(f"Dry Run:         {args.dry_run}")
    print("=" * 80 + "\n", flush=True)

    cmd = [
        PYTHON, str(ROOT / "run_benchmark_pipeline.py"),
        "--dataset_root", cfg["dataset_root"],
        "--outs_dir", cfg["outs_dir"],
        "--splits_dir", cfg["splits_dir"],
        "--train_sizes"] + cfg["train_sizes"] + [
        "--image_sizes"] + cfg["image_sizes"] + [
        "--max_iters"] + cfg["max_iters"] + [
        "--batch_size", cfg["batch_size"],
        "--gpus", args.gpus,
        "--seed", cfg["seed"]
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"\n[ERROR] leishi_026 supplementation pipeline failed with exit code {proc.returncode} after {elapsed/60:.2f} min!")
        sys.exit(proc.returncode)
    else:
        print(f"\n[SUCCESS] leishi_026 supplementation pipeline completed successfully in {elapsed/60:.2f} min!")


if __name__ == "__main__":
    main()
