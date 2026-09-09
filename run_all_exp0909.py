#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Master Benchmark Pipeline Runner for exp0909 Datasets.

Replicates the exact same datasets, configurations, and parameters from exp0908,
outputting to /data/wt/exp0909:
  - Datasets: shimo-all, 台达散热片栅格, 台达散热片正反面
  - Train Sample Sizes N: [100, 200, 400]
  - Image Sizes S: [224, 448, 672]
  - Iterations: [1000, 2000, 5000, 20000] (Strictly excluding 40000)
  - Batch Size: 8
  - GPUs: 8x RTX 4090 (auto)
  - Outs Dir: /data/wt/exp0909/{dataset_name}
  - Splits Dir: /data/wt/exp0909/{dataset_name}/data_splits
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
        "splits_dir": "/data/wt/exp0909/shimo-all/data_splits",
        "outs_dir": "/data/wt/exp0909/shimo-all",
    },
    {
        "name": "台达散热片栅格",
        "dataset_root": "/data/wt/data/台达散热片栅格",
        "splits_dir": "/data/wt/exp0909/台达散热片栅格/data_splits",
        "outs_dir": "/data/wt/exp0909/台达散热片栅格",
    },
    {
        "name": "台达散热片正反面",
        "dataset_root": "/data/wt/data/台达散热片正反面",
        "splits_dir": "/data/wt/exp0909/台达散热片正反面/data_splits",
        "outs_dir": "/data/wt/exp0909/台达散热片正反面",
    },
]

TRAIN_SIZES = ["100", "200", "400"]
IMAGE_SIZES = ["224", "448", "672"]
MAX_ITERS = ["1000", "2000", "5000", "20000"]
BATCH_SIZE = "8"
SEED = "2024"


def ensure_splits_and_masks(dataset_name: str, dataset_root: str, out_dir: Path, seed: str):
    """Ensure stratified proportional splits and GT masks are ready for exp0909."""
    import shutil
    splits_dir = out_dir / "data_splits"
    gt_dir = out_dir / "ground_truth"
    test_txt = splits_dir / "test_full.txt"
    if test_txt.is_file() and test_txt.stat().st_size > 0:
        print(f"[INFO] Data splits already exist for {dataset_name} at {splits_dir}")
        return

    src_exp0908 = Path("/data/wt/exp0908") / dataset_name
    src_splits = src_exp0908 / "data_splits"
    if (src_splits / "test_full.txt").is_file():
        print(f"[INFO] Copying valid data_splits from {src_splits} to {splits_dir}...")
        splits_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_splits, splits_dir, dirs_exist_ok=True)
        if (src_exp0908 / "ground_truth").is_dir() and not gt_dir.exists():
            gt_dir.symlink_to(src_exp0908 / "ground_truth")
        return

    print(f"\n[INFO] Preparing data splits and masks for {dataset_name} in {out_dir}...")
    prep_cmd = [
        PYTHON, str(ROOT / "prepare_splits_exp0908.py"),
        "--dataset", dataset_name,
        "--dataset_root", dataset_root,
        "--out_dir", str(out_dir),
        "--train_sizes"] + TRAIN_SIZES + [
        "--seed", seed
    ]
    proc = subprocess.run(prep_cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to prepare splits for {dataset_name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master Runner for exp0909 Industrial Datasets")
    parser.add_argument("--dataset", type=str, default="all", help="Target dataset ('all', 'shimo-all', '台达散热片栅格', '台达散热片正反面')")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Dry run verification")
    parser.add_argument("--gpus", type=str, default="auto", help="GPUs to use")
    parser.add_argument("--seed", type=str, default=SEED, help="Random seed")
    parser.add_argument("--wait_for_exp0908", action="store_true", help="Wait for exp0908 tasks to finish before starting")
    args = parser.parse_args()

    if args.wait_for_exp0908:
        print("[INFO] Checking if exp0908 master pipeline is still running...")
        my_pid = os.getpid()
        while True:
            out = subprocess.run(["pgrep", "-f", "run_all_exp0908.py"], stdout=subprocess.PIPE, text=True).stdout.strip()
            pids = [int(p) for p in out.split() if p.isdigit() and int(p) != my_pid]
            if not pids:
                print("[INFO] exp0908 has finished! Commencing exp0909 now.")
                break
            print(f"[WAIT] exp0908 still running (PIDs: {pids}). Sleeping 30s...", flush=True)
            time.sleep(30)

    active_datasets = [d for d in DATASETS if args.dataset == "all" or args.dataset == d["name"]]
    if not active_datasets:
        print(f"[ERROR] Unknown dataset: {args.dataset}. Available: {[d['name'] for d in DATASETS]}")
        sys.exit(1)

    print("=" * 80)
    print("=== Master Pipeline Orchestrator for exp0909 ===")
    print(f"Datasets to run ({len(active_datasets)}): {[d['name'] for d in active_datasets]}")
    print(f"Train Sizes N:   {TRAIN_SIZES}")
    print(f"Image Sizes S:   {IMAGE_SIZES}")
    print(f"Max Iters:       {MAX_ITERS} (excluding 40000)")
    print(f"Batch Size:      {BATCH_SIZE}")
    print(f"GPUs:            {args.gpus}")
    print(f"Seed:            {args.seed}")
    print(f"Dry Run:         {args.dry_run}")
    print("=" * 80 + "\n", flush=True)

    t_global_start = time.perf_counter()
    results = {}

    for idx, ds in enumerate(active_datasets):
        ds_name = ds["name"]
        outs_dir = Path(ds["outs_dir"])
        print("\n" + "#" * 80)
        print(f"### [{idx+1}/{len(active_datasets)}] Running Full Benchmark Pipeline: {ds_name} -> {outs_dir} ###")
        print("#" * 80 + "\n", flush=True)

        ensure_splits_and_masks(ds_name, ds["dataset_root"], outs_dir, args.seed)

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
            "--seed", args.seed
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
    print("=== All exp0909 Benchmark Pipelines Finished Successfully! ===")
    print(f"Total Time: {total_elapsed/60:.2f} minutes")
    for name, r in results.items():
        print(f"  - {name}: {r['status']} ({r['elapsed_min']:.2f} min)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
