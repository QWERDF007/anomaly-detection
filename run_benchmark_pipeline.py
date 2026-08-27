#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic Full Benchmark Pipeline Runner.

Automates the complete multi-size multi-sample benchmark workflow:
  1. Automated Data Splitting (sampling N normal images, gathering full test set)
  2. Dinomaly2 Multi-Scale Training
  3. PatchCore Multi-Scale Training & FAISS GPU Indexing
  4. Two-Stage Defect/Normal Feature Bank Construction
  5. Multi-Scale Full Test Set Evaluation (evaluate_benchmark.py)
  6. Real Hardware Performance Visualization (plot_evaluation_charts.py)
  7. Automated Markdown Benchmark Report Generation (generate_final_report_multisize.py)
"""
import os
import sys
import time
import json
import random
import subprocess
import argparse
from pathlib import Path

if sys.platform == "win32":
    py_dir = Path(sys.executable).parent
    for p in [py_dir, py_dir / "Library" / "bin", py_dir / "DLLs"]:
        if p.is_dir():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)

def run_cmd(cmd, cwd=None):
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n[RUN] {cmd_str}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[ERR] Failed in {elapsed:.1f}s, returncode={proc.returncode}")
        print(f"STDOUT:\n{proc.stdout[-2000:]}")
        print(f"STDERR:\n{proc.stderr[-2000:]}")
        return False, elapsed
    else:
        print(f"[OK] Completed in {elapsed:.1f}s")
        return True, elapsed

def discover_dataset_images(dataset_root: Path):
    """Auto-discovers normal (good/OK) and anomalous (NG/defect) image paths."""
    all_imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        all_imgs.extend(list(dataset_root.rglob(ext)))

    good_imgs = []
    bad_imgs = []
    for p in all_imgs:
        p_str = str(p).lower()
        if "建库数据" in str(p) or "feature_bank" in p_str:
            continue
        if "\\ok\\" in str(p) or "/ok/" in str(p) or "good" in p_str or "normal" in p_str:
            good_imgs.append(p)
        elif "\\ng\\" in str(p) or "/ng/" in str(p) or "defect" in p_str or "bad" in p_str or "anomaly" in p_str:
            bad_imgs.append(p)

    good_imgs = sorted(list(set(good_imgs)))
    bad_imgs = sorted(list(set(bad_imgs)))
    return good_imgs, bad_imgs

def main():
    parser = argparse.ArgumentParser(description="Generic Full Benchmark Pipeline Runner")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root folder of industrial inspection dataset")
    parser.add_argument("--bank_data", type=str, default="", help="Path to defect/normal ROI images for two-stage bank")
    parser.add_argument("--outs_dir", type=str, required=True, help="Output directory for experiments, models, charts, and report")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[50, 100, 200, 400], help="Sample sizes N to evaluate")
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[224, 448, 672], help="Resolution sizes to evaluate")
    parser.add_argument("--max_iters", type=int, default=2000, help="Training iterations for Dinomaly2")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    outs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = outs_dir / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    bank_data = Path(args.bank_data).expanduser().resolve() if args.bank_data else (dataset_root / "建库数据" if (dataset_root / "建库数据").is_dir() else None)

    print("=" * 80)
    print(f"=== Generic Anomaly Detection Benchmark Pipeline ===")
    print(f"Dataset Root: {dataset_root}")
    print(f"Bank Data: {bank_data}")
    print(f"Outs Dir: {outs_dir}")
    print(f"Train Sizes N: {args.train_sizes}")
    print(f"Image Sizes S: {args.image_sizes}")
    print(f"Max Iters: {args.max_iters}")
    print("=" * 80)

    # 1. Data Split Generation
    good_imgs, bad_imgs = discover_dataset_images(dataset_root)
    print(f"Discovered {len(good_imgs)} normal images and {len(bad_imgs)} anomalous images.")

    random.seed(args.seed)
    shuffled_good = list(good_imgs)
    random.shuffle(shuffled_good)

    valid_n_samples = [n for n in args.train_sizes if n <= len(shuffled_good)]
    if not valid_n_samples:
        raise ValueError(f"Not enough normal images ({len(good_imgs)}) for requested sample sizes {args.train_sizes}")

    for n in valid_n_samples:
        train_p = splits_dir / f"train_n{n}.txt"
        with open(train_p, "w", encoding="utf-8") as f:
            for p in shuffled_good[:n]:
                f.write(f"{p}\n")

    # Unified Full Test Set
    test_full_p = splits_dir / "test_full.txt"
    test_imgs = [(p, 0) for p in good_imgs] + [(p, 1) for p in bad_imgs]
    with open(test_full_p, "w", encoding="utf-8") as f:
        for p, lbl in test_imgs:
            f.write(f"{p}\t{lbl}\n")
    print(f"Created unified test split: {len(test_imgs)} total images -> {test_full_p}")

    # 2. Dinomaly2 Training
    print("\n" + "=" * 80)
    print("=== Step 1: Training Dinomaly2 Models ===")
    print("=" * 80)
    for s in args.image_sizes:
        for n in valid_n_samples:
            task_name = f"dinomaly2_n{n}_s{s}_seed{args.seed}"
            task_out = outs_dir / task_name
            train_txt = splits_dir / f"train_n{n}.txt"
            model_file = task_out / "model.pth"

            if model_file.exists() or any((task_out).glob("*/model.pth")):
                print(f"[SKIP] Dinomaly2 Task {task_name} already trained.")
                continue

            cmd = [
                str(PYTHON), str(ROOT / "Dinomaly2" / "train.py"),
                "-d", "custom",
                "--custom_train_data", str(train_txt),
                "--custom_test_data", str(test_full_p),
                "--img_size", str(s),
                "--max_iters", str(args.max_iters),
                "--output_dir", str(task_out),
                "--val_freq", "2000",
                "--batch_size", "2" if s >= 672 else "4",
            ]
            run_cmd(cmd)

    # 3. PatchCore Training
    print("\n" + "=" * 80)
    print("=== Step 2: Training PatchCore Baselines & GPU Indexing ===")
    print("=" * 80)
    for s in args.image_sizes:
        for n in valid_n_samples:
            task_name = f"patchcore_n{n}_s{s}_seed{args.seed}"
            task_out = outs_dir / task_name
            train_txt = splits_dir / f"train_n{n}.txt"

            if list(task_out.glob("*/*patchcore_params.pkl")):
                print(f"[SKIP] PatchCore Task {task_name} already trained.")
                continue

            cmd = [
                str(PYTHON), str(ROOT / "patchcore-inspection" / "bin" / "run_patchcore.py"),
                "--gpu", str(args.cuda),
                "--seed", str(args.seed),
                "--save_patchcore_model",
                "--log_group", task_name,
                "--log_project", str(outs_dir),
                "patch_core",
                "-b", "wideresnet50",
                "-le", "layer2",
                "-le", "layer3",
                "--faiss_on_gpu",
                "--sampler", "approx_greedy_coreset",
                "--percentage", "0.01",
                "dataset",
                "--resize", str(s),
                "--imagesize", str(s),
                "custom",
                "--train_data", str(train_txt),
                "--test_data", str(test_full_p),
            ]
            run_cmd(cmd)

    # 4. Two-Stage Feature Bank Construction
    print("\n" + "=" * 80)
    print("=== Step 3: Building Two-Stage Defect & Normal Feature Banks ===")
    print("=" * 80)
    if bank_data and bank_data.is_dir():
        for s in args.image_sizes:
            for n in valid_n_samples:
                task_out = outs_dir / f"dinomaly2_n{n}_s{s}_seed{args.seed}"
                bank_file = task_out / "feature_bank.npz"
                train_txt = splits_dir / f"train_n{n}.txt"

                if bank_file.exists():
                    print(f"[SKIP] Feature bank already built for N={n} Size={s}")
                    continue

                cmd = [
                    str(PYTHON), str(ROOT / "two_stage" / "build_bank.py"),
                    "--data_root", str(bank_data),
                    "--normal_list", str(train_txt),
                    "--output", str(bank_file),
                    "--image_size", str(s),
                    "--device", f"cuda:{args.cuda}",
                ]
                run_cmd(cmd)

    # 5. Full Evaluation
    print("\n" + "=" * 80)
    print("=== Step 4: Running Full Unified Evaluation ===")
    print("=" * 80)
    eval_cmd = [
        str(PYTHON), str(ROOT / "evaluate_benchmark.py"),
        "--outs_dir", str(outs_dir),
        "--test_list", str(test_full_p),
        "--train_sizes"] + [str(n) for n in valid_n_samples] + [
        "--image_sizes"] + [str(s) for s in args.image_sizes] + [
        "--cuda", str(args.cuda)
    ]
    run_cmd(eval_cmd)

    # 6. Plot Evaluation Charts
    print("\n" + "=" * 80)
    print("=== Step 5: Generating Benchmark Visualization Charts ===")
    print("=" * 80)
    plot_cmd = [
        str(PYTHON), str(ROOT / "plot_evaluation_charts.py"),
        "--outs_dir", str(outs_dir),
        "--chart_dir", str(outs_dir / "charts"),
        "--full_benchmark"
    ]
    run_cmd(plot_cmd)

    # 7. Generate Final Report
    print("\n" + "=" * 80)
    print("=== Step 6: Generating Comprehensive Markdown Benchmark Report ===")
    print("=" * 80)
    report_cmd = [
        str(PYTHON), str(ROOT / "generate_final_report_multisize.py"),
        "--outs_dir", str(outs_dir)
    ]
    run_cmd(report_cmd)

    print("\n" + "=" * 80)
    print(f"=== Benchmark Completed Successfully! ===")
    print(f"Report: {outs_dir / 'FINAL_BENCHMARK_REPORT.md'}")
    print(f"Charts: {outs_dir / 'charts'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
