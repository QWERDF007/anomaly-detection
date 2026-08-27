#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full benchmark runner for 透气膜 (Breathable Membrane) dataset.
Executes complete pipeline:
  1. Data splitting (train_sizes=[20, 50, 100, 150])
  2. Dinomaly2 training (2000 iters per task, 12 tasks)
  3. PatchCore baseline training & index construction
  4. Two-stage ROI patch feature bank extraction (FAISS GPU)
  5. Full unified test set evaluation (468 images: 194 OK + 274 NG)
  6. Performance plotting & Markdown report generation
"""
import os
import sys
import time
import json
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

def main():
    parser = argparse.ArgumentParser(description="Full Benchmark Pipeline for 透气膜")
    parser.add_argument("--dataset_root", type=str, default=r"F:\data\异常检测测试报告数据\透气膜")
    parser.add_argument("--bank_data", type=str, default=r"F:\data\异常检测测试报告数据\透气膜\建库数据")
    parser.add_argument("--outs_dir", type=str, default=r"F:\tmp\0827")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[20, 50, 100, 150])
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[224, 448, 672])
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    bank_data = Path(args.bank_data)
    outs_dir = Path(args.outs_dir)
    outs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = outs_dir / "data_splits"

    print("=" * 80)
    print(f"=== Starting Full Benchmark on 透气膜 Dataset ===")
    print(f"Dataset: {dataset_root}")
    print(f"Bank Data: {bank_data}")
    print(f"Outs Dir: {outs_dir}")
    print(f"Train Sizes: {args.train_sizes}")
    print(f"Image Sizes: {args.image_sizes}")
    print(f"Max Iters: {args.max_iters}")
    print("=" * 80)

    # 1. Data Split Generation
    print("\n>>> Step 1: Generating Data Splits...")
    cmd = [
        str(PYTHON), str(ROOT / "prepare_splits.py"),
        "--dataset_root", str(dataset_root),
        "--outs_dir", str(outs_dir),
        "--train_sizes"
    ] + [str(n) for n in args.train_sizes] + ["--seeds", "2024", "2025", "2026"]
    run_cmd(cmd, cwd=str(ROOT))

    # Unified Full Test List (194 OK + 274 NG = 468 images)
    ok_files = sorted(list((dataset_root / "OK").glob("*.*")))
    ng_files = sorted(list((dataset_root / "NG").glob("*.*")))
    all_test = sorted(ok_files + ng_files, key=lambda p: str(p).lower())
    test_full_path = splits_dir / "test_full.txt"
    test_full_path.write_text("\n".join(str(p) for p in all_test) + "\n", encoding="utf-8")
    print(f"[prepare] Saved full unified test list ({len(all_test)} images: {len(ok_files)} OK + {len(ng_files)} NG) -> {test_full_path}")

    # 2. Training Loop
    batch_map = {224: 8, 448: 4, 672: 2}
    total_configs = len(args.image_sizes) * len(args.train_sizes)
    curr_config = 0

    for sz in args.image_sizes:
        bs = batch_map.get(sz, 4)
        for n in args.train_sizes:
            curr_config += 1
            print(f"\n{'='*80}\n>>> Step 2 & 3: [{curr_config}/{total_configs}] Config: N={n} Size={sz} BatchSize={bs}\n{'='*80}")
            train_txt = splits_dir / f"train_{n}_seed2024.txt"
            d_save = outs_dir / f"dinomaly2_n{n}_s{sz}_seed2024"
            p_save = outs_dir / f"patchcore_n{n}_s{sz}_seed2024"

            # 2.1 Dinomaly2 Training
            d_models = list(d_save.rglob("model.pth")) if d_save.is_dir() else []
            if d_models:
                print(f"[skip] Dinomaly2 N={n} Size={sz} already trained: {d_models[0]}")
            else:
                cmd = [
                    str(PYTHON), str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
                    "--data_path", str(train_txt),
                    "--dataset", "custom",
                    "--image_size", str(sz),
                    "--crop_size", str(sz),
                    "--batch-size", str(bs),
                    "--max-iters", str(args.max_iters),
                    "--save_dir", str(d_save),
                    "--cuda", str(args.cuda),
                    "--backbone", "dinov2reg_vit_base_14"
                ]
                run_cmd(cmd, cwd=str(ROOT / "Dinomaly2"))

            # 2.2 PatchCore Training
            p_models = list(p_save.rglob("nnscorer_search_index.faiss")) if p_save.is_dir() else []
            if p_models:
                print(f"[skip] PatchCore N={n} Size={sz} already indexed: {p_models[0]}")
            else:
                cmd = [
                    str(PYTHON), str(ROOT / "patchcore-inspection" / "train.py"),
                    "--data_path", str(train_txt),
                    "--dataset", "custom",
                    "--backbone", "wideresnet50",
                    "-imgsz", str(sz),
                    "-csz", str(sz),
                    "--batch_size", str(bs),
                    "--save_dir", str(p_save),
                    "--gpu", str(args.cuda)
                ]
                run_cmd(cmd, cwd=str(ROOT / "patchcore-inspection"))

            # 2.3 Two-Stage Feature Bank Building
            d_models = list(d_save.rglob("model.pth")) if d_save.is_dir() else []
            if d_models:
                model_pth = sorted(d_models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                bank_npz = d_save / "feature_bank.npz"
                if bank_npz.is_file():
                    print(f"[skip] Feature bank already exists: {bank_npz}")
                else:
                    cmd = [
                        str(PYTHON), str(ROOT / "two_stage" / "build_bank.py"),
                        "--model", str(model_pth),
                        "--data_dir", str(bank_data),
                        "--save_bank", str(bank_npz),
                        "--save_dir", str(d_save),
                        "--image_size", str(sz),
                        "--cuda", str(args.cuda)
                    ]
                    run_cmd(cmd, cwd=str(ROOT))

    # 3. Full Unified Evaluation (eval_all_models_full)
    print("\n>>> Step 4: Evaluating All Models on Full 468 Unified Test Set...")
    cmd = [
        str(PYTHON), str(ROOT / "eval_all_models_full_1733.py"),
        "--outs_dir", str(outs_dir),
        "--test_list", str(test_full_path),
        "--bank_data", str(bank_data),
        "--cuda", str(args.cuda)
    ]
    run_cmd(cmd, cwd=str(ROOT))

    # 4. Generate Visual Charts & Markdown Benchmark Report
    print("\n>>> Step 5: Generating Publication-Grade Evaluation Charts...")
    cmd = [
        str(PYTHON), str(ROOT / "plot_evaluation_charts.py"),
        "--outs_dir", str(outs_dir),
        "--chart_dir", str(outs_dir / "charts"),
        "--full_benchmark"
    ]
    run_cmd(cmd, cwd=str(ROOT))

    print("\n>>> Step 6: Compiling Final Comprehensive Benchmark Markdown Report...")
    cmd = [
        str(PYTHON), str(ROOT / "generate_final_report_multisize.py"),
        "--outs_dir", str(outs_dir)
    ]
    run_cmd(cmd, cwd=str(ROOT))

    print("\n" + "=" * 80)
    print(f"🎉 All benchmark steps for 透气膜 completed successfully!")
    print(f"Report -> {outs_dir / 'FINAL_BENCHMARK_REPORT.md'}")
    print(f"Charts -> {outs_dir / 'charts'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
