#!/usr/bin/env python
"""Multi-GPU Full Benchmark Runner: Parallel Matrix Execution Across GPUs.

Distributes the 12 tasks (50/100/200/400 x 224/448/672) across specified GPUs
(e.g., --gpus 0,1,2,3 or --gpus auto) using multiprocessing.Queue with 0 DDP overhead.
Each GPU worker independently runs:
  1. Dinomaly2 Stage-1 Training
  2. PatchCore Training & Feature Subsampling
  3. Two-Stage Feature Bank Building
  4. Two-Stage Dual-Threshold Short-Circuit Inference
Upon all tasks completion, automatically invokes:
  5. Multi-metric report aggregation (generate_final_report_multisize.py & analyze_and_report.py)
  6. Comprehensive comparison chart plotting (plot_evaluation_charts.py)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = Path(r"D:\Software\anaconda3\envs\py312\python.exe")
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)
ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-GPU Full Benchmark Matrix Runner")
    parser.add_argument("--gpus", type=str, default="auto", help="List of GPU IDs (e.g. '0,1,2,3' or 'auto')")
    parser.add_argument("--outs_dir", type=str, default=r"F:\tmp\0826", help="Output directory for all benchmark artifacts")
    parser.add_argument("--splits_dir", type=str, default=r"F:\tmp\outs\data_splits", help="Directory containing train_N_seed2024.txt")
    parser.add_argument("--bank_data", type=str, default=r"F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据", help="Feature bank dataset directory")
    parser.add_argument("--max_iters", type=int, default=2000, help="Stage 1 training iterations")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[224, 448, 672])
    return parser


def _worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, outs_dir: Path, bank_data: Path, max_iters: int):
    """Worker process assigned to a dedicated GPU."""
    log_file = outs_dir / f"gpu_{gpu_id}.log"

    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}][GPU {gpu_id}] {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run_cmd(cmd, cwd=None):
        log(f"RUN: {' '.join(str(c) for c in cmd)}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - t0
        log(f"EXIT {proc.returncode} in {elapsed:.1f}s")
        if proc.returncode != 0:
            log(f"STDERR:\n{proc.stderr[-2000:]}")
        return proc.returncode == 0, elapsed

    log(f"Worker initialized on GPU {gpu_id}")

    while True:
        try:
            task = task_queue.get_nowait()
        except Exception:
            break

        n = task["n"]
        sz = task["sz"]
        bs = task["bs"]
        train_txt = task["train_txt"]
        test_txt = task["test_txt"]

        log(f"\n{'='*70}\nStarting Task: N={n} Size={sz} BatchSize={bs}\n{'='*70}")
        t_task_start = time.perf_counter()

        d_save = outs_dir / f"dinomaly2_n{n}_s{sz}_seed2024"
        p_save = outs_dir / f"patchcore_n{n}_s{sz}_seed2024"
        out_e2e = outs_dir / f"e2e_out_n{n}_s{sz}"

        # 1. Dinomaly2 Stage 1 Training
        d_models = list(d_save.rglob("model.pth")) if d_save.is_dir() else []
        if d_models:
            log(f"Skip Dinomaly2: already exists {d_models[0]}")
        else:
            cmd = [str(PYTHON), str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
                   "--data_path", str(train_txt),
                   "--dataset", "custom",
                   "--image_size", str(sz),
                   "--crop_size", str(sz),
                   "--batch-size", str(bs),
                   "--max-iters", str(max_iters),
                   "--save_dir", str(d_save),
                   "--cuda", str(gpu_id),
                   "--backbone", "dinov2reg_vit_base_14"]
            run_cmd(cmd, cwd=str(ROOT / "Dinomaly2"))

        # 2. PatchCore Training / Coreset Subsampling
        p_models = list(p_save.rglob("nnscorer_search_index.faiss")) if p_save.is_dir() else []
        if p_models:
            log(f"Skip PatchCore: already exists {p_models[0]}")
        else:
            cmd = [str(PYTHON), str(ROOT / "patchcore-inspection" / "train.py"),
                   "--data_path", str(train_txt),
                   "--dataset", "custom",
                   "--backbone", "wideresnet50",
                   "-imgsz", str(sz),
                   "-csz", str(sz),
                   "--batch_size", str(bs),
                   "--save_dir", str(p_save),
                   "--gpu", str(gpu_id)]
            run_cmd(cmd, cwd=str(ROOT / "patchcore-inspection"))

        # 3. Two-Stage Feature Bank Building
        d_models = list(d_save.rglob("model.pth")) if d_save.is_dir() else []
        if d_models:
            model_path = sorted(d_models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            bank_npz = d_save / "feature_bank.npz"
            if not bank_npz.is_file():
                cmd = [str(PYTHON), str(ROOT / "two_stage" / "build_bank.py"),
                       "--model", str(model_path),
                       "--data_dir", str(bank_data),
                       "--save_bank", str(bank_npz),
                       "--image_size", str(sz),
                       "--cuda", str(gpu_id)]
                run_cmd(cmd, cwd=str(ROOT))

            # 4. Two-Stage E2E Inference & Evaluation
            cmd = [str(PYTHON), str(ROOT / "run_e2e.py"),
                   "--dinomaly_model", str(model_path),
                   "--bank_data", str(bank_data),
                   "--test_list", str(test_txt),
                   "--output_dir", str(out_e2e),
                   "--cuda", str(gpu_id),
                   "--low", "0.019",
                   "--high", "0.024",
                   "--image_size", str(sz)]
            run_cmd(cmd, cwd=str(ROOT))

        elapsed = time.perf_counter() - t_task_start
        log(f"Finished Task: N={n} Size={sz} in {elapsed:.1f}s ({elapsed/60:.2f} min)")
        result_queue.put({"n": n, "sz": sz, "gpu": gpu_id, "elapsed": elapsed})


def main():
    args = build_parser().parse_args()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    outs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    bank_data = Path(args.bank_data).expanduser().resolve()

    import torch
    if args.gpus.lower() == "auto":
        gpu_count = torch.cuda.device_count()
        gpu_list = list(range(gpu_count)) if gpu_count > 0 else [0]
    else:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    print(f"=== Multi-GPU Full Benchmark Dispatcher ===")
    print(f"Active GPUs: {gpu_list}")
    print(f"Outputs Dir: {outs_dir}")
    print(f"Splits Dir: {splits_dir}")

    # Build task matrix
    batch_map = {224: 8, 448: 4, 672: 2}
    tasks = []
    for sz in args.image_sizes:
        bs = batch_map.get(sz, 4)
        for n in args.train_sizes:
            train_txt = splits_dir / f"train_{n}_seed2024.txt"
            test_txt = splits_dir / f"test_{n}_seed2024.txt"
            tasks.append({
                "n": n,
                "sz": sz,
                "bs": bs,
                "train_txt": train_txt,
                "test_txt": test_txt,
            })

    print(f"Total benchmark tasks queued: {len(tasks)}")

    # Setup multiprocessing queue
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task in tasks:
        task_queue.put(task)

    t0 = time.perf_counter()
    workers = []
    for gpu_id in gpu_list:
        p = ctx.Process(
            target=_worker_process,
            args=(gpu_id, task_queue, result_queue, outs_dir, bank_data, args.max_iters),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    total_time = time.perf_counter() - t0
    print(f"\n[Multi-GPU] All {len(tasks)} tasks finished in {total_time:.1f}s ({total_time/3600:.2f} hours)!")

    # 5. Speed Benchmark
    print("\n=== Running Speed Benchmark ===")
    subprocess.run([str(PYTHON), str(ROOT / "benchmark_speed.py"), "--gpus", ",".join(str(g) for g in gpu_list), "--outs_dir", str(outs_dir)], cwd=str(ROOT))

    # 6. Generate Markdown Reports
    print("\n=== Generating Comprehensive Markdown Reports ===")
    subprocess.run([str(PYTHON), str(ROOT / "generate_final_report_multisize.py"), "--outs_dir", str(outs_dir)], cwd=str(ROOT))
    subprocess.run([str(PYTHON), str(ROOT / "analyze_and_report.py"), "--outs_dir", str(outs_dir)], cwd=str(ROOT))

    # 7. Plot All Benchmark Charts (including Training Time and Inference Time)
    chart_dir = outs_dir / "charts"
    print(f"\n=== Plotting Full Benchmark Visual Charts into {chart_dir} ===")
    subprocess.run([str(PYTHON), str(ROOT / "plot_evaluation_charts.py"), "--chart_dir", str(chart_dir), "--full_benchmark"], cwd=str(ROOT))

    print(f"\n[ALL DONE] All benchmark reports, matrices, and charts are ready in -> {outs_dir}")


if __name__ == "__main__":
    main()
