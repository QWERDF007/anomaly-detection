#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic Multi-GPU Full Benchmark Pipeline Runner.

Automates the complete multi-size multi-sample benchmark workflow:
  1. Automated Data Splitting (sampling N normal images, gathering full test set)
  2. Multi-GPU Parallel Task Dispatching:
     - Step 1: Dinomaly2 Multi-Scale Training
     - Step 2: PatchCore Multi-Scale Training & FAISS GPU Indexing
     - Step 3: Two-Stage Defect/Normal Feature Bank Construction
  3. Multi-Scale Full Test Set Evaluation (evaluate_benchmark.py)
  4. Real Hardware Performance Visualization (plot_evaluation_charts.py)
  5. Automated Markdown Benchmark Report Generation (generate_final_report_multisize.py)
"""
import os
import sys
import time
import json
import random
import subprocess
import argparse
import multiprocessing as mp
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

def run_cmd(cmd, cwd=None, env=None):
    if env is None:
        env = os.environ.copy()
    pythonpaths = [str(ROOT), str(ROOT / "Dinomaly2"), str(ROOT / "patchcore-inspection" / "src")]
    curr_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths + ([curr_pp] if curr_pp else []))
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd_str = " ".join(str(c) for c in cmd)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[ERR] Failed in {elapsed:.1f}s, cmd: {cmd_str[:120]}...", flush=True)
        print(f"STDERR:\n{proc.stderr[-2000:]}", flush=True)
        return False, elapsed
    else:
        return True, elapsed

def discover_dataset_images(dataset_root: Path):
    """Auto-discovers normal (good/OK) and anomalous (NG/defect) image paths."""
    all_imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        all_imgs.extend(list(dataset_root.rglob(ext)))

    good_imgs = []
    bad_imgs = []
    for p in all_imgs:
        parts_lower = [part.lower() for part in p.parts]
        if "建库数据" in p.parts or "feature_bank" in parts_lower:
            continue
        if any(k in parts_lower for k in ["ok", "good", "normal", "良品", "正常"]):
            good_imgs.append(p)
        elif any(k in parts_lower for k in ["ng", "bad", "defect", "anomaly", "不良品", "瑕疵", "缺陷"]):
            bad_imgs.append(p)

    good_imgs = sorted(list(set(good_imgs)))
    bad_imgs = sorted(list(set(bad_imgs)))
    return good_imgs, bad_imgs

def _worker_gpu(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """Dedicated worker process for a specific GPU."""
    while True:
        try:
            task = task_queue.get(timeout=2)
        except Exception:
            break
        if task is None:
            break

        task_type, cmd, desc = task
        print(f"[GPU {gpu_id}] Starting: {desc}...", flush=True)
        env = os.environ.copy()
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        ok, elapsed = run_cmd(cmd, env=env)
        status = "OK" if ok else "FAILED"
        print(f"[GPU {gpu_id}] Finished [{status}] in {elapsed:.1f}s: {desc}", flush=True)
        result_queue.put((task_type, desc, ok, elapsed))

def run_task_batch_on_gpus(task_list: list, gpu_list: list[int], phase_name: str):
    if not task_list:
        print(f"[INFO] {phase_name}: All tasks already completed, skipping.")
        return

    print(f"\n=== {phase_name} ({len(task_list)} tasks across {len(gpu_list)} GPUs) ===", flush=True)
    if len(gpu_list) > 1:
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        for t in task_list:
            task_queue.put(t)

        processes = []
        for gid in gpu_list:
            p = mp.Process(target=_worker_gpu, args=(gid, task_queue, result_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        single_gid = gpu_list[0]
        for task_type, cmd, desc in task_list:
            print(f"[GPU {single_gid}] Starting: {desc}...", flush=True)
            env = os.environ.copy()
            if single_gid >= 0:
                env["CUDA_VISIBLE_DEVICES"] = str(single_gid)
            ok, elapsed = run_cmd(cmd, env=env)
            status = "OK" if ok else "FAILED"
            print(f"[GPU {single_gid}] Finished [{status}] in {elapsed:.1f}s: {desc}", flush=True)

def parse_gpu_list(gpus_arg: str) -> list[int]:
    import torch
    num_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpus_arg.strip().lower() == "auto":
        return list(range(num_cuda)) if num_cuda > 0 else [-1]
    gpus = []
    for part in gpus_arg.split(","):
        part = part.strip()
        if part.isdigit():
            gpus.append(int(part))
    return gpus if gpus else [0]

def main():
    parser = argparse.ArgumentParser(description="Generic Multi-GPU Full Benchmark Pipeline Runner")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root folder of industrial inspection dataset")
    parser.add_argument("--bank_data", type=str, default="", help="Path to defect/normal ROI images for two-stage bank")
    parser.add_argument("--outs_dir", type=str, required=True, help="Output directory for experiments, models, charts, and report")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[50, 100, 200, 400], help="Sample sizes N to evaluate")
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[224, 448, 672], help="Resolution sizes to evaluate")
    parser.add_argument("--max_iters", type=int, default=2000, help="Training iterations for Dinomaly2")
    parser.add_argument("--gpus", type=str, default="auto", help="GPU device IDs (e.g. '0', '0,1', '0,1,2,3', or 'auto')")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    outs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = outs_dir / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Automatic Tee logger to outs_dir/pipeline_execution.log
    log_file_path = outs_dir / "pipeline_execution.log"
    class TeeLogger:
        def __init__(self, original_stdout, log_file):
            self.terminal = original_stdout
            self.log = log_file
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    log_file = open(log_file_path, "a", encoding="utf-8")
    sys.stdout = TeeLogger(sys.stdout, log_file)
    sys.stderr = TeeLogger(sys.stderr, log_file)

    gpu_list = parse_gpu_list(args.gpus)
    bank_data = Path(args.bank_data).expanduser().resolve() if args.bank_data else (dataset_root / "建库数据" if (dataset_root / "建库数据").is_dir() else None)

    print("=" * 80)
    print(f"=== Multi-GPU Anomaly Detection Benchmark Pipeline ===")
    print(f"Dataset Root: {dataset_root}")
    print(f"Bank Data: {bank_data}")
    print(f"Outs Dir: {outs_dir}")
    print(f"Train Sizes N: {args.train_sizes}")
    print(f"Image Sizes S: {args.image_sizes}")
    print(f"GPUs Allocated: {gpu_list} (Total: {len(gpu_list)} GPU workers)")
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

    # Step 1: Dinomaly2 Training Tasks
    dino_tasks = []
    for s in args.image_sizes:
        for n in valid_n_samples:
            task_name = f"dinomaly2_n{n}_s{s}_seed{args.seed}"
            task_out = outs_dir / task_name
            train_txt = splits_dir / f"train_n{n}.txt"
            model_file = task_out / "model.pth"

            if not model_file.exists() and not any(task_out.glob("*/model.pth")):
                cmd = [
                    str(PYTHON), str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
                    "--dataset", "custom",
                    "--data_path", str(train_txt),
                    "--test_path", str(test_full_p),
                    "--image_size", str(s),
                    "--crop_size", str(s),
                    "--batch_size", "2" if s >= 672 else "4",
                    "--max_iters", str(args.max_iters),
                    "--save_dir", str(task_out),
                    "--eval_interval", "-1",
                    "--cuda", "0",
                ]
                dino_tasks.append(("dinomaly2", cmd, f"Dinomaly2 N={n} Size={s}"))
    run_task_batch_on_gpus(dino_tasks, gpu_list, "Step 1: Training Dinomaly2 Models")

    # Step 2: PatchCore Training Tasks
    patch_tasks = []
    for s in args.image_sizes:
        for n in valid_n_samples:
            task_name = f"patchcore_n{n}_s{s}_seed{args.seed}"
            task_out = outs_dir / task_name
            train_txt = splits_dir / f"train_n{n}.txt"

            if not list(task_out.glob("*/*patchcore_params.pkl")) and not (task_out / "patchcore_params.pkl").exists() and not (task_out / "models" / "patchcore_params.pkl").exists():
                cmd = [
                    str(PYTHON), str(ROOT / "patchcore-inspection" / "train.py"),
                    "--data_path", str(train_txt),
                    "--dataset", "custom",
                    "--backbone", "wideresnet50",
                    "--layers", "layer2",
                    "--layers", "layer3",
                    "--sampler", "approx_greedy_coreset",
                    "--sampling_percentage", "0.1",
                    "--resize", str(s),
                    "--image_size", str(s),
                    "--save_dir", str(task_out),
                    "--gpu", "0",
                    "--seed", str(args.seed),
                    "--skip_eval",
                ]
                patch_tasks.append(("patchcore", cmd, f"PatchCore N={n} Size={s}"))
    run_task_batch_on_gpus(patch_tasks, gpu_list, "Step 2: Training PatchCore Baselines & GPU Indexing")

    # Step 3: Two-Stage Feature Bank Tasks
    bank_tasks = []
    if bank_data and bank_data.is_dir():
        for s in args.image_sizes:
            for n in valid_n_samples:
                task_out = outs_dir / f"dinomaly2_n{n}_s{s}_seed{args.seed}"
                model_file = task_out / "model.pth"
                bank_file = task_out / "feature_bank.npz"
                d_cands = sorted(list(task_out.glob("*/model.pth")) + list(task_out.glob("model.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
                m_p = d_cands[0] if d_cands else model_file

                if not bank_file.exists() and d_cands:
                    cmd = [
                        str(PYTHON), str(ROOT / "two_stage" / "build_bank.py"),
                        "--model", str(m_p),
                        "--data_dir", str(bank_data),
                        "--save_bank", str(bank_file),
                        "--image_size", str(s),
                        "--cuda", "0",
                    ]
                    bank_tasks.append(("feature_bank", cmd, f"FeatureBank N={n} Size={s}"))
    run_task_batch_on_gpus(bank_tasks, gpu_list, "Step 3: Building Two-Stage Defect & Normal Feature Banks")

    # Step 4: Full Evaluation
    print("\n" + "=" * 80)
    print("=== Step 4: Running Full Unified Evaluation ===")
    print("=" * 80)
    primary_gpu = gpu_list[0] if gpu_list[0] >= 0 else 0
    eval_cmd = [
        str(PYTHON), str(ROOT / "evaluate_benchmark.py"),
        "--outs_dir", str(outs_dir),
        "--test_list", str(test_full_p),
        "--train_sizes"] + [str(n) for n in valid_n_samples] + [
        "--image_sizes"] + [str(s) for s in args.image_sizes] + [
        "--cuda", str(primary_gpu)
    ]
    env_eval = os.environ.copy()
    if primary_gpu >= 0:
        env_eval["CUDA_VISIBLE_DEVICES"] = str(primary_gpu)
    run_cmd(eval_cmd, env=env_eval)

    # Step 5: Plot Evaluation Charts
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

    # Step 6: Generate Final Report
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
    mp.freeze_support()
    main()
