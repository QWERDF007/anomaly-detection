#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic Multi-GPU Full Benchmark Pipeline Runner.

Automates the complete multi-size multi-sample benchmark workflow:
  1. Automated Data Splitting (sampling N normal images, gathering full test set & clean in-domain set)
  2. Step 1: PatchCore Baseline Training & FAISS GPU Indexing (12 tasks across 8 GPUs, ~2 mins)
  3. Step 2: Dinomaly2 Training with LPT (Longest Processing Time First) scheduling across 8 GPUs
  4. Step 3: Two-Stage Defect/Normal Feature Bank Construction
  5. Step 4: Multi-GPU Parallel Unified Benchmark Evaluation (evaluate_benchmark.py across all GPUs)
  6. Step 5: High-Fidelity Performance Chart Generation (plot_evaluation_charts.py)
  7. Step 6: Automated Markdown Benchmark Report Generation (generate_final_report_multisize.py)

STRICT DATA INTEGRITY:
  - NO synthetic or hardcoded metrics allowed.
  - All metrics, times, latency, throughput, and VRAM are collected from live execution logs and hardware profilers.
"""
from __future__ import annotations

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


def run_cmd(cmd: list[str], cwd: Path | str | None = None, env: dict | None = None) -> tuple[bool, float, str, str]:
    """Runs command, returning (success, elapsed_sec, stdout, stderr)."""
    if env is None:
        env = os.environ.copy()
    pythonpaths = [str(ROOT), str(ROOT / "Dinomaly2"), str(ROOT / "patchcore-inspection" / "src")]
    curr_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(pythonpaths + ([curr_pp] if curr_pp else []))
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd_str = " ".join(str(c) for c in cmd)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[ERR] Failed in {elapsed:.1f}s, cmd: {cmd_str[:160]}...", flush=True)
        print(f"STDERR:\n{proc.stderr[-2000:]}", flush=True)
        return False, elapsed, proc.stdout, proc.stderr
    else:
        return True, elapsed, proc.stdout, proc.stderr


def discover_dataset_images(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    """Auto-discovers normal (good/OK) and anomalous (NG/defect) image paths."""
    all_imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        all_imgs.extend(list(dataset_root.rglob(ext)))

    good_imgs = []
    bad_imgs = []
    for p in all_imgs:
        parts_lower = [part.lower() for part in p.parts]
        if any(skip in parts_lower for skip in ["建库数据", "feature_bank", "ground_truth", "masks", "mask", "gt"]):
            continue
        if any(k in parts_lower for k in ["ok", "good", "normal", "良品", "正常"]):
            good_imgs.append(p)
        elif any(k in parts_lower for k in ["ng", "bad", "defect", "anomaly", "不良品", "瑕疵", "缺陷"]):
            bad_imgs.append(p)

    good_imgs = sorted(list(set(good_imgs)))
    bad_imgs = sorted(list(set(bad_imgs)))
    return good_imgs, bad_imgs


def _worker_gpu(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    while True:
        try:
            task = task_queue.get(timeout=2)
        except Exception:
            break
        if task is None:
            break

        task_type, cmd, desc, task_out_str, meta_dict = task
        print(f"[GPU {gpu_id}] Starting: {desc}...", flush=True)
        env = os.environ.copy()
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        t_start = time.time()
        ok, elapsed, stdout, stderr = run_cmd(cmd, env=env)
        t_end = time.time()
        status = "OK" if ok else "FAILED"
        print(f"[GPU {gpu_id}] Finished [{status}] in {elapsed:.1f}s ({elapsed/60.0:.2f}m): {desc}", flush=True)

        peak_mem = 0.0
        if task_out_str:
            out_p = Path(task_out_str)
            out_p.mkdir(parents=True, exist_ok=True)
            # Find metrics.json saved by training script
            m_files = sorted(list(out_p.rglob("metrics.json")), key=lambda p: p.stat().st_mtime, reverse=True)
            if m_files:
                try:
                    m_data = json.loads(m_files[0].read_text(encoding="utf-8"))
                    peak_mem = float(m_data.get("peak_gpu_mem_mb", 0.0))
                except Exception:
                    pass

            task_metrics = {
                "task_type": task_type,
                "desc": desc,
                "gpu_id": gpu_id,
                "start_time": t_start,
                "end_time": t_end,
                "elapsed_sec": round(elapsed, 2),
                "elapsed_min": round(elapsed / 60.0, 2),
                "peak_gpu_mem_mb": round(peak_mem, 2),
                "status": status,
                **(meta_dict or {})
            }
            try:
                metrics_file = "bank_task_metrics.json" if task_type == "feature_bank" else "task_metrics.json"
                with open(out_p / metrics_file, "w", encoding="utf-8") as f:
                    json.dump(task_metrics, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[warn] Could not save task metrics to {out_p}: {e}", flush=True)

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
        for task_type, cmd, desc, task_out_str, meta_dict in task_list:
            print(f"[GPU {single_gid}] Starting: {desc}...", flush=True)
            env = os.environ.copy()
            if single_gid >= 0:
                env["CUDA_VISIBLE_DEVICES"] = str(single_gid)
            t_start = time.time()
            ok, elapsed, stdout, stderr = run_cmd(cmd, env=env)
            t_end = time.time()
            status = "OK" if ok else "FAILED"
            print(f"[GPU {single_gid}] Finished [{status}] in {elapsed:.1f}s ({elapsed/60.0:.2f}m): {desc}", flush=True)

            if task_out_str:
                out_p = Path(task_out_str)
                out_p.mkdir(parents=True, exist_ok=True)
                peak_mem = 0.0
                m_files = sorted(list(out_p.rglob("metrics.json")), key=lambda p: p.stat().st_mtime, reverse=True)
                if m_files:
                    try:
                        m_data = json.loads(m_files[0].read_text(encoding="utf-8"))
                        peak_mem = float(m_data.get("peak_gpu_mem_mb", 0.0))
                    except Exception:
                        pass

                task_metrics = {
                    "task_type": task_type,
                    "desc": desc,
                    "gpu_id": single_gid,
                    "start_time": t_start,
                    "end_time": t_end,
                    "elapsed_sec": round(elapsed, 2),
                    "elapsed_min": round(elapsed / 60.0, 2),
                    "peak_gpu_mem_mb": round(peak_mem, 2),
                    "status": status,
                    **(meta_dict or {})
                }
                try:
                    metrics_file = "bank_task_metrics.json" if task_type == "feature_bank" else "task_metrics.json"
                    with open(out_p / metrics_file, "w", encoding="utf-8") as f:
                        json.dump(task_metrics, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[warn] Could not save task metrics to {out_p}: {e}", flush=True)


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
    parser.add_argument("--dataset_root", type=str, default="/data/wt/ramdisk/透气膜/透气膜", help="Root folder of industrial inspection dataset")
    parser.add_argument("--bank_data", type=str, default="", help="Path to defect/normal ROI images for two-stage bank")
    parser.add_argument("--outs_dir", type=str, default="", help="Output directory for experiments, models, charts, and report")
    parser.add_argument("--batch_size", type=int, default=8, help="Fixed batch size for training (default: 8)")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[20, 50, 100, 150], help="Sample sizes N to evaluate")
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[224, 448, 672], help="Resolution sizes to evaluate")
    parser.add_argument("--backbone", type=str, default="dinov2reg_vit_base_14", help="Backbone encoder model for Dinomaly2")
    parser.add_argument("--max_iters", type=int, nargs="+", default=[1000, 2000, 5000, 20000, 40000], help="Training iterations for Dinomaly2")
    parser.add_argument("--source-dg", "--source_dg", action="store_true", dest="source_dg", help="Enable Source-DG normal-shift generalization for Dinomaly2")
    parser.add_argument("--pseudo-anomaly", "--pseudo_anomaly", action="store_true", dest="pseudo_anomaly", help="Enable pseudo-anomaly preservation training for Source-DG")
    parser.add_argument("--gpus", type=str, default="auto", help="GPU device IDs (e.g. '0', '0,1', '0,1,2,3', or 'auto')")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", dest="dry_run", help="Print planned tasks without executing")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    
    # Auto-detect outs_dir if omitted
    if not args.outs_dir:
        ds_name = dataset_root.name
        if ds_name in ["透气膜", "铜色异常检测4相机", "铜色异常检测6相机", "leishi_026"]:
            outs_dir = Path(f"/data/wt/exp0906/{ds_name}").resolve()
        elif dataset_root.parent.name in ["透气膜", "铜色异常检测4相机", "铜色异常检测6相机"]:
            outs_dir = Path(f"/data/wt/exp0906/{dataset_root.parent.name}").resolve()
        else:
            outs_dir = Path(f"/data/wt/exp0906/{ds_name}").resolve()
    else:
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
    bank_data = Path(args.bank_data).expanduser().resolve() if args.bank_data else (
        (dataset_root / "建库数据") if (dataset_root / "建库数据").is_dir() else (
            (dataset_root.parent / "建库数据") if (dataset_root.parent / "建库数据").is_dir() else None
        )
    )

    print("=" * 80)
    print("=== Multi-GPU Anomaly Detection Benchmark Pipeline ===")
    print(f"Dataset Root:    {dataset_root}")
    print(f"Bank Data:       {bank_data}")
    print(f"Outs Dir:        {outs_dir}")
    print(f"Backbone:        {args.backbone}")
    print(f"Train Sizes N:   {args.train_sizes}")
    print(f"Image Sizes S:   {args.image_sizes}")
    print(f"Fixed Batch:     {args.batch_size}")
    print(f"Max Iters:       {args.max_iters}")
    print(f"GPUs Allocated:  {gpu_list} (Total: {len(gpu_list)} GPU workers)")
    print(f"Seed:            {args.seed}")
    print("=" * 80, flush=True)

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

    # Unified Full Test Set (all good + all bad)
    test_full_p = splits_dir / "test_full.txt"
    test_imgs = [(p, 0) for p in good_imgs] + [(p, 1) for p in bad_imgs]
    with open(test_full_p, "w", encoding="utf-8") as f:
        for p, lbl in test_imgs:
            f.write(f"{p}\t{lbl}\n")
    print(f"Created unified test split: {len(test_imgs)} total images -> {test_full_p}")

    # Clean In-Domain Test Set (Normal images used in training, for verifying clean domain false positive rate)
    max_train_n = max(valid_n_samples)
    clean_train_imgs = shuffled_good[:max_train_n]
    clean_indomain_p = splits_dir / "test_clean_in_domain.txt"
    with open(clean_indomain_p, "w", encoding="utf-8") as f:
        for p in clean_train_imgs:
            f.write(f"{p}\t0\n")
    print(f"Created clean in-domain split: {len(clean_train_imgs)} normal training images -> {clean_indomain_p}")

    # Step 1: PatchCore Training Tasks (Fast: ~2 mins across 8 GPUs)
    patch_tasks = []
    for s in args.image_sizes:
        for n in valid_n_samples:
            task_name = f"patchcore_n{n}_s{s}_seed{args.seed}"
            task_out = outs_dir / task_name
            train_txt = splits_dir / f"train_n{n}.txt"

            if not list(task_out.glob("**/patchcore_params.pkl")):
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
                    "--batch_size", str(args.batch_size),
                    "--save_dir", str(task_out),
                    "--gpu", "0",
                    "--seed", str(args.seed),
                    "--skip_eval",
                ]
                desc = f"PatchCore N={n} Size={s} (BS={args.batch_size})"
                meta = {"n": n, "size": s, "batch_size": args.batch_size, "model": "patchcore"}
                patch_tasks.append(("patchcore", cmd, desc, str(task_out), meta))

    # Step 2: Dinomaly2 Training Tasks (LPT Scheduling: heaviest tasks first)
    dino_tasks = []
    for iters in args.max_iters:
        for s in args.image_sizes:
            dino_s = (s // 14) * 14 if s % 14 != 0 else s
            for n in valid_n_samples:
                task_name = f"dinomaly2_n{n}_s{s}_iter{iters}_seed{args.seed}"
                task_out = outs_dir / task_name
                train_txt = splits_dir / f"train_n{n}.txt"
                model_file = task_out / "model.pth"

                if not model_file.exists() and not any(task_out.glob("*/model.pth")):
                    cmd = [
                        str(PYTHON), str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
                        "--dataset", "custom",
                        "--data_path", str(train_txt),
                        "--test_path", str(test_full_p),
                        "--backbone", str(args.backbone),
                        "--image_size", str(dino_s),
                        "--crop_size", str(dino_s),
                        "--batch_size", str(args.batch_size),
                        "--max_iters", str(iters),
                        "--save_dir", str(task_out),
                        "--eval_interval", "-1",
                        "--cuda", "0",
                    ]
                    if args.source_dg:
                        cmd.append("--source-dg")
                        if args.pseudo_anomaly:
                            cmd.append("--pseudo-anomaly")
                    desc = f"Dinomaly2 N={n} Size={s} Iter={iters} (BS={args.batch_size}){' (Source-DG)' if args.source_dg else ''}"
                    meta = {"n": n, "size": s, "iters": iters, "batch_size": args.batch_size, "model": "dinomaly2"}
                    dino_tasks.append(("dinomaly2", cmd, desc, str(task_out), meta))

    # Apply LPT (Longest Processing Time First) heuristic to prevent trailing GPU starvation
    def task_weight(t):
        meta = t[4]
        return meta.get("iters", 0) * ((meta.get("size", 224) / 224.0) ** 2) * (meta.get("n", 1) ** 0.5)

    dino_tasks.sort(key=task_weight, reverse=True)

    if args.dry_run:
        print("\n" + "=" * 80)
        print("=== DRY RUN MODE: Tasks Plan Summary ===")
        print(f"Total PatchCore Tasks: {len(patch_tasks)}")
        print(f"Total Dinomaly2 Tasks: {len(dino_tasks)}")
        for _, _, desc, out, _ in dino_tasks[:6]:
            print(f"  - {desc} -> {Path(out).name}")
        if len(dino_tasks) > 6:
            print(f"  ... and {len(dino_tasks) - 6} more")
        print(f"Total GPUs: {len(gpu_list)} ({gpu_list})")
        print("=" * 80)
        return

    # Execute Step 1: PatchCore (non-blocking baselines, fast execution)
    run_task_batch_on_gpus(patch_tasks, gpu_list, "Step 1: Training PatchCore Baselines & GPU Indexing")

    # Execute Step 2: Dinomaly2 (LPT scheduling across 8 GPUs)
    run_task_batch_on_gpus(dino_tasks, gpu_list, "Step 2: Training Dinomaly2 Models (LPT Multi-Iteration)")

    # Step 3: Two-Stage Feature Bank Tasks
    bank_tasks = []
    if bank_data and bank_data.is_dir():
        for iters in args.max_iters:
            for s in args.image_sizes:
                dino_s = (s // 14) * 14 if s % 14 != 0 else s
                for n in valid_n_samples:
                    task_out = outs_dir / f"dinomaly2_n{n}_s{s}_iter{iters}_seed{args.seed}"
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
                            "--backbone", str(args.backbone),
                            "--image_size", str(dino_s),
                            "--cuda", "0",
                        ]
                        desc = f"FeatureBank N={n} Size={s} Iter={iters}"
                        meta = {"n": n, "size": s, "iters": iters, "task_type": "feature_bank"}
                        bank_tasks.append(("feature_bank", cmd, desc, str(task_out), meta))

    run_task_batch_on_gpus(bank_tasks, gpu_list, "Step 3: Building Two-Stage Defect & Normal Feature Banks")

    # Step 4: Multi-GPU Parallel Full Evaluation
    print("\n" + "=" * 80)
    print("=== Step 4: Running Multi-GPU Parallel Unified Evaluation ===")
    print("=" * 80)

    eval_cmd = [
        str(PYTHON), str(ROOT / "evaluate_benchmark.py"),
        "--outs_dir", str(outs_dir),
        "--test_list", str(test_full_p),
        "--clean_test_list", str(clean_indomain_p),
        "--train_sizes"] + [str(n) for n in valid_n_samples] + [
        "--image_sizes"] + [str(s) for s in args.image_sizes] + [
        "--max_iters"] + [str(it) for it in args.max_iters] + [
        "--gpus", args.gpus
    ]
    env_eval = os.environ.copy()
    pythonpaths = [str(ROOT), str(ROOT / "Dinomaly2"), str(ROOT / "patchcore-inspection" / "src")]
    curr_pp = env_eval.get("PYTHONPATH", "")
    env_eval["PYTHONPATH"] = os.pathsep.join(pythonpaths + ([curr_pp] if curr_pp else []))
    env_eval["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    t0_eval = time.perf_counter()
    proc_eval = subprocess.run(eval_cmd, env=env_eval)
    eval_elapsed = time.perf_counter() - t0_eval
    if proc_eval.returncode != 0:
        print(f"[ERR] Multi-GPU evaluation failed with exit code {proc_eval.returncode}")
        sys.exit(proc_eval.returncode)
    else:
        print(f"[SUCCESS] Multi-GPU parallel evaluation finished in {eval_elapsed:.1f}s ({eval_elapsed/60.0:.2f}m)!")

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
    subprocess.run(plot_cmd, check=True)

    # Step 6: Generate Comprehensive Markdown Benchmark Report
    print("\n" + "=" * 80)
    print("=== Step 6: Generating Comprehensive Markdown Benchmark Report ===")
    print("=" * 80)
    report_cmd = [
        str(PYTHON), str(ROOT / "generate_final_report_multisize.py"),
        "--outs_dir", str(outs_dir)
    ]
    subprocess.run(report_cmd, check=True)

    print("\n" + "=" * 80)
    print("=== Benchmark Pipeline Completed Successfully! ===")
    print(f"Report: {outs_dir / 'FINAL_BENCHMARK_REPORT.md'}")
    print(f"Charts: {outs_dir / 'charts'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
