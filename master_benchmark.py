#!/usr/bin/env python3
"""Master Benchmark Orchestrator for Dinomaly2 & PatchCore on 8 GPUs.

Features:
- Full matrix: 3 Seeds (42, 100, 2024) x 4 Sample Sizes (50, 100, 200, 400) x 3 Resolutions (224, 448, 672) x 2 Models = 72 tasks.
- Concurrent execution across 8 GPUs (1 GPU per task worker).
- Automatic resumption and validation of existing results.
- Output reports to WORKLOG, CSV, and JSON.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from typing import Any, Dict, List

PYTHON_BIN = "/home/dell/miniconda3/envs/anomaly/bin/python"
BASE_DIR = Path("/data/wt/anomaly-detection")
OUT_ROOT = Path("/data/wt/outs")
SPLITS_DIR = OUT_ROOT / "data_splits"

GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]


def generate_tasks() -> List[Dict[str, Any]]:
    tasks = []
    seeds = [42, 100, 2024]
    sample_sizes = [50, 100, 200, 400]
    image_sizes = [224, 448, 672]

    # Dinomaly2 tasks (max_iters=2000)
    for seed in seeds:
        for n in sample_sizes:
            for size in image_sizes:
                batch_size = 16 if size == 224 else (8 if size == 448 else 4)
                exp_id = f"dinomaly2_n{n}_s{size}_seed{seed}"
                out_dir = OUT_ROOT / exp_id
                tasks.append({
                    "exp_id": exp_id,
                    "model": "Dinomaly2",
                    "sample_count": n,
                    "image_size": size,
                    "batch_size": batch_size,
                    "seed": seed,
                    "train_file": str(SPLITS_DIR / f"train_{n}_seed{seed}.txt"),
                    "test_file": str(SPLITS_DIR / f"test_{n}_seed{seed}.txt"),
                    "out_dir": str(out_dir),
                    "cmd": [
                        PYTHON_BIN,
                        str(BASE_DIR / "Dinomaly2" / "dinomaly_2D.py"),
                        "--data_path", str(SPLITS_DIR / f"train_{n}_seed{seed}.txt"),
                        "--dataset", "custom",
                        "--image_size", str(size),
                        "--crop_size", str(size),
                        "--batch_size", str(batch_size),
                        "--max_iters", "2000",
                        "--eval_interval", "-1",
                        "--save_dir", str(out_dir),
                    ],
                    "cwd": str(BASE_DIR / "Dinomaly2"),
                })

    # PatchCore tasks (sampling_percentage=0.1)
    for seed in seeds:
        for n in sample_sizes:
            for size in image_sizes:
                batch_size = 16 if size == 224 else (8 if size == 448 else 4)
                exp_id = f"patchcore_n{n}_s{size}_seed{seed}"
                out_dir = OUT_ROOT / exp_id
                tasks.append({
                    "exp_id": exp_id,
                    "model": "PatchCore",
                    "sample_count": n,
                    "image_size": size,
                    "batch_size": batch_size,
                    "seed": seed,
                    "train_file": str(SPLITS_DIR / f"train_{n}_seed{seed}.txt"),
                    "test_file": str(SPLITS_DIR / f"test_{n}_seed{seed}.txt"),
                    "out_dir": str(out_dir),
                    "cmd": [
                        PYTHON_BIN,
                        str(BASE_DIR / "patchcore-inspection" / "train.py"),
                        "--data_path", str(SPLITS_DIR / f"train_{n}_seed{seed}.txt"),
                        "--dataset", "custom",
                        "--backbone", "wideresnet50",
                        "--resize", str(size),
                        "--imagesize", str(size),
                        "--batch_size", str(batch_size),
                        "--sampling_percentage", "0.1",
                        "--save_dir", str(out_dir),
                    ],
                    "cwd": str(BASE_DIR / "patchcore-inspection"),
                })

    return tasks


def run_single_task(task: Dict[str, Any], gpu_id: int) -> Dict[str, Any]:
    exp_id = task["exp_id"]
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"

    # Check if already completed
    candidate_json_files = list(out_dir.rglob("metrics.json"))
    candidate_csv_files = list(out_dir.rglob("results.csv"))

    if candidate_json_files:
        candidate_json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(candidate_json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("I-AUROC") is not None and data.get("I-AP") is not None:
                    # Attempt to extract elapsed time from log
                    elapsed_sec = 0.0
                    if log_file.exists():
                        with open(log_file, "r", encoding="utf-8", errors="ignore") as fl:
                            lcontent = fl.read()
                            el_m = re.findall(r"elapsed:([0-9:]+)", lcontent)
                            if el_m:
                                parts = [int(p) for p in el_m[-1].split(":")]
                                if len(parts) == 3:
                                    elapsed_sec = parts[0]*3600 + parts[1]*60 + parts[2]
                                elif len(parts) == 2:
                                    elapsed_sec = parts[0]*60 + parts[1]
                    print(
                        f"[{time.strftime('%H:%M:%S')}] [GPU {gpu_id}] [CACHED] {exp_id} -> "
                        f"I-AUROC: {data.get('I-AUROC')}, I-AP: {data.get('I-AP')}, I-F1: {data.get('I-F1')}",
                        flush=True,
                    )
                    return {
                        "exp_id": exp_id,
                        "model": task["model"],
                        "sample_count": task["sample_count"],
                        "image_size": task["image_size"],
                        "seed": task["seed"],
                        "batch_size": task["batch_size"],
                        "status": "CACHED",
                        "elapsed_sec": round(elapsed_sec, 1),
                        "peak_gpu_mem_mb": data.get("peak_gpu_mem_mb"),
                        "I-AUROC": data.get("I-AUROC"),
                        "I-AP": data.get("I-AP"),
                        "I-F1": data.get("I-F1"),
                        "out_dir": task["out_dir"],
                    }
        except Exception:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{BASE_DIR / 'patchcore-inspection' / 'src'}:{BASE_DIR / 'Dinomaly2'}:{env.get('PYTHONPATH', '')}"

    print(
        f"[{time.strftime('%H:%M:%S')}] [GPU {gpu_id}] >>> STARTING {exp_id} "
        f"(Model: {task['model']}, N={task['sample_count']}, Size={task['image_size']}, Seed={task['seed']})",
        flush=True,
    )

    t0 = time.time()
    try:
        with open(log_file, "w", encoding="utf-8") as f_log:
            proc = subprocess.run(
                task["cmd"],
                cwd=task["cwd"],
                env=env,
                stdout=f_log,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
        elapsed_sec = time.time() - t0
        status = "SUCCESS"
    except subprocess.CalledProcessError as e:
        elapsed_sec = time.time() - t0
        status = f"FAILED (code {e.returncode})"
        print(f"[!] [GPU {gpu_id}] FAILED {exp_id}: code {e.returncode}. Log: {log_file}", flush=True)
    except Exception as e:
        elapsed_sec = time.time() - t0
        status = f"ERROR ({e})"
        print(f"[!] [GPU {gpu_id}] ERROR {exp_id}: {e}", flush=True)

    metrics = {
        "I-AUROC": None,
        "I-AP": None,
        "I-F1": None,
        "peak_gpu_mem_mb": None,
    }

    # Parse metrics from files or log
    candidate_json_files = list(out_dir.rglob("metrics.json"))
    candidate_csv_files = list(out_dir.rglob("results.csv"))

    if candidate_json_files:
        candidate_json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(candidate_json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                for k in metrics.keys():
                    if k in data and data[k] is not None:
                        metrics[k] = data[k]
        except Exception:
            pass

    if (metrics["I-AUROC"] is None or metrics["I-AP"] is None) and candidate_csv_files:
        candidate_csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(candidate_csv_files[0], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k in metrics.keys():
                        if k in row and row[k]:
                            try:
                                metrics[k] = float(row[k])
                            except ValueError:
                                pass
        except Exception:
            pass

    # Fallback parse from log
    if (metrics["I-AUROC"] is None or metrics["peak_gpu_mem_mb"] is None) and log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if metrics["peak_gpu_mem_mb"] is None:
                    mem_m = re.findall(r"Peak GPU Memory:\s*([0-9.]+)\s*MB", content)
                    if mem_m:
                        metrics["peak_gpu_mem_mb"] = float(mem_m[-1])
                if metrics["I-AUROC"] is None:
                    auroc_m = re.findall(r"I-Auroc:\s*([0-9.]+)", content, re.IGNORECASE)
                    if auroc_m:
                        metrics["I-AUROC"] = float(auroc_m[-1])
                if metrics["I-AP"] is None:
                    ap_m = re.findall(r"I-AP:\s*([0-9.]+)", content, re.IGNORECASE)
                    if ap_m:
                        metrics["I-AP"] = float(ap_m[-1])
                if metrics["I-F1"] is None:
                    f1_m = re.findall(r"I-F1:\s*([0-9.]+)", content, re.IGNORECASE)
                    if f1_m:
                        metrics["I-F1"] = float(f1_m[-1])
        except Exception:
            pass

    res_item = {
        "exp_id": exp_id,
        "model": task["model"],
        "sample_count": task["sample_count"],
        "image_size": task["image_size"],
        "seed": task["seed"],
        "batch_size": task["batch_size"],
        "status": status,
        "elapsed_sec": round(elapsed_sec, 1),
        "peak_gpu_mem_mb": metrics["peak_gpu_mem_mb"],
        "I-AUROC": metrics["I-AUROC"],
        "I-AP": metrics["I-AP"],
        "I-F1": metrics["I-F1"],
        "out_dir": task["out_dir"],
    }

    print(
        f"[{time.strftime('%H:%M:%S')}] [GPU {gpu_id}] <<< FINISHED {exp_id} ({status}) "
        f"in {elapsed_sec:.1f}s | Mem: {metrics['peak_gpu_mem_mb']} MB | "
        f"I-AUROC: {metrics['I-AUROC']} | I-AP: {metrics['I-AP']} | I-F1: {metrics['I-F1']}",
        flush=True,
    )
    return res_item


def worker_loop(gpu_id: int, task_q: queue.Queue, results: List[Dict[str, Any]], lock: threading.Lock):
    while True:
        try:
            task = task_q.get_nowait()
        except queue.Empty:
            break
        res = run_single_task(task, gpu_id)
        with lock:
            results.append(res)
        task_q.task_done()


def main():
    print("=" * 80)
    print("STARTING MASTER MULTI-SEED BENCHMARK (72 TASKS ON 8 GPUS)")
    print("=" * 80)

    tasks = generate_tasks()
    print(f"[+] Total tasks queued: {len(tasks)}")

    task_q: queue.Queue = queue.Queue()
    for t in tasks:
        task_q.put(t)

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    t_start = time.time()
    threads = []
    for gpu_id in GPU_LIST:
        th = threading.Thread(target=worker_loop, args=(gpu_id, task_q, results, lock), daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL {len(results)} TASKS COMPLETED in {total_time:.1f}s ({total_time/60.0:.2f} mins)")
    print("=" * 80)

    # Save raw results
    summary_file = OUT_ROOT / "all_experiments_raw.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved raw results to: {summary_file}")


if __name__ == "__main__":
    main()
