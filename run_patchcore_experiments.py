#!/usr/bin/env python3
"""Run all 9 PatchCore experiments across available GPUs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Dict, List

PYTHON_BIN = "/home/dell/miniconda3/envs/anomaly/bin/python"
BASE_DIR = Path("/data/wt/anomaly-detection")
OUT_ROOT = Path("/data/wt/outs")
SPLITS_DIR = OUT_ROOT / "data_splits"

# Available GPUs
GPU_LIST = [1, 2, 4, 5, 0, 3, 6, 7]


def generate_patchcore_tasks() -> List[Dict[str, Any]]:
    tasks = []
    sample_counts = [50, 100, 200]
    image_sizes = [224, 448, 672]

    for count in sample_counts:
        for size in image_sizes:
            batch_size = 16 if size == 224 else (8 if size == 448 else 4)
            exp_id = f"patchcore_n{count}_s{size}"
            out_dir = OUT_ROOT / exp_id
            tasks.append({
                "exp_id": exp_id,
                "model": "PatchCore",
                "sample_count": count,
                "image_size": size,
                "batch_size": batch_size,
                "train_file": str(SPLITS_DIR / f"train_{count}.txt"),
                "test_file": str(SPLITS_DIR / f"test_{count}.txt"),
                "out_dir": str(out_dir),
                "cmd": [
                    PYTHON_BIN,
                    str(BASE_DIR / "patchcore-inspection" / "train.py"),
                    "--data_path", str(SPLITS_DIR / f"train_{count}.txt"),
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


def run_single_patchcore(task: Dict[str, Any], gpu_id: int) -> Dict[str, Any]:
    exp_id = task["exp_id"]
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = f"{BASE_DIR / 'patchcore-inspection' / 'src'}:{BASE_DIR / 'Dinomaly2'}:{env.get('PYTHONPATH', '')}"

    print(f"[{time.strftime('%H:%M:%S')}] [GPU {gpu_id}] >>> STARTING {exp_id} (Model: {task['model']}, N={task['sample_count']}, Size={task['image_size']})", flush=True)

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
        "P-AUROC": None,
        "P-AP": None,
        "P-F1": None,
        "P-AUPRO": None,
        "peak_gpu_mem_mb": None,
    }

    # Parse metrics from metrics.json or results.csv
    candidate_json_files = list(out_dir.rglob("metrics.json"))
    candidate_csv_files = list(out_dir.rglob("results.csv"))

    if candidate_json_files:
        candidate_json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(candidate_json_files[0], "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for k in metrics.keys():
                    if k in loaded and loaded[k] is not None:
                        metrics[k] = loaded[k]
        except Exception as e:
            print(f"[!] Failed to parse metrics.json for {exp_id}: {e}")

    if candidate_csv_files and (metrics["I-AUROC"] is None or metrics["peak_gpu_mem_mb"] is None):
        candidate_csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(candidate_csv_files[0], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k in metrics.keys():
                        if k in row and row[k] != "" and metrics[k] is None:
                            try:
                                metrics[k] = float(row[k])
                            except ValueError:
                                pass
        except Exception as e:
            print(f"[!] Failed to parse results.csv for {exp_id}: {e}")

    result_record = {
        "exp_id": exp_id,
        "model": task["model"],
        "sample_count": task["sample_count"],
        "image_size": task["image_size"],
        "batch_size": task["batch_size"],
        "gpu_id": gpu_id,
        "status": status,
        "elapsed_sec": round(elapsed_sec, 2),
        "peak_gpu_mem_mb": metrics["peak_gpu_mem_mb"],
        "I-AUROC": metrics["I-AUROC"],
        "I-AP": metrics["I-AP"],
        "I-F1": metrics["I-F1"],
        "P-AUROC": metrics["P-AUROC"],
        "P-AP": metrics["P-AP"],
        "P-F1": metrics["P-F1"],
        "P-AUPRO": metrics["P-AUPRO"],
        "log_file": str(log_file),
    }

    print(
        f"[{time.strftime('%H:%M:%S')}] [GPU {gpu_id}] <<< FINISHED {exp_id} ({status}) in {elapsed_sec:.1f}s | "
        f"Mem: {metrics['peak_gpu_mem_mb']} MB | I-AUROC: {metrics['I-AUROC']} | I-AP: {metrics['I-AP']} | I-F1: {metrics['I-F1']}",
        flush=True,
    )
    return result_record


def worker_loop(
    gpu_id: int,
    task_q: queue.Queue,
    results_list: List[Dict[str, Any]],
    lock: threading.Lock,
):
    while True:
        try:
            task = task_q.get_nowait()
        except queue.Empty:
            break

        res = run_single_patchcore(task, gpu_id)
        with lock:
            results_list.append(res)
        task_q.task_done()


def main():
    print("=" * 80)
    print("Running PatchCore Experiments (9 combinations)")
    print(f"GPUs available: {len(GPU_LIST)} ({GPU_LIST})")
    print("=" * 80)

    tasks = generate_patchcore_tasks()
    task_q = queue.Queue()
    for t in tasks:
        task_q.put(t)

    results_list: List[Dict[str, Any]] = []
    lock = threading.Lock()

    t_start = time.time()
    threads = []
    for gpu_id in GPU_LIST:
        t = threading.Thread(
            target=worker_loop,
            args=(gpu_id, task_q, results_list, lock),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"ALL {len(tasks)} PATCHCORE EXPERIMENTS COMPLETED in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
