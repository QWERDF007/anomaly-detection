#!/usr/bin/env python3
"""Multi-GPU Master Experiment Orchestrator for Copper Anomaly Detection.

Runs 18 experiments in parallel across 8 GPUs:
- 3 Sample sizes: 50, 100, 200
- 3 Image resolutions: 224, 448, 672
- 2 Models: Dinomaly2 (max-iters=2000), PatchCore (sampling_percentage=0.1)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List


PYTHON_BIN = "/home/dell/miniconda3/envs/anomaly/bin/python"
BASE_DIR = Path("/data/wt/anomaly-detection")
OUT_ROOT = Path("/data/wt/outs")
SPLITS_DIR = OUT_ROOT / "data_splits"

GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]


def generate_task_list() -> List[Dict[str, Any]]:
    tasks = []
    
    sample_counts = [50, 100, 200]
    image_sizes = [224, 448, 672]
    
    # 1. Dinomaly2 experiments
    for count in sample_counts:
        for size in image_sizes:
            batch_size = 16 if size == 224 else (8 if size == 448 else 4)
            exp_id = f"dinomaly2_n{count}_s{size}"
            out_dir = OUT_ROOT / exp_id
            tasks.append({
                "exp_id": exp_id,
                "model": "Dinomaly2",
                "sample_count": count,
                "image_size": size,
                "batch_size": batch_size,
                "train_file": str(SPLITS_DIR / f"train_{count}.txt"),
                "test_file": str(SPLITS_DIR / f"test_{count}.txt"),
                "out_dir": str(out_dir),
                "cmd": [
                    PYTHON_BIN,
                    str(BASE_DIR / "Dinomaly2" / "dinomaly_2D.py"),
                    "--data_path", str(SPLITS_DIR / f"train_{count}.txt"),
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

    # 2. PatchCore experiments
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


def run_single_task(task: Dict[str, Any], gpu_id: int) -> Dict[str, Any]:
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
        print(f"[!] [GPU {gpu_id}] FAILED {exp_id}: return code {e.returncode}. See log: {log_file}", flush=True)
    except Exception as e:
        elapsed_sec = time.time() - t0
        status = f"ERROR ({e})"
        print(f"[!] [GPU {gpu_id}] ERROR {exp_id}: {e}", flush=True)

    # Parse metrics from metrics.json or results.csv or log
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

    # Find the newest subdirectory in out_dir (since Dinomaly2 / PatchCore creates timestamped folders)
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

    # Fallback: parse directly from run.log if any metric is still None
    if log_file.exists() and (metrics["I-AUROC"] is None or metrics["peak_gpu_mem_mb"] is None):
        import re
        try:
            log_text = log_file.read_text(encoding="utf-8")
            mem_match = re.findall(r"Peak GPU Memory:\s*([0-9.]+)\s*MB", log_text)
            if mem_match and metrics["peak_gpu_mem_mb"] is None:
                metrics["peak_gpu_mem_mb"] = float(mem_match[-1])
            metric_match = re.findall(
                r"Mean:\s*I-Auroc:([0-9.]+),\s*I-AP:([0-9.]+),\s*I-F1:([0-9.]+)",
                log_text,
            )
            if metric_match:
                metrics["I-AUROC"] = float(metric_match[-1][0])
                metrics["I-AP"] = float(metric_match[-1][1])
                metrics["I-F1"] = float(metric_match[-1][2])
                if status.startswith("FAILED") and "NameError" in log_text:
                    status = "SUCCESS"
        except Exception as e:
            print(f"[!] Failed to parse log fallback for {exp_id}: {e}")

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
        
        res = run_single_task(task, gpu_id)
        with lock:
            results_list.append(res)
        task_q.task_done()


def main():
    print("=" * 80)
    print("Multi-GPU Anomaly Detection Benchmark: Copper Dataset (6 Cameras Unified)")
    print(f"GPUs available: {len(GPU_LIST)} ({GPU_LIST})")
    print(f"Output root: {OUT_ROOT}")
    print("=" * 80)

    tasks = generate_task_list()
    print(f"Total experiments planned: {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"  [{i+1:02d}] {t['exp_id']} (Model: {t['model']}, OK={t['sample_count']}, Size={t['image_size']})")

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
    print(f"ALL {len(tasks)} EXPERIMENTS COMPLETED in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print("=" * 80)

    # Sort results deterministically by model, sample_count, image_size
    results_list.sort(key=lambda r: (r["model"], r["sample_count"], r["image_size"]))

    # Save summary CSV
    summary_csv = OUT_ROOT / "experiment_summary.csv"
    fields = [
        "model",
        "sample_count",
        "image_size",
        "I-AUROC",
        "I-AP",
        "I-F1",
        "peak_gpu_mem_mb",
        "elapsed_sec",
        "status",
        "exp_id",
        "gpu_id",
        "log_file",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_list)
    print(f"[+] Summary CSV saved to: {summary_csv}")

    # Save summary JSON
    summary_json = OUT_ROOT / "experiment_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_time_sec": round(total_time, 2),
                "num_experiments": len(results_list),
                "results": results_list,
            },
            f,
            indent=2,
        )
    print(f"[+] Summary JSON saved to: {summary_json}")


if __name__ == "__main__":
    main()
