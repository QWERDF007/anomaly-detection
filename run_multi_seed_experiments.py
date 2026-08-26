#!/usr/bin/env python3
"""Run multi-seed experiments (Seed 100 and Seed 2024) across all combinations on 8 GPUs."""

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
    seeds = [100, 2024]
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
                loaded = json.load(f)
                for k in metrics.keys():
                    if k in loaded and loaded[k] is not None:
                        metrics[k] = loaded[k]
        except Exception:
            pass

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
        except Exception:
            pass

    if log_file.exists() and (metrics["I-AUROC"] is None or metrics["peak_gpu_mem_mb"] is None):
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
                status = "SUCCESS"
        except Exception:
            pass

    result_record = {
        "exp_id": exp_id,
        "model": task["model"],
        "sample_count": task["sample_count"],
        "image_size": task["image_size"],
        "batch_size": task["batch_size"],
        "seed": task["seed"],
        "gpu_id": gpu_id,
        "status": status,
        "elapsed_sec": round(elapsed_sec, 2),
        "peak_gpu_mem_mb": metrics["peak_gpu_mem_mb"],
        "I-AUROC": metrics["I-AUROC"],
        "I-AP": metrics["I-AP"],
        "I-F1": metrics["I-F1"],
        "log_file": str(log_file),
    }

    # Save individual metric record
    try:
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result_record, f, indent=2)
    except Exception:
        pass

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
    print("Running Multi-Seed Experiments (Seed 100 and Seed 2024)")
    print(f"GPUs available: {len(GPU_LIST)} ({GPU_LIST})")
    print("=" * 80)

    tasks = generate_tasks()
    print(f"Total tasks in queue: {len(tasks)}")
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
    print(f"ALL {len(tasks)} MULTI-SEED EXPERIMENTS COMPLETED in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
