#!/usr/bin/env python3
"""Benchmark Inference Latency (ms/image) and Throughput (FPS) for Dinomaly2 and PatchCore.

Tests:
1. Dinomaly2 at sizes 224, 448, 672:
   - Batch size 1 (pure latency ms/img, FPS)
   - Standard batch size (16 for 224, 8 for 448, 4 for 672) (throughput FPS)
2. PatchCore across sizes (224, 448, 672) and sample sizes (50, 100, 200, 400):
   - Batch size 1 (latency ms/img, FPS)
   - Standard batch size (throughput FPS)

GPU公平对比：PatchCore与Dinomaly2同在GPU上检索（FaissNN on_gpu=True），与train.py:158保持一致
支持多卡并行：--gpus 0,1,2,3,4,5,6,7 或 --gpus auto (默认8卡)
"""

from __future__ import annotations

from functools import partial
import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path("/data/wt/anomaly-detection")
sys.path.insert(0, str(BASE_DIR / "patchcore-inspection" / "src"))
sys.path.insert(0, str(BASE_DIR / "Dinomaly2"))

import patchcore.patchcore
import patchcore.backbones
from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, LinearAttention2


def build_dinomaly2(image_size: int, device: torch.device):
    encoder_name = 'dinov2reg_vit_small_14'
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 384, 6
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

    dropout = 0.4
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
        nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
                      nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout))
    ])

    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=partial(LinearAttention2, eps=1e-8))
        for _ in range(8)
    ])

    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=True,
    ).to(device)
    model.eval()
    return model


def benchmark_dinomaly2(image_size: int, device: torch.device) -> Dict[str, Any]:
    print(f"\n[+] Benchmarking Dinomaly2 (Size={image_size}) on {device}...", flush=True)
    model = build_dinomaly2(image_size, device)

    results = {}

    # 1. Batch size 1 (Single Image Latency)
    dummy_bs1 = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_bs1)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None

    iters = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_bs1)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    total_time = time.perf_counter() - t0
    latency_bs1_ms = (total_time / iters) * 1000.0
    fps_bs1 = iters / total_time
    results["bs1_latency_ms"] = round(latency_bs1_ms, 2)
    results["bs1_fps"] = round(fps_bs1, 1)

    # 2. Standard batch size (Throughput)
    batch_size = 16 if image_size == 224 else (8 if image_size == 448 else 4)
    dummy_batch = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_batch)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_batch)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    total_time = time.perf_counter() - t0
    batch_latency_ms = (total_time / iters) * 1000.0
    throughput_fps = (iters * batch_size) / total_time
    latency_per_img_ms = batch_latency_ms / batch_size

    results["batch_size"] = batch_size
    results["batch_latency_ms"] = round(batch_latency_ms, 2)
    results["batch_latency_per_img_ms"] = round(latency_per_img_ms, 2)
    results["batch_throughput_fps"] = round(throughput_fps, 1)

    print(f"    BS=1: {latency_bs1_ms:.2f} ms/img ({fps_bs1:.1f} FPS) | BS={batch_size}: {latency_per_img_ms:.2f} ms/img ({throughput_fps:.1f} FPS)", flush=True)
    # cleanup
    del model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    return results


def benchmark_patchcore(
    exp_dir: Path,
    image_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n[+] Benchmarking PatchCore ({exp_dir} Size={image_size}) on {device}...", flush=True)
    import patchcore.common
    model = patchcore.patchcore.PatchCore(device)
    # 公平对比：PatchCore与Dinomaly2同在GPU上检索，与train.py:158 on_gpu=True保持一致
    use_gpu = device.type == "cuda"
    model.load_from_path(
        str(exp_dir),
        device=device,
        nn_method=patchcore.common.FaissNN(use_gpu, 4),
    )

    results = {}

    # 1. Batch size 1
    dummy_bs1 = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(2):
            _ = model._predict(dummy_bs1)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None

    iters = 5
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model._predict(dummy_bs1)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    total_time = time.perf_counter() - t0
    latency_bs1_ms = (total_time / iters) * 1000.0
    fps_bs1 = iters / total_time
    results["bs1_latency_ms"] = round(latency_bs1_ms, 2)
    results["bs1_fps"] = round(fps_bs1, 1)

    # 2. Standard batch size
    batch_size = 16 if image_size == 224 else (8 if image_size == 448 else 4)
    dummy_batch = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(2):
            _ = model._predict(dummy_batch)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model._predict(dummy_batch)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    total_time = time.perf_counter() - t0
    batch_latency_ms = (total_time / iters) * 1000.0
    throughput_fps = (iters * batch_size) / total_time
    latency_per_img_ms = batch_latency_ms / batch_size

    results["batch_size"] = batch_size
    results["batch_latency_ms"] = round(batch_latency_ms, 2)
    results["batch_latency_per_img_ms"] = round(latency_per_img_ms, 2)
    results["batch_throughput_fps"] = round(throughput_fps, 1)

    print(f"    BS=1: {latency_bs1_ms:.2f} ms/img ({fps_bs1:.1f} FPS) | BS={batch_size}: {latency_per_img_ms:.2f} ms/img ({throughput_fps:.1f} FPS)", flush=True)
    del model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    return results


def find_patchcore_checkpoint(outs_dir: Path, n: int, size: int) -> Path | None:
    """查找 PatchCore 对应 N/size 的最新 checkpoint 目录（含 nnscorer_search_index.faiss）"""
    # 匹配 patchcore_n{N}_s{size}_seed* 目录
    pattern = f"patchcore_n{n}_s{size}_seed*"
    candidates = []
    for seed_dir in outs_dir.glob(pattern):
        if not seed_dir.is_dir():
            continue
        # 在 seed_dir 下查找所有包含 faiss 的时间戳子目录
        for sub in seed_dir.iterdir():
            faiss_file = sub / "nnscorer_search_index.faiss"
            if faiss_file.is_file():
                candidates.append((faiss_file.stat().st_mtime, sub))
        # 也兼容直接在 seed_dir 下（无时间戳）的情况
        direct_faiss = seed_dir / "nnscorer_search_index.faiss"
        if direct_faiss.is_file():
            candidates.append((direct_faiss.stat().st_mtime, seed_dir))
    if not candidates:
        # fallback: 全盘搜索（兼容旧命名）
        for p in outs_dir.rglob("nnscorer_search_index.faiss"):
            # 检查路径是否包含 patchcore_n{n}_s{size}
            if f"n{n}_s{size}" in str(p):
                candidates.append((p.stat().st_mtime, p.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def generate_tasks(outs_dir: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    # Dinomaly2 tasks
    for size in [224, 448, 672]:
        tasks.append({
            "type": "dinomaly2",
            "size": size,
            "key": f"size_{size}",
            "desc": f"Dinomaly2 size={size}"
        })
    # PatchCore tasks
    for size in [224, 448, 672]:
        for n in [50, 100, 200, 400]:
            exp_dir = find_patchcore_checkpoint(outs_dir, n, size)
            if exp_dir is None:
                print(f"[!] No checkpoint found for PatchCore n={n} size={size}, skipped", flush=True)
                continue
            tasks.append({
                "type": "patchcore",
                "n": n,
                "size": size,
                "exp_dir": exp_dir,
                "key": f"n{n}_s{size}",
                "desc": f"PatchCore n={n} size={size} ({exp_dir})"
            })
    return tasks


def worker_loop(gpu_id: int, task_q: queue.Queue, results: Dict[str, Any], lock: threading.Lock, outs_dir: Path):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # 绑定当前线程到指定 GPU
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print(f"[Worker GPU {gpu_id}] started on {device}", flush=True)
    while True:
        try:
            task = task_q.get_nowait()
        except queue.Empty:
            break
        desc = task["desc"]
        print(f"[GPU {gpu_id}] >>> START {desc}", flush=True)
        t0 = time.time()
        try:
            if task["type"] == "dinomaly2":
                res = benchmark_dinomaly2(task["size"], device)
                with lock:
                    results["dinomaly2"][task["key"]] = res
            else:  # patchcore
                res = benchmark_patchcore(task["exp_dir"], task["size"], device)
                with lock:
                    results["patchcore"][task["key"]] = res
            elapsed = time.time() - t0
            print(f"[GPU {gpu_id}] <<< DONE {desc} in {elapsed:.1f}s", flush=True)
        except Exception as e:
            import traceback
            print(f"[GPU {gpu_id}] !!! FAILED {desc}: {e}", flush=True)
            traceback.print_exc()
        finally:
            task_q.task_done()


def _run_task_in_process(task: Dict[str, Any], gpu_id: int) -> tuple[str, str, Dict[str, Any]]:
    """在独立进程中执行单个压测任务（避免线程级CUDA竞争）"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 子进程内需重新初始化CUDA可见设备，统一用 cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    # 重新设置BASE_DIR路径（子进程继承）
    task_type = task["type"]
    if task_type == "dinomaly2":
        res = benchmark_dinomaly2(task["size"], device)
        return ("dinomaly2", task["key"], res)
    else:
        res = benchmark_patchcore(task["exp_dir"], task["size"], device)
        return ("patchcore", task["key"], res)


def main():
    parser = argparse.ArgumentParser(description="GPU benchmark for Dinomaly2 & PatchCore (支持多卡并行)")
    parser.add_argument("--gpus", type=str, default="auto", help="GPU ids comma-separated, e.g. 0,1,2 or auto for all available (default: auto)")
    parser.add_argument("--outs_dir", type=str, default="/data/wt/outs", help="outs directory")
    parser.add_argument("--parallel", action="store_true", help="启用多卡并行（多进程，默认串行以保证精度）")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数，默认等于GPU数")
    args = parser.parse_args()

    if args.gpus == "auto":
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_list = list(range(num_gpus)) if num_gpus > 0 else []
        if not gpu_list:
            gpu_list = [0]
    else:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]

    if torch.cuda.is_available():
        print("=" * 80)
        mode = "并行多进程" if args.parallel else "串行"
        print(f"BENCHMARKING INFERENCE LATENCY & THROUGHPUT ({mode}, GPUs: {gpu_list}, Faiss on GPU)")
        for gid in gpu_list:
            try:
                print(f"  GPU {gid}: {torch.cuda.get_device_name(gid)}")
            except:
                print(f"  GPU {gid}: unknown")
        print("=" * 80)
    else:
        print("CUDA not available, falling back to CPU")
        gpu_list = [0]

    outs_dir = Path(args.outs_dir)
    tasks = generate_tasks(outs_dir)
    print(f"[+] Total tasks: {len(tasks)} (Dinomaly2: {sum(1 for t in tasks if t['type']=='dinomaly2')}, PatchCore: {sum(1 for t in tasks if t['type']=='patchcore')})")
    for t in tasks:
        print(f"  - {t['desc']}")

    summary: Dict[str, Any] = {
        "dinomaly2": {},
        "patchcore": {},
    }

    t_start = time.time()
    if args.parallel:
        # 多进程并行：每个任务独立进程，通过 CUDA_VISIBLE_DEVICES 隔离，避免线程级CUDA竞争
        import concurrent.futures
        workers = args.workers or len(gpu_list)
        print(f"[+] 并行模式：{workers} workers, {len(gpu_list)} GPUs 轮询", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, task in enumerate(tasks):
                gid = gpu_list[idx % len(gpu_list)]
                fut = executor.submit(_run_task_in_process, task, gid)
                futures[fut] = task["desc"]
            for fut in concurrent.futures.as_completed(futures):
                desc = futures[fut]
                try:
                    cat, key, res = fut.result()
                    summary[cat][key] = res
                    print(f"[DONE] {desc} -> {res.get('bs1_latency_ms')} ms", flush=True)
                except Exception as e:
                    import traceback
                    print(f"[FAILED] {desc}: {e}", flush=True)
                    traceback.print_exc()
    else:
        # 串行模式：单GPU顺序执行，结果最精确（无资源竞争），与之前隔离压测一致
        device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        print(f"[+] 串行模式：使用 {device} 顺序执行（精度最高）", flush=True)
        for task in tasks:
            desc = task["desc"]
            print(f">>> START {desc} on {device}", flush=True)
            try:
                if task["type"] == "dinomaly2":
                    res = benchmark_dinomaly2(task["size"], device)
                    summary["dinomaly2"][task["key"]] = res
                else:
                    res = benchmark_patchcore(task["exp_dir"], task["size"], device)
                    summary["patchcore"][task["key"]] = res
                print(f"<<< DONE {desc}", flush=True)
            except Exception as e:
                import traceback
                print(f"!!! FAILED {desc}: {e}", flush=True)
                traceback.print_exc()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL {len(tasks)} TASKS COMPLETED in {total_time:.1f}s")
    print("=" * 80)
    # 按 key 排序后保存
    summary["dinomaly2"] = dict(sorted(summary["dinomaly2"].items()))
    summary["patchcore"] = dict(sorted(summary["patchcore"].items()))

    out_file = outs_dir / "speed_benchmark_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved speed benchmark summary to: {out_file}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
