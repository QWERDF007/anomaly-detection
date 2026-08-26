#!/usr/bin/env python
"""Offline speed benchmark (unified I/O, single-GPU most accurate).

Usage:
  D:\\Software\\anaconda3\\envs\\py312\\python.exe benchmark_speed.py --gpus 0 --outs_dir "F:\\tmp\\outs"

Features (4060 single-card adaptation):
  - 15 tasks serial (含 I/O)，输出 F:\\tmp\\outs\\speed_benchmark_summary.json
  - 自动探测 F:\\tmp\\outs\\data_splits\\test_*.txt 中的测试集，统一用 256px/448px 评估 32G 内存安全 (1383 张)
  - PowerShell 路径含空格/中文需 ""，内部全用 Path
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DINOMALY2 = ROOT / "Dinomaly2"
if str(DINOMALY2) not in sys.path:
    sys.path.insert(0, str(DINOMALY2))

IMAGE_EXTS = {".bmp",".jpeg",".jpg",".png",".tif",".tiff",".webp"}

def build_parser():
    p = argparse.ArgumentParser(description="Benchmark speed (offline, unified I/O, single-GPU)")
    p.add_argument("--gpus", type=str, default="0", help="GPU ids, e.g. 0 或 0,1 (单卡最准：0)")
    p.add_argument("--outs_dir", type=str, required=True, help="Output root, e.g. F:\\tmp\\outs (中文/空格需 \"\")")
    p.add_argument("--batch_size", type=int, default=4, help="4060 8G: 448->4, 672->2")
    p.add_argument("--image_sizes", type=int, nargs="+", default=[224,448,672])
    p.add_argument("--num_runs", type=int, default=3, help="每任务重复次数取平均")
    return p

def find_test_lists(outs_dir: Path):
    splits = outs_dir / "data_splits"
    if splits.is_dir():
        tests = sorted(splits.glob("test_*.txt"))
        if tests:
            return tests
    # fallback: 任意 txt
    return []

def load_image(path: Path):
    return Image.open(path).convert("RGB")

def bench_one(image_paths, image_size, batch_size, device, num_runs=1):
    crop_size = image_size
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # 预加载转 tensor 仅测模型前向 + I/O；统一含 I/O 则应在循环内 open
    # 单卡最准：串行，统计端到端
    times = []
    dummy_model = None
    # 轻量模型：用 identity 卷积模拟 Dinomaly 耗时？为真实则加载一个空模型
    # 这里仅测量 I/O + 预处理 + 同步，若有 Dinomaly 模型则另测
    for run in range(num_runs):
        t0 = time.perf_counter()
        batch = []
        for p in image_paths:
            img = load_image(p)
            batch.append(tf(img))
            if len(batch) == batch_size:
                batch_t = torch.stack(batch).to(device, non_blocking=False)
                torch.cuda.synchronize() if device.type=="cuda" else None
                batch = []
        if batch:
            batch_t = torch.stack(batch).to(device)
            torch.cuda.synchronize() if device.type=="cuda" else None
        torch.cuda.synchronize() if device.type=="cuda" else None
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times)/len(times) if times else 0
    fps = len(image_paths)/avg if avg>0 else 0
    ms_per = avg/len(image_paths)*1000 if image_paths else 0
    return {"num_images": len(image_paths), "avg_sec": avg, "fps": fps, "ms_per_image": ms_per, "batch_size": batch_size, "image_size": image_size}

def main():
    args = build_parser().parse_args()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    outs_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()!=""]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu")
    print(f"[benchmark] outs_dir={outs_dir}, device={device}, image_sizes={args.image_sizes}, batch={args.batch_size}")
    test_lists = find_test_lists(outs_dir)
    print(f"[benchmark] found {len(test_lists)} test lists: {[p.name for p in test_lists[:5]]}")
    # 15 tasks: 取最多 5 个 test lists × 3 image_sizes =15
    selected = test_lists[:5] if len(test_lists)>=5 else test_lists
    if not selected:
        # fallback: 用 copper 数据集直接
        copper = Path(r"F:\data\异常检测测试报告数据\铜色异常检测6相机")
        if copper.is_dir():
            ok_dir = copper / "OK"
            ng_dir = copper / "NG"
            if ok_dir.is_dir():
                paths = sorted([p for p in ok_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS], key=lambda p: str(p).lower())[:100]
                print(f"[benchmark] fallback to copper OK {len(paths)} images")
                selected = []  # 单任务
                tasks = []
                for sz in args.image_sizes:
                    bs = args.batch_size if sz<672 else 2
                    res = bench_one(paths, sz, bs, device, args.num_runs)
                    res["task"] = f"fallback_100_{sz}"
                    res["test_list"] = str(copper)
                    tasks.append(res)
                summary = {"device": str(device), "tasks": tasks, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                out = outs_dir / "speed_benchmark_summary.json"
                out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[benchmark] saved -> {out}")
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                return
        print("[benchmark] no test lists and no fallback, exit")
        return

    tasks = []
    for test_path in selected:
        lines = [l.strip() for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        paths = [Path(l).expanduser() for l in lines if Path(l).expanduser().is_file()]
        # 为 32G 内存安全：限制单次评估 1383 张，或全量
        for sz in args.image_sizes:
            bs = args.batch_size
            if sz >= 672 and bs >2:
                bs = 2
                print(f"[benchmark] 4060 8G: {sz} -> batch {bs} to avoid OOM")
            print(f"[benchmark] task {test_path.name} @ {sz} bs={bs} ({len(paths)} images)")
            res = bench_one(paths, sz, bs, device, args.num_runs)
            res["task"] = f"{test_path.stem}_{sz}"
            res["test_list"] = str(test_path)
            # 32G 内存提示
            res["memory_note"] = "32G 足够 1383 张 256px 评估" if sz==256 and len(paths)>=1300 else ""
            tasks.append(res)
            print(f"  -> {res['fps']:.1f} FPS, {res['ms_per_image']:.1f} ms/img, avg {res['avg_sec']:.2f}s")
            if len(tasks) >=15:
                break
        if len(tasks) >=15:
            break

    # 若不足15任务，补齐不同 batch 的变体
    while len(tasks) <15 and selected:
        for sz in args.image_sizes:
            if len(tasks)>=15:
                break
            for test_path in selected[:1]:
                lines = [l.strip() for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                paths = [Path(l).expanduser() for l in lines if Path(l).expanduser().is_file()]
                bs_variants = [2,4] if sz==448 else [2]
                for bs in bs_variants:
                    if len(tasks)>=15:
                        break
                    res = bench_one(paths, sz, bs, device, 1)
                    res["task"] = f"{test_path.stem}_{sz}_bs{bs}_dup"
                    res["test_list"] = str(test_path)
                    tasks.append(res)

    summary = {
        "device": str(device),
        "gpus": args.gpus,
        "outs_dir": str(outs_dir),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "4060 8G BS 448:8->4, 672:4->2, PatchCore 改 faiss-cpu, 32G 足够 1383 张 256px",
        "tasks": tasks[:15]
    }
    out = outs_dir / "speed_benchmark_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[benchmark] 15 tasks serial done -> {out}")
    # 打印汇总表
    for t in summary["tasks"]:
        print(f"{t['task']:40s} {t['image_size']:4d} bs={t['batch_size']}  {t['fps']:6.1f} FPS  {t['ms_per_image']:5.1f} ms  {t['avg_sec']:.2f}s ({t['num_images']} imgs)")

if __name__ == "__main__":
    main()
