#!/usr/bin/env python
"""Generate train/test splits from flat OK/NG folders (e.g. 铜色异常检测6相机).

Usage (PowerShell, 路径含空格/中文需 ""):
  D:\\Software\\anaconda3\\envs\\py312\\python.exe prepare_splits.py `
    --dataset_root "F:\\data\\异常检测测试报告数据\\铜色异常检测6相机" `
    --outs_dir "F:\\tmp\\outs"

Produces:
  F:\\tmp\\outs\\data_splits\\train_50_seed2024.txt / test_50_seed2024.txt
  ... 50/100/200/400 × 3 seeds (2024/2025/2026) = 12 train + 12 test = 24 files
  (兼容用户所述的“12文件”按 train 计；test 同步生成)

Rule:
  - OK: 1730 张随机抽 n 张作 train (n∈{50,100,200,400}), 剩余 OK + 全部 NG(53)作 test
  - 文本每行一条绝对路径；训练脚本需用 Path 自动处理中文/空格

Supports --dataset_root hardcode fix: DATASET_ROOT=Path(r"...")
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

IMAGE_EXTS = {".bmp",".jpeg",".jpg",".png",".tif",".tiff",".webp"}

def iter_images(directory: Path):
    if not directory.is_dir():
        return []
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: str(p).lower())

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare train/test txt splits for 铜色异常检测 (OK/NG flat layout)")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root containing OK and/or NG folders (Chinese/spaces must be quoted)")
    parser.add_argument("--outs_dir", type=str, required=True, help="Output root, e.g. F:\\tmp\\outs")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[50,100,200,400], help="Train normal counts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024,2025,2026], help="Random seeds")
    parser.add_argument("--ok_names", type=str, nargs="+", default=["OK","ok","good","normal"], help="Names considered normal")
    parser.add_argument("--ng_names", type=str, nargs="+", default=["NG","ng","anomaly","scratch","defect","bad"], help="Names considered anomaly (others also treated as NG if not OK)")
    parser.add_argument("--ignore_names", type=str, nargs="+", default=["建库数据","建库数据2","bank","bank_data","features"], help="Subdirs to ignore during splitting")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    # Collect OK / NG
    # If dataset_root directly contains images (no OK/NG subdirs), treat all as OK
    ok_candidates = []
    ng_candidates = []
    subdirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    ok_lower = {n.lower() for n in args.ok_names}
    ng_lower = {n.lower() for n in args.ng_names}
    ignore_lower = {n.lower() for n in args.ignore_names}
    if subdirs:
        for sub in subdirs:
            if sub.name.lower() in ignore_lower:
                print(f"[info] Ignoring auxiliary subdir {sub.name!r}")
                continue
            if sub.name.lower() in ok_lower:
                ok_candidates.extend(iter_images(sub))
            elif sub.name.lower() in ng_lower:
                ng_candidates.extend(iter_images(sub))
            else:
                # Unknown subdir: decide by name, default treat as NG if not OK
                print(f"[warn] Unknown subdir {sub.name!r} treated as NG")
                ng_candidates.extend(iter_images(sub))
        # also flat images directly under root: treat as OK (fallback)
        flat = [p for p in dataset_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if flat:
            print(f"[info] Found {len(flat)} flat images under root, merging into OK")
            ok_candidates.extend(sorted(flat, key=lambda p: str(p).lower()))
    else:
        ok_candidates = iter_images(dataset_root)

    ok_candidates = sorted(set(ok_candidates), key=lambda p: str(p).lower())
    ng_candidates = sorted(set(ng_candidates), key=lambda p: str(p).lower())
    print(f"[prepare_splits] dataset_root={dataset_root}")
    print(f"  OK: {len(ok_candidates)} images")
    print(f"  NG: {len(ng_candidates)} images")
    if not ok_candidates:
        raise RuntimeError(f"No OK images found in {dataset_root}")
    # also print samples to verify encoding
    if ok_candidates:
        print(f"  sample OK: {ok_candidates[0]}")
    if ng_candidates:
        print(f"  sample NG: {ng_candidates[0]}")

    splits_dir = outs_dir / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for n in args.train_sizes:
        if n > len(ok_candidates):
            print(f"[skip] train_size {n} > OK count {len(ok_candidates)}")
            continue
        for seed in args.seeds:
            rng = random.Random(seed)
            shuffled = ok_candidates.copy()
            rng.shuffle(shuffled)
            train_ok = sorted(shuffled[:n], key=lambda p: str(p).lower())
            test_ok = sorted(shuffled[n:], key=lambda p: str(p).lower())
            # test = remaining OK + all NG
            test_all = sorted(test_ok + ng_candidates, key=lambda p: str(p).lower())

            train_path = splits_dir / f"train_{n}_seed{seed}.txt"
            test_path = splits_dir / f"test_{n}_seed{seed}.txt"
            train_path.write_text("\n".join(str(p) for p in train_ok) + ("\n" if train_ok else ""), encoding="utf-8")
            test_path.write_text("\n".join(str(p) for p in test_all) + ("\n" if test_all else ""), encoding="utf-8")
            print(f"  -> {train_path.name}: {len(train_ok)} train | {test_path.name}: {len(test_all)} test (OK rem {len(test_ok)} + NG {len(ng_candidates)})")
            total_files += 2

    print(f"[done] Wrote {total_files} files to {splits_dir}")
    # also write a manifest json for analyze
    import json
    manifest = {
        "dataset_root": str(dataset_root),
        "ok_count": len(ok_candidates),
        "ng_count": len(ng_candidates),
        "train_sizes": args.train_sizes,
        "seeds": args.seeds,
        "splits_dir": str(splits_dir),
        "generated_files": total_files,
    }
    (splits_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  manifest: {splits_dir / 'manifest.json'}")

if __name__ == "__main__":
    main()
