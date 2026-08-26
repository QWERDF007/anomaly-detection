#!/usr/bin/env python3
"""Generate hierarchical train/test splits for copper anomaly detection using sample_images_to_txt."""

import os
from pathlib import Path
import random

DATASET_ROOT = Path("/data/wt/ramdisk/铜色异常检测6相机")
OK_DIR = DATASET_ROOT / "OK"
NG_DIR = DATASET_ROOT / "NG"
SPLITS_DIR = Path("/data/wt/outs/data_splits")

SEEDS = [42, 100, 2024]
SAMPLE_SIZES = [50, 100, 200, 400]

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Scan all OK and NG images
    ok_images = sorted([p.resolve() for p in OK_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}])
    ng_images = sorted([p.resolve() for p in NG_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}])
    
    print(f"Total OK images found: {len(ok_images)}")
    print(f"Total NG images found: {len(ng_images)}")
    assert len(ok_images) >= 400, f"Insufficient OK images: {len(ok_images)}"
    assert len(ng_images) > 0, f"No NG images found: {len(ng_images)}"
    
    for seed in SEEDS:
        rng = random.Random(seed)
        # Sample 400 in random order
        sampled_400 = rng.sample(ok_images, 400)
        
        for n in SAMPLE_SIZES:
            train_ok = sampled_400[:n]
            train_ok_set = set(train_ok)
            test_ok = [p for p in ok_images if p not in train_ok_set]
            
            # Train file (label 0)
            train_file = SPLITS_DIR / f"train_{n}_seed{seed}.txt"
            with open(train_file, "w", encoding="utf-8") as f:
                for p in sorted(train_ok, key=lambda x: str(x).lower()):
                    f.write(f"{p} 0\n")
            
            # Test file (remaining OK: label 0, all NG: label 1)
            test_file = SPLITS_DIR / f"test_{n}_seed{seed}.txt"
            with open(test_file, "w", encoding="utf-8") as f:
                for p in sorted(test_ok, key=lambda x: str(x).lower()):
                    f.write(f"{p} 0\n")
                for p in sorted(ng_images, key=lambda x: str(x).lower()):
                    f.write(f"{p} 1\n")
            
            print(f"[Seed {seed}] N={n}: Train={len(train_ok)} (OK), Test={len(test_ok)} OK + {len(ng_images)} NG = {len(test_ok) + len(ng_images)} total")

if __name__ == "__main__":
    main()
