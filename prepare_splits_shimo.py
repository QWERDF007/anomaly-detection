import os
import json
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def render_mask(json_path, out_mask_path, img_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    h = data.get('imageHeight')
    w = data.get('imageWidth')
    if h is None or w is None or h <= 0 or w <= 0:
        with Image.open(img_path) as img:
            w, h = img.size
    
    mask = np.zeros((h, w), dtype=np.uint8)
    shapes = data.get('shapes', [])
    
    for s in shapes:
        st = (s.get('shape_type') or 'polygon').lower()
        pts = s.get('points', [])
        if not pts:
            continue
        pts_arr = np.array(pts, dtype=np.float32)
        if st == 'rectangle':
            if len(pts_arr) >= 2:
                x1, y1 = np.rint(pts_arr[0]).astype(np.int32)
                x2, y2 = np.rint(pts_arr[1]).astype(np.int32)
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
        elif st == 'polygon':
            if len(pts_arr) >= 3:
                int_pts = np.rint(pts_arr).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [int_pts], 255)
        elif st == 'circle':
            if len(pts_arr) >= 2:
                center = tuple(np.rint(pts_arr[0]).astype(np.int32))
                radius = int(round(np.linalg.norm(pts_arr[1] - pts_arr[0])))
                cv2.circle(mask, center, max(1, radius), 255, -1)
        elif st in {'line', 'linestrip'}:
            if len(pts_arr) >= 2:
                int_pts = np.rint(pts_arr).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(mask, [int_pts], isClosed=False, color=255, thickness=5)
        elif st == 'point':
            center = tuple(np.rint(pts_arr[0]).astype(np.int32))
            cv2.circle(mask, center, 5, 255, -1)
        else:
            if len(pts_arr) >= 3:
                int_pts = np.rint(pts_arr).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [int_pts], 255)

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask_path), mask)
    return out_mask_path

def process_shimo_subdir(sub_path, sub_name, out_dir, target_train_n=200, seed=2024):
    sub = Path(sub_path)
    splits_dir = out_dir / 'data_splits'
    gt_dir = out_dir / 'ground_truth' / sub_name
    splits_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"Processing {sub_name} from {sub} ...")
    print(f"=======================================================")
    
    imgs = sorted([f for f in (sub / 'images').iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    ann_dir = sub / 'annotations'
    
    ok_items = []
    ng_items = []
    
    for idx, img in enumerate(imgs):
        json_file = ann_dir / (img.stem + '.json')
        is_ng = False
        if json_file.is_file():
            try:
                with open(json_file, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                shapes = data.get('shapes', [])
                if len(shapes) > 0:
                    is_ng = True
            except Exception as e:
                print(f"Warning reading {json_file}: {e}")
                is_ng = False
        
        if is_ng:
            mask_p = gt_dir / f"{img.stem}_mask.png"
            if not mask_p.is_file():
                render_mask(json_file, mask_p, img)
            ng_items.append((img, mask_p, 1))
        else:
            ok_items.append((img, None, 0))
            
    print(f"[{sub_name}] Total OK: {len(ok_items)}, Total NG: {len(ng_items)}")
    
    # Deterministic shuffle for reproducible train/test sampling
    rng = random.Random(seed)
    rng.shuffle(ok_items)
    
    n_train = min(target_train_n, len(ok_items))
    train_ok = ok_items[:n_train]
    test_ok = ok_items[n_train:]
    test_ng = ng_items
    
    print(f"[{sub_name}] Selected Train OK (N={n_train})")
    print(f"[{sub_name}] Test Set: {len(test_ok)} OK, {len(test_ng)} NG (Total {len(test_ok) + len(test_ng)})")
    
    # Write train split: image_path mask_path label
    train_file = splits_dir / f"{sub_name}_train_n{n_train}.txt"
    with open(train_file, 'w', encoding='utf-8') as fp:
        for img, mask_p, lbl in train_ok:
            fp.write(f"{img.resolve()} {'None' if mask_p is None else mask_p.resolve()} {lbl}\n")
            
    # Write test split
    test_file = splits_dir / f"{sub_name}_test.txt"
    with open(test_file, 'w', encoding='utf-8') as fp:
        for img, mask_p, lbl in test_ok:
            fp.write(f"{img.resolve()} {'None' if mask_p is None else mask_p.resolve()} {lbl}\n")
        for img, mask_p, lbl in test_ng:
            fp.write(f"{img.resolve()} {'None' if mask_p is None else mask_p.resolve()} {lbl}\n")
            
    print(f"[{sub_name}] Saved train split to: {train_file}")
    print(f"[{sub_name}] Saved test split to: {test_file}")

def main():
    root = Path("/data/wt/data/shimo-all")
    out_dir = Path("/data/wt/exp_shimo")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for sub_name in ['ori-10', 'ori-21']:
        process_shimo_subdir(root / sub_name, sub_name, out_dir, target_train_n=200, seed=2024)
        
    print("\n[SUCCESS] All shimo-all splits and ground truth masks generated successfully!")

if __name__ == '__main__':
    main()
