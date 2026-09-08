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
            # fallback polygon
            if len(pts_arr) >= 3:
                int_pts = np.rint(pts_arr).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [int_pts], 255)

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask_path), mask)
    return out_mask_path

def process_dataset(root_path, ds_name, out_dir, target_train_n=200, seed=2024):
    root = Path(root_path)
    splits_dir = out_dir / 'data_splits'
    gt_dir = out_dir / 'ground_truth' / ds_name
    splits_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {ds_name} from {root}...")
    
    # 1. Gather all image directories
    all_img_dirs = []
    for p in root.rglob('*'):
        if p.is_dir() and any(k in p.name.lower() for k in ['train', 'val', 'test']):
            if 'label' not in p.name.lower():
                all_img_dirs.append(p)
                
    train_ok_imgs = []
    train_ng_imgs = []
    eval_ok_imgs = []
    eval_ng_imgs = []
    
    mask_cache = {}
    
    for d in sorted(all_img_dirs):
        is_train = 'train' in d.name.lower()
        
        # Locate corresponding label dir
        parent = d.parent
        lbl_name = d.name.replace('Image', 'Label').replace('image', 'label')
        lbl_dir = parent / lbl_name
        if not lbl_dir.is_dir():
            cands = [x for x in parent.iterdir() if x.is_dir() and x.name.lower() == lbl_name.lower()]
            lbl_dir = cands[0] if cands else None
            
        imgs = sorted([f for f in d.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        
        for img in imgs:
            json_file = None
            if lbl_dir and lbl_dir.is_dir():
                cand = lbl_dir / (img.stem + '.json')
                if cand.is_file():
                    json_file = cand
            if not json_file:
                cand = d / (img.stem + '.json')
                if cand.is_file():
                    json_file = cand
            
            is_ng = False
            if json_file and json_file.is_file():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if len(meta.get('shapes', [])) > 0:
                        is_ng = True
                except:
                    is_ng = False
            
            if is_ng:
                # Render mask
                rel_stem = img.relative_to(root).with_suffix('.png')
                mask_p = gt_dir / rel_stem
                if not mask_p.is_file():
                    render_mask(json_file, mask_p, img)
                mask_cache[str(img)] = str(mask_p)
                
                if is_train:
                    train_ng_imgs.append((str(img), str(mask_p)))
                else:
                    eval_ng_imgs.append((str(img), str(mask_p)))
            else:
                if is_train:
                    train_ok_imgs.append(str(img))
                else:
                    eval_ok_imgs.append(str(img))
                    
    print(f"[{ds_name}] Raw Train: {len(train_ok_imgs)} OK, {len(train_ng_imgs)} NG")
    print(f"[{ds_name}] Raw Eval (Val+Test): {len(eval_ok_imgs)} OK, {len(eval_ng_imgs)} NG")
    
    # Sample training OK
    rng = random.Random(seed)
    train_ok_shuffled = list(train_ok_imgs)
    rng.shuffle(train_ok_shuffled)
    
    actual_train_n = min(len(train_ok_shuffled), target_train_n)
    selected_train_ok = train_ok_shuffled[:actual_train_n]
    remaining_train_ok = train_ok_shuffled[actual_train_n:]
    
    # Construct Test split:
    # Always include eval NG
    test_lines = []
    
    if len(remaining_train_ok) > 0:
        # We have plenty of OK from train, use remaining OK + eval OK
        test_ok = eval_ok_imgs + remaining_train_ok
    else:
        # Train OK was <= target_train_n (all used for training)
        # Include eval OK plus train OK in test to ensure statistically valid ROC/FPR evaluation
        test_ok = eval_ok_imgs + selected_train_ok
        
    for ok_p in sorted(test_ok):
        test_lines.append(f"{ok_p} 0\n")
        
    for ng_p, mask_p in sorted(eval_ng_imgs):
        test_lines.append(f'"{ng_p}","{mask_p}",bad\n')
        
    train_lines = [f"{p} 0\n" for p in sorted(selected_train_ok)]
    
    train_txt_path = splits_dir / f"{ds_name}_train_n{actual_train_n}.txt"
    test_txt_path = splits_dir / f"{ds_name}_test.txt"
    
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
        
    with open(test_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
        
    print(f"[{ds_name}] Generated {train_txt_path.name}: {len(train_lines)} train OK images")
    print(f"[{ds_name}] Generated {test_txt_path.name}: {len(test_lines)} test images ({len(test_ok)} OK, {len(eval_ng_imgs)} NG)")
    return actual_train_n, train_txt_path, test_txt_path

if __name__ == '__main__':
    out_dir = Path('/data/wt/exp0903')
    vc_n, vc_train, vc_test = process_dataset('/data/wt/data/VC训练数据导出', 'VC', out_dir, target_train_n=200, seed=2024)
    zs_n, zs_train, zs_test = process_dataset('/data/wt/data/中石训练数据导出', '中石', out_dir, target_train_n=200, seed=2024)
