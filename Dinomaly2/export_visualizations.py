"""Export side-by-side PNG visualizations for Dinomaly2 predictions.
Generates: [Original Image] | [GT Mask (if any)] | [Color Heatmap Overlay]
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from dinomaly_two_stage import load_mask

def main():
    parser = argparse.ArgumentParser(description="Export PNG visualizations")
    parser.add_argument("--root", default="/data/wt/two_stages/base_672_15k", help="Two-stage prediction root")
    parser.add_argument("--data_root", default="/data/wt/ramdisk/leishi_026/test", help="Test images root")
    parser.add_argument("--ground_truth", default="/data/wt/ramdisk/leishi_026/ground_truth", help="GT root")
    parser.add_argument("--out_dir", default=None, help="Output directory for PNGs")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to export")
    args = parser.parse_args()

    root = Path(args.root)
    data_root = Path(args.data_root)
    gt_root = Path(args.ground_truth)
    out_dir = Path(args.out_dir) if args.out_dir else root / "preds" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    details_dir = root / "preds" / "details"
    score_maps_dir = root / "preds" / "score_maps"
    detail_files = sorted(list(details_dir.rglob("*.json")))

    if not detail_files:
        print(f"No detail files found in {details_dir}")
        return

    print(f"Exporting visualizations to: {out_dir}")
    exported = 0
    for df in tqdm(detail_files):
        try:
            d = json.loads(df.read_text())
        except Exception:
            continue
            
        img_rel = Path(d["image_relative"])
        img_path = data_root / img_rel
        if not img_path.is_file():
            # Search candidate paths
            candidates = list(data_root.rglob(img_rel.name))
            if candidates:
                img_path = candidates[0]
            else:
                continue

        score_path = score_maps_dir / img_rel.with_suffix(".npy")
        if not score_path.is_file():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        score_map = np.load(score_path).astype(np.float32)
        score_map_resized = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize score map to 0-255 for Jet Colormap
        norm_score = np.clip(score_map_resized / 0.035 * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_score, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        # Draw binary contour if > 0.014
        binary = (score_map_resized >= 0.014).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # Load GT Mask if present
        gt_img = np.zeros((h, w, 3), dtype=np.uint8)
        for ext in [".png", ".jpg", ".npy", ".json"]:
            gp = gt_root / img_rel.with_suffix(ext)
            if gp.is_file():
                try:
                    gt_mask = load_mask(gp, (h, w))
                    gt_img[gt_mask > 0] = [0, 255, 0]
                    cv2.drawContours(img_bgr, cv2.findContours((gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
                except Exception:
                    pass
                break

        # Put Label & Score text
        final_lbl = d.get("final_label", "unknown")
        raw_s = d.get("raw_score", 0.0)
        adj_s = d.get("adjusted_score", raw_s)
        color = (0, 0, 255) if final_lbl != "good" else (0, 255, 0)
        cv2.putText(overlay, f"{final_lbl.upper()} | Score: {adj_s:.4f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Combine into side-by-side: [Original+GT] | [Heatmap Overlay]
        combined = np.hstack([img_bgr, overlay])
        save_path = out_dir / f"{img_rel.parent.name}_{img_rel.stem}.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), combined)

        exported += 1
        if args.max_samples > 0 and exported >= args.max_samples:
            break

    print(f"Successfully exported {exported} visualization images to {out_dir}")

if __name__ == "__main__":
    main()
