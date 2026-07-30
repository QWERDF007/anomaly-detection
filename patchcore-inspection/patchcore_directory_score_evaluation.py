"""Evaluate cached PatchCore score maps by child directory, without inference.

``data_root`` contains child directories with ``images/`` and optional
``masks/``.  Any child without ``masks/`` is treated as a normal class.  Score
maps are searched recursively in one or more output directories and matched
by image filename stem.

Image scores are always calculated from the score map's highest 1% pixels,
which is the Dinomaly2 training definition used by both model families.  No
sidecar metadata is read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PROJECT_ROOT = ROOT.parent
SHARED_UTILS = PROJECT_ROOT / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS))

from score_workflow_common import (  # noqa: E402
    build_score_index,
    find_mask,
    find_score,
    iter_data_directories,
    iter_images,
    load_score_map as common_load_score_map,
    pixel_f1_score_and_threshold,
    region_detection_metrics,
    save_classification_threshold,
    write_metric_report,
)

from patchcore_evaluation import (
    CLASSIFICATION_METRIC_NAMES,
    METRIC_NAMES,
    REPORT_METRIC_NAMES,
    classification_metrics,
    compute_evaluation_metrics,
    evaluate_pixel_metrics,
    select_optimal_threshold,
    training_image_score,
    write_per_image_pixel_metrics,
)
from patchcore.datasets.custom import get_data_transforms

def _mask_transform(args):
    """Return PatchCore's optional GT transform from explicit CLI settings."""

    resize = args.resize
    imagesize = args.imagesize
    try:
        resize = int(resize) if resize is not None else None
        imagesize = int(imagesize) if imagesize is not None else None
    except (TypeError, ValueError):
        return None
    if resize is None or imagesize is None or resize < 1 or imagesize < 1:
        return None
    _image_transform, mask_transform = get_data_transforms(resize, imagesize)
    return mask_transform


def _load_mask(mask_path: Path, shape: Tuple[int, int], mask_transform) -> np.ndarray:
    if mask_transform is not None:
        mask = np.squeeze(mask_transform(Image.open(mask_path).convert("L")).numpy() > 0)
        mask = mask.astype(np.uint8)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Cannot read mask: {mask_path}")
        mask = (mask > 0).astype(np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _resize_for_metrics(score_map: np.ndarray, gt_mask: np.ndarray, metric_size: Optional[int]):
    if metric_size is None:
        return score_map, gt_mask
    size = (metric_size, metric_size)
    return (
        cv2.resize(score_map, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(gt_mask, size, interpolation=cv2.INTER_NEAREST),
    )


def _evaluate_directory(
    data_directory: Path,
    images_dir: Path,
    masks_dir: Optional[Path],
    score_index: Mapping[str, Sequence[Tuple[Path, Path]]],
    args,
) -> Tuple[Dict[str, float], List[Dict[str, object]], np.ndarray, np.ndarray]:
    gt_maps = []
    score_maps = []
    image_labels = []
    image_scores = []
    records: List[Dict[str, object]] = []
    for image_path in tqdm(
        iter_images(images_dir), desc=f"Evaluate {data_directory.name}", unit="image", dynamic_ncols=True
    ):
        score_path = find_score(image_path, data_directory, score_index)
        score_map = common_load_score_map(score_path)
        mask_path = find_mask(image_path, images_dir, masks_dir)
        mask_transform = _mask_transform(args)
        gt_mask = (
            _load_mask(mask_path, score_map.shape, mask_transform)
            if mask_path is not None
            else np.zeros(score_map.shape, dtype=np.uint8)
        )
        score_map, gt_mask = _resize_for_metrics(score_map, gt_mask, args.metric_size)
        image_label = int(gt_mask.any())
        image_score = training_image_score(score_map)
        pixel_metrics = evaluate_pixel_metrics(gt_mask, score_map)
        gt_maps.append(gt_mask)
        score_maps.append(score_map)
        image_labels.append(image_label)
        image_scores.append(image_score)
        records.append(
            {
                "directory": str(data_directory),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "score_path": str(score_path),
                "image_label": image_label,
                "image_score": image_score,
                "gt_positive_pixels": int(gt_mask.sum()),
                **{name: pixel_metrics[name] for name in pixel_metrics if name.startswith("P-")},
            }
        )
    gt_array = np.stack(gt_maps)
    score_array = np.stack(score_maps)
    return (
        compute_evaluation_metrics(image_scores, image_labels, score_array, gt_array),
        records,
        gt_array,
        score_array,
    )


def _write_results(
    results: Mapping[str, Mapping[str, float]], records: Sequence[Mapping[str, object]], output_dir: Path
) -> None:
    write_metric_report(results, output_dir, METRIC_NAMES, "directory")
    write_per_image_pixel_metrics(records, output_dir / "pixel_metrics.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PatchCore score maps by data_root child directory; no inference is run.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "每个 data_root 子目录必须有 images/，masks/ 可省略，省略即正常样本。\n"
            "score_output_dir 可指定一个或多个目录，脚本递归查找 .npy/.npz 并按图像 stem 匹配。\n"
            "图像分数统一按分数图最高 1% 像素均值计算"
        ),
    )
    parser.add_argument("-i", "--data_root", type=Path, required=True)
    parser.add_argument("-s", "--score_output_dir", "--score_dir", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=None)
    parser.add_argument("-imgsz", dest="resize", type=int, default=None, help="可选的 GT 变换 Resize；需同时指定 -csz。")
    parser.add_argument("-csz", dest="imagesize", type=int, default=None, help="可选的 GT 变换 CenterCrop。")
    parser.add_argument("--metric_size", type=int, default=256, help="计算指标前统一缩放到的正方形边长（默认：256）。")
    parser.add_argument("--score_threshold", type=float, default=None, help="图像判定阈值；不指定时在所有子目录上按最大平衡准确率自动选择。")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric_size is not None and args.metric_size < 1:
        raise ValueError("metric_size must be positive")
    if (args.resize is None) != (args.imagesize is None):
        raise ValueError("-imgsz and -csz must be given together")
    if args.resize is not None and (args.resize < 1 or args.imagesize < 1):
        raise ValueError("resize and imagesize must be positive")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    score_dirs = []
    seen = set()
    for group in args.score_output_dir:
        for directory in group:
            directory = directory.expanduser().resolve()
            if not directory.is_dir():
                raise FileNotFoundError(f"Score output directory does not exist: {directory}")
            if directory not in seen:
                score_dirs.append(directory)
                seen.add(directory)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else data_root / "evaluation_metrics"
    directories = iter_data_directories(data_root, excluded_directories=[output_dir, *score_dirs])
    score_index = build_score_index(score_dirs)
    results: Dict[str, Dict[str, float]] = {}
    records: List[Dict[str, object]] = []
    region_inputs = {}
    for data_directory, images_dir, masks_dir in directories:
        metrics, directory_records, gt_maps, score_maps = _evaluate_directory(data_directory, images_dir, masks_dir, score_index, args)
        directory_key = str(data_directory)
        results[directory_key] = metrics
        records.extend(directory_records)
        region_inputs[directory_key] = (gt_maps, score_maps, directory_records)

    labels = np.asarray([record["image_label"] for record in records], dtype=np.uint8)
    scores = np.asarray([record["image_score"] for record in records], dtype=np.float32)
    if args.score_threshold is None:
        threshold, threshold_method, global_threshold_metrics = select_optimal_threshold(
            labels, scores
        )
    else:
        threshold = float(args.score_threshold)
        threshold_method = "manual"
        global_threshold_metrics = classification_metrics(labels, scores, threshold)

    _global_pixel_f1, global_pixel_threshold = pixel_f1_score_and_threshold(
        np.concatenate([item[0].reshape(-1) for item in region_inputs.values()]),
        np.concatenate([item[1].reshape(-1) for item in region_inputs.values()]),
    )
    effective_pixel_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else global_pixel_threshold
    )

    for directory, metrics in results.items():
        directory_records = [
            record for record in records if record["directory"] == directory
        ]
        directory_metrics = classification_metrics(
            [record["image_label"] for record in directory_records],
            [record["image_score"] for record in directory_records],
            threshold,
        )
        metrics.update(
            {name: directory_metrics[name] for name in CLASSIFICATION_METRIC_NAMES}
        )
        gt_maps, score_maps, directory_records = region_inputs[directory]
        metrics.update(
            region_detection_metrics(
                gt_maps,
                score_maps,
                effective_pixel_threshold,
                directory_records,
                p_f1_threshold=global_pixel_threshold,
            )
        )
        print(f"\n===== {directory} =====", flush=True)
        print(
            "  "
            + "  ".join(
                f"{name}={metrics.get(name, float('nan')):.6f}"
                for name in REPORT_METRIC_NAMES
            ),
            flush=True,
        )
    _write_results(results, records, output_dir)
    save_classification_threshold(
        output_dir / "classification_threshold.json",
        threshold,
        threshold_method,
        global_threshold_metrics,
    )
    print(
        f"\nImage threshold={threshold:.6f} ({threshold_method}); "
        f"FPR={global_threshold_metrics['FPR']:.6f}, "
        f"TNR={global_threshold_metrics['TNR']:.6f}, "
        f"Accuracy={global_threshold_metrics['Accuracy']:.6f}",
        flush=True,
    )
    print(f"\nMetrics written to {output_dir / 'metrics.csv'}")
    print(f"Per-image pixel metrics written to {output_dir / 'pixel_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
