"""Evaluate score maps for child directories containing ``images``/``masks``.

This entry point is independent of model inference.  It scans one data root::

    data_root/
    ├── good/
    │   └── images/
    └── class_a/
        ├── images/
        └── masks/

Any child directory may omit ``masks/``; it is treated as normal and assigned
an all-zero GT.  A child directory that has ``masks/`` uses those masks.

Score maps are searched recursively under one or more ``score_output_dir``
directories and matched by image stem.  Results are printed separately for
each child directory and written to ``metrics.csv``, ``metrics.json`` and
``pixel_metrics.csv`` under ``output_dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SHARED_UTILS = PROJECT_ROOT / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(1, str(SHARED_UTILS))

from score_workflow_common import (  # noqa: E402
    CLASSIFICATION_METRIC_NAMES,
    build_score_index,
    classification_metrics,
    find_mask,
    find_score,
    iter_data_directories,
    iter_images,
    load_score_map,
    pixel_f1_score_and_threshold,
    region_detection_metrics,
    report_metric_names,
    save_classification_threshold,
    select_optimal_threshold,
    write_metric_report,
)
from dinomaly_evaluation import (
    METRIC_NAMES,
    compute_evaluation_metrics,
    evaluate_pixel_metrics,
    training_image_score,
    write_per_image_pixel_metrics,
)

REPORT_METRIC_NAMES = report_metric_names(METRIC_NAMES)


def _load_mask(mask_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(mask > 0, dtype=np.uint8)


def _evaluate_directory(
    data_directory: Path,
    images_dir: Path,
    masks_dir: Path | None,
    score_index: Mapping[str, Sequence[Tuple[Path, Path]]],
    metric_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]], np.ndarray, np.ndarray]:
    gt_maps = []
    score_maps = []
    image_labels = []
    image_scores = []
    records: List[Dict[str, object]] = []
    image_paths = iter_images(images_dir)

    for image_path in tqdm(
        image_paths,
        desc=f"Evaluate {data_directory.name}",
        unit="image",
        dynamic_ncols=True,
    ):
        score_path = find_score(
            image_path,
            data_directory,
            score_index,
        )
        score_map = load_score_map(score_path)
        mask_path = find_mask(image_path, images_dir, masks_dir)
        gt_mask = (
            _load_mask(mask_path, score_map.shape)
            if mask_path is not None
            else np.zeros(score_map.shape, dtype=np.uint8)
        )
        resized_gt = cv2.resize(
            gt_mask,
            (metric_size, metric_size),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_score = cv2.resize(
            score_map,
            (metric_size, metric_size),
            interpolation=cv2.INTER_LINEAR,
        )
        image_label = bool(resized_gt.any())
        image_score = training_image_score(resized_score)
        gt_maps.append(resized_gt)
        score_maps.append(resized_score)
        image_labels.append(image_label)
        image_scores.append(image_score)
        records.append(
            {
                "directory": str(data_directory),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "score_path": str(score_path),
                "image_label": int(image_label),
                "image_score": image_score,
                "gt_positive_pixels": int(resized_gt.astype(bool).sum()),
                **evaluate_pixel_metrics(resized_gt, resized_score),
            }
        )

    gt_array = np.stack(gt_maps, axis=0)
    score_array = np.stack(score_maps, axis=0)
    metrics = compute_evaluation_metrics(
        image_scores,
        image_labels,
        score_array,
        gt_array,
    )
    return metrics, records, gt_array, score_array


def _write_results(
    results: Mapping[str, Mapping[str, float]],
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    write_metric_report(results, output_dir, METRIC_NAMES, "directory")

    write_per_image_pixel_metrics(records, output_dir / "pixel_metrics.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate score maps by data_root child directory. Each child "
            "must contain images/; masks/ is optional and omitted means "
            "normal. No model inference is run."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "目录示例（good 可省略 masks/）：\n"
            "  data_root/category_a/images/a.png\n"
            "  data_root/category_a/masks/a.png\n"
            "  data_root/category_b/images/b.png\n"
            "  data_root/category_b/masks/b.png\n\n"
            "score_output_dir 可重复或空格指定多个目录；脚本递归搜索"
            ".npy/.npz，并按图像 stem 匹配。"
        ),
    )
    parser.add_argument(
        "-i",
        "--data_root",
        required=True,
        type=Path,
        help="包含多个 images/ 和 masks/ 子目录的数据根目录。",
    )
    parser.add_argument(
        "-s",
        "--score_output_dir",
        "--score_dir",
        required=True,
        nargs="+",
        action="append",
        type=Path,
        help="一个或多个 score map 搜索目录；可空格分隔或重复指定。",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="评估结果目录；默认写入 data_root/evaluation_metrics。",
    )
    parser.add_argument(
        "-msz",
        "--metric_size",
        type=int,
        default=256,
        help="计算指标前统一缩放到的正方形边长（默认：256）。",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="图像判定阈值；不指定时在全部子目录上按最大平衡准确率自动选择。",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if args.metric_size < 1:
        raise ValueError("metric_size must be positive")

    score_output_dirs = []
    seen_score_dirs = set()
    for directory_group in args.score_output_dir:
        for directory in directory_group:
            directory = directory.expanduser().resolve()
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Score output directory does not exist: {directory}"
                )
            if directory not in seen_score_dirs:
                score_output_dirs.append(directory)
                seen_score_dirs.add(directory)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "evaluation_metrics"
    )
    data_directories = iter_data_directories(
        data_root,
        excluded_directories=[output_dir, *score_output_dirs],
    )
    score_index = build_score_index(score_output_dirs)
    results: Dict[str, Dict[str, float]] = {}
    all_records: List[Dict[str, object]] = []
    region_inputs = {}

    for data_directory, images_dir, masks_dir in data_directories:
        metrics, records, gt_maps, score_maps = _evaluate_directory(
            data_directory,
            images_dir,
            masks_dir,
            score_index,
            args.metric_size,
        )
        directory_key = str(data_directory)
        results[directory_key] = metrics
        all_records.extend(records)
        region_inputs[directory_key] = (gt_maps, score_maps, records)
    labels = np.asarray(
        [record["image_label"] for record in all_records], dtype=np.uint8
    )
    scores = np.asarray(
        [record["image_score"] for record in all_records], dtype=np.float32
    )
    if args.score_threshold is None:
        threshold, threshold_method, global_threshold_metrics = select_optimal_threshold(
            labels, scores
        )
    else:
        threshold = float(args.score_threshold)
        threshold_method = "manual"
        global_threshold_metrics = classification_metrics(labels, scores, threshold)

    _global_pixel_f1, global_pixel_threshold = pixel_f1_score_and_threshold(
        np.concatenate([item[0] for item in region_inputs.values()], axis=0),
        np.concatenate([item[1] for item in region_inputs.values()], axis=0),
    )
    effective_pixel_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else global_pixel_threshold
    )

    for directory, metrics in results.items():
        directory_records = [
            record for record in all_records if record["directory"] == directory
        ]
        directory_threshold_metrics = classification_metrics(
            [record["image_label"] for record in directory_records],
            [record["image_score"] for record in directory_records],
            threshold,
        )
        metrics.update(
            {
                name: directory_threshold_metrics[name]
                for name in CLASSIFICATION_METRIC_NAMES
            }
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

    _write_results(results, all_records, output_dir)
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
