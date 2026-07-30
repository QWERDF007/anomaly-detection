"""Evaluate cached anomaly score maps by ``images/`` and optional ``masks/``.

This is a model-agnostic, no-inference entry point.  It evaluates only cached
``.npy``/``.npz`` score maps, so it can be used for both Dinomaly2 and
PatchCore (or any other model producing a two-dimensional anomaly map).

Dataset layout::

    data_root/
    ├── normal/images/
    └── defect/
        ├── images/
        └── masks/

Every direct child must provide ``images/``.  A child without ``masks/`` is a
normal class.  Score maps are found by filename stem below one or more score
directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from anomaly_evaluation import (
    CLASSIFICATION_METRIC_NAMES,
    METRIC_NAMES,
    REPORT_METRIC_NAMES,
    classification_metrics,
    compute_evaluation_metrics,
    evaluate_pixel_metrics,
    pixel_f1_score_and_threshold,
    region_detection_metrics,
    select_optimal_threshold,
    training_image_score,
    write_per_image_pixel_metrics,
)
from score_workflow_common import (
    build_score_index,
    find_mask,
    find_score,
    iter_data_directories,
    iter_images,
    load_score_map,
    save_classification_threshold,
    write_metric_report,
)


def _load_mask(mask_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"Cannot read mask: {mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


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

    for image_path in tqdm(
        iter_images(images_dir),
        desc=f"Evaluate {data_directory.name}",
        unit="image",
        dynamic_ncols=True,
    ):
        score_path = find_score(image_path, data_directory, score_index)
        score_map = load_score_map(score_path)
        mask_path = find_mask(image_path, images_dir, masks_dir)
        gt_mask = (
            _load_mask(mask_path, score_map.shape)
            if mask_path is not None
            else np.zeros(score_map.shape, dtype=np.uint8)
        )
        resized_score = cv2.resize(
            score_map,
            (metric_size, metric_size),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_gt = cv2.resize(
            gt_mask,
            (metric_size, metric_size),
            interpolation=cv2.INTER_NEAREST,
        )
        image_label = int(np.any(resized_gt))
        image_score = training_image_score(resized_score)
        pixel_metrics = evaluate_pixel_metrics(resized_gt, resized_score)

        gt_maps.append((resized_gt > 0).astype(np.uint8))
        score_maps.append(resized_score)
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
                "gt_positive_pixels": int(np.asarray(resized_gt, dtype=bool).sum()),
                **pixel_metrics,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate cached .npy/.npz anomaly maps by data-root child directory; no inference is run.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Each data_root child needs images/ and may omit masks/ (normal class).\n"
            "score_output_dir accepts one or more directories and recursively matches .npy/.npz by image stem."
        ),
    )
    parser.add_argument("-i", "--data_root", required=True, type=Path)
    parser.add_argument(
        "-s",
        "--score_output_dir",
        "--score_dir",
        required=True,
        nargs="+",
        action="append",
        type=Path,
        help="One or more score-map roots; repeat the option or separate paths with spaces.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="Report directory (default: data_root/evaluation_metrics).",
    )
    parser.add_argument(
        "-msz",
        "--metric_size",
        type=int,
        default=256,
        help="Square metric resolution (default: 256).",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Manual image/pixel threshold; default chooses the image threshold by max balanced accuracy.",
    )
    return parser


def _score_directories(values: Sequence[Sequence[Path]]) -> List[Path]:
    directories = []
    seen = set()
    for group in values:
        for directory in group:
            directory = directory.expanduser().resolve()
            if not directory.is_dir():
                raise FileNotFoundError(f"Score output directory does not exist: {directory}")
            if directory not in seen:
                directories.append(directory)
                seen.add(directory)
    return directories


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric_size < 1:
        raise ValueError("metric_size must be positive")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    score_directories = _score_directories(args.score_output_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "evaluation_metrics"
    )
    data_directories = iter_data_directories(
        data_root,
        excluded_directories=[output_dir, *score_directories],
    )
    score_index = build_score_index(score_directories)

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
        key = str(data_directory)
        results[key] = metrics
        all_records.extend(records)
        region_inputs[key] = (gt_maps, score_maps, records)

    labels = np.asarray([record["image_label"] for record in all_records], dtype=np.uint8)
    scores = np.asarray([record["image_score"] for record in all_records], dtype=np.float32)
    if args.score_threshold is None:
        threshold, threshold_method, global_threshold_metrics = select_optimal_threshold(
            labels, scores
        )
    else:
        threshold = float(args.score_threshold)
        threshold_method = "manual"
        global_threshold_metrics = classification_metrics(labels, scores, threshold)

    _global_pixel_f1, pixel_f1_threshold = pixel_f1_score_and_threshold(
        np.concatenate([item[0] for item in region_inputs.values()], axis=0),
        np.concatenate([item[1] for item in region_inputs.values()], axis=0),
    )
    effective_pixel_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else pixel_f1_threshold
    )
    for directory, metrics in results.items():
        directory_records = [record for record in all_records if record["directory"] == directory]
        threshold_metrics = classification_metrics(
            [record["image_label"] for record in directory_records],
            [record["image_score"] for record in directory_records],
            threshold,
        )
        metrics.update({name: threshold_metrics[name] for name in CLASSIFICATION_METRIC_NAMES})
        gt_maps, score_maps, records = region_inputs[directory]
        metrics.update(
            region_detection_metrics(
                gt_maps,
                score_maps,
                effective_pixel_threshold,
                records,
                p_f1_threshold=pixel_f1_threshold,
            )
        )
        print(f"\n===== {directory} =====", flush=True)
        print(
            "  ".join(
                f"{name}={metrics.get(name, float('nan')):.6f}"
                for name in REPORT_METRIC_NAMES
            ),
            flush=True,
        )

    write_metric_report(results, output_dir, METRIC_NAMES, "directory")
    write_per_image_pixel_metrics(all_records, output_dir / "pixel_metrics.csv")
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
