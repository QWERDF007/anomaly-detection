"""Shared evaluation utilities for Dinomaly2 score maps.

The evaluator operates on already-generated score maps.  It is intentionally
independent of model construction and inference so score-map pipelines and a
standalone evaluation entry point can use exactly the same metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from skimage import measure
from tqdm import tqdm

from dinomaly_pipeline_common import load_ground_truth, load_score_map


METRIC_NAMES = (
    "I-AUROC",
    "I-AP",
    "I-F1",
    "P-AUROC",
    "P-AP",
    "P-F1",
    "P-AUPRO",
)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return float("nan")


def max_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
    except ValueError:
        return float("nan")
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
    return float(np.nanmax(f1))


def safe_aupro(masks: np.ndarray, scores: np.ndarray) -> float:
    if not np.any(masks):
        return float("nan")
    if float(scores.max()) <= float(scores.min()):
        return 0.0
    try:
        return float(compute_pro_fast(masks.astype(np.uint8), scores))
    except (AssertionError, ValueError, ZeroDivisionError):
        return float("nan")


def compute_pro_fast(
    masks: np.ndarray,
    amaps: np.ndarray,
    num_th: int = 200,
) -> float:
    """Compute PRO with the threshold sweep used by both pipelines."""

    masks = np.asarray(masks)
    amaps = np.asarray(amaps)
    if masks.ndim != 3 or amaps.ndim != 3 or masks.shape != amaps.shape:
        raise ValueError("masks and amaps must be equally shaped 3D arrays")
    if set(np.unique(masks).tolist()) != {0, 1}:
        raise AssertionError("masks must contain exactly 0 and 1")
    if not isinstance(num_th, int) or num_th <= 0:
        raise ValueError("num_th must be a positive integer")

    region_labels = []
    region_areas = []
    for mask in masks:
        labels = measure.label(mask)
        areas = np.bincount(labels.reshape(-1))[1:].astype(np.float64)
        region_labels.append(labels)
        region_areas.append(areas)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    if delta <= 0:
        return 0.0

    inverse_pixels = np.logical_not(masks.astype(bool))
    inverse_count = int(inverse_pixels.sum())
    total_regions = sum(len(areas) for areas in region_areas)
    if total_regions == 0 or inverse_count == 0:
        return float("nan")

    thresholds = np.arange(min_th, max_th, delta)
    pro_sums = np.zeros(thresholds.shape, dtype=np.float64)
    false_positive_counts = np.zeros(thresholds.shape, dtype=np.int64)

    for label_map, areas, amap, inverse_mask in tqdm(
        zip(region_labels, region_areas, amaps, inverse_pixels),
        total=len(amaps),
        desc="Compute PRO",
        unit="image",
        dynamic_ncols=True,
        leave=False,
    ):
        outside_values = np.sort(amap[inverse_mask])
        false_positive_counts += (
            outside_values.size
            - np.searchsorted(outside_values, thresholds, side="right")
        )
        for region_id, area in enumerate(areas, start=1):
            region_values = np.sort(amap[label_map == region_id])
            hits = region_values.size - np.searchsorted(
                region_values,
                thresholds,
                side="right",
            )
            pro_sums += hits / area

    pros = pro_sums / total_regions
    fprs = false_positive_counts / inverse_count
    valid = fprs < 0.3
    if not np.any(valid):
        return float("nan")
    fprs = fprs[valid]
    pros = pros[valid]
    max_fpr = fprs.max()
    if max_fpr <= 0:
        return float("nan")
    return float(auc(fprs / max_fpr, pros))


def _load_stage_score_map(sample: Dict, score_map_key: str) -> np.ndarray:
    try:
        source = sample[score_map_key]
    except KeyError as error:
        raise KeyError(
            f"Sample does not contain score-map field {score_map_key!r}."
        ) from error

    if isinstance(source, (str, Path)):
        return load_score_map(Path(source))

    score_map = np.asarray(source, dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Score map in {score_map_key!r} must be 2D; got {score_map.shape}"
        )
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def evaluate_stage(
    samples: Sequence[Dict],
    ground_truth_dir: Optional[Path],
    metric_size: int,
    score_map_key: str = "score_path",
    image_score_key: Optional[str] = None,
    stage_name: str = "score maps",
) -> Dict[str, float]:
    """Evaluate one score-map stage.

    ``score_map_key`` may point to a cached ``.npy`` path or an in-memory
    NumPy array.  If ``image_score_key`` is omitted, the image score is the
    maximum value of the loaded map; otherwise the named sample field is used.
    This supports raw cached maps as well as ROI-filtered in-memory maps.
    """

    if metric_size < 1:
        raise ValueError("metric_size must be positive")

    evaluation_samples = [
        sample
        for sample in samples
        if sample["group_key"] in {"test_good", "test_anomaly"}
    ]
    if not evaluation_samples:
        raise RuntimeError("No Test/good or Test/anomaly samples were found.")

    image_labels = []
    image_scores = []
    gt_pixels = []
    score_pixels = []
    with tqdm(
        evaluation_samples,
        desc=f"Evaluate {stage_name}",
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        for sample in progress:
            score_map = _load_stage_score_map(sample, score_map_key)
            gt_mask = load_ground_truth(
                sample,
                ground_truth_dir,
                score_map.shape,
            )
            image_labels.append(sample["group_key"] == "test_anomaly")
            if image_score_key is None:
                image_scores.append(float(score_map.max()))
            else:
                image_scores.append(float(sample[image_score_key]))

            gt_pixels.append(
                cv2.resize(
                    gt_mask,
                    (metric_size, metric_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            score_pixels.append(
                cv2.resize(
                    score_map,
                    (metric_size, metric_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            )

    image_labels_array = np.asarray(image_labels, dtype=np.uint8)
    image_scores_array = np.asarray(image_scores, dtype=np.float32)
    gt_pixels_array = np.stack(gt_pixels, axis=0)
    score_pixels_array = np.stack(score_pixels, axis=0)
    pixel_labels = gt_pixels_array.reshape(-1)
    pixel_scores = score_pixels_array.reshape(-1)
    return {
        "I-AUROC": safe_auroc(image_labels_array, image_scores_array),
        "I-AP": safe_ap(image_labels_array, image_scores_array),
        "I-F1": max_f1(image_labels_array, image_scores_array),
        "P-AUROC": safe_auroc(pixel_labels, pixel_scores),
        "P-AP": safe_ap(pixel_labels, pixel_scores),
        "P-F1": max_f1(pixel_labels, pixel_scores),
        "P-AUPRO": safe_aupro(gt_pixels_array, score_pixels_array),
    }


def print_and_save_metrics(
    results: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> None:
    """Save metrics as JSON/CSV and print the shared table format."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        str(stage): dict(metrics)
        for stage, metrics in results.items()
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
    with (output_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["stage"] + list(METRIC_NAMES),
        )
        writer.writeheader()
        for stage, metrics in result.items():
            writer.writerow(
                {
                    "stage": stage,
                    **{
                        name: metrics.get(name, float("nan"))
                        for name in METRIC_NAMES
                    },
                }
            )

    print("\nEvaluation metrics")
    print(
        "stage                         "
        + "  ".join(f"{name:>10}" for name in METRIC_NAMES)
    )
    for stage, metrics in result.items():
        values = "  ".join(
            f"{metrics.get(name, float('nan')):10.6f}"
            for name in METRIC_NAMES
        )
        print(f"{stage:<29}{values}")
    print()
