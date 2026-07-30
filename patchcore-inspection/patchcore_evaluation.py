"""Shared PatchCore-compatible metrics for cached anomaly score maps.

The formulas in this module are deliberately the same as ``train.py``.  In
particular, image-level metrics consume PatchCore's native image score rather
than a value derived from the anomaly map, because those two values are not
generally identical.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from skimage import measure
from sklearn import metrics as sklearn_metrics


METRIC_NAMES = (
    "I-AUROC",
    "I-AP",
    "I-F1",
    "P-AUROC",
    "P-AP",
    "P-F1",
    "P-AUPRO",
)

PIXEL_METRIC_NAMES = ("P-AUROC", "P-AP", "P-F1", "P-AUPRO")


def safe_auroc(labels, scores) -> float:
    """Return the AUROC calculation used by the PatchCore trainer."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) == 0 or len(labels) != len(scores) or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(sklearn_metrics.roc_auc_score(labels, scores))
    except (ValueError, RuntimeError):
        return float("nan")


def safe_average_precision(labels, scores) -> float:
    """Return the average-precision calculation used by the trainer."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) == 0 or len(labels) != len(scores) or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(sklearn_metrics.average_precision_score(labels, scores))
    except (ValueError, RuntimeError):
        return float("nan")


def safe_f1_max(labels, scores) -> float:
    """Return the maximum PR-curve F1 calculation used by the trainer."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) == 0 or len(labels) != len(scores) or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        precision, recall, _ = sklearn_metrics.precision_recall_curve(labels, scores)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
        # ``precision_recall_curve`` has one endpoint without a threshold.
        if len(f1) > 1:
            f1 = f1[:-1]
        return float(np.nanmax(f1))
    except (ValueError, RuntimeError):
        return float("nan")


def prepare_pixel_arrays(segmentations, masks):
    """Normalize PatchCore score-map and mask containers to ``[N,H,W]``."""

    segmentations = np.asarray(segmentations, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if segmentations.size == 0 or masks.size == 0:
        return None, None

    if segmentations.ndim == 4 and segmentations.shape[1] == 1:
        segmentations = segmentations[:, 0]
    if masks.ndim == 4:
        masks = masks[:, 0] if masks.shape[1] == 1 else masks.max(axis=1)
    if segmentations.ndim != 3 or masks.ndim != 3:
        return None, None
    if segmentations.shape != masks.shape:
        return None, None
    return segmentations, (masks > 0).astype(np.uint8)


def compute_pro(masks, anomaly_maps, num_thresholds: int = 200) -> float:
    """Compute P-AUPRO with exactly the training threshold sweep."""

    if masks is None or anomaly_maps is None:
        return float("nan")
    if masks.ndim != 3 or anomaly_maps.ndim != 3 or masks.shape != anomaly_maps.shape:
        return float("nan")
    if not np.any(masks) or np.all(masks):
        return float("nan")

    min_score = float(anomaly_maps.min())
    max_score = float(anomaly_maps.max())
    if not np.isfinite(min_score) or not np.isfinite(max_score) or max_score <= min_score:
        return float("nan")

    thresholds = np.linspace(min_score, max_score, num_thresholds, endpoint=False)
    pros = []
    fprs = []
    background = masks == 0
    background_pixels = int(background.sum())
    if background_pixels == 0:
        return float("nan")

    regions_per_image = [
        measure.regionprops(measure.label(mask.astype(np.uint8))) for mask in masks
    ]
    if not any(regions_per_image):
        return float("nan")

    for threshold in thresholds:
        binary_maps = anomaly_maps > threshold
        region_overlaps = []
        for binary_map, regions in zip(binary_maps, regions_per_image):
            for region in regions:
                coords = region.coords
                region_overlaps.append(
                    float(binary_map[coords[:, 0], coords[:, 1]].sum()) / region.area
                )
        if not region_overlaps:
            continue

        false_positive_rate = float(
            np.logical_and(background, binary_maps).sum()
        ) / background_pixels
        if false_positive_rate <= 0.3:
            pros.append(float(np.mean(region_overlaps)))
            fprs.append(false_positive_rate)

    if len(fprs) < 2 or max(fprs) <= 0:
        return float("nan")

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]
    return float(sklearn_metrics.auc(fprs / fprs.max(), pros))


def compute_evaluation_metrics(scores, labels, segmentations, masks) -> Dict[str, float]:
    """Calculate all image/pixel metrics with the trainer's definitions."""

    image_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    image_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    anomaly_maps, gt_masks = prepare_pixel_arrays(segmentations, masks)

    if anomaly_maps is None or gt_masks is None:
        pixel_scores = np.asarray([], dtype=np.float32)
        pixel_labels = np.asarray([], dtype=np.uint8)
        p_aupro = float("nan")
    else:
        pixel_scores = anomaly_maps.reshape(-1)
        pixel_labels = gt_masks.reshape(-1)
        p_aupro = compute_pro(gt_masks, anomaly_maps)

    return {
        "I-AUROC": safe_auroc(image_labels, image_scores),
        "I-AP": safe_average_precision(image_labels, image_scores),
        "I-F1": safe_f1_max(image_labels, image_scores),
        "P-AUROC": safe_auroc(pixel_labels, pixel_scores),
        "P-AP": safe_average_precision(pixel_labels, pixel_scores),
        "P-F1": safe_f1_max(pixel_labels, pixel_scores),
        "P-AUPRO": p_aupro,
    }


def evaluate_pixel_metrics(gt_mask, score_map) -> Dict[str, float]:
    """Calculate the PatchCore-compatible pixel metrics for one image."""

    gt_mask = np.asarray(gt_mask)
    score_map = np.asarray(score_map, dtype=np.float32)
    if gt_mask.shape != score_map.shape or gt_mask.ndim != 2:
        raise ValueError(
            "Per-image pixel metrics need equally shaped 2D arrays; got "
            f"{gt_mask.shape} and {score_map.shape}."
        )
    metrics = compute_evaluation_metrics(
        scores=np.asarray([0.0], dtype=np.float32),
        labels=np.asarray([0], dtype=np.uint8),
        segmentations=score_map[None, ...],
        masks=(gt_mask > 0).astype(np.uint8)[None, ...],
    )
    return {name: metrics[name] for name in PIXEL_METRIC_NAMES}


def load_score_map(score_path: Path) -> np.ndarray:
    """Load a 2D ``.npy``/``.npz`` score map and make non-finite values safe."""

    score_path = Path(score_path)
    if score_path.suffix.lower() == ".npz":
        archive = np.load(score_path)
        try:
            if not archive.files:
                raise ValueError(f"Score archive is empty: {score_path}")
            score_map = np.asarray(archive[archive.files[0]], dtype=np.float32)
        finally:
            archive.close()
    else:
        score_map = np.asarray(np.load(score_path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(f"Score map must be 2D: {score_path}; got {score_map.shape}")
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def write_metrics(
    results: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    label_name: str = "stage",
) -> None:
    """Print and save a metric table in JSON and CSV form."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {str(label): dict(metrics) for label, metrics in results.items()}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[label_name, *METRIC_NAMES])
        writer.writeheader()
        for label, metrics in result.items():
            writer.writerow(
                {
                    label_name: label,
                    **{name: metrics.get(name, float("nan")) for name in METRIC_NAMES},
                }
            )

    print("\nEvaluation metrics")
    print(f"{label_name:<29}" + "  ".join(f"{name:>10}" for name in METRIC_NAMES))
    for label, metrics in result.items():
        values = "  ".join(
            f"{metrics.get(name, float('nan')):10.6f}" for name in METRIC_NAMES
        )
        print(f"{label:<29}{values}")
    print()


def write_per_image_pixel_metrics(
    records: Sequence[Mapping[str, object]], output_path: Path, extra_fields=()
) -> None:
    """Write one pixel-metric row for every evaluated image."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        *extra_fields,
        "image_path",
        "score_path",
        "image_label",
        "image_score",
        "gt_positive_pixels",
        *PIXEL_METRIC_NAMES,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, float("nan")) for field in fields})
