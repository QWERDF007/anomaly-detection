"""Canonical anomaly-detection metrics shared by Dinomaly2 and PatchCore.

The definitions in this module follow Dinomaly2's training evaluator:

* image score: mean of the highest 1% pixels in a score map;
* image/pixel AUROC, AP and maximum F1: scikit-learn PR/ROC curves;
* P-AUPRO: the Dinomaly2 200-threshold PRO sweep below FPR 0.3.

Model-specific scripts are responsible only for producing score maps and
loading their masks.  Keeping the calculation here prevents training,
visualization and offline directory evaluation from drifting apart.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score
from skimage import measure
from tqdm import tqdm

from score_workflow_common import (
    CLASSIFICATION_METRIC_NAMES,
    classification_metrics,
    load_score_map,
    pixel_f1_score_and_threshold,
    region_detection_metrics,
    report_metric_names,
    select_optimal_threshold,
    write_metric_report,
    write_per_image_report,
)


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
REPORT_METRIC_NAMES = report_metric_names(METRIC_NAMES)
TRAINING_IMAGE_SCORE_RATIO = 0.01

PER_IMAGE_METRIC_FIELDS = (
    "stage",
    "directory",
    "group",
    "image_path",
    "mask_path",
    "score_path",
    "image_label",
    "image_score",
    "gt_positive_pixels",
    "gt_region_count",
    "detected_region_count",
    "missed_region_count",
    "tp_region_count",
    "fp_region_count",
    *PIXEL_METRIC_NAMES,
    "R-MissRate",
    "R-PixelCoverage",
    "R-FPR",
)


def _normalise_arrays(labels, scores) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("labels and scores must be non-empty arrays of equal length")
    return labels, scores


def safe_auroc(labels, scores) -> float:
    """Dinomaly2 AUROC with an undefined single-class result represented by NaN."""

    try:
        labels, scores = _normalise_arrays(labels, scores)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def safe_ap(labels, scores) -> float:
    """Dinomaly2 AP, including the defined all-normal value of zero."""

    try:
        labels, scores = _normalise_arrays(labels, scores)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            value = float(average_precision_score(labels, scores))
        return 0.0 if value == 0.0 else value
    except ValueError:
        return float("nan")


def max_f1(labels, scores) -> float:
    """Maximum F1 over the Dinomaly2 precision-recall sweep."""

    try:
        labels, scores = _normalise_arrays(labels, scores)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            precision, recall, _ = precision_recall_curve(labels, scores)
    except ValueError:
        return float("nan")
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
    # The final PR point has no threshold; Dinomaly2 excludes it.
    return float(np.nanmax(f1[:-1])) if len(f1) > 1 else 0.0


def training_image_score(
    score_map: np.ndarray,
    top_ratio: float = TRAINING_IMAGE_SCORE_RATIO,
) -> float:
    """Return Dinomaly2's highest-pixel-ratio mean score for one image."""

    score_map = np.asarray(score_map, dtype=np.float32)
    if score_map.size == 0:
        raise ValueError("Cannot calculate an image score from an empty score map.")
    if not 0.0 < float(top_ratio) <= 1.0:
        raise ValueError("top_ratio must be in (0, 1].")
    top_count = max(1, int(score_map.size * float(top_ratio)))
    return float(np.sort(score_map.reshape(-1))[-top_count:].mean())


def compute_pro_fast(
    masks: np.ndarray,
    amaps: np.ndarray,
    num_th: int = 200,
    show_progress: bool = True,
) -> float:
    """Compute Dinomaly2's P-AUPRO implementation without a pandas loop."""

    masks = np.asarray(masks, dtype=np.uint8)
    amaps = np.asarray(amaps, dtype=np.float32)
    if masks.ndim != 3 or amaps.ndim != 3 or masks.shape != amaps.shape:
        raise ValueError("masks and amaps must be equally shaped [N,H,W] arrays")
    if set(np.unique(masks).tolist()) != {0, 1}:
        raise AssertionError("masks must contain both 0 and 1")
    if not isinstance(num_th, int) or num_th <= 0:
        raise ValueError("num_th must be a positive integer")

    minimum = float(amaps.min())
    maximum = float(amaps.max())
    delta = (maximum - minimum) / num_th
    if delta <= 0.0:
        return 0.0
    thresholds = np.arange(minimum, maximum, delta, dtype=np.float32)

    inverse_masks = np.logical_not(masks.astype(bool))
    inverse_count = int(inverse_masks.sum())
    if inverse_count == 0:
        return float("nan")

    labels_per_image = []
    region_areas = []
    for mask in masks:
        labels = measure.label(mask)
        areas = np.bincount(labels.reshape(-1))[1:].astype(np.float64)
        labels_per_image.append(labels)
        region_areas.append(areas)
    total_regions = sum(len(areas) for areas in region_areas)
    if total_regions == 0:
        return float("nan")

    pro_sums = np.zeros_like(thresholds, dtype=np.float64)
    false_positive_counts = np.zeros_like(thresholds, dtype=np.int64)
    entries = zip(labels_per_image, region_areas, amaps, inverse_masks)
    if show_progress:
        entries = tqdm(
            entries,
            total=len(amaps),
            desc="Compute PRO",
            unit="image",
            dynamic_ncols=True,
            leave=False,
        )
    for labels, areas, amap, inverse_mask in entries:
        outside_values = np.sort(amap[inverse_mask])
        false_positive_counts += outside_values.size - np.searchsorted(
            outside_values, thresholds, side="right"
        )
        for region_id, area in enumerate(areas, start=1):
            region_values = np.sort(amap[labels == region_id])
            hits = region_values.size - np.searchsorted(
                region_values, thresholds, side="right"
            )
            pro_sums += hits / area

    pros = pro_sums / total_regions
    fprs = false_positive_counts / inverse_count
    valid = fprs < 0.3
    if not np.any(valid):
        return float("nan")
    fprs = fprs[valid]
    pros = pros[valid]
    max_fpr = float(fprs.max())
    if max_fpr <= 0.0:
        return float("nan")
    # ``np.arange`` visits thresholds from low to high, which is the same
    # decreasing-FPR order used by Dinomaly2/utils.py before calling auc().
    return float(auc(fprs / max_fpr, pros))


def safe_aupro(
    masks: np.ndarray,
    scores: np.ndarray,
    show_progress: bool = True,
) -> float:
    """Return Dinomaly2 P-AUPRO and preserve its undefined-value policy."""

    masks = np.asarray(masks, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)
    if masks.shape != scores.shape or masks.ndim != 3 or not np.any(masks):
        return float("nan")
    if float(scores.max()) <= float(scores.min()):
        return 0.0
    try:
        return compute_pro_fast(masks, scores, show_progress=show_progress)
    except (AssertionError, ValueError, ZeroDivisionError):
        return float("nan")


def prepare_pixel_arrays(segmentations, masks) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize common score/mask containers to matching ``[N,H,W]`` arrays."""

    segmentations = np.asarray(segmentations, dtype=np.float32)
    masks = np.asarray(masks)
    if segmentations.size == 0 or masks.size == 0:
        return None, None
    if segmentations.ndim == 4 and segmentations.shape[1] == 1:
        segmentations = segmentations[:, 0]
    if masks.ndim == 4:
        masks = masks[:, 0] if masks.shape[1] == 1 else masks.max(axis=1)
    if segmentations.ndim != 3 or masks.ndim != 3 or segmentations.shape != masks.shape:
        return None, None
    return segmentations, (masks > 0).astype(np.uint8)


def resize_metric_arrays(
    score_maps: np.ndarray,
    masks: np.ndarray,
    metric_size: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize score maps linearly and GT masks with nearest-neighbour sampling."""

    if metric_size is None:
        return score_maps, masks
    if int(metric_size) < 1:
        raise ValueError("metric_size must be positive")
    target = (int(metric_size), int(metric_size))
    if score_maps.shape[1:] == target:
        return score_maps, masks
    resized_scores = np.stack(
        [cv2.resize(item, target, interpolation=cv2.INTER_LINEAR) for item in score_maps]
    ).astype(np.float32)
    resized_masks = np.stack(
        [cv2.resize(item, target, interpolation=cv2.INTER_NEAREST) for item in masks]
    ).astype(np.uint8)
    return resized_scores, (resized_masks > 0).astype(np.uint8)


def compute_evaluation_metrics(
    scores,
    labels,
    segmentations,
    masks,
    metric_size: Optional[int] = None,
    image_score_ratio: Optional[float] = TRAINING_IMAGE_SCORE_RATIO,
) -> Dict[str, float]:
    """Calculate canonical image and pixel metrics.

    Whenever pixel score maps are present, their image scores are deliberately
    recomputed with :func:`training_image_score`.  Passing
    ``image_score_ratio=None`` retains an explicitly supplied legacy image
    score (needed only by older Dinomaly2 calls that used a map maximum).
    """

    image_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    anomaly_maps, gt_masks = prepare_pixel_arrays(segmentations, masks)
    if anomaly_maps is None or gt_masks is None:
        image_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        pixel_labels = np.asarray([], dtype=np.uint8)
        pixel_scores = np.asarray([], dtype=np.float32)
        pixel_f1 = float("nan")
        pixel_threshold = float("nan")
        p_aupro = float("nan")
    else:
        anomaly_maps, gt_masks = resize_metric_arrays(anomaly_maps, gt_masks, metric_size)
        if len(image_labels) != len(anomaly_maps):
            raise ValueError("labels must contain one item per score map")
        if image_score_ratio is None:
            image_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            if len(image_scores) != len(anomaly_maps):
                raise ValueError("scores must contain one item per score map")
        else:
            image_scores = np.asarray(
                [
                    training_image_score(score_map, image_score_ratio)
                    for score_map in anomaly_maps
                ],
                dtype=np.float32,
            )
        pixel_labels = gt_masks.reshape(-1)
        pixel_scores = anomaly_maps.reshape(-1)
        pixel_f1, pixel_threshold = pixel_f1_score_and_threshold(gt_masks, anomaly_maps)
        p_aupro = safe_aupro(gt_masks, anomaly_maps, show_progress=False)

    return {
        "I-AUROC": safe_auroc(image_labels, image_scores),
        "I-AP": safe_ap(image_labels, image_scores),
        "I-F1": max_f1(image_labels, image_scores),
        "P-AUROC": safe_auroc(pixel_labels, pixel_scores),
        "P-AP": safe_ap(pixel_labels, pixel_scores),
        "P-F1": pixel_f1,
        "P-AUPRO": p_aupro,
        "P-F1-Threshold": pixel_threshold,
    }


def evaluate_pixel_metrics(gt_mask: np.ndarray, score_map: np.ndarray) -> Dict[str, float]:
    """Calculate the canonical pixel metrics for one image."""

    gt_mask = np.asarray(gt_mask, dtype=np.uint8)
    score_map = np.asarray(score_map, dtype=np.float32)
    if gt_mask.ndim != 2 or gt_mask.shape != score_map.shape:
        raise ValueError(
            "Per-image pixel metrics need equally shaped 2D arrays; got "
            f"{gt_mask.shape} and {score_map.shape}."
        )
    labels = (gt_mask > 0).reshape(-1).astype(np.uint8)
    scores = score_map.reshape(-1)
    has_positive = bool(np.any(labels))
    has_negative = bool(np.any(labels == 0))
    return {
        "P-AUROC": safe_auroc(labels, scores) if has_positive and has_negative else float("nan"),
        "P-AP": safe_ap(labels, scores) if has_positive and has_negative else 0.0,
        "P-F1": max_f1(labels, scores) if has_positive else 0.0,
        "P-AUPRO": (
            safe_aupro(gt_mask[None, ...], score_map[None, ...], show_progress=False)
            if has_positive and has_negative
            else float("nan")
        ),
    }


def write_metrics(
    results: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    label_name: str = "stage",
) -> None:
    """Write the standardized aggregate metric report."""

    write_metric_report(results, output_dir, METRIC_NAMES, label_name)


def write_per_image_pixel_metrics(
    records: Sequence[Mapping[str, object]], output_path: Path
) -> None:
    """Write standardized per-image pixel and region metrics."""

    write_per_image_report(records, output_path, PER_IMAGE_METRIC_FIELDS)


__all__ = (
    "CLASSIFICATION_METRIC_NAMES",
    "METRIC_NAMES",
    "PIXEL_METRIC_NAMES",
    "REPORT_METRIC_NAMES",
    "PER_IMAGE_METRIC_FIELDS",
    "TRAINING_IMAGE_SCORE_RATIO",
    "classification_metrics",
    "compute_evaluation_metrics",
    "compute_pro_fast",
    "evaluate_pixel_metrics",
    "load_score_map",
    "max_f1",
    "pixel_f1_score_and_threshold",
    "prepare_pixel_arrays",
    "region_detection_metrics",
    "report_metric_names",
    "resize_metric_arrays",
    "safe_ap",
    "safe_aupro",
    "safe_auroc",
    "select_optimal_threshold",
    "training_image_score",
    "write_metric_report",
    "write_metrics",
    "write_per_image_pixel_metrics",
    "write_per_image_report",
)
