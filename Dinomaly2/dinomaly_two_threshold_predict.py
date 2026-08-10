"""Dinomaly2 prediction with good/anomaly score thresholds.

Images are classified in three initial bands:

* ``raw_score < good_threshold``: good, without feature-library search;
* ``raw_score > anomaly_threshold``: anomaly, without feature-library search;
* otherwise: extract regions using the good threshold, search both ROI
  libraries, apply the distance-based offset, and make a final binary
  good/anomaly decision.

The offset magnitude is the distance to the nearer library
(``min(d_good, d_anomaly)``), signed negative when the good library is
nearer and positive when the anomaly library is nearer.  After the offset
is applied, scores at or above the good threshold (including the middle
band) are final anomaly.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import logging
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(1, str(_UTILS_DIR))

from anomaly_evaluation import (  # noqa: E402
    max_f1,
    pixel_f1_score_and_threshold,
    region_detection_metrics,
    safe_ap,
    safe_auroc,
    safe_aupro,
    training_image_score,
    write_metrics,
)

from dinomaly_two_stage import (
    _json_safe,
    _model_feature_mask,
    add_model_arguments,
    build_transform,
    calculate_distance_offset,
    connected_components,
    dilate_mask,
    feature_patch_geometry,
    infer_image,
    iter_image_paths,
    l2_normalize,
    load_dinomaly_model,
    load_feature_library,
    load_mask,
    load_patch_backbone,
    mask_bbox,
    patch_center_mask_with_fallback,
    preprocess_mask,
    record_for_vector_id,
    roi_align_masked,
    resize_score_map_to_feature,
    search_library,
    select_device,
    select_patch_positions,
    select_strongest_region,
    validate_args,
    validate_library_compatibility,
)
from utils import get_gaussian_kernel


LOGGER = logging.getLogger("dinomaly_two_threshold_predict")


def collect_data_root_images(
    data_root: Path,
) -> List[Tuple[Path, Path, str]]:
    """Collect images grouped by the first-level directory under ``data_root``."""

    data_root = Path(data_root).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    group_dirs = sorted(
        [child for child in data_root.iterdir() if child.is_dir()],
        key=lambda path: str(path).lower(),
    )
    if not group_dirs:
        raise RuntimeError(
            f"No subdirectories found under {data_root}; expected good/ and anomaly directories"
        )

    entries: List[Tuple[Path, Path, str]] = []
    for group_dir in group_dirs:
        dataset_label = "good" if group_dir.name.casefold() == "good" else "anomaly"
        for image_path in iter_image_paths(group_dir):
            entries.append(
                (
                    image_path,
                    image_path.relative_to(data_root),
                    dataset_label,
                )
            )
    return entries


def output_artifact_path(
    output_dir: Path,
    artifact_name: str,
    image_relative: Path,
    suffix: str,
) -> Path:
    """Build an output path while preserving each input's relative layout."""

    result = Path(output_dir) / artifact_name / image_relative.with_suffix(suffix)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def write_score_density_plot(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    density_points: int,
    good_threshold: float,
    anomaly_threshold: float,
) -> None:
    """Write vertically stacked raw/adjusted continuous KDE curves."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required to write score_density.png"
        ) from error

    grouped: Dict[str, Dict[str, List[float]]] = {
        "good": {"raw": [], "adjusted": []},
        "anomaly": {"raw": [], "adjusted": []},
    }
    for row in rows:
        group = str(row.get("dataset_label", "anomaly"))
        if group not in grouped:
            group = "anomaly"
        for score_key, output_key in (
            ("raw_score", "raw"),
            ("adjusted_score", "adjusted"),
        ):
            value = float(row.get(score_key, np.nan))
            if np.isfinite(value):
                grouped[group][output_key].append(value)

    all_values = [
        value
        for group_values in grouped.values()
        for score_values in group_values.values()
        for value in score_values
    ]
    if not all_values:
        return

    minimum = min(all_values)
    maximum = max(all_values)
    score_range = maximum - minimum
    padding = max(score_range * 0.08, abs(minimum) * 0.02, 0.01)
    if np.isclose(score_range, 0.0):
        padding = max(abs(minimum) * 0.05, 0.05)
    grid = np.linspace(
        minimum - padding,
        maximum + padding,
        max(int(density_points), 100),
    )

    def density_curve(values: Sequence[float]) -> np.ndarray:
        """Estimate a continuous density, including for tiny samples."""

        array = np.asarray(values, dtype=np.float64)
        if array.size >= 2 and np.ptp(array) > 1e-12:
            try:
                from scipy.stats import gaussian_kde

                return np.asarray(gaussian_kde(array)(grid), dtype=np.float64)
            except (ImportError, np.linalg.LinAlgError, ValueError):
                pass

        # A Gaussian mixture keeps the curve continuous when KDE is singular
        # for one sample or for identical scores.
        bandwidth = 1.06 * float(np.std(array)) * max(array.size, 1) ** (-0.2)
        grid_step = float(grid[1] - grid[0]) if len(grid) > 1 else 1e-3
        bandwidth = max(bandwidth, grid_step * 1.5, 1e-6)
        normalized = (grid[:, None] - array[None, :]) / bandwidth
        return np.mean(
            np.exp(-0.5 * normalized * normalized)
            / (np.sqrt(2.0 * np.pi) * bandwidth),
            axis=1,
        )

    colors = {"good": "#2ca02c", "anomaly": "#d62728"}
    labels = {"good": "good", "anomaly": "anomaly"}
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, score_key, title in (
        (axes[0], "raw", "Before correction: raw score density"),
        (axes[1], "adjusted", "After correction: adjusted score density"),
    ):
        for group in ("good", "anomaly"):
            values = grouped[group][score_key]
            if not values:
                continue
            density = density_curve(values)
            axis.plot(
                grid,
                density,
                color=colors[group],
                linewidth=2.0,
                label=f"{labels[group]} (n={len(values)})",
            )
            axis.fill_between(
                grid,
                density,
                color=colors[group],
                alpha=0.12,
            )
        axis.axvline(
            float(good_threshold),
            color="#555555",
            linestyle="--",
            linewidth=1,
            label="good threshold",
        )
        axis.axvline(
            float(anomaly_threshold),
            color="#111111",
            linestyle=":",
            linewidth=1,
            label="anomaly threshold",
        )
        axis.set_title(title)
        axis.set_ylabel("Density")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("Score")
    figure.suptitle("Dinomaly2 score density before and after correction")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def initial_score_label(
    score: float,
    good_threshold: float,
    anomaly_threshold: float,
) -> str:
    """Return the initial good/anomaly/middle band label."""

    if float(score) < float(good_threshold):
        return "good"
    if float(score) > float(anomaly_threshold):
        return "anomaly"
    return "middle"


def final_score_label(
    adjusted_score: float,
    good_threshold: float,
    anomaly_threshold: float,
    similar_library: str = "",
) -> tuple[str, str]:
    """Make a binary final decision and return ``(label, reason)``.

    Scores strictly below the good threshold are normal; anything at or
    above it, including the middle band, is anomaly.
    """

    if float(adjusted_score) < float(good_threshold):
        return "good", "adjusted_below_good_threshold"
    if float(adjusted_score) > float(anomaly_threshold):
        return "anomaly", "adjusted_above_anomaly_threshold"
    return "anomaly", "adjusted_in_middle_band"


def _match_metadata(
    library,
    vector_id: int,
    prefix: str,
) -> Dict[str, Any]:
    """Flatten one FAISS neighbour's reverse mapping into a result row."""

    fields: Dict[str, Any] = {f"{prefix}_vector_id": int(vector_id)}
    try:
        record = record_for_vector_id(library.metadata, int(vector_id))
    except (KeyError, TypeError, ValueError):
        return fields
    for key in (
        "image_id",
        "roi_id",
        "image_name",
        "image_path",
        "mask_path",
        "bbox_original",
        "patch_bbox_original",
        "patch_center_original",
        "feature_shape",
    ):
        value = record.get(key, "")
        fields[f"{prefix}_{key}"] = value
    return fields


def top_ratio_mean(values: np.ndarray, ratio: float) -> float:
    """Mean of the highest ``ratio`` fraction of an array (anti-noise)."""

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1].")
    top_count = max(1, int(values.size * float(ratio)))
    return float(np.sort(values)[-top_count:].mean())


def _build_region_result(
    component: Mapping[str, Any],
    score_map: np.ndarray,
    feature: np.ndarray,
    good_library,
    anomaly_library,
    args,
    device,
    image_shape: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    """Search both libraries for one score-map connected component."""

    query_mask = dilate_mask(component["mask"], args.roi_dilation)
    library_mode = str(good_library.metadata.get("library_mode", "roi"))
    model_mask = preprocess_mask(
        query_mask,
        args.image_size,
        args.crop_size,
    )
    if library_mode == "patch":
        # Patch mode has an explicit geometric rule: only feature cells whose
        # centre is inside the transformed query mask may be searched.  Tiny
        # regions or regions outside the CenterCrop fall back to the nearest
        # feature cell (in original image space when needed) so the ROI is
        # still queried instead of being dropped.
        if not model_mask.any():
            model_mask = np.asarray(query_mask, dtype=bool)
        mask_feature = patch_center_mask_with_fallback(
            model_mask,
            feature.shape[-2:],
        )
    else:
        mask_feature = _model_feature_mask(query_mask, feature.shape[-2:], args)
    bbox_feature = mask_bbox(mask_feature)
    if bbox_feature is None and library_mode != "patch":
        mask_feature = dilate_mask(mask_feature, 1)
        bbox_feature = mask_bbox(mask_feature)
    if bbox_feature is None:
        return None

    if library_mode == "patch":
        patch_ratio = float(good_library.metadata.get("patch_top_ratio", 0.5))
        score_feature = resize_score_map_to_feature(
            score_map,
            feature.shape[-2:],
            args.image_size,
            args.crop_size,
        )
        positions = select_patch_positions(
            score_feature,
            mask_feature,
            patch_ratio,
        )
        if positions.shape[0] == 0:
            return None
        # 建库用区域内前 x% 的 patch 全量入库；查询只用区域内
        # 分数最高的单个 patch（select_patch_positions 降序，首行即最大）。
        positions = positions[:1]
        good_matches = []
        anomaly_matches = []
        for row, col in positions:
            patch_vector = feature[:, int(row), int(col)]
            if bool(good_library.metadata.get("normalize", True)):
                patch_vector = l2_normalize(patch_vector)
            good_distance, good_neighbour = search_library(
                good_library,
                patch_vector,
            )
            anomaly_distance, anomaly_neighbour = search_library(
                anomaly_library,
                patch_vector,
            )
            good_matches.append((good_distance, good_neighbour))
            anomaly_matches.append((anomaly_distance, anomaly_neighbour))
        good_distance, good_neighbour = min(
            good_matches,
            key=lambda match: match[0],
        )
        anomaly_distance, anomaly_neighbour = min(
            anomaly_matches,
            key=lambda match: match[0],
        )
        best_row, best_col = positions[
            min(range(len(good_matches)), key=lambda i: good_matches[i][0])
        ]
        patch_geometry = feature_patch_geometry(
            int(best_row),
            int(best_col),
            feature.shape[-2:],
            image_shape,
            args.image_size,
            args.crop_size,
        )
        decision = calculate_distance_offset(
            good_distance,
            anomaly_distance,
            args.offset_scale,
            args.max_offset,
            args.offset_eps,
        )
        region = {
            "region_id": int(component["component_id"]),
            "region_score": top_ratio_mean(
                score_map[component["mask"]],
                args.region_top_ratio,
            ),
            "library_mode": "patch",
            "patch_count": int(positions.shape[0]),
            "patch_top_ratio": patch_ratio,
            "best_patch_row": int(best_row),
            "best_patch_col": int(best_col),
            "feature_shape": [
                int(feature.shape[-2]),
                int(feature.shape[-1]),
            ],
            "best_patch_bbox_feature": patch_geometry["bbox_feature"],
            "best_patch_center_feature": patch_geometry["center_feature"],
            "best_patch_bbox_resized": patch_geometry["bbox_resized"],
            "best_patch_center_resized": patch_geometry["center_resized"],
            "best_patch_bbox_original": patch_geometry["bbox_original"],
            "best_patch_center_original": patch_geometry["center_original"],
            "area": int(component["area"]),
            "bbox_original": [float(value) for value in component["bbox"]],
            "bbox_feature": [float(value) for value in bbox_feature],
            "good_distance": float(good_distance),
            "anomaly_distance": float(anomaly_distance),
            **decision,
            **_match_metadata(good_library, good_neighbour, "good"),
            **_match_metadata(anomaly_library, anomaly_neighbour, "anomaly"),
        }
        region["good_neighbour"] = int(good_neighbour)
        region["anomaly_neighbour"] = int(anomaly_neighbour)
        return region

    vector = roi_align_masked(
        feature,
        mask_feature,
        args.roi_size,
        device,
    )
    if bool(good_library.metadata.get("normalize", True)):
        vector = l2_normalize(vector)
    good_distance, good_neighbour = search_library(good_library, vector)
    anomaly_distance, anomaly_neighbour = search_library(anomaly_library, vector)
    decision = calculate_distance_offset(
        good_distance,
        anomaly_distance,
        args.offset_scale,
        args.max_offset,
        args.offset_eps,
    )
    region = {
        "region_id": int(component["component_id"]),
        "region_score": top_ratio_mean(
            score_map[component["mask"]],
            args.region_top_ratio,
        ),
        "library_mode": "roi",
        "area": int(component["area"]),
        "bbox_original": [float(value) for value in component["bbox"]],
        "bbox_feature": [float(value) for value in bbox_feature],
        "good_distance": float(good_distance),
        "anomaly_distance": float(anomaly_distance),
        **decision,
        **_match_metadata(good_library, good_neighbour, "good"),
        **_match_metadata(anomaly_library, anomaly_neighbour, "anomaly"),
    }
    # Keep the integer neighbour fields alongside the reverse mapping fields.
    region["good_neighbour"] = int(good_neighbour)
    region["anomaly_neighbour"] = int(anomaly_neighbour)
    return region


def evaluate_image_level(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Evaluate the raw and adjusted image scores with canonical metrics.

    Reuses the same image-level metrics as dinomaly_score_visualization
    (I-AUROC / I-AP / I-F1); anomaly is the positive class.
    """

    labels = np.asarray(
        [
            1 if str(row.get("dataset_label", "")) != "good" else 0
            for row in rows
        ],
        dtype=np.uint8,
    )
    evaluations: Dict[str, Dict[str, float]] = {}
    for key, name in (("raw_score", "raw"), ("adjusted_score", "adjusted")):
        scores = np.asarray(
            [float(row.get(key, np.nan)) for row in rows],
            dtype=np.float32,
        )
        finite = np.isfinite(scores)
        if not np.all(finite):
            scores = scores[finite]
            valid_labels = labels[finite]
        else:
            valid_labels = labels
        evaluations[name] = {
            "I-AUROC": safe_auroc(valid_labels, scores),
            "I-AP": safe_ap(valid_labels, scores),
            "I-F1": max_f1(valid_labels, scores),
        }
    write_metrics(evaluations, output_dir)
    print("\n图像级评估（原始 vs 调整后）：", flush=True)
    for name, metrics in evaluations.items():
        line = "  ".join(f"{metric}={value:.4f}" for metric, value in metrics.items())
        print(f"  {name}: {line}", flush=True)
    return evaluations


def _apply_two_stage_overlay(
    score_map: np.ndarray,
    output_dir: Path,
    image_relative: Path,
) -> np.ndarray:
    """Return a copy of ``score_map`` with the two-stage results overwritten.

    Every middle-band ROI pixel is replaced by its adjusted score when the
    region is finally judged anomaly, or by zero when judged good.  This lets
    pixel-level and region-level evaluation reflect the second stage instead
    of the raw Dinomaly2 map.  Regions are re-derived from the saved
    ``candidate_regions`` mask, whose connected-component labels match the
    ``region_id`` values written to ``details``.
    """

    detail_path = output_dir / "details" / image_relative.with_suffix(".json")
    if not detail_path.is_file():
        return score_map
    try:
        with detail_path.open("r", encoding="utf-8") as file:
            detail = json.load(file)
    except (OSError, json.JSONDecodeError):
        return score_map
    regions = detail.get("regions", [])
    if not regions:
        return score_map
    candidate_path = (
        output_dir / "candidate_regions" / image_relative.with_suffix(".png")
    )
    if not candidate_path.is_file():
        return score_map
    candidate_mask = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
    if (
        candidate_mask is None
        or candidate_mask.shape != score_map.shape[:2]
    ):
        return score_map
    count, labels, _stats, _ = cv2.connectedComponentsWithStats(
        (candidate_mask > 0).astype(np.uint8),
        8,
    )
    if count <= 1:
        return score_map
    good_threshold = float(detail.get("good_threshold", 0.0))
    anomaly_threshold = float(detail.get("anomaly_threshold", 0.0))
    overlay = score_map.copy()
    for region in regions:
        region_id = int(region.get("region_id", -1))
        if not 0 < region_id < count:
            continue
        region_mask = labels == region_id
        region_score = float(region.get("region_score", 0.0))
        signed_offset = float(region.get("signed_offset", 0.0))
        adjusted_score = region_score + signed_offset
        label, _reason = final_score_label(
            adjusted_score,
            good_threshold,
            anomaly_threshold,
            str(region.get("similar_library", "")),
        )
        if label == "good":
            overlay[region_mask] = 0.0
        else:
            overlay[region_mask] = adjusted_score
    return overlay


def evaluate_pixel_level(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    ground_truth_dir: Optional[Path],
    metric_size: int,
    good_threshold: float = 0.0,
    adjusted: bool = False,
) -> Dict[str, float]:
    """Evaluate pixel-level metrics on the saved score maps.

    With ``adjusted=False`` the raw saved score maps are evaluated.  With
    ``adjusted=True`` each score map is copied and the two-stage region
    results are overwritten on the copy (:func:`_apply_two_stage_overlay`),
    then pixel metrics and GT-region detection metrics (R-MissRate /
    R-PixelCoverage at the good threshold) are computed on it.  Only anomaly
    images with a ground-truth mask contribute.
    """

    if ground_truth_dir is None:
        print("未提供 GT 目录，跳过像素级评估。", flush=True)
        return {}
    anomaly_maps = []
    gt_masks = []
    skipped = 0
    for row in rows:
        if str(row.get("dataset_label", "")) == "good":
            continue
        image_relative = Path(row["image_relative"])
        score_path = (
            output_dir / "score_maps" / image_relative.with_suffix(".npy")
        )
        if not score_path.is_file():
            continue
        score_map = np.asarray(np.load(score_path), dtype=np.float32)
        if adjusted:
            score_map = _apply_two_stage_overlay(
                score_map,
                output_dir,
                image_relative,
            )
        gt_path = None
        for suffix in (".png", ".jpg", ".jpeg", ".npy", ".tif", ".json"):
            candidate = ground_truth_dir / image_relative.with_suffix(suffix)
            if candidate.is_file():
                gt_path = candidate
                break
        if gt_path is None:
            skipped += 1
            continue
        try:
            gt_mask = load_mask(gt_path, score_map.shape[:2])
        except (OSError, ValueError) as error:
            LOGGER.warning("Skipping GT %s: %s", gt_path, error)
            skipped += 1
            continue
        anomaly_maps.append(score_map)
        gt_masks.append(gt_mask.astype(np.uint8))
    if not anomaly_maps:
        print(f"无可用 GT 掩码（跳过 {skipped} 张），无法计算像素级指标。", flush=True)
        return {}
    target = (int(metric_size), int(metric_size))
    maps = np.stack(
        [
            cv2.resize(score_map, target, interpolation=cv2.INTER_LINEAR)
            for score_map in anomaly_maps
        ]
    ).astype(np.float32)
    masks = np.stack(
        [
            cv2.resize(gt_mask, target, interpolation=cv2.INTER_NEAREST)
            for gt_mask in gt_masks
        ]
    ).astype(np.uint8)
    masks = (masks > 0).astype(np.uint8)
    pixel_labels = masks.reshape(-1)
    pixel_scores = maps.reshape(-1)
    pixel_f1, _ = pixel_f1_score_and_threshold(masks, maps)
    metrics = {
        "P-AUROC": safe_auroc(pixel_labels, pixel_scores),
        "P-AP": safe_ap(pixel_labels, pixel_scores),
        "P-F1": pixel_f1,
        "P-AUPRO": safe_aupro(masks, maps, show_progress=False),
    }
    metrics.update(
        region_detection_metrics(
            masks,
            maps,
            float(good_threshold),
        )
    )
    label_text = "调整后" if adjusted else "原始"
    print(
        f"\n像素级/区域级评估（{label_text} score maps）："
        + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        flush=True,
    )
    return metrics


def apply_library_metadata(args, metadata: Mapping[str, Any]) -> None:
    """Overlay prediction parameters from the library metadata onto ``args``.

    The library records the exact feature configuration used at build time
    (image/crop size, ROI size, backbone, merge and feature source), so the
    prediction command only needs the model, data root, root and thresholds.
    Metadata wins; values absent from the metadata keep the CLI/defaults.
    """

    for arg_key, meta_key, cast in (
        ("image_size", "image_size", int),
        ("crop_size", "crop_size", int),
        ("roi_size", "roi_size", int),
        ("backbone", "backbone", str),
        ("feature_merge", "feature_merge", str),
    ):
        value = metadata.get(meta_key)
        if value is None:
            continue
        setattr(args, arg_key, cast(value))
    source = str(metadata.get("feature_source", ""))
    if source == "raw_patch":
        args.feature_source = "raw_patch"
    elif source == "dinomaly_encoder_output":
        args.feature_source = "dinomaly"


_WORKER: Dict[str, Any] = {}


def _worker_init(args, output_dir: Path, gpu_index: int) -> None:
    """Load per-process model, libraries and transform into the global state.

    Every worker process (and the single-process path) owns its own Dinomaly2
    model and FAISS indexes; nothing is shared across processes.  Spawn is
    used for multiprocessing so each worker initialises its CUDA context
    cleanly.
    """

    global _WORKER
    device = select_device(int(gpu_index))
    faiss_on_gpu = device.type == "cuda"
    root = Path(output_dir).parent
    good_library = load_feature_library(root / "good", device, faiss_on_gpu)
    anomaly_library = load_feature_library(root / "anomaly", device, faiss_on_gpu)
    apply_library_metadata(args, good_library.metadata)
    validate_library_compatibility(good_library, anomaly_library, args)
    model = load_dinomaly_model(args, device)
    patch_backbone = None
    if args.feature_source == "raw_patch":
        patch_backbone = load_patch_backbone(args, device)
    transform = build_transform(args)
    gaussian_filter = get_gaussian_kernel(5, 4).to(device)
    cache_root = Path(output_dir) / (
        "features_raw_patch" if args.feature_source == "raw_patch" else "features"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    _WORKER.update(
        {
            "args": args,
            "device": device,
            "good_library": good_library,
            "anomaly_library": anomaly_library,
            "model": model,
            "patch_backbone": patch_backbone,
            "transform": transform,
            "gaussian_filter": gaussian_filter,
            "output_dir": Path(output_dir),
            "cache_root": cache_root,
            "process_size": int(args.process_size),
        }
    )


def _process_one_entry(
    args,
    output_dir: Path,
    gpu_index: int,
    entry: Tuple[Path, Path, str],
) -> Optional[
    Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], int]
]:
    """Predict one image; returns ``(row, detail, roi_rows, cache_hit)``.

    Artifacts are written to their own paths, so worker processes never
    contend; only the row/detail/roi rows are returned to the parent.  The
    worker state is initialised lazily on the first task: multiprocessing
    swallows exceptions raised by a Pool ``initializer``, which hid the real
    failure behind a later ``KeyError``, so initialisation happens here and
    any error propagates to the parent through the task result.
    """

    global _WORKER
    if not _WORKER:
        _worker_init(args, output_dir, gpu_index)
    image_path, image_relative, dataset_label = entry
    worker = _WORKER
    args = worker["args"]
    device = worker["device"]
    good_library = worker["good_library"]
    anomaly_library = worker["anomaly_library"]
    model = worker["model"]
    patch_backbone = worker["patch_backbone"]
    transform = worker["transform"]
    gaussian_filter = worker["gaussian_filter"]
    output_dir = worker["output_dir"]
    cache_root = worker["cache_root"]
    process_size = worker["process_size"]

    with Image.open(image_path) as image:
        original_image_shape = (int(image.height), int(image.width))
    score_path = output_artifact_path(
        output_dir,
        "score_maps",
        image_relative,
        ".npy",
    )
    feature_path = output_artifact_path(
        cache_root,
        "",
        image_relative,
        ".npy",
    )
    cached = (
        not args.recompute_features
        and score_path.is_file()
        and feature_path.is_file()
    )
    if cached:
        score_map = np.load(score_path)
        feature = np.load(feature_path)
        cache_hit = 1
    else:
        score_map, feature = infer_image(
            model,
            image_path,
            transform,
            device,
            args.feature_merge,
            gaussian_filter,
            patch_backbone=patch_backbone,
            feature_source=args.feature_source,
        )
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(score_path, score_map)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, feature)
        cache_hit = 0
    if process_size > 0:
        score_map = cv2.resize(
            score_map,
            (process_size, process_size),
            interpolation=cv2.INTER_LINEAR,
        )
    raw_score = (
        float(training_image_score(score_map)) if score_map.size else 0.0
    )
    initial_label = initial_score_label(
        raw_score,
        args.good_threshold,
        args.anomaly_threshold,
    )
    regions: List[Dict[str, Any]] = []
    # Always save the direct Dinomaly2 threshold result.  This is kept
    # separate from candidate_regions, which contains only the regions
    # used by the second-stage feature-library search.
    raw_region_mask = (
        np.asarray(score_map >= float(args.good_threshold), dtype=np.uint8)
    )
    candidate_mask = np.zeros(score_map.shape, dtype=np.uint8)

    if initial_label == "middle":
        # Equality belongs to the middle band because the direct rules
        # intentionally use strict < and > comparisons.
        min_area = 1
        if float(args.min_area_pct) > 0.0:
            min_area = max(
                min_area,
                int(
                    round(
                        float(args.min_area_pct) / 100.0 * score_map.size
                    )
                ),
            )
        components = connected_components(
            score_map >= float(args.good_threshold),
            min_area=min_area,
            max_regions=args.max_regions,
        )
        for component in components:
            candidate_mask[component["mask"]] = 1
            try:
                region = _build_region_result(
                    component,
                    score_map,
                    feature,
                    good_library,
                    anomaly_library,
                    args,
                    device,
                    original_image_shape,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                LOGGER.warning(
                    "Skipping ROI %s in %s: %s",
                    component["component_id"],
                    image_path,
                    error,
                )
                continue
            if region is not None:
                regions.append(region)

    selected = select_strongest_region(regions)
    signed_offset = float(selected["signed_offset"]) if selected else 0.0
    adjusted_anomaly_mask = np.zeros(score_map.shape, dtype=np.uint8)
    overlay = score_map.copy()
    if regions:
        count, labels, _stats, _ = cv2.connectedComponentsWithStats(
            (candidate_mask > 0).astype(np.uint8),
            8,
        )
        for region in regions:
            region_id = int(region["region_id"])
            if not 0 < region_id < count:
                continue
            region_mask = labels == region_id
            region_adjusted = float(region["region_score"]) + float(
                region["signed_offset"]
            )
            region_label, _region_reason = final_score_label(
                region_adjusted,
                args.good_threshold,
                args.anomaly_threshold,
                str(region.get("similar_library", "")),
            )
            if region_label == "good":
                overlay[region_mask] = 0.0
            else:
                overlay[region_mask] = region_adjusted
                adjusted_anomaly_mask[region_mask] = 1
        adjusted_score = (
            float(training_image_score(overlay)) if overlay.size else 0.0
        )
    else:
        adjusted_score = raw_score
    final_label, decision_reason = final_score_label(
        adjusted_score,
        args.good_threshold,
        args.anomaly_threshold,
        str(selected.get("similar_library", "")) if selected else "",
    )
    relative_text = image_relative.as_posix()
    roi_rows_for_image: List[Dict[str, Any]] = []
    for region in regions:
        roi_rows_for_image.append(
            {
                "image_path": str(image_path),
                "image_relative": relative_text,
                "dataset_label": dataset_label,
                "raw_score": raw_score,
                "good_threshold": float(args.good_threshold),
                "anomaly_threshold": float(args.anomaly_threshold),
                **region,
            }
        )

    raw_region_path = output_artifact_path(
        output_dir,
        "raw_regions",
        image_relative,
        ".png",
    )
    region_path = output_artifact_path(
        output_dir,
        "candidate_regions",
        image_relative,
        ".png",
    )
    adjusted_region_path = output_artifact_path(
        output_dir,
        "adjusted_candidate_regions",
        image_relative,
        ".png",
    )
    detail_path = output_artifact_path(
        output_dir,
        "details",
        image_relative,
        ".json",
    )
    if not cv2.imwrite(str(raw_region_path), raw_region_mask * 255):
        raise OSError(f"Cannot write raw threshold region mask: {raw_region_path}")
    if not cv2.imwrite(str(region_path), candidate_mask * 255):
        raise OSError(f"Cannot write candidate region mask: {region_path}")
    # This is the mask used by the GUI/final visualization.  A region
    # judged good by stage two is intentionally absent here.  Keep the
    # original candidate mask above because pixel-level adjusted metrics
    # still need to know which regions must be overwritten with zero.
    if not cv2.imwrite(
        str(adjusted_region_path),
        adjusted_anomaly_mask * 255,
    ):
        raise OSError(
            "Cannot write adjusted candidate region mask: "
            f"{adjusted_region_path}"
        )

    row = {
        "image_path": str(image_path),
        "image_relative": relative_text,
        "dataset_label": dataset_label,
        "raw_score": raw_score,
        "good_threshold": float(args.good_threshold),
        "anomaly_threshold": float(args.anomaly_threshold),
        "process_size": process_size,
        "initial_label": initial_label,
        "adjusted_score": adjusted_score,
        "final_label": final_label,
        "decision_reason": decision_reason,
        "stage2_applied": bool(initial_label == "middle" and regions),
        "region_count": len(regions),
        "selected_region_id": selected.get("region_id", "") if selected else "",
        "good_distance": selected.get("good_distance", "") if selected else "",
        "anomaly_distance": selected.get("anomaly_distance", "") if selected else "",
        "similar_library": selected.get("similar_library", "") if selected else "",
        "confidence": selected.get("confidence", 0.0) if selected else 0.0,
        "offset": selected.get("offset", 0.0) if selected else 0.0,
        "signed_offset": signed_offset,
    }
    detail = {
        **row,
        "score_map_path": str(score_path),
        "raw_region_path": str(raw_region_path),
        "candidate_region_path": str(region_path),
        "adjusted_candidate_region_path": str(adjusted_region_path),
        "regions": regions,
    }
    with detail_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(detail), file, ensure_ascii=False, indent=2)
    LOGGER.info(
        "%s raw=%.6f (%s) adjusted=%.6f (%s) regions=%d library=%s offset=%+.6f",
        image_path,
        raw_score,
        initial_label,
        adjusted_score,
        final_label,
        len(regions),
        selected.get("similar_library", "none") if selected else "none",
        signed_offset,
    )
    return row, detail, roi_rows_for_image, cache_hit


def predict_images(args) -> int:
    data_root = Path(args.data_root).expanduser()
    image_entries = collect_data_root_images(data_root)
    if not image_entries:
        raise RuntimeError(f"No images found under {data_root}")

    root = Path(args.root).expanduser()
    output_dir = root / "preds"
    output_dir.mkdir(parents=True, exist_ok=True)
    # The parent process only needs the library metadata (for run.json and
    # the feature configuration); models and indexes are loaded per worker.
    good_metadata: Dict[str, Any] = {}
    good_metadata_path = root / "good" / "metadata.json"
    if good_metadata_path.is_file():
        try:
            with good_metadata_path.open("r", encoding="utf-8") as file:
                good_metadata = json.load(file)
            apply_library_metadata(args, good_metadata)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            LOGGER.warning("Cannot read library metadata: %s", error)

    gpu_devices = [int(value) for value in args.gpu_devices]
    entries = list(image_entries)
    start_time = time.time()
    if len(gpu_devices) <= 1:
        _worker_init(args, output_dir, gpu_devices[0])
        collected = []
        for entry in tqdm(
            entries,
            desc="Dinomaly2 dual-threshold prediction",
            unit="image",
            dynamic_ncols=True,
        ):
            result = _process_one_entry(args, output_dir, gpu_devices[0], entry)
            if result is not None:
                collected.append(result)
    else:
        context = multiprocessing.get_context("spawn")
        chunk_count = len(gpu_devices)
        chunks = [
            entries[index::chunk_count] for index in range(chunk_count)
        ]
        pools = []
        iterators = []
        for gpu_index, chunk in zip(gpu_devices, chunks):
            pool = context.Pool(processes=1)
            task = functools.partial(
                _process_one_entry,
                args,
                output_dir,
                gpu_index,
            )
            pools.append(pool)
            iterators.append(pool.imap(task, chunk))
        collected = []
        alive = list(range(len(iterators)))
        try:
            with tqdm(
                total=len(entries),
                desc=f"Dinomaly2 dual-threshold prediction (GPUs {gpu_devices})",
                unit="image",
                dynamic_ncols=True,
            ) as progress:
                while alive:
                    index = alive.pop(0)
                    iterator = iterators[index]
                    try:
                        result = next(iterator)
                    except StopIteration:
                        continue
                    else:
                        if result is not None:
                            collected.append(result)
                        progress.update(1)
                        alive.append(index)
        except Exception:
            for pool in pools:
                pool.terminate()
            raise
        finally:
            for pool in pools:
                pool.close()
            for pool in pools:
                pool.join()

    rows = [result[0] for result in collected]
    details = [result[1] for result in collected]
    roi_rows = [roi for result in collected for roi in result[2]]
    cache_hits = sum(result[3] for result in collected)
    computed = len(entries) - cache_hits

    csv_path = output_dir / "results.csv"
    fieldnames = [
        "image_path",
        "dataset_label",
        "raw_score",
        "initial_label",
        "adjusted_score",
        "final_label",
        "stage2_applied",
        "region_count",
        "selected_region_id",
        "good_distance",
        "anomaly_distance",
        "confidence",
        "signed_offset",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    density_path = output_dir / "score_density.png"
    write_score_density_plot(
        rows,
        density_path,
        args.density_points,
        args.good_threshold,
        args.anomaly_threshold,
    )

    roi_csv_path = output_dir / "roi_results.csv"
    roi_fieldnames = [
        "image_path",
        "dataset_label",
        "raw_score",
        "region_id",
        "region_score",
        "library_mode",
        "patch_count",
        "patch_top_ratio",
        "best_patch_row",
        "best_patch_col",
        "feature_shape",
        "best_patch_bbox_feature",
        "best_patch_center_feature",
        "best_patch_bbox_resized",
        "best_patch_center_resized",
        "best_patch_bbox_original",
        "best_patch_center_original",
        "area",
        "bbox_original",
        "bbox_feature",
        "good_distance",
        "good_neighbour",
        "good_vector_id",
        "good_image_id",
        "good_roi_id",
        "good_image_name",
        "good_image_path",
        "good_mask_path",
        "good_bbox_original",
        "anomaly_distance",
        "anomaly_neighbour",
        "anomaly_vector_id",
        "anomaly_image_id",
        "anomaly_roi_id",
        "anomaly_image_name",
        "anomaly_image_path",
        "anomaly_mask_path",
        "anomaly_bbox_original",
        "confidence",
        "signed_offset",
    ]
    with roi_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=roi_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for roi_row in roi_rows:
            row_for_csv = dict(roi_row)
            row_for_csv["bbox_original"] = json.dumps(
                row_for_csv["bbox_original"],
                ensure_ascii=False,
            )
            row_for_csv["bbox_feature"] = json.dumps(
                row_for_csv["bbox_feature"],
                ensure_ascii=False,
            )
            for key in (
                "good_bbox_original",
                "anomaly_bbox_original",
                "feature_shape",
                "best_patch_bbox_feature",
                "best_patch_center_feature",
                "best_patch_bbox_resized",
                "best_patch_center_resized",
                "best_patch_bbox_original",
                "best_patch_center_original",
            ):
                if key in row_for_csv:
                    row_for_csv[key] = json.dumps(row_for_csv[key], ensure_ascii=False)
            writer.writerow(row_for_csv)

    with (output_dir / "run.json").open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(
                {
                    "good_threshold": args.good_threshold,
                    "anomaly_threshold": args.anomaly_threshold,
                    "offset_scale": args.offset_scale,
                    "max_offset": args.max_offset,
                    "offset_eps": args.offset_eps,
                    "roi_dilation": args.roi_dilation,
                    "region_top_ratio": args.region_top_ratio,
                    "library_mode": str(
                        good_metadata.get("library_mode", "roi")
                    ),
                    "patch_top_ratio": float(
                        good_metadata.get("patch_top_ratio", 0.5)
                    ),
                    "patch_selection_rule": str(
                        good_metadata.get(
                            "patch_selection_rule",
                            "top_ratio_by_score_among_feature_cells_whose_center_is_inside_mask",
                        )
                    ),
                    "feature_shape": good_metadata.get(
                        "feature_shape",
                        [],
                    ),
                    "min_area_pct": args.min_area_pct,
                    "max_regions": args.max_regions,
                    "gpus": [int(value) for value in args.gpu_devices],
                    "process_size": int(args.process_size),
                    "density_points": args.density_points,
                    "feature_merge": args.feature_merge,
                    "roi_size": args.roi_size,
                    "score_density_plot": str(density_path),
                    "results": details,
                }
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )
    elapsed = time.time() - start_time
    print(
        f"Prediction finished: {len(image_entries)} images, "
        f"{cache_hits} cache hits, {computed} computed, "
        f"elapsed {elapsed:.1f}s "
        f"({elapsed / max(len(image_entries), 1) * 1000.0:.0f} ms/image)",
        flush=True,
    )
    evaluations = evaluate_image_level(rows, output_dir)
    if args.ground_truth_dir:
        ground_truth_dir = Path(args.ground_truth_dir).expanduser()
    else:
        ground_truth_dir = data_root / "ground_truth"
    if not ground_truth_dir.is_dir():
        ground_truth_dir = None
    pixel_raw = evaluate_pixel_level(
        rows,
        output_dir,
        ground_truth_dir,
        args.metric_size,
        good_threshold=args.good_threshold,
    )
    pixel_adjusted = evaluate_pixel_level(
        rows,
        output_dir,
        ground_truth_dir,
        args.metric_size,
        good_threshold=args.good_threshold,
        adjusted=True,
    )
    if pixel_raw and pixel_adjusted:
        print("\n像素级/区域级评估对比（原始 vs 二次调整后）：", flush=True)
        for name in (
            "P-AUROC",
            "P-AP",
            "P-F1",
            "P-AUPRO",
            "R-MissRate",
            "R-PixelCoverage",
            "R-FPR",
            "R-FP-RegionCount",
        ):
            raw_value = float(pixel_raw.get(name, float("nan")))
            adjusted_value = float(pixel_adjusted.get(name, float("nan")))
            print(
                f"  {name:>16}   raw={raw_value:.4f}   "
                f"adjusted={adjusted_value:.4f}",
                flush=True,
            )
        print(flush=True)
    if pixel_raw:
        evaluations["raw"].update(pixel_raw)
    if pixel_adjusted:
        evaluations["adjusted"].update(pixel_adjusted)
    write_metrics(evaluations, output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dinomaly2 prediction with good/anomaly score thresholds"
    )
    add_model_arguments(parser)
    # The shared --gpu argument is a single int; replace it with a GPU-index
    # list so one worker process is spawned per GPU.  _remove_action does
    # not clean _option_string_actions, so remove those mappings explicitly.
    for action in list(parser._actions):
        if getattr(action, "dest", None) == "gpu":
            parser._remove_action(action)
            for option_string in ("--gpu", "--cuda"):
                parser._option_string_actions.pop(option_string, None)
    parser.add_argument(
        "--gpu",
        "--cuda",
        dest="gpu",
        nargs="+",
        type=int,
        default=[0],
        help=(
            "GPU index(es); one worker process is spawned per GPU "
            "(e.g. '--gpu 0 1 2 3' runs 4 workers on GPUs 0-3). Each worker "
            "loads its own model and FAISS indexes. Defaults to [0]."
        ),
    )
    parser.add_argument(
        "--data_root",
        "--input",
        dest="data_root",
        required=True,
        help="Root containing first-level good/other anomaly directories",
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Root directory containing good/ and anomaly/ feature libraries; "
            "all prediction output is written under --root/preds/ "
            "(score_maps/, raw_regions/, candidate_regions/, details/, "
            "adjusted_candidate_regions/, results.csv, roi_results.csv, "
            "score_density.png, run.json)"
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        required=True,
        help=(
            "Two score thresholds in order: good_threshold anomaly_threshold "
            "(e.g. '--thresholds 0.02 0.06'). Scores strictly below the "
            "first are good, strictly above the second are anomaly."
        ),
    )
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument(
        "--min_area_pct",
        type=float,
        default=0.0,
        help=(
            "Minimum connected-component area as a percentage of the image "
            "area (e.g. 0.1 = 0.1%)"
        ),
    )
    parser.add_argument(
        "--process_size",
        type=int,
        default=0,
        help=(
            "Downsample the score map to this square resolution before "
            "thresholding, connected components and mask writing "
            "(0 = keep the original resolution)"
        ),
    )
    parser.add_argument("--max_regions", type=int, default=0)
    parser.add_argument(
        "--density_points",
        type=int,
        default=400,
        help="Number of points used to draw continuous density curves",
    )
    parser.add_argument(
        "--roi_dilation",
        type=int,
        default=0,
        help="Dilate each middle-band score-map component before ROIAlign",
    )
    parser.add_argument(
        "--region_top_ratio",
        type=float,
        default=0.10,
        help=(
            "Fraction of highest-pixel scores averaged into each ROI's "
            "region_score for anti-noise robustness (default: 0.10 = 10%)"
        ),
    )
    parser.add_argument(
        "--offset_scale",
        type=float,
        default=1.0,
        help="Maximum score correction generated by the normalized distance margin",
    )
    parser.add_argument("--max_offset", type=float, default=None)
    parser.add_argument("--offset_eps", type=float, default=1e-8)
    parser.add_argument(
        "--ground_truth_dir",
        default=None,
        help=(
            "GT anomaly mask directory mirroring --data_root's layout; "
            "defaults to --data_root/ground_truth. Used for pixel-level and "
            "region-level evaluation of the raw and two-stage-adjusted score "
            "maps."
        ),
    )
    parser.add_argument(
        "--metric_size",
        type=int,
        default=256,
        help="Side length used for pixel-metric arrays (default: 256)",
    )
    parser.add_argument(
        "--recompute_features",
        action="store_true",
        help=(
            "Recompute and overwrite the cached score maps / second-stage "
            "features instead of reusing preds/score_maps and "
            "preds/features (or features_raw_patch)"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    gpu_devices = [int(value) for value in args.gpu]
    if not gpu_devices:
        raise ValueError("at least one GPU index is required")
    if len(set(gpu_devices)) != len(gpu_devices):
        raise ValueError("GPU indices must be unique")
    args.gpu_devices = gpu_devices
    validate_args(args)
    if len(args.thresholds) != 2:
        raise ValueError(
            "--thresholds requires exactly two values: "
            "good_threshold anomaly_threshold"
        )
    args.good_threshold, args.anomaly_threshold = [
        float(value) for value in args.thresholds
    ]
    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise ValueError(f"--root does not exist: {root}")
    for subdir in ("good", "anomaly"):
        if not (root / subdir).is_dir():
            raise ValueError(f"--root must contain a {subdir}/ directory: {root}")
    if not np.isfinite(args.good_threshold) or not np.isfinite(args.anomaly_threshold):
        raise ValueError("good_threshold and anomaly_threshold must be finite")
    if args.good_threshold >= args.anomaly_threshold:
        raise ValueError("good_threshold must be smaller than anomaly_threshold")
    if args.density_points < 100:
        raise ValueError("density_points must be at least 100")
    if args.min_area_pct < 0:
        raise ValueError("min_area_pct cannot be negative")
    if not 0.0 < args.region_top_ratio <= 1.0:
        raise ValueError("region_top_ratio must be in (0, 1]")
    return predict_images(args)


if __name__ == "__main__":
    raise SystemExit(main())
