"""Dinomaly2 prediction with good/anomaly score thresholds.

Images are classified in three initial bands:

* ``raw_score < good_threshold``: good, without feature-library search;
* ``raw_score > anomaly_threshold``: anomaly, without feature-library search;
* otherwise: extract regions using the good threshold, search both ROI
  libraries, apply the distance-based offset, and make a final binary
  good/anomaly decision.

If the adjusted score remains between the two thresholds, the nearest feature
library decides the final label.  If there is no valid ROI match, the midpoint
of the two thresholds is used as the deterministic fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from dinomaly_two_stage import (
    _json_safe,
    _model_feature_mask,
    add_model_arguments,
    build_transform,
    calculate_distance_offset,
    connected_components,
    dilate_mask,
    infer_image,
    iter_image_paths,
    l2_normalize,
    load_dinomaly_model,
    load_feature_library,
    load_patch_backbone,
    mask_bbox,
    record_for_vector_id,
    roi_align_masked,
    search_library,
    select_device,
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
    """Make a binary final decision and return ``(label, reason)``."""

    if float(adjusted_score) < float(good_threshold):
        return "good", "adjusted_below_good_threshold"
    if float(adjusted_score) > float(anomaly_threshold):
        return "anomaly", "adjusted_above_anomaly_threshold"
    if similar_library in {"good", "anomaly"}:
        return similar_library, f"feature_library_{similar_library}"
    midpoint = (float(good_threshold) + float(anomaly_threshold)) / 2.0
    return (
        ("good" if float(adjusted_score) < midpoint else "anomaly"),
        "threshold_midpoint_fallback",
    )


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
    ):
        value = record.get(key, "")
        fields[f"{prefix}_{key}"] = value
    return fields


def _build_region_result(
    component: Mapping[str, Any],
    score_map: np.ndarray,
    feature: np.ndarray,
    good_library,
    anomaly_library,
    args,
    device,
) -> Optional[Dict[str, Any]]:
    """Search both libraries for one score-map connected component."""

    query_mask = dilate_mask(component["mask"], args.roi_dilation)
    mask_feature = _model_feature_mask(query_mask, feature.shape[-2:], args)
    bbox_feature = mask_bbox(mask_feature)
    if bbox_feature is None:
        mask_feature = dilate_mask(mask_feature, 1)
        bbox_feature = mask_bbox(mask_feature)
    if bbox_feature is None:
        return None

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
        "region_score": float(score_map[component["mask"]].max()),
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


def predict_images(args) -> int:
    data_root = Path(args.data_root).expanduser()
    image_entries = collect_data_root_images(data_root)
    if not image_entries:
        raise RuntimeError(f"No images found under {data_root}")

    root = Path(args.root).expanduser()
    output_dir = root / "preds"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.gpu)
    good_library = load_feature_library(
        root / "good",
        device,
        args.faiss_on_gpu,
    )
    anomaly_library = load_feature_library(
        root / "anomaly",
        device,
        args.faiss_on_gpu,
    )
    validate_library_compatibility(good_library, anomaly_library, args)
    model = load_dinomaly_model(args, device)
    patch_backbone = None
    if args.feature_source == "raw_patch":
        patch_backbone = load_patch_backbone(args, device)
    transform = build_transform(args)
    gaussian_filter = get_gaussian_kernel(5, 4).to(device)

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    roi_rows: List[Dict[str, Any]] = []
    for image_path, image_relative, dataset_label in tqdm(
        image_entries,
        desc="Dinomaly2 dual-threshold prediction",
        unit="image",
        dynamic_ncols=True,
    ):
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
        raw_score = float(np.max(score_map)) if score_map.size else 0.0
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
            components = connected_components(
                score_map >= float(args.good_threshold),
                min_area=args.min_area,
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
        adjusted_score = float(raw_score + signed_offset)
        final_label, decision_reason = final_score_label(
            adjusted_score,
            args.good_threshold,
            args.anomaly_threshold,
            str(selected.get("similar_library", "")) if selected else "",
        )
        relative_text = image_relative.as_posix()
        for region in regions:
            roi_rows.append(
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

        score_path = output_artifact_path(
            output_dir,
            "score_maps",
            image_relative,
            ".npy",
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
        detail_path = output_artifact_path(
            output_dir,
            "details",
            image_relative,
            ".json",
        )
        np.save(score_path, score_map)
        if not cv2.imwrite(str(raw_region_path), raw_region_mask * 255):
            raise OSError(f"Cannot write raw threshold region mask: {raw_region_path}")
        if not cv2.imwrite(str(region_path), candidate_mask * 255):
            raise OSError(f"Cannot write candidate region mask: {region_path}")

        row = {
            "image_path": str(image_path),
            "image_relative": relative_text,
            "dataset_label": dataset_label,
            "raw_score": raw_score,
            "good_threshold": float(args.good_threshold),
            "anomaly_threshold": float(args.anomaly_threshold),
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
            "regions": regions,
        }
        with detail_path.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(detail), file, ensure_ascii=False, indent=2)
        rows.append(row)
        details.append(detail)

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
                    "min_area": args.min_area,
                    "max_regions": args.max_regions,
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dinomaly2 prediction with good/anomaly score thresholds"
    )
    add_model_arguments(parser)
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
            "results.csv, roi_results.csv, score_density.png, run.json)"
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
        "--offset_scale",
        type=float,
        default=1.0,
        help="Maximum score correction generated by the normalized distance margin",
    )
    parser.add_argument("--max_offset", type=float, default=None)
    parser.add_argument("--offset_eps", type=float, default=1e-8)
    parser.add_argument(
        "--faiss_on_gpu",
        action="store_true",
        help="Move both FAISS indexes to the selected CUDA device",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
    return predict_images(args)


if __name__ == "__main__":
    raise SystemExit(main())
