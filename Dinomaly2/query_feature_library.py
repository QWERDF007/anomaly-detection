"""Reverse-lookup Dinomaly2 ROI features in one or more feature libraries.

Given one image and an anomaly-region mask (or a saved Dinomaly2 score map),
this script extracts the Dinomaly2 encoder feature, applies the same masked
ROIAlign operation used while building the libraries, and reports the nearest
library vectors.  Every result is resolved through ``vector_id`` to the
stored ``image_id``/``roi_id`` and the original image/ROI mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from dinomaly_two_stage import (
    connected_components,
    dilate_mask,
    feature_patch_geometry,
    l2_normalize,
    load_feature_library,
    load_mask,
    make_image_id,
    mask_bbox,
    patch_center_mask_with_fallback,
    preprocess_mask,
    record_for_vector_id,
    resize_mask_to_feature,
    resize_score_map_to_feature,
    roi_align_masked,
    search_library_topk,
    select_device,
    select_patch_positions,
)


LOGGER = logging.getLogger("query_feature_library")


def load_score_region_mask(
    score_path: Path,
    image_shape: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Load a score map, resize it to the image, and threshold it strictly."""

    score_map = np.asarray(np.load(score_path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Score map must be 2D: {score_path}; got {score_map.shape}"
        )
    height, width = image_shape
    if score_map.shape != (height, width):
        score_map = cv2.resize(
            score_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    score_map = np.nan_to_num(score_map, nan=0.0, posinf=np.finfo(np.float32).max)
    return np.asarray(score_map > float(threshold), dtype=bool)


def load_cached_feature(args, input_path: Path, libraries) -> np.ndarray:
    """Load the cached second-stage feature map for the query image.

    The cache root is derived from the libraries: ``<root>/preds/features``
    (or ``features_raw_patch`` when the library uses raw patch tokens).  The
    image is matched by its file name; the query fails unless exactly one
    cached feature exists, so the model is never needed.
    """

    library = libraries[0]
    root = library.index_path.parent.parent
    source = str(library.metadata.get("feature_source", ""))
    cache_root = root / "preds" / (
        "features_raw_patch" if source == "raw_patch" else "features"
    )
    if not cache_root.is_dir():
        raise RuntimeError(
            f"Feature cache directory missing: {cache_root}. "
            "Run dinomaly_two_threshold_predict.py first."
        )
    matches = sorted(
        path
        for path in cache_root.rglob(f"{input_path.stem}.npy")
        if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one cached feature for {input_path} under "
            f"{cache_root}; found {len(matches)}. "
            "Run dinomaly_two_threshold_predict.py first."
        )
    feature = np.asarray(np.load(matches[0]), dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError(
            f"Cached feature must be CHW: {matches[0]}; got {feature.shape}"
        )
    LOGGER.info("Using cached feature: %s", matches[0])
    return np.nan_to_num(feature)


def patch_to_image_coords(
    row: int,
    col: int,
    feature_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
    image_size: int,
    crop_size: int,
) -> Tuple[float, float]:
    """Map a feature-grid patch centre back to original-image coordinates."""

    feature_height, feature_width = feature_shape
    crop_offset = (int(image_size) - int(crop_size)) / 2.0
    x_resized = (float(col) + 0.5) / float(feature_width) * float(crop_size) + crop_offset
    y_resized = (float(row) + 0.5) / float(feature_height) * float(crop_size) + crop_offset
    scale_x = float(image_shape[1]) / float(image_size)
    scale_y = float(image_shape[0]) / float(image_size)
    return x_resized * scale_x, y_resized * scale_y


def load_run_config(libraries) -> Dict[str, Any]:
    """Load ``preds/run.json`` beside the libraries for prediction settings."""

    library = libraries[0]
    root = library.index_path.parent.parent
    run_path = root / "preds" / "run.json"
    if not run_path.is_file():
        return {}
    try:
        with run_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Cannot read %s: %s", run_path, error)
        return {}
    return config if isinstance(config, dict) else {}


def load_cached_score_map(args, input_path: Path, libraries) -> np.ndarray:
    """Load the cached score map for the query image (patch library mode)."""

    library = libraries[0]
    root = library.index_path.parent.parent
    cache_root = root / "preds" / "score_maps"
    if not cache_root.is_dir():
        raise RuntimeError(
            f"Score-map cache directory missing: {cache_root}. "
            "Run dinomaly_two_threshold_predict.py first."
        )
    matches = sorted(
        path
        for path in cache_root.rglob(f"{input_path.stem}.npy")
        if path.is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one cached score map for {input_path} under "
            f"{cache_root}; found {len(matches)}. "
            "Run dinomaly_two_threshold_predict.py first."
        )
    score_map = np.asarray(np.load(matches[0]), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Cached score map must be 2D: {matches[0]}; got {score_map.shape}"
        )
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
    )


def resolve_library_paths(args) -> List[Path]:
    paths = [Path(path).expanduser() for path in args.library]
    paths.extend(
        Path(path).expanduser()
        for path in (args.good_library, args.anomaly_library)
        if path
    )
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise ValueError(
            "At least one feature library is required via --library, "
            "--good_library, or --anomaly_library."
        )
    return unique


def validate_query_libraries(libraries, args) -> None:
    """Check that all libraries share one feature configuration.

    The query parameters come from the cached features and the library
    metadata; the first library defines the expected ``roi_size`` and
    feature layout, and every library must agree with it.
    """

    reference = libraries[0]
    for key in (
        "feature_dim",
        "feature_source",
        "roi_size",
        "normalize",
        "backbone",
        "feature_shape",
        "patch_selection_rule",
    ):
        stored = reference.metadata.get(key)
        for library in libraries[1:]:
            other = library.metadata.get(key)
            if stored is not None and other is not None and stored != other:
                raise ValueError(
                    f"Library {key} differs: {reference.index_path} {stored!r} "
                    f"vs {library.index_path} {other!r}"
                )
    mode = str(reference.metadata.get("library_mode", "roi"))
    for library in libraries[1:]:
        if str(library.metadata.get("library_mode", "roi")) != mode:
            raise ValueError(
                f"Library library_mode differs: {reference.index_path} {mode!r} "
                f"vs {library.index_path} "
                f"{library.metadata.get('library_mode', 'roi')!r}"
            )


def query_feature_library(args) -> int:
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    with Image.open(input_path) as image:
        image_shape = (image.height, image.width)

    if args.region_mask:
        region_mask = load_mask(
            Path(args.region_mask).expanduser(),
            image_shape,
            args.mask_threshold,
        )
        region_source = str(Path(args.region_mask).expanduser())
    else:
        if args.score_threshold is None:
            raise ValueError("--score_threshold is required with --score_map")
        region_mask = load_score_region_mask(
            Path(args.score_map).expanduser(),
            image_shape,
            args.score_threshold,
        )
        region_source = str(Path(args.score_map).expanduser())

    components = connected_components(
        region_mask,
        min_area=args.min_area,
        max_regions=args.max_regions,
    )
    if not components:
        raise RuntimeError(
            "No anomaly ROI was found. Check the input mask/score threshold."
        )

    library_paths = resolve_library_paths(args)
    device = select_device(args.gpu)
    libraries = [
        load_feature_library(path, device, args.faiss_on_gpu)
        for path in library_paths
    ]
    validate_query_libraries(libraries, args)
    feature = load_cached_feature(args, input_path, libraries)
    feature_shape = feature.shape[-2:]
    query_image_relative = input_path.name
    query_image_id = make_image_id(query_image_relative)
    results: List[Dict[str, Any]] = []
    vanished_count = 0
    image_size = int(libraries[0].metadata.get("image_size", 672))
    crop_size = int(libraries[0].metadata.get("crop_size", 672))
    roi_size = int(libraries[0].metadata.get("roi_size", 7))
    library_mode = str(libraries[0].metadata.get("library_mode", "roi"))
    stored_feature_shape = libraries[0].metadata.get("feature_shape")
    if library_mode == "patch" and isinstance(
        stored_feature_shape,
        (list, tuple),
    ):
        expected_feature_shape = [int(value) for value in feature_shape]
        if [int(value) for value in stored_feature_shape] != expected_feature_shape:
            raise ValueError(
                "Cached feature shape does not match patch-library metadata: "
                f"{expected_feature_shape} vs {stored_feature_shape}"
            )
    patch_ratio = (
        args.patch_top_ratio
        if args.patch_top_ratio is not None
        else float(libraries[0].metadata.get("patch_top_ratio", 0.5))
    )
    score_map = None
    if library_mode == "patch":
        score_map = load_cached_score_map(args, input_path, libraries)
        run_config = load_run_config(libraries)
        process_size = int(run_config.get("process_size", 0) or 0)
        if process_size > 0:
            score_map = cv2.resize(
                score_map,
                (process_size, process_size),
                interpolation=cv2.INTER_LINEAR,
            )

    def add_result(
        library,
        distance: float,
        vector_id: int,
        rank: int,
        patch_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = record_for_vector_id(library.metadata, vector_id)
        if not record.get("image_id") or not record.get("roi_id"):
            raise RuntimeError(
                f"{library.index_path} does not contain image_id/roi_id mapping. "
                "Rebuild it with the current dinomaly_two_stage.py."
            )
        result = {
            "query_image_id": query_image_id,
            "query_image_name": input_path.name,
            "query_image_path": str(input_path.resolve()),
            "query_region_id": int(component["component_id"]),
            "query_region_area": int(component["area"]),
            "query_region_bbox": [
                float(value) for value in component["bbox"]
            ],
            "region_source": region_source,
            "library_type": library_type,
            "library_path": str(library.index_path.parent),
            "rank": int(rank),
            "distance": float(distance),
            "vector_id": int(vector_id),
            "image_id": str(record["image_id"]),
            "roi_id": str(record["roi_id"]),
            "image_name": str(record.get("image_name", "")),
            "image_path": str(record.get("image_path", "")),
            "image_relative": str(record.get("image_relative", "")),
            "mask_path": str(record.get("mask_path", "")),
            "component_id": int(record.get("component_id", -1)),
            "area": int(record.get("area", 0)),
            "bbox_original": [
                float(value) for value in record.get("bbox_original", [])
            ],
            "bbox_feature": [
                float(value) for value in record.get("bbox_feature", [])
            ],
            "patch_bbox_original": [
                float(value) for value in record.get("patch_bbox_original", [])
            ],
            "patch_center_original": [
                float(value) for value in record.get("patch_center_original", [])
            ],
            "patch_center_inside_mask": bool(
                record.get("patch_center_inside_mask", False)
            ),
            "patch_bbox_resized": [
                float(value) for value in record.get("patch_bbox_resized", [])
            ],
            "patch_center_resized": [
                float(value) for value in record.get("patch_center_resized", [])
            ],
            "feature_shape": [
                int(value)
                for value in record.get("feature_shape", feature_shape)
            ],
        }
        if patch_info is not None:
            result.update(patch_info)
        results.append(result)

    for component in components:
        model_mask = preprocess_mask(
            component["mask"],
            image_size,
            crop_size,
        )
        if library_mode == "patch":
            # Keep query patch selection identical to library construction:
            # the feature-cell centre, not OpenCV's nearest-neighbour sample,
            # must be inside the preprocessed ROI mask.  Tiny regions or
            # regions outside the CenterCrop fall back to the nearest feature
            # cell (in original image space when needed) so the ROI is still
            # queried instead of being dropped.
            if not model_mask.any():
                model_mask = np.asarray(component["mask"], dtype=bool)
            mask_feature = patch_center_mask_with_fallback(
                model_mask,
                feature_shape,
            )
        else:
            mask_feature = resize_mask_to_feature(model_mask, feature_shape)
        if mask_bbox(mask_feature) is None and library_mode != "patch":
            mask_feature = dilate_mask(mask_feature, 1)
        if mask_bbox(mask_feature) is None:
            vanished_count += 1
            LOGGER.warning(
                "Query ROI %s vanished after preprocessing; skipping",
                component["component_id"],
            )
            continue
        if library_mode == "patch":
            score_feature = resize_score_map_to_feature(
                score_map,
                feature_shape,
                image_size,
                crop_size,
            )
            positions = select_patch_positions(
                score_feature,
                mask_feature,
                patch_ratio,
            )
            if positions.shape[0] == 0:
                vanished_count += 1
                continue
            # 与预测一致：只用区域内分数最高的单个 patch 查询
            # （select_patch_positions 按分数降序，首行即最大）。
            positions = positions[:1]
            query_patch_geometry = feature_patch_geometry(
                int(positions[0, 0]),
                int(positions[0, 1]),
                feature_shape,
                image_shape,
                image_size,
                crop_size,
            )
            patch_vectors = []
            for row, col in positions:
                patch_vector = feature[:, int(row), int(col)]
                if bool(libraries[0].metadata.get("normalize", True)):
                    patch_vector = l2_normalize(patch_vector)
                patch_vectors.append((int(row), int(col), patch_vector))
            for library in libraries:
                library_type = str(
                    library.metadata.get(
                        "library_type",
                        library.index_path.parent.name,
                    )
                )
                best_match = None
                for patch_index, (row, col, patch_vector) in enumerate(
                    patch_vectors
                ):
                    neighbours = search_library_topk(
                        library,
                        patch_vector,
                        args.top_k,
                    )
                    for rank, (distance, vector_id) in enumerate(
                        neighbours, start=1
                    ):
                        if (
                            best_match is None
                            or distance < best_match["distance"]
                        ):
                            best_match = {
                                "distance": float(distance),
                                "vector_id": int(vector_id),
                                "rank": int(rank),
                                "patch_info": {
                                    "query_patch_index": int(patch_index),
                                    "patch_row": int(row),
                                    "patch_col": int(col),
                                    "query_feature_shape": query_patch_geometry[
                                        "feature_shape"
                                    ],
                                    "query_patch_bbox_feature": (
                                        query_patch_geometry["bbox_feature"]
                                    ),
                                    "query_patch_center_feature": (
                                        query_patch_geometry["center_feature"]
                                    ),
                                    "query_patch_bbox_resized": (
                                        query_patch_geometry["bbox_resized"]
                                    ),
                                    "query_patch_center_resized": (
                                        query_patch_geometry["center_resized"]
                                    ),
                                    "query_patch_bbox_original": (
                                        query_patch_geometry["bbox_original"]
                                    ),
                                    "query_patch_center_original": (
                                        query_patch_geometry["center_original"]
                                    ),
                                },
                            }
                if best_match is not None:
                    add_result(
                        library,
                        best_match["distance"],
                        best_match["vector_id"],
                        best_match["rank"],
                        patch_info=best_match["patch_info"],
                    )
            continue
        vector = roi_align_masked(
            feature,
            mask_feature,
            roi_size,
            device,
        )
        for library in libraries:
            if bool(library.metadata.get("normalize", True)):
                query_vector = l2_normalize(vector)
            else:
                query_vector = vector
            neighbours = search_library_topk(
                library,
                query_vector,
                args.top_k,
            )
            library_type = str(
                library.metadata.get(
                    "library_type",
                    library.index_path.parent.name,
                )
            )
            for rank, (distance, vector_id) in enumerate(neighbours, start=1):
                add_result(library, distance, vector_id, rank)

    if not results:
        if vanished_count:
            raise RuntimeError(
                f"No valid feature-library match was produced: {vanished_count} "
                "query region(s) vanished after preprocessing. The ROI is too "
                "small (below one feature-map pixel) or lies outside the "
                "CenterCrop area; enlarge it or redraw it."
            )
        raise RuntimeError("No valid feature-library match was produced.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "lookup_results.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "query_image": str(input_path.resolve()),
                "region_source": region_source,
                "top_k": int(args.top_k),
                "library_mode": library_mode,
                "feature_shape": [int(value) for value in feature_shape],
                "image_size": image_size,
                "crop_size": crop_size,
                "patch_selection_rule": str(
                    libraries[0].metadata.get(
                        "patch_selection_rule",
                        "top_ratio_by_score_among_feature_cells_whose_center_is_inside_mask",
                    )
                ),
                "results": results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = output_dir / "lookup_results.csv"
    fieldnames = [
        "query_image_id",
        "query_image_name",
        "query_image_path",
        "query_region_id",
        "query_region_area",
        "query_region_bbox",
        "query_patch_index",
        "patch_row",
        "patch_col",
        "query_feature_shape",
        "query_patch_bbox_feature",
        "query_patch_center_feature",
        "query_patch_bbox_resized",
        "query_patch_center_resized",
        "query_patch_bbox_original",
        "query_patch_center_original",
        "region_source",
        "library_type",
        "library_path",
        "rank",
        "distance",
        "vector_id",
        "image_id",
        "roi_id",
        "image_name",
        "image_path",
        "image_relative",
        "mask_path",
        "component_id",
        "area",
        "bbox_original",
        "bbox_feature",
        "patch_bbox_original",
        "patch_center_original",
        "patch_center_inside_mask",
        "patch_bbox_resized",
        "patch_center_resized",
        "feature_shape",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            for key in ("query_region_bbox", "bbox_original", "bbox_feature"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)

    for result in results:
        print(
            f"query_roi={result['query_region_id']} "
            f"library={result['library_type']} rank={result['rank']} "
            f"distance={result['distance']:.6f} "
            f"vector_id={result['vector_id']} "
            f"image_id={result['image_id']} roi_id={result['roi_id']} "
            f"image={result['image_path']} "
            f"roi_bbox={result['bbox_original']}",
            flush=True,
        )
    print(f"Wrote lookup results to {csv_path} and {json_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find source images and ROIs for an input Dinomaly2 anomaly region"
    )
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument("--input", required=True, help="One query image")
    region = parser.add_mutually_exclusive_group(required=True)
    region.add_argument("--region_mask", "--anomaly_mask", dest="region_mask")
    region.add_argument("--score_map", help="Saved Dinomaly2 score-map .npy")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Strict threshold used with --score_map",
    )
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--max_regions", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--patch_top_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of highest-score patches queried per ROI in patch "
            "library mode; defaults to the library's metadata value"
        ),
    )
    parser.add_argument("--library", nargs="+", default=[])
    parser.add_argument("--good_library", default=None)
    parser.add_argument("--anomaly_library", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    return parser


def validate_args(args) -> None:
    if args.min_area < 1:
        raise ValueError("min_area must be at least 1")
    if args.max_regions < 0:
        raise ValueError("max_regions cannot be negative")
    if args.top_k < 1:
        raise ValueError("top_k must be at least 1")
    if args.score_map and args.score_threshold is None:
        raise ValueError("--score_threshold is required with --score_map")
    if args.patch_top_ratio is not None and not 0.0 < args.patch_top_ratio <= 1.0:
        raise ValueError("patch_top_ratio must be in (0, 1]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return query_feature_library(args)


if __name__ == "__main__":
    raise SystemExit(main())
