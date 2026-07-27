"""Refine score-map anomaly regions with DINO ROI feature retrieval."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from roi_feature_utils import (
    feature_path_for_image,
    iter_image_paths,
    l2_normalize,
    load_feature_map,
    load_manifest,
    load_search_index,
    mask_bbox,
    relative_posix,
    resize_mask_to_feature,
    roi_align_vector,
)


LOGGER = logging.getLogger("refine_anomaly_regions")


def _score_path_for_image(
    image_path: Path,
    score_dir: Path,
    image_root: Path,
) -> Optional[Path]:
    relative = Path(relative_posix(image_path, image_root))
    candidates = [
        score_dir / relative.with_suffix(".npy"),
        score_dir / f"{image_path.stem}.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(score_dir.rglob(f"{image_path.stem}.npy"))
    if not matches:
        # PatchCore predict.py prefixes files with an ordinal, for example
        # 00001_image.npy, while Dinomaly2 uses image.npy.
        matches = sorted(score_dir.rglob(f"*_{image_path.stem}.npy"))
    return matches[0] if matches else None


def _load_score_map(path: Path) -> np.ndarray:
    score_map = np.asarray(np.load(path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(f"Expected a 2D score map in {path}, got {score_map.shape}")
    return np.nan_to_num(score_map)


def _find_components(
    score_map: np.ndarray,
    score_threshold: float,
    min_area: int,
):
    binary = np.asarray(score_map >= score_threshold, dtype=np.uint8)
    number, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for component_id in range(1, number):
        x, y, width, height, area = stats[component_id].tolist()
        if area < min_area:
            continue
        component_mask = labels == component_id
        components.append(
            {
                "component_id": int(component_id),
                "bbox": [float(x), float(y), float(x + width), float(y + height)],
                "area": int(area),
                "mask": component_mask,
            }
        )
    return binary, components


def _load_metadata(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def refine(args) -> int:
    image_root = Path(args.image_root).expanduser()
    score_dir = Path(args.score_dir).expanduser()
    feature_dir = Path(args.feature_dir).expanduser()
    index_path = Path(args.index).expanduser()
    metadata_path = Path(args.metadata).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(metadata_path)
    manifest = load_manifest(feature_dir)
    roi_size = int(args.roi_size or metadata["roi_size"])
    normalize = bool(metadata.get("normalize", True))
    index, _resources = load_search_index(
        index_path,
        on_gpu=args.faiss_on_gpu and args.gpu >= 0,
        gpu_id=args.gpu,
    )
    if index.d != int(metadata["feature_dim"]):
        raise ValueError(
            f"FAISS dimension {index.d} does not match metadata dimension "
            f"{metadata['feature_dim']}."
        )

    image_paths = iter_image_paths(image_root, recursive=args.recursive)
    for image_path in image_paths:
        score_path = _score_path_for_image(image_path, score_dir, image_root)
        feature_path = feature_path_for_image(
            image_path, feature_dir, image_root, manifest
        )
        if score_path is None or feature_path is None:
            LOGGER.warning(
                "Skipping %s: score map or DINO feature not found.",
                image_path,
            )
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            LOGGER.warning("Cannot read image: %s", image_path)
            continue
        image_height, image_width = image_bgr.shape[:2]
        score_map = _load_score_map(score_path)
        score_map = cv2.resize(
            score_map,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        _, components = _find_components(
            score_map, args.score_threshold, args.min_area
        )
        feature = load_feature_map(feature_path)
        feature_shape = feature.shape[-2:]
        refined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        refined_count = 0

        for component in components:
            component_mask = component["mask"]
            mask_feature = resize_mask_to_feature(component_mask, feature_shape)
            bbox_feature = mask_bbox(mask_feature)
            if bbox_feature is None:
                continue
            vector = roi_align_vector(
                feature,
                bbox_feature,
                mask_feature=mask_feature,
                output_size=roi_size,
            )
            if normalize:
                vector = l2_normalize(vector)
            distances, _neighbors = index.search(
                np.asarray([vector], dtype=np.float32), 1
            )
            distance = float(distances[0, 0])
            if distance >= args.distance_threshold:
                refined_mask[component_mask] = 1
                refined_count += 1

        relative = image_path.relative_to(image_root).with_suffix(".png")
        refined_mask_path = mask_dir / relative
        refined_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(refined_mask_path), refined_mask * 255):
            raise IOError(f"Failed to write refined mask: {refined_mask_path}")
        LOGGER.info(
            "%s: score_regions=%d, refined_regions=%d -> %s",
            image_path.name,
            len(components),
            refined_count,
            refined_mask_path,
        )

    LOGGER.info("Wrote refined masks to %s", mask_dir)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Refine score-map regions with DINO ROI FAISS retrieval."
    )
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--score_dir", required=True, help="Directory of score-map .npy files.")
    parser.add_argument("--feature_dir", required=True, help="Directory from extract_dino_features.py.")
    parser.add_argument("--index", required=True, help="FAISS ROI index.")
    parser.add_argument("--metadata", required=True, help="ROI index metadata JSON.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_threshold", type=float, required=True)
    parser.add_argument("--distance_threshold", type=float, required=True)
    parser.add_argument("--min_area", type=int, default=16)
    parser.add_argument("--roi_size", type=int, default=None)
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument(
        "--faiss_on_gpu",
        dest="faiss_on_gpu",
        action="store_true",
        default=True,
        help="Search the FAISS index on GPU (enabled by default).",
    )
    parser.add_argument(
        "--no_faiss_on_gpu",
        "--no-faiss-on-gpu",
        dest="faiss_on_gpu",
        action="store_false",
    )
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return refine(args)


if __name__ == "__main__":
    raise SystemExit(main())
