"""Build a FAISS index from Labelme polygon ROIs and DINO feature maps."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from roi_feature_utils import (
    annotation_path_for_image,
    load_feature_map,
    load_labelme_annotation,
    load_manifest,
    mask_bbox,
    polygon_to_feature_mask,
    relative_posix,
    roi_align_vector,
    save_json,
)


LOGGER = logging.getLogger("build_roi_faiss_index")


def _image_path_from_record(
    record: Dict,
    image_root: Optional[Path],
    feature_dir: Path,
) -> Optional[Path]:
    candidate = Path(record.get("image_path", ""))
    if candidate.is_file():
        return candidate
    if image_root and record.get("image_relative"):
        candidate = image_root / record["image_relative"]
        if candidate.is_file():
            return candidate

    feature_path = feature_dir / record["feature_path"]
    if image_root:
        relative = Path(record.get("image_relative", feature_path.relative_to(feature_dir)))
        candidate = image_root / relative
        if candidate.is_file():
            return candidate
        for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            candidate = (image_root / relative).with_suffix(extension)
            if candidate.is_file():
                return candidate
    return None


def _fallback_records(feature_dir: Path, image_root: Optional[Path]):
    if image_root is None:
        raise ValueError(
            "No manifest.json found. --image_root is required to match .npy files to images."
        )
    image_extensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
    }
    records = []
    for feature_path in sorted(feature_dir.rglob("*.npy")):
        feature_relative = feature_path.relative_to(feature_dir)
        relative_stem = feature_relative.with_suffix("")
        image_path = None

        # Preserve filenames containing multiple dots, for example
        # camera.01.png -> camera.01.npy.
        for extension in sorted(image_extensions):
            candidate = image_root / Path(str(relative_stem) + extension)
            if candidate.is_file():
                image_path = candidate
                break

        # If image_root is a dataset root while feature_dir points to a
        # subdirectory such as train/good, fall back to a unique stem match.
        if image_path is None:
            matches = sorted(
                [
                    path for path in image_root.rglob("*")
                    if (
                        path.is_file()
                        and path.suffix.lower() in image_extensions
                        and path.stem == feature_path.stem
                    )
                ],
                key=lambda path: str(path).lower(),
            )
            if matches:
                image_path = matches[0]

        if image_path is not None:
            records.append(
                {
                    "feature_path": feature_path.relative_to(feature_dir).as_posix(),
                    "image_path": str(image_path),
                    "image_relative": image_path.relative_to(image_root).with_suffix("").as_posix(),
                }
            )
    return records


def build_index(args) -> int:
    feature_dir = Path(args.feature_dir).expanduser()
    annotation_dir = Path(args.annotation_dir).expanduser()
    image_root = Path(args.image_root).expanduser() if args.image_root else None
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(feature_dir)
    records = manifest if manifest else _fallback_records(feature_dir, image_root)
    if not records:
        feature_count = len(list(feature_dir.rglob("*.npy")))
        raise RuntimeError(
            f"No feature records found in {feature_dir}. Found {feature_count} .npy "
            f"files, but none matched image_root={image_root}. "
            "Pass the corresponding original-image directory with --image_root."
        )

    vectors = []
    roi_records: List[Dict] = []
    feature_dim = None
    skipped_images = 0

    for record in records:
        feature_path = feature_dir / record["feature_path"]
        image_path = _image_path_from_record(record, image_root, feature_dir)
        if not feature_path.is_file() or image_path is None:
            skipped_images += 1
            continue

        annotation_path = annotation_path_for_image(
            image_path, annotation_dir, image_root
        )
        if annotation_path is None:
            LOGGER.warning("No Labelme annotation for %s", image_path)
            skipped_images += 1
            continue

        annotation = load_labelme_annotation(annotation_path)
        image_size = (
            int(annotation["imageWidth"]),
            int(annotation["imageHeight"]),
        )
        feature = load_feature_map(feature_path)
        feature_shape = feature.shape[-2:]
        shapes = annotation.get("shapes", [])
        image_roi_count = 0

        for shape_index, shape in enumerate(shapes):
            if shape.get("shape_type", "polygon") != "polygon":
                continue
            points = shape.get("points", [])
            if len(points) < 3:
                continue
            mask = polygon_to_feature_mask(points, image_size, feature_shape)
            bbox = mask_bbox(mask)
            if bbox is None:
                continue

            vector = roi_align_vector(
                feature,
                bbox,
                mask_feature=mask,
                output_size=args.roi_size,
            )
            if args.normalize:
                norm = np.linalg.norm(vector)
                if norm > 1e-12:
                    vector = vector / norm

            if feature_dim is None:
                feature_dim = int(vector.shape[0])
            if vector.shape[0] != feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch: {feature_path} has {vector.shape[0]}, "
                    f"expected {feature_dim}."
                )

            vectors.append(vector.astype(np.float32, copy=False))
            roi_records.append(
                {
                    "id": len(roi_records),
                    "image_path": str(image_path),
                    "image_relative": relative_posix(image_path, image_root),
                    "feature_path": str(feature_path),
                    "annotation_path": str(annotation_path),
                    "shape_index": shape_index,
                    "label": shape.get("label", ""),
                    "points": points,
                    "bbox_feature": [float(value) for value in bbox],
                }
            )
            image_roi_count += 1

        if image_roi_count == 0:
            LOGGER.warning("No polygon ROI found in %s", annotation_path)

    if not vectors:
        raise RuntimeError("No polygon ROI features were collected.")

    vectors_array = np.stack(vectors).astype(np.float32)
    index = faiss.IndexFlatL2(vectors_array.shape[1])
    index.add(vectors_array)
    index_path = output_dir / args.index_name
    metadata_path = output_dir / args.metadata_name
    faiss.write_index(index, str(index_path))
    np.save(output_dir / "roi_vectors.npy", vectors_array)

    metadata = {
        "index_type": "IndexFlatL2",
        "feature_dim": int(vectors_array.shape[1]),
        "roi_size": int(args.roi_size),
        "normalize": bool(args.normalize),
        "feature_layout": "CHW",
        "records": roi_records,
    }
    save_json(metadata_path, metadata)
    LOGGER.info(
        "Built FAISS index with %d ROI vectors; skipped %d image records. Index: %s",
        len(roi_records),
        skipped_images,
        index_path,
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from DINO feature maps and Labelme polygons."
    )
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument(
        "--image_root",
        default=None,
        help="Original image root. Usually optional when feature manifest.json exists.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--roi_size", type=int, default=7)
    parser.add_argument("--index_name", default="roi_index.faiss")
    parser.add_argument("--metadata_name", default="roi_index.json")
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize ROI features before indexing (default: enabled).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable ROI feature normalization.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return build_index(args)


if __name__ == "__main__":
    raise SystemExit(main())
