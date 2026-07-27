"""Convert LabelMe JSON annotations to three-value training masks.

The conversion rule is intentionally simple:

* ``0``: background/normal
* ``good_value``: a shape whose label is ``good``
* ``anomaly_value``: every other labeled shape

The comparison with ``good`` ignores leading/trailing whitespace and letter
case.  No other label mapping is performed.  If good and anomaly shapes
overlap, anomaly takes precedence.

Example::

    python convert_labelme_to_mask.py \
        --input_dir /path/to/labelme_annotations \
        --output_dir /path/to/masks \
        --image_root /path/to/images

The relative directory structure below ``input_dir`` is retained in
``output_dir`` and each ``.json`` is converted to a same-named ``.png``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def _positive_int(value: Any) -> Optional[int]:
    """Return a positive integer or ``None`` for an invalid value."""

    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result > 0 else None


def _size_from_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def _size_from_image_data(image_data: Any) -> Optional[tuple[int, int]]:
    """Read the dimensions of embedded LabelMe image data, if available."""

    if not isinstance(image_data, str) or not image_data:
        return None

    # LabelMe normally stores raw base64, but accepting a data URI costs
    # nothing and makes the fallback work with more exported JSON variants.
    encoded = image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data
    try:
        data = base64.b64decode(encoded)
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
        return int(width), int(height)
    except (ValueError, OSError, TypeError):
        return None


def _image_candidates(
    annotation: dict[str, Any],
    annotation_path: Path,
    input_dir: Path,
    image_root: Optional[Path],
) -> Iterable[Path]:
    image_path = annotation.get("imagePath")
    if not isinstance(image_path, str) or not image_path.strip():
        return

    # LabelMe can write Windows separators even when the JSON is processed on
    # another platform.  Normalize them before constructing candidates.
    normalized = image_path.replace("\\", os.sep)
    relative_image_path = Path(normalized)

    if relative_image_path.is_absolute():
        yield relative_image_path
    else:
        yield annotation_path.parent / relative_image_path
        if image_root is not None:
            yield image_root / relative_image_path

            # This also handles an image_root that mirrors input_dir, e.g.
            # annotations/Train/good/a.json and images/Train/good/a.jpg.
            try:
                relative_annotation = annotation_path.relative_to(input_dir)
            except ValueError:
                relative_annotation = None
            if relative_annotation is not None:
                yield image_root / relative_annotation.parent / relative_image_path.name


def _annotation_size(
    annotation: dict[str, Any],
    annotation_path: Path,
    input_dir: Path,
    image_root: Optional[Path],
) -> tuple[int, int]:
    width = _positive_int(annotation.get("imageWidth"))
    height = _positive_int(annotation.get("imageHeight"))
    if width is not None and height is not None:
        return width, height

    for candidate in _image_candidates(
        annotation, annotation_path, input_dir, image_root
    ):
        if candidate.is_file():
            try:
                return _size_from_image(candidate)
            except (OSError, ValueError):
                continue

    embedded_size = _size_from_image_data(annotation.get("imageData"))
    if embedded_size is not None:
        return embedded_size

    raise ValueError(
        "imageWidth/imageHeight are missing or invalid, and the image could "
        f"not be found for {annotation_path}"
    )


def _shape_points(shape: dict[str, Any]) -> Optional[np.ndarray]:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) == 0:
        return None

    converted = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        converted.append((x, y))
    return np.asarray(converted, dtype=np.float32)


def _to_int_points(points: np.ndarray) -> np.ndarray:
    # LabelMe coordinates are floats.  OpenCV drawing operations use integer
    # coordinates; rounding matches the nearest source pixel.
    return np.rint(points).astype(np.int32).reshape((-1, 1, 2))


def _draw_shape(canvas: np.ndarray, shape: dict[str, Any], line_width: int) -> str:
    """Draw one shape and return its status.

    The return value is one of ``drawn``, ``invalid`` or ``unsupported`` and
    is used only for the conversion summary.
    """

    points = _shape_points(shape)
    if points is None:
        return "invalid"

    shape_type = shape.get("shape_type") or "polygon"
    if not isinstance(shape_type, str):
        return "unsupported"
    shape_type = shape_type.lower()
    color = 255

    if shape_type == "polygon":
        if len(points) < 3:
            return "invalid"
        cv2.fillPoly(canvas, [_to_int_points(points)], color)
        return "drawn"

    if shape_type == "rectangle":
        if len(points) < 2:
            return "invalid"
        first, second = points[0], points[1]
        top_left = np.rint(np.minimum(first, second)).astype(np.int32)
        bottom_right = np.rint(np.maximum(first, second)).astype(np.int32)
        cv2.rectangle(
            canvas,
            tuple(top_left.tolist()),
            tuple(bottom_right.tolist()),
            color,
            thickness=-1,
        )
        return "drawn"

    if shape_type == "circle":
        if len(points) < 2:
            return "invalid"
        center = np.rint(points[0]).astype(np.int32)
        radius = float(np.linalg.norm(points[1] - points[0]))
        radius = max(1, int(round(radius)))
        cv2.circle(canvas, tuple(center.tolist()), radius, color, thickness=-1)
        return "drawn"

    if shape_type in {"line", "linestrip"}:
        if len(points) < 2:
            return "invalid"
        cv2.polylines(
            canvas,
            [_to_int_points(points)],
            isClosed=False,
            color=color,
            thickness=line_width,
            lineType=cv2.LINE_8,
        )
        return "drawn"

    if shape_type == "point":
        center = np.rint(points[0]).astype(np.int32)
        radius = max(1, line_width // 2)
        cv2.circle(canvas, tuple(center.tolist()), radius, color, thickness=-1)
        return "drawn"

    return "unsupported"


def _label_is_good(label: Any) -> bool:
    return isinstance(label, str) and label.strip().lower() == "good"


def convert_annotation(
    annotation_path: Path,
    output_path: Path,
    input_dir: Path,
    image_root: Optional[Path],
    good_value: int,
    anomaly_value: int,
    line_width: int,
    label_counts: Counter[str],
    shape_status_counts: Counter[str],
) -> tuple[int, int, int]:
    """Convert one JSON file.

    Returns ``(good_shape_count, anomaly_shape_count, overlap_pixel_count)``.
    """

    with annotation_path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)
    if not isinstance(annotation, dict):
        raise ValueError("LabelMe annotation must contain a JSON object")

    width, height = _annotation_size(
        annotation, annotation_path, input_dir, image_root
    )
    good_canvas = np.zeros((height, width), dtype=np.uint8)
    anomaly_canvas = np.zeros((height, width), dtype=np.uint8)

    good_count = 0
    anomaly_count = 0
    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError("LabelMe 'shapes' must be a list")

    for shape in shapes:
        if not isinstance(shape, dict):
            shape_status_counts["invalid"] += 1
            continue

        label = shape.get("label", "<missing>")
        label_key = str(label).strip() if label is not None else "<missing>"
        label_counts[label_key] += 1
        target = good_canvas if _label_is_good(label) else anomaly_canvas
        status = _draw_shape(target, shape, line_width)
        shape_status_counts[status] += 1
        if status != "drawn":
            continue
        if target is good_canvas:
            good_count += 1
        else:
            anomaly_count += 1

    overlap_pixels = int(np.count_nonzero(good_canvas & anomaly_canvas))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[good_canvas != 0] = np.uint8(good_value)
    # Anomaly deliberately overwrites good on overlapping pixels.
    mask[anomaly_canvas != 0] = np.uint8(anomaly_value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), mask):
        raise OSError(f"OpenCV could not write mask: {output_path}")
    return good_count, anomaly_count, overlap_pixels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LabelMe JSON files to masks where label 'good' is "
            "good and every other label is anomaly."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing LabelMe .json files (searched recursively).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory in which the relative .png mask tree is written.",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=None,
        help=(
            "Optional image root used when imageWidth/imageHeight are absent "
            "from a JSON file."
        ),
    )
    parser.add_argument(
        "--good_value",
        type=int,
        default=1,
        help="Pixel value for label 'good' (default: 1).",
    )
    parser.add_argument(
        "--anomaly_value",
        type=int,
        default=2,
        help="Pixel value for every label other than 'good' (default: 2).",
    )
    parser.add_argument(
        "--line_width",
        type=int,
        default=3,
        help="Width in pixels for line/linestrip/point shapes (default: 3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output PNG.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve() if args.image_root else None

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not 1 <= args.good_value <= 255:
        print("--good_value must be in [1, 255].", file=sys.stderr)
        return 2
    if not 1 <= args.anomaly_value <= 255:
        print("--anomaly_value must be in [1, 255].", file=sys.stderr)
        return 2
    if args.good_value == args.anomaly_value:
        print("--good_value and --anomaly_value must differ.", file=sys.stderr)
        return 2
    if args.line_width < 1:
        print("--line_width must be at least 1.", file=sys.stderr)
        return 2

    json_paths = sorted(
        (path for path in input_dir.rglob("*.json") if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    if not json_paths:
        print(f"No .json files found under {input_dir}.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    label_counts: Counter[str] = Counter()
    shape_status_counts: Counter[str] = Counter()
    converted = 0
    skipped = 0
    failed = 0
    total_good_shapes = 0
    total_anomaly_shapes = 0
    total_overlap_pixels = 0

    progress = tqdm(json_paths, desc="Convert LabelMe masks", unit="file")
    for annotation_path in progress:
        relative_path = annotation_path.relative_to(input_dir).with_suffix(".png")
        output_path = output_dir / relative_path
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            good_shapes, anomaly_shapes, overlap_pixels = convert_annotation(
                annotation_path=annotation_path,
                output_path=output_path,
                input_dir=input_dir,
                image_root=image_root,
                good_value=args.good_value,
                anomaly_value=args.anomaly_value,
                line_width=args.line_width,
                label_counts=label_counts,
                shape_status_counts=shape_status_counts,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failed += 1
            print(f"[ERROR] {annotation_path}: {error}", file=sys.stderr)
            continue
        converted += 1
        total_good_shapes += good_shapes
        total_anomaly_shapes += anomaly_shapes
        total_overlap_pixels += overlap_pixels

    print(f"Converted: {converted}")
    print(f"Skipped (exists, use --overwrite to replace): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print(
        "Shape counts: "
        f"good={total_good_shapes}, anomaly={total_anomaly_shapes}"
    )
    if total_overlap_pixels:
        print(
            "Good/anomaly overlap pixels (anomaly takes precedence): "
            f"{total_overlap_pixels}"
        )
    if label_counts:
        print("Labels:")
        for label, count in sorted(label_counts.items(), key=lambda item: item[0].lower()):
            category = "good" if label.strip().lower() == "good" else "anomaly"
            print(f"  {label!r}: {count} ({category})")
    if shape_status_counts.get("unsupported") or shape_status_counts.get("invalid"):
        print(
            "Shape status: "
            f"drawn={shape_status_counts.get('drawn', 0)}, "
            f"invalid={shape_status_counts.get('invalid', 0)}, "
            f"unsupported={shape_status_counts.get('unsupported', 0)}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
