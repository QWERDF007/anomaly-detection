"""Create heatmap and binary-mask images from saved ``.npy`` score maps.

The score maps saved by ``predict.py`` contain the raw anomaly map.  This
script normalizes the map with a user-provided threshold, converts it to a
JET heatmap, fuses it with the original image, and saves the fused heatmap
and thresholded binary mask into separate directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import tqdm


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def iter_image_paths(source: Path, recursive: bool) -> list[Path]:
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {source}")

    iterator: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    return sorted(
        [
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def score_path_for_image(
    image_path: Path,
    score_source: Path,
    image_root: Optional[Path],
) -> Optional[Path]:
    """Find the score map corresponding to an image."""

    score_source = Path(score_source)
    if score_source.is_file():
        return score_source if score_source.suffix.lower() == ".npy" else None
    if not score_source.is_dir():
        return None

    candidates = []
    if image_root is not None:
        try:
            relative = image_path.relative_to(image_root)
            candidates.append(score_source / relative.with_suffix(".npy"))
        except ValueError:
            pass
    candidates.append(score_source / f"{image_path.stem}.npy")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # PatchCore may prefix score-map names with an ordinal, for example
    # 00001_image.npy.  Dinomaly2 normally writes image.npy.
    matches = sorted(score_source.rglob(f"{image_path.stem}.npy"))
    if not matches:
        matches = sorted(score_source.rglob(f"*_{image_path.stem}.npy"))
    return matches[0] if matches else None


def load_score_map(path: Path) -> np.ndarray:
    score_map = np.asarray(np.load(path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D score map in {path}, got shape {score_map.shape}"
        )
    return np.nan_to_num(score_map, copy=False)


def create_fused_image(
    image_path: Path,
    score_path: Path,
    threshold: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    score_map = load_score_map(score_path)
    height, width = image.shape[:2]
    score_map = cv2.resize(
        score_map,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    normalized = np.clip(score_map / threshold, 0.0, 1.0)
    heatmap = cv2.applyColorMap(
        np.uint8(np.round(normalized * 255.0)),
        cv2.COLORMAP_JET,
    )
    mask = np.where(score_map >= threshold, 255, 0).astype(np.uint8)
    blended = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0.0)
    fused = image.copy()
    anomaly_pixels = mask > 0
    fused[anomaly_pixels] = blended[anomaly_pixels]
    return fused, mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse original images with anomaly heatmaps generated from .npy "
            "score maps."
        )
    )
    parser.add_argument(
        "--input",
        "--image",
        "--image_root",
        dest="image_source",
        required=True,
        help="Original image file or image directory.",
    )
    parser.add_argument(
        "--npy",
        "--score_dir",
        "--score_root",
        dest="score_source",
        required=True,
        help="Matching .npy score-map file or directory.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root directory. Outputs are written to heatmap/ and mask/.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Score value mapped to the maximum heatmap intensity.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Heatmap fusion weight, in [0, 1]. Default: 0.5.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix added before .png in both output directories.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not search image subdirectories.",
    )
    parser.set_defaults(recursive=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.threshold <= 0:
        raise ValueError("--threshold must be greater than 0.")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    image_source = Path(args.image_source).expanduser()
    score_source = Path(args.score_source).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    heatmap_dir = output_dir / "heatmap"
    mask_dir = output_dir / "mask"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(image_source, args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_source}")
    if score_source.is_file() and len(image_paths) != 1:
        raise ValueError(
            "A single .npy score-map file can only be used with one image. "
            "Use a score-map directory for batch processing."
        )

    image_root = image_source if image_source.is_dir() else None
    processed = 0
    skipped = 0
    for image_path in tqdm.tqdm(image_paths, total=len(image_paths)):
        score_path = score_path_for_image(image_path, score_source, image_root)
        if score_path is None:
            print(f"[skip] score map not found for {image_path}")
            skipped += 1
            continue

        fused, mask = create_fused_image(
            image_path=image_path,
            score_path=score_path,
            threshold=args.threshold,
            alpha=args.alpha,
        )

        if image_root is not None:
            relative = image_path.relative_to(image_root)
            heatmap_path = heatmap_dir / relative.with_name(
                f"{relative.stem}{args.suffix}.png"
            )
            mask_path = mask_dir / relative.with_name(
                f"{relative.stem}{args.suffix}.png"
            )
        else:
            heatmap_path = heatmap_dir / f"{image_path.stem}{args.suffix}.png"
            mask_path = mask_dir / f"{image_path.stem}{args.suffix}.png"
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(heatmap_path), fused):
            raise IOError(f"Failed to write heatmap image: {heatmap_path}")
        if not cv2.imwrite(str(mask_path), mask):
            raise IOError(f"Failed to write binary mask: {mask_path}")

        # print(
        #     f"[ok] {image_path} + {score_path} -> "
        #     f"heatmap: {heatmap_path}, mask: {mask_path}"
        # )
        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
