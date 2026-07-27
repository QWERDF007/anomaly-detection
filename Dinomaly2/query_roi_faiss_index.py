"""Plot FAISS distances for ROIs found in Dinomaly2 score maps.

For every score-map ROI above score_threshold, the corresponding DINO feature
map is ROIAligned and searched against the ROI FAISS index. The script prints
all matching distances and writes only one distance-distribution plot.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roi_feature_utils import (
    load_feature_map,
    load_search_index,
    l2_normalize,
    mask_bbox,
    resize_mask_to_feature,
    roi_align_vector,
)


LOGGER = logging.getLogger("query_roi_faiss_index")


def iter_score_paths(source: Path, recursive: bool = True) -> list[Path]:
    source = Path(source).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".npy":
            raise ValueError(f"Expected a score-map .npy file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Score-map path does not exist: {source}")
    iterator = source.rglob("*.npy") if recursive else source.glob("*.npy")
    return sorted(
        [path for path in iterator if path.is_file()],
        key=lambda path: str(path).lower(),
    )


def feature_path_for_score(
    score_path: Path,
    score_root: Path,
    feature_dir: Path,
) -> Optional[Path]:
    feature_dir = Path(feature_dir)
    try:
        relative = score_path.relative_to(score_root)
    except ValueError:
        relative = Path(score_path.name)

    candidates = [
        feature_dir / relative.with_suffix(".npy"),
        feature_dir / f"{score_path.stem}.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(feature_dir.rglob(f"{score_path.stem}.npy"))
    if not matches:
        matches = sorted(feature_dir.rglob(f"*_{score_path.stem}.npy"))
    return matches[0] if matches else None


def load_score_map(path: Path) -> np.ndarray:
    score_map = np.asarray(np.load(path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D score map in {path}, got shape {score_map.shape}"
        )
    return np.nan_to_num(score_map)


def find_components(
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
        components.append(
            {
                "id": int(component_id),
                "bbox": [float(x), float(y), float(x + width), float(y + height)],
                "area": int(area),
                "mask": labels == component_id,
            }
        )
    return components


def load_metadata(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_distance_distribution(
    distances: list[float],
    output_path: Path,
    bins: int,
) -> None:
    if not distances:
        raise RuntimeError(
            "No ROI distances were collected. Check score_threshold and "
            "score/feature file matching."
        )

    values = np.asarray(distances, dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        margin = max(abs(low) * 0.05, 1e-6)
        edges = np.linspace(low - margin, high + margin, bins + 1)
    else:
        edges = np.linspace(low, high, bins + 1)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=edges, color="darkorange", alpha=0.8)
    plt.axvline(
        float(values.mean()),
        color="blue",
        linestyle="--",
        label=f"mean={values.mean():.6f}",
    )
    plt.xlabel("FAISS squared L2 distance")
    plt.ylabel("Frequency")
    plt.title("ROI FAISS Distance Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run(args) -> int:
    score_source = Path(args.score_dir).expanduser()
    score_root = score_source if score_source.is_dir() else score_source.parent
    feature_dir = Path(args.feature_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(Path(args.metadata).expanduser())
    roi_size = int(args.roi_size or metadata["roi_size"])
    normalize = bool(metadata.get("normalize", True))
    expected_dim = int(metadata["feature_dim"])
    index, _resources = load_search_index(
        Path(args.index).expanduser(),
        on_gpu=args.faiss_on_gpu and args.gpu >= 0,
        gpu_id=args.gpu,
    )
    if index.d != expected_dim:
        raise ValueError(
            f"FAISS dimension {index.d} does not match metadata dimension "
            f"{expected_dim}."
        )

    score_paths = iter_score_paths(score_source, recursive=args.recursive)
    if not score_paths:
        raise RuntimeError(f"No score-map .npy files found in {score_source}.")

    distances: list[float] = []
    processed_images = 0
    processed_rois = 0
    for score_path in score_paths:
        feature_path = feature_path_for_score(
            score_path,
            score_root,
            feature_dir,
        )
        if feature_path is None:
            LOGGER.warning(
                "Skipping %s: matching DINO feature .npy not found.",
                score_path,
            )
            continue

        score_map = load_score_map(score_path)
        components = find_components(
            score_map,
            args.score_threshold,
            args.min_area,
        )
        feature = load_feature_map(feature_path)
        feature_shape = feature.shape[-2:]
        channels = feature.shape[0]
        if channels != expected_dim:
            raise ValueError(
                f"Feature dimension {channels} in {feature_path} does not "
                f"match FAISS dimension {expected_dim}."
            )

        processed_images += 1
        for component in components:
            mask_feature = resize_mask_to_feature(
                component["mask"],
                feature_shape,
            )
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
            searched_distances, neighbours = index.search(
                np.asarray([vector], dtype=np.float32),
                1,
            )
            distance = float(searched_distances[0, 0])
            matched_index = int(neighbours[0, 0])
            distances.append(distance)
            processed_rois += 1

            print(
                f"{score_path.name} "
                f"ROI {component['id']} "
                f"bbox={component['bbox']} "
                f"distance={distance:.6f} "
                f"matched_index={matched_index}"
            )

        LOGGER.info(
            "%s: score_regions=%d",
            score_path.name,
            len(components),
        )

    output_path = output_dir / args.distance_plot
    plot_distance_distribution(distances, output_path, args.bins)
    print(
        f"Distance summary: count={len(distances)}, "
        f"max={max(distances):.6f}, mean={np.mean(distances):.6f}"
    )
    print(f"Distance distribution saved to: {output_path}")
    LOGGER.info(
        "Processed score maps=%d, ROIs=%d",
        processed_images,
        processed_rois,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find ROIs in Dinomaly2 score maps, ROIAlign DINO features, "
            "and plot FAISS matching distances."
        )
    )
    parser.add_argument(
        "--score_dir",
        "--score_npy",
        dest="score_dir",
        required=True,
        help="Dinomaly2 score-map .npy file or directory.",
    )
    parser.add_argument(
        "--feature_dir",
        "--feature_npy",
        dest="feature_dir",
        required=True,
        help="DINO NCHW/CHW feature .npy directory.",
    )
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata JSON generated by build_roi_faiss_index.py.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_threshold", type=float, required=True)
    parser.add_argument("--min_area", type=int, default=16)
    parser.add_argument("--roi_size", type=int, default=None)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument(
        "--distance_plot",
        default="distance_distribution.png",
    )
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument(
        "--faiss_on_gpu",
        action="store_true",
        default=True,
        help="Search FAISS on GPU (enabled by default).",
    )
    parser.add_argument(
        "--no-faiss-on-gpu",
        dest="faiss_on_gpu",
        action="store_false",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
    )
    parser.set_defaults(recursive=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.score_threshold <= 0:
        raise ValueError("--score_threshold must be greater than 0.")
    if args.min_area < 1:
        raise ValueError("--min_area must be at least 1.")
    if args.bins < 1:
        raise ValueError("--bins must be greater than 0.")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
