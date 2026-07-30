"""Evaluate cached PatchCore score maps by child directory, without inference.

``data_root`` contains child directories with ``images/`` and optional
``masks/``.  Any child without ``masks/`` is treated as a normal class.  Score
maps are searched recursively in one or more output directories and matched
by image filename stem.

When a score map was made by ``patchcore_score_visualization.py``, its
``.npy.json`` sidecar supplies the native PatchCore image score.  That makes
image-level metrics identical to ``train.py``.  External maps without this
sidecar fall back to the map maximum and are marked in ``pixel_metrics.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PROJECT_ROOT = ROOT.parent
SHARED_UTILS = PROJECT_ROOT / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS))

from score_workflow_common import (  # noqa: E402
    build_score_index,
    find_mask,
    find_score,
    iter_data_directories,
    iter_images,
    load_score_map as common_load_score_map,
    write_metric_report,
    write_per_image_report,
)

from patchcore_evaluation import (
    CLASSIFICATION_METRIC_NAMES,
    METRIC_NAMES,
    REPORT_METRIC_NAMES,
    classification_metrics,
    compute_evaluation_metrics,
    evaluate_pixel_metrics,
    load_score_map,
    select_optimal_threshold,
)
from patchcore.datasets.custom import get_data_transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
SCORE_EXTENSIONS = (".npy", ".npz")


def _find_child_directory(root: Path, name: str) -> Optional[Path]:
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def _iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def _iter_data_directories(
    data_root: Path, excluded_directories: Sequence[Path] = ()
) -> List[Tuple[Path, Path, Optional[Path]]]:
    excluded = {Path(path).resolve() for path in excluded_directories}
    result = []
    for child in sorted(
        (path for path in data_root.iterdir() if path.is_dir()), key=lambda path: str(path).lower()
    ):
        if child.resolve() in excluded:
            continue
        images_dir = _find_child_directory(child, "images")
        masks_dir = _find_child_directory(child, "masks")
        if images_dir is None:
            raise FileNotFoundError(f"Each data_root child must contain images/: {child}")
        if not _iter_images(images_dir):
            raise RuntimeError(f"No images found in {images_dir}")
        result.append((child, images_dir, masks_dir))
    if not result:
        raise RuntimeError(f"No child directories containing images/ found in {data_root}")
    return result


def _find_mask(image_path: Path, images_dir: Path, masks_dir: Optional[Path]) -> Optional[Path]:
    if masks_dir is None:
        return None
    relative = image_path.relative_to(images_dir)
    for extension in MASK_EXTENSIONS:
        candidate = masks_dir / relative.with_suffix(extension)
        if candidate.is_file():
            return candidate
    stem = image_path.stem.lower()
    matches = sorted(
        (
            path
            for path in masks_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in MASK_EXTENSIONS
            and path.stem.lower() in {stem, f"{stem}_mask", f"{stem}-mask"}
        ),
        key=lambda path: str(path).lower(),
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Mask not found for image {image_path} under {masks_dir}")
    raise RuntimeError(f"Multiple masks match {image_path}: " + ", ".join(map(str, matches)))


def _score_keys(score_path: Path) -> Iterable[str]:
    stem = score_path.stem.lower()
    yield stem
    for extension in IMAGE_EXTENSIONS:
        if stem.endswith(extension):
            yield stem[: -len(extension)]


def _build_score_index(score_output_dirs: Sequence[Path]) -> Dict[str, List[Tuple[Path, Path]]]:
    index: Dict[str, List[Tuple[Path, Path]]] = {}
    for root in score_output_dirs:
        for score_path in root.rglob("*"):
            if not score_path.is_file() or score_path.suffix.lower() not in SCORE_EXTENSIONS:
                continue
            for key in set(_score_keys(score_path)):
                index.setdefault(key, []).append((root, score_path))
    if not index:
        raise FileNotFoundError("No .npy or .npz score maps found under score_output_dir.")
    return index


def _find_score(
    image_path: Path, data_directory: Path, score_index: Mapping[str, Sequence[Tuple[Path, Path]]]
) -> Path:
    matches = list(score_index.get(image_path.stem.lower(), ()))
    unique = {score_path.resolve(): score_path for _root, score_path in matches}
    matches = list(unique.values())
    if len(matches) > 1:
        directory_name = data_directory.name.lower()
        preferred = [
            path
            for path in matches
            if directory_name in {part.lower() for part in path.parts}
        ]
        if len(preferred) == 1:
            matches = preferred
    if not matches:
        raise FileNotFoundError(
            f"Score map not found by filename {image_path.name} under score_output_dir."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple score maps match {image_path.name}:\n" + "\n".join(map(str, matches))
        )
    return matches[0]


def _load_sidecar(score_path: Path) -> Dict[str, object]:
    sidecar = score_path.with_suffix(score_path.suffix + ".json")
    if not sidecar.is_file():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _native_image_score(score_path: Path, score_map: np.ndarray) -> Tuple[float, str, Dict[str, object]]:
    metadata = _load_sidecar(score_path)
    try:
        score = float(metadata["image_score"])
    except (KeyError, TypeError, ValueError):
        score = float("nan")
    if np.isfinite(score):
        return score, "patchcore-native-sidecar", metadata
    return float(score_map.max()), "score-map-max-fallback", metadata


def _metadata_transform(metadata: Mapping[str, object], args):
    resize = args.resize if args.resize is not None else metadata.get("resize")
    imagesize = args.imagesize if args.imagesize is not None else metadata.get("imagesize")
    try:
        resize = int(resize) if resize is not None else None
        imagesize = int(imagesize) if imagesize is not None else None
    except (TypeError, ValueError):
        return None
    if resize is None or imagesize is None or resize < 1 or imagesize < 1:
        return None
    _image_transform, mask_transform = get_data_transforms(resize, imagesize)
    return mask_transform


def _load_mask(mask_path: Path, shape: Tuple[int, int], mask_transform) -> np.ndarray:
    if mask_transform is not None:
        mask = np.squeeze(mask_transform(Image.open(mask_path).convert("L")).numpy() > 0)
        mask = mask.astype(np.uint8)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Cannot read mask: {mask_path}")
        mask = (mask > 0).astype(np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _resize_for_metrics(score_map: np.ndarray, gt_mask: np.ndarray, metric_size: Optional[int]):
    if metric_size is None:
        return score_map, gt_mask
    size = (metric_size, metric_size)
    return (
        cv2.resize(score_map, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(gt_mask, size, interpolation=cv2.INTER_NEAREST),
    )


def _evaluate_directory(
    data_directory: Path,
    images_dir: Path,
    masks_dir: Optional[Path],
    score_index: Mapping[str, Sequence[Tuple[Path, Path]]],
    args,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    gt_maps = []
    score_maps = []
    image_labels = []
    image_scores = []
    records: List[Dict[str, object]] = []
    for image_path in tqdm(
        iter_images(images_dir), desc=f"Evaluate {data_directory.name}", unit="image", dynamic_ncols=True
    ):
        score_path = find_score(image_path, data_directory, score_index)
        score_map = common_load_score_map(score_path)
        image_score, image_score_source, metadata = _native_image_score(score_path, score_map)
        mask_path = find_mask(image_path, images_dir, masks_dir)
        mask_transform = _metadata_transform(metadata, args)
        gt_mask = (
            _load_mask(mask_path, score_map.shape, mask_transform)
            if mask_path is not None
            else np.zeros(score_map.shape, dtype=np.uint8)
        )
        score_map, gt_mask = _resize_for_metrics(score_map, gt_mask, args.metric_size)
        image_label = int(gt_mask.any())
        pixel_metrics = evaluate_pixel_metrics(gt_mask, score_map)
        gt_maps.append(gt_mask)
        score_maps.append(score_map)
        image_labels.append(image_label)
        image_scores.append(image_score)
        records.append(
            {
                "directory": str(data_directory),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "score_path": str(score_path),
                "image_label": image_label,
                "image_score": image_score,
                "image_score_source": image_score_source,
                "gt_positive_pixels": int(gt_mask.sum()),
                **{name: pixel_metrics[name] for name in pixel_metrics if name.startswith("P-")},
            }
        )
    return (
        compute_evaluation_metrics(
            image_scores, image_labels, np.stack(score_maps), np.stack(gt_maps)
        ),
        records,
    )


def _write_results(
    results: Mapping[str, Mapping[str, float]], records: Sequence[Mapping[str, object]], output_dir: Path
) -> None:
    write_metric_report(results, output_dir, METRIC_NAMES, "directory")
    fields = (
        "directory",
        "mask_path",
        "image_score_source",
        "image_path",
        "score_path",
        "image_label",
        "image_score",
        "gt_positive_pixels",
        "P-AUROC",
        "P-AP",
        "P-F1",
        "P-AUPRO",
    )
    write_per_image_report(records, output_dir / "pixel_metrics.csv", fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PatchCore score maps by data_root child directory; no inference is run.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "每个 data_root 子目录必须有 images/，masks/ 可省略，省略即正常样本。\n"
            "score_output_dir 可指定一个或多个目录，脚本递归查找 .npy/.npz 并按图像 stem 匹配。\n"
            "由 patchcore_score_visualization.py 生成的 .npy.json 会提供训练同口径的 image score。"
        ),
    )
    parser.add_argument("-i", "--data_root", type=Path, required=True)
    parser.add_argument("-s", "--score_output_dir", "--score_dir", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, default=None)
    parser.add_argument("--resize", type=int, default=None, help="GT 变换 Resize；默认读取 score sidecar。")
    parser.add_argument("--imagesize", "--crop_size", dest="imagesize", type=int, default=None, help="GT 变换 CenterCrop；默认读取 score sidecar。")
    parser.add_argument("--metric_size", type=int, default=None, help="可选统一指标大小；设置后像素指标不再是训练的原始分辨率。")
    parser.add_argument("--score_threshold", type=float, default=None, help="图像判定阈值；不指定时在所有子目录上按最大平衡准确率自动选择。")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric_size is not None and args.metric_size < 1:
        raise ValueError("metric_size must be positive")
    if (args.resize is None) != (args.imagesize is None):
        raise ValueError("--resize and --imagesize must be given together")
    if args.resize is not None and (args.resize < 1 or args.imagesize < 1):
        raise ValueError("resize and imagesize must be positive")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    score_dirs = []
    seen = set()
    for group in args.score_output_dir:
        for directory in group:
            directory = directory.expanduser().resolve()
            if not directory.is_dir():
                raise FileNotFoundError(f"Score output directory does not exist: {directory}")
            if directory not in seen:
                score_dirs.append(directory)
                seen.add(directory)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else data_root / "evaluation_metrics"
    directories = iter_data_directories(data_root, excluded_directories=[output_dir, *score_dirs])
    score_index = build_score_index(score_dirs)
    results: Dict[str, Dict[str, float]] = {}
    records: List[Dict[str, object]] = []
    for data_directory, images_dir, masks_dir in directories:
        metrics, directory_records = _evaluate_directory(data_directory, images_dir, masks_dir, score_index, args)
        results[str(data_directory)] = metrics
        records.extend(directory_records)

    labels = np.asarray([record["image_label"] for record in records], dtype=np.uint8)
    scores = np.asarray([record["image_score"] for record in records], dtype=np.float32)
    if args.score_threshold is None:
        threshold, threshold_method, global_threshold_metrics = select_optimal_threshold(
            labels, scores
        )
    else:
        threshold = float(args.score_threshold)
        threshold_method = "manual"
        global_threshold_metrics = classification_metrics(labels, scores, threshold)

    for directory, metrics in results.items():
        directory_records = [
            record for record in records if record["directory"] == directory
        ]
        directory_metrics = classification_metrics(
            [record["image_label"] for record in directory_records],
            [record["image_score"] for record in directory_records],
            threshold,
        )
        metrics.update(
            {name: directory_metrics[name] for name in CLASSIFICATION_METRIC_NAMES}
        )
        print(f"\n===== {directory} =====", flush=True)
        print(
            "  "
            + "  ".join(
                f"{name}={metrics.get(name, float('nan')):.6f}"
                for name in REPORT_METRIC_NAMES
            ),
            flush=True,
        )
    _write_results(results, records, output_dir)
    with (output_dir / "classification_threshold.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "threshold": threshold,
                "method": threshold_method,
                **{
                    name: global_threshold_metrics[name]
                    for name in CLASSIFICATION_METRIC_NAMES
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"\nImage threshold={threshold:.6f} ({threshold_method}); "
        f"FPR={global_threshold_metrics['FPR']:.6f}, "
        f"TNR={global_threshold_metrics['TNR']:.6f}, "
        f"Accuracy={global_threshold_metrics['Accuracy']:.6f}",
        flush=True,
    )
    print(f"\nMetrics written to {output_dir / 'metrics.csv'}")
    print(f"Per-image pixel metrics written to {output_dir / 'pixel_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
