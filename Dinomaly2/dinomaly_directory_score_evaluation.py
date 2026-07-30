"""Evaluate score maps for child directories containing ``images``/``masks``.

This entry point is independent of model inference.  It scans one data root::

    data_root/
    ├── good/
    │   └── images/
    └── class_a/
        ├── images/
        └── masks/

Any child directory may omit ``masks/``; it is treated as normal and assigned
an all-zero GT.  A child directory that has ``masks/`` uses those masks.

Score maps are searched recursively under one or more ``score_output_dir``
directories and matched by image stem.  Results are printed separately for
each child directory and written to ``metrics.csv``, ``metrics.json`` and
``pixel_metrics.csv`` under ``output_dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SHARED_UTILS = PROJECT_ROOT / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(1, str(SHARED_UTILS))

from score_workflow_common import (  # noqa: E402
    CLASSIFICATION_METRIC_NAMES,
    build_score_index,
    classification_metrics,
    find_mask,
    find_score,
    iter_data_directories,
    iter_images,
    load_score_map,
    report_metric_names,
    save_classification_threshold,
    select_optimal_threshold,
    write_metric_report,
    write_per_image_report,
)
from dinomaly_evaluation import (
    METRIC_NAMES,
    evaluate_pixel_metrics,
    max_f1,
    safe_ap,
    safe_aupro,
    safe_auroc,
    training_image_score,
)


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
SCORE_EXTENSIONS = (".npy", ".npz")
REPORT_METRIC_NAMES = report_metric_names(METRIC_NAMES)


def _find_child_directory(root: Path, name: str) -> Path | None:
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def _iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def _iter_data_directories(
    data_root: Path,
    excluded_directories: Sequence[Path] = (),
) -> List[Tuple[Path, Path, Path | None]]:
    excluded = {Path(path).resolve() for path in excluded_directories}
    directories = []
    for child in sorted(
        [path for path in data_root.iterdir() if path.is_dir()],
        key=lambda path: str(path).lower(),
    ):
        if child.resolve() in excluded:
            continue
        images_dir = _find_child_directory(child, "images")
        masks_dir = _find_child_directory(child, "masks")
        if images_dir is None:
            raise FileNotFoundError(
                f"Each data_root child must contain images/: {child}"
            )
        if not _iter_images(images_dir):
            raise RuntimeError(f"No images found in {images_dir}")
        directories.append((child, images_dir, masks_dir))
    if not directories:
        raise RuntimeError(
            f"No child directories containing images/ found in "
            f"{data_root}"
        )
    return directories


def _find_mask(
    image_path: Path,
    images_dir: Path,
    masks_dir: Path | None,
) -> Path | None:
    if masks_dir is None:
        return None
    relative = image_path.relative_to(images_dir)
    candidates = [masks_dir / relative.with_suffix(extension) for extension in MASK_EXTENSIONS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    stem = image_path.stem.lower()
    matches = sorted(
        [
            path
            for path in masks_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in MASK_EXTENSIONS
            and path.stem.lower() in {stem, f"{stem}_mask", f"{stem}-mask"}
        ],
        key=lambda path: str(path).lower(),
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Mask not found for image {image_path} under {masks_dir}"
        )
    raise RuntimeError(
        f"Multiple masks match image {image_path}: "
        + ", ".join(str(path) for path in matches)
    )


def _score_keys(score_path: Path) -> Iterable[str]:
    stem = score_path.stem.lower()
    yield stem
    for extension in IMAGE_EXTENSIONS:
        suffix = extension.lower()
        if stem.endswith(suffix):
            yield stem[: -len(suffix)]


def _build_score_index(score_output_dirs: Sequence[Path]):
    index: Dict[str, List[Tuple[Path, Path]]] = {}
    for score_output_dir in score_output_dirs:
        for score_path in score_output_dir.rglob("*"):
            if not score_path.is_file() or score_path.suffix.lower() not in SCORE_EXTENSIONS:
                continue
            for key in set(_score_keys(score_path)):
                index.setdefault(key, []).append((score_output_dir, score_path))
    if not index:
        raise FileNotFoundError(
            "No .npy or .npz score maps were found under the configured "
            "score_output_dir."
        )
    return index


def _find_score(
    image_path: Path,
    data_directory: Path,
    score_index: Mapping[str, Sequence[Tuple[Path, Path]]],
) -> Tuple[Path, Path]:
    matches = list(score_index.get(image_path.stem.lower(), ()))
    unique = {}
    for score_output_dir, score_path in matches:
        unique[score_path.resolve()] = (score_output_dir, score_path)
    matches = list(unique.values())

    if len(matches) > 1:
        preferred = [
            item
            for item in matches
            if item[0].name.lower() == data_directory.name.lower()
            or data_directory.name.lower()
            in {part.lower() for part in item[1].parts}
        ]
        if len(preferred) == 1:
            matches = preferred

    if not matches:
        raise FileNotFoundError(
            f"Score map not found by filename {image_path.name} under the "
            "configured score_output_dir directories."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple score maps match {image_path.name}; "
            "use non-overlapping score_output_dir directories or put the "
            "matching score under a directory named after the data folder:\n"
            + "\n".join(str(item[1]) for item in matches)
        )
    return matches[0][0], matches[0][1]


def _load_score(score_path: Path) -> np.ndarray:
    if score_path.suffix.lower() == ".npz":
        archive = np.load(score_path)
        try:
            if not archive.files:
                raise ValueError(f"Score archive is empty: {score_path}")
            score_map = np.asarray(archive[archive.files[0]], dtype=np.float32)
        finally:
            archive.close()
    else:
        score_map = np.asarray(np.load(score_path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Score map must be 2D: {score_path}; got {score_map.shape}"
        )
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def _load_mask(mask_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(mask > 0, dtype=np.uint8)


def _evaluate_directory(
    data_directory: Path,
    images_dir: Path,
    masks_dir: Path | None,
    score_index: Mapping[str, Sequence[Tuple[Path, Path]]],
    metric_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    gt_maps = []
    score_maps = []
    image_labels = []
    image_scores = []
    records: List[Dict[str, object]] = []
    image_paths = iter_images(images_dir)

    for image_path in tqdm(
        image_paths,
        desc=f"Evaluate {data_directory.name}",
        unit="image",
        dynamic_ncols=True,
    ):
        score_path = find_score(
            image_path,
            data_directory,
            score_index,
        )
        score_map = load_score_map(score_path)
        mask_path = find_mask(image_path, images_dir, masks_dir)
        gt_mask = (
            _load_mask(mask_path, score_map.shape)
            if mask_path is not None
            else np.zeros(score_map.shape, dtype=np.uint8)
        )
        resized_gt = cv2.resize(
            gt_mask,
            (metric_size, metric_size),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_score = cv2.resize(
            score_map,
            (metric_size, metric_size),
            interpolation=cv2.INTER_LINEAR,
        )
        image_label = bool(resized_gt.any())
        image_score = training_image_score(resized_score)
        gt_maps.append(resized_gt)
        score_maps.append(resized_score)
        image_labels.append(image_label)
        image_scores.append(image_score)
        records.append(
            {
                "directory": str(data_directory),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "score_path": str(score_path),
                "image_label": int(image_label),
                "image_score": image_score,
                "gt_positive_pixels": int(resized_gt.astype(bool).sum()),
                **evaluate_pixel_metrics(
                    resized_gt,
                    resized_score,
                    show_progress=False,
                ),
            }
        )

    gt_array = np.stack(gt_maps, axis=0)
    score_array = np.stack(score_maps, axis=0)
    pixel_labels = gt_array.reshape(-1)
    pixel_scores = score_array.reshape(-1)
    metrics = {
        "I-AUROC": safe_auroc(
            np.asarray(image_labels, dtype=np.uint8),
            np.asarray(image_scores, dtype=np.float32),
        ),
        "I-AP": safe_ap(
            np.asarray(image_labels, dtype=np.uint8),
            np.asarray(image_scores, dtype=np.float32),
        ),
        "I-F1": max_f1(
            np.asarray(image_labels, dtype=np.uint8),
            np.asarray(image_scores, dtype=np.float32),
        ),
        "P-AUROC": safe_auroc(pixel_labels, pixel_scores),
        "P-AP": safe_ap(pixel_labels, pixel_scores),
        "P-F1": max_f1(pixel_labels, pixel_scores),
        "P-AUPRO": safe_aupro(
            gt_array,
            score_array,
            show_progress=False,
        ),
    }
    return metrics, records


def _write_results(
    results: Mapping[str, Mapping[str, float]],
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    write_metric_report(results, output_dir, METRIC_NAMES, "directory")

    pixel_fields = [
        "directory",
        "image_path",
        "mask_path",
        "score_path",
        "image_label",
        "image_score",
        "gt_positive_pixels",
        "P-AUROC",
        "P-AP",
        "P-F1",
        "P-AUPRO",
    ]
    write_per_image_report(records, output_dir / "pixel_metrics.csv", pixel_fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate score maps by data_root child directory. Each child "
            "must contain images/; masks/ is optional and omitted means "
            "normal. No model inference is run."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "目录示例（good 可省略 masks/）：\n"
            "  data_root/category_a/images/a.png\n"
            "  data_root/category_a/masks/a.png\n"
            "  data_root/category_b/images/b.png\n"
            "  data_root/category_b/masks/b.png\n\n"
            "score_output_dir 可重复或空格指定多个目录；脚本递归搜索"
            ".npy/.npz，并按图像 stem 匹配。"
        ),
    )
    parser.add_argument(
        "-i",
        "--data_root",
        required=True,
        type=Path,
        help="包含多个 images/ 和 masks/ 子目录的数据根目录。",
    )
    parser.add_argument(
        "-s",
        "--score_output_dir",
        "--score_dir",
        required=True,
        nargs="+",
        action="append",
        type=Path,
        help="一个或多个 score map 搜索目录；可空格分隔或重复指定。",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="评估结果目录；默认写入 data_root/evaluation_metrics。",
    )
    parser.add_argument(
        "-msz",
        "--metric_size",
        type=int,
        default=256,
        help="计算指标前统一缩放到的正方形边长（默认：256）。",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="图像判定阈值；不指定时在全部子目录上按最大平衡准确率自动选择。",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if args.metric_size < 1:
        raise ValueError("metric_size must be positive")

    score_output_dirs = []
    seen_score_dirs = set()
    for directory_group in args.score_output_dir:
        for directory in directory_group:
            directory = directory.expanduser().resolve()
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Score output directory does not exist: {directory}"
                )
            if directory not in seen_score_dirs:
                score_output_dirs.append(directory)
                seen_score_dirs.add(directory)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "evaluation_metrics"
    )
    data_directories = iter_data_directories(
        data_root,
        excluded_directories=[output_dir, *score_output_dirs],
    )
    score_index = build_score_index(score_output_dirs)
    results: Dict[str, Dict[str, float]] = {}
    all_records: List[Dict[str, object]] = []

    for data_directory, images_dir, masks_dir in data_directories:
        metrics, records = _evaluate_directory(
            data_directory,
            images_dir,
            masks_dir,
            score_index,
            args.metric_size,
        )
        directory_key = str(data_directory)
        results[directory_key] = metrics
        all_records.extend(records)
    labels = np.asarray(
        [record["image_label"] for record in all_records], dtype=np.uint8
    )
    scores = np.asarray(
        [record["image_score"] for record in all_records], dtype=np.float32
    )
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
            record for record in all_records if record["directory"] == directory
        ]
        directory_threshold_metrics = classification_metrics(
            [record["image_label"] for record in directory_records],
            [record["image_score"] for record in directory_records],
            threshold,
        )
        metrics.update(
            {
                name: directory_threshold_metrics[name]
                for name in CLASSIFICATION_METRIC_NAMES
            }
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

    _write_results(results, all_records, output_dir)
    save_classification_threshold(
        output_dir / "classification_threshold.json",
        threshold,
        threshold_method,
        global_threshold_metrics,
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
