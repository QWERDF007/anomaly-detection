"""Evaluate cached score maps for the standard train/test anomaly layout.

No model is loaded and no inference is run.  The script reads cached
``.npy``/``.npz`` maps below ``score_output_dir/scores`` and evaluates the
standard layout used by both Dinomaly2 and PatchCore::

    data_root/train/good/
    data_root/test/good/
    data_root/test/<anomaly_type>/
    data_root/ground_truth/<anomaly_type>/  # or --ground_truth_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from anomaly_evaluation import (
    METRIC_NAMES,
    REPORT_METRIC_NAMES,
    compute_evaluation_metrics,
    evaluate_pixel_metrics,
    region_detection_metrics,
    training_image_score,
    write_per_image_pixel_metrics,
)
from score_workflow_common import (
    IMAGE_EXTENSIONS,
    MASK_EXTENSIONS,
    find_child_directory,
    load_score_map,
)


GROUPS = (
    ("train_good", "Train / Good"),
    ("test_good", "Test / Good"),
    ("test_anomaly", "Test / Anomaly"),
)


def _iter_images(root: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def _groups(data_root: Path) -> Dict[str, List[Path]]:
    train_root = find_child_directory(data_root, "train")
    test_root = find_child_directory(data_root, "test")
    train_good = find_child_directory(train_root, "good") if train_root else None
    test_good = find_child_directory(test_root, "good") if test_root else None
    if train_good is None or test_good is None:
        raise FileNotFoundError("data_root must contain train/good and test/good directories")
    anomalies = sorted(
        (path for path in test_root.iterdir() if path.is_dir() and path.name.lower() != "good"),
        key=lambda path: str(path).lower(),
    )
    if not anomalies:
        raise FileNotFoundError("data_root/test must contain at least one non-good anomaly directory")
    return {"train_good": [train_good], "test_good": [test_good], "test_anomaly": anomalies}


def _score_candidates(
    score_output_dir: Path,
    group_key: str,
    image_root: Path,
    image_path: Path,
    root_count: int,
) -> Iterable[Path]:
    relative_without_suffix = image_path.relative_to(image_root).with_suffix("")
    root = score_output_dir / "scores" / group_key
    roots = [root]
    if group_key == "test_anomaly":
        # Dinomaly2 stores the anomaly type only for multiple roots, while
        # PatchCore always stores it.  Accept both cache layouts.
        typed_root = root / image_root.name
        roots = [typed_root, root] if root_count > 1 else [root, typed_root]
    for candidate_root in roots:
        for extension in (".npy", ".npz"):
            yield candidate_root / relative_without_suffix.with_suffix(extension)


def _find_cached_score(
    score_output_dir: Path,
    group_key: str,
    image_root: Path,
    image_path: Path,
    root_count: int,
) -> Path:
    candidates = list(
        _score_candidates(score_output_dir, group_key, image_root, image_path, root_count)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cached score map not found for {image_path}. Tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def _find_ground_truth(
    ground_truth_dir: Path,
    image_path: Path,
    image_root: Path,
    root_count: int,
) -> Optional[Path]:
    relative = image_path.relative_to(image_root)
    roots = [ground_truth_dir]
    typed_root = ground_truth_dir / image_root.name
    if typed_root.is_dir():
        roots.insert(0, typed_root)
    if root_count > 1:
        relative_paths = [Path(image_root.name) / relative, relative]
    else:
        relative_paths = [relative]
    candidates = []
    for root in roots:
        for relative_path in relative_paths:
            for extension in MASK_EXTENSIONS:
                candidates.extend(
                    (
                        root / relative_path.with_suffix(extension),
                        root / f"{image_path.stem}{extension}",
                        root / f"{image_path.stem}_mask{extension}",
                        root / f"{image_path.stem}-mask{extension}",
                    )
                )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = []
    for root in roots:
        if root.is_dir():
            matches.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in MASK_EXTENSIONS
                and path.stem.lower()
                in {
                    image_path.stem.lower(),
                    f"{image_path.stem.lower()}_mask",
                    f"{image_path.stem.lower()}-mask",
                }
            )
    unique = sorted({path.resolve() for path in matches}, key=lambda path: str(path).lower())
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(f"Multiple GT masks match {image_path}: " + ", ".join(map(str, unique)))
    return None


def _load_mask(mask_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"Cannot read ground-truth mask: {mask_path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _print_metrics(metrics: Dict[str, float]) -> None:
    print("\nEvaluation metrics")
    print("stage                         " + "  ".join(f"{name:>10}" for name in REPORT_METRIC_NAMES))
    print(
        f"{'score_maps':<29}"
        + "  ".join(f"{metrics.get(name, float('nan')):10.6f}" for name in REPORT_METRIC_NAMES)
    )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate cached .npy/.npz maps for the standard train/test layout; no inference is run.",
    )
    parser.add_argument("-i", "--data_root", required=True, type=Path)
    parser.add_argument(
        "-s",
        "--score_output_dir",
        "--score_dir",
        required=True,
        type=Path,
        help="Score-visualization output directory containing scores/.",
    )
    parser.add_argument(
        "-gt",
        "--ground_truth_dir",
        type=Path,
        default=None,
        help="GT-mask root (default: data_root/ground_truth).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="Per-image CSV directory (default: score_output_dir).",
    )
    parser.add_argument(
        "-msz",
        "--metric_size",
        type=int,
        default=256,
        help="Square metric resolution (default: 256).",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Optional pixel threshold for region miss rate and coverage; default reuses the best P-F1 threshold.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric_size < 1:
        raise ValueError("metric_size must be positive")
    data_root = args.data_root.expanduser().resolve()
    score_output_dir = args.score_output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not score_output_dir.is_dir():
        raise FileNotFoundError(f"Score output directory does not exist: {score_output_dir}")
    ground_truth_dir = (
        args.ground_truth_dir.expanduser().resolve()
        if args.ground_truth_dir is not None
        else find_child_directory(data_root, "ground_truth")
    )
    if ground_truth_dir is None or not ground_truth_dir.is_dir():
        raise FileNotFoundError("Ground-truth masks are required; pass --ground_truth_dir or create data_root/ground_truth")

    groups = _groups(data_root)
    labels = dict(GROUPS)
    image_labels = []
    image_scores = []
    score_maps = []
    gt_maps = []
    records: List[Dict[str, object]] = []
    for group_key, _display in GROUPS:
        # Training-good maps are not part of the evaluation metrics.  They
        # may be absent when only test maps were cached, which is valid here.
        if group_key == "train_good":
            continue
        roots = groups[group_key]
        for image_root in roots:
            for image_path in tqdm(
                _iter_images(image_root),
                desc=f"Evaluate {group_key}/{image_root.name}",
                unit="image",
                dynamic_ncols=True,
            ):
                score_path = _find_cached_score(
                    score_output_dir,
                    group_key,
                    image_root,
                    image_path,
                    len(roots),
                )
                score_map = load_score_map(score_path)
                if group_key == "test_anomaly":
                    mask_path = _find_ground_truth(
                        ground_truth_dir,
                        image_path,
                        image_root,
                        len(roots),
                    )
                    if mask_path is None:
                        raise FileNotFoundError(f"Ground-truth mask not found for {image_path}")
                    gt_mask = _load_mask(mask_path, score_map.shape)
                else:
                    mask_path = None
                    gt_mask = np.zeros(score_map.shape, dtype=np.uint8)

                resized_score = cv2.resize(
                    score_map,
                    (args.metric_size, args.metric_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_gt = cv2.resize(
                    gt_mask,
                    (args.metric_size, args.metric_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                image_label = int(group_key == "test_anomaly")
                image_score = training_image_score(resized_score)
                pixel_metrics = evaluate_pixel_metrics(resized_gt, resized_score)
                image_labels.append(image_label)
                image_scores.append(image_score)
                score_maps.append(resized_score)
                gt_maps.append((resized_gt > 0).astype(np.uint8))
                records.append(
                    {
                        "stage": "score maps",
                        "group": labels[group_key],
                        "image_path": str(image_path),
                        "mask_path": str(mask_path) if mask_path is not None else "",
                        "score_path": str(score_path),
                        "image_label": image_label,
                        "image_score": image_score,
                        "gt_positive_pixels": int(np.asarray(resized_gt, dtype=bool).sum()),
                        **pixel_metrics,
                    }
                )

    if not score_maps:
        raise RuntimeError("No Test/good or Test/anomaly cached score maps were found")
    gt_array = np.stack(gt_maps, axis=0)
    score_array = np.stack(score_maps, axis=0)
    metrics = compute_evaluation_metrics(image_scores, image_labels, score_array, gt_array)
    p_f1_threshold = metrics["P-F1-Threshold"]
    metrics.update(
        region_detection_metrics(
            gt_array,
            score_array,
            p_f1_threshold if args.score_threshold is None else float(args.score_threshold),
            records,
            p_f1_threshold=p_f1_threshold,
        )
    )
    _print_metrics(metrics)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else score_output_dir
    output_path = output_dir / "pixel_metrics.csv"
    write_per_image_pixel_metrics(records, output_path)
    print(f"Per-image pixel metrics written to {output_path}")
    print(f"Done. evaluated {len(records)} cached test score maps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
