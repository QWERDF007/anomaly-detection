"""Model-agnostic helpers for anomaly-score visualization and evaluation.

The functions here intentionally know nothing about Dinomaly2 or PatchCore.
Each model keeps its own inference and training-compatible AUROC/AP/F1/AUPRO
adapter, while both use this module for score-file lookup, directory layout,
threshold selection, classification reports, score tables and distributions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STANDARD_GROUPS = (
    ("train_good", "Train / Good"),
    ("test_good", "Test / Good"),
    ("test_anomaly", "Test / Anomaly"),
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
CLASSIFICATION_METRIC_NAMES = ("Threshold", "FPR", "TNR", "Accuracy")


def find_child_directory(root: Path, name: str) -> Optional[Path]:
    """Find an immediate child directory case-insensitively."""

    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def iter_images(images_dir: Path) -> List[Path]:
    """Recursively return supported image files in deterministic order."""

    return sorted(
        (
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def iter_data_directories(
    data_root: Path, excluded_directories: Sequence[Path] = ()
) -> List[Tuple[Path, Path, Optional[Path]]]:
    """Discover direct children containing ``images/`` and optional ``masks/``."""

    excluded = {Path(path).resolve() for path in excluded_directories}
    directories = []
    for child in sorted(
        (path for path in data_root.iterdir() if path.is_dir()),
        key=lambda path: str(path).lower(),
    ):
        if child.resolve() in excluded:
            continue
        images_dir = find_child_directory(child, "images")
        masks_dir = find_child_directory(child, "masks")
        if images_dir is None:
            raise FileNotFoundError(f"Each data_root child must contain images/: {child}")
        if not iter_images(images_dir):
            raise RuntimeError(f"No images found in {images_dir}")
        directories.append((child, images_dir, masks_dir))
    if not directories:
        raise RuntimeError(f"No child directories containing images/ found in {data_root}")
    return directories


def find_mask(
    image_path: Path, images_dir: Path, masks_dir: Optional[Path]
) -> Optional[Path]:
    """Resolve a mask with matching relative name, stem, or ``_mask`` suffix."""

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


def build_score_index(score_output_dirs: Sequence[Path]) -> Dict[str, List[Tuple[Path, Path]]]:
    """Index ``.npy``/``.npz`` score maps by image filename stem."""

    index: Dict[str, List[Tuple[Path, Path]]] = {}
    for score_output_dir in score_output_dirs:
        for score_path in score_output_dir.rglob("*"):
            if not score_path.is_file() or score_path.suffix.lower() not in SCORE_EXTENSIONS:
                continue
            for key in set(_score_keys(score_path)):
                index.setdefault(key, []).append((score_output_dir, score_path))
    if not index:
        raise FileNotFoundError("No .npy or .npz score maps found under score_output_dir.")
    return index


def find_score(
    image_path: Path,
    data_directory: Path,
    score_index: Mapping[str, Sequence[Tuple[Path, Path]]],
) -> Path:
    """Resolve one score map, preferring a directory named after the data class."""

    matches = list(score_index.get(image_path.stem.lower(), ()))
    unique = {score_path.resolve(): (root, score_path) for root, score_path in matches}
    matches = list(unique.values())
    if len(matches) > 1:
        directory_name = data_directory.name.lower()
        preferred = [
            item
            for item in matches
            if item[0].name.lower() == directory_name
            or directory_name in {part.lower() for part in item[1].parts}
        ]
        if len(preferred) == 1:
            matches = preferred
    if not matches:
        raise FileNotFoundError(
            f"Score map not found by filename {image_path.name} under score_output_dir."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple score maps match {image_path.name}:\n"
            + "\n".join(str(item[1]) for item in matches)
        )
    return matches[0][1]


def load_score_map(score_path: Path) -> np.ndarray:
    """Load a finite two-dimensional score map from ``.npy`` or ``.npz``."""

    score_path = Path(score_path)
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
        raise ValueError(f"Score map must be 2D: {score_path}; got {score_map.shape}")
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def classification_metrics(labels, scores, threshold: float) -> Dict[str, float]:
    """Calculate FPR/TNR/accuracy with anomalies as the positive class."""

    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty arrays of equal length")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    predictions = scores >= float(threshold)
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.logical_and(positives, predictions).sum())
    true_negative = int(np.logical_and(negatives, ~predictions).sum())
    false_positive = int(np.logical_and(negatives, predictions).sum())
    false_negative = int(np.logical_and(positives, ~predictions).sum())
    normal_count = true_negative + false_positive
    anomaly_count = true_positive + false_negative
    return {
        "Threshold": float(threshold),
        "FPR": float(false_positive / normal_count) if normal_count else float("nan"),
        "TNR": float(true_negative / normal_count) if normal_count else float("nan"),
        "Accuracy": float((true_positive + true_negative) / len(labels)),
        "_TPR": float(true_positive / anomaly_count) if anomaly_count else float("nan"),
    }


def select_optimal_threshold(labels, scores) -> Tuple[float, str, Dict[str, float]]:
    """Choose the threshold that maximizes balanced accuracy.

    Ties prefer accuracy, then TNR, then a larger threshold.  Single-class
    inputs choose the no-error threshold for the available class.
    """

    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty arrays of equal length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite for automatic threshold selection")
    unique_scores = np.unique(scores)
    if np.all(labels == 0):
        threshold = float(np.nextafter(unique_scores[-1], np.inf))
        return threshold, "all-normal", classification_metrics(labels, scores, threshold)
    if np.all(labels == 1):
        threshold = float(np.nextafter(unique_scores[0], -np.inf))
        return threshold, "all-anomaly", classification_metrics(labels, scores, threshold)

    candidates = np.concatenate(
        (
            [np.nextafter(unique_scores[0], -np.inf)],
            unique_scores,
            [np.nextafter(unique_scores[-1], np.inf)],
        )
    )
    best = None
    for candidate in candidates:
        metrics = classification_metrics(labels, scores, float(candidate))
        balanced_accuracy = (metrics["_TPR"] + metrics["TNR"]) / 2.0
        key = (balanced_accuracy, metrics["Accuracy"], metrics["TNR"], float(candidate))
        if best is None or key > best[0]:
            best = (key, metrics)
    assert best is not None
    return float(best[1]["Threshold"]), "max-balanced-accuracy", best[1]


def report_metric_names(base_metric_names: Sequence[str]) -> Tuple[str, ...]:
    return (*base_metric_names, *CLASSIFICATION_METRIC_NAMES)


def write_metric_report(
    results: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    base_metric_names: Sequence[str],
    label_name: str = "stage",
) -> None:
    """Print and save a common JSON/CSV report for any model adapter."""

    metric_names = report_metric_names(base_metric_names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {str(label): dict(metrics) for label, metrics in results.items()}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[label_name, *metric_names])
        writer.writeheader()
        for label, metrics in result.items():
            writer.writerow(
                {
                    label_name: label,
                    **{name: metrics.get(name, float("nan")) for name in metric_names},
                }
            )

    print("\nEvaluation metrics")
    print(f"{label_name:<29}" + "  ".join(f"{name:>10}" for name in metric_names))
    for label, metrics in result.items():
        values = "  ".join(
            f"{metrics.get(name, float('nan')):10.6f}" for name in metric_names
        )
        print(f"{label:<29}{values}")
    print()


def write_per_image_report(
    records: Sequence[Mapping[str, object]], output_path: Path, fieldnames: Sequence[str]
) -> None:
    """Write arbitrary per-image metric rows with a shared CSV implementation."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: record.get(field, float("nan")) for field in fieldnames}
            )


def group_score_values(
    samples: Sequence[Mapping[str, object]], score_key: str = "score"
) -> Dict[str, List[float]]:
    """Collect finite image scores by the standard visualization group label."""

    values = {label: [] for _key, label in STANDARD_GROUPS}
    for sample in samples:
        label = str(sample["group_label"])
        score = float(sample[score_key])
        if np.isfinite(score):
            values.setdefault(label, []).append(score)
    return values


def plot_score_distribution(
    groups: Mapping[str, Sequence[float]],
    output_path: Path,
    threshold: Optional[float],
    bins: int,
    title: str,
    xlabel: str,
    gt_score_values: Optional[Sequence[float]] = None,
) -> None:
    """Plot the common Train/Good, Test/Good, Test/Anomaly and Test/GT rows."""

    entries = [*STANDARD_GROUPS, ("test_gt", "Test / GT")]
    values = {**groups, "Test / GT": list(gt_score_values or ())}
    figure, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    for axis, (_key, label) in zip(axes, entries):
        group_values = np.asarray(values.get(label, ()), dtype=np.float64)
        group_values = group_values[np.isfinite(group_values)]
        if group_values.size:
            color = "crimson" if label == "Test / GT" else "steelblue"
            axis.hist(group_values, bins=max(1, int(bins)), alpha=0.7, color=color)
        else:
            axis.text(0.5, 0.5, "No samples", ha="center", va="center", transform=axis.transAxes)
        if threshold is not None:
            axis.axvline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"threshold={threshold:.6f}",
            )
        axis.set_title(f"{title}: {label} (n={group_values.size})")
        axis.set_ylabel("Count")
        axis.grid(True, alpha=0.3)
        if threshold is not None:
            axis.legend()
    axes[-1].set_xlabel(xlabel)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_score_table(
    samples: Sequence[Mapping[str, object]],
    output_path: Path,
    score_key: str = "score",
    include_score_path: bool = False,
) -> None:
    """Write scores and append one ``Test / GT`` maximum row when present."""

    fields = ["group", "image_path", "score"]
    if include_score_path:
        fields.append("score_path")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {
                "group": sample["group_label"],
                "image_path": str(sample["image_path"]),
                "score": float(sample[score_key]),
            }
            if include_score_path:
                row["score_path"] = str(sample["score_path"])
            writer.writerow(row)
            if sample.get("gt_score") is not None:
                row = {
                    "group": "Test / GT",
                    "image_path": str(sample["image_path"]),
                    "score": float(sample["gt_score"]),
                }
                if include_score_path:
                    row["score_path"] = str(sample["score_path"])
                writer.writerow(row)


def save_classification_threshold(
    output_path: Path,
    threshold: float,
    method: str,
    metrics: Mapping[str, float],
    extra: Optional[Mapping[str, object]] = None,
) -> None:
    """Persist the selected image threshold and common classification metrics."""

    payload: Dict[str, object] = {
        "threshold": float(threshold),
        "method": str(method),
        **{name: float(metrics[name]) for name in CLASSIFICATION_METRIC_NAMES},
    }
    if extra:
        payload.update(extra)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
