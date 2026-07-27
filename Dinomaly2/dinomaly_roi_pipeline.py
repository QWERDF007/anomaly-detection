"""Run the complete Dinomaly2 score/DINO ROI/FAISS pipeline.

The pipeline:

1. Predicts Dinomaly2 score maps for train/good, test/good, and every
   non-good directory under test/.
2. Extracts DINO patch features from the same images.
3. Builds a normal ROI FAISS index from Train/good Labelme polygons.
4. Finds score-map ROIs, ROIAligns DINO features, and plots FAISS distances.
5. Filters ROIs by the selected distance threshold and compares image scores
   before and after filtering in one two-panel figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    message=r"xFormers is not available.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*",
)

import cv2
import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from skimage import measure
from tqdm import tqdm
from torchvision import transforms
from torchvision.ops import roi_align

from extract_dino_features import extract_feature_map
from models.uad import Dinomaly
from predict import build_model
from roi_feature_utils import (
    IMAGE_EXTENSIONS,
    annotation_path_for_image,
    l2_normalize,
    load_labelme_annotation,
    load_feature_map,
    load_search_index,
    mask_bbox,
    polygon_to_feature_mask,
    resize_mask_to_feature,
)
from utils import cal_anomaly_maps, get_gaussian_kernel


LOGGER = logging.getLogger("dinomaly_roi_pipeline")

GROUPS = (
    ("train_good", "Train / Good"),
    ("test_good", "Test / Good"),
    ("test_anomaly", "Test / Anomaly"),
)

COLORS = {
    "Train / Good": "green",
    "Test / Good": "blue",
    "Test / Anomaly": "red",
}


def find_child_directory(root: Path, name: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def resolve_group_directory(
    data_root: Path,
    explicit: Optional[str],
    split: str,
    category: str,
) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
    else:
        split_dir = find_child_directory(data_root, split)
        path = find_child_directory(split_dir, category) if split_dir else None
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            f"Cannot find {split}/{category} directory. "
            f"Use the corresponding explicit argument."
        )
    return path


def resolve_non_good_directories(
    data_root: Path,
) -> List[Path]:
    """Resolve every Test child directory except the directory named good."""

    test_dir = find_child_directory(data_root, "test")
    roots = (
        [
            child
            for child in test_dir.iterdir()
            if child.is_dir() and child.name.lower() != "good"
        ]
        if test_dir is not None
        else []
    )

    roots = sorted(set(roots), key=lambda path: str(path).lower())
    if not roots:
        raise FileNotFoundError(
            "Cannot find any non-good directory under test/. "
            "Expected directories such as test/bad or test/gg."
        )
    return roots


def group_roots(groups: Dict, group_key: str) -> List[Path]:
    roots = groups[group_key]
    if isinstance(roots, Path):
        return [roots]
    return list(roots)


def artifact_root(
    output_dir: Path,
    artifact_name: str,
    group_key: str,
    image_root: Path,
    root_count: int,
) -> Path:
    root = output_dir / artifact_name / group_key
    if group_key == "test_anomaly" and root_count > 1:
        root = root / image_root.name
    return root


def iter_image_paths(root: Path) -> list[Path]:
    root = Path(root)
    iterator: Iterable[Path] = root.rglob("*")
    return sorted(
        [
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def load_transform(args):
    return transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(args.crop_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def select_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        if gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu} is not available; "
                f"{torch.cuda.device_count()} device(s) found."
            )
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def infer_score_and_feature(
    model: Dinomaly,
    image_path: Path,
    transform,
    device: torch.device,
    layers: Sequence[int],
    feature_merge: str,
    gaussian_filter: Optional[torch.nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original = np.asarray(image)
        image_tensor = transform(image).unsqueeze(0).to(device)

    layers = sorted(set(int(layer) for layer in layers))
    captured = {}
    handles = []
    encoder_blocks = getattr(model.encoder, "blocks", None)
    target_layers = getattr(model, "target_layers", ())
    can_reuse_encoder = (
        encoder_blocks is not None
        and layers
        and all(0 <= layer < len(encoder_blocks) for layer in layers)
        and (not target_layers or max(layers) <= max(target_layers))
    )

    if can_reuse_encoder:
        def capture(layer_index):
            def hook(_module, _inputs, output):
                captured[layer_index] = output[0] if isinstance(output, tuple) else output

            return hook

        for layer in layers:
            handles.append(encoder_blocks[layer].register_forward_hook(capture(layer)))

    try:
        with torch.no_grad():
            encoder_output, decoder_output = model(image_tensor)
            anomaly_map, _ = cal_anomaly_maps(
                encoder_output,
                decoder_output,
                original.shape[:2],
            )
            if gaussian_filter is None:
                gaussian_filter = get_gaussian_kernel(
                    kernel_size=5,
                    sigma=4,
                ).to(device)
            anomaly_map = gaussian_filter(anomaly_map)
    finally:
        for handle in handles:
            handle.remove()

    if can_reuse_encoder and all(layer in captured for layer in layers):
        register_tokens = int(getattr(model.encoder, "num_register_tokens", 0))
        feature_maps = []
        for layer in layers:
            layer_tokens = captured[layer][:, 1 + register_tokens:, :]
            side = int(layer_tokens.shape[1] ** 0.5)
            if side * side != layer_tokens.shape[1]:
                raise ValueError(
                    f"Layer {layer} has {layer_tokens.shape[1]} spatial tokens, "
                    "which cannot be reshaped into a square feature map."
                )
            feature_maps.append(
                layer_tokens.transpose(1, 2).reshape(
                    layer_tokens.shape[0], layer_tokens.shape[2], side, side
                )
            )
        if feature_merge == "mean":
            feature_map = torch.stack(feature_maps, dim=1).mean(dim=1)
        elif feature_merge == "concat":
            feature_map = torch.cat(feature_maps, dim=1)
        else:
            raise ValueError(f"Unsupported feature merge mode: {feature_merge}")
    else:
        # Keep the original behavior for unusual layer selections outside the
        # layers traversed by Dinomaly.forward.
        with torch.no_grad():
            feature_map = extract_feature_map(
                model.encoder,
                image_tensor,
                layers,
                feature_merge,
            )

    score_map = anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32)
    feature_nchw = feature_map.detach().cpu().numpy().astype(np.float32)
    return score_map, feature_nchw


def relative_output_path(
    image_path: Path,
    image_root: Path,
    output_root: Path,
    extension: str,
) -> Path:
    relative = image_path.relative_to(image_root).with_suffix(extension)
    output_path = output_root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def has_cached_outputs(
    groups: Dict,
    output_dir: Path,
) -> bool:
    """Return whether every discovered image already has both cached arrays."""

    for group_key, _display_name in GROUPS:
        roots = group_roots(groups, group_key)
        for image_root in roots:
            for image_path in iter_image_paths(image_root):
                score_path = relative_output_path(
                    image_path,
                    image_root,
                    artifact_root(
                        output_dir,
                        "scores",
                        group_key,
                        image_root,
                        len(roots),
                    ),
                    ".npy",
                )
                feature_path = relative_output_path(
                    image_path,
                    image_root,
                    artifact_root(
                        output_dir,
                        "features",
                        group_key,
                        image_root,
                        len(roots),
                    ),
                    ".npy",
                )
                if not score_path.is_file() or not feature_path.is_file():
                    return False
    return True


def prepare_samples(
    groups: Dict,
    output_dir: Path,
    model: Optional[Dinomaly],
    transform,
    device: torch.device,
    args,
) -> List[Dict]:
    samples: List[Dict] = []
    jobs = []
    gaussian_filter = (
        get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
        if model is not None
        else None
    )

    for group_key, _display_name in GROUPS:
        roots = group_roots(groups, group_key)
        for image_root in roots:
            image_paths = iter_image_paths(image_root)
            if not image_paths:
                LOGGER.warning("No images found in %s", image_root)
                continue
            jobs.extend(
                (group_key, roots, image_root, image_path)
                for image_path in image_paths
            )

    with tqdm(
        jobs,
        desc="Dinomaly2 score/features",
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        for group_key, roots, image_root, image_path in progress:
            score_path = relative_output_path(
                image_path,
                image_root,
                artifact_root(
                    output_dir,
                    "scores",
                    group_key,
                    image_root,
                    len(roots),
                ),
                ".npy",
            )
            feature_path = relative_output_path(
                image_path,
                image_root,
                artifact_root(
                    output_dir,
                    "features",
                    group_key,
                    image_root,
                    len(roots),
                ),
                ".npy",
            )
            use_cache = score_path.is_file() and feature_path.is_file()
            progress.set_postfix(mode="cache" if use_cache else "infer")
            if use_cache:
                score_map = np.asarray(
                    np.load(score_path),
                    dtype=np.float32,
                )
                score_map = np.squeeze(score_map)
                feature_nchw = np.asarray(
                    np.load(feature_path),
                    dtype=np.float32,
                )
                if score_map.ndim != 2:
                    raise ValueError(
                        f"Cached score map must be 2D: {score_path}; "
                        f"got {score_map.shape}"
                    )
                if feature_nchw.ndim not in (3, 4):
                    raise ValueError(
                        f"Cached feature map must be CHW or NCHW: {feature_path}; "
                        f"got {feature_nchw.shape}"
                    )
            else:
                if model is None:
                    raise RuntimeError(
                        "Cached score/features are incomplete and no model was loaded."
                    )
                score_map, feature_nchw = infer_score_and_feature(
                    model,
                    image_path,
                    transform,
                    device,
                    args.layers,
                    args.feature_merge,
                    gaussian_filter,
                )
                np.save(score_path, score_map)
                np.save(feature_path, feature_nchw)
            ground_truth_relative = image_path.relative_to(image_root)
            if group_key == "test_anomaly" and len(roots) > 1:
                ground_truth_relative = Path(image_root.name) / ground_truth_relative
            samples.append(
                {
                    "group_key": group_key,
                    "group_label": dict(GROUPS)[group_key],
                    "image_path": image_path,
                    "image_root": image_root,
                    "ground_truth_relative": ground_truth_relative,
                    "score_path": score_path,
                    "feature_path": feature_path,
                    "before_score": float(score_map.max()),
                    "rois": [],
                    "after_score": 0.0,
                }
            )
    return samples


def score_values_by_group(
    samples: Sequence[Dict],
    key: str = "before_score",
) -> Dict[str, List[float]]:
    values = {display: [] for _, display in GROUPS}
    for sample in samples:
        values[sample["group_label"]].append(float(sample[key]))
    return {label: scores for label, scores in values.items() if scores}


def common_grid(groups: Dict[str, List[float]], bins: int) -> np.ndarray:
    values = [
        value
        for scores in groups.values()
        for value in scores
    ]
    if not values:
        raise ValueError("No values are available for plotting.")
    low = float(min(values))
    high = float(max(values))
    if high <= low:
        margin = max(abs(low) * 0.05, 1e-6)
        low -= margin
        high += margin
    return np.linspace(low, high, max(256, int(bins) * 8))


def kde_density(values: Sequence[float], grid: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return np.zeros_like(grid, dtype=np.float64)

    standard_deviation = float(np.std(values_array))
    interquartile_range = float(
        np.percentile(values_array, 75)
        - np.percentile(values_array, 25)
    )
    scale = min(
        standard_deviation,
        interquartile_range / 1.34,
    ) if interquartile_range > 0 else standard_deviation
    data_range = max(float(grid[-1] - grid[0]), 1e-12)
    bandwidth = 0.9 * scale * values_array.size ** -0.2
    if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
        bandwidth = data_range / 50.0

    density = np.zeros_like(grid, dtype=np.float64)
    normalizer = bandwidth * np.sqrt(2.0 * np.pi)
    chunk_size = 4096
    for start in range(0, values_array.size, chunk_size):
        chunk = values_array[start : start + chunk_size]
        distance = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * distance * distance).sum(axis=1)
    density /= values_array.size * normalizer
    return density


def plot_group_density(
    axis,
    groups: Dict[str, List[float]],
    grid: np.ndarray,
    title: str,
    threshold: Optional[float] = None,
    xlabel: str = "Anomaly Score",
) -> None:
    for label, values in groups.items():
        if not values:
            continue
        density = kde_density(values, grid)
        color = COLORS.get(label, "steelblue")
        axis.plot(
            grid,
            density,
            color=color,
            linewidth=2.0,
            label=f"{label} (n={len(values)})",
        )
        axis.fill_between(grid, density, color=color, alpha=0.12)
    if threshold is not None:
        axis.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold={threshold:.6f}",
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.grid(True, alpha=0.3)
    axis.legend()


def plot_distance_distribution(
    groups: Dict[str, List[float]],
    output_path: Path,
    threshold: Optional[float],
    bins: int,
) -> None:
    grid = common_grid(groups, bins)
    figure, axis = plt.subplots(figsize=(10, 6))
    plot_group_density(
        axis,
        groups,
        grid,
        "ROI FAISS Distance Distribution",
        threshold,
        xlabel="FAISS squared L2 distance",
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_score_comparison(
    before: Dict[str, List[float]],
    after: Dict[str, List[float]],
    output_path: Path,
    score_threshold: float,
    bins: int,
) -> None:
    merged = {
        label: before.get(label, []) + after.get(label, [])
        for label in set(before) | set(after)
    }
    grid = common_grid(merged, bins)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 11),
        sharex=True,
    )
    plot_group_density(
        axes[0],
        before,
        grid,
        "Before Distance Filtering",
        score_threshold,
    )
    plot_group_density(
        axes[1],
        after,
        grid,
        "After Distance Filtering",
        score_threshold,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def distribution_valley_threshold(
    normal_scores: Sequence[float],
    anomaly_scores: Sequence[float],
    bins: int,
) -> Tuple[Optional[float], str]:
    """Find the valley between normal and anomaly score distributions."""

    normal = np.asarray(normal_scores, dtype=np.float64)
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    normal = normal[np.isfinite(normal)]
    anomaly = anomaly[np.isfinite(anomaly)]
    if normal.size == 0 or anomaly.size == 0:
        return None, "missing-normal-or-anomaly"

    values = np.concatenate((normal, anomaly))
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return low, "constant-distribution"

    grid = common_grid(
        {"normal": normal.tolist(), "anomaly": anomaly.tolist()},
        bins,
    )
    normal_density = kde_density(normal, grid)
    anomaly_density = kde_density(anomaly, grid)
    combined_density = normal_density + anomaly_density

    normal_peak = int(np.argmax(normal_density))
    anomaly_peak = int(np.argmax(anomaly_density))
    left_peak, right_peak = sorted((normal_peak, anomaly_peak))
    if right_peak - left_peak >= 2:
        valley_start = left_peak + 1
        valley_end = right_peak - 1
        valley_index = valley_start + int(
            np.argmin(combined_density[valley_start : valley_end + 1])
        )
        return (
            float(grid[valley_index]),
            "distribution-valley",
        )

    # If the two modes overlap, use the largest empty/low-density gap as the
    # closest available valley rather than a percentile threshold.
    sorted_values = np.sort(values)
    gaps = np.diff(sorted_values)
    if gaps.size:
        gap_index = int(np.argmax(gaps))
        if gaps[gap_index] > 0:
            return (
                float((sorted_values[gap_index] + sorted_values[gap_index + 1]) / 2.0),
                "distribution-largest-gap",
            )

    return float((np.median(normal) + np.median(anomaly)) / 2.0), (
        "distribution-median-midpoint"
    )


def choose_threshold(
    groups: Dict[str, List[float]],
    explicit: Optional[float],
    bins: int,
) -> Tuple[float, str]:
    if explicit is not None:
        return float(explicit), "manual"

    normal_scores = groups.get("Train / Good", []) + groups.get(
        "Test / Good", []
    )
    anomaly_scores = groups.get("Test / Anomaly", [])
    threshold, method = distribution_valley_threshold(
        normal_scores,
        anomaly_scores,
        bins,
    )
    if threshold is None:
        raise RuntimeError(
            "Cannot choose an automatic threshold: both normal and anomaly "
            "score distributions are required. Pass the threshold explicitly."
        )
    return threshold, method


def mask_components(mask: np.ndarray, min_area: int = 1):
    binary = np.asarray(mask > 0, dtype=np.uint8)
    number, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for component_id in range(1, number):
        x, y, width, height, area = stats[component_id].tolist()
        if area < min_area:
            continue
        components.append(
            {
                "id": int(component_id),
                "mask": labels == component_id,
                "bbox": [float(x), float(y), float(x + width), float(y + height)],
                "area": int(area),
            }
        )
    return components


def roi_align_vectors(
    feature_chw: np.ndarray,
    entries: Sequence[Dict],
    output_size: int,
) -> np.ndarray:
    """ROIAlign and mask-pool all ROIs from one feature map in one call."""

    if not entries:
        return np.empty((0, feature_chw.shape[0]), dtype=np.float32)

    feature = np.asarray(feature_chw, dtype=np.float32)
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"Expected a CHW feature map, got shape {feature.shape}")
    feature = np.nan_to_num(feature, copy=False)
    channels, height, width = feature.shape

    boxes = []
    masks = []
    for entry in entries:
        x1, y1, x2, y2 = [float(value) for value in entry["bbox_feature"]]
        x1 = max(0.0, min(x1, width - 1e-3))
        y1 = max(0.0, min(y1, height - 1e-3))
        x2 = max(x1 + 1e-3, min(x2, float(width)))
        y2 = max(y1 + 1e-3, min(y2, float(height)))
        boxes.append([0.0, x1, y1, x2, y2])
        masks.append(np.asarray(entry["mask_feature"], dtype=np.float32))

    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    feature_tensor = torch.from_numpy(feature).unsqueeze(0)
    pooled = roi_align(
        feature_tensor,
        boxes_tensor,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True,
    )

    mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1)
    mask_boxes = boxes_tensor.clone()
    mask_boxes[:, 0] = torch.arange(len(entries), dtype=torch.float32)
    pooled_mask = roi_align(
        mask_tensor,
        mask_boxes,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True,
    ).clamp_min(0.0)
    weight = pooled_mask.sum(dim=(2, 3), keepdim=True)
    pooled = torch.where(
        weight > 1e-6,
        (pooled * pooled_mask).sum(dim=(2, 3), keepdim=True) / weight,
        pooled.mean(dim=(2, 3), keepdim=True),
    )
    return pooled.reshape(len(entries), channels).numpy().astype(
        np.float32,
        copy=False,
    )


def build_roi_index(
    samples: Sequence[Dict],
    annotation_root: Path,
    output_dir: Path,
    args,
) -> Tuple[Path, Path]:
    vectors = []
    records = []
    feature_dim = None
    train_samples = [
        sample for sample in samples if sample["group_key"] == "train_good"
    ]

    for sample in tqdm(
        train_samples,
        desc="Build ROI FAISS index",
        unit="image",
        dynamic_ncols=True,
    ):
        annotation_path = annotation_path_for_image(
            sample["image_path"],
            annotation_root,
            sample["image_root"],
        )
        if annotation_path is None:
            # LOGGER.warning(
            #     "Labelme annotation not found for %s",
            #     sample["image_path"],
            # )
            continue
        try:
            annotation = load_labelme_annotation(annotation_path)
            image_size = (
                int(annotation["imageWidth"]),
                int(annotation["imageHeight"]),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            LOGGER.warning("Cannot read Labelme annotation %s: %s", annotation_path, error)
            continue

        feature = load_feature_map(sample["feature_path"])
        feature_shape = feature.shape[-2:]
        shapes = annotation.get("shapes", [])
        if not isinstance(shapes, list):
            LOGGER.warning("Labelme shapes is not a list: %s", annotation_path)
            continue
        roi_entries = []
        for shape_index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                continue
            if str(shape.get("shape_type", "polygon")).lower() != "polygon":
                continue
            points = shape.get("points", [])
            try:
                roi_mask = polygon_to_feature_mask(
                    points,
                    image_size,
                    feature_shape,
                )
            except (TypeError, ValueError) as error:
                LOGGER.warning(
                    "Invalid polygon %d in %s: %s",
                    shape_index,
                    annotation_path,
                    error,
                )
                continue
            area = int(np.count_nonzero(roi_mask))
            bbox_feature = mask_bbox(roi_mask)
            if bbox_feature is None:
                continue
            roi_entries.append(
                {
                    "mask_feature": roi_mask,
                    "bbox_feature": bbox_feature,
                    "shape_index": shape_index,
                    "shape": shape,
                    "points": points,
                    "area": area,
                }
            )

        vectors_for_image = roi_align_vectors(
            feature,
            roi_entries,
            output_size=args.roi_size,
        )
        for entry, vector in zip(roi_entries, vectors_for_image):
            if args.normalize:
                vector = l2_normalize(vector)
            if feature_dim is None:
                feature_dim = int(vector.shape[0])
            if vector.shape[0] != feature_dim:
                raise ValueError("Feature dimensions are inconsistent.")
            vectors.append(vector.astype(np.float32, copy=False))
            records.append(
                {
                    "id": len(records),
                    "image_path": str(sample["image_path"]),
                    "annotation_path": str(annotation_path),
                    "shape_index": int(entry["shape_index"]),
                    "label": str(entry["shape"].get("label", "")),
                    "points": [
                        [float(point[0]), float(point[1])]
                        for point in entry["points"]
                    ],
                    "bbox_feature": [
                        float(value) for value in entry["bbox_feature"]
                    ],
                    "area": entry["area"],
                }
            )

    if not vectors:
        raise RuntimeError(
            "No Train/good polygon ROI features were collected. "
            "Check --train_annotation_dir and Labelme JSON files."
        )

    vectors_array = np.stack(vectors).astype(np.float32)
    cpu_index = faiss.IndexFlatL2(vectors_array.shape[1])
    cpu_index.add(vectors_array)
    index_dir = output_dir / "roi_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "roi_index.faiss"
    metadata_path = index_dir / "roi_index.json"
    faiss.write_index(cpu_index, str(index_path))
    metadata = {
        "index_type": "IndexFlatL2",
        "feature_dim": int(vectors_array.shape[1]),
        "roi_size": int(args.roi_size),
        "normalize": bool(args.normalize),
        "feature_layout": "NCHW",
        "records": records,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    LOGGER.info(
        "Built ROI index: %d vectors -> %s",
        len(records),
        index_path,
    )
    return index_path, metadata_path


def query_score_rois(
    samples: Sequence[Dict],
    index_path: Path,
    metadata_path: Path,
    args,
) -> Dict[str, List[float]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index, _resources = load_search_index(
        index_path,
        on_gpu=args.gpu >= 0,
        gpu_id=args.gpu,
    )
    roi_size = int(args.roi_size)
    normalize = bool(metadata.get("normalize", True))
    expected_dim = int(metadata["feature_dim"])
    if index.d != expected_dim:
        raise ValueError("FAISS/index metadata feature dimensions do not match.")

    distance_groups: Dict[str, List[float]] = {
        display: [] for _, display in GROUPS
    }
    for sample in tqdm(
        samples,
        desc="Query ROI FAISS distances",
        unit="image",
        dynamic_ncols=True,
    ):
        score_map = np.asarray(
            np.load(sample["score_path"]),
            dtype=np.float32,
        )
        score_map = np.squeeze(score_map)
        if score_map.ndim != 2:
            raise ValueError(f"Expected 2D score map: {sample['score_path']}")
        score_map = np.nan_to_num(score_map)
        components = mask_components(score_map >= args.score_threshold)
        feature = load_feature_map(sample["feature_path"])
        if feature.shape[0] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch in {sample['feature_path']}."
            )
        feature_shape = feature.shape[-2:]
        roi_entries = []
        for component in components:
            mask_feature = resize_mask_to_feature(
                component["mask"],
                feature_shape,
            )
            bbox_feature = mask_bbox(mask_feature)
            if bbox_feature is None:
                continue
            roi_entries.append(
                {
                    "mask_feature": mask_feature,
                    "bbox_feature": bbox_feature,
                    "mask": component["mask"],
                    "score": float(score_map[component["mask"]].max()),
                }
            )

        vectors = roi_align_vectors(feature, roi_entries, output_size=roi_size)
        if normalize and len(vectors):
            # Keep the original per-ROI normalization rule while batching the
            # expensive FAISS query below.
            vectors = np.stack([l2_normalize(vector) for vector in vectors])
        if len(vectors):
            distances, neighbours = index.search(
                np.asarray(vectors, dtype=np.float32),
                1,
            )
        else:
            distances = np.empty((0, 1), dtype=np.float32)
            neighbours = np.empty((0, 1), dtype=np.int64)

        for entry, distance_row, neighbour_row in zip(
            roi_entries,
            distances,
            neighbours,
        ):
            distance = float(distance_row[0])
            matched_index = int(neighbour_row[0])
            sample["rois"].append(
                {
                    "mask": entry["mask"],
                    "score": entry["score"],
                    "distance": distance,
                    "matched_index": matched_index,
                }
            )
            distance_groups[sample["group_label"]].append(distance)
            # LOGGER.info(
            #     "[distance] %s %s ROI %s: %.6f",
            #     sample["group_label"],
            #     sample["image_path"].name,
            #     component["id"],
            #     distance,
            # )

        if sample["rois"]:
            sample["after_score"] = 0.0
    return {
        label: values
        for label, values in distance_groups.items()
        if values
    }


def calculate_after_scores(
    samples: Sequence[Dict],
    distance_threshold: float,
) -> None:
    for sample in samples:
        score_map = np.asarray(
            np.load(sample["score_path"]),
            dtype=np.float32,
        )
        score_map = np.squeeze(score_map)
        filtered_score_map = np.zeros_like(score_map, dtype=np.float32)
        before_roi_mask = np.zeros_like(score_map, dtype=np.uint8)
        after_roi_mask = np.zeros_like(score_map, dtype=np.uint8)
        filtered_roi_mask = np.zeros_like(score_map, dtype=np.uint8)
        kept_scores = [
            roi["score"]
            for roi in sample["rois"]
            if roi["distance"] >= distance_threshold
        ]
        for roi in sample["rois"]:
            before_roi_mask[roi["mask"]] = 1
            if roi["distance"] >= distance_threshold:
                after_roi_mask[roi["mask"]] = 1
                filtered_score_map[roi["mask"]] = score_map[roi["mask"]]
            else:
                filtered_roi_mask[roi["mask"]] = 1
        sample["after_score"] = max(kept_scores, default=0.0)
        sample["filtered_score_map"] = filtered_score_map
        sample["before_roi_mask"] = before_roi_mask
        sample["after_roi_mask"] = after_roi_mask
        sample["filtered_roi_mask"] = filtered_roi_mask


def visualization_path(
    output_dir: Path,
    stage: str,
    artifact: str,
    sample: Dict,
) -> Path:
    relative = Path(sample["ground_truth_relative"])
    extension = ".jpg" if artifact == "heatmap" else ".png"
    path = (
        output_dir
        / "visualizations"
        / stage
        / sample["group_key"]
        / artifact
        / relative.with_suffix(extension)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_fused_heatmap(
    score_map: np.ndarray,
    image_path: Path,
    output_path: Path,
    reference_min: float,
    reference_max: float,
) -> None:
    score_map = np.asarray(score_map, dtype=np.float32)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot read source image: {image_path}")

    image_height, image_width = image.shape[:2]
    if score_map.shape != (image_height, image_width):
        zero_mask = cv2.resize(
            (score_map == 0).astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        score_map = cv2.resize(
            score_map,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        zero_mask = score_map == 0

    if reference_max <= reference_min:
        normalized = np.zeros(score_map.shape, dtype=np.uint8)
    else:
        normalized = np.clip(
            (score_map - reference_min)
            / (reference_max - reference_min)
            * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    fused = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0.0)
    fused[zero_mask] = image[zero_mask]
    if not cv2.imwrite(str(output_path), fused):
        raise OSError(f"Cannot write heatmap: {output_path}")


def save_roi_visualizations_and_report(
    samples: Sequence[Dict],
    output_dir: Path,
    distance_threshold: float,
) -> None:
    """Save before/after ROI masks and heatmaps plus a per-image report."""

    report_path = output_dir / "roi_filter_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group",
                "image_path",
                "before_score",
                "after_score",
                "candidate_roi_count",
                "kept_roi_count",
                "filtered_roi_count",
                "roi_filtered",
                "filtered_roi_distances",
            ],
        )
        writer.writeheader()

        for sample in samples:
            score_map = np.asarray(
                np.load(sample["score_path"]),
                dtype=np.float32,
            )
            score_map = np.squeeze(score_map)
            after_score_map = np.asarray(
                sample["filtered_score_map"],
                dtype=np.float32,
            )
            finite_scores = score_map[np.isfinite(score_map)]
            if finite_scores.size:
                reference_min = float(finite_scores.min())
                reference_max = float(finite_scores.max())
            else:
                reference_min = 0.0
                reference_max = 0.0

            save_fused_heatmap(
                score_map,
                sample["image_path"],
                visualization_path(output_dir, "before", "heatmap", sample),
                reference_min,
                reference_max,
            )
            save_fused_heatmap(
                after_score_map,
                sample["image_path"],
                visualization_path(output_dir, "after", "heatmap", sample),
                reference_min,
                reference_max,
            )
            cv2.imwrite(
                str(visualization_path(output_dir, "before", "mask", sample)),
                np.asarray(sample["before_roi_mask"], dtype=np.uint8) * 255,
            )
            cv2.imwrite(
                str(visualization_path(output_dir, "after", "mask", sample)),
                np.asarray(sample["after_roi_mask"], dtype=np.uint8) * 255,
            )

            filtered_distances = [
                float(roi["distance"])
                for roi in sample["rois"]
                if roi["distance"] < distance_threshold
            ]
            kept_count = len(sample["rois"]) - len(filtered_distances)
            writer.writerow(
                {
                    "group": sample["group_label"],
                    "image_path": str(sample["image_path"]),
                    "before_score": sample["before_score"],
                    "after_score": sample["after_score"],
                    "candidate_roi_count": len(sample["rois"]),
                    "kept_roi_count": kept_count,
                    "filtered_roi_count": len(filtered_distances),
                    "roi_filtered": bool(filtered_distances),
                    "filtered_roi_distances": ";".join(
                        f"{distance:.8f}" for distance in filtered_distances
                    ),
                }
            )


def find_ground_truth_path(
    sample: Dict,
    ground_truth_dir: Path,
) -> Optional[Path]:
    image_path = sample["image_path"]
    image_root = sample["image_root"]
    relative = Path(
        sample.get(
            "ground_truth_relative",
            image_path.relative_to(image_root),
        )
    )
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    candidates = []
    for extension in extensions:
        candidates.extend(
            [
                ground_truth_dir / relative.with_suffix(extension),
                ground_truth_dir / f"{image_path.stem}{extension}",
                ground_truth_dir / f"{image_path.stem}_mask{extension}",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = []
    for extension in extensions:
        matches.extend(ground_truth_dir.rglob(f"{image_path.stem}{extension}"))
        matches.extend(
            ground_truth_dir.rglob(f"{image_path.stem}_mask{extension}")
        )
    matches = sorted(set(matches), key=lambda path: str(path).lower())
    return matches[0] if matches else None


def load_ground_truth(
    sample: Dict,
    ground_truth_dir: Optional[Path],
    shape: Tuple[int, int],
) -> np.ndarray:
    if sample["group_key"] == "test_good":
        return np.zeros(shape, dtype=np.uint8)
    if ground_truth_dir is None:
        raise FileNotFoundError(
            "Ground-truth directory is required for Test/anomaly pixel metrics."
        )
    ground_truth_path = find_ground_truth_path(sample, ground_truth_dir)
    if ground_truth_path is None:
        raise FileNotFoundError(
            f"Ground-truth mask not found for {sample['image_path']}. "
            f"Search root: {ground_truth_dir}"
        )
    mask = cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read ground-truth mask: {ground_truth_path}")
    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(mask > 0, dtype=np.uint8)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return float("nan")


def max_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
    except ValueError:
        return float("nan")
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
    return float(np.nanmax(f1))


def safe_aupro(masks: np.ndarray, scores: np.ndarray) -> float:
    if not np.any(masks):
        return float("nan")
    if float(scores.max()) <= float(scores.min()):
        return 0.0
    try:
        return float(compute_pro_fast(masks.astype(np.uint8), scores))
    except (AssertionError, ValueError, ZeroDivisionError):
        return float("nan")


def compute_pro_fast(
    masks: np.ndarray,
    amaps: np.ndarray,
    num_th: int = 200,
) -> float:
    """Vectorized equivalent of ``utils.compute_pro``.

    The threshold sweep and the 0--0.3 FPR integration range intentionally
    match the original implementation. Connected-component labels are built
    once; per-threshold region hits then use ``np.bincount`` instead of a
    Python loop over every region.
    """

    masks = np.asarray(masks)
    amaps = np.asarray(amaps)
    if masks.ndim != 3 or amaps.ndim != 3 or masks.shape != amaps.shape:
        raise ValueError("masks and amaps must be equally shaped 3D arrays")
    if set(np.unique(masks).tolist()) != {0, 1}:
        raise AssertionError("masks must contain exactly 0 and 1")
    if not isinstance(num_th, int) or num_th <= 0:
        raise ValueError("num_th must be a positive integer")

    region_labels = []
    region_areas = []
    for mask in masks:
        labels = measure.label(mask)
        areas = np.bincount(labels.reshape(-1))[1:].astype(np.float64)
        region_labels.append(labels)
        region_areas.append(areas)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    if delta <= 0:
        return 0.0

    pros = []
    fprs = []
    inverse_pixels = np.logical_not(masks.astype(bool))
    inverse_count = int(inverse_pixels.sum())
    total_regions = sum(len(areas) for areas in region_areas)
    if total_regions == 0 or inverse_count == 0:
        return float("nan")

    for threshold in np.arange(min_th, max_th, delta):
        binary = amaps > threshold
        pro_sum = 0.0
        false_positive_count = 0
        for label_map, areas, amap_binary, inverse_mask in zip(
            region_labels,
            region_areas,
            binary,
            inverse_pixels,
        ):
            hits = np.bincount(
                label_map[amap_binary].reshape(-1),
                minlength=len(areas) + 1,
            )[1:]
            pro_sum += float(np.divide(hits, areas).sum())
            false_positive_count += int(np.logical_and(inverse_mask, amap_binary).sum())
        pros.append(pro_sum / total_regions)
        fprs.append(false_positive_count / inverse_count)

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    valid = fprs < 0.3
    if not np.any(valid):
        return float("nan")
    fprs = fprs[valid]
    pros = pros[valid]
    max_fpr = fprs.max()
    if max_fpr <= 0:
        # The original implementation divides by zero in this degenerate
        # case; preserve its resulting undefined metric rather than inventing
        # a new score.
        return float("nan")
    return float(auc(fprs / max_fpr, pros))


def evaluate_stage(
    samples: Sequence[Dict],
    ground_truth_dir: Optional[Path],
    after_filter: bool,
    metric_size: int,
) -> Dict[str, float]:
    evaluation_samples = [
        sample
        for sample in samples
        if sample["group_key"] in {"test_good", "test_anomaly"}
    ]
    if not evaluation_samples:
        raise RuntimeError("No Test/good or Test/anomaly samples were found.")

    image_labels = np.asarray(
        [sample["group_key"] == "test_anomaly" for sample in evaluation_samples],
        dtype=np.uint8,
    )
    image_scores = np.asarray(
        [
            sample["after_score"] if after_filter else sample["before_score"]
            for sample in evaluation_samples
        ],
        dtype=np.float32,
    )
    gt_pixels = []
    score_pixels = []
    stage_name = "after filtering" if after_filter else "before filtering"
    with tqdm(
        evaluation_samples,
        desc=f"Evaluate {stage_name}",
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        for sample in progress:
            score_map = np.asarray(
                sample["filtered_score_map"]
                if after_filter
                else np.load(sample["score_path"]),
                dtype=np.float32,
            )
            score_map = np.squeeze(score_map)
            original_shape = score_map.shape
            gt_mask = load_ground_truth(
                sample,
                ground_truth_dir,
                original_shape,
            )
            score_map = cv2.resize(
                score_map,
                (metric_size, metric_size),
                interpolation=cv2.INTER_LINEAR,
            )
            gt_mask = cv2.resize(
                gt_mask,
                (metric_size, metric_size),
                interpolation=cv2.INTER_NEAREST,
            )
            gt_pixels.append(gt_mask)
            score_pixels.append(score_map)

    gt_pixels_array = np.stack(gt_pixels, axis=0)
    score_pixels_array = np.stack(score_pixels, axis=0)
    pixel_labels = gt_pixels_array.reshape(-1)
    pixel_scores = score_pixels_array.reshape(-1)
    return {
        "I-AUROC": safe_auroc(image_labels, image_scores),
        "I-AP": safe_ap(image_labels, image_scores),
        "I-F1": max_f1(image_labels, image_scores),
        "P-AUROC": safe_auroc(pixel_labels, pixel_scores),
        "P-AP": safe_ap(pixel_labels, pixel_scores),
        "P-F1": max_f1(pixel_labels, pixel_scores),
        "P-AUPRO": safe_aupro(gt_pixels_array, score_pixels_array),
    }


def print_and_save_metrics(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
    output_dir: Path,
) -> None:
    metric_names = [
        "I-AUROC",
        "I-AP",
        "I-F1",
        "P-AUROC",
        "P-AP",
        "P-F1",
        "P-AUPRO",
    ]
    result = {
        "before_distance_filtering": before_metrics,
        "after_distance_filtering": after_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=True)
    with (output_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["stage"] + metric_names,
        )
        writer.writeheader()
        for stage, metrics in result.items():
            writer.writerow(
                {
                    "stage": stage,
                    **{
                        name: metrics.get(name, float("nan"))
                        for name in metric_names
                    },
                }
            )

    print("\nEvaluation metrics")
    print(
        "stage                         "
        + "  ".join(f"{name:>10}" for name in metric_names)
    )
    for stage, metrics in result.items():
        values = "  ".join(
            f"{metrics.get(name, float('nan')):10.6f}"
            for name in metric_names
        )
        print(f"{stage:<29}{values}")
    print()

def save_score_table(
    samples: Sequence[Dict],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["group", "image_path", "before_score", "after_score"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "group": sample["group_label"],
                    "image_path": str(sample["image_path"]),
                    "before_score": sample["before_score"],
                    "after_score": sample["after_score"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dinomaly2 + DINO 特征 + ROIAlign + FAISS 综合异常检测流程。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "数据目录示例：\n"
            "  data_root/\n"
            "  ├── train/good/       训练正常图像\n"
            "  ├── test/good/        测试正常图像\n"
            "  ├── test/<非good目录>/ 测试异常图像（可有多个目录）\n"
            "  ├── ground_truth/     非good测试图像的像素标注\n"
            "  └── labelme/          Train/good 对应的 Labelme JSON\n\n"
            "完整使用说明、输出目录和参数解释请查看：\n"
            "  Dinomaly2/ROI_FEATURE_PIPELINE.md"
        ),
    )
    parser.add_argument(
        "-i",
        "--data_root",
        required=True,
        help="数据集根目录；固定查找 train/good、test/good 和 test 下所有非 good 目录。",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Dinomaly2 训练得到的模型权重（.pth）；首次生成 score/features 时使用。",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="输出目录；保存 scores、features、roi_index、分布图和评估指标。",
    )
    parser.add_argument(
        "-gt",
        "--ground_truth_dir",
        default=None,
        help="像素级 Ground Truth 掩码目录；不指定时使用 data_root/ground_truth。",
    )
    parser.add_argument(
        "-ann",
        "--train_annotation_dir",
        required=True,
        help="Train/good 的 Labelme JSON 标注目录；JSON 文件名需与图像文件名对应。",
    )
    parser.add_argument(
        "--backbone",
        default="dinov2reg_vit_small_14",
        help="Dinomaly2 使用的 DINOv2 backbone 名称。",
    )
    parser.add_argument(
        "-imgsz",
        "--image_size",
        type=int,
        default=672,
        help="输入图像先缩放到的正方形边长；默认 672。",
    )
    parser.add_argument(
        "-csz",
        "--crop_size",
        type=int,
        default=672,
        help="缩放后中心裁剪的正方形边长；建议与 image_size 相同以保持标注坐标对应。",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dinomaly2 解码器 dropout；需与训练模型配置一致。",
    )
    parser.add_argument(
        "--la",
        type=int,
        default=1,
        help="Dinomaly2 解码器参数 la；需与训练模型配置一致。",
    )
    parser.add_argument(
        "--lc",
        type=int,
        default=2,
        help="Dinomaly2 解码器参数 lc；需与训练模型配置一致。",
    )
    parser.add_argument(
        "--cr",
        type=int,
        default=1,
        help="Dinomaly2 解码器参数 cr；需与训练模型配置一致。",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9],
        help="提取 DINO patch token 的 Transformer 层编号列表。",
    )
    parser.add_argument(
        "--feature_merge",
        choices=["mean", "concat"],
        default="mean",
        help="多层 patch token 合并方式：mean 为逐层平均，concat 为通道拼接。",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        default=7,
        help="ROIAlign 输出的空间尺寸 roi_size×roi_size；最终会池化成一个 ROI 特征向量。",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="密度曲线的采样分辨率基数；数值越大，波谷定位越细。",
    )
    parser.add_argument(
        "-msz",
        "--metric_size",
        type=int,
        default=256,
        help="计算像素级指标前统一缩放到的正方形边长；默认 256。",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Dinomaly2 异常分数阈值；不指定时从正常+异常分数分布的波谷自动选择。",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=None,
        help="FAISS ROI 距离阈值；不指定时从正常+异常 ROI 距离分布的波谷自动选择。",
    )
    parser.add_argument(
        "--gpu",
        "--cuda",
        dest="gpu",
        type=int,
        default=0,
        help="Dinomaly2 和 FAISS 使用的 GPU 编号；设为 -1 使用 CPU。",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="不对 ROI 特征做 L2 归一化；默认会归一化。",
    )
    parser.add_argument(
        "--vis",
        dest="save_visualizations",
        action="store_true",
        help="输出过滤前后的 heatmap、mask 和 ROI 过滤报告。",
    )
    parser.set_defaults(normalize=True, save_visualizations=False)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size < 1 or args.crop_size < 1:
        raise ValueError("image_size and crop_size must be positive.")
    if args.roi_size < 1:
        raise ValueError("roi_size must be positive.")
    if args.metric_size < 1:
        raise ValueError("metric_size must be positive.")
    data_root = Path(args.data_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                output_dir / "pipeline.log",
                encoding="utf-8",
            )
        ],
        force=True,
    )
    groups = {
        "train_good": resolve_group_directory(
            data_root, None, "train", "good"
        ),
        "test_good": resolve_group_directory(
            data_root, None, "test", "good"
        ),
        "test_anomaly": resolve_non_good_directories(data_root),
    }
    train_annotation_dir = Path(args.train_annotation_dir).expanduser()
    if not train_annotation_dir.is_dir():
        raise FileNotFoundError(
            "Train annotation directory does not exist: "
            f"{train_annotation_dir}"
        )
    ground_truth_dir = (
        Path(args.ground_truth_dir).expanduser()
        if args.ground_truth_dir
        else find_child_directory(data_root, "ground_truth")
    )
    if ground_truth_dir is None or not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            "Ground-truth directory is required for pixel metrics. "
            "Pass --ground_truth_dir or create data_root/ground_truth."
        )

    device = select_device(args.gpu)
    if has_cached_outputs(groups, output_dir):
        LOGGER.info(
            "Reusing all cached score maps and DINO features from %s",
            output_dir,
        )
        model = None
    else:
        LOGGER.info("Cached outputs are incomplete; loading Dinomaly2 model.")
        model = build_model(args, device)
        checkpoint = torch.load(
            Path(args.model).expanduser(),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
    transform = load_transform(args)

    samples = prepare_samples(
        groups,
        output_dir,
        model,
        transform,
        device,
        args,
    )
    before_groups = score_values_by_group(samples, "before_score")
    score_threshold, score_method = choose_threshold(
        before_groups,
        args.score_threshold,
        args.bins,
    )
    LOGGER.info(
        "Selected score threshold: %.6f (%s)",
        score_threshold,
        score_method,
    )
    index_path, metadata_path = build_roi_index(
        samples,
        train_annotation_dir,
        output_dir,
        args,
    )
    distance_groups = query_score_rois(
        samples,
        index_path,
        metadata_path,
        argparse.Namespace(
            roi_size=args.roi_size,
            score_threshold=score_threshold,
            gpu=args.gpu,
        ),
    )
    distance_threshold, distance_method = choose_threshold(
        distance_groups,
        args.distance_threshold,
        args.bins,
    )
    LOGGER.info(
        "Selected distance threshold: %.6f (%s)",
        distance_threshold,
        distance_method,
    )
    plot_distance_distribution(
        distance_groups,
        output_dir / "distance_distribution.png",
        distance_threshold,
        args.bins,
    )

    calculate_after_scores(samples, distance_threshold)
    if args.save_visualizations:
        save_roi_visualizations_and_report(
            samples,
            output_dir,
            distance_threshold,
        )
    else:
        LOGGER.info("Visualization output disabled; pass --vis to enable")
    after_groups = score_values_by_group(samples, "after_score")
    before_metrics = evaluate_stage(
        samples,
        ground_truth_dir,
        after_filter=False,
        metric_size=args.metric_size,
    )
    after_metrics = evaluate_stage(
        samples,
        ground_truth_dir,
        after_filter=True,
        metric_size=args.metric_size,
    )
    print_and_save_metrics(
        before_metrics,
        after_metrics,
        output_dir,
    )
    plot_score_comparison(
        before_groups,
        after_groups,
        output_dir / "score_distribution_comparison.png",
        score_threshold,
        args.bins,
    )
    save_score_table(samples, output_dir / "score_values.csv")
    LOGGER.info(
        "Done. score_threshold=%.6f, distance_threshold=%.6f",
        score_threshold,
        distance_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
