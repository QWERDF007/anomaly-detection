"""Shared utilities for the Dinomaly2 score and ROI pipelines.

This module contains only functionality used by both pipelines.  ROIAlign,
FAISS, ROI filtering and score-only visualization policy remain in their
respective entry-point scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils import cal_anomaly_maps, get_gaussian_kernel


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

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

GROUND_TRUTH_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_child_directory(root: Optional[Path], name: str) -> Optional[Path]:
    if root is None or not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def resolve_group_directory(
    data_root: Path,
    explicit: Optional[str] = None,
    split: str = "train",
    category: str = "good",
) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
    else:
        split_dir = find_child_directory(data_root, split)
        path = find_child_directory(split_dir, category)
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            f"Cannot find {split}/{category} directory. "
            "Use the corresponding explicit argument."
        )
    return path


def resolve_non_good_directories(data_root: Path) -> List[Path]:
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
            "Expected directories such as test/bad or test/anomaly."
        )
    return roots


def group_roots(groups: Dict[str, object], group_key: str) -> List[Path]:
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


def ground_truth_relative_path(
    group_key: str,
    image_path: Path,
    image_root: Path,
    root_count: int,
) -> Path:
    """Return the GT-relative path used for a discovered sample.

    When Test contains multiple non-good roots, retaining the root directory
    name prevents equal image names in different anomaly categories from
    being mapped to the same GT file.
    """

    relative = image_path.relative_to(image_root)
    if group_key == "test_anomaly" and root_count > 1:
        return Path(image_root.name) / relative
    return relative


def iter_image_paths(root: Path) -> List[Path]:
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


def score_map_from_outputs(
    encoder_output,
    decoder_output,
    original_shape: Tuple[int, int],
    device: torch.device,
    gaussian_filter: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    anomaly_map, _ = cal_anomaly_maps(
        encoder_output,
        decoder_output,
        original_shape,
    )
    if gaussian_filter is None:
        gaussian_filter = get_gaussian_kernel(
            kernel_size=5,
            sigma=4,
        ).to(device)
    anomaly_map = gaussian_filter(anomaly_map)
    score_map = anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def infer_score_map(
    model,
    image_path: Path,
    transform,
    device: torch.device,
    gaussian_filter: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original = np.asarray(image)
        image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_output, decoder_output = model(image_tensor)
        return score_map_from_outputs(
            encoder_output,
            decoder_output,
            original.shape[:2],
            device,
            gaussian_filter,
        )


def extract_feature_map(
    encoder,
    images: torch.Tensor,
    layers: Sequence[int],
    feature_merge: str,
) -> torch.Tensor:
    """Extract spatial DINO patch-token feature maps.

    The CLS token and register tokens are discarded. The returned tensor has
    shape ``[N, C, H, W]`` and is used by the ROI pipeline when the requested
    layers cannot be captured during the Dinomaly forward pass.
    """

    layers = sorted(set(int(layer) for layer in layers))
    if not layers:
        raise ValueError("At least one DINO layer is required.")
    if not hasattr(encoder, "prepare_tokens") or not hasattr(
        encoder,
        "blocks",
    ):
        raise RuntimeError(
            "This encoder does not expose Dinomaly2's "
            "prepare_tokens/blocks interface."
        )

    outputs: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        tokens = encoder.prepare_tokens(images)
        for index, block in enumerate(encoder.blocks):
            if index > layers[-1]:
                break
            tokens = block(tokens)
            if index in layers:
                outputs[index] = tokens

    register_tokens = int(getattr(encoder, "num_register_tokens", 0))
    feature_maps = []
    for layer in layers:
        if layer not in outputs:
            raise ValueError(
                f"Requested DINO layer {layer}, but the encoder has only "
                f"{len(encoder.blocks)} blocks."
            )
        layer_tokens = outputs[layer][:, 1 + register_tokens:, :]
        side = int(layer_tokens.shape[1] ** 0.5)
        if side * side != layer_tokens.shape[1]:
            raise ValueError(
                f"Layer {layer} has {layer_tokens.shape[1]} spatial tokens, "
                "which cannot be reshaped into a square feature map."
            )
        feature_maps.append(
            layer_tokens.transpose(1, 2).reshape(
                layer_tokens.shape[0],
                layer_tokens.shape[2],
                side,
                side,
            )
        )

    if feature_merge == "mean":
        return torch.stack(feature_maps, dim=1).mean(dim=1)
    if feature_merge == "concat":
        return torch.cat(feature_maps, dim=1)
    raise ValueError(f"Unsupported feature merge mode: {feature_merge}")


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


def load_score_map(score_path: Path) -> np.ndarray:
    score_map = np.asarray(np.load(score_path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Cached score map must be 2D: {score_path}; "
            f"got {score_map.shape}"
        )
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def score_values_by_group(
    samples: Sequence[Dict],
    key: str = "score",
) -> Dict[str, List[float]]:
    values = {display: [] for _, display in GROUPS}
    for sample in samples:
        values[sample["group_label"]].append(float(sample[key]))
    return {label: scores for label, scores in values.items() if scores}


def common_grid(groups: Dict[str, List[float]], bins: int) -> np.ndarray:
    values = [value for scores in groups.values() for value in scores]
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
    scale = (
        min(standard_deviation, interquartile_range / 1.34)
        if interquartile_range > 0
        else standard_deviation
    )
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
    bins: int = 30,
    color_overrides: Optional[Dict[str, str]] = None,
) -> None:
    histogram_values = []
    histogram_labels = []
    histogram_colors = []
    for label, values in groups.items():
        values_array = np.asarray(values, dtype=np.float64)
        values_array = values_array[np.isfinite(values_array)]
        if values_array.size == 0:
            continue
        histogram_values.append(values_array)
        histogram_labels.append(f"{label} (n={values_array.size})")
        histogram_colors.append(
            (color_overrides or {}).get(
                label,
                COLORS.get(label, "steelblue"),
            )
        )

    if histogram_values:
        axis.hist(
            histogram_values,
            bins=max(1, int(bins)),
            alpha=0.45,
            edgecolor="none",
            color=histogram_colors,
            label=histogram_labels,
        )
        axis.set_xlim(float(grid[0]), float(grid[-1]))
    else:
        axis.text(
            0.5,
            0.5,
            "No samples",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

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
    axis.set_ylabel("Count")
    axis.grid(True, alpha=0.3)
    handles, _labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend()


def distribution_valley_threshold(
    normal_scores: Sequence[float],
    anomaly_scores: Sequence[float],
    bins: int,
) -> Tuple[Optional[float], str]:
    """Find a valley or largest gap between normal and anomaly scores."""

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
        return float(grid[valley_index]), "distribution-valley"

    sorted_values = np.sort(values)
    gaps = np.diff(sorted_values)
    if gaps.size:
        gap_index = int(np.argmax(gaps))
        if gaps[gap_index] > 0:
            return (
                float(
                    (sorted_values[gap_index] + sorted_values[gap_index + 1])
                    / 2.0
                ),
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


def save_fused_heatmap(
    score_map: np.ndarray,
    image_path: Path,
    output_path: Path,
    reference_min: float,
    reference_max: float,
    source_image: Optional[np.ndarray] = None,
) -> None:
    score_map = np.asarray(score_map, dtype=np.float32)
    image = (
        source_image
        if source_image is not None
        else cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), fused):
        raise OSError(f"Cannot write heatmap: {output_path}")


def find_ground_truth_path(
    sample: Dict,
    ground_truth_dir: Path,
) -> Optional[Path]:
    image_path = Path(sample["image_path"])
    image_root = Path(sample["image_root"])
    relative = Path(
        sample.get(
            "ground_truth_relative",
            image_path.relative_to(image_root),
        )
    )
    candidates = []
    for extension in GROUND_TRUTH_EXTENSIONS:
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
    for extension in GROUND_TRUTH_EXTENSIONS:
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
    if sample["group_key"] in {"train_good", "test_good"}:
        return np.zeros(shape, dtype=np.uint8)
    if ground_truth_dir is None:
        raise FileNotFoundError(
            "Ground-truth directory is required for Test/anomaly scores."
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
