"""Dinomaly2 score-map and visualization pipeline without ROI processing.

This is a standalone subset of ``dinomaly_roi_pipeline.py``.  It only:

1. predicts and caches Dinomaly2 score maps;
2. plots Train/Good, Test/Good and Test/Anomaly score distributions;
3. selects an automatic score threshold from the distributions; and
4. optionally saves threshold masks and fused heatmaps.

It does not extract DINO patch features, run ROIAlign, build/query FAISS, or
filter ROI distances.  The original ROI pipeline is not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from models.uad import Dinomaly
from predict import build_model
from utils import cal_anomaly_maps, get_gaussian_kernel


LOGGER = logging.getLogger("dinomaly_score_visualization")

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


def find_child_directory(root: Optional[Path], name: str) -> Optional[Path]:
    if root is None or not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def resolve_group_directory(
    data_root: Path,
    split: str,
    category: str,
) -> Path:
    split_dir = find_child_directory(data_root, split)
    path = find_child_directory(split_dir, category)
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            f"Cannot find {split}/{category} under {data_root}."
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


def iter_image_paths(root: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in root.rglob("*")
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


def infer_score_map(
    model: Dinomaly,
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

    score_map = anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


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


def has_cached_scores(groups: Dict[str, object], output_dir: Path) -> bool:
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
                if not score_path.is_file():
                    return False
    return True


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


def prepare_samples(
    groups: Dict[str, object],
    output_dir: Path,
    model: Optional[Dinomaly],
    transform,
    device: torch.device,
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
        desc="Dinomaly2 score maps",
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
            use_cache = score_path.is_file()
            progress.set_postfix(mode="cache" if use_cache else "infer")
            if use_cache:
                score_map = load_score_map(score_path)
            else:
                if model is None:
                    raise RuntimeError(
                        "Cached score maps are incomplete and no model was loaded."
                    )
                score_map = infer_score_map(
                    model,
                    image_path,
                    transform,
                    device,
                    gaussian_filter,
                )
                np.save(score_path, score_map)

            samples.append(
                {
                    "group_key": group_key,
                    "group_label": dict(GROUPS)[group_key],
                    "image_path": image_path,
                    "image_root": image_root,
                    "root_count": len(roots),
                    "score_path": score_path,
                    "score_map_shape": score_map.shape,
                    "score": float(score_map.max()),
                }
            )
    return samples


def score_values_by_group(samples: Sequence[Dict]) -> Dict[str, List[float]]:
    values = {display: [] for _, display in GROUPS}
    for sample in samples:
        values[sample["group_label"]].append(float(sample["score"]))
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


def plot_group_histogram(
    axis,
    groups: Dict[str, List[float]],
    grid: np.ndarray,
    title: str,
    threshold: Optional[float],
    bins: int,
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
        histogram_colors.append(COLORS.get(label, "steelblue"))

    if histogram_values:
        axis.hist(
            histogram_values,
            bins=max(1, int(bins)),
            alpha=0.55,
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
    axis.set_ylabel("Count")
    axis.grid(True, alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend()


def plot_score_distribution(
    groups: Dict[str, List[float]],
    output_path: Path,
    threshold: Optional[float],
    bins: int,
) -> None:
    grid = common_grid(groups, bins)
    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for axis, (_, label) in zip(axes, GROUPS):
        plot_group_histogram(
            axis,
            {label: groups.get(label, [])},
            grid,
            label,
            threshold,
            bins,
        )
    axes[-1].set_xlabel("Dinomaly2 image anomaly score")
    figure.suptitle("Dinomaly2 Score Distribution", y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


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
            "score distributions are required. Pass --score_threshold explicitly."
        )
    return threshold, method


def visualization_path(
    output_dir: Path,
    sample: Dict,
    artifact: str,
) -> Path:
    extension = ".jpg" if artifact == "heatmap" else ".png"
    relative = sample["image_path"].relative_to(sample["image_root"])
    root = output_dir / "visualizations" / sample["group_key"]
    if sample["group_key"] == "test_anomaly" and sample["root_count"] > 1:
        root = root / sample["image_root"].name
    path = root / artifact / relative.with_suffix(extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def save_visualization_payload(payload: Dict) -> None:
    score_map = load_score_map(Path(payload["score_path"]))
    threshold = float(payload["score_threshold"])
    anomaly_mask = (score_map >= threshold).astype(np.uint8)
    thresholded_score_map = np.where(
        anomaly_mask > 0,
        score_map,
        0.0,
    ).astype(np.float32, copy=False)

    image_path = Path(payload["image_path"])
    source_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise OSError(f"Cannot read source image: {image_path}")

    finite_scores = score_map[np.isfinite(score_map)]
    if finite_scores.size:
        reference_min = float(finite_scores.min())
        reference_max = float(finite_scores.max())
    else:
        reference_min = 0.0
        reference_max = 0.0

    output_dir = Path(payload["output_dir"])
    sample = {
        "group_key": payload["group_key"],
        "image_path": image_path,
        "image_root": Path(payload["image_root"]),
        "root_count": int(payload["root_count"]),
    }
    save_fused_heatmap(
        thresholded_score_map,
        image_path,
        visualization_path(output_dir, sample, "heatmap"),
        reference_min,
        reference_max,
        source_image=source_image,
    )
    mask_path = visualization_path(output_dir, sample, "mask")
    if not cv2.imwrite(str(mask_path), anomaly_mask * 255):
        raise OSError(f"Cannot write mask: {mask_path}")


def visualization_payload(
    sample: Dict,
    output_dir: Path,
    score_threshold: float,
) -> Dict:
    return {
        "output_dir": str(output_dir),
        "score_path": str(sample["score_path"]),
        "image_path": str(sample["image_path"]),
        "image_root": str(sample["image_root"]),
        "root_count": int(sample["root_count"]),
        "group_key": sample["group_key"],
        "score_threshold": float(score_threshold),
    }


def save_visualizations(
    samples: Sequence[Dict],
    output_dir: Path,
    score_threshold: float,
    workers: int,
) -> None:
    workers = max(1, int(workers))
    payloads = [
        visualization_payload(sample, output_dir, score_threshold)
        for sample in samples
    ]
    if workers == 1:
        for payload in tqdm(
            payloads,
            desc="Save score visualizations",
            unit="image",
            dynamic_ncols=True,
        ):
            save_visualization_payload(payload)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(save_visualization_payload, payload)
            for payload in payloads
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Save score visualizations ({workers} processes)",
            unit="image",
            dynamic_ncols=True,
        ):
            future.result()


def save_score_table(samples: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["group", "image_path", "score"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "group": sample["group_label"],
                    "image_path": str(sample["image_path"]),
                    "score": sample["score"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dinomaly2 score map, automatic threshold, heatmap and mask pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "数据目录示例：\n"
            "  data_root/\n"
            "  ├── train/good/       训练正常图像\n"
            "  ├── test/good/        测试正常图像\n"
            "  └── test/<非good目录>/ 测试异常图像\n\n"
            "本脚本不使用 DINO patch features、ROIAlign 或 FAISS。"
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
        help="Dinomaly2 训练得到的 model.pth。",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="输出目录；保存 scores、分布图、阈值、score 表和可视化结果。",
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
        help="输入图像先缩放到的正方形边长；需与模型训练配置匹配。",
    )
    parser.add_argument(
        "-csz",
        "--crop_size",
        type=int,
        default=672,
        help="缩放后中心裁剪尺寸；默认与 image_size 相同。",
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
        help="Dinomaly2 Linear Attention 参数；需与训练模型配置一致。",
    )
    parser.add_argument(
        "--lc",
        type=int,
        default=2,
        help="Dinomaly2 Loose Constraint 参数；需与训练模型配置一致。",
    )
    parser.add_argument(
        "--cr",
        type=int,
        default=1,
        help="Dinomaly2 Context-aware recentering 参数；需与训练模型配置一致。",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="分布图直方图 bin 数以及自动阈值搜索分辨率基数。",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="分数阈值；不指定时根据正常/异常分布自动选择。",
    )
    parser.add_argument(
        "--gpu",
        "--cuda",
        dest="gpu",
        type=int,
        default=0,
        help="Dinomaly2 使用的 GPU 编号；设为 -1 使用 CPU。",
    )
    parser.add_argument(
        "--vis",
        dest="save_visualizations",
        action="store_true",
        help="输出按 score_threshold 生成的 heatmap 和 mask。",
    )
    parser.add_argument(
        "--vis_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="保存 heatmap/mask 的进程数；仅在指定 --vis 时生效。",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="忽略 output_dir/scores 中已有的 .npy，重新推理分数图。",
    )
    parser.set_defaults(save_visualizations=False)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size < 1 or args.crop_size < 1:
        raise ValueError("image_size and crop_size must be positive.")
    if args.crop_size > args.image_size:
        raise ValueError("crop_size must not be greater than image_size.")
    if args.bins < 1:
        raise ValueError("bins must be positive.")
    if args.vis_workers < 1:
        raise ValueError("vis_workers must be positive.")

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                output_dir / "score_pipeline.log",
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    groups = {
        "train_good": resolve_group_directory(data_root, "train", "good"),
        "test_good": resolve_group_directory(data_root, "test", "good"),
        "test_anomaly": resolve_non_good_directories(data_root),
    }
    device = select_device(args.gpu)
    LOGGER.info("Device: %s", device)

    if has_cached_scores(groups, output_dir) and not args.force_recompute:
        LOGGER.info("Reusing cached score maps from %s", output_dir)
        model = None
    else:
        LOGGER.info("Loading Dinomaly2 model from %s", args.model)
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
    )
    if not samples:
        raise RuntimeError("No images were found in the configured groups.")

    score_groups = score_values_by_group(samples)
    score_threshold, threshold_method = choose_threshold(
        score_groups,
        args.score_threshold,
        args.bins,
    )
    LOGGER.info(
        "Selected score threshold: %.6f (%s)",
        score_threshold,
        threshold_method,
    )

    plot_score_distribution(
        score_groups,
        output_dir / "score_distribution.png",
        score_threshold,
        args.bins,
    )
    save_score_table(samples, output_dir / "score_values.csv")
    with (output_dir / "score_threshold.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "score_threshold": score_threshold,
                "method": threshold_method,
                "groups": {
                    label: len(values)
                    for label, values in score_groups.items()
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    if args.save_visualizations:
        print("Saving score heatmaps and masks...", flush=True)
        save_visualizations(
            samples,
            output_dir,
            score_threshold,
            args.vis_workers,
        )
    else:
        LOGGER.info("Visualizations disabled; pass --vis to enable.")

    print(
        "Done. "
        f"score_threshold={score_threshold:.6f} "
        f"({threshold_method}), "
        f"images={len(samples)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
