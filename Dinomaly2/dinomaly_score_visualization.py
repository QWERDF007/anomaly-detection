"""Dinomaly2 score-map and visualization pipeline without ROI processing.

This is a standalone subset of ``dinomaly_roi_pipeline.py``.  It only:

1. predicts and caches Dinomaly2 score maps;
2. plots Train/Good, Test/Good, Test/Anomaly and optional GT-region score
   distributions;
3. selects an automatic score threshold from the distributions; and
4. optionally saves threshold masks and fused heatmaps.

The score table also contains a ``Test / GT`` row for each annotated anomaly
image.  Its score is the maximum score-map value inside that image's GT mask.

It does not extract DINO patch features, run ROIAlign, build/query FAISS, or
filter ROI distances.  Shared behavior is implemented in
``dinomaly_pipeline_common.py`` and reused by the ROI entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import measure
from tqdm import tqdm

from dinomaly_evaluation import (
    evaluate_stage,
    print_and_save_metrics,
    write_per_image_pixel_metrics,
)
from dinomaly_pipeline_common import (
    GROUPS,
    artifact_root,
    choose_threshold,
    common_grid,
    find_child_directory,
    group_roots,
    ground_truth_relative_path,
    load_ground_truth,
    infer_score_map,
    iter_image_paths,
    load_score_map,
    load_transform,
    plot_group_density,
    relative_output_path,
    resolve_group_directory,
    resolve_non_good_directories,
    save_fused_heatmap,
    score_values_by_group,
    select_device,
)
from models.uad import Dinomaly
from predict import build_model
from utils import get_gaussian_kernel


LOGGER = logging.getLogger("dinomaly_score_visualization")


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
                    "ground_truth_relative": ground_truth_relative_path(
                        group_key,
                        image_path,
                        image_root,
                        len(roots),
                    ),
                    "score_path": score_path,
                    "score_map_shape": score_map.shape,
                    "score": float(score_map.max()),
                }
            )
    return samples


def collect_gt_score_values(
    samples: Sequence[Dict],
    ground_truth_dir: Optional[Path],
) -> List[float]:
    """Collect one maximum score for every connected GT anomaly region.

    The image-level distributions use one maximum score per image.  This
    fourth distribution uses one maximum score per connected GT region, so
    large regions do not contribute more samples than small regions.
    """

    if ground_truth_dir is None:
        return []

    values: List[float] = []
    for sample in samples:
        if sample["group_key"] != "test_anomaly":
            continue
        score_map = load_score_map(Path(sample["score_path"]))
        gt_mask = load_ground_truth(
            sample,
            ground_truth_dir,
            score_map.shape,
        ).astype(bool, copy=False)
        labels = measure.label(gt_mask)
        region_scores: List[float] = []
        for region_id in range(1, int(labels.max()) + 1):
            region_values = score_map[labels == region_id]
            region_values = region_values[np.isfinite(region_values)]
            if region_values.size:
                region_scores.append(float(region_values.max()))
        if region_scores:
            # Keep the per-image maximum for the CSV's Test / GT group while
            # using one maximum score per connected region in the plot.
            sample["gt_score"] = max(region_scores)
            sample["gt_region_scores"] = region_scores
            sample["gt_region_count"] = len(region_scores)
            values.extend(region_scores)
    return values


def plot_score_distribution(
    groups: Dict[str, List[float]],
    output_path: Path,
    threshold: Optional[float],
    bins: int,
    gt_score_values: Optional[Sequence[float]] = None,
) -> None:
    gt_label = "Test / Anomaly / GT regions"
    plot_groups = dict(groups)
    plot_groups[gt_label] = (
        list(gt_score_values) if gt_score_values is not None else []
    )
    grid = common_grid(plot_groups, bins)
    figure, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    for axis, (_, label) in zip(axes[:3], GROUPS):
        plot_group_density(
            axis,
            {label: groups.get(label, [])},
            grid,
            label,
            threshold,
            bins=bins,
            xlabel="Dinomaly2 image anomaly score",
        )
    plot_group_density(
        axes[3],
        {gt_label: plot_groups[gt_label]},
        grid,
        "Test / Anomaly / GT region maximum scores",
        threshold,
        bins=bins,
        xlabel="Dinomaly2 region maximum anomaly score",
        color_overrides={gt_label: "crimson"},
    )
    axes[-1].set_xlabel("Dinomaly2 anomaly score")
    figure.suptitle("Dinomaly2 Score Distribution", y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


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
            gt_score = sample.get("gt_score")
            if gt_score is not None:
                writer.writerow(
                    {
                        "group": "Test / GT",
                        "image_path": str(sample["image_path"]),
                        "score": gt_score,
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
            "  ├── test/<非good目录>/ 测试异常图像\n"
            "  └── ground_truth/     Test/Anomaly 的像素 GT 掩码（可选）\n\n"
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
        "-gt",
        "--ground_truth_dir",
        default=None,
        help=(
            "Test/Anomaly 像素 GT 掩码目录；不指定时自动使用 "
            "data_root/ground_truth，找不到时第四幅图为空。"
        ),
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
        "--metric_size",
        type=int,
        default=256,
        help="计算评估指标前统一缩放到的正方形边长；与 ROI pipeline 默认值一致。",
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
    if args.metric_size < 1:
        raise ValueError("metric_size must be positive.")
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
        "train_good": resolve_group_directory(
            data_root,
            split="train",
            category="good",
        ),
        "test_good": resolve_group_directory(
            data_root,
            split="test",
            category="good",
        ),
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

    ground_truth_dir = (
        Path(args.ground_truth_dir).expanduser()
        if args.ground_truth_dir
        else find_child_directory(data_root, "ground_truth")
    )
    if ground_truth_dir is not None and not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory does not exist: {ground_truth_dir}"
        )
    if ground_truth_dir is None:
        LOGGER.info(
            "No ground-truth directory found; GT score distribution will be empty."
        )
    else:
        LOGGER.info("Collecting GT score distribution from %s", ground_truth_dir)
    gt_score_values = collect_gt_score_values(samples, ground_truth_dir)
    LOGGER.info(
        "GT anomaly region maximum scores: %d regions",
        len(gt_score_values),
    )

    plot_score_distribution(
        score_groups,
        output_dir / "score_distribution.png",
        score_threshold,
        args.bins,
        gt_score_values,
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
                "gt_anomaly_region_score_count": len(gt_score_values),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    if ground_truth_dir is not None:
        print("Evaluating score maps...", flush=True)
        pixel_metric_records: List[Dict[str, object]] = []
        metrics = evaluate_stage(
            samples,
            ground_truth_dir,
            args.metric_size,
            score_map_key="score_path",
            stage_name="score maps",
            per_image_records=pixel_metric_records,
        )
        print_and_save_metrics({"score_maps": metrics}, output_dir)
        write_per_image_pixel_metrics(
            pixel_metric_records,
            output_dir / "pixel_metrics.csv",
        )
        print(
            f"Per-image pixel metrics written to "
            f"{output_dir / 'pixel_metrics.csv'}",
            flush=True,
        )
    else:
        LOGGER.info(
            "Evaluation metrics skipped; no ground-truth directory found."
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
