"""Evaluate cached Dinomaly2 score maps without running model inference.

The score maps are expected to come from ``dinomaly_score_visualization.py``.
This entry point only resolves dataset metadata, loads the cached ``.npy``
maps and computes the shared image- and pixel-level metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dinomaly_evaluation import evaluate_stage, print_metrics
from dinomaly_pipeline_common import (
    collect_cached_score_samples,
    find_child_directory,
    resolve_group_directory,
    resolve_non_good_directories,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dinomaly2 score maps cached by "
            "dinomaly_score_visualization.py; no model inference is run."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例：\n"
            "  python dinomaly_score_evaluation.py \\\n"
            "    --data_root /path/to/data \\\n"
            "    --score_output_dir /path/to/score_pipeline \\\n"
            "    --ground_truth_dir /path/to/data/ground_truth\n\n"
            "--score_output_dir 应指向 score visualization 的输出目录，"
            "脚本会读取其下的 scores/。"
        ),
    )
    parser.add_argument(
        "-i",
        "--data_root",
        required=True,
        type=Path,
        help="数据集根目录；用于解析 Test/good、Test/Anomaly 和 GT 路径。",
    )
    parser.add_argument(
        "-s",
        "--score_output_dir",
        "--score_dir",
        required=True,
        type=Path,
        help=(
            "dinomaly_score_visualization 的输出目录；"
            "其中应包含 scores/。"
        ),
    )
    parser.add_argument(
        "-gt",
        "--ground_truth_dir",
        type=Path,
        default=None,
        help="像素级 GT 目录；未指定时使用 data_root/ground_truth。",
    )
    parser.add_argument(
        "-msz",
        "--metric_size",
        type=int,
        default=256,
        help="计算像素指标前统一缩放到的正方形边长（默认：256）。",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric_size < 1:
        raise ValueError("metric_size must be positive.")

    data_root = args.data_root.expanduser().resolve()
    score_output_dir = args.score_output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not score_output_dir.is_dir():
        raise FileNotFoundError(
            f"Score output directory does not exist: {score_output_dir}"
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
    ground_truth_dir = (
        args.ground_truth_dir.expanduser().resolve()
        if args.ground_truth_dir is not None
        else find_child_directory(data_root, "ground_truth")
    )
    if ground_truth_dir is None or not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            "Ground-truth directory is required for evaluation. "
            "Pass --ground_truth_dir or create data_root/ground_truth."
        )

    print(
        f"Loading cached score maps from {score_output_dir / 'scores'}...",
        flush=True,
    )
    samples = collect_cached_score_samples(groups, score_output_dir)
    metrics = evaluate_stage(
        samples,
        ground_truth_dir,
        args.metric_size,
        score_map_key="score_path",
        image_score_key="score",
        stage_name="score maps",
    )
    print_metrics({"score_maps": metrics})
    print(
        f"Done. evaluated score maps for {len(samples)} images.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
