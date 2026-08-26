"""Train and evaluate PatchCore on MVTec or a custom image dataset.

The original project exposes a powerful chained Click command in
``bin/run_patchcore.py``.  This script is a smaller, argparse-based entry point
for the workflow used by Dinomaly2: point it at one dataset directory, build a
normal-only memory bank, evaluate every available test split, and save a model
that can be consumed by ``predict.py`` or the original PatchCore loader.

Examples::

    python train.py --data_path /data/widget --dataset custom \
        --backbone wideresnet50 --save_dir ./saved_results

    python train.py --data_path /data/mvtec --dataset mvtec \
        --category bottle --category cable --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from patchcore_evaluation import (
    METRIC_NAMES,
    compute_evaluation_metrics,
    compute_pro,
    safe_average_precision,
    safe_auroc,
    safe_f1_max,
)


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import patchcore.backbones  # noqa: E402
import patchcore.common  # noqa: E402
import patchcore.patchcore  # noqa: E402
import patchcore.sampler  # noqa: E402
import patchcore.utils  # noqa: E402
from patchcore.datasets.custom import CustomDataset  # noqa: E402
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset  # noqa: E402


LOGGER = logging.getLogger("patchcore.train")


def setup_seed(seed: int, device: torch.device) -> None:
    """Make data ordering and coreset sampling reproducible."""

    patchcore.utils.fix_seeds(seed, with_cuda=device.type == "cuda")
    torch.backends.cudnn.benchmark = False


def select_device(gpu: int) -> torch.device:
    """Return the requested CUDA device; 4060 8G 单卡适配：GPU 不可用时自动回退 CPU，兼容 faiss-cpu。

    原实现强制要求 GPU FAISS，已改为：若 torch.cuda 不可用或 gpu<0 则回退 cpu；
    faiss-cpu 场景下使用 CPU 索引，避免 faiss-gpu 的 cu12 与 OOM 问题。
    """
    if gpu is not None and int(gpu) >= 0 and torch.cuda.is_available():
        gpu = int(gpu)
        if 0 <= gpu < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu}")
        LOGGER.warning("GPU %d unavailable (%d found), fallback to CPU", gpu, torch.cuda.device_count())
    if torch.cuda.is_available() and (gpu is None or int(gpu) < 0):
        # 未指定 gpu 但有 cuda：默认 cuda:0
        return torch.device("cuda:0")
    LOGGER.info("Using CPU (faiss-cpu mode) - 4060 8G 推荐")
    return torch.device("cpu")


def _split_categories(values: Optional[Sequence[str]]) -> List[Optional[str]]:
    if not values:
        return [None]
    categories: List[Optional[str]] = []
    for value in values:
        categories.extend(item.strip() for item in value.split(",") if item.strip())
    return categories or [None]


def _dataset_class(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name in {"mvtec", "mvtec-ad", "mvtec_ad"}:
        return MVTecDataset
    if dataset_name in {"custom", "folder", "imagefolder"}:
        return CustomDataset
    raise ValueError(f"Unsupported dataset type: {dataset_name}")


def make_dataset(args, category: Optional[str], split: DatasetSplit):
    dataset_cls = _dataset_class(args.dataset)
    # 4060 单卡：支持 data_path 为 txt 列表（中文/空格需 ""），自动处理 test/train 切分 txt 的 test sibling
    source = args.data_path
    if isinstance(source, (str, Path)) and str(source).lower().endswith(".txt"):
        src_path = Path(source).expanduser()
        # 若是 test split 且 source 是 train_*.txt，尝试定位 test_*.txt
        if split == DatasetSplit.TEST and "train" in src_path.name:
            cand = src_path.parent / src_path.name.replace("train", "test")
            if cand.is_file():
                source = str(cand)
                LOGGER.info("PatchCore txt split: train %s -> test %s", src_path, cand)
            else:
                LOGGER.warning("PatchCore txt test sibling not found: %s, using train txt for TEST split", cand)
    kwargs = {
        "source": source,
        "classname": category,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "split": split,
        "train_val_split": args.train_val_split,
        "seed": args.seed,
    }
    if dataset_cls is CustomDataset and args.mask_dir:
        kwargs["mask_dir"] = args.mask_dir
    return dataset_cls(**kwargs)


def make_sampler(
    args, device: torch.device, number_of_features: Optional[int] = None
):
    if args.sampler == "identity" or args.sampling_percentage >= 1.0:
        return patchcore.sampler.IdentitySampler()
    if number_of_features is not None and int(number_of_features * args.sampling_percentage) < 1:
        LOGGER.warning(
            "Sampling percentage would select no features; using identity sampling."
        )


def _split_categories(values: Optional[Sequence[str]]) -> List[Optional[str]]:
    if not values:
        return [None]
    categories: List[Optional[str]] = []
    for value in values:
        categories.extend(item.strip() for item in value.split(",") if item.strip())
    return categories or [None]


def _dataset_class(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name in {"mvtec", "mvtec-ad", "mvtec_ad"}:
        return MVTecDataset
    if dataset_name in {"custom", "folder", "imagefolder"}:
        return CustomDataset
    raise ValueError(f"Unsupported dataset type: {dataset_name}")


def make_dataset(args, category: Optional[str], split: DatasetSplit):
    dataset_cls = _dataset_class(args.dataset)
    # 4060 单卡：支持 data_path 为 txt 列表（中文/空格需 ""），自动处理 test/train 切分 txt 的 test sibling
    source = args.data_path
    if isinstance(source, (str, Path)) and str(source).lower().endswith(".txt"):
        src_path = Path(source).expanduser()
        # 若是 test split 且 source 是 train_*.txt，尝试定位 test_*.txt
        if split == DatasetSplit.TEST and "train" in src_path.name:
            cand = src_path.parent / src_path.name.replace("train", "test")
            if cand.is_file():
                source = str(cand)
                LOGGER.info("PatchCore txt split: train %s -> test %s", src_path, cand)
            else:
                LOGGER.warning("PatchCore txt test sibling not found: %s, using train txt for TEST split", cand)
    kwargs = {
        "source": source,
        "classname": category,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "split": split,
        "train_val_split": args.train_val_split,
        "seed": args.seed,
    }
    if dataset_cls is CustomDataset and args.mask_dir:
        kwargs["mask_dir"] = args.mask_dir
    return dataset_cls(**kwargs)


def make_sampler(
    args, device: torch.device, number_of_features: Optional[int] = None
):
    if args.sampler == "identity" or args.sampling_percentage >= 1.0:
        return patchcore.sampler.IdentitySampler()
    if number_of_features is not None and int(number_of_features * args.sampling_percentage) < 1:
        LOGGER.warning(
            "Sampling percentage would select no features; using identity sampling."
        )
        return patchcore.sampler.IdentitySampler()
    if args.sampling_percentage <= 0:
        raise ValueError("sampling_percentage must be greater than zero.")
    # 4060 8G 单卡：672 大图 + 大 N 时 FAISS 采样在 GPU 易 OOM (10.77 GiB)，自动切 CPU 采样 + 降低比例
    sampler_device = device
    eff_percentage = float(args.sampling_percentage)
    if getattr(args, "imagesize", 0) >= 672 and number_of_features is not None and number_of_features > 80000:
        LOGGER.warning("672 large feature bank (%d patches) -> switch sampler to CPU and reduce percentage to 0.05 to avoid OOM", number_of_features)
        sampler_device = torch.device("cpu")
        eff_percentage = min(eff_percentage, 0.05)
    elif getattr(args, "imagesize", 0) >= 672 and device.type == "cuda":
        # 672 即使中等也建议 CPU 采样，避免显存峰值
        sampler_device = torch.device("cpu")
        LOGGER.info("672 uses CPU sampler to save VRAM (4060 8G)")
    if args.sampler == "identity":
        return patchcore.sampler.IdentitySampler()
    if args.sampler == "greedy_coreset":
        return patchcore.sampler.GreedyCoresetSampler(eff_percentage, sampler_device)
    if args.sampler == "approx_greedy_coreset":
        return patchcore.sampler.ApproximateGreedyCoresetSampler(
            eff_percentage, sampler_device
        )
    raise ValueError(f"Unsupported sampler: {args.sampler}")


def build_patchcore(
    args,
    input_shape: Sequence[int],
    device: torch.device,
    number_of_features: Optional[int] = None,
):
    """Create a PatchCore instance with the same parameters used for saving."""

    layers = list(args.layers_to_extract_from or ["layer2", "layer3"])
    backbone = patchcore.backbones.load(args.backbone)
    backbone.name = args.backbone
    backbone.seed = args.seed

    device_id = device.index if device.type == "cuda" and device.index is not None else 0
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=layers,
        device=device,
        input_shape=tuple(input_shape),
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        patchstride=args.patchstride,
        featuresampler=make_sampler(args, device, number_of_features),
        anomaly_score_num_nn=args.anomaly_scorer_num_nn,
        nn_method=patchcore.common.FaissNN(
            on_gpu=None if device.type == "cuda" else False,
            num_workers=args.faiss_num_workers,
            device_id=device_id,
        ),
    )
    return model
_safe_auroc = safe_auroc
_safe_average_precision = safe_average_precision
_safe_f1_max = safe_f1_max
_compute_pro = compute_pro
_compute_evaluation_metrics = compute_evaluation_metrics


def _write_predictions(path: Path, dataset, scores) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image_path", "anomaly", "anomaly_score"])
        for item, score in zip(dataset.data_to_iterate, scores):
            writer.writerow([item[2], item[1] != "good", float(score)])


def evaluate(model, dataloader, dataset, output_dir: Path) -> Dict[str, float]:
    scores, segmentations, labels_gt, masks_gt = model.predict(dataloader)
    _write_predictions(output_dir / "predictions.csv", dataset, scores)
    # Canonical metrics deliberately derive image scores from the maps rather
    # than PatchCore's native score, and use Dinomaly2's 256px metric grid.
    metrics = _compute_evaluation_metrics(
        scores,
        labels_gt,
        segmentations,
        masks_gt,
        metric_size=256,
    )
    return {name: metrics[name] for name in METRIC_NAMES}


def _category_output_name(category: Optional[str], args) -> str:
    if category:
        return category.replace("/", "_").replace("\\", "_")
    if args.dataset == "custom":
        return Path(args.data_path).name or "custom"
    return "all"


def train(args) -> List[Dict[str, object]]:
    device = select_device(args.gpu)
    setup_seed(args.seed, device)
    LOGGER.info("Using device: %s", device)

    output_root = Path(args.save_dir).expanduser() / datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for category in _split_categories(args.category):
        category_name = _category_output_name(category, args)
        LOGGER.info("Preparing category %s", category_name)
        train_dataset = make_dataset(args, category, DatasetSplit.TRAIN)
        if len(train_dataset) == 0:
            raise RuntimeError(f"No normal training images found for {category_name}.")

        test_dataset = None
        try:
            test_dataset = make_dataset(args, category, DatasetSplit.TEST)
        except (FileNotFoundError, RuntimeError) as exc:
            LOGGER.warning("Skipping evaluation for %s: %s", category_name, exc)

        pin_memory = device.type == "cuda"
        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
        }
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
        test_loader = (
            torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                **loader_kwargs,
            )
            if test_dataset is not None
            else None
        )

        input_shape = train_dataset.imagesize
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_patchcore(args, input_shape, device, len(train_dataset))
        LOGGER.info("Fitting PatchCore with %d training images", len(train_dataset))
        model.fit(train_loader)

        peak_gpu_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )
        LOGGER.info("Peak GPU Memory: %.1f MB", peak_gpu_mem_mb)

        model.save_to_path(str(output_root))
        LOGGER.info("Saved model to %s", output_root)

        metrics: Dict[str, float] = {}
        if test_loader is not None:
            metrics = evaluate(model, test_loader, test_dataset, output_root)
            metric_text = ", ".join(
                f"{name}={metrics[name]:.4f}" for name in metrics
            )
            LOGGER.info("%s: %s", category_name, metric_text)
        results.append({"category": category_name, "peak_gpu_mem_mb": round(peak_gpu_mem_mb, 2), **metrics})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_path = output_root / "results.csv"
    fields = [
        "category",
        "peak_gpu_mem_mb",
        "I-AUROC",
        "I-AP",
        "I-F1",
        "P-AUROC",
        "P-AP",
        "P-F1",
        "P-AUPRO",
    ]
    with result_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    import json
    with (output_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results[0] if results else {}, f, indent=2)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PatchCore using normal images from a custom or MVTec dataset."
    )
    parser.add_argument("--data_path", required=True, help="Dataset root directory.")
    parser.add_argument(
        "--dataset",
        default="custom",
        choices=["custom", "mvtec"],
        help="Dataset format: custom uses one data_path; mvtec uses the original MVTec layout.",
    )
    parser.add_argument(
        "--category",
        "--categories",
        "-c",
        action="append",
        help="Category name; repeat or comma-separate for multiple categories.",
    )
    parser.add_argument("--save_dir", default="./saved_results")
    parser.add_argument("--mask_dir", default=None)

    parser.add_argument("--backbone", "-b", default="wideresnet50")
    parser.add_argument(
        "--layers_to_extract_from",
        "--layers",
        "-le",
        action="append",
        default=None,
        help="Backbone layers, e.g. --layers layer2 --layers layer3.",
    )
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1024)
    parser.add_argument("--target_embed_dimension", type=int, default=1024)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--patchstride", type=int, default=1)
    parser.add_argument("--anomaly_scorer_num_nn", type=int, default=1)
    parser.add_argument(
        "--sampler",
        choices=["identity", "greedy_coreset", "approx_greedy_coreset"],
        default="approx_greedy_coreset",
    )
    parser.add_argument("--sampling_percentage", "-p", type=float, default=0.1)
    parser.add_argument("--faiss_num_workers", type=int, default=4)

    parser.add_argument(
        "-imgsz",
        "--resize",
        dest="resize",
        type=int,
        default=256,
        help="Initial resize before center cropping.",
    )
    parser.add_argument(
        "-csz",
        "--imagesize",
        "--image_size",
        dest="imagesize",
        type=int,
        default=224,
        help="Final model input size.",
    )
    parser.add_argument("--train_val_split", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--gpu",
        "--cuda",
        dest="gpu",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
