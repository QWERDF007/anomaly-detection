"""Cache PatchCore score maps, plot score distributions, and evaluate them.

The dataset layout follows MVTec AD and the custom PatchCore dataset::

    data_root/train/good/
    data_root/test/good/
    data_root/test/<anomaly_type>/
    data_root/ground_truth/<anomaly_type>/   # optional

Raw score maps are saved under ``output_dir/scores``.  Every map has a
``.npy.json`` sidecar containing PatchCore's native image score, allowing
cache-only evaluation to use exactly the same image-level score as training.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import measure
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SHARED_UTILS = PROJECT_ROOT / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS))

from score_workflow_common import (  # noqa: E402
    IMAGE_EXTENSIONS,
    MASK_EXTENSIONS,
    STANDARD_GROUPS,
    find_child_directory as common_find_child_directory,
    group_score_values,
    iter_images as common_iter_images,
    plot_score_distribution as plot_common_score_distribution,
    save_classification_threshold,
    write_score_table,
)
from patchcore_evaluation import (
    CLASSIFICATION_METRIC_NAMES,
    classification_metrics,
    compute_evaluation_metrics,
    evaluate_pixel_metrics,
    load_score_map,
    select_optimal_threshold,
    write_metrics,
    write_per_image_pixel_metrics,
)
from predict import _aggregate_outputs, load_models, select_device
from patchcore.datasets.custom import get_data_transforms


LOGGER = logging.getLogger("patchcore.score_visualization")
GROUPS = STANDARD_GROUPS


def _find_child_directory(root: Path, name: str) -> Optional[Path]:
    return common_find_child_directory(root, name)


def _iter_images(root: Path) -> List[Path]:
    return common_iter_images(root)


def _resolve_groups(data_root: Path) -> Dict[str, List[Path]]:
    train_root = _find_child_directory(data_root, "train")
    test_root = _find_child_directory(data_root, "test")
    if train_root is None or test_root is None:
        raise FileNotFoundError(
            "data_root must contain train/ and test/ directories: " f"{data_root}"
        )
    train_good = _find_child_directory(train_root, "good")
    test_good = _find_child_directory(test_root, "good")
    if train_good is None or test_good is None:
        raise FileNotFoundError(
            "data_root must contain train/good/ and test/good/ directories."
        )
    anomalous = sorted(
        (
            path
            for path in test_root.iterdir()
            if path.is_dir() and path.name.lower() != "good"
        ),
        key=lambda path: str(path).lower(),
    )
    if not anomalous:
        LOGGER.warning("No anomalous test subdirectory found under %s", test_root)
    return {"train_good": [train_good], "test_good": [test_good], "test_anomaly": anomalous}


def _score_path(output_dir: Path, group_key: str, image_root: Path, image_path: Path) -> Path:
    root = output_dir / "scores" / group_key
    if group_key == "test_anomaly":
        root = root / image_root.name
    return root / image_path.relative_to(image_root).with_suffix(".npy")


def _sidecar_path(score_path: Path) -> Path:
    return score_path.with_suffix(score_path.suffix + ".json")


def _load_cached_native_score(score_path: Path) -> Optional[float]:
    sidecar = _sidecar_path(score_path)
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as file:
            value = float(json.load(file)["image_score"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if np.isfinite(value) else None


def _save_native_score(
    score_path: Path,
    image_score: float,
    image_path: Path,
    resize: Optional[int] = None,
    imagesize: Optional[int] = None,
) -> None:
    sidecar = _sidecar_path(score_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "image_score": float(image_score),
                "image_path": str(image_path),
                "resize": resize,
                "imagesize": imagesize,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def _predict_jobs(models, jobs: Sequence[Tuple[str, Path, Path]], transform):
    """Predict all jobs, then apply the same ensemble aggregation as predict.py.

    Ensemble member outputs are normalized across the full input set before
    averaging.  Aggregating one image at a time would turn every one-image
    score range into zero, so the batch-wide aggregation here is essential.
    """

    model_scores = [[] for _ in models]
    model_maps = [[] for _ in models]
    for _group_key, _image_root, image_path in tqdm(
        jobs, desc="PatchCore inference", unit="image", dynamic_ncols=True
    ):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        for index, model in enumerate(models):
            scores, maps = model.predict(tensor)
            model_scores[index].append(float(scores[0]))
            model_maps[index].append(np.asarray(maps[0], dtype=np.float32))
    scores, maps = _aggregate_outputs(model_scores, model_maps)
    return [
        (float(score), np.squeeze(np.asarray(score_map, dtype=np.float32)))
        for score, score_map in zip(scores, maps)
    ]


def _iter_jobs(groups: Dict[str, List[Path]]) -> Iterable[Tuple[str, Path, Path]]:
    for group_key, _label in GROUPS:
        for image_root in groups[group_key]:
            for image_path in _iter_images(image_root):
                yield group_key, image_root, image_path


def prepare_samples(
    groups: Dict[str, List[Path]],
    output_dir: Path,
    models,
    transform,
    force_recompute: bool,
    resize: int,
    imagesize: int,
) -> List[Dict[str, object]]:
    """Load cached maps or infer only the individual missing map/score pairs."""

    labels = dict(GROUPS)
    samples: List[Dict[str, object]] = []
    jobs = list(_iter_jobs(groups))
    cached = []
    for group_key, image_root, image_path in jobs:
        score_path = _score_path(output_dir, group_key, image_root, image_path)
        image_score = None if force_recompute else _load_cached_native_score(score_path)
        cached.append(
            (score_path.is_file() and image_score is not None, score_path, image_score)
        )

    # A single PatchCore model can fill only missing cache entries.  An
    # ensemble must recompute all entries when the cache is incomplete because
    # its normalization is defined over the complete input collection.
    recompute_all = force_recompute or (len(models) > 1 and not all(item[0] for item in cached))
    predictions = (
        _predict_jobs(models, jobs, transform)
        if recompute_all
        else [None] * len(jobs)
    )

    for index, (group_key, image_root, image_path) in enumerate(tqdm(
        jobs, desc="Prepare score cache", unit="image", dynamic_ncols=True
    )):
        is_cached, score_path, image_score = cached[index]
        if recompute_all:
            image_score, score_map = predictions[index]
            score_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(score_path, score_map)
            _save_native_score(score_path, image_score, image_path, resize, imagesize)
        elif is_cached:
            score_map = load_score_map(score_path)
        else:
            image_score, score_map = _predict_jobs(models, [jobs[index]], transform)[0]
            score_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(score_path, score_map)
            _save_native_score(score_path, image_score, image_path, resize, imagesize)

        samples.append(
            {
                "group_key": group_key,
                "group_label": labels[group_key],
                "image_path": image_path,
                "image_root": image_root,
                "anomaly_type": image_root.name if group_key == "test_anomaly" else "",
                "score_path": score_path,
                "score": float(image_score),
                "score_map_shape": tuple(score_map.shape),
            }
        )
    return samples


def _find_ground_truth_mask(sample: Dict[str, object], ground_truth_dir: Path) -> Optional[Path]:
    image_path = Path(sample["image_path"])
    image_root = Path(sample["image_root"])
    anomaly_type = str(sample["anomaly_type"])
    roots = [ground_truth_dir / anomaly_type, ground_truth_dir]
    relative = image_path.relative_to(image_root)
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for extension in MASK_EXTENSIONS:
            candidates.extend(
                (
                    root / relative.with_suffix(extension),
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
                in {image_path.stem.lower(), f"{image_path.stem.lower()}_mask", f"{image_path.stem.lower()}-mask"}
            )
    unique = sorted({path.resolve() for path in matches}, key=lambda path: str(path).lower())
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(
            f"Multiple ground-truth masks match {image_path}: " + ", ".join(map(str, unique))
        )
    return None


def _load_transformed_mask(mask_path: Path, mask_transform, expected_shape: Tuple[int, int]) -> np.ndarray:
    mask = mask_transform(Image.open(mask_path).convert("L")).numpy()
    mask = np.squeeze(mask > 0).astype(np.uint8)
    if mask.shape != expected_shape:
        mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def collect_gt_scores(samples: Sequence[Dict[str, object]], ground_truth_dir: Optional[Path], mask_transform) -> List[float]:
    """Attach one GT-mask maximum to each anomalous image and return region maxima."""

    if ground_truth_dir is None:
        return []
    values: List[float] = []
    for sample in samples:
        if sample["group_key"] != "test_anomaly":
            continue
        score_map = load_score_map(Path(sample["score_path"]))
        mask_path = _find_ground_truth_mask(sample, ground_truth_dir)
        if mask_path is None:
            raise FileNotFoundError(f"Ground-truth mask not found for {sample['image_path']}")
        gt_mask = _load_transformed_mask(mask_path, mask_transform, score_map.shape)
        labels = measure.label(gt_mask)
        region_scores = []
        for region_id in range(1, int(labels.max()) + 1):
            region_values = score_map[labels == region_id]
            region_values = region_values[np.isfinite(region_values)]
            if region_values.size:
                region_scores.append(float(region_values.max()))
        if region_scores:
            sample["gt_score"] = max(region_scores)
            values.extend(region_scores)
    return values


def _group_scores(samples: Sequence[Dict[str, object]]) -> Dict[str, List[float]]:
    return group_score_values(samples)


def _choose_threshold(
    samples: Sequence[Dict[str, object]], explicit: Optional[float]
) -> Tuple[float, str, Dict[str, float]]:
    evaluation_samples = [sample for sample in samples if sample["group_key"] != "train_good"]
    labels = np.asarray(
        [sample["group_key"] == "test_anomaly" for sample in evaluation_samples],
        dtype=np.uint8,
    )
    scores = np.asarray([sample["score"] for sample in evaluation_samples], dtype=np.float32)
    if explicit is not None:
        threshold = float(explicit)
        return threshold, "manual", classification_metrics(labels, scores, threshold)
    return select_optimal_threshold(labels, scores)


def _plot_distribution(
    groups: Dict[str, List[float]], gt_scores: Sequence[float], threshold: float, output_path: Path, bins: int
) -> None:
    plot_common_score_distribution(
        groups,
        output_path,
        threshold,
        bins,
        title="PatchCore score distribution",
        xlabel="PatchCore anomaly score",
        gt_score_values=gt_scores,
    )


def _save_score_table(samples: Sequence[Dict[str, object]], output_path: Path) -> None:
    write_score_table(samples, output_path, include_score_path=True)


def _evaluate(
    samples: Sequence[Dict[str, object]], ground_truth_dir: Path, mask_transform
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    labels = []
    scores = []
    maps = []
    masks = []
    records: List[Dict[str, object]] = []
    evaluation_samples = [sample for sample in samples if sample["group_key"] != "train_good"]
    for sample in tqdm(evaluation_samples, desc="Evaluate score maps", unit="image", dynamic_ncols=True):
        score_map = load_score_map(Path(sample["score_path"]))
        if sample["group_key"] == "test_anomaly":
            mask_path = _find_ground_truth_mask(sample, ground_truth_dir)
            if mask_path is None:
                raise FileNotFoundError(f"Ground-truth mask not found for {sample['image_path']}")
            gt_mask = _load_transformed_mask(mask_path, mask_transform, score_map.shape)
        else:
            mask_path = None
            gt_mask = np.zeros(score_map.shape, dtype=np.uint8)
        image_label = int(sample["group_key"] == "test_anomaly")
        labels.append(image_label)
        scores.append(float(sample["score"]))
        maps.append(score_map)
        masks.append(gt_mask)
        pixel_metrics = evaluate_pixel_metrics(gt_mask, score_map)
        records.append(
            {
                "group": sample["group_label"],
                "image_path": str(sample["image_path"]),
                "score_path": str(sample["score_path"]),
                "image_label": image_label,
                "image_score": float(sample["score"]),
                "gt_positive_pixels": int(gt_mask.sum()),
                **{name: pixel_metrics[name] for name in pixel_metrics if name.startswith("P-")},
            }
        )
    return compute_evaluation_metrics(scores, labels, np.stack(maps), np.stack(masks)), records


def _image_only_metrics(samples: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Calculate image metrics when pixel GT is unavailable."""

    evaluation_samples = [sample for sample in samples if sample["group_key"] != "train_good"]
    return compute_evaluation_metrics(
        [float(sample["score"]) for sample in evaluation_samples],
        [sample["group_key"] == "test_anomaly" for sample in evaluation_samples],
        np.asarray([], dtype=np.float32),
        np.asarray([], dtype=np.uint8),
    )


def _save_visualizations(samples: Sequence[Dict[str, object]], output_dir: Path, threshold: float) -> None:
    finite_values = []
    for sample in samples:
        values = load_score_map(Path(sample["score_path"]))
        finite_values.extend(values[np.isfinite(values)].reshape(-1).tolist())
    low = float(min(finite_values)) if finite_values else 0.0
    high = float(max(finite_values)) if finite_values else low
    for sample in tqdm(samples, desc="Save visualizations", unit="image", dynamic_ncols=True):
        score_map = load_score_map(Path(sample["score_path"]))
        image_path = Path(sample["image_path"])
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise OSError(f"Cannot read source image: {image_path}")
        height, width = source.shape[:2]
        display_map = cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR)
        normalized = np.zeros_like(display_map, dtype=np.uint8) if high <= low else np.uint8(np.clip((display_map - low) / (high - low) * 255.0, 0, 255))
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(source, 0.5, heatmap, 0.5, 0.0)
        mask = cv2.resize((score_map >= threshold).astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
        root = output_dir / "visualizations" / str(sample["group_key"])
        if sample["group_key"] == "test_anomaly":
            root = root / str(sample["anomaly_type"])
        relative = image_path.relative_to(Path(sample["image_root"]))
        heatmap_path = root / "heatmap" / relative.with_suffix(".png")
        overlay_path = root / "overlay" / relative.with_suffix(".png")
        mask_path = root / "mask" / relative.with_suffix(".png")
        for path in (heatmap_path, overlay_path, mask_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(heatmap_path), heatmap) or not cv2.imwrite(str(overlay_path), overlay) or not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"Cannot save visualization for {image_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PatchCore score-map cache, distribution plot, and evaluation.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "数据目录：data_root/train/good、data_root/test/good、"
            "data_root/test/<异常类型>，GT 可位于 data_root/ground_truth/<异常类型>。\n"
            "缓存 .npy 的同名 .npy.json 保存 PatchCore 原生 image score，"
            "离线评估可与训练的图像指标一致。"
        ),
    )
    parser.add_argument("-i", "--data_root", required=True, type=Path)
    parser.add_argument("-m", "--model", required=True, help="PatchCore 模型目录或 patchcore_params.pkl。")
    parser.add_argument("-o", "--output_dir", required=True, type=Path)
    parser.add_argument("-gt", "--ground_truth_dir", type=Path, default=None)
    parser.add_argument("--category", default=None, help="模型含多个类别时选择的类别。")
    parser.add_argument("--resize", "--image_size", dest="resize", type=int, default=None, help="需与训练时 Resize 一致。")
    parser.add_argument("--imagesize", "--crop_size", dest="imagesize", type=int, default=None, help="需与训练时 CenterCrop 一致。")
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=4)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="图像判定阈值；不指定时按最大平衡准确率自动选择。",
    )
    parser.add_argument("--vis", dest="save_visualizations", action="store_true")
    parser.add_argument("--force_recompute", action="store_true")
    parser.set_defaults(save_visualizations=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bins < 1:
        raise ValueError("bins must be positive")
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = select_device(args.gpu)
    models = load_models(args.model, device, args.faiss_on_gpu, args.faiss_num_workers, args.category)
    input_height = int(models[0].input_shape[-2])
    resize = args.resize or input_height
    imagesize = args.imagesize or input_height
    transform, mask_transform = get_data_transforms(resize, imagesize)
    groups = _resolve_groups(data_root)
    samples = prepare_samples(
        groups,
        output_dir,
        models,
        transform,
        args.force_recompute,
        resize,
        imagesize,
    )
    if not samples:
        raise RuntimeError("No images found in the configured dataset groups.")

    ground_truth_dir = args.ground_truth_dir.expanduser().resolve() if args.ground_truth_dir else _find_child_directory(data_root, "ground_truth")
    if ground_truth_dir is not None and not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {ground_truth_dir}")
    gt_scores = collect_gt_scores(samples, ground_truth_dir, mask_transform)
    score_groups = _group_scores(samples)
    threshold, threshold_method, threshold_metrics = _choose_threshold(
        samples, args.score_threshold
    )
    _plot_distribution(score_groups, gt_scores, threshold, output_dir / "score_distribution.png", args.bins)
    _save_score_table(samples, output_dir / "score_values.csv")
    save_classification_threshold(
        output_dir / "score_threshold.json",
        threshold,
        threshold_method,
        threshold_metrics,
        extra={"score_threshold": threshold, "gt_region_count": len(gt_scores)},
    )

    if ground_truth_dir is not None:
        metrics, records = _evaluate(samples, ground_truth_dir, mask_transform)
        write_per_image_pixel_metrics(records, output_dir / "pixel_metrics.csv", extra_fields=("group",))
    else:
        metrics = _image_only_metrics(samples)
        LOGGER.info("No ground_truth directory; pixel-level metrics are unavailable.")
    metrics.update(
        {name: threshold_metrics[name] for name in CLASSIFICATION_METRIC_NAMES}
    )
    write_metrics({"score_maps": metrics}, output_dir)

    if args.save_visualizations:
        _save_visualizations(samples, output_dir, threshold)
    print(f"Done. score_threshold={threshold:.6f} ({threshold_method}), images={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
