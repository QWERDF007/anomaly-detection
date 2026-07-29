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
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from skimage import measure
from tqdm import tqdm
from torchvision.ops import roi_align

from dinomaly_evaluation import evaluate_stage, print_and_save_metrics
from dinomaly_pipeline_common import (
    GROUPS,
    artifact_root,
    choose_threshold,
    common_grid,
    extract_feature_map,
    find_child_directory,
    group_roots,
    ground_truth_relative_path,
    iter_image_paths,
    load_ground_truth,
    load_transform,
    plot_group_density,
    relative_output_path,
    resolve_group_directory,
    resolve_non_good_directories,
    save_fused_heatmap,
    score_map_from_outputs,
    score_values_by_group,
    select_device,
)
from models.uad import Dinomaly
from predict import build_model
from roi_feature_utils import (
    annotation_path_for_image,
    l2_normalize,
    load_labelme_annotation,
    load_feature_map,
    load_search_index,
    mask_bbox,
    polygon_to_feature_mask,
    resize_mask_to_feature,
)
from utils import get_gaussian_kernel


LOGGER = logging.getLogger("dinomaly_roi_pipeline")


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
            score_map = score_map_from_outputs(
                encoder_output,
                decoder_output,
                original.shape[:2],
                device,
                gaussian_filter,
            )
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

    feature_nchw = feature_map.detach().cpu().numpy().astype(np.float32)
    return score_map, feature_nchw


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
            samples.append(
                {
                    "group_key": group_key,
                    "group_label": dict(GROUPS)[group_key],
                    "image_path": image_path,
                    "image_root": image_root,
                    "ground_truth_relative": ground_truth_relative_path(
                        group_key,
                        image_path,
                        image_root,
                        len(roots),
                    ),
                    "score_path": score_path,
                    "feature_path": feature_path,
                    "before_score": float(score_map.max()),
                    "rois": [],
                    "after_score": 0.0,
                }
            )
    return samples


def plot_distance_distribution(
    groups: Dict[str, List[float]],
    output_path: Path,
    threshold: Optional[float],
    bins: int,
    gt_anomaly_roi_distances: Optional[Sequence[float]] = None,
) -> None:
    gt_anomaly_roi_distances = list(gt_anomaly_roi_distances or [])
    all_groups = dict(groups)
    all_groups["Test / Anomaly / GT-overlap ROI"] = gt_anomaly_roi_distances
    grid = common_grid(all_groups, bins)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(10, 16),
        sharex=True,
    )
    for axis, (_, label) in zip(axes[:3], GROUPS):
        plot_group_density(
            axis,
            {label: groups.get(label, [])},
            grid,
            label,
            threshold,
            xlabel="FAISS squared L2 distance",
            bins=bins,
        )
    plot_group_density(
        axes[3],
        {"Test / Anomaly / GT-overlap ROI": gt_anomaly_roi_distances},
        grid,
        "Test / Anomaly / GT-overlap ROI",
        threshold,
        xlabel="FAISS squared L2 distance",
        bins=bins,
        color_overrides={
            "Test / Anomaly / GT-overlap ROI": "crimson",
        },
    )
    figure.suptitle("ROI FAISS Distance Distribution", y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def collect_gt_anomaly_roi_distances(
    samples: Sequence[Dict],
    ground_truth_dir: Path,
) -> List[float]:
    """Collect distances for Test/Anomaly ROIs overlapping GT anomalies."""

    distances: List[float] = []
    for sample in samples:
        if sample["group_key"] != "test_anomaly":
            continue

        score_map = np.asarray(
            np.load(sample["score_path"]),
            dtype=np.float32,
        )
        score_map = np.squeeze(score_map)
        if score_map.ndim != 2:
            raise ValueError(f"Expected 2D score map: {sample['score_path']}")

        gt_mask = load_ground_truth(
            sample,
            ground_truth_dir,
            score_map.shape,
        ).astype(bool, copy=False)
        for roi in sample["rois"]:
            roi_mask = np.asarray(roi["mask"], dtype=bool)
            if roi_mask.shape != gt_mask.shape:
                raise ValueError(
                    "ROI mask and ground-truth mask shapes do not match for "
                    f"{sample['image_path']}"
                )
            if np.any(roi_mask & gt_mask):
                distance = float(roi["distance"])
                if np.isfinite(distance):
                    distances.append(distance)
    return distances


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
        3,
        1,
        figsize=(10, 12),
        sharex=True,
    )
    for axis, (_, label) in zip(axes, GROUPS):
        plot_group_density(
            axis,
            {
                "Before": before.get(label, []),
                "After": after.get(label, []),
            },
            grid,
            label,
            score_threshold,
            bins=bins,
            color_overrides={
                "Before": "darkorange",
                "After": "royalblue",
            },
        )
    figure.suptitle("Score Distribution Before/After Distance Filtering", y=0.995)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


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


def dilate_binary_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Expand a binary mask by ``iterations`` 8-connected pixel rings."""

    if iterations <= 0:
        return np.asarray(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(
        np.asarray(mask, dtype=np.uint8),
        kernel,
        iterations=int(iterations),
    )


def roi_align_vectors(
    feature_chw: np.ndarray,
    entries: Sequence[Dict],
    output_size: int,
    device: Optional[torch.device] = None,
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

    roi_device = device or torch.device("cpu")
    boxes_tensor = torch.as_tensor(
        boxes,
        dtype=torch.float32,
        device=roi_device,
    )
    feature_tensor = torch.from_numpy(feature).unsqueeze(0).to(roi_device)
    pooled = roi_align(
        feature_tensor,
        boxes_tensor,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True,
    )

    mask_tensor = torch.from_numpy(
        np.stack(masks, axis=0),
    ).unsqueeze(1).to(roi_device)
    mask_boxes = boxes_tensor.clone()
    mask_boxes[:, 0] = torch.arange(
        len(entries),
        dtype=torch.float32,
        device=roi_device,
    )
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
    return pooled.reshape(len(entries), channels).detach().cpu().numpy().astype(
        np.float32,
        copy=False,
    )


def annotation_path_for_sample(
    sample: Dict,
    annotation_root: Path,
    data_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve Labelme JSON for both split-preserving and legacy layouts.

    Preferred layout mirrors the dataset relative to ``data_root``::

        annotations/train/good/image.json
        annotations/test/bad/image.json
        annotations/test/scratch/image.json

    Legacy flat/name-based layouts remain supported as fallbacks.
    """

    annotation_root = Path(annotation_root)
    image_path = Path(sample["image_path"])
    image_root = Path(sample["image_root"])
    candidates: List[Path] = []

    if data_root is not None:
        try:
            relative = image_path.relative_to(Path(data_root))
        except ValueError:
            relative = None
        if relative is not None:
            candidates.append(annotation_root / relative.with_suffix(".json"))

    try:
        relative_to_group = image_path.relative_to(image_root)
    except ValueError:
        relative_to_group = Path(image_path.name)

    group_key = sample.get("group_key")
    if group_key == "train_good":
        candidates.append(
            annotation_root
            / "train"
            / "good"
            / relative_to_group.with_suffix(".json")
        )
    elif group_key == "test_anomaly":
        candidates.append(
            annotation_root
            / "test"
            / image_root.name
            / relative_to_group.with_suffix(".json")
        )

    # Keep compatibility with the previous flat annotation directory and
    # basename-based lookup.
    candidates.extend(
        [
            annotation_root / relative_to_group.with_suffix(".json"),
            annotation_root / f"{image_path.stem}.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return annotation_path_for_image(
        image_path,
        annotation_root,
        image_root,
    )


def build_roi_index(
    samples: Sequence[Dict],
    annotation_root: Path,
    output_dir: Path,
    args,
    data_root: Optional[Path] = None,
    roi_device: Optional[torch.device] = None,
) -> Tuple[Path, Path]:
    vectors = []
    records = []
    feature_dim = None
    # Every discovered sample is eligible. Samples without an annotation or
    # without valid polygon shapes are skipped below.
    index_samples = list(samples)

    for sample in tqdm(
        index_samples,
        desc="Build ROI FAISS index",
        unit="image",
        dynamic_ncols=True,
    ):
        annotation_path = annotation_path_for_sample(
            sample,
            annotation_root,
            data_root,
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
            device=roi_device,
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
                    "group": sample["group_label"],
                    "group_key": sample["group_key"],
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
            "No polygon ROI features were collected from any sample. "
            "Check --train_annotation_dir and the annotation layout."
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
    roi_device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index, _resources = load_search_index(
        index_path,
        on_gpu=args.gpu >= 0,
        gpu_id=args.gpu,
    )
    roi_size = int(args.roi_size)
    roi_dilation = int(getattr(args, "roi_dilation", 0))
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
            # Use the expanded region only to build the feature-map ROI.
            # Keep the original component mask for score filtering and output.
            feature_roi_mask = dilate_binary_mask(
                component["mask"],
                roi_dilation,
            )
            mask_feature = resize_mask_to_feature(
                feature_roi_mask,
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
                    "component_id": int(component["id"]),
                    "score": float(score_map[component["mask"]].max()),
                }
            )

        vectors = roi_align_vectors(
            feature,
            roi_entries,
            output_size=roi_size,
            device=roi_device,
        )
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
                    "roi_id": entry["component_id"],
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


def save_visualization_payload(payload: Dict) -> None:
    score_map = np.asarray(
        np.load(payload["score_path"]),
        dtype=np.float32,
    )
    score_map = np.squeeze(score_map)
    mask_shape = tuple(payload["mask_shape"])
    pixel_count = int(np.prod(mask_shape))
    before_roi_mask = np.unpackbits(
        np.frombuffer(payload["before_mask_bits"], dtype=np.uint8),
    )[:pixel_count].reshape(mask_shape).astype(np.uint8, copy=False)
    after_roi_mask = np.unpackbits(
        np.frombuffer(payload["after_mask_bits"], dtype=np.uint8),
    )[:pixel_count].reshape(mask_shape).astype(np.uint8, copy=False)
    after_score_map = np.where(after_roi_mask > 0, score_map, 0.0).astype(
        np.float32,
        copy=False,
    )

    sample = {
        "group_key": payload["group_key"],
        "ground_truth_relative": Path(payload["ground_truth_relative"]),
    }
    image_path = Path(payload["image_path"])
    output_dir = Path(payload["output_dir"])
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

    save_fused_heatmap(
        score_map,
        image_path,
        visualization_path(output_dir, "before", "heatmap", sample),
        reference_min,
        reference_max,
        source_image=source_image,
    )
    save_fused_heatmap(
        after_score_map,
        image_path,
        visualization_path(output_dir, "after", "heatmap", sample),
        reference_min,
        reference_max,
        source_image=source_image,
    )
    if not cv2.imwrite(
        str(visualization_path(output_dir, "before", "mask", sample)),
        before_roi_mask * 255,
    ):
        raise OSError(f"Cannot write before mask for {image_path}")
    if not cv2.imwrite(
        str(visualization_path(output_dir, "after", "mask", sample)),
        after_roi_mask * 255,
    ):
        raise OSError(f"Cannot write after mask for {image_path}")


def visualization_payload(sample: Dict, output_dir: Path) -> Dict:
    before_mask = np.asarray(sample["before_roi_mask"], dtype=np.uint8)
    after_mask = np.asarray(sample["after_roi_mask"], dtype=np.uint8)
    return {
        "output_dir": str(output_dir),
        "score_path": str(sample["score_path"]),
        "image_path": str(sample["image_path"]),
        "group_key": sample["group_key"],
        "ground_truth_relative": str(sample["ground_truth_relative"]),
        "mask_shape": before_mask.shape,
        "before_mask_bits": np.packbits(before_mask.reshape(-1)).tobytes(),
        "after_mask_bits": np.packbits(after_mask.reshape(-1)).tobytes(),
    }


def save_visualization_artifacts(
    sample: Dict,
    output_dir: Path,
) -> None:
    """Compatibility wrapper for saving one visualization payload."""

    save_visualization_payload(visualization_payload(sample, output_dir))


def save_roi_visualizations_and_report(
    samples: Sequence[Dict],
    output_dir: Path,
    distance_threshold: float,
    workers: int = 8,
) -> None:
    """Save visualizations in separate processes plus a per-image report."""

    workers = max(1, int(workers))
    payloads = [
        visualization_payload(sample, output_dir)
        for sample in samples
    ]
    process_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=process_context,
    ) as executor:
        futures = [
            executor.submit(save_visualization_payload, payload)
            for payload in payloads
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Save visualizations ({workers} processes)",
            unit="image",
            dynamic_ncols=True,
        ):
            future.result()

    report_path = output_dir / "roi_filter_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group",
                "image_path",
                "roi_id",
                "roi_score",
                "before_score",
                "after_score",
                "score_difference",
                "distance_threshold",
                "roi_distance_before",
                "roi_distance_after",
                "filter_status",
            ],
        )
        writer.writeheader()

        for sample in samples:
            for roi_index, roi in enumerate(sample["rois"]):
                distance = float(roi["distance"])
                kept = distance >= distance_threshold
                writer.writerow(
                    {
                        "group": sample["group_label"],
                        "image_path": str(sample["image_path"]),
                        "roi_id": roi.get("roi_id", roi_index),
                        "roi_score": roi["score"],
                        "before_score": sample["before_score"],
                        "after_score": sample["after_score"],
                        "score_difference": (
                            sample["before_score"] - sample["after_score"]
                        ),
                        "distance_threshold": distance_threshold,
                        "roi_distance_before": distance,
                        "roi_distance_after": distance if kept else "",
                        "filter_status": "kept" if kept else "filtered",
                    }
                )

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
        "--roi_dilation",
        type=int,
        default=0,
        help=(
            "ROIAlign 前在 score map 上对每个异常区域做的 8 邻域膨胀圈数；"
            "0 表示不膨胀，1 表示向外扩大一圈。"
        ),
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
    parser.add_argument(
        "--vis_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="可视化图片保存进程数；仅在指定 --vis 时生效。",
    )
    parser.set_defaults(normalize=True, save_visualizations=False)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size < 1 or args.crop_size < 1:
        raise ValueError("image_size and crop_size must be positive.")
    if args.roi_size < 1:
        raise ValueError("roi_size must be positive.")
    if args.roi_dilation < 0:
        raise ValueError("roi_dilation must be non-negative.")
    if args.vis_workers < 1:
        raise ValueError("vis_workers must be positive.")
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
    LOGGER.info(
        "ROI feature-mask dilation: %d ring(s)",
        args.roi_dilation,
    )
    LOGGER.info("ROIAlign device: %s", device)
    index_path, metadata_path = build_roi_index(
        samples,
        train_annotation_dir,
        output_dir,
        args,
        data_root=data_root,
        roi_device=device,
    )
    distance_groups = query_score_rois(
        samples,
        index_path,
        metadata_path,
        argparse.Namespace(
            roi_size=args.roi_size,
            roi_dilation=args.roi_dilation,
            score_threshold=score_threshold,
            gpu=args.gpu,
        ),
        roi_device=device,
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
    print(
        "Plotting ROI distance distribution "
        f"({sum(len(values) for values in distance_groups.values())} ROIs)...",
        flush=True,
    )
    gt_anomaly_roi_distances = collect_gt_anomaly_roi_distances(
        samples,
        ground_truth_dir,
    )
    print(
        "GT-overlap Test/Anomaly ROI distances: "
        f"{len(gt_anomaly_roi_distances)} ROIs",
        flush=True,
    )
    plot_distance_distribution(
        distance_groups,
        output_dir / "distance_distribution.png",
        distance_threshold,
        args.bins,
        gt_anomaly_roi_distances=gt_anomaly_roi_distances,
    )

    print("Applying ROI distance filtering...", flush=True)
    calculate_after_scores(samples, distance_threshold)
    if args.save_visualizations:
        print("Saving before/after visualizations...", flush=True)
        save_roi_visualizations_and_report(
            samples,
            output_dir,
            distance_threshold,
            workers=args.vis_workers,
        )
    else:
        LOGGER.info("Visualization output disabled; pass --vis to enable")
    after_groups = score_values_by_group(samples, "after_score")
    print("Evaluating before filtering...", flush=True)
    before_metrics = evaluate_stage(
        samples,
        ground_truth_dir,
        metric_size=args.metric_size,
        score_map_key="score_path",
        image_score_key="before_score",
        stage_name="before filtering",
    )
    print("Evaluating after filtering...", flush=True)
    after_metrics = evaluate_stage(
        samples,
        ground_truth_dir,
        metric_size=args.metric_size,
        score_map_key="filtered_score_map",
        image_score_key="after_score",
        stage_name="after filtering",
    )
    print_and_save_metrics(
        {
            "before_distance_filtering": before_metrics,
            "after_distance_filtering": after_metrics,
        },
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
