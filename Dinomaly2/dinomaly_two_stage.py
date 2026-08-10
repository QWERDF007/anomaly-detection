"""Two-stage Dinomaly2 inference with good/anomaly ROI feature libraries.

The first stage is the regular Dinomaly2 image score.  Only images whose
score is strictly greater than ``--score_threshold`` enter the second stage:

* connected components of the score map become candidate anomaly regions;
* the same region is mapped through Dinomaly2's Resize + CenterCrop geometry;
* the encoder output is ROIAligned and queried against both a good and an
  anomaly FAISS library;
* the distance margin is converted into a bounded score offset; and
* a good match subtracts the offset while an anomaly match adds it.

The ``build-library`` subcommand creates either library from images and a
binary mask (PNG/TIFF/NPY) or a Labelme JSON mask.  The ``build-libraries``
subcommand can additionally split Labelme shapes into the good and anomaly
libraries according to their ``label`` values.  By default both library types
use the encoder features returned by ``Dinomaly.forward`` rather than the
decoder or a second feature extractor, so the library and query
representations stay in the same feature space.

With ``--feature_source raw_patch``, the second stage instead uses the final
patch-token output (``x_norm_patchtokens``) of the same ``--backbone``
loaded standalone from the project's local ``dinov2.hub.backbones`` (or the
``vit_encoder`` for DINOv3).  The Dinomaly2 model still produces the stage-1
anomaly map; only the library/query representation switches to the raw patch
tokens, reshaped to an NCHW feature map and processed by the same ROI
mapping and masked ROIAlign.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
from tqdm import tqdm

from predict import build_model
from utils import cal_anomaly_maps, get_gaussian_kernel


LOGGER = logging.getLogger("dinomaly_two_stage")

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MASK_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".npy",
    ".json",
)


@dataclass
class SearchLibrary:
    """A loaded FAISS index and the metadata needed to query it."""

    index: Any
    metadata: Dict[str, Any]
    index_path: Path
    metadata_path: Path
    resources: Any = None


def classify_score(score: float, threshold: float) -> str:
    """Classify a score using the pipeline's strict ``>`` threshold rule."""

    return "anomaly" if float(score) > float(threshold) else "normal"


def calculate_distance_offset(
    good_distance: float,
    anomaly_distance: float,
    offset_scale: float = 1.0,
    max_offset: Optional[float] = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Turn two nearest-neighbour distances into a signed score adjustment.

    FAISS returns smaller distances for more similar features.  The offset
    magnitude is the smaller of the two distances (the distance to the
    nearer library)::

        offset = min(d_good, d_anomaly) * offset_scale   (capped by max_offset)

    A good match has a negative signed offset; an anomaly match has a
    positive signed offset.  Equal or invalid distances produce no
    correction and are reported as ``tie``/``invalid``.
    """

    good = float(good_distance)
    anomaly = float(anomaly_distance)
    if not np.isfinite(good) or not np.isfinite(anomaly):
        return {
            "similar_library": "invalid",
            "confidence": 0.0,
            "offset": 0.0,
            "signed_offset": 0.0,
        }

    good = max(good, 0.0)
    anomaly = max(anomaly, 0.0)
    if abs(good - anomaly) <= max(float(eps), 0.0):
        return {
            "similar_library": "tie",
            "confidence": 0.0,
            "offset": 0.0,
            "signed_offset": 0.0,
        }

    nearer = min(good, anomaly)
    offset = nearer * max(float(offset_scale), 0.0)
    if max_offset is not None:
        offset = min(offset, max(float(max_offset), 0.0))

    similar_library = "good" if good < anomaly else "anomaly"
    signed_offset = -offset if similar_library == "good" else offset
    return {
        "similar_library": similar_library,
        "confidence": float(nearer),
        "offset": float(offset),
        "signed_offset": float(signed_offset),
    }


def select_strongest_region(regions: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """Select the region with the strongest adjusted score.

    A single image may contain multiple score-map components.  The largest
    adjusted score (``region_score + signed_offset``, with ``region_score``
    as the tiebreak) is used for the image-level correction; the ROI details
    for all components are still written to the result JSON.
    """

    if not regions:
        return None
    return max(
        regions,
        key=lambda region: (
            float(region.get("region_score", 0.0))
            + float(region.get("signed_offset", 0.0)),
            float(region.get("region_score", 0.0)),
        ),
    )


def select_device(gpu: int) -> torch.device:
    if int(gpu) >= 0 and torch.cuda.is_available():
        gpu = int(gpu)
        if gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu} is unavailable; {torch.cuda.device_count()} device(s) found."
            )
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def build_transform(args) -> transforms.Compose:
    if args.image_size < 1 or args.crop_size < 1:
        raise ValueError("image_size and crop_size must be positive")
    if args.crop_size > args.image_size:
        raise ValueError("crop_size cannot be greater than image_size")
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


def load_dinomaly_model(args, device: torch.device):
    """Build Dinomaly2 and load a plain or wrapped state-dict checkpoint."""

    checkpoint_path = Path(args.model).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Dinomaly2 checkpoint does not exist: {checkpoint_path}")

    model = build_model(args, device)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch releases without weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                state_dict = candidate
                break
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain a model state-dict."
        )

    state_dict = dict(state_dict)
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key)[len("module."):]: value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    LOGGER.info("Loaded Dinomaly2 checkpoint: %s", checkpoint_path)
    return model


def iter_image_paths(source: Path) -> List[Path]:
    source = Path(source).expanduser()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {source}")
    return sorted(
        [
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def _relative_image_path(image_path: Path, image_root: Path) -> Path:
    image_root = Path(image_root)
    if image_root.is_file():
        image_root = image_root.parent
    try:
        return Path(image_path).relative_to(image_root)
    except ValueError:
        return Path(image_path.name)


def make_image_id(image_relative: str) -> str:
    """Create a stable image ID from the image path relative to its library root."""

    normalized = str(image_relative).replace("\\", "/")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"img_{digest}"


def make_roi_id(image_id: str, component_id: int) -> str:
    """Create a stable ROI ID for one connected mask component."""

    return f"{image_id}_roi_{int(component_id):04d}"


def resolve_mask_path(
    image_path: Path,
    image_root: Path,
    mask_root: Path,
) -> Optional[Path]:
    """Find a mask by relative path, split/category path, or image stem."""

    image_path = Path(image_path)
    image_root = Path(image_root)
    mask_root = Path(mask_root).expanduser()
    if mask_root.is_file():
        return mask_root
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask path does not exist: {mask_root}")

    relative = _relative_image_path(image_path, image_root)
    relative_without_suffix = relative.with_suffix("")
    candidates: List[Path] = []

    def add_variants(base: Path) -> None:
        for extension in MASK_EXTENSIONS:
            # Do not call Path.with_suffix here: a filename such as
            # ``camera.01.jpg`` becomes ``camera.01`` after the first
            # replacement, and a second with_suffix would incorrectly turn
            # it into ``camera.png`` instead of ``camera.01.png``.
            candidates.append(Path(str(base) + extension))

    add_variants(mask_root / relative_without_suffix)
    add_variants(mask_root / f"{relative_without_suffix}_mask")
    add_variants(mask_root / image_path.stem)
    add_variants(mask_root / f"{image_path.stem}_mask")

    root_name = image_root.name if image_root.is_dir() else image_root.parent.name
    if root_name:
        add_variants(mask_root / root_name / relative_without_suffix)
        add_variants(mask_root / root_name / f"{relative_without_suffix}_mask")

    parent_name = image_root.parent.name if image_root.is_dir() else ""
    if parent_name:
        add_variants(mask_root / parent_name / root_name / relative_without_suffix)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    matches = []
    names = {f"{image_path.stem}{extension}" for extension in MASK_EXTENSIONS}
    names.update(
        f"{image_path.stem}_mask{extension}" for extension in MASK_EXTENSIONS
    )
    for candidate in mask_root.rglob("*"):
        if candidate.is_file() and candidate.name.lower() in {
            name.lower() for name in names
        }:
            matches.append(candidate)
    matches.sort(key=lambda path: str(path).lower())
    if len(matches) > 1:
        LOGGER.warning(
            "Multiple masks match %s; using %s",
            image_path,
            matches[0],
        )
    return matches[0] if matches else None


def _draw_labelme_shape(mask: np.ndarray, shape: Mapping[str, Any]) -> bool:
    """Rasterize one Labelme shape and report whether it was supported."""

    points = np.asarray(shape.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
        return False
    points = np.round(points).astype(np.int32)
    shape_type = str(shape.get("shape_type", "polygon")).lower()
    if shape_type == "rectangle" and points.shape[0] >= 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
        return True
    if shape_type == "circle" and points.shape[0] >= 2:
        center = tuple(int(value) for value in points[0])
        radius = int(np.linalg.norm(points[1] - points[0]))
        cv2.circle(mask, center, max(radius, 1), 1, -1)
        return True
    if shape_type in {"line", "linestrip"}:
        cv2.polylines(mask, [points], False, 1, thickness=1)
        return True
    if points.shape[0] >= 3:
        cv2.fillPoly(mask, [points], 1)
        return True
    return False


def _normalized_labels(labels: Sequence[str]) -> set[str]:
    """Normalize Labelme labels for case-insensitive matching."""

    return {
        str(label).strip().casefold()
        for label in labels
        if str(label).strip()
    }


def _labelme_library_masks(
    annotation: Mapping[str, Any],
    good_labels: Sequence[str],
    ignore_labels: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Return separate good/anomaly masks from one Labelme annotation.

    A shape whose normalized label is in ``good_labels`` is assigned to the
    good mask.  A shape in ``ignore_labels`` is skipped.  Every other shape
    is assigned to the anomaly mask, matching ``convert_labelme_to_mask.py``.
    """

    width = int(annotation.get("imageWidth", 0))
    height = int(annotation.get("imageHeight", 0))
    if width < 1 or height < 1:
        raise ValueError("Labelme JSON must contain positive imageWidth/imageHeight")

    good_mask = np.zeros((height, width), dtype=np.uint8)
    anomaly_mask = np.zeros((height, width), dtype=np.uint8)
    good_label_set = _normalized_labels(good_labels)
    ignore_label_set = _normalized_labels(ignore_labels)
    for shape in annotation.get("shapes", []):
        if not isinstance(shape, Mapping):
            continue
        label = str(shape.get("label", "")).strip().casefold()
        if label in ignore_label_set:
            continue
        target = good_mask if label in good_label_set else anomaly_mask
        _draw_labelme_shape(target, shape)
    return {
        "good": good_mask.astype(bool, copy=False),
        "anomaly": anomaly_mask.astype(bool, copy=False),
    }


def _labelme_mask(annotation: Mapping[str, Any]) -> np.ndarray:
    """Rasterize all Labelme shapes into one mask for legacy mask loading."""

    masks = _labelme_library_masks(annotation, good_labels=(), ignore_labels=())
    return np.logical_or(masks["good"], masks["anomaly"])


def load_mask(mask_path: Path, image_shape: Tuple[int, int], threshold: float = 0.0) -> np.ndarray:
    """Load a binary image/NPY/Labelme mask and resize it to the image."""

    mask_path = Path(mask_path)
    height, width = image_shape
    if mask_path.suffix.lower() == ".json":
        with mask_path.open("r", encoding="utf-8") as file:
            mask = _labelme_mask(json.load(file))
        mask = mask.astype(np.uint8)
    elif mask_path.suffix.lower() == ".npy":
        mask = np.asarray(np.load(mask_path, allow_pickle=False))
        if mask.ndim == 3:
            mask = np.any(mask > threshold, axis=-1)
        if mask.ndim != 2:
            raise ValueError(f"Mask NPY must be 2D: {mask_path}; got {mask.shape}")
        mask = mask.astype(np.float32, copy=False)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise OSError(f"Cannot read mask: {mask_path}")
        if mask.ndim == 3:
            mask = np.any(mask > threshold, axis=2)
        mask = np.asarray(mask)

    if mask.shape != (height, width):
        mask = cv2.resize(
            np.asarray(mask, dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(mask > threshold, dtype=bool)


def load_labelme_library_mask(
    mask_path: Path,
    image_shape: Tuple[int, int],
    library_type: str,
    good_labels: Sequence[str],
    ignore_labels: Sequence[str],
) -> np.ndarray:
    """Load only the Labelme shapes assigned to one library.

    This is intentionally separate from :func:`load_mask`: the regular mask
    loader keeps its historical behavior and treats every shape as one ROI,
    while automatic label-based library construction needs two masks from the
    same JSON document.
    """

    mask_path = Path(mask_path)
    if mask_path.suffix.lower() != ".json":
        raise ValueError(
            "Label-based library construction requires Labelme JSON masks; "
            f"got {mask_path}"
        )
    if library_type not in {"good", "anomaly"}:
        raise ValueError(f"Unsupported library type: {library_type}")
    with mask_path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)
    masks = _labelme_library_masks(annotation, good_labels, ignore_labels)
    mask = masks[library_type]
    height, width = image_shape
    if mask.shape != (height, width):
        mask = cv2.resize(
            np.asarray(mask, dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.asarray(mask != 0, dtype=bool)


def preprocess_mask(mask: np.ndarray, image_size: int, crop_size: int) -> np.ndarray:
    """Apply the same geometry as Dinomaly2's Resize + CenterCrop transform."""

    if crop_size > image_size:
        raise ValueError("crop_size cannot be greater than image_size")
    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_NEAREST,
    )
    top = (int(image_size) - int(crop_size)) // 2
    left = (int(image_size) - int(crop_size)) // 2
    cropped = resized[top:top + int(crop_size), left:left + int(crop_size)]
    return np.asarray(cropped > 0, dtype=bool)


def resize_score_map_to_feature(
    score_map: np.ndarray,
    feature_shape: Tuple[int, int],
    image_size: int,
    crop_size: int,
) -> np.ndarray:
    """Apply the image transform before sampling a score map on the feature grid.

    Score maps produced by inference are usually at the original image
    resolution (or at ``process_size``).  Resizing such a non-square map
    directly to the feature grid skips Dinomaly's Resize + CenterCrop geometry
    and can select a patch from the wrong vertical location.
    """

    score = np.asarray(score_map, dtype=np.float32)
    if score.ndim != 2:
        raise ValueError(f"Score map must be 2D: {score.shape}")
    resized = cv2.resize(
        score,
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    top = (int(image_size) - int(crop_size)) // 2
    left = (int(image_size) - int(crop_size)) // 2
    cropped = resized[
        top:top + int(crop_size),
        left:left + int(crop_size),
    ]
    feature_height, feature_width = [int(value) for value in feature_shape]
    return cv2.resize(
        cropped,
        (feature_width, feature_height),
        interpolation=cv2.INTER_LINEAR,
    )


def patch_center_mask(
    mask: np.ndarray,
    feature_shape: Tuple[int, int],
) -> np.ndarray:
    """Return feature cells whose geometric centre is inside ``mask``.

    ``resize_mask_to_feature(..., INTER_NEAREST)`` answers which source pixel
    OpenCV samples for a cell.  That is not the same as testing the cell centre
    and is especially visible for thin annotations at a feature-grid
    boundary.  Patch-library selection uses this explicit centre rule.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D: {mask.shape}")
    feature_height, feature_width = [int(value) for value in feature_shape]
    if feature_height < 1 or feature_width < 1:
        raise ValueError(f"Invalid feature shape: {feature_shape}")
    height, width = mask.shape
    rows = np.floor(
        (np.arange(feature_height, dtype=np.float64) + 0.5)
        * float(height)
        / float(feature_height)
    ).astype(np.int64)
    cols = np.floor(
        (np.arange(feature_width, dtype=np.float64) + 0.5)
        * float(width)
        / float(feature_width)
    ).astype(np.int64)
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    return mask[rows[:, None], cols[None, :]]


def nearest_feature_cell(
    mask: np.ndarray,
    feature_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Return the ``(row, col)`` feature cell whose centre is nearest the mask.

    Used when no feature-cell centre lies inside a tiny region or a region
    that fell outside the CenterCrop: the mask centroid is matched against
    every feature-cell centre, and the closest cell is returned.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D: {mask.shape}")
    feature_height, feature_width = [int(value) for value in feature_shape]
    height, width = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cannot find a nearest feature cell for an empty mask")
    center_y = float(np.mean(ys))
    center_x = float(np.mean(xs))
    grid_y = (np.arange(feature_height, dtype=np.float64) + 0.5) * float(height) / float(feature_height)
    grid_x = (np.arange(feature_width, dtype=np.float64) + 0.5) * float(width) / float(feature_width)
    best_row, best_col = 0, 0
    best_distance = float("inf")
    for row in range(feature_height):
        delta_y = grid_y[row] - center_y
        for col in range(feature_width):
            delta_x = grid_x[col] - center_x
            distance = delta_y * delta_y + delta_x * delta_x
            if distance < best_distance:
                best_distance = distance
                best_row, best_col = row, col
    return int(best_row), int(best_col)


def patch_center_mask_with_fallback(
    mask: np.ndarray,
    feature_shape: Tuple[int, int],
) -> np.ndarray:
    """Feature cells with centre inside ``mask``; nearest cell as fallback.

    Tiny regions or regions outside the CenterCrop may have no feature-cell
    centre inside them.  Instead of dropping the ROI, the single feature cell
    nearest to the mask is used so the region still contributes a patch.
    """

    mask = np.asarray(mask, dtype=bool)
    cells = patch_center_mask(mask, feature_shape)
    if cells.any():
        return cells
    row, col = nearest_feature_cell(mask, feature_shape)
    cells = np.zeros(cells.shape, dtype=bool)
    cells[row, col] = True
    return cells


def linear_mask_to_feature(
    mask: np.ndarray,
    feature_shape: Tuple[int, int],
) -> np.ndarray:
    """Legacy linear mapping: region mask resized straight onto the feature grid.

    Mirrors dinomaly2_feature_bank_labelme_0807_add0.7ad.py (no Resize +
    CenterCrop geometry, just ``scale = feature / original``).
    """

    feature_height, feature_width = [int(value) for value in feature_shape]
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (feature_width, feature_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)


def linear_score_to_feature(
    score_map: np.ndarray,
    feature_shape: Tuple[int, int],
) -> np.ndarray:
    """Legacy linear mapping: score map resized straight onto the feature grid."""

    feature_height, feature_width = [int(value) for value in feature_shape]
    return cv2.resize(
        np.asarray(score_map, dtype=np.float32),
        (feature_width, feature_height),
        interpolation=cv2.INTER_LINEAR,
    )


def linear_patch_geometry(
    row: int,
    col: int,
    feature_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Legacy linear geometry for one feature cell (no crop offset)."""

    feature_height, feature_width = [int(value) for value in feature_shape]
    image_height, image_width = [int(value) for value in image_shape]
    scale_x = float(image_width) / float(feature_width)
    scale_y = float(image_height) / float(feature_height)
    bbox_original = [
        float(col) * scale_x,
        float(row) * scale_y,
        float(col + 1) * scale_x,
        float(row + 1) * scale_y,
    ]
    center_original = [
        (float(col) + 0.5) * scale_x,
        (float(row) + 0.5) * scale_y,
    ]
    return {
        "feature_shape": [feature_height, feature_width],
        "bbox_feature": [
            float(col),
            float(row),
            float(col + 1),
            float(row + 1),
        ],
        "center_feature": [float(col) + 0.5, float(row) + 0.5],
        # 线性映射无 Resize+CenterCrop 空间，resized 字段与原始空间一致，
        # 保持下游记录字段完整。
        "bbox_resized": bbox_original,
        "center_resized": center_original,
        "bbox_original": bbox_original,
        "center_original": center_original,
    }


def feature_patch_geometry(
    row: int,
    col: int,
    feature_shape: Tuple[int, int],
    image_shape: Tuple[int, int],
    image_size: int,
    crop_size: int,
) -> Dict[str, Any]:
    """Return the exact feature-cell geometry in resized and original space."""

    feature_height, feature_width = [int(value) for value in feature_shape]
    image_height, image_width = [int(value) for value in image_shape]
    crop_offset = (int(image_size) - int(crop_size)) / 2.0
    x1_resized = float(col) / float(feature_width) * float(crop_size) + crop_offset
    x2_resized = (
        float(col + 1) / float(feature_width) * float(crop_size) + crop_offset
    )
    y1_resized = float(row) / float(feature_height) * float(crop_size) + crop_offset
    y2_resized = (
        float(row + 1) / float(feature_height) * float(crop_size) + crop_offset
    )
    scale_x = float(image_width) / float(image_size)
    scale_y = float(image_height) / float(image_size)
    bbox_resized = [x1_resized, y1_resized, x2_resized, y2_resized]
    bbox_original = [
        x1_resized * scale_x,
        y1_resized * scale_y,
        x2_resized * scale_x,
        y2_resized * scale_y,
    ]
    center_feature = [float(col) + 0.5, float(row) + 0.5]
    center_resized = [
        (float(col) + 0.5) / float(feature_width) * float(crop_size)
        + crop_offset,
        (float(row) + 0.5) / float(feature_height) * float(crop_size)
        + crop_offset,
    ]
    center_original = [
        center_resized[0] * scale_x,
        center_resized[1] * scale_y,
    ]
    return {
        "feature_shape": [feature_height, feature_width],
        "bbox_feature": [
            float(col),
            float(row),
            float(col + 1),
            float(row + 1),
        ],
        "center_feature": center_feature,
        "bbox_resized": bbox_resized,
        "center_resized": center_resized,
        "bbox_original": bbox_original,
        "center_original": center_original,
    }


def resize_mask_to_feature(mask: np.ndarray, feature_shape: Tuple[int, int]) -> np.ndarray:
    height, width = [int(value) for value in feature_shape]
    mask = np.asarray(mask, dtype=np.uint8)
    resized = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)
    if resized.any() or not np.any(mask):
        return resized
    # A tiny non-empty ROI can disappear under nearest-neighbour
    # downsampling; fall back to projecting its bounding box so it covers
    # at least one feature cell.
    src_height, src_width = mask.shape[:2]
    bbox = mask_bbox(mask)
    if bbox is None:
        return resized
    x1, y1, x2, y2 = bbox
    fx1 = int(np.floor(x1 * width / src_width))
    fx2 = int(np.ceil(x2 * width / src_width))
    fy1 = int(np.floor(y1 * height / src_height))
    fy2 = int(np.ceil(y2 * height / src_height))
    fx1 = max(0, min(fx1, width - 1))
    fx2 = max(fx1 + 1, min(fx2, width))
    fy1 = max(0, min(fy1, height - 1))
    fy2 = max(fy1 + 1, min(fy2, height))
    resized[fy1:fy2, fx1:fx2] = True
    return resized


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def connected_components(mask: np.ndarray, min_area: int = 1, max_regions: int = 0) -> List[Dict[str, Any]]:
    if int(min_area) < 1:
        raise ValueError("min_area must be at least 1")
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components: List[Dict[str, Any]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        component_mask = labels == component_id
        bbox = mask_bbox(component_mask)
        if bbox is None:
            continue
        components.append(
            {
                "component_id": int(component_id),
                "area": area,
                "mask": component_mask,
                "bbox": bbox,
            }
        )
    components.sort(key=lambda component: (-component["area"], component["component_id"]))
    if int(max_regions) > 0:
        components = components[:int(max_regions)]
    return components


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if int(iterations) <= 0:
        return np.asarray(mask, dtype=bool)
    return cv2.dilate(
        np.asarray(mask, dtype=np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=int(iterations),
    ).astype(bool, copy=False)


def _load_image_tensor(image_path: Path, transform, device: torch.device):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original = np.asarray(image)
        image_tensor = transform(image).unsqueeze(0).to(device)
    return original, image_tensor


def infer_image(
    model,
    image_path: Path,
    transform,
    device: torch.device,
    gaussian_filter: Optional[torch.nn.Module] = None,
    patch_backbone=None,
    feature_source: str = "dinomaly",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(original-resolution score_map, encoder feature CHW)``.

    ``feature_source`` selects the representation used for the second stage:
    ``dinomaly`` (default) uses the last encoder feature group (``en[-1]``),
    while ``raw_patch`` uses the final patch-token output of a standalone
    DINOv2/DINOv3 backbone loaded by :func:`load_patch_backbone`.
    """

    original, image_tensor = _load_image_tensor(image_path, transform, device)
    with torch.no_grad():
        encoder_outputs, decoder_outputs = model(image_tensor)
        anomaly_map, _ = cal_anomaly_maps(
            encoder_outputs,
            decoder_outputs,
            original.shape[:2],
        )
        if gaussian_filter is None:
            gaussian_filter = get_gaussian_kernel(5, 4).to(device)
        anomaly_map = gaussian_filter(anomaly_map)
        score_map = anomaly_map[0, 0].detach().cpu().numpy()
        if feature_source == "raw_patch":
            feature = extract_raw_patch_feature(
                patch_backbone,
                image_path,
                transform,
                device,
            )
        else:
            # en[-1]：最后一层 encoder 特征图，建库/预测/查询三处一致。
            feature = encoder_outputs[-1][0]
    score_map = np.nan_to_num(
        np.asarray(score_map, dtype=np.float32),
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().numpy().astype(np.float32, copy=False)
    return score_map, np.nan_to_num(feature)


def extract_encoder_feature(
    model,
    image_path: Path,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Extract the last encoder feature group (``en[-1]``) for library building."""

    _original, image_tensor = _load_image_tensor(image_path, transform, device)
    with torch.no_grad():
        encoder_outputs, _decoder_outputs = model(image_tensor)
        feature = encoder_outputs[-1][0]
    return np.nan_to_num(
        feature.detach().cpu().numpy().astype(np.float32, copy=False)
    )


def extract_raw_patch_feature(
    backbone,
    image_path: Path,
    transform,
    device: torch.device,
) -> np.ndarray:
    """Extract the raw DINOv2/DINOv3 final patch-token map as CHW.

    ``backbone`` is a standalone pretrained DINOv2/DINOv3 model.  The final
    normed patch tokens are taken (``x_norm_patchtokens`` from
    ``forward_features``, or the last output of ``get_intermediate_layers``
    for the project's local dinov2 package) and reshaped into an NCHW feature
    map; the ROI mapping and masked ROIAlign afterwards are identical to the
    regular Dinomaly2 feature path.
    """

    _original, image_tensor = _load_image_tensor(image_path, transform, device)
    num_extra = int(
        getattr(backbone, "num_register_tokens", 0)
        or getattr(backbone, "n_storage_tokens", 0)
        or 0
    )
    with torch.no_grad():
        if callable(getattr(backbone, "forward_features", None)):
            output = backbone.forward_features(image_tensor)
            if isinstance(output, Mapping) and "x_norm_patchtokens" in output:
                patch = output["x_norm_patchtokens"]
            elif isinstance(output, Mapping) and "x_prenorm" in output:
                patch = output["x_prenorm"][:, num_extra + 1:, :]
            else:
                raise RuntimeError(
                    f"Backbone forward_features returned unsupported output: "
                    f"{type(output).__name__}"
                )
        elif callable(getattr(backbone, "get_intermediate_layers", None)):
            outputs = backbone.get_intermediate_layers(image_tensor, n=1)
            patch = outputs[-1][:, num_extra + 1:, :]
        else:
            raise RuntimeError(
                "Backbone exposes neither forward_features nor "
                "get_intermediate_layers"
            )
        side = int(round(float(patch.shape[1]) ** 0.5))
        if side * side != int(patch.shape[1]):
            raise ValueError(f"Non-square patch token grid: {patch.shape[1]}")
        feature = patch.permute(0, 2, 1).reshape(
            1, int(patch.shape[-1]), side, side
        )
    return np.nan_to_num(
        feature[0].detach().cpu().numpy().astype(np.float32, copy=False)
    )


def hub_backbone_name(backbone_name: str) -> str:
    """Map a Dinomaly2 backbone name to the local dinov2.hub.backbones entry.

    ``dinov2reg_vit_small_14`` -> ``dinov2_vits14_reg``,
    ``dinov2_vit_base_14`` -> ``dinov2_vitb14``.
    """

    parts = str(backbone_name).split("_")
    family = parts[0]
    if not family.startswith("dinov2"):
        raise ValueError(
            f"--backbone {backbone_name!r} is not a torch.hub DINOv2 model; "
            "use a dinov2/dinov2reg name such as 'dinov2_vitl14' or "
            "'dinov2reg_vit_small_14'"
        )
    if len(parts) < 3:
        raise ValueError(f"Cannot parse backbone name: {backbone_name!r}")
    arch = parts[-2]
    patch = parts[-1]
    letters = {"small": "s", "base": "b", "large": "l", "giant": "g"}
    if arch not in letters:
        raise ValueError(f"Unsupported backbone size {arch!r}: {backbone_name!r}")
    hub_name = f"dinov2_vit{letters[arch]}{patch}"
    if "reg" in family:
        hub_name += "_reg"
    return hub_name


def load_patch_backbone(args, device: torch.device):
    """Load the raw patch-token backbone, reusing ``--backbone``.

    torch.hub is deliberately avoided: the project ships its own ``dinov2``
    package that shadows the hub checkout (whose hubconf.py imports a
    ``cell_dino`` module missing from the cache).  DINOv2/DINOv2reg backbones
    are therefore loaded from the local ``dinov2.hub.backbones`` entries,
    and DINOv3 from the project's ``vit_encoder``.
    """

    name = args.backbone
    if name.startswith("dinov3"):
        from models import vit_encoder

        backbone = vit_encoder.load(name)
    else:
        from dinov2.hub.backbones import (
            dinov2_vitb14,
            dinov2_vitb14_reg,
            dinov2_vitg14,
            dinov2_vitg14_reg,
            dinov2_vitl14,
            dinov2_vitl14_reg,
            dinov2_vits14,
            dinov2_vits14_reg,
        )

        entry_points = {
            "dinov2_vits14": dinov2_vits14,
            "dinov2_vitb14": dinov2_vitb14,
            "dinov2_vitl14": dinov2_vitl14,
            "dinov2_vitg14": dinov2_vitg14,
            "dinov2_vits14_reg": dinov2_vits14_reg,
            "dinov2_vitb14_reg": dinov2_vitb14_reg,
            "dinov2_vitl14_reg": dinov2_vitl14_reg,
            "dinov2_vitg14_reg": dinov2_vitg14_reg,
        }
        entry = entry_points.get(hub_backbone_name(name))
        if entry is None:
            raise ValueError(f"Unsupported raw_patch backbone: {name!r}")
        backbone = entry(pretrained=True)
    backbone.to(device).eval()
    LOGGER.info("Loaded raw patch backbone: %s", name)
    return backbone


def roi_align_masked(
    feature_chw: np.ndarray,
    mask_feature: np.ndarray,
    output_size: int,
    device: torch.device,
) -> np.ndarray:
    """ROIAlign a feature map and average only the pixels covered by a mask."""

    feature = np.asarray(feature_chw, dtype=np.float32)
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"Expected CHW feature map, got {feature.shape}")
    mask = np.asarray(mask_feature, dtype=bool)
    if mask.shape != tuple(feature.shape[-2:]):
        raise ValueError(
            f"Feature/mask shape mismatch: {feature.shape[-2:]} vs {mask.shape}"
        )
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("Cannot ROIAlign an empty mask")
    channels, height, width = feature.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(x1, width - 1e-3))
    y1 = max(0.0, min(y1, height - 1e-3))
    x2 = max(x1 + 1e-3, min(x2, float(width)))
    y2 = max(y1 + 1e-3, min(y2, float(height)))
    boxes = torch.tensor(
        [[0.0, x1, y1, x2, y2]],
        dtype=torch.float32,
        device=device,
    )
    feature_tensor = torch.from_numpy(np.ascontiguousarray(feature)).unsqueeze(0).to(device)
    mask_tensor = (
        torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        pooled = roi_align(
            feature_tensor,
            boxes,
            output_size=(int(output_size), int(output_size)),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        pooled_mask = roi_align(
            mask_tensor,
            boxes,
            output_size=(int(output_size), int(output_size)),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        ).clamp_min(0.0)
        weight = pooled_mask.sum(dim=(2, 3), keepdim=True)
        pooled_mean = pooled.mean(dim=(2, 3), keepdim=True)
        pooled = torch.where(
            weight > 1e-6,
            (pooled * pooled_mask).sum(dim=(2, 3), keepdim=True) / weight,
            pooled_mean,
        )
    vector = pooled.reshape(1, channels).detach().cpu().numpy()[0]
    return np.nan_to_num(vector.astype(np.float32, copy=False))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector = vector / norm
    return vector.astype(np.float32, copy=False)


def _build_index_is_l2(args) -> bool:
    """True when the library index type is ``l2`` (normalised + Euclidean)."""

    return str(getattr(args, "index_type", "l2")).casefold() == "l2"


def _require_faiss():
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS is required for the good/anomaly feature libraries. "
            "Install the project's requirements (faiss-cpu) or faiss-gpu."
        ) from error
    return faiss


def _library_paths(path: Path) -> Tuple[Path, Path]:
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".faiss":
            raise ValueError(f"Library file must have .faiss suffix: {path}")
        return path, path.with_suffix(".json")
    if not path.is_dir():
        raise FileNotFoundError(f"Feature library does not exist: {path}")
    index_path = path / "index.faiss"
    if not index_path.is_file():
        candidates = sorted(path.glob("*.faiss"), key=lambda item: str(item).lower())
        if len(candidates) == 1:
            index_path = candidates[0]
    if not index_path.is_file():
        raise FileNotFoundError(f"No .faiss index found in {path}")
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        metadata_path = index_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No metadata JSON found for {index_path}")
    return index_path, metadata_path


def write_feature_library(
    output_dir: Path,
    vectors: np.ndarray,
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    faiss = _require_faiss()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise ValueError(f"Feature library vectors must be non-empty 2D: {vectors.shape}")
    index = faiss.IndexFlatL2(int(vectors.shape[1]))
    if str(metadata.get("index_type", "IndexFlatL2")).casefold() == "indexflatip":
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors))
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.json"
    faiss.write_index(index, str(index_path))
    np.save(output_dir / "vectors.npy", vectors)
    metadata = dict(metadata)
    metadata.update(
        {
            "format_version": 2,
            "index_type": str(metadata.get("index_type", "IndexFlatL2")),
            "distance_metric": (
                "1 - inner_product"
                if str(metadata.get("index_type", "IndexFlatL2")).casefold()
                == "indexflatip"
                else "L2 (Euclidean)"
            ),
            "feature_dim": int(vectors.shape[1]),
            "vector_count": int(vectors.shape[0]),
            "index_file": index_path.name,
            "vectors_file": "vectors.npy",
            "id_mapping_csv": "id_mapping.csv",
            "id_mapping_json": "id_mapping.json",
        }
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    write_id_mapping(output_dir, metadata)
    return index_path, metadata_path


def write_id_mapping(output_dir: Path, metadata: Mapping[str, Any]) -> None:
    """Write the FAISS-row -> image/ROI mapping beside a feature library."""

    output_dir = Path(output_dir)
    records = list(metadata.get("records", []))
    library_type = str(metadata.get("library_type", "unknown"))
    mapping_records = []
    for fallback_vector_id, record in enumerate(records):
        vector_id = int(record.get("vector_id", record.get("id", fallback_vector_id)))
        mapping_records.append(
            {
                "vector_id": vector_id,
                "image_id": str(record.get("image_id", "")),
                "roi_id": str(record.get("roi_id", "")),
                "library_type": library_type,
                "image_name": str(record.get("image_name", Path(record.get("image_path", "")).name)),
                "image_path": str(record.get("image_path", "")),
                "image_relative": str(record.get("image_relative", "")),
                "mask_path": str(record.get("mask_path", "")),
                "component_id": int(record.get("component_id", -1)),
                "area": int(record.get("area", 0)),
                "bbox_original": [float(value) for value in record.get("bbox_original", [])],
                "bbox_feature": [float(value) for value in record.get("bbox_feature", [])],
                "patch_index": int(record.get("patch_index", -1)),
                "patch_row": int(record.get("patch_row", -1)),
                "patch_col": int(record.get("patch_col", -1)),
                "patch_center_inside_mask": bool(
                    record.get("patch_center_inside_mask", False)
                ),
                "feature_shape": [
                    int(value) for value in record.get("feature_shape", [])
                ],
                "patch_center_feature": [
                    float(value)
                    for value in record.get("patch_center_feature", [])
                ],
                "patch_bbox_resized": [
                    float(value)
                    for value in record.get("patch_bbox_resized", [])
                ],
                "patch_center_resized": [
                    float(value)
                    for value in record.get("patch_center_resized", [])
                ],
                "patch_bbox_original": [
                    float(value)
                    for value in record.get("patch_bbox_original", [])
                ],
                "patch_center_original": [
                    float(value)
                    for value in record.get("patch_center_original", [])
                ],
            }
        )
    mapping_records.sort(key=lambda record: record["vector_id"])
    mapping_json = {
        "format_version": 1,
        "library_type": library_type,
        "records": mapping_records,
    }
    with (output_dir / "id_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(mapping_json, file, ensure_ascii=False, indent=2)

    fieldnames = [
        "vector_id",
        "image_id",
        "roi_id",
        "library_type",
        "image_name",
        "image_path",
        "image_relative",
        "mask_path",
        "component_id",
        "area",
        "bbox_original",
        "bbox_feature",
        "patch_index",
        "patch_row",
        "patch_col",
        "patch_center_inside_mask",
        "feature_shape",
        "patch_center_feature",
        "patch_bbox_resized",
        "patch_center_resized",
        "patch_bbox_original",
        "patch_center_original",
    ]
    with (output_dir / "id_mapping.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in mapping_records:
            row = dict(record)
            row["bbox_original"] = json.dumps(row["bbox_original"], ensure_ascii=False)
            row["bbox_feature"] = json.dumps(row["bbox_feature"], ensure_ascii=False)
            for key in (
                "feature_shape",
                "patch_center_feature",
                "patch_bbox_resized",
                "patch_center_resized",
                "patch_bbox_original",
                "patch_center_original",
            ):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def record_for_vector_id(
    metadata: Mapping[str, Any],
    vector_id: int,
) -> Dict[str, Any]:
    """Resolve one FAISS row to its image/ROI record."""

    records = metadata.get(
        "id_mapping_records",
        metadata.get("records", []),
    )
    vector_id = int(vector_id)
    for fallback_id, record in enumerate(records):
        record_id = int(record.get("vector_id", record.get("id", fallback_id)))
        if record_id == vector_id:
            resolved = dict(record)
            resolved["vector_id"] = record_id
            return resolved
    raise KeyError(
        f"FAISS vector_id={vector_id} has no image/ROI mapping. "
        "Rebuild the library with the current dinomaly_two_stage.py."
    )


def _uses_inner_product(library: SearchLibrary) -> bool:
    """True when the library uses ``IndexFlatIP`` (inner product)."""

    return (
        str(library.metadata.get("index_type", "")).casefold()
        in {"indexflatip", "ip"}
    )


def search_library_topk(
    library: SearchLibrary,
    vector: np.ndarray,
    top_k: int = 1,
) -> List[Tuple[float, int]]:
    """Search a library and return ``(distance, vector_id)`` pairs.

    For ``IndexFlatIP`` the reported "distance" is ``1 - inner_product``;
    for ``IndexFlatL2`` the raw Euclidean distance is returned.  Smaller is
    always more similar, keeping the distance semantics used by
    :func:`calculate_distance_offset`.
    """

    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    top_k = max(1, min(int(top_k), int(library.index.ntotal)))
    if vector.shape[1] != int(library.index.d):
        raise ValueError(
            f"Query feature dimension {vector.shape[1]} does not match "
            f"library dimension {library.index.d}."
        )
    raw_distances, neighbours = library.index.search(
        np.ascontiguousarray(vector),
        top_k,
    )
    if _uses_inner_product(library):
        return [
            (1.0 - float(raw_distance), int(neighbour))
            for raw_distance, neighbour in zip(raw_distances[0], neighbours[0])
            if int(neighbour) >= 0
        ]
    return [
        (float(raw_distance), int(neighbour))
        for raw_distance, neighbour in zip(raw_distances[0], neighbours[0])
        if int(neighbour) >= 0
    ]


def load_feature_library(
    path: Path,
    device: torch.device,
    faiss_on_gpu: bool = False,
) -> SearchLibrary:
    faiss = _require_faiss()
    index_path, metadata_path = _library_paths(path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    mapping_name = metadata.get("id_mapping_json", "id_mapping.json")
    mapping_path = metadata_path.parent / str(mapping_name)
    if mapping_path.is_file():
        with mapping_path.open("r", encoding="utf-8") as file:
            mapping = json.load(file)
        mapping_records = mapping.get("records", [])
        if mapping_records:
            # The external mapping is the source of truth for reverse lookup;
            # keep metadata.records for compatibility with older consumers.
            metadata["id_mapping_records"] = mapping_records
            metadata["id_mapping_path"] = str(mapping_path)
    index = faiss.read_index(str(index_path))
    feature_dim = int(metadata.get("feature_dim", index.d))
    if int(index.d) != feature_dim:
        raise ValueError(
            f"FAISS dimension {index.d} does not match metadata {feature_dim}: {metadata_path}"
        )
    if int(index.ntotal) < 1:
        raise ValueError(f"Feature library is empty: {index_path}")

    resources = None
    if faiss_on_gpu:
        if device.type != "cuda":
            raise ValueError("--faiss_on_gpu requires a CUDA device")
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("This FAISS installation has no GPU support")
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, device.index or 0, index)
    return SearchLibrary(index, metadata, index_path, metadata_path, resources)


def validate_library_compatibility(
    good_library: SearchLibrary,
    anomaly_library: SearchLibrary,
    args,
) -> None:
    good = good_library.metadata
    anomaly = anomaly_library.metadata
    for key in (
        "feature_dim",
        "feature_source",
        "roi_size",
        "normalize",
        "backbone",
        "library_mode",
        "feature_shape",
        "patch_selection_rule",
    ):
        good_value = good.get(key)
        anomaly_value = anomaly.get(key)
        if good_value != anomaly_value:
            raise ValueError(
                f"Good/anomaly library {key} differs: {good_value!r} vs {anomaly_value!r}"
            )
    expected = {
        "roi_size": int(args.roi_size),
        "image_size": int(args.image_size),
        "crop_size": int(args.crop_size),
    }
    if str(good.get("feature_source", "")) == "raw_patch":
        expected["backbone"] = args.backbone
        expected["feature_source"] = "raw_patch"
    else:
        expected["backbone"] = args.backbone
        expected["feature_merge"] = args.feature_merge
        expected["feature_source"] = "dinomaly_encoder_output"
    for key, value in expected.items():
        stored = good.get(key)
        # Libraries built by earlier patch-mode versions recorded the raw
        # CLI value "dinomaly" instead of "dinomaly_encoder_output".
        if key == "feature_source" and stored == "dinomaly":
            stored = "dinomaly_encoder_output"
        if stored is not None and stored != value:
            raise ValueError(
                f"Prediction {key}={value!r} does not match library metadata {stored!r}."
            )


def search_library(library: SearchLibrary, vector: np.ndarray) -> Tuple[float, int]:
    """Search one library; returns ``(distance, vector_id)``.

    Distance is the raw L2 norm for ``IndexFlatL2`` and
    ``1 - inner_product`` for ``IndexFlatIP``.
    """

    vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if vector.shape[1] != int(library.index.d):
        raise ValueError(
            f"Query feature dimension {vector.shape[1]} does not match "
            f"library dimension {library.index.d}."
        )
    raw_distances, neighbours = library.index.search(
        np.ascontiguousarray(vector),
        1,
    )
    distance = float(raw_distances[0, 0])
    if _uses_inner_product(library):
        distance = 1.0 - distance
    return distance, int(neighbours[0, 0])


def _model_feature_mask(
    original_mask: np.ndarray,
    feature_shape: Tuple[int, int],
    args,
) -> np.ndarray:
    model_mask = preprocess_mask(original_mask, args.image_size, args.crop_size)
    return resize_mask_to_feature(model_mask, feature_shape)


def _model_patch_center_mask(
    original_mask: np.ndarray,
    feature_shape: Tuple[int, int],
    args,
) -> np.ndarray:
    """Map an original mask to feature cells by the explicit centre rule.

    When the whole region falls outside the CenterCrop, the nearest feature
    cell is resolved in the original image space so the region still
    contributes a patch.
    """

    model_mask = preprocess_mask(original_mask, args.image_size, args.crop_size)
    if not model_mask.any():
        model_mask = np.asarray(original_mask, dtype=bool)
    return patch_center_mask_with_fallback(model_mask, feature_shape)


def select_patch_positions(
    score_map: np.ndarray,
    mask_feature: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Return the ``(row, col)`` positions of the highest-score patches inside a region.

    The score map is resampled onto the feature-map grid, then the
    ``ratio`` fraction of feature pixels with the highest scores inside
    ``mask_feature`` is selected (stable sort, descending).  Used by the
    ``patch`` library mode where every selected patch is stored/queried as
    its own vector instead of one pooled ROIAlign vector.
    """

    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ratio must be in (0, 1].")
    mask_feature = np.asarray(mask_feature, dtype=bool)
    score_map = np.asarray(score_map, dtype=np.float32)
    feature_height, feature_width = mask_feature.shape
    if score_map.shape[:2] != (feature_height, feature_width):
        score_map = cv2.resize(
            score_map,
            (feature_width, feature_height),
            interpolation=cv2.INTER_LINEAR,
        )
    coords = np.argwhere(mask_feature)
    if coords.shape[0] == 0:
        return coords
    scores = score_map[coords[:, 0], coords[:, 1]]
    count = max(1, int(round(coords.shape[0] * float(ratio))))
    order = np.argsort(-scores, kind="stable")[:count]
    return coords[order]


def _build_feature_library(
    args,
    images_root: Path,
    mask_root: Path,
    image_paths: Sequence[Path],
    library_type: str,
    output_dir: Path,
    mask_loader=None,
    model=None,
    transform=None,
    device: Optional[torch.device] = None,
    label_routing: Optional[Mapping[str, Any]] = None,
    patch_backbone=None,
) -> int:
    """Extract and write one library, optionally using a custom mask loader."""

    if device is None:
        device = select_device(args.gpu)
    if transform is None:
        transform = build_transform(args)
    if model is None:
        model = load_dinomaly_model(args, device)
    if args.feature_source == "raw_patch":
        if patch_backbone is None:
            patch_backbone = load_patch_backbone(args, device)
    library_mode = str(getattr(args, "library_mode", "roi"))
    if library_mode not in ("roi", "patch"):
        raise ValueError(f"Unsupported library_mode: {library_mode}")
    vectors: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []

    for image_path in tqdm(
        image_paths,
        desc=f"Build {library_type} feature library",
        unit="image",
        dynamic_ncols=True,
    ):
        mask_path = resolve_mask_path(image_path, images_root, mask_root)
        if mask_path is None:
            LOGGER.warning("No mask found for %s; skipping", image_path)
            continue
        try:
            with Image.open(image_path) as image:
                image_shape = (image.height, image.width)
            if mask_loader is None:
                mask = load_mask(mask_path, image_shape, args.mask_threshold)
            else:
                mask = mask_loader(mask_path, image_shape)
            components = connected_components(
                mask,
                min_area=args.min_area,
                max_regions=args.max_regions,
            )
            if not components:
                continue
            if library_mode == "patch":
                score_map, feature = infer_image(
                    model,
                    image_path,
                    transform,
                    device,
                    patch_backbone=patch_backbone,
                    feature_source=args.feature_source,
                )
            elif args.feature_source == "raw_patch":
                feature = extract_raw_patch_feature(
                    patch_backbone,
                    image_path,
                    transform,
                    device,
                )
            else:
                feature = extract_encoder_feature(
                    model,
                    image_path,
                    transform,
                    device,
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            LOGGER.warning("Skipping %s: %s", image_path, error)
            continue

        feature_shape = feature.shape[-2:]
        image_relative = _relative_image_path(image_path, images_root).as_posix()
        image_id = make_image_id(image_relative)
        for component in components:
            if library_mode == "patch":
                # 建库前对区域膨胀（good/anomaly 分开控制）：在原始图像
                # 空间按像素膨胀（每圈一次 3x3 结构元素），扩大一圈后再
                # 挑选前 x% 的 patch。
                dilation = int(
                    getattr(args, "good_dilation", 0)
                    if library_type == "good"
                    else getattr(args, "anomaly_dilation", 0)
                )
                region_mask = component["mask"]
                if dilation > 0:
                    region_mask = dilate_mask(region_mask, dilation)
                if not _build_index_is_l2(args):
                    # ip 模式：完全照旧脚本流程（dinomaly2_feature_bank_
                    # labelme_0807_add0.7ad.py）——区域 mask 线性缩放到
                    # 特征网格，不做 Resize+CenterCrop 几何。
                    mask_feature = linear_mask_to_feature(
                        region_mask,
                        feature_shape,
                    )
                    if not mask_feature.any():
                        # 旧脚本：区域太小无特征像素时取质心所在 cell。
                        row, col = nearest_feature_cell(
                            region_mask,
                            feature_shape,
                        )
                        mask_feature = np.zeros(feature_shape, dtype=bool)
                        mask_feature[row, col] = True
                else:
                    mask_feature = _model_patch_center_mask(
                        region_mask,
                        feature_shape,
                        args,
                    )
            else:
                mask_feature = _model_feature_mask(
                    component["mask"],
                    feature_shape,
                    args,
                )
            bbox_feature = mask_bbox(mask_feature)
            if bbox_feature is None:
                continue
            if library_mode == "patch":
                # 良品库/异常库分开控制入库比例：默认良品 100% 全量入库，
                # 异常库取区域内分数最高的前 anomaly_patch_ratio。
                patch_ratio = float(
                    getattr(args, "good_patch_ratio", 1.0)
                    if library_type == "good"
                    else getattr(args, "anomaly_patch_ratio", 0.5)
                )
                if not _build_index_is_l2(args):
                    # ip 模式：score map 直接线性缩放到特征网格（旧脚本式）。
                    score_feature = linear_score_to_feature(
                        score_map,
                        feature_shape,
                    )
                else:
                    score_feature = resize_score_map_to_feature(
                        score_map,
                        feature_shape,
                        args.image_size,
                        args.crop_size,
                    )
                positions = select_patch_positions(
                    score_feature,
                    mask_feature,
                    patch_ratio,
                )
                if positions.shape[0] == 0:
                    continue
                base_roi_id = make_roi_id(
                    image_id,
                    component["component_id"],
                )
                for patch_index, (row, col) in enumerate(positions):
                    # 特征来源 en[-1]；两种模式均 L2 归一化：
                    # l2 用 IndexFlatL2 欧氏距离，ip 用 IndexFlatIP
                    # （归一化后内积=余弦，距离=1-内积 ∈ [0,2]，照旧脚本）。
                    vector = feature[:, int(row), int(col)]
                    vector = l2_normalize(vector)
                    if not _build_index_is_l2(args):
                        # ip 模式记录旧脚本式线性几何（无 crop 偏移）。
                        patch_geometry = linear_patch_geometry(
                            int(row),
                            int(col),
                            feature_shape,
                            image_shape,
                        )
                    else:
                        patch_geometry = feature_patch_geometry(
                            int(row),
                            int(col),
                            feature_shape,
                            image_shape,
                            args.image_size,
                            args.crop_size,
                        )
                    patch_bbox_resized = patch_geometry.get(
                        "bbox_resized",
                        patch_geometry["bbox_original"],
                    )
                    patch_center_resized = patch_geometry.get(
                        "center_resized",
                        patch_geometry["center_original"],
                    )
                    vector_id = len(vectors)
                    vectors.append(vector)
                    records.append(
                        {
                            "vector_id": vector_id,
                            "id": vector_id,
                            "image_id": image_id,
                            "roi_id": f"{base_roi_id}_p{patch_index}",
                            "image_name": image_path.name,
                            "image_path": str(image_path.resolve()),
                            "image_relative": image_relative,
                            "mask_path": str(mask_path.resolve()),
                            "component_id": int(component["component_id"]),
                            "patch_index": int(patch_index),
                            "patch_row": int(row),
                            "patch_col": int(col),
                            "patch_center_inside_mask": True,
                            "feature_shape": patch_geometry["feature_shape"],
                            "patch_center_feature": patch_geometry["center_feature"],
                            "patch_bbox_resized": patch_bbox_resized,
                            "patch_center_resized": patch_center_resized,
                            "patch_bbox_original": patch_geometry["bbox_original"],
                            "patch_center_original": patch_geometry["center_original"],
                            "area": 1,
                            "bbox_original": [
                                float(value) for value in component["bbox"]
                            ],
                            "bbox_feature": [
                                float(col),
                                float(row),
                                float(col + 1),
                                float(row + 1),
                            ],
                        }
                    )
                continue
            vector = roi_align_masked(
                feature,
                mask_feature,
                args.roi_size,
                device,
            )
            vector = l2_normalize(vector)
            vector_id = len(vectors)
            vectors.append(vector)
            records.append(
                {
                    "vector_id": vector_id,
                    "id": vector_id,
                    "image_id": image_id,
                    "roi_id": make_roi_id(
                        image_id,
                        component["component_id"],
                    ),
                    "image_name": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "image_relative": image_relative,
                    "mask_path": str(mask_path.resolve()),
                    "component_id": int(component["component_id"]),
                    "area": int(component["area"]),
                    "bbox_original": [float(value) for value in component["bbox"]],
                    "bbox_feature": [float(value) for value in bbox_feature],
                }
            )

    if not vectors:
        raise RuntimeError(
            f"No valid ROI features were collected for the {library_type} library. "
            "Check --masks, mask geometry, and --min_area."
        )

    vectors_array = np.stack(vectors).astype(np.float32, copy=False)
    if library_mode == "patch":
        metadata = {
            "library_type": library_type,
            "library_mode": "patch",
            "patch_top_ratio": float(
                getattr(args, "good_patch_ratio", 1.0)
                if library_type == "good"
                else getattr(args, "anomaly_patch_ratio", 0.5)
            ),
            "good_patch_ratio": float(getattr(args, "good_patch_ratio", 1.0)),
            "anomaly_patch_ratio": float(
                getattr(args, "anomaly_patch_ratio", 0.5)
            ),
            "patch_selection_rule": (
                "top_ratio_by_score_among_feature_cells_whose_center_is_inside_mask"
            ),
            "patch_center_coordinate_system": (
                "preprocessed_mask_crop_feature_cell_center"
            ),
            "feature_source": (
                "raw_patch"
                if args.feature_source == "raw_patch"
                else "dinomaly_encoder_output"
            ),
            "feature_layout": (
                "per-patch en[-1] feature vectors; each patch is one vector"
            ),
            "roi_size": int(args.roi_size),
            "index_type": (
                "IndexFlatL2" if _build_index_is_l2(args) else "IndexFlatIP"
            ),
            "normalize": True,
            "distance_metric": (
                "L2 (Euclidean) on L2-normalised vectors"
                if _build_index_is_l2(args)
                else "1 - inner_product"
            ),
            "image_size": int(args.image_size),
            "crop_size": int(args.crop_size),
            "backbone": args.backbone,
            "model": str(Path(args.model).expanduser()),
            "feature_shape": [int(feature_shape[0]), int(feature_shape[1])],
            "region_dilation": int(
                getattr(args, "good_dilation", 0)
                if library_type == "good"
                else getattr(args, "anomaly_dilation", 0)
            ),
            "records": records,
        }
    elif args.feature_source == "raw_patch":
        metadata = {
            "library_type": library_type,
            "library_mode": "roi",
            "feature_source": "raw_patch",
            "feature_layout": "final normed patch tokens (x_norm_patchtokens) before ROIAlign",
            "roi_size": int(args.roi_size),
            "index_type": (
                "IndexFlatL2" if _build_index_is_l2(args) else "IndexFlatIP"
            ),
            "normalize": True,
            "distance_metric": (
                "L2 (Euclidean) on L2-normalised vectors"
                if _build_index_is_l2(args)
                else "1 - inner_product"
            ),
            "image_size": int(args.image_size),
            "crop_size": int(args.crop_size),
            "backbone": args.backbone,
            "model": args.backbone,
            "records": records,
        }
    else:
        metadata = {
            "library_type": library_type,
            "library_mode": "roi",
            "feature_source": "dinomaly_encoder_output",
            "feature_layout": "en[-1] CHW before ROIAlign",
            "roi_size": int(args.roi_size),
            "index_type": (
                "IndexFlatL2" if _build_index_is_l2(args) else "IndexFlatIP"
            ),
            "normalize": True,
            "distance_metric": (
                "L2 (Euclidean) on L2-normalised vectors"
                if _build_index_is_l2(args)
                else "1 - inner_product"
            ),
            "image_size": int(args.image_size),
            "crop_size": int(args.crop_size),
            "backbone": args.backbone,
            "model": str(Path(args.model).expanduser()),
            "records": records,
        }
    if label_routing is not None:
        metadata["label_routing"] = dict(label_routing)
    index_path, metadata_path = write_feature_library(
        Path(output_dir).expanduser(),
        vectors_array,
        metadata,
    )
    print(
        f"Built {library_type} library: {len(records)} ROI vectors -> "
        f"{index_path}",
        flush=True,
    )
    return 0


def build_library(args) -> int:
    """Build one library from a binary mask or an unfiltered Labelme mask."""

    images_root = Path(args.images).expanduser()
    mask_root = Path(args.masks).expanduser()
    image_paths = iter_image_paths(images_root)
    if not image_paths:
        raise RuntimeError(f"No images found under {images_root}")
    return _build_feature_library(
        args,
        images_root,
        mask_root,
        image_paths,
        args.library,
        Path(args.output_dir).expanduser(),
    )


def build_libraries_by_label(args) -> int:
    """Build good/anomaly libraries by routing Labelme shapes by ``label``."""

    images_root = Path(args.images).expanduser()
    mask_root = Path(args.masks).expanduser()
    image_paths = iter_image_paths(images_root)
    if not image_paths:
        raise RuntimeError(f"No images found under {images_root}")

    good_labels = _normalized_labels(args.good_labels)
    ignore_labels = _normalized_labels(args.ignore_labels)
    overlap = sorted(good_labels & ignore_labels)
    if overlap:
        raise ValueError(
            "A label cannot be both good and ignore: "
            + ", ".join(overlap)
        )
    if not good_labels:
        raise ValueError("--good_labels must contain at least one non-empty label")

    device = select_device(args.gpu)
    transform = build_transform(args)
    model = load_dinomaly_model(args, device)
    output_root = Path(args.output_dir).expanduser()
    label_routing = {
        "good_labels": sorted(good_labels),
        "ignore_labels": sorted(ignore_labels),
        "other_labels": "anomaly",
    }

    for library_type in ("good", "anomaly"):
        def mask_loader(mask_path, image_shape, target=library_type):
            return load_labelme_library_mask(
                mask_path,
                image_shape,
                target,
                good_labels,
                ignore_labels,
            )

        _build_feature_library(
            args,
            images_root,
            mask_root,
            image_paths,
            library_type,
            output_root / library_type,
            mask_loader=mask_loader,
            model=model,
            transform=transform,
            device=device,
            label_routing=label_routing,
        )
    return 0


def _output_relative_path(image_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    relative = _relative_image_path(image_path, input_root).with_suffix(suffix)
    result = Path(output_root) / relative
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def predict_images(args) -> int:
    input_path = Path(args.input).expanduser()
    image_paths = iter_image_paths(input_path)
    if not image_paths:
        raise RuntimeError(f"No images found under {input_path}")

    device = select_device(args.gpu)
    good_library = load_feature_library(
        Path(args.good_library).expanduser(),
        device,
        args.faiss_on_gpu,
    )
    anomaly_library = load_feature_library(
        Path(args.anomaly_library).expanduser(),
        device,
        args.faiss_on_gpu,
    )
    validate_library_compatibility(good_library, anomaly_library, args)
    model = load_dinomaly_model(args, device)
    patch_backbone = None
    if args.feature_source == "raw_patch":
        patch_backbone = load_patch_backbone(args, device)
    transform = build_transform(args)
    gaussian_filter = get_gaussian_kernel(5, 4).to(device)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = input_path if input_path.is_dir() else input_path.parent

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    roi_rows: List[Dict[str, Any]] = []
    for image_path in tqdm(
        image_paths,
        desc="Two-stage Dinomaly2 inference",
        unit="image",
        dynamic_ncols=True,
    ):
        score_path = _output_relative_path(
            image_path,
            input_root,
            output_dir / "score_maps",
            ".npy",
        )
        feature_path = _output_relative_path(
            image_path,
            input_root,
            output_dir / ("features_raw_patch" if args.feature_source == "raw_patch" else "features"),
            ".npy",
        )
        cached = (
            not args.recompute_features
            and score_path.is_file()
            and feature_path.is_file()
        )
        if cached:
            score_map = np.load(score_path)
            feature = np.load(feature_path)
        else:
            score_map, feature = infer_image(
                model,
                image_path,
                transform,
                device,
                gaussian_filter,
                patch_backbone=patch_backbone,
                feature_source=args.feature_source,
            )
            np.save(score_path, score_map)
            np.save(feature_path, feature)
        raw_score = float(np.max(score_map)) if score_map.size else 0.0
        stage1_label = classify_score(raw_score, args.score_threshold)
        regions: List[Dict[str, Any]] = []
        candidate_mask = np.zeros(score_map.shape, dtype=np.uint8)
        if stage1_label == "anomaly":
            components = connected_components(
                score_map > float(args.score_threshold),
                min_area=args.min_area,
                max_regions=args.max_regions,
            )
            feature_shape = feature.shape[-2:]
            for component in components:
                candidate_mask[component["mask"]] = 1
                query_mask = dilate_mask(component["mask"], args.roi_dilation)
                mask_feature = _model_feature_mask(query_mask, feature_shape, args)
                bbox_feature = mask_bbox(mask_feature)
                if bbox_feature is None:
                    mask_feature = dilate_mask(mask_feature, 1)
                    bbox_feature = mask_bbox(mask_feature)
                if bbox_feature is None:
                    continue
                vector = roi_align_masked(
                    feature,
                    mask_feature,
                    args.roi_size,
                    device,
                )
                if bool(good_library.metadata.get("normalize", True)):
                    vector = l2_normalize(vector)
                good_distance, good_neighbour = search_library(good_library, vector)
                anomaly_distance, anomaly_neighbour = search_library(
                    anomaly_library,
                    vector,
                )
                decision = calculate_distance_offset(
                    good_distance,
                    anomaly_distance,
                    args.offset_scale,
                    args.max_offset,
                    args.offset_eps,
                )
                regions.append(
                    {
                        "region_id": int(component["component_id"]),
                        "region_score": float(score_map[component["mask"]].max()),
                        "area": int(component["area"]),
                        "bbox_original": [float(value) for value in component["bbox"]],
                        "bbox_feature": [float(value) for value in bbox_feature],
                        "good_distance": float(good_distance),
                        "good_neighbour": int(good_neighbour),
                        "anomaly_distance": float(anomaly_distance),
                        "anomaly_neighbour": int(anomaly_neighbour),
                        **decision,
                    }
                )

        selected = select_strongest_region(regions)
        signed_offset = float(selected["signed_offset"]) if selected else 0.0
        adjusted_score = float(raw_score + signed_offset)
        final_label = classify_score(adjusted_score, args.score_threshold)
        relative = _relative_image_path(image_path, input_root)
        for region in regions:
            roi_rows.append(
                {
                    "image_path": str(image_path),
                    "image_relative": relative.as_posix(),
                    "raw_score": raw_score,
                    "score_threshold": float(args.score_threshold),
                    **region,
                }
            )
        region_path = _output_relative_path(
            image_path,
            input_root,
            output_dir / "candidate_regions",
            ".png",
        )
        detail_path = _output_relative_path(
            image_path,
            input_root,
            output_dir / "details",
            ".json",
        )
        if not cv2.imwrite(str(region_path), candidate_mask * 255):
            raise OSError(f"Cannot write candidate region mask: {region_path}")
        row = {
            "image_path": str(image_path),
            "image_relative": relative.as_posix(),
            "raw_score": raw_score,
            "adjusted_score": adjusted_score,
            "score_threshold": float(args.score_threshold),
            "stage1_label": stage1_label,
            "final_label": final_label,
            "stage2_applied": bool(stage1_label == "anomaly" and regions),
            "region_count": len(regions),
            "selected_region_id": selected.get("region_id", "") if selected else "",
            "good_distance": selected.get("good_distance", "") if selected else "",
            "anomaly_distance": selected.get("anomaly_distance", "") if selected else "",
            "similar_library": selected.get("similar_library", "") if selected else "",
            "confidence": selected.get("confidence", 0.0) if selected else 0.0,
            "offset": selected.get("offset", 0.0) if selected else 0.0,
            "signed_offset": signed_offset,
        }
        detail = {
            **row,
            "score_map_path": str(score_path),
            "candidate_region_path": str(region_path),
            "regions": regions,
        }
        with detail_path.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(detail), file, ensure_ascii=False, indent=2)
        rows.append(row)
        details.append(detail)
        LOGGER.info(
            "%s raw=%.6f (%s) adjusted=%.6f (%s) regions=%d library=%s offset=%+.6f",
            image_path,
            raw_score,
            stage1_label,
            adjusted_score,
            final_label,
            len(regions),
            selected.get("similar_library", "none") if selected else "none",
            signed_offset,
        )

    csv_path = output_dir / "results.csv"
    fieldnames = [
        "image_path",
        "image_relative",
        "raw_score",
        "adjusted_score",
        "score_threshold",
        "stage1_label",
        "final_label",
        "stage2_applied",
        "region_count",
        "selected_region_id",
        "good_distance",
        "anomaly_distance",
        "similar_library",
        "confidence",
        "offset",
        "signed_offset",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    roi_csv_path = output_dir / "roi_results.csv"
    roi_fieldnames = [
        "image_path",
        "image_relative",
        "raw_score",
        "score_threshold",
        "region_id",
        "region_score",
        "area",
        "bbox_original",
        "bbox_feature",
        "good_distance",
        "good_neighbour",
        "anomaly_distance",
        "anomaly_neighbour",
        "similar_library",
        "confidence",
        "offset",
        "signed_offset",
    ]
    with roi_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=roi_fieldnames)
        writer.writeheader()
        for roi_row in roi_rows:
            row_for_csv = dict(roi_row)
            row_for_csv["bbox_original"] = json.dumps(
                row_for_csv["bbox_original"],
                ensure_ascii=False,
            )
            row_for_csv["bbox_feature"] = json.dumps(
                row_for_csv["bbox_feature"],
                ensure_ascii=False,
            )
            writer.writerow(row_for_csv)
    with (output_dir / "run.json").open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(
                {
                    "score_threshold": args.score_threshold,
                    "offset_scale": args.offset_scale,
                    "max_offset": args.max_offset,
                    "roi_dilation": args.roi_dilation,
                    "min_area": args.min_area,
                    "feature_merge": args.feature_merge,
                    "roi_size": args.roi_size,
                    "results": details,
                }
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"Wrote two-stage results to {csv_path} and {roi_csv_path}",
        flush=True,
    )
    return 0


def add_model_arguments(parser: argparse.ArgumentParser, model_required: bool = True) -> None:
    parser.add_argument(
        "--model",
        required=model_required,
        default=None,
        help="Dinomaly2 model.pth checkpoint",
    )
    parser.add_argument(
        "--backbone",
        default="dinov2reg_vit_small_14",
        help=(
            "DINOv2 backbone used during training, and (with "
            "--feature_source raw_patch) loaded standalone for patch-token "
            "features. Available: dinov2_vit_small_14, dinov2_vit_base_14, "
            "dinov2_vit_large_14; dinov2reg_vit_small_14, "
            "dinov2reg_vit_base_14, dinov2reg_vit_large_14 (4 register "
            "tokens); dinov3_vit_small_16, dinov3_vit_base_16, "
            "dinov3_vit_large_16 (raw_patch loads dinov3 through the local "
            "vit_encoder, dinov2/dinov2reg through dinov2.hub.backbones); "
            "also dinov1 (dino_vit_small_8/16, dino_vit_base_8/16), "
            "tips_vit_small_14/base_14/large_14, beit_vit_base_16, "
            "deit_vit_small_16/base_16 (these are NOT usable with "
            "--feature_source raw_patch)"
        ),
    )
    parser.add_argument("--image_size", type=int, default=672)
    parser.add_argument("--crop_size", type=int, default=672)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--la", type=int, default=1)
    parser.add_argument("--lc", type=int, default=2)
    parser.add_argument("--cr", type=int, default=1)
    parser.add_argument(
        "--feature_merge",
        choices=("mean", "concat"),
        default="mean",
        help="Fuse the encoder feature groups by mean or channel concatenation",
    )
    parser.add_argument("--roi_size", type=int, default=7)
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument(
        "--feature_source",
        choices=("dinomaly", "raw_patch"),
        default="dinomaly",
        help=(
            "Second-stage feature representation: 'dinomaly' uses the "
            "Dinomaly2 encoder output; 'raw_patch' uses the final "
            "patch-token output (x_norm_patchtokens) of the same --backbone "
            "loaded standalone from the local dinov2/dinov3 packages"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dinomaly2 two-stage score correction with good/anomaly ROI libraries"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-library",
        help="Extract masked Dinomaly2 encoder ROI features and build one FAISS library",
    )
    add_model_arguments(build)
    build.add_argument("--images", "--input", dest="images", required=True)
    build.add_argument("--masks", required=True, help="Mask directory/file or Labelme JSON directory")
    build.add_argument(
        "--library",
        "--library_type",
        dest="library",
        choices=("good", "anomaly"),
        required=True,
    )
    build.add_argument("--output_dir", required=True)
    build.add_argument("--min_area", type=int, default=1)
    build.add_argument("--max_regions", type=int, default=0)
    build.add_argument("--mask_threshold", type=float, default=0.0)
    build.add_argument(
        "--library_mode",
        choices=("roi", "patch"),
        default="roi",
        help=(
            "roi = one pooled ROIAlign vector per annotated region "
            "(default); patch = one vector per highest-score patch inside "
            "each region (no ROIAlign)"
        ),
    )
    build.add_argument(
        "--index_type",
        choices=("l2", "ip"),
        default="l2",
        help=(
            "l2 = L2-normalise features and build an IndexFlatL2 index "
            "(Euclidean distance, default); ip = keep features unnormalised "
            "and build an IndexFlatIP index (distance = 1 - inner product)"
        ),
    )
    build.add_argument(
        "--good_patch_ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of feature patches stored per region in the good "
            "library (default: 1.0 = all patches stored)"
        ),
    )
    build.add_argument(
        "--anomaly_patch_ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of highest-score feature patches stored per region in "
            "the anomaly library (default: 0.5 = top 50%)"
        ),
    )
    build.add_argument(
        "--good_dilation",
        type=int,
        default=0,
        help=(
            "Dilate each annotated region in the original image space (one "
            "3x3 pixel iteration per unit) before selecting patches when "
            "building the good library (default: 0 = off)"
        ),
    )
    build.add_argument(
        "--anomaly_dilation",
        type=int,
        default=0,
        help=(
            "Dilate each annotated region in the original image space before "
            "selecting patches when building the anomaly library "
            "(default: 0 = off)"
        ),
    )
    build_by_label = subparsers.add_parser(
        "build-libraries",
        help="Build good/anomaly FAISS libraries by routing Labelme labels",
    )
    add_model_arguments(build_by_label)
    build_by_label.add_argument("--images", "--input", dest="images", required=True)
    build_by_label.add_argument(
        "--masks",
        required=True,
        help="Directory containing Labelme JSON files matched to the images",
    )
    build_by_label.add_argument(
        "--output_dir",
        required=True,
        help="Root directory; good/ and anomaly/ libraries are created below it",
    )
    build_by_label.add_argument(
        "--good_labels",
        nargs="+",
        default=["good"],
        help="Labels routed to the good library (default: good)",
    )
    build_by_label.add_argument(
        "--ignore_labels",
        nargs="+",
        default=["ignore"],
        help="Labels skipped during routing (default: ignore)",
    )
    build_by_label.add_argument("--min_area", type=int, default=1)
    build_by_label.add_argument("--max_regions", type=int, default=0)
    build_by_label.add_argument("--mask_threshold", type=float, default=0.0)
    build_by_label.add_argument(
        "--library_mode",
        choices=("roi", "patch"),
        default="roi",
        help=(
            "roi = one pooled ROIAlign vector per annotated region "
            "(default); patch = one vector per highest-score patch inside "
            "each region (no ROIAlign)"
        ),
    )
    build_by_label.add_argument(
        "--index_type",
        choices=("l2", "ip"),
        default="l2",
        help=(
            "l2 = L2-normalise features and build an IndexFlatL2 index "
            "(Euclidean distance, default); ip = keep features unnormalised "
            "and build an IndexFlatIP index (distance = 1 - inner product)"
        ),
    )
    build_by_label.add_argument(
        "--good_patch_ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of feature patches stored per region in the good "
            "library (default: 1.0 = all patches stored)"
        ),
    )
    build_by_label.add_argument(
        "--anomaly_patch_ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of highest-score feature patches stored per region in "
            "the anomaly library (default: 0.5 = top 50%)"
        ),
    )
    build_by_label.add_argument(
        "--good_dilation",
        type=int,
        default=0,
        help=(
            "Dilate each annotated region in the original image space (one "
            "3x3 pixel iteration per unit) before selecting patches when "
            "building the good library (default: 0 = off)"
        ),
    )
    build_by_label.add_argument(
        "--anomaly_dilation",
        type=int,
        default=0,
        help=(
            "Dilate each annotated region in the original image space before "
            "selecting patches when building the anomaly library "
            "(default: 0 = off)"
        ),
    )
    predict = subparsers.add_parser(
        "predict",
        help="Run Dinomaly2 stage 1 and good/anomaly-library stage 2",
    )
    add_model_arguments(predict)
    predict.add_argument("--input", required=True, help="One image or a recursive image directory")
    predict.add_argument("--good_library", required=True)
    predict.add_argument("--anomaly_library", required=True)
    predict.add_argument(
        "--score_threshold",
        "--threshold",
        dest="score_threshold",
        type=float,
        required=True,
        help="Stage-1 and final score threshold; scores strictly greater than it are anomaly",
    )
    predict.add_argument("--output_dir", required=True)
    predict.add_argument("--min_area", type=int, default=1)
    predict.add_argument("--max_regions", type=int, default=0)
    predict.add_argument(
        "--roi_dilation",
        type=int,
        default=0,
        help="Dilate each score-map component before encoder ROIAlign",
    )
    predict.add_argument(
        "--offset_scale",
        type=float,
        default=1.0,
        help="Maximum score correction generated by the normalized distance margin",
    )
    predict.add_argument(
        "--max_offset",
        type=float,
        default=None,
        help="Optional hard cap on the score correction",
    )
    predict.add_argument("--offset_eps", type=float, default=1e-8)
    predict.add_argument(
        "--recompute_features",
        action="store_true",
        help=(
            "Recompute and overwrite the cached score maps / second-stage "
            "features instead of reusing output_dir/score_maps and "
            "output_dir/features (or features_raw_patch)"
        ),
    )
    predict.add_argument(
        "--faiss_on_gpu",
        action="store_true",
        help="Move both FAISS indexes to the selected CUDA device",
    )
    return parser


def validate_args(args) -> None:
    if args.image_size < 1 or args.crop_size < 1:
        raise ValueError("image_size and crop_size must be positive")
    if args.crop_size > args.image_size:
        raise ValueError("crop_size cannot be greater than image_size")
    if args.roi_size < 1:
        raise ValueError("roi_size must be positive")
    if args.min_area < 1:
        raise ValueError("min_area must be at least 1")
    if args.max_regions < 0:
        raise ValueError("max_regions cannot be negative")
    if hasattr(args, "roi_dilation") and args.roi_dilation < 0:
        raise ValueError("roi_dilation cannot be negative")
    for key in ("good_dilation", "anomaly_dilation"):
        if hasattr(args, key) and getattr(args, key) < 0:
            raise ValueError(f"{key} cannot be negative")
    if hasattr(args, "offset_scale") and args.offset_scale < 0:
        raise ValueError("offset_scale cannot be negative")
    if hasattr(args, "max_offset") and args.max_offset is not None and args.max_offset < 0:
        raise ValueError("max_offset cannot be negative")
    if hasattr(args, "offset_eps") and args.offset_eps < 0:
        raise ValueError("offset_eps cannot be negative")
    if hasattr(args, "score_threshold") and not np.isfinite(args.score_threshold):
        raise ValueError("score_threshold must be finite")
    if hasattr(args, "library_mode") and args.library_mode == "patch":
        for key in ("good_patch_ratio", "anomaly_patch_ratio"):
            if not 0.0 < getattr(args, key) <= 1.0:
                raise ValueError(f"{key} must be in (0, 1]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.command == "build-library":
        return build_library(args)
    if args.command == "build-libraries":
        return build_libraries_by_label(args)
    if args.command == "predict":
        return predict_images(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
