"""Utilities shared by the DINO ROI feature/index scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from torchvision.ops import roi_align


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def iter_image_paths(source: Path, recursive: bool = True):
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {source}")
    iterator = source.rglob("*") if recursive else source.iterdir()
    return sorted(
        [
            path for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def relative_posix(path: Path, root: Optional[Path] = None) -> str:
    path = Path(path)
    if root is not None:
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
    return path.as_posix()


def load_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return image.width, image.height


def load_manifest(feature_dir: Path):
    manifest_path = Path(feature_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("records", payload) if isinstance(payload, dict) else payload


def feature_path_for_image(
    image_path: Path,
    feature_dir: Path,
    image_root: Optional[Path] = None,
    manifest=None,
) -> Optional[Path]:
    feature_dir = Path(feature_dir)
    image_path = Path(image_path)
    relative = relative_posix(image_path, image_root)

    if manifest:
        image_absolute = str(image_path.resolve())
        for record in manifest:
            record_relative = record.get("image_relative", record.get("relative_path", ""))
            record_image = record.get("image_path", "")
            if (
                record_relative.replace("\\", "/") == relative
                or record_image == str(image_path)
                or record_image == image_absolute
                or Path(record_image).name == image_path.name
            ):
                candidate = feature_dir / record["feature_path"]
                if candidate.is_file():
                    return candidate

    candidates = [
        feature_dir / Path(relative).with_suffix(".npy"),
        feature_dir / f"{image_path.stem}.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(feature_dir.rglob(f"{image_path.stem}.npy"))
    return matches[0] if matches else None


def annotation_path_for_image(
    image_path: Path,
    annotation_dir: Path,
    image_root: Optional[Path] = None,
) -> Optional[Path]:
    annotation_dir = Path(annotation_dir)
    image_path = Path(image_path)
    relative = Path(relative_posix(image_path, image_root))
    candidates = [
        annotation_dir / relative.with_suffix(".json"),
        annotation_dir / f"{image_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(annotation_dir.rglob(f"{image_path.stem}.json"))
    return matches[0] if matches else None


def load_labelme_annotation(annotation_path: Path) -> Dict:
    with Path(annotation_path).open("r", encoding="utf-8") as file:
        annotation = json.load(file)
    if "imageWidth" not in annotation or "imageHeight" not in annotation:
        raise ValueError(
            f"Labelme annotation has no imageWidth/imageHeight: {annotation_path}"
        )
    return annotation


def load_feature_map(feature_path: Path) -> np.ndarray:
    feature = np.asarray(np.load(feature_path), dtype=np.float32)
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(
            f"Expected a CHW feature map in {feature_path}, got shape {feature.shape}"
        )
    feature = np.nan_to_num(feature, copy=False)
    return feature


def polygon_to_feature_mask(
    points: Sequence[Sequence[float]],
    image_size: Tuple[int, int],
    feature_shape: Tuple[int, int],
) -> np.ndarray:
    image_width, image_height = image_size
    feature_height, feature_width = feature_shape
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("A polygon must contain at least three [x, y] points.")
    polygon = polygon.copy()
    polygon[:, 0] *= feature_width / max(float(image_width), 1.0)
    polygon[:, 1] *= feature_height / max(float(image_height), 1.0)
    polygon = np.round(polygon).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(feature_width - 1, 0))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(feature_height - 1, 0))
    mask = np.zeros((feature_height, feature_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def scale_bbox_to_feature(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    feature_shape: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    image_width, image_height = image_size
    feature_height, feature_width = feature_shape
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (
        x1 * feature_width / max(float(image_width), 1.0),
        y1 * feature_height / max(float(image_height), 1.0),
        x2 * feature_width / max(float(image_width), 1.0),
        y2 * feature_height / max(float(image_height), 1.0),
    )


def resize_mask_to_feature(mask: np.ndarray, feature_shape: Tuple[int, int]) -> np.ndarray:
    feature_height, feature_width = feature_shape
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (feature_width, feature_height),
        interpolation=cv2.INTER_NEAREST,
    )


def roi_align_vector(
    feature_chw: np.ndarray,
    bbox_feature: Sequence[float],
    mask_feature: Optional[np.ndarray] = None,
    output_size: int = 7,
) -> np.ndarray:
    """ROIAlign a feature-map region and return one pooled feature vector."""

    feature = load_feature_map_from_array(feature_chw)
    channels, feature_height, feature_width = feature.shape
    x1, y1, x2, y2 = [float(value) for value in bbox_feature]
    x1 = max(0.0, min(x1, feature_width - 1e-3))
    y1 = max(0.0, min(y1, feature_height - 1e-3))
    x2 = max(x1 + 1e-3, min(x2, float(feature_width)))
    y2 = max(y1 + 1e-3, min(y2, float(feature_height)))

    feature_tensor = torch.from_numpy(feature).unsqueeze(0)
    boxes = torch.tensor([[0.0, x1, y1, x2, y2]], dtype=torch.float32)
    pooled = roi_align(
        feature_tensor,
        boxes,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=True,
    )

    if mask_feature is not None:
        mask_tensor = torch.from_numpy(
            np.asarray(mask_feature, dtype=np.float32)
        ).unsqueeze(0).unsqueeze(0)
        pooled_mask = roi_align(
            mask_tensor,
            boxes,
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
    else:
        pooled = pooled.mean(dim=(2, 3), keepdim=True)

    vector = pooled.reshape(1, channels).numpy()[0]
    return np.asarray(vector, dtype=np.float32)


def load_feature_map_from_array(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 4 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f"Expected a CHW feature map, got shape {feature.shape}")
    return np.nan_to_num(feature, copy=False)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        vector = vector / norm
    return vector.astype(np.float32, copy=False)


def load_search_index(
    index_path: Path,
    on_gpu: bool = False,
    gpu_id: int = 0,
):
    index = faiss.read_index(str(index_path))
    resources = None
    if on_gpu:
        required = ("StandardGpuResources", "index_cpu_to_gpu")
        missing = [name for name in required if not hasattr(faiss, name)]
        if missing:
            raise RuntimeError(
                "GPU FAISS is not available. Missing: "
                + ", ".join(missing)
                + ". Install faiss-gpu or disable --faiss_on_gpu."
            )
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
    return index, resources


def save_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
