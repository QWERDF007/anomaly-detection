"""Dataset utilities for the one-pass Dinomaly2 mask-constraint training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
SKIP_DIRECTORY_NAMES = {"mask", "masks", "ground_truth", "annotations"}


def _find_child(directory: Path, name: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    name = name.lower()
    for child in directory.iterdir():
        if child.is_dir() and child.name.lower() == name:
            return child
    return None


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.stem.lower() not in {
                "mask",
                "ground_truth",
            }
            and not path.stem.lower().endswith(("_mask", "-mask"))
            and not any(
                part.lower() in SKIP_DIRECTORY_NAMES
                for part in path.relative_to(directory).parts[:-1]
            )
        ),
        key=lambda path: str(path).lower(),
    )


class MaskConstraintTrainDataset(Dataset):
    """Read ``Train/good`` and all non-good directories with optional masks.

    A missing mask is intentional and means that the whole image contributes
    only to the default Dinomaly2 loss. A present mask may contain BG,
    ``good_value``, ``anomaly_value`` and ``ignore_value``. Pixels with
    ``ignore_value`` are kept in the returned mask and excluded by the loss
    function; they are not silently converted to BG.
    """

    def __init__(
        self,
        root: str | Path,
        image_transform,
        image_size: int,
        crop_size: int,
        mask_dir: str | Path | None = None,
        good_value: int = 1,
        anomaly_value: int = 2,
        ignore_value: int = 255,
        joint_transform=None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.train_root = _find_child(self.root, "train")
        if self.train_root is None:
            raise FileNotFoundError(
                f"Cannot find Train/ directory under {self.root}."
            )
        self.image_transform = image_transform
        self.mask_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.mask_crop = transforms.CenterCrop(crop_size)
        # Training masks are isolated from evaluation ground-truth masks.
        # A missing mask_dir means exactly root/masks.
        self.mask_dir = (
            Path(mask_dir).expanduser()
            if mask_dir is not None
            else self.root / "masks"
        )
        self.good_value = int(good_value)
        self.anomaly_value = int(anomaly_value)
        self.ignore_value = int(ignore_value)
        self.joint_transform = joint_transform
        configured_values = {
            "good_value": self.good_value,
            "anomaly_value": self.anomaly_value,
            "ignore_value": self.ignore_value,
        }
        if any(value == 0 for value in configured_values.values()):
            raise ValueError(
                "good_value, anomaly_value and ignore_value must differ from 0."
            )
        if len(set(configured_values.values())) != len(configured_values):
            raise ValueError(
                "good_value, anomaly_value and ignore_value must be different."
            )

        self.samples = [
            {
                "image_path": image_path,
                "mask_path": self._find_mask(image_path),
            }
            for image_path in _iter_images(self.train_root)
        ]
        if not self.samples:
            raise RuntimeError(f"No training images found under {self.train_root}.")

    def _candidate_mask_paths(self, image_path: Path):
        relative = image_path.relative_to(self.train_root)
        relative_stem = relative.with_suffix("")
        # Only the configured training-mask root is searched. Evaluation
        # data_root/ground_truth must never be treated as a training mask.
        roots = [self.mask_dir]

        for root in roots:
            for extension in MASK_EXTENSIONS:
                yield root / relative.with_suffix(extension)
                yield root / self.train_root.name / relative.with_suffix(extension)
                yield root / "train" / relative.with_suffix(extension)
                yield root / "Train" / relative.with_suffix(extension)
                yield root / relative_stem.parent / (
                    f"{relative_stem.name}_mask{extension}"
                )
                yield root / relative_stem.parent / (
                    f"{relative_stem.name}-mask{extension}"
                )


    def _find_mask(self, image_path: Path) -> Optional[Path]:
        seen = set()
        for candidate in self._candidate_mask_paths(image_path):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load_mask(mask_path: Path) -> np.ndarray:
        if mask_path.suffix.lower() == ".npy":
            mask = np.asarray(np.load(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
        else:
            image = Image.open(mask_path).convert("L")
            mask = np.asarray(image)
        if mask.ndim != 2:
            raise ValueError(
                f"Mask must be 2D: {mask_path}; got shape {mask.shape}"
            )
        return np.rint(mask).astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")

        mask_path = item["mask_path"]
        mask_image = None
        if mask_path is not None:
            mask_array = self._load_mask(mask_path)
            mask_image = Image.fromarray(mask_array.astype(np.int32), mode="I")

        if self.joint_transform is not None:
            image_tensor, mask_image = self.joint_transform(image, mask_image)
        else:
            image_tensor = self.image_transform(image)

        if mask_image is None:
            mask = torch.zeros(
                (image_tensor.shape[-2], image_tensor.shape[-1]),
                dtype=torch.long,
            )
            has_mask = False
        else:
            if self.joint_transform is None:
                mask_image = self.mask_resize(mask_image)
                mask_image = self.mask_crop(mask_image)
            mask = torch.from_numpy(
                np.asarray(mask_image, dtype=np.int64).copy()
            ).long()
            valid_values = {
                0,
                self.good_value,
                self.anomaly_value,
                self.ignore_value,
            }
            actual_values = set(np.unique(mask.numpy()).tolist())
            invalid_values = actual_values - valid_values
            if invalid_values:
                raise ValueError(
                    f"Invalid values {sorted(invalid_values)} in mask {mask_path}. "
                    f"Expected only {sorted(valid_values)}."
                )
            has_mask = True

        return (
            image_tensor,
            mask,
            torch.tensor(has_mask, dtype=torch.bool),
            str(image_path),
        )
