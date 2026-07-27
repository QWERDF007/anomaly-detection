"""Dataset helpers for PatchCore-style custom image datasets.

The adapter intentionally follows the directory convention used by Dinomaly2
and MVTec AD while being a little more permissive for small, private datasets.
The most common supported layout is::

    dataset/
    |-- train/
    |   `-- good/
    |-- test/
    |   |-- good/
    |   `-- scratch/
    `-- ground_truth/
        `-- scratch/

``dataset`` may also contain one or more category directories.  In that case
``classname`` selects one category, or ``None`` makes the dataset enumerate all
categories.  Training samples are always treated as normal samples; anomaly
labels and masks are only used by the test split.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import PIL.Image
import torch
from torchvision import transforms

from .mvtec import DatasetSplit


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
NORMAL_NAMES = frozenset({"good", "normal", "normals", "ok", "positive"})
MASK_DIRECTORY_NAMES = frozenset({"ground_truth", "ground-truth", "masks", "mask"})


def _as_size(size: Union[int, Sequence[int]]) -> Union[int, Tuple[int, int]]:
    if isinstance(size, int):
        return size
    size = tuple(size)
    if len(size) != 2:
        raise ValueError("Image size must be an int or a two-element sequence.")
    return size[0], size[1]


def _as_hw(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    size = _as_size(size)
    if isinstance(size, int):
        return size, size
    return size


def get_data_transforms(
    size: Union[int, Sequence[int]],
    imagesize: Union[int, Sequence[int]],
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """Build the image and mask transforms used by the custom dataset.

    The order is the same as the original PatchCore/MVTec pipeline: resize,
    center crop, convert to tensor, and normalize the image only.
    """

    mean = list(IMAGENET_MEAN if mean is None else mean)
    std = list(IMAGENET_STD if std is None else std)
    resize = _as_size(size)
    crop = _as_size(imagesize)
    image_transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    mask_transform = transforms.Compose(
        [transforms.Resize(resize), transforms.CenterCrop(crop), transforms.ToTensor()]
    )
    return image_transform, mask_transform


def _iter_images(directory: Path, recursive: bool = True) -> List[Path]:
    if not directory.is_dir():
        return []
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def _split_value(split: Union[DatasetSplit, str]) -> DatasetSplit:
    if isinstance(split, DatasetSplit):
        return split
    try:
        return DatasetSplit(str(split).lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in DatasetSplit)
        raise ValueError(f"Unknown dataset split {split!r}; expected one of {valid}.") from exc


def _find_normal_directory(
    directory: Path, normal_names: Iterable[str] = NORMAL_NAMES
) -> Optional[Path]:
    if not directory.is_dir():
        return None
    for child in sorted(directory.iterdir()):
        if child.is_dir() and child.name.lower() in normal_names:
            return child
    return None


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


class CustomDataset(torch.utils.data.Dataset):
    """A directory-based image anomaly dataset.

    Args:
        source: Dataset root, or a directory containing category directories.
        classname: Optional category name.  If omitted and ``source`` itself
            has ``train``/``test`` directories, it is treated as one category;
            otherwise all category directories below ``source`` are used.
        resize: Initial resize passed to :class:`torchvision.transforms.Resize`.
        imagesize: Final center-crop size.  PatchCore expects a square input in
            its command line tools, but a two-element size is also accepted.
        split: ``DatasetSplit.TRAIN``, ``VAL`` or ``TEST`` (or the string form).
        train_val_split: Fraction of normal training images used for TRAIN.
            The remaining images are exposed by VAL.
        mask_dir: Optional root directory containing anomaly masks.  By default
            ``ground_truth`` below each category root is used.
        normal_names: Names of directories that represent normal images.
        recursive: Recursively discover images inside each split directory.
    """

    def __init__(
        self,
        source: Optional[Union[str, os.PathLike]] = None,
        classname: Optional[str] = None,
        resize: Union[int, Sequence[int]] = 256,
        imagesize: Union[int, Sequence[int]] = 224,
        split: Union[DatasetSplit, str] = DatasetSplit.TRAIN,
        train_val_split: float = 1.0,
        mask_dir: Optional[Union[str, os.PathLike]] = None,
        normal_names: Optional[Sequence[str]] = None,
        recursive: bool = True,
        **kwargs,
    ):
        super().__init__()
        root = kwargs.pop("root", None)
        if source is None:
            source = root
        if source is None:
            raise TypeError("CustomDataset requires a source or root directory.")
        del kwargs  # Keep compatibility with the MVTec command-line adapter.

        self.source = Path(source).expanduser()
        if not self.source.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.source}")

        self.split = _split_value(split)
        self.train_val_split = float(train_val_split)
        if not 0.0 < self.train_val_split <= 1.0:
            raise ValueError("train_val_split must be in the interval (0, 1].")

        self.normal_names = frozenset(
            name.lower() for name in (normal_names or NORMAL_NAMES)
        )
        self.recursive = recursive
        self.mask_dir = Path(mask_dir).expanduser() if mask_dir is not None else None
        self.resize = resize
        self.imagesize = (3, *_as_hw(imagesize))

        self.transform_mean = list(IMAGENET_MEAN)
        self.transform_std = list(IMAGENET_STD)
        self.transform_img, self.transform_mask = get_data_transforms(
            resize, imagesize, self.transform_mean, self.transform_std
        )

        self._category_roots = self._resolve_category_roots(classname)
        self.classnames_to_use = [name for name, _ in self._category_roots]
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

    def _resolve_category_roots(self, classname: Optional[str]):
        if classname is not None:
            candidate = self.source / classname
            if (candidate / "train").is_dir() or (candidate / "test").is_dir():
                return [(str(classname), candidate)]
            if (self.source / "train").is_dir() or (self.source / "test").is_dir():
                return [(str(classname), self.source)]
            raise FileNotFoundError(
                f"Could not find train/test directories for category {classname!r} "
                f"under {self.source}."
            )

        if (self.source / "train").is_dir() or (self.source / "test").is_dir():
            return [(self.source.name, self.source)]

        categories = [
            child
            for child in sorted(self.source.iterdir())
            if child.is_dir()
            and not child.name.startswith(".")
            and ((child / "train").is_dir() or (child / "test").is_dir())
        ]
        if categories:
            return [(child.name, child) for child in categories]
        raise FileNotFoundError(
            f"No train/test directory found below {self.source}. Expected a "
            "train/good and test/<anomaly> style dataset."
        )

    def _mask_root(self, category_root: Path, classname: str) -> Path:
        if self.mask_dir is None:
            return category_root / "ground_truth"
        if (self.mask_dir / classname).is_dir():
            return self.mask_dir / classname
        return self.mask_dir

    def _find_mask(self, mask_root: Path, anomaly: str, image_path: Path) -> Optional[str]:
        if not mask_root.is_dir():
            return None

        anomaly_root = mask_root / anomaly
        search_roots = [anomaly_root] if anomaly_root.is_dir() else [mask_root]
        stem = image_path.stem
        candidates = []
        for root in search_roots:
            for suffix in ("_mask", "-mask", ""):
                for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                    candidates.append(root / f"{stem}{suffix}{extension}")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        matches = []
        for root in search_roots:
            matches.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.stem in {stem, f"{stem}_mask", f"{stem}-mask"}
            )
        return str(sorted(matches)[0]) if matches else None

    def _normal_images(self, train_root: Path) -> List[Path]:
        normal_root = _find_normal_directory(train_root, self.normal_names)
        if normal_root is not None:
            return _iter_images(normal_root, self.recursive)

        # A flat train directory is a convenient custom-dataset shorthand.  If
        # it has no explicit normal folder, every image in it is a normal sample.
        return _iter_images(train_root, self.recursive)

    def _get_train_items(self, category: str, category_root: Path):
        train_root = category_root / "train"
        if not train_root.is_dir():
            raise FileNotFoundError(f"Training directory does not exist: {train_root}")
        images = self._normal_images(train_root)
        split_index = int(len(images) * self.train_val_split)
        if images:
            split_index = max(1, min(len(images), split_index))
        if self.split == DatasetSplit.TRAIN:
            images = images[:split_index]
        elif self.split == DatasetSplit.VAL:
            images = images[split_index:]
        return [
            [category, "good", str(path), None]
            for path in images
        ]

    def _get_test_items(self, category: str, category_root: Path):
        test_root = category_root / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(f"Test directory does not exist: {test_root}")

        mask_root = self._mask_root(category_root, category)
        class_directories = [
            child
            for child in sorted(test_root.iterdir())
            if child.is_dir()
            and not child.name.startswith(".")
            and child.name.lower() not in MASK_DIRECTORY_NAMES
        ]
        if not class_directories:
            return [[category, "good", str(path), None] for path in _iter_images(test_root, self.recursive)]

        items = []
        for anomaly_root in class_directories:
            anomaly = anomaly_root.name
            is_normal = anomaly.lower() in self.normal_names
            for image_path in _iter_images(anomaly_root, self.recursive):
                mask_path = (
                    None
                    if is_normal
                    else self._find_mask(mask_root, anomaly, image_path)
                )
                items.append(
                    [category, "good" if is_normal else anomaly, str(image_path), mask_path]
                )
        return items

    def get_image_data(self):
        """Return the same data structures as the original MVTec adapter."""

        imgpaths_per_class = {}
        data_to_iterate = []
        for category, category_root in self._category_roots:
            if self.split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
                items = self._get_train_items(category, category_root)
            else:
                items = self._get_test_items(category, category_root)
            imgpaths_per_class[category] = {}
            for item in items:
                _, anomaly, image_path, _ = item
                imgpaths_per_class[category].setdefault(anomaly, []).append(image_path)
            data_to_iterate.extend(items)
        if not data_to_iterate:
            split = self.split.value
            raise RuntimeError(f"No images found for split {split!r} in {self.source}.")
        return imgpaths_per_class, data_to_iterate

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, index):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[index]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask > 0).to(torch.float32)
        else:
            mask = torch.zeros((1, *image.shape[-2:]), dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": _safe_relative_path(Path(image_path), self.source),
            "image_path": image_path,
        }


class ImageInferenceDataset(torch.utils.data.Dataset):
    """Dataset used by ``predict.py`` for a single image or an image folder."""

    def __init__(
        self,
        source: Union[str, os.PathLike],
        transform,
        recursive: bool = True,
    ):
        self.source = Path(source).expanduser()
        if self.source.is_file():
            self.image_paths = [self.source]
        elif self.source.is_dir():
            self.image_paths = _iter_images(self.source, recursive=recursive)
        else:
            raise FileNotFoundError(f"Input image or directory does not exist: {self.source}")
        if not self.image_paths:
            raise RuntimeError(f"No supported images found in {self.source}.")
        self.transform_img = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return {
            "image": image,
            "image_path": str(image_path),
            "image_name": image_path.name,
            "mask": torch.zeros((1, *image.shape[-2:]), dtype=torch.float32),
            "is_anomaly": 0,
        }


# A short alias is convenient for callers that prefer the name used in
# torchvision and in the Dinomaly2 scripts.
CustomImageDataset = CustomDataset


__all__ = [
    "CustomDataset",
    "CustomImageDataset",
    "DatasetSplit",
    "ImageInferenceDataset",
    "get_data_transforms",
]




