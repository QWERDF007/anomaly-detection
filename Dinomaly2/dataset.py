import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import tifffile as tiff
import cv2
from torchvision.transforms import functional as F
from pathlib import Path
from natsort import natsorted

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class MaskConstraintTrainTransform:
    """Training transform with mask-safe horizontal flipping.

    Horizontal flipping is applied to the image and its optional mask using
    the same random decision. ColorJitter is applied only to the image. No
    other geometric augmentation is performed here.
    """

    def __init__(
        self,
        size,
        isize,
        hflip_prob=0.5,
        brightness=0.2,
        contrast=0.2,
        hue=0.1,
        mean_train=None,
        std_train=None,
    ):
        if not 0.0 <= hflip_prob <= 1.0:
            raise ValueError("hflip_prob must be in [0, 1].")
        if brightness < 0 or contrast < 0:
            raise ValueError("brightness and contrast must be non-negative.")
        if not 0.0 <= hue <= 0.5:
            raise ValueError("hue must be in [0, 0.5].")

        mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
        std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
        self.hflip_prob = hflip_prob
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                hue=hue,
            ),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train, std=std_train),
        ])
        self.mask_resize = transforms.Resize(
            (size, size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.mask_crop = transforms.CenterCrop(isize)

    def __call__(self, image, mask=None):
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)

        image = self.image_transform(image)
        if mask is not None:
            mask = self.mask_resize(mask)
            mask = self.mask_crop(mask)
        return image, mask


def get_mask_constraint_train_transform(
    size,
    isize,
    hflip_prob=0.5,
    brightness=0.2,
    contrast=0.2,
    hue=0.1,
    mean_train=None,
    std_train=None,
):
    """Build the mask-aware transform used only for mask training images."""

    return MaskConstraintTrainTransform(
        size=size,
        isize=isize,
        hflip_prob=hflip_prob,
        brightness=brightness,
        contrast=contrast,
        hue=hue,
        mean_train=mean_train,
        std_train=std_train,
    )


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    scale = (isize / size) * (isize / size)
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(scale, scale), ratio=(0.95, 1.05)),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)])
    return data_transforms


class RandomRotate90(object):
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return F.rotate(img, angle)


class CustomDataset(torch.utils.data.Dataset):
    """Dataset adapter for one custom anomaly-detection dataset.

    Supports two modes:
    1. Directory layout:
        dataset/
        ├── train/
        │   └── good/                 # good is optional; flat train is OK
        ├── test/
        │   ├── good/
        │   └── scratch/
        └── ground_truth/
            └── scratch/

    2. Text file / list file (e.g. ``train.txt`` or ``--data_path train_list.txt``):
        Each line can be:
        - ``<image_path>`` (normal image for training or testing)
        - ``<image_path> <label_or_type_or_mask>`` (e.g. ``path/001.png 0`` or ``path/002.png crack`` or ``path/002.png mask.png``)
        - ``<image_path> <mask_path> <label_or_type>``

        Relative paths in the text file are resolved relative to the text file directory,
        or current working directory.
    """

    IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
    LIST_EXTENSIONS = {'.txt', '.list', '.csv', '.tsv'}
    NORMAL_NAMES = {'good', 'normal', 'normals', 'ok', 'positive'}

    def __init__(self, root, transform, gt_transform, phase, txt_file=None):
        if phase not in {'train', 'test'}:
            raise ValueError("phase must be 'train' or 'test'")

        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.cls_idx = 0

        target_path = Path(txt_file).expanduser() if txt_file else Path(root).expanduser()

        # Mode 1: root or txt_file is directly a list file (.txt, .list, etc.)
        if target_path.is_file() and (target_path.suffix.lower() in self.LIST_EXTENSIONS or txt_file is not None):
            self.root = target_path.parent if txt_file is None else Path(root).expanduser()
            file_to_load = target_path

            if txt_file is None and phase == 'test':
                stem_lower = target_path.stem.lower()
                if 'train' in stem_lower:
                    # Look for corresponding test list file in the same directory
                    test_name = target_path.name.replace('train', 'test').replace('Train', 'Test')
                    test_cand = target_path.parent / test_name
                    if test_cand.is_file():
                        file_to_load = test_cand
                    elif (target_path.parent / 'test.txt').is_file():
                        file_to_load = target_path.parent / 'test.txt'
                    elif (target_path.parent / 'test_list.txt').is_file():
                        file_to_load = target_path.parent / 'test_list.txt'
                    else:
                        raise FileNotFoundError(
                            f"Custom test split file list not found for {target_path}. "
                            f"(Checked {test_cand} and {target_path.parent / 'test.txt'})"
                        )

            self.img_path = file_to_load
            self.gt_path = None
            self.img_paths, self.gt_paths, self.labels, self.types = self._load_from_txt_file(file_to_load)
            return

        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f'Custom dataset directory or file does not exist: {self.root}')

        # Mode 2: root directory contains phase.txt (e.g. root/train.txt or root/test.txt)
        if self.root.is_dir():
            phase_txt = self.root / f"{phase}.txt"
            phase_list_txt = self.root / f"{phase}_list.txt"
            if phase_txt.is_file():
                self.img_path = phase_txt
                self.gt_path = None
                self.img_paths, self.gt_paths, self.labels, self.types = self._load_from_txt_file(phase_txt)
                return
            elif phase_list_txt.is_file():
                self.img_path = phase_list_txt
                self.gt_path = None
                self.img_paths, self.gt_paths, self.labels, self.types = self._load_from_txt_file(phase_list_txt)
                return

        # Mode 3: Standard directory layout (root/train and root/test)
        if not self.root.is_dir():
            raise FileNotFoundError(f'Custom dataset directory does not exist: {self.root}')

        self.img_path = next(
            (
                child
                for child in self.root.iterdir()
                if child.is_dir() and child.name.lower() == phase.lower()
            ),
            self.root / phase,
        )
        self.gt_path = next(
            (
                child
                for child in self.root.iterdir()
                if child.is_dir() and child.name.lower() == 'ground_truth'
            ),
            self.root / 'ground_truth',
        )

        if not self.img_path.is_dir():
            raise FileNotFoundError(
                f'Custom {phase} directory does not exist: {self.img_path}. '
                "Expected a dataset root containing 'train' and 'test', or a valid .txt image list."
            )

        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

    def _load_from_txt_file(self, file_path: Path):
        file_path = Path(file_path).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"File list does not exist: {file_path}")

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        base_dir = file_path.parent

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        for line in lines:
            if ',' in line and not os.path.exists(line):
                parts = [p.strip().strip('\'"') for p in line.split(',') if p.strip()]
            else:
                parts = [p.strip().strip('\'"') for p in line.split() if p.strip()]

            if not parts:
                continue

            raw_img = parts[0]
            img_p = Path(raw_img)
            if not img_p.is_absolute():
                if (base_dir / img_p).is_file():
                    img_p = (base_dir / img_p).resolve()
                elif (Path.cwd() / img_p).is_file():
                    img_p = (Path.cwd() / img_p).resolve()
                elif hasattr(self, 'root') and (self.root / img_p).is_file():
                    img_p = (self.root / img_p).resolve()
                else:
                    img_p = (base_dir / img_p).resolve()

            gt_p = None
            label = 0
            img_type = 'good'

            if len(parts) == 1:
                label = 0
                img_type = 'good'
                gt_p = None
            elif len(parts) == 2:
                second = parts[1]
                if second.isdigit() or second in {'0', '1'}:
                    label = int(second)
                    img_type = 'good' if label == 0 else 'anomaly'
                    gt_p = None
                elif second.lower() in self.NORMAL_NAMES:
                    label = 0
                    img_type = 'good'
                    gt_p = None
                elif any(second.lower().endswith(ext) for ext in self.IMAGE_EXTENSIONS):
                    mask_candidate = Path(second)
                    if not mask_candidate.is_absolute():
                        if (base_dir / mask_candidate).is_file():
                            mask_candidate = (base_dir / mask_candidate).resolve()
                        else:
                            mask_candidate = (Path.cwd() / mask_candidate).resolve()
                    gt_p = str(mask_candidate)
                    label = 1
                    img_type = 'anomaly'
                else:
                    img_type = second
                    label = 0 if img_type.lower() in self.NORMAL_NAMES else 1
                    gt_p = None
            else:
                p1, p2 = parts[1], parts[2]
                if any(p1.lower().endswith(ext) for ext in self.IMAGE_EXTENSIONS):
                    mask_str, type_or_label = p1, p2
                else:
                    type_or_label, mask_str = p1, p2

                mask_candidate = Path(mask_str)
                if not mask_candidate.is_absolute():
                    if (base_dir / mask_candidate).is_file():
                        mask_candidate = (base_dir / mask_candidate).resolve()
                    else:
                        mask_candidate = (Path.cwd() / mask_candidate).resolve()
                gt_p = str(mask_candidate)

                if type_or_label.isdigit():
                    label = int(type_or_label)
                    img_type = 'good' if label == 0 else 'anomaly'
                elif type_or_label.lower() in self.NORMAL_NAMES:
                    label = 0
                    img_type = 'good'
                else:
                    img_type = type_or_label
                    label = 1

            img_tot_paths.append(str(img_p))
            gt_tot_paths.append(gt_p)
            tot_labels.append(label)
            tot_types.append(img_type)

        if not img_tot_paths:
            raise RuntimeError(f"No valid image entries found in file list: {file_path}")

        return (
            np.asarray(img_tot_paths, dtype=object),
            np.asarray(gt_tot_paths, dtype=object),
            np.asarray(tot_labels, dtype=np.int64),
            np.asarray(tot_types, dtype=object),
        )

    def _load_from_txt(self):
        """从 txt 列表加载；兼容 train/test 语义与无 GT 的 4060 测试场景。"""
        lines = [line.strip() for line in self.root.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [Path(line).expanduser() for line in lines]
        # 过滤不存在路径（警告但不中断，PowerShell 中文路径常见编码问题）
        valid = []
        missing = []
        for p in paths:
            if p.is_file():
                valid.append(p)
            else:
                # 尝试 resolve 相对路径相对于 txt 目录
                alt = (self.root.parent / p).resolve()
                if alt.is_file():
                    valid.append(alt)
                else:
                    missing.append(str(p))
        if missing:
            print(f"[CustomDataset txt] Warning: {len(missing)} paths not found (first 3: {missing[:3]})")
        img_tot_paths = [str(p) for p in valid]
        # 按父目录名推断 label：OK/good/normal -> 0, 其他 -> 1 (仅 test 需要；train 统一 0)
        if self.phase == "train":
            tot_labels = [0] * len(img_tot_paths)
            tot_types = ["good"] * len(img_tot_paths)
            gt_tot_paths = [None] * len(img_tot_paths)
        else:
            tot_labels = []
            tot_types = []
            gt_tot_paths = []
            for p in valid:
                parent = p.parent.name.lower()
                # flat 列表中若来自 OK/good 视为 normal，否则 anomaly（无 mask 将返回全 0 mask）
                is_normal = parent in self.NORMAL_NAMES or "ok" in parent or "good" in parent
                # 更严格：若文件名含 NG/anomaly 也可判异常，但父目录最可靠
                if p.name.lower().startswith("picture") and "ng" not in str(p).lower():
                    # 对于铜色数据，NG 图片也在 NG 目录下，已按 parent 区分；此处保底
                    pass
                tot_labels.append(0 if is_normal else 1)
                tot_types.append("good" if is_normal else "anomaly")
                gt_tot_paths.append(None)
            # 若全部被判 normal 但实际含 NG，则尝试根据路径字符串含 NG/anomaly 修正
            # 已在上面按 parent 处理，通常已正确

        if not img_tot_paths:
            raise RuntimeError(f"No images found in txt list: {self.root} (checked {len(lines)} lines, {len(valid)} valid)")

        return (
            np.asarray(img_tot_paths, dtype=object),
            np.asarray(gt_tot_paths, dtype=object),
            np.asarray(tot_labels, dtype=np.int64),
            np.asarray(tot_types, dtype=object),
        )

    @classmethod
    def _iter_images(cls, directory):
        if not directory.is_dir():
            return []
        return sorted(
            [
                path for path in directory.rglob('*')
                if path.is_file() and path.suffix.lower() in cls.IMAGE_EXTENSIONS
            ],
            key=lambda path: str(path).lower(),
        )

    def _find_mask(self, defect_type, image_path, defect_root):
        """Find a mask by relative path, stem, or the common _mask name."""

        mask_root = self.gt_path / defect_type
        if not mask_root.is_dir():
            return None

        candidates = []
        try:
            relative_path = image_path.relative_to(defect_root)
            candidates.append(mask_root / relative_path)
        except ValueError:
            pass

        for suffix in ('', '_mask', '-mask'):
            for extension in self.IMAGE_EXTENSIONS:
                candidates.append(mask_root / f'{image_path.stem}{suffix}{extension}')

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        matches = [
            path for path in mask_root.rglob('*')
            if path.is_file()
            and path.suffix.lower() in self.IMAGE_EXTENSIONS
            and path.stem in {
                image_path.stem,
                f'{image_path.stem}_mask',
                f'{image_path.stem}-mask',
            }
        ]
        return str(sorted(matches, key=lambda path: str(path).lower())[0]) if matches else None

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        if self.phase == 'train':
            normal_dirs = [
                child for child in sorted(self.img_path.iterdir())
                if child.is_dir() and child.name.lower() in self.NORMAL_NAMES
            ]
            image_roots = normal_dirs if normal_dirs else [self.img_path]
            for image_root in image_roots:
                img_paths = self._iter_images(image_root)
                img_tot_paths.extend(str(path) for path in img_paths)
                gt_tot_paths.extend([None] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
        else:
            test_dirs = [
                child for child in sorted(self.img_path.iterdir())
                if child.is_dir() and child.name.lower() != 'ground_truth'
            ]

            # A flat test directory is interpreted as normal images.
            if not test_dirs:
                img_paths = self._iter_images(self.img_path)
                img_tot_paths.extend(str(path) for path in img_paths)
                gt_tot_paths.extend([None] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                for defect_root in test_dirs:
                    defect_type = defect_root.name
                    img_paths = self._iter_images(defect_root)
                    is_normal = defect_type.lower() in self.NORMAL_NAMES
                    img_tot_paths.extend(str(path) for path in img_paths)
                    tot_labels.extend([0 if is_normal else 1] * len(img_paths))
                    tot_types.extend(['good' if is_normal else defect_type] * len(img_paths))

                    if is_normal:
                        gt_tot_paths.extend([None] * len(img_paths))
                    else:
                        masks = [
                            self._find_mask(defect_type, path, defect_root)
                            for path in img_paths
                        ]
                        missing = [
                            path for path, mask in zip(img_paths, masks) if mask is None
                        ]
                        if missing:
                            examples = ', '.join(str(path) for path in missing[:3])
                            raise FileNotFoundError(
                                f'No ground-truth mask found for {len(missing)} custom '
                                f'anomaly image(s), for example: {examples}. Expected '
                                f'masks under {self.gt_path / defect_type}.'
                            )
                        gt_tot_paths.extend(masks)

        if not img_tot_paths:
            raise RuntimeError(
                f'No images found in custom {self.phase} split: {self.img_path}'
            )

        return (
            np.asarray(img_tot_paths, dtype=object),
            np.asarray(gt_tot_paths, dtype=object),
            np.asarray(tot_labels, dtype=np.int64),
            np.asarray(tot_types, dtype=object),
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, 0

        gt = self.gt_paths[idx]
        label = int(self.labels[idx])
        if label == 0 or gt is None or (isinstance(gt, float) and str(gt) == 'nan') or (isinstance(gt, str) and gt.lower() in {'none', '0', ''}):
            gt = torch.zeros((1, img.size(-2), img.size(-1)), dtype=torch.float32)
        else:
            try:
                gt = Image.open(gt).convert('L')
                gt = self.gt_transform(gt)
            except Exception:
                gt = torch.zeros((1, img.size(-2), img.size(-1)), dtype=torch.float32)

        assert img.size()[1:] == gt.size()[1:], 'image.size != gt.size !!!'
        return img, gt, label, img_path


class CustomRAMDataset(CustomDataset):
    """RAM-cached version of CustomDataset."""

    def __init__(self, root, transform, gt_transform, phase, txt_file=None):
        super().__init__(root, transform, gt_transform, phase, txt_file=txt_file)
        self.cached_items = [
            CustomDataset.__getitem__(self, idx) for idx in range(len(self))
        ]

    def __getitem__(self, idx):
        return self.cached_items[idx]


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png") + \
                           glob.glob(os.path.join(self.gt_path, defect_type) + "/*.bmp")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        print('img:')
        print(len(img_tot_paths))
        print()
        print("gt:")
        print(len(gt_tot_paths))
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = gt.convert('L')
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MVTecRAMDataset(MVTecDataset):
    """MVTecDataset that preloads all transformed images and GTs into RAM."""

    def __init__(self, root, transform, gt_transform, phase):
        super().__init__(root, transform, gt_transform, phase)
        self.cached_imgs = []
        self.cached_gts = []
        for idx in range(len(self.img_paths)):
            img, gt, _, _ = super().__getitem__(idx)
            self.cached_imgs.append(img)
            self.cached_gts.append(gt)

    def __getitem__(self, idx):
        return self.cached_imgs[idx], self.cached_gts[idx], self.labels[idx], self.img_paths[idx]


class MVTec3DDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, paired_transform=None, cat_3D=True, shot=None, seed=42):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')

        self.transform = transform
        self.gt_transform = gt_transform
        self.paired_transform = paired_transform
        self.phase = phase
        self.shot = shot
        self.seed = seed

        # load dataset
        self.img_paths, self.xyz_paths, self.gt_paths, self.labels, self.types = self.load_dataset()

        # Apply few-shot sampling for training phase
        if self.phase == 'train' and self.shot is not None:
            self._apply_few_shot_sampling()

        # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0
        self.cat_3D = cat_3D

    def _apply_few_shot_sampling(self):
        """Randomly sample 'shot' number of samples from training set"""
        np.random.seed(self.seed)

        total_samples = len(self.img_paths)

        if self.shot >= total_samples:
            print(f"Warning: shot ({self.shot}) >= total samples ({total_samples}), using all samples")
            return

        # Randomly select indices
        selected_indices = np.random.choice(total_samples, size=self.shot, replace=False)
        selected_indices = np.sort(selected_indices)  # Sort to maintain some order

        # Sample the data
        self.img_paths = self.img_paths[selected_indices]
        self.xyz_paths = self.xyz_paths[selected_indices]
        self.gt_paths = self.gt_paths[selected_indices]
        self.labels = self.labels[selected_indices]
        self.types = self.types[selected_indices]

        # print(f"Few-shot sampling: selected {self.shot} samples from training set (seed={self.seed})")
        # print(f"Label distribution: good={np.sum(self.labels == 0)}, anomaly={np.sum(self.labels == 1)}")

    def load_dataset(self):
        img_tot_paths = []
        xyz_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                img_tot_paths.extend(img_paths)
                xyz_paths = [path.replace('rgb', 'z').replace('png', 'tiff') for path in img_paths]
                xyz_tot_paths.extend(xyz_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                xyz_paths = [path.replace('rgb', 'z').replace('png', 'tiff') for path in img_paths]
                xyz_tot_paths.extend(xyz_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(xyz_tot_paths), np.array(gt_tot_paths), \
            np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, xyz_path, gt, label, img_type = self.img_paths[idx], self.xyz_paths[idx], self.gt_paths[idx], \
            self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        depth = tiff.imread(xyz_path)

        depth_mask = depth > 0.
        min_value = depth[depth_mask].min()
        max_value = depth.max()
        depth = (depth - min_value) / (max_value - min_value)
        depth = 1 - depth * 0.8
        depth[depth_mask == 0] = 0

        # depth = (np.repeat(np.expand_dims(depth, axis=2), 3, axis=2) * 255.0).astype(np.uint8)
        depth = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_HOT)  # convert to obvious heatmap
        depth = depth[:, :, ::-1]

        img = np.array(img)
        img[depth_mask == 0] = 0
        img = Image.fromarray(img.astype(np.uint8))
        depth = Image.fromarray(depth.astype(np.uint8))

        if self.paired_transform is not None:
            img, depth = self.paired_transform(img, depth)
        else:
            img = self.transform(img)
            depth = self.transform(depth)

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        if self.cat_3D:
            img = torch.cat([img, depth], dim=0)

        return img, gt, label, img_path

    def cvt2heatmap(self, gray):
        heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
        return heatmap


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, five_view_train=False):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.five_view_train = five_view_train

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels = [], [], []

        if phase == 'train':
            if self.five_view_train:
                data_set = class_json[phase]
                current_id = '0000'
                for i, sample in enumerate(data_set):
                    object_id = sample['image_path'].split('/')[:-1]
                    object_id = ''.join(object_id)
                    if object_id != current_id:
                        self.img_paths.append([])
                        self.labels.append([])
                        self.gt_paths.append([])
                        current_id = object_id
                    self.img_paths[-1].append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                    label = False
                    self.labels[-1].append(label)
            else:
                data_set = class_json[phase]
                for sample in data_set:
                    self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                    label = sample['anomaly_class'] != 'OK'
                    if label:
                        self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
                    else:
                        self.gt_paths.append(None)
                    self.labels.append(label)
        elif phase == 'test':
            data_set = class_json[phase]
            current_id = '0000'
            for i, sample in enumerate(data_set):
                object_id = sample['image_path'].split('/')[:-1]
                object_id = ''.join(object_id)
                if object_id != current_id:
                    self.img_paths.append([])
                    self.labels.append([])
                    self.gt_paths.append([])
                    current_id = object_id

                self.img_paths[-1].append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
                label = sample['anomaly_class'] != 'OK'
                if label:
                    self.gt_paths[-1].append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
                else:
                    self.gt_paths[-1].append(None)
                self.labels[-1].append(label)
        else:
            raise 'phase must be train or test'

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.phase == 'train':
            if self.five_view_train:
                img_paths = self.img_paths[idx]
                img_5view, gt_5view, label_5view = [], [], []
                for img_path in img_paths:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    img_5view.append(img)

                img_5view = torch.stack(img_5view, dim=0)
                label = torch.ones(5, dtype=torch.uint8)
                return img_5view, label

            else:
                img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)

                return img, label
        else:
            img_paths, gts, labels = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
            img_5view, gt_5view, label_5view = [], [], []
            for img_path, gt, label in zip(img_paths, gts, labels):
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                if label == 0:
                    gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
                else:
                    gt = Image.open(gt)
                    gt = self.gt_transform(gt)
                img_5view.append(img)
                gt_5view.append(gt)
                label_5view.append(label)

            img_5view = torch.stack(img_5view, dim=0)
            gt_5view = torch.stack(gt_5view, dim=0)
            label_5view = torch.tensor(label_5view)
            return img_5view, gt_5view, label_5view, list(img_paths)


class RealIADDatasetv2(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase, five_view_train=False,
                 split='realiad_jsons', version='realiad_1024'):
        self.img_path = os.path.join(root, version, category)
        if os.path.isdir(os.path.join(self.img_path, category)):
            self.img_path = os.path.join(self.img_path, category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        self.five_view_train = five_view_train
        self.version = version

        json_path = os.path.join(root, split, category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels = [], [], []

        if phase == 'train':
            if self.five_view_train:
                data_set = class_json[phase]
                # 使用字典收集同一物体的所有view
                object_dict = {}
                for i, sample in enumerate(data_set):
                    object_id = sample['image_path'].split('/')[:-1]
                    object_id = ''.join(object_id)

                    if object_id not in object_dict:
                        object_dict[object_id] = {
                            'img_paths': [],
                            'labels': [],
                            'gt_paths': []
                        }

                    object_dict[object_id]['img_paths'].append(
                        os.path.join(self.img_path, sample['image_path'])
                    )
                    label = False
                    object_dict[object_id]['labels'].append(label)

                # 将字典转换为列表
                for object_id in sorted(object_dict.keys()):  # sorted保证顺序一致
                    self.img_paths.append(object_dict[object_id]['img_paths'])
                    self.labels.append(object_dict[object_id]['labels'])
                    self.gt_paths.append(object_dict[object_id]['gt_paths'])
            else:
                data_set = class_json[phase]
                for sample in data_set:
                    self.img_paths.append(os.path.join(self.img_path, sample['image_path']))
                    label = sample['anomaly_class'] != 'OK'
                    if label:
                        self.gt_paths.append(os.path.join(self.img_path, sample['mask_path']))
                    else:
                        self.gt_paths.append(None)
                    self.labels.append(label)
        elif phase == 'test':
            data_set = class_json[phase]
            # 使用字典收集同一物体的所有view
            object_dict = {}
            for i, sample in enumerate(data_set):
                object_id = sample['image_path'].split('/')[:-1]
                object_id = ''.join(object_id)

                if object_id not in object_dict:
                    object_dict[object_id] = {
                        'img_paths': [],
                        'labels': [],
                        'gt_paths': []
                    }

                object_dict[object_id]['img_paths'].append(
                    os.path.join(self.img_path, sample['image_path'])
                )

                label = sample['anomaly_class'] != 'OK'
                if sample['mask_path'] is None:
                    label = False

                if label:
                    object_dict[object_id]['gt_paths'].append(
                        os.path.join(self.img_path, sample['mask_path'])
                    )
                else:
                    object_dict[object_id]['gt_paths'].append(None)

                object_dict[object_id]['labels'].append(label)

            # 将字典转换为列表
            for object_id in sorted(object_dict.keys()):  # sorted保证顺序一致
                self.img_paths.append(object_dict[object_id]['img_paths'])
                self.labels.append(object_dict[object_id]['labels'])
                self.gt_paths.append(object_dict[object_id]['gt_paths'])
        else:
            raise ValueError('phase must be train or test')

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.phase == 'train':
            if self.five_view_train:
                img_paths = self.img_paths[idx]
                img_5view, gt_5view, label_5view = [], [], []
                for img_path in img_paths:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                    img_5view.append(img)

                img_5view = torch.stack(img_5view, dim=0)
                label = torch.ones(5, dtype=torch.uint8)
                return img_5view, label

            else:
                img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)

                return img, label
        else:
            img_paths, gts, labels = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
            img_5view, gt_5view, label_5view = [], [], []
            for img_path, gt, label in zip(img_paths, gts, labels):
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                if label == 0:
                    gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
                else:
                    gt = Image.open(gt)
                    gt = self.gt_transform(gt)
                img_5view.append(img)
                gt_5view.append(gt)
                label_5view.append(label)

            img_5view = torch.stack(img_5view, dim=0)
            gt_5view = torch.stack(gt_5view, dim=0)
            label_5view = torch.tensor(label_5view)
            return img_5view, gt_5view, label_5view, list(img_paths)

class DroneAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        """
        Simplified Drone Anomaly Dataset for individual images

        Args:
            root: Path to specific category (e.g., './Drone-Anomaly/Vehicle Roundabout')
            transform: Image transforms
            phase: 'train' or 'test'
        """
        self.root = Path(root)
        self.transform = transform
        self.phase = phase

        # Load dataset
        self.img_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []

        # Find all sequence directories in the category root
        sequence_dirs = [d for d in self.root.iterdir() if d.is_dir() and d.name.startswith('sequence')]

        for seq_dir in sequence_dirs:
            phase_path = seq_dir / self.phase
            if not phase_path.exists():
                continue

            if self.phase == 'train':
                # Training: all images are normal (label = 0)
                video_dirs = [d for d in phase_path.iterdir() if d.is_dir()]
                for video_dir in video_dirs:
                    img_paths = glob.glob(str(video_dir / "*.jpg"))
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([0] * len(img_paths))  # All training images are normal

            else:  # test phase
                # Testing: load images and their corresponding labels from .npy files
                video_dirs = [d for d in phase_path.iterdir() if d.is_dir()]
                for video_dir in video_dirs:
                    # Get image paths
                    img_paths = natsorted(glob.glob(str(video_dir / "*.jpg")))

                    # Load corresponding labels from .npy file
                    label_file = phase_path / f"{video_dir.name}.npy"
                    if label_file.exists():
                        try:
                            labels = np.load(label_file)
                            # Make sure we have labels for all images
                            if len(labels) == len(img_paths):
                                img_tot_paths.extend(img_paths)
                                tot_labels.extend(labels.tolist())
                            else:
                                print(
                                    f"Warning: Label count mismatch for {video_dir}: {len(labels)} labels vs {len(img_paths)} images")
                        except Exception as e:
                            print(f"Error loading labels for {video_dir}: {e}")
                            # If no labels available, assume all are normal
                            img_tot_paths.extend(img_paths)
                            tot_labels.extend([0] * len(img_paths))
                    else:
                        # If no label file, assume all are normal
                        img_tot_paths.extend(img_paths)
                        tot_labels.extend([0] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, img_path


class MANTATinyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0
        self.phase = phase

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        img_5view = [img[:, :256], img[:, 1 * 256:2 * 256], img[:, 2 * 256:3 * 256], img[:, 3 * 256:4 * 256],
                     img[:, 4 * 256:]]

        if self.phase == 'train':
            random_view_idx = random.randint(0, 4)
            img = img_5view[random_view_idx]
            img = self.transform(Image.fromarray(img))
            gt = torch.zeros([5, 1, img.shape[1], img.shape[2]])
            label = 0
            return img, label

        else:
            img_5view = [self.transform(Image.fromarray(img)) for img in img_5view]

            if label == 0:
                gt = torch.zeros([5, 1, img_5view[0].shape[1], img_5view[0].shape[2]])
            else:
                gt = Image.open(gt)
                gt = np.array(gt)
                gt_5view = [gt[:, :256], gt[:, 1 * 256:2 * 256], gt[:, 2 * 256:3 * 256], gt[:, 3 * 256:4 * 256],
                            gt[:, 4 * 256:]]
                gt_5view = [self.gt_transform(Image.fromarray(gt)) for gt in gt_5view]
                gt = torch.stack(gt_5view, dim=0)

            img = torch.stack(img_5view, dim=0)
            label = gt.flatten(1).max(dim=1)[0].int()
            return img, gt, label, img_path



class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label
