"""Extract DINO patch-token features and save them as NCHW NumPy arrays.

The CLS token and register tokens are discarded. Only spatial patch tokens
are reshaped into a feature map and saved with a batch dimension:
[1, channels, height, width].
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import vit_encoder
from roi_feature_utils import IMAGE_EXTENSIONS, iter_image_paths, relative_posix


LOGGER = logging.getLogger("extract_dino_features")


class ImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_root: Path, transform):
        self.image_paths = list(image_paths)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            tensor = self.transform(image)
        relative = relative_posix(image_path, self.image_root)
        return tensor, str(image_path.resolve()), relative, width, height


def select_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        if gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu} is not available; {torch.cuda.device_count()} device(s) found."
            )
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def load_trained_encoder(encoder, model_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported Dinomaly checkpoint format: {model_path}")

    encoder_state = {
        key[len("encoder."):]: value
        for key, value in checkpoint.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(
            f"No encoder.* weights found in Dinomaly checkpoint: {model_path}"
        )
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    LOGGER.info(
        "Loaded DINO encoder from %s (missing=%d, unexpected=%d)",
        model_path,
        len(missing),
        len(unexpected),
    )


def extract_feature_map(
    encoder,
    images: torch.Tensor,
    layers: Sequence[int],
    feature_merge: str,
) -> torch.Tensor:
    layers = sorted(set(int(layer) for layer in layers))
    if not layers:
        raise ValueError("At least one DINO layer is required.")
    if not hasattr(encoder, "prepare_tokens") or not hasattr(encoder, "blocks"):
        raise RuntimeError(
            "This encoder does not expose Dinomaly2's prepare_tokens/blocks interface."
        )

    outputs: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        tokens = encoder.prepare_tokens(images)
        for index, block in enumerate(encoder.blocks):
            if index > layers[-1]:
                break
            tokens = block(tokens)
            if index in layers:
                outputs[index] = tokens

    register_tokens = int(getattr(encoder, "num_register_tokens", 0))
    feature_maps = []
    for layer in layers:
        if layer not in outputs:
            raise ValueError(
                f"Requested DINO layer {layer}, but the encoder has only "
                f"{len(encoder.blocks)} blocks."
            )
        # Token layout is [CLS, register tokens, patch tokens]. Keep only
        # patch tokens so the saved feature map is spatially aligned.
        layer_tokens = outputs[layer][:, 1 + register_tokens:, :]
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
        return torch.stack(feature_maps, dim=1).mean(dim=1)
    if feature_merge == "concat":
        return torch.cat(feature_maps, dim=1)
    raise ValueError(f"Unsupported feature merge mode: {feature_merge}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract spatial DINO feature maps from images."
    )
    parser.add_argument("--input", required=True, help="Image file or image directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for .npy features.")
    parser.add_argument("--model", default=None, help="Optional trained Dinomaly model.pth.")
    parser.add_argument("--backbone", default="dinov2reg_vit_small_14")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9],
        help="DINO block indices to extract.",
    )
    parser.add_argument("--feature_merge", choices=["mean", "concat"], default="mean")
    parser.add_argument("--image_size", type=int, default=672)
    parser.add_argument("--crop_size", type=int, default=672)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_image_paths(input_path, recursive=args.recursive)
    if not image_paths:
        raise RuntimeError(f"No supported images found in {input_path}.")
    image_root = input_path if input_path.is_dir() else input_path.parent

    device = select_device(args.gpu)
    weights_dir = (
        Path(args.weights_dir).expanduser()
        if args.weights_dir
        else Path(__file__).resolve().parent / "backbones" / "weights"
    )
    encoder = vit_encoder.load(args.backbone, WEIGHTS_DIR=str(weights_dir))
    if args.model:
        load_trained_encoder(encoder, Path(args.model).expanduser(), device)
    encoder = encoder.to(device).eval()

    transform = transforms.Compose(
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
    dataset = ImageDataset(image_paths, image_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    LOGGER.info("Extracting %d images on %s", len(dataset), device)
    with torch.no_grad():
        for images, image_paths_batch, relative_batch, widths, heights in loader:
            images = images.to(device, non_blocking=True)
            feature_maps = extract_feature_map(
                encoder, images, args.layers, args.feature_merge
            ).cpu().numpy().astype(np.float32)
            for index, (image_path, relative, width, height) in enumerate(
                zip(image_paths_batch, relative_batch, widths.tolist(), heights.tolist())
            ):
                feature_relative = Path(relative).with_suffix(".npy")
                feature_path = output_dir / feature_relative
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                # feature_maps is [B, C, H, W]. Keep N=1 when saving each
                # image so every file has an explicit NCHW layout.
                feature_nchw = feature_maps[index:index + 1]
                np.save(feature_path, feature_nchw)

    LOGGER.info("Saved %d NCHW patch-token feature maps to %s", len(dataset), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
