"""Run PatchCore inference on one image or a directory of images.

The script accepts a directory produced by ``train.py`` (or a directory that
contains ``patchcore_params.pkl`` and the corresponding FAISS index), writes a
CSV with image-level scores, and stores raw anomaly maps as ``.npy`` files
together with heatmap/overlay images.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import patchcore.common  # noqa: E402
import patchcore.patchcore  # noqa: E402
from patchcore.datasets.custom import (  # noqa: E402
    ImageInferenceDataset,
    get_data_transforms,
)


LOGGER = logging.getLogger("patchcore.predict")
PARAMS_SUFFIX = "patchcore_params.pkl"
INDEX_SUFFIX = "nnscorer_search_index.faiss"
_RESAMPLING = getattr(Image, "Resampling", Image)


def select_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        if gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu} is not available; {torch.cuda.device_count()} device(s) found."
            )
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def _prefix_from_params(path: Path) -> str:
    if not path.name.endswith(PARAMS_SUFFIX):
        raise ValueError(f"Not a PatchCore parameter file: {path}")
    return path.name[: -len(PARAMS_SUFFIX)]


def _model_parameter_files(model_path: Path, category: Optional[str]) -> List[Path]:
    """Resolve a model directory, a training output directory, or one pkl."""

    if model_path.is_file():
        if model_path.name.endswith(PARAMS_SUFFIX):
            return [model_path]
        raise ValueError(
            "--model must point to a PatchCore model directory or patchcore_params.pkl."
        )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    direct = sorted(model_path.glob(f"*{PARAMS_SUFFIX}"))
    if direct:
        return direct

    candidates_root = model_path / "models"
    if candidates_root.is_dir():
        if category:
            candidates_root = candidates_root / category
        direct = sorted(candidates_root.glob(f"*{PARAMS_SUFFIX}"))
        if direct:
            return direct

    candidates = sorted(model_path.rglob(f"*{PARAMS_SUFFIX}"))
    if category:
        candidates = [
            path
            for path in candidates
            if category in {part for part in path.parent.parts}
        ]
    parent_dirs = sorted({path.parent for path in candidates})
    if len(parent_dirs) > 1:
        names = ", ".join(str(path) for path in parent_dirs)
        raise ValueError(
            "The model path contains multiple categories. Pass --category to select one: "
            + names
        )
    if not candidates:
        raise FileNotFoundError(
            f"No {PARAMS_SUFFIX} found below {model_path}."
        )
    return candidates


def load_models(
    model_path: str,
    device: torch.device,
    faiss_on_gpu: bool = False,
    faiss_num_workers: int = 4,
    category: Optional[str] = None,
):
    """Load one model or all ensemble members found in ``model_path``."""

    parameter_files = _model_parameter_files(Path(model_path).expanduser(), category)
    models = []
    for parameter_file in parameter_files:
        model = patchcore.patchcore.PatchCore(device)
        model.load_from_path(
            load_path=str(parameter_file.parent),
            device=device,
            prepend=_prefix_from_params(parameter_file),
            nn_method=patchcore.common.FaissNN(
                on_gpu=faiss_on_gpu and device.type == "cuda",
                num_workers=faiss_num_workers,
            ),
        )
        models.append(model)
        LOGGER.info("Loaded model from %s", parameter_file)
    return models


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.nanmin(values)
    high = np.nanmax(values)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def _jet_colormap(values: np.ndarray) -> Image.Image:
    """Create a dependency-light JET-like RGB heatmap."""

    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)), mode="RGB")


def predict_one(model, image_path: str, transform, device: torch.device):
    """Predict one image and return ``(score, raw_map, original_image)``."""

    original = Image.open(image_path).convert("RGB")
    image = transform(original).unsqueeze(0).to(device)
    with torch.no_grad():
        scores, masks = model.predict(image)
    return float(scores[0]), np.asarray(masks[0], dtype=np.float32), original


def _output_stem(image_path: Path, input_path: Path, index: int) -> str:
    if input_path.is_dir():
        try:
            relative = image_path.relative_to(input_path)
        except ValueError:
            relative = Path(image_path.name)
    else:
        relative = Path(image_path.name)
    stem = "__".join(relative.with_suffix("").parts)
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in stem)
    return f"{index:05d}_{safe or 'image'}"


def _aggregate_outputs(model_scores, model_masks):
    scores = np.stack([np.asarray(item, dtype=np.float32) for item in model_scores])
    masks = np.stack(
        [np.stack([np.asarray(mask, dtype=np.float32) for mask in item]) for item in model_masks]
    )
    if len(scores) == 1:
        return scores[0], masks[0]
    normalised_scores = np.stack([_normalise(item) for item in scores])
    normalised_masks = np.stack([_normalise(item) for item in masks])
    return normalised_scores.mean(axis=0), normalised_masks.mean(axis=0)


def predict(args):
    device = select_device(args.gpu)
    models = load_models(
        args.model,
        device,
        faiss_on_gpu=args.faiss_on_gpu,
        faiss_num_workers=args.faiss_num_workers,
        category=args.category,
    )
    input_height, input_width = models[0].input_shape[-2:]
    resize = args.resize or input_height
    imagesize = args.imagesize or input_height
    transform, _ = get_data_transforms(resize, imagesize)
    input_path = Path(args.input).expanduser()
    dataset = ImageInferenceDataset(input_path, transform, recursive=args.recursive)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Keep the output order identical for every ensemble member and for the CSV.
    model_scores = []
    model_masks = []
    for model_index, model in enumerate(models):
        scores = []
        masks = []
        for batch in dataloader:
            batch_scores, batch_masks = model.predict(batch["image"])
            scores.extend(float(score) for score in batch_scores)
            masks.extend(np.asarray(mask, dtype=np.float32) for mask in batch_masks)
        model_scores.append(scores)
        model_masks.append(masks)
        LOGGER.info("Predicted with model %d/%d", model_index + 1, len(models))

    scores, maps = _aggregate_outputs(model_scores, model_masks)
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "anomaly_score"])
        for index, (image_path, score, raw_map) in enumerate(
            zip(dataset.image_paths, scores, maps), start=1
        ):
            image_path = Path(image_path)
            stem = _output_stem(image_path, input_path, index - 1)
            raw_map = np.squeeze(np.asarray(raw_map, dtype=np.float32))
            np.save(output_dir / f"{stem}.npy", raw_map)

            display_map = (
                np.clip(raw_map / args.threshold, 0.0, 1.0)
                if args.threshold is not None and args.threshold > 0
                else _normalise(raw_map)
            )
            original = Image.open(image_path).convert("RGB")
            heatmap = _jet_colormap(display_map).resize(original.size, _RESAMPLING.BILINEAR)
            overlay = Image.blend(original, heatmap, alpha=0.5)
            heatmap.save(output_dir / f"{stem}_heatmap.png")
            overlay.save(output_dir / f"{stem}_overlay.png")
            if args.threshold is not None:
                binary = np.uint8(raw_map >= args.threshold) * 255
                Image.fromarray(binary, mode="L").resize(
                    original.size, _RESAMPLING.NEAREST
                ).save(output_dir / f"{stem}_mask.png")

            writer.writerow([str(image_path), float(score)])
            LOGGER.info(
                "[%d/%d] %s: score=%.6f",
                index,
                len(dataset),
                image_path.name,
                float(score),
            )
    LOGGER.info("Wrote predictions to %s", output_dir)
    return scores, maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict anomaly maps with PatchCore.")
    parser.add_argument("--model", required=True, help="Model directory or patchcore_params.pkl.")
    parser.add_argument("--input", required=True, help="Input image or image directory.")
    parser.add_argument("--output", default="./predictions")
    parser.add_argument("--category", default=None, help="Category when --model contains multiple categories.")
    parser.add_argument("--resize", "--image_size", dest="resize", type=int, default=None)
    parser.add_argument("--imagesize", "--crop_size", dest="imagesize", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", "--cuda", dest="gpu", type=int, default=0)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    predict(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


