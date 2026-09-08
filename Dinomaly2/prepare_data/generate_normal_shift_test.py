"""Generate a deterministic normal-only distribution-shift test set.

Example:
    python prepare_data/generate_normal_shift_test.py \
        --input ../my_dataset/test/good \
        --output ../my_dataset_normal_shift

The output follows ``CustomDataset`` layout: ``<output>/test/good``. A
``manifest.csv`` records the source and transform for every generated image.
These images are evaluation-only and must not be added to training data.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
Transform = Tuple[str, str, Callable[[Image.Image], Image.Image]]


def _translate(image: Image.Image, dx: int, dy: int) -> Image.Image:
    array = np.asarray(image.convert('RGB'))
    height, width = array.shape[:2]
    pad_x, pad_y = abs(dx), abs(dy)
    mode = 'reflect' if height > 1 and width > 1 else 'edge'
    padded = np.pad(
        array,
        ((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
        mode=mode,
    )
    start_y = pad_y - dy
    start_x = pad_x - dx
    shifted = padded[start_y:start_y + height, start_x:start_x + width]
    return Image.fromarray(shifted)


def _rotate(image: Image.Image, degrees: float) -> Image.Image:
    image = image.convert('RGB')
    array = np.asarray(image)
    height, width = array.shape[:2]
    pad = max(4, int(math.ceil(max(height, width) * 0.04)))
    mode = 'reflect' if height > 1 and width > 1 else 'edge'
    padded = np.pad(array, ((pad, pad), (pad, pad), (0, 0)), mode=mode)
    padded_image = Image.fromarray(padded)
    rotated = padded_image.rotate(
        degrees,
        resample=Image.Resampling.BICUBIC,
        expand=False,
    )
    return rotated.crop((pad, pad, pad + width, pad + height))


def _gamma(image: Image.Image, gamma: float) -> Image.Image:
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    corrected = np.power(array, gamma)
    return Image.fromarray(np.clip(corrected * 255.0, 0, 255).astype(np.uint8))


def _hue(image: Image.Image, degrees: float) -> Image.Image:
    hsv = np.array(image.convert('HSV'), dtype=np.uint8, copy=True)
    offset = int(round(degrees / 360.0 * 256.0))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + offset) % 256
    return Image.fromarray(hsv, mode='HSV').convert('RGB')


def build_transforms() -> List[Transform]:
    transforms: List[Transform] = []
    for pixels in (-8, -4, -2, 2, 4, 8):
        transforms.append(
            ('translation', f'x{pixels:+d}px', lambda image, value=pixels: _translate(image, value, 0))
        )
        transforms.append(
            ('translation', f'y{pixels:+d}px', lambda image, value=pixels: _translate(image, 0, value))
        )
    for degrees in (-2.0, -1.0, 1.0, 2.0):
        transforms.append(
            ('rotation', f'{degrees:+g}deg', lambda image, value=degrees: _rotate(image, value))
        )
    for factor in (0.90, 0.95, 1.05, 1.10):
        transforms.append(
            ('brightness', f'{factor:.2f}x', lambda image, value=factor: ImageEnhance.Brightness(image).enhance(value))
        )
        transforms.append(
            ('contrast', f'{factor:.2f}x', lambda image, value=factor: ImageEnhance.Contrast(image).enhance(value))
        )
    for gamma in (0.9, 1.1):
        transforms.append(('gamma', f'{gamma:.1f}', lambda image, value=gamma: _gamma(image, value)))
    transforms.append(('gaussian_blur', 'radius1.0', lambda image: image.filter(ImageFilter.GaussianBlur(1.0))))
    for degrees in (-5.0, 5.0):
        transforms.append(('hue', f'{degrees:+g}deg', lambda image, value=degrees: _hue(image, value)))
    for factor in (0.95, 1.05):
        transforms.append(
            ('saturation', f'{factor:.2f}x', lambda image, value=factor: ImageEnhance.Color(image).enhance(value))
        )
    return transforms


def iter_images(root: Path) -> Iterable[Path]:
    return sorted(
        (
            path for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def _safe_parameter_name(parameter: str) -> str:
    return (
        parameter.replace('+', 'p')
        .replace('-', 'm')
        .replace('.', '_')
        .replace('%', 'pct')
    )


def generate(input_root: Path, output_root: Path, overwrite: bool = False) -> int:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f'normal image directory does not exist: {input_root}')
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError('output must not be the input directory or a child of it')

    sources = list(iter_images(input_root))
    if not sources:
        raise RuntimeError(f'no supported images found under {input_root}')
    destination_root = output_root / 'test' / 'good'
    destination_root.mkdir(parents=True, exist_ok=True)

    rows = []
    generated = 0
    for source in sources:
        relative = source.relative_to(input_root)
        with Image.open(source) as opened:
            image = opened.convert('RGB')
            for shift_name, parameter, transform in build_transforms():
                suffix = f'__{shift_name}_{_safe_parameter_name(parameter)}.png'
                destination = destination_root / relative.parent / f'{relative.stem}{suffix}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not destination.exists():
                    transform(image.copy()).save(destination, format='PNG')
                    generated += 1
                rows.append(
                    {
                        'source': str(source),
                        'output': str(destination),
                        'shift': shift_name,
                        'parameter': parameter,
                    }
                )

    manifest_path = output_root / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=['source', 'output', 'shift', 'parameter'],
        )
        writer.writeheader()
        writer.writerows(rows)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate evaluation-only normal distribution shifts.'
    )
    parser.add_argument('--input', required=True, type=Path, help='Directory containing only normal images.')
    parser.add_argument('--output', required=True, type=Path, help='Output dataset root.')
    parser.add_argument('--overwrite', action='store_true', help='Replace already generated image files.')
    args = parser.parse_args()
    generated = generate(args.input, args.output, overwrite=args.overwrite)
    print(f'Generated {generated} images in {args.output.resolve()}')


if __name__ == '__main__':
    main()
