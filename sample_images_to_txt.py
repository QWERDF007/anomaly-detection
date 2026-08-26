#!/usr/bin/env python3
"""Script to scan a directory for images, randomly sample a specified count, and output paths to a .txt file.

Usage examples:
    # 1. Randomly sample 100 images from a directory (absolute paths)
    python sample_images_to_txt.py -i /data/dataset/train/good -o train_list.txt -n 100

    # 2. Sample 50 images with a specific random seed
    python sample_images_to_txt.py -i /data/dataset -o sample_50.txt -n 50 --seed 123

    # 3. Export all images (without sampling limit)
    python sample_images_to_txt.py -i /data/dataset/train/good -o all_train.txt

    # 4. Save relative paths and attach label 0
    python sample_images_to_txt.py -i ./images -o train.txt -n 200 --relative --label 0

    # 5. Split sampled images into train (80%) and test (20%) lists
    python sample_images_to_txt.py -i /data/dataset -o train_list.txt --split_ratio 0.8 -n 500
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
from typing import Iterable, List, Optional, Sequence, Union


DEFAULT_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def scan_images(
    directory: Union[str, Path],
    extensions: Optional[Iterable[str]] = None,
    recursive: bool = True,
    include_filter: Optional[str] = None,
    exclude_filter: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Scan a directory for image files matching extensions and filter criteria."""
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {dir_path}")

    ext_set = (
        {f".{e.lstrip('.').lower()}" for e in extensions}
        if extensions
        else DEFAULT_EXTENSIONS
    )

    iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()
    image_paths: List[Path] = []

    exclude_list = [e.lower() for e in (exclude_filter or []) if e]

    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in ext_set:
            continue
        # Skip hidden files or files in hidden folders
        try:
            rel = path.relative_to(dir_path)
            if any(part.startswith(".") for part in rel.parts):
                continue
        except ValueError:
            pass

        path_str = str(path)
        path_str_lower = path_str.lower()

        # Check exclude keywords
        if any(exc in path_str_lower for exc in exclude_list):
            continue

        # Check include filter
        if include_filter and include_filter.lower() not in path_str_lower:
            continue

        image_paths.append(path)

    image_paths.sort(key=lambda p: str(p).lower())
    return image_paths


def sample_images(
    image_paths: Sequence[Path],
    num_samples: Optional[int] = None,
    seed: int = 42,
    sort_output: bool = True,
) -> List[Path]:
    """Randomly sample images using a fixed random seed."""
    total = len(image_paths)
    if not image_paths:
        return []

    if num_samples is None or num_samples <= 0 or num_samples >= total:
        selected = list(image_paths)
    else:
        rng = random.Random(seed)
        selected = rng.sample(list(image_paths), num_samples)

    if sort_output:
        selected.sort(key=lambda p: str(p).lower())

    return selected


def write_image_list_file(
    image_paths: Sequence[Path],
    output_txt: Union[str, Path],
    relative: bool = False,
    relative_to: Optional[Union[str, Path]] = None,
    label: Optional[Union[str, int]] = None,
) -> Path:
    """Write image paths to a .txt file."""
    out_path = Path(output_txt).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir = (
        Path(relative_to).expanduser().resolve()
        if relative_to is not None
        else out_path.parent
    )

    lines = []
    for path in image_paths:
        if relative:
            try:
                line_path = os.path.relpath(str(path), str(base_dir))
            except ValueError:
                line_path = str(path.resolve())
        else:
            line_path = str(path.resolve())

        if label is not None:
            lines.append(f"{line_path} {label}")
        else:
            lines.append(line_path)

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan directory for images, randomly sample a specified count, and output paths to a .txt file."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        "--dir",
        type=str,
        required=True,
        help="Input directory containing images (e.g. /data/dataset/train/good).",
    )
    parser.add_argument(
        "-o",
        "--output_txt",
        "--out",
        type=str,
        required=True,
        help="Output .txt file path (e.g. train_list.txt).",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        "--count",
        "--num",
        type=int,
        default=None,
        help="Number of images to randomly sample. Omit or 0 to select all images.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subdirectories recursively.",
    )
    parser.add_argument(
        "-e",
        "--ext",
        "--extensions",
        type=str,
        default=None,
        help="Comma-separated image extensions (e.g. 'jpg,png,bmp'). Default: jpg,jpeg,png,bmp,tif,tiff,webp.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Save relative paths instead of absolute paths.",
    )
    parser.add_argument(
        "--relative_to",
        type=str,
        default=None,
        help="Base directory for relative paths (defaults to directory of output_txt).",
    )
    parser.add_argument(
        "--filter",
        "--include",
        type=str,
        default=None,
        help="Only include image paths containing this keyword.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Exclude image paths containing these comma-separated keywords (e.g. 'mask,ground_truth').",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label to append to each line (e.g. 0, good, 1).",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=None,
        help="If set (e.g. 0.8), splits the sampled images into train and test lists.",
    )
    parser.add_argument(
        "--output_test",
        type=str,
        default=None,
        help="Output path for test split when --split_ratio is set (default: replaces 'train' with 'test').",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort output paths alphabetically.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    extensions = (
        [ext.strip() for ext in args.ext.split(",") if ext.strip()]
        if args.ext
        else None
    )
    exclude_filter = (
        [e.strip() for e in args.exclude.split(",") if e.strip()]
        if args.exclude
        else None
    )

    print(f"[*] Scanning directory: {args.input_dir}")
    all_images = scan_images(
        directory=args.input_dir,
        extensions=extensions,
        recursive=not args.no_recursive,
        include_filter=args.filter,
        exclude_filter=exclude_filter,
    )
    total_found = len(all_images)
    print(f"[*] Found {total_found} image(s) matching criteria.")

    if total_found == 0:
        print("[!] Warning: No images found matching the criteria. Output file not created.")
        sys.exit(1)

    target_num = args.num_samples if (args.num_samples and args.num_samples > 0) else total_found
    target_num = min(target_num, total_found)

    sampled = sample_images(
        image_paths=all_images,
        num_samples=target_num,
        seed=args.seed,
        sort_output=not args.no_sort,
    )
    print(f"[*] Randomly selected {len(sampled)} / {total_found} image(s) (seed={args.seed}).")

    if args.split_ratio is not None and 0.0 < args.split_ratio < 1.0:
        # Split into train and test
        rng = random.Random(args.seed)
        shuffled = list(sampled)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * args.split_ratio)
        split_idx = max(1, min(len(shuffled) - 1, split_idx))

        train_set = shuffled[:split_idx]
        test_set = shuffled[split_idx:]
        if not args.no_sort:
            train_set.sort(key=lambda p: str(p).lower())
            test_set.sort(key=lambda p: str(p).lower())

        train_out = write_image_list_file(
            train_set,
            args.output_txt,
            relative=args.relative,
            relative_to=args.relative_to,
            label=args.label,
        )
        print(f"[+] Train split ({len(train_set)} images) saved to: {train_out}")

        test_out_path = args.output_test
        if not test_out_path:
            out_p = Path(args.output_txt)
            if "train" in out_p.name.lower():
                test_name = out_p.name.replace("train", "test").replace("Train", "Test")
            else:
                test_name = f"test_{out_p.name}"
            test_out_path = out_p.parent / test_name

        test_out = write_image_list_file(
            test_set,
            test_out_path,
            relative=args.relative,
            relative_to=args.relative_to,
            label=args.label,
        )
        print(f"[+] Test split ({len(test_set)} images) saved to: {test_out}")
    else:
        out_file = write_image_list_file(
            sampled,
            args.output_txt,
            relative=args.relative,
            relative_to=args.relative_to,
            label=args.label,
        )
        print(f"[+] Output saved to: {out_file}")

    # Show preview
    preview_count = min(3, len(sampled))
    print(f"[*] Preview of first {preview_count} lines:")
    with open(args.output_txt, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 5:
                break
            print(f"    [{idx + 1}] {line.strip()}")


if __name__ == "__main__":
    main()
