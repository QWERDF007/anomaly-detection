"""Plot image-level anomaly-score distributions from saved score maps.

Each ``.npy`` file is expected to contain a pixel-level anomaly map.  The
maximum value of the map is used as the image-level anomaly score, matching
the score calculation in ``predict.py`` and ``get_score.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def iter_npy_paths(source: Path, recursive: bool = True) -> list[Path]:
    source = Path(source).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Score-map path does not exist: {source}")

    iterator: Iterable[Path] = source.rglob("*.npy") if recursive else source.glob("*.npy")
    return sorted(
        [path for path in iterator if path.is_file()],
        key=lambda path: str(path).lower(),
    )


def score_from_npy(path: Path) -> float:
    score_map = np.asarray(np.load(path), dtype=np.float32)
    score_map = np.squeeze(score_map)
    if score_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D score map in {path}, got shape {score_map.shape}"
        )
    score_map = np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
    return float(score_map.max())


def infer_label(path: Path, root: Path) -> str:
    """Infer labels from directory names for the default input layout."""

    try:
        relative_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]
    parts = {part.lower() for part in relative_parts}
    if root.name.lower() in {"good", "normal"} or parts.intersection(
        {"good", "normal"}
    ):
        return "Good"
    return "Anomaly"


def collect_scores(
    source: Path,
    label: str,
    recursive: bool,
) -> Dict[str, list[float]]:
    scores: list[float] = []
    for path in iter_npy_paths(source, recursive=recursive):
        try:
            scores.append(score_from_npy(path))
        except (OSError, ValueError) as error:
            print(f"[skip] {error}")
    return {label: scores}


def collect_auto_scores(source: Path, recursive: bool) -> Dict[str, list[float]]:
    grouped: Dict[str, list[float]] = {"Good": [], "Anomaly": []}
    for path in iter_npy_paths(source, recursive=recursive):
        try:
            grouped[infer_label(path, source)].append(score_from_npy(path))
        except (OSError, ValueError) as error:
            print(f"[skip] {error}")
    return {label: values for label, values in grouped.items() if values}


def collect_split_scores(
    source: Path,
    split_name: str,
    recursive: bool,
) -> Dict[str, list[float]]:
    """Collect Good/Anomaly score distributions for one train or test split."""

    source = Path(source).expanduser()
    grouped: Dict[str, list[float]] = {}
    for path in iter_npy_paths(source, recursive=recursive):
        try:
            relative_parts = path.relative_to(source).parts
        except ValueError:
            relative_parts = path.parts
        is_good = (
            bool(relative_parts)
            and relative_parts[0].lower() in {"good", "normal"}
        )
        label = f"{split_name} / {'Good' if is_good else 'Anomaly'}"
        try:
            grouped.setdefault(label, []).append(score_from_npy(path))
        except (OSError, ValueError) as error:
            print(f"[skip] {error}")
    return grouped


def merge_scores(
    target: Dict[str, list[float]],
    source: Optional[Dict[str, list[float]]],
) -> None:
    if not source:
        return
    for label, values in source.items():
        target.setdefault(label, []).extend(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot score distributions using the maximum value of each .npy map."
    )
    parser.add_argument(
        "--input",
        help=(
            "Root directory or one .npy file. With --label auto, directories "
            "named good/normal are treated as Good and the rest as Anomaly."
        ),
    )
    parser.add_argument(
        "--train",
        help="Train score-map directory, normally containing good and/or anomaly subdirectories.",
    )
    parser.add_argument(
        "--test",
        help="Test score-map directory, normally containing good and anomaly subdirectories.",
    )
    parser.add_argument(
        "--good_dir",
        help="Optional directory/file containing Good score maps.",
    )
    parser.add_argument(
        "--anomaly_dir",
        help="Optional directory/file containing Anomaly score maps.",
    )
    parser.add_argument(
        "--output",
        default="score_distribution.png",
        help="Output plot path. Default: score_distribution.png.",
    )
    parser.add_argument(
        "--label",
        choices=("auto", "good", "anomaly"),
        default="auto",
        help="Label for --input when it is not organized by subdirectories.",
    )
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not search score-map subdirectories.",
    )
    parser.set_defaults(recursive=True)
    return parser


def plot_distribution(
    grouped_scores: Dict[str, list[float]],
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    all_scores = [score for values in grouped_scores.values() for score in values]
    if not all_scores:
        raise ValueError("No valid .npy score maps were found.")

    low = float(min(all_scores))
    high = float(max(all_scores))
    if high <= low:
        margin = max(abs(low) * 0.05, 1e-6)
        edges = np.linspace(low - margin, high + margin, bins + 1)
    else:
        edges = np.linspace(low, high, bins + 1)

    colors = {
        "Train / Good": "green",
        "Train / Anomaly": "orange",
        "Test / Good": "blue",
        "Test / Anomaly": "red",
        "Good": "green",
        "Anomaly": "red",
    }
    plt.figure(figsize=(10, 6))
    for label, values in grouped_scores.items():
        if not values:
            continue
        plt.hist(
            values,
            bins=edges,
            alpha=0.7,
            color=colors.get(label, "steelblue"),
            label=f"{label} (n={len(values)})",
        )

    plt.xlabel("Anomaly Score (maximum of score map)")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.bins < 1:
        raise ValueError("--bins must be greater than 0.")
    if args.dpi < 1:
        raise ValueError("--dpi must be greater than 0.")
    if not any(
        [args.input, args.train, args.test, args.good_dir, args.anomaly_dir]
    ):
        raise ValueError(
            "Provide --input, --train, --test, --good_dir, or --anomaly_dir."
        )

    grouped_scores: Dict[str, list[float]] = {}
    if args.train or args.test:
        if args.input:
            print("[info] --train/--test specified; ignoring --input.")
        if args.train:
            merge_scores(
                grouped_scores,
                collect_split_scores(
                    Path(args.train), "Train", args.recursive
                ),
            )
        if args.test:
            merge_scores(
                grouped_scores,
                collect_split_scores(
                    Path(args.test), "Test", args.recursive
                ),
            )
    elif args.input:
        input_path = Path(args.input).expanduser()
        if args.label == "auto":
            merge_scores(grouped_scores, collect_auto_scores(input_path, args.recursive))
        elif args.label == "good":
            merge_scores(
                grouped_scores,
                collect_scores(input_path, "Good", args.recursive),
            )
        else:
            merge_scores(
                grouped_scores,
                collect_scores(input_path, "Anomaly", args.recursive),
            )

    if args.good_dir:
        merge_scores(
            grouped_scores,
            collect_scores(Path(args.good_dir).expanduser(), "Good", args.recursive),
        )
    if args.anomaly_dir:
        merge_scores(
            grouped_scores,
            collect_scores(
                Path(args.anomaly_dir).expanduser(), "Anomaly", args.recursive
            ),
        )

    output_path = Path(args.output).expanduser()
    plot_distribution(grouped_scores, output_path, args.bins, args.dpi)

    for label, values in grouped_scores.items():
        print(
            f"{label}: count={len(values)}, "
            f"max={max(values):.6f}, mean={np.mean(values):.6f}"
        )
    print(f"Distribution plot saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
