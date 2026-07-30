"""Merge directory-evaluation ``metrics.csv`` files into three metric tables.

Expected layout::

    eval_root/
    ├── patchcore/
    │   ├── parameter_a/metrics.csv
    │   └── parameter_b/metrics.csv
    └── dinomaly2/
        ├── parameter_a/metrics.csv
        └── parameter_b/metrics.csv

Each source row is preserved.  The output ``group`` value identifies the
model and parameter result directory, for example ``patchcore/parameter_a``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_METRICS = (
    "I-AUROC",
    "I-AP",
    "I-F1",
    "FPR",
    "TNR",
    "Accuracy",
)
PIXEL_METRICS = (
    "P-AUROC",
    "P-AP",
    "P-F1",
    "P-AUPRO",
)
REGION_METRICS = (
    "R-MissRate",
    "R-PixelCoverage",
    "R-GT-ImageCount",
    "R-GT-RegionCount",
)
MODEL_NAMES = ("patchcore", "dinomaly2")


def _result_directories(eval_root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield ``(group, metrics.csv)`` pairs in deterministic order."""

    for model_name in MODEL_NAMES:
        model_dir = eval_root / model_name
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing model result directory: {model_dir}")
        result_dirs = sorted(
            (path for path in model_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        )
        if not result_dirs:
            raise RuntimeError(f"No parameter result directories found in {model_dir}")
        for result_dir in result_dirs:
            metrics_path = result_dir / "metrics.csv"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing metrics.csv for {model_name}/{result_dir.name}: {metrics_path}")
            yield f"{model_name}/{result_dir.name}", metrics_path


def _read_rows(group: str, metrics_path: Path) -> List[Dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "directory" not in reader.fieldnames:
            raise ValueError(f"metrics.csv must contain a directory column: {metrics_path}")
        rows = []
        for row in reader:
            if not row.get("directory"):
                continue
            rows.append({"group": group, **row})
    if not rows:
        raise RuntimeError(f"metrics.csv has no directory rows: {metrics_path}")
    return rows


def _write_table(
    output_path: Path,
    rows: Sequence[Dict[str, str]],
    metrics: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("group", "directory", *metrics)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge PatchCore and Dinomaly2 directory-evaluation metrics into image/pixel/region CSV files.",
    )
    parser.add_argument(
        "-i",
        "--eval_dir",
        "--eval_root",
        required=True,
        type=Path,
        help="Directory containing patchcore/ and dinomaly2/ parameter-result directories.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: eval_dir).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_root = args.eval_dir.expanduser().resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Evaluation root does not exist: {eval_root}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else eval_root

    rows: List[Dict[str, str]] = []
    for group, metrics_path in _result_directories(eval_root):
        source_rows = _read_rows(group, metrics_path)
        rows.extend(source_rows)
        print(f"Read {len(source_rows)} row(s): {group}")

    outputs = (
        ("image_metrics.csv", IMAGE_METRICS),
        ("pixel_metrics.csv", PIXEL_METRICS),
        ("region_metrics.csv", REGION_METRICS),
    )
    for filename, metrics in outputs:
        path = output_dir / filename
        _write_table(path, rows, metrics)
        print(f"Wrote {len(rows)} row(s): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
