"""Dinomaly2 adapter for the canonical shared anomaly evaluator.

The common metric definitions are in ``utils/anomaly_evaluation.py``.  This
file only resolves Dinomaly2 ground-truth masks and keeps the historical
``evaluate_stage`` API used by the score and ROI pipelines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SHARED_UTILS = ROOT.parent / "utils"
if str(SHARED_UTILS) not in sys.path:
    # Do not use ``from utils...`` here: Dinomaly2 itself has utils.py.
    sys.path.insert(1, str(SHARED_UTILS))

from anomaly_evaluation import (  # noqa: E402,F401
    CLASSIFICATION_METRIC_NAMES,
    METRIC_NAMES,
    PIXEL_METRIC_NAMES,
    REPORT_METRIC_NAMES,
    TRAINING_IMAGE_SCORE_RATIO,
    classification_metrics,
    compute_evaluation_metrics,
    compute_pro_fast,
    evaluate_pixel_metrics,
    load_score_map,
    max_f1,
    pixel_f1_score_and_threshold,
    region_detection_metrics,
    report_metric_names,
    safe_ap,
    safe_aupro,
    safe_auroc,
    select_optimal_threshold,
    training_image_score,
    write_metrics,
    write_per_image_pixel_metrics,
)
from dinomaly_pipeline_common import load_ground_truth


def _load_stage_score_map(sample: Dict, score_map_key: str) -> np.ndarray:
    try:
        source = sample[score_map_key]
    except KeyError as error:
        raise KeyError(f"Sample does not contain score-map field {score_map_key!r}.") from error
    if isinstance(source, (str, Path)):
        return load_score_map(Path(source))
    score_map = np.squeeze(np.asarray(source, dtype=np.float32))
    if score_map.ndim != 2:
        raise ValueError(f"Score map in {score_map_key!r} must be 2D; got {score_map.shape}")
    return np.nan_to_num(
        score_map,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )


def evaluate_stage(
    samples: Sequence[Dict],
    ground_truth_dir: Optional[Path],
    metric_size: int,
    score_map_key: str = "score_path",
    image_score_key: Optional[str] = None,
    stage_name: str = "score maps",
    per_image_records: Optional[list[Dict[str, object]]] = None,
    pixel_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate already-generated Dinomaly2 maps with canonical metrics.

    ``image_score_key`` remains accepted for source compatibility but is
    intentionally ignored when score maps are available: image scores always
    use the Dinomaly2 highest-1% rule.
    """

    if metric_size < 1:
        raise ValueError("metric_size must be positive")
    evaluation_samples = [
        sample
        for sample in samples
        if sample["group_key"] in {"test_good", "test_anomaly"}
    ]
    if not evaluation_samples:
        raise RuntimeError("No Test/good or Test/anomaly samples were found.")

    image_labels = []
    image_scores = []
    gt_pixels = []
    score_pixels = []
    with tqdm(
        evaluation_samples,
        desc=f"Evaluate {stage_name}",
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        for sample in progress:
            score_source = sample[score_map_key]
            score_map = _load_stage_score_map(sample, score_map_key)
            gt_mask = load_ground_truth(sample, ground_truth_dir, score_map.shape)
            resized_gt = cv2.resize(
                gt_mask, (metric_size, metric_size), interpolation=cv2.INTER_NEAREST
            )
            resized_score = cv2.resize(
                score_map, (metric_size, metric_size), interpolation=cv2.INTER_LINEAR
            )
            image_label = int(sample["group_key"] == "test_anomaly")
            image_score = training_image_score(resized_score)
            image_labels.append(image_label)
            image_scores.append(image_score)
            gt_pixels.append((resized_gt > 0).astype(np.uint8))
            score_pixels.append(resized_score)

            if per_image_records is not None:
                per_image_records.append(
                    {
                        "stage": stage_name,
                        "group": sample.get("group_label", sample["group_key"]),
                        "image_path": str(sample["image_path"]),
                        "score_path": (
                            str(score_source)
                            if isinstance(score_source, (str, Path))
                            else ""
                        ),
                        "image_label": image_label,
                        "image_score": image_score,
                        "gt_positive_pixels": int(np.asarray(resized_gt, dtype=bool).sum()),
                        **evaluate_pixel_metrics(resized_gt, resized_score),
                    }
                )

    gt_array = np.stack(gt_pixels, axis=0)
    score_array = np.stack(score_pixels, axis=0)
    metrics = compute_evaluation_metrics(
        image_scores,
        image_labels,
        score_array,
        gt_array,
    )
    p_f1_threshold = metrics["P-F1-Threshold"]
    metrics.update(
        region_detection_metrics(
            gt_array,
            score_array,
            p_f1_threshold if pixel_threshold is None else float(pixel_threshold),
            per_image_records,
            p_f1_threshold=p_f1_threshold,
        )
    )
    return metrics


def print_metrics(results: Mapping[str, Mapping[str, float]]) -> None:
    """Print standardized metrics without writing files."""

    print("\nEvaluation metrics")
    print("stage                         " + "  ".join(f"{name:>10}" for name in REPORT_METRIC_NAMES))
    for stage, metrics in results.items():
        values = "  ".join(f"{metrics.get(name, float('nan')):10.6f}" for name in REPORT_METRIC_NAMES)
        print(f"{stage:<29}{values}")
    print()


def print_and_save_metrics(
    results: Mapping[str, Mapping[str, float]], output_dir: Path
) -> None:
    """Save and print the standardized metrics report."""

    write_metrics(results, output_dir, "stage")
