"""PatchCore compatibility imports for the canonical Dinomaly2 metrics.

All metric formulas live in ``utils/anomaly_evaluation.py``.  This module is
kept only so existing PatchCore entry points and third-party imports continue
to work without carrying a second metric implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SHARED_UTILS = ROOT.parent / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS))

from anomaly_evaluation import (  # noqa: E402,F401
    CLASSIFICATION_METRIC_NAMES,
    METRIC_NAMES,
    PER_IMAGE_METRIC_FIELDS,
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
    prepare_pixel_arrays,
    region_detection_metrics,
    report_metric_names,
    resize_metric_arrays,
    safe_ap,
    safe_aupro,
    safe_auroc,
    select_optimal_threshold,
    training_image_score,
    write_metric_report,
    write_metrics,
    write_per_image_pixel_metrics,
    write_per_image_report,
)


# Legacy PatchCore names intentionally point at the same Dinomaly2 functions.
safe_average_precision = safe_ap
safe_f1_max = max_f1
compute_pro = compute_pro_fast

