"""Metrics for source-only distribution-generalization experiments."""

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


def production_threshold(clean_normal_scores, target_fpr: float = 0.01) -> float:
    """Calibrate a deployment threshold using clean normal scores only."""

    if not 0 < target_fpr < 1:
        raise ValueError('target_fpr must be in (0, 1)')
    scores = np.asarray(clean_normal_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError('clean_normal_scores must be non-empty')
    # ``higher`` guarantees that no more than the requested empirical tail is
    # accepted as an anomaly when the sample count is small.
    return float(np.quantile(scores, 1.0 - target_fpr, method='higher'))


def shift_fpr_at_tpr(
    shifted_normal_scores,
    ng_scores,
    target_tpr: float = 0.95,
) -> Tuple[float, float, float]:
    """Return ``(FPR, threshold, achieved_ng_recall)`` for Shift-FPR@TPR."""

    if not 0 < target_tpr <= 1:
        raise ValueError('target_tpr must be in (0, 1]')
    shifted = np.asarray(shifted_normal_scores, dtype=np.float64).reshape(-1)
    ng = np.asarray(ng_scores, dtype=np.float64).reshape(-1)
    if shifted.size == 0 or ng.size == 0:
        raise ValueError('shifted_normal_scores and ng_scores must be non-empty')
    threshold = float(np.quantile(ng, 1.0 - target_tpr, method='lower'))
    return (
        float(np.mean(shifted >= threshold)),
        threshold,
        float(np.mean(ng >= threshold)),
    )


def bootstrap_mean_ci(
    values,
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 1,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a mean without scipy dependencies."""

    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')
    if iterations <= 0:
        raise ValueError('iterations must be positive')
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError('values must be non-empty')
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, values.size), replace=True)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(samples.mean(axis=1), alpha)),
        float(np.quantile(samples.mean(axis=1), 1.0 - alpha)),
    )


def summarize_shift_scores(
    clean_normal_scores,
    shifted_scores_by_family: Mapping[str, Sequence[float]],
    ng_scores,
    target_fpr: float = 0.01,
    target_tpr: float = 0.95,
) -> Dict[str, object]:
    """Summarize production and TPR-constrained shift metrics by family."""

    clean = np.asarray(clean_normal_scores, dtype=np.float64).reshape(-1)
    ng = np.asarray(ng_scores, dtype=np.float64).reshape(-1)
    if clean.size == 0 or ng.size == 0:
        raise ValueError('clean_normal_scores and ng_scores must be non-empty')
    prod_threshold = production_threshold(clean, target_fpr=target_fpr)
    family_metrics: Dict[str, Dict[str, float]] = {}
    production_fprs = []
    constrained_fprs = []
    for family, values in shifted_scores_by_family.items():
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
        if scores.size == 0:
            continue
        constrained_fpr, tpr_threshold, achieved_tpr = shift_fpr_at_tpr(
            scores, ng, target_tpr=target_tpr
        )
        production_fpr = float(np.mean(scores >= prod_threshold))
        family_metrics[family] = {
            'count': int(scores.size),
            'production_fpr': production_fpr,
            'shift_fpr_at_tpr': constrained_fpr,
            'tpr_threshold': tpr_threshold,
            'achieved_ng_recall': achieved_tpr,
            'p50': float(np.quantile(scores, 0.50)),
            'p95': float(np.quantile(scores, 0.95)),
            'p99': float(np.quantile(scores, 0.99)),
        }
        production_fprs.append(production_fpr)
        constrained_fprs.append(constrained_fpr)
    return {
        'production_threshold': prod_threshold,
        'clean_fpr': float(np.mean(clean >= prod_threshold)),
        'mean_production_shift_fpr': float(np.mean(production_fprs)) if production_fprs else float('nan'),
        'worst_production_shift_fpr': float(np.max(production_fprs)) if production_fprs else float('nan'),
        'mean_shift_fpr_at_tpr': float(np.mean(constrained_fprs)) if constrained_fprs else float('nan'),
        'worst_shift_fpr_at_tpr': float(np.max(constrained_fprs)) if constrained_fprs else float('nan'),
        'families': family_metrics,
    }
