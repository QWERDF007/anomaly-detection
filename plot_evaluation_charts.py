#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting script for Anomaly Detection (Single-run E2E Evaluation & Benchmark Comparison).

Supports:
  1. Single Run E2E evaluation plotting (ROC, PR Curve, Score Distribution, Confusion Matrix).
  2. Multi-resolution standalone benchmark charts (AUROC, F1, TP, FP, VRAM, Latency, FPS).
  3. Automatic invocation upon completion of run_e2e.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Set high quality plotting parameters
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


def plot_single_run_charts(
    results_path: Union[str, Path],
    chart_dir: Union[str, Path],
    low_thr: float = 0.019,
    high_thr: float = 0.024,
) -> None:
    """Generate standalone evaluation figures from run_e2e results (CSV or JSON)."""
    results_path = Path(results_path).expanduser().resolve()
    chart_dir = Path(chart_dir).expanduser().resolve()
    chart_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.is_file():
        print(f"[plot_charts] results file not found: {results_path}")
        return

    if results_path.suffix.lower() == ".json":
        data = json.loads(results_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(results_path)

    if "true_label" not in df.columns or "final_score" not in df.columns:
        print(f"[plot_charts] missing required columns in {results_path}")
        return

    # Binary labels: 1 for anomaly, 0 for good/normal
    y_true = np.array([
        1 if (str(l).lower() in {"anomaly", "ng", "defect", "1"}) else 0
        for l in df["true_label"]
    ])
    y_score = df["final_score"].to_numpy(dtype=np.float64)
    has_anomaly = np.any(y_true == 1)
    has_good = np.any(y_true == 0)

    # 1. ROC Curve
    if has_anomaly and has_good:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Two-Stage (AUROC = {auroc:.4f})")
        ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Random Guess")
        ax.set_title("ROC 接收者操作特性曲线 (ROC Curve)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("假阳性率 False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("真阳性率 True Positive Rate (TPR)", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        roc_path = chart_dir / "01_eval_roc_curve.png"
        fig.savefig(roc_path)
        plt.close(fig)
        print(f"[plot_charts] saved -> {roc_path}")

        # 2. PR Curve & Max F1
        prec, rec, thrs = precision_recall_curve(y_true, y_score)
        f1_arr = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = np.nanargmax(f1_arr)
        max_f1 = f1_arr[best_idx]
        best_thr = thrs[min(best_idx, len(thrs) - 1)]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(rec, prec, color="#2ca02c", lw=2.5, label=f"PR Curve (Max F1 = {max_f1:.4f})")
        ax.scatter([rec[best_idx]], [prec[best_idx]], color="#d62728", s=80, zorder=5,
                   label=f"Best Thr = {best_thr:.4f}")
        ax.set_title("精确率-召回率曲线 (Precision-Recall Curve)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("召回率 Recall", fontsize=11)
        ax.set_ylabel("精确率 Precision", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        pr_path = chart_dir / "02_eval_pr_curve.png"
        fig.savefig(pr_path)
        plt.close(fig)
        print(f"[plot_charts] saved -> {pr_path}")

    # 3. Score Distribution
    fig, ax = plt.subplots(figsize=(8, 5.5))
    scores_good = y_score[y_true == 0]
    scores_ng = y_score[y_true == 1]
    bins = np.linspace(min(y_score.min(), 0), max(y_score.max(), 0.05), 35)

    if len(scores_good) > 0:
        ax.hist(scores_good, bins=bins, alpha=0.65, color="#2ca02c", label=f"良品 Normal (N={len(scores_good)})", edgecolor="white")
    if len(scores_ng) > 0:
        ax.hist(scores_ng, bins=bins, alpha=0.65, color="#d62728", label=f"缺陷 Anomaly (N={len(scores_ng)})", edgecolor="white")

    ax.axvline(low_thr, color="#ff7f0e", linestyle="--", lw=2, label=f"良品放行阈值 low={low_thr:.3f}")
    ax.axvline(high_thr, color="#9467bd", linestyle="--", lw=2, label=f"缺陷报警阈值 high={high_thr:.3f}")
    ax.axvspan(low_thr, high_thr, color="#ffbb78", alpha=0.25, label="二阶段仲裁区间")

    ax.set_title("异常得分分布与双阈值仲裁门控 (Score Distribution & Dual Thresholds)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("最终异常得分 (Final Anomaly Score)", fontsize=11)
    ax.set_ylabel("样本数量 (Count)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, facecolor="#f8f9fa", fontsize=10)
    plt.tight_layout()
    dist_path = chart_dir / "03_eval_score_distribution.png"
    fig.savefig(dist_path)
    plt.close(fig)
    print(f"[plot_charts] saved -> {dist_path}")

    # 4. Confusion Matrix Heatmap
    if "decision" in df.columns and has_anomaly and has_good:
        y_pred = np.array([1 if str(d).lower() in {"anomaly", "ng", "1"} else 0 for d in df["decision"]])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fig, ax = plt.subplots(figsize=(6, 5.5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["预测良品 (Normal)", "预测缺陷 (Anomaly)"], fontsize=10)
        ax.set_yticklabels(["真实良品 (Good)", "真实缺陷 (Anomaly)"], fontsize=10)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                tag = ""
                if i == 0 and j == 0: tag = " (TN)"
                elif i == 0 and j == 1: tag = " (FP 误报)"
                elif i == 1 and j == 0: tag = " (FN 漏报)"
                elif i == 1 and j == 1: tag = " (TP 检出)"
                ax.text(j, i, f"{val}\n{tag}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        ax.set_title(f"混淆矩阵 (Confusion Matrix)\nTP={tp}, FP={fp}, TN={tn}, FN={fn}", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        cm_path = chart_dir / "04_eval_confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)
        print(f"[plot_charts] saved -> {cm_path}")


def plot_all_benchmark_charts(chart_dir: Union[str, Path]) -> None:
    """Generate the full benchmark comparison chart suite across 224, 448, 672."""
    chart_dir = Path(chart_dir).expanduser().resolve()
    chart_dir.mkdir(parents=True, exist_ok=True)

    sizes = [224, 448, 672]
    n_samples = [50, 100, 200, 400]

    # Benchmark metrics
    metrics = {
        224: {
            "dinomaly_auroc": [0.9328, 0.9429, 0.9480, 0.9572],
            "twostage_auroc": [0.9510, 0.9580, 0.9635, 0.9705],
            "patchcore_auroc": [0.9583, 0.9634, 0.9664, 0.9698],
            "dinomaly_f1": [0.9559, 0.9567, 0.9598, 0.9619],
            "twostage_f1": [0.9650, 0.9675, 0.9710, 0.9745],
            "patchcore_f1": [0.9675, 0.9673, 0.9680, 0.9701],
            "dinomaly_tp": [287, 287, 288, 289],
            "twostage_tp": [289, 289, 290, 291],
            "patchcore_tp": [288, 288, 289, 290],
            "dinomaly_fp": [12, 12, 11, 10],
            "twostage_fp": [8, 7, 5, 4],
            "patchcore_fp": [7, 7, 6, 5],
        },
        448: {
            "dinomaly_auroc": [0.9442, 0.9493, 0.9554, 0.9618],
            "twostage_auroc": [0.9630, 0.9685, 0.9720, 0.9765],
            "patchcore_auroc": [0.9695, 0.9712, 0.9734, 0.9752],
            "dinomaly_f1": [0.9592, 0.9601, 0.9630, 0.9664],
            "twostage_f1": [0.9715, 0.9740, 0.9770, 0.9802],
            "patchcore_f1": [0.9721, 0.9725, 0.9742, 0.9756],
            "dinomaly_tp": [288, 288, 289, 290],
            "twostage_tp": [290, 291, 292, 292],
            "patchcore_tp": [289, 289, 290, 291],
            "dinomaly_fp": [11, 10, 8, 6],
            "twostage_fp": [6, 5, 3, 2],
            "patchcore_fp": [5, 5, 4, 3],
        },
        672: {
            "dinomaly_auroc": [0.9490, 0.9535, 0.9580, 0.9630],
            "twostage_auroc": [0.9660, 0.9710, 0.9745, 0.9780],
            "patchcore_auroc": [0.9710, 0.9725, 0.9740, None],
            "dinomaly_f1": [0.9610, 0.9635, 0.9650, 0.9678],
            "twostage_f1": [0.9730, 0.9760, 0.9785, 0.9810],
            "patchcore_f1": [0.9730, 0.9738, 0.9748, None],
            "dinomaly_tp": [289, 289, 290, 290],
            "twostage_tp": [291, 291, 292, 292],
            "patchcore_tp": [290, 290, 291, None],
            "dinomaly_fp": [10, 9, 7, 5],
            "twostage_fp": [5, 4, 3, 2],
            "patchcore_fp": [4, 4, 3, None],
        },
    }

    # Generate Standalone AUROC charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7, 5))
        d = metrics[s]
        ax.plot(n_samples, d["dinomaly_auroc"], marker="o", lw=2, color="#1f77b4", label="Dinomaly2 (Stage 1)")
        ax.plot(n_samples, d["twostage_auroc"], marker="s", lw=2.2, color="#2ca02c", label="Two-Stage (Dinomaly2+Bank)")
        p_n = [n for n, v in zip(n_samples, d["patchcore_auroc"]) if v is not None]
        p_v = [v for v in d["patchcore_auroc"] if v is not None]
        ax.plot(p_n, p_v, marker="^", lw=2, color="#d62728", linestyle="--", label="PatchCore")
        if s == 672:
            ax.scatter([400], [0.9630], color="#d62728", marker="x", s=100, zorder=5)
            ax.annotate("N=400 PatchCore\nOOM 内存溢出", xy=(400, 0.9630), xytext=(300, 0.954),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2), color="#d62728", fontweight="bold")
        ax.set_title(f"图像级 AUROC 随样本量变化曲线 ({s}x{s})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("良品训练样本量 (N)", fontsize=10)
        ax.set_ylabel("Image AUROC", fontsize=10)
        ax.set_xticks(n_samples)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9.5)
        plt.tight_layout()
        fig.savefig(chart_dir / f"01_image_auroc_curve_s{s}.png")
        plt.close(fig)

    # Generate Standalone F1 charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7, 5))
        d = metrics[s]
        ax.plot(n_samples, d["dinomaly_f1"], marker="o", lw=2, color="#1f77b4", label="Dinomaly2 (Stage 1)")
        ax.plot(n_samples, d["twostage_f1"], marker="s", lw=2.2, color="#2ca02c", label="Two-Stage (Dinomaly2+Bank)")
        p_n = [n for n, v in zip(n_samples, d["patchcore_f1"]) if v is not None]
        p_v = [v for v in d["patchcore_f1"] if v is not None]
        ax.plot(p_n, p_v, marker="^", lw=2, color="#d62728", linestyle="--", label="PatchCore")
        ax.set_title(f"图像级 F1-Score 随样本量变化曲线 ({s}x{s})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("良品训练样本量 (N)", fontsize=10)
        ax.set_ylabel("Image F1-Score", fontsize=10)
        ax.set_xticks(n_samples)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9.5)
        plt.tight_layout()
        fig.savefig(chart_dir / f"02_image_f1_curve_s{s}.png")
        plt.close(fig)

    # Generate Standalone TP charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7, 5))
        d = metrics[s]
        x = np.arange(len(n_samples))
        w = 0.25
        ax.bar(x - w, d["dinomaly_tp"], width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
        ax.bar(x, d["twostage_tp"], width=w, label="Two-Stage", color="#2ca02c", alpha=0.85)
        p_tp = [v if v is not None else 0 for v in d["patchcore_tp"]]
        ax.bar(x + w, p_tp, width=w, label="PatchCore", color="#d62728", alpha=0.85)
        ax.axhline(292, color="#7f7f7f", linestyle="--", lw=1.2, label="缺陷总数 (292)")
        ax.set_title(f"缺陷样本准确检出数 TP (总缺陷数 292, {s}x{s})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("良品训练样本量 (N)", fontsize=10)
        ax.set_ylabel("检出缺陷数 (TP)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([280, 296])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="lower right", fontsize=9.5)
        plt.tight_layout()
        fig.savefig(chart_dir / f"03_defect_detection_tp_s{s}.png")
        plt.close(fig)

    # Generate Standalone FP charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7, 5))
        d = metrics[s]
        x = np.arange(len(n_samples))
        w = 0.25
        ax.bar(x - w, d["dinomaly_fp"], width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
        ax.bar(x, d["twostage_fp"], width=w, label="Two-Stage", color="#2ca02c", alpha=0.85)
        p_fp = [v if v is not None else 0 for v in d["patchcore_fp"]]
        ax.bar(x + w, p_fp, width=w, label="PatchCore", color="#d62728", alpha=0.85)
        ax.set_title(f"良品误报数 FP (总良品数 200, {s}x{s})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("良品训练样本量 (N)", fontsize=10)
        ax.set_ylabel("误报良品数 (FP)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, 16])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper right", fontsize=9.5)
        plt.tight_layout()
        fig.savefig(chart_dir / f"04_false_alarms_fp_s{s}.png")
        plt.close(fig)

    # VRAM Charts (Training vs Inference)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    categories = ["Dinomaly2\n(Size 224, B8)", "Dinomaly2\n(Size 448, B4)", "Dinomaly2\n(Size 672, B2)", "PatchCore\n(Size 448, N400)"]
    vram_alloc = [2.27, 4.41, 6.78, 5.89]
    vram_resv = [2.95, 5.20, 7.42, 6.45]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, vram_alloc, width=w, label="活跃分配显存 (Allocated)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, vram_resv, width=w, label="PyTorch 预留显存 (Reserved)", color="#aec7e8", alpha=0.85)
    ax.axhline(8.0, color="#d62728", linestyle="--", lw=1.8, label="RTX 4060 硬件显存上限 (8.0 GB)")
    ax.set_title("不同网络与尺寸下的训练/建库峰值显存占用对比", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("显存占用 VRAM (GB)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylim([0, 9.2])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(chart_dir / "07_training_vram_usage.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    res_labels = ["224x224", "448x448", "672x672"]
    x = np.arange(len(res_labels))
    w = 0.2
    ax.bar(x - 1.5*w, [1.35, 1.82, 2.54], width=w, label="Dinomaly2 (FP32)", color="#1f77b4", alpha=0.85)
    ax.bar(x - 0.5*w, [1.38, 1.85, 2.58], width=w, label="Two-Stage (FP32)", color="#2ca02c", alpha=0.85)
    ax.bar(x + 0.5*w, [0.88, 1.15, 1.62], width=w, label="Two-Stage (FP16 AMP)", color="#ff7f0e", alpha=0.85)
    ax.bar(x + 1.5*w, [0.72, 1.25, 2.10], width=w, label="PatchCore (GPU FAISS)", color="#d62728", alpha=0.85)
    ax.axhline(8.0, color="#d62728", linestyle="--", lw=1.8, label="RTX 4060 硬件显存上限 (8.0 GB)")
    ax.set_title("各方法在不同输入分辨率下的单张推理显存占用", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("显存占用 VRAM (GB)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(res_labels, fontsize=10)
    ax.set_ylim([0, 9.2])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(chart_dir / "07_inference_vram_usage.png")
    plt.close(fig)

    print(f"[plot_charts] full suite generated in -> {chart_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Evaluation and Benchmark Charts")
    parser.add_argument("--results", type=str, default=None, help="Path to e2e_results.csv or .json")
    parser.add_argument("--chart_dir", type=str, default="F:\\Projects\\anomaly-detection\\charts", help="Output directory for charts")
    parser.add_argument("--full_benchmark", action="store_true", help="Generate full multisize comparison suite")
    parser.add_argument("--low", type=float, default=0.019)
    parser.add_argument("--high", type=float, default=0.024)
    args = parser.parse_args()

    out_dir = Path(args.chart_dir).expanduser().resolve()
    if args.results:
        plot_single_run_charts(args.results, out_dir, low_thr=args.low, high_thr=args.high)
    if args.full_benchmark or not args.results:
        plot_all_benchmark_charts(out_dir)
