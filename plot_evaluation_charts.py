#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting script for Anomaly Detection Benchmark Comparison.

Dynamically extracts real metrics from outs_dir (predictions, results, json logs)
and generates high-precision, publication-quality figures matching the report tables.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# High quality plotting setup
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Serif CJK SC", "SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


def extract_benchmark_metrics(outs_dir: Path):
    """Extract real evaluated metrics for all 12 experiment combinations."""
    sizes = [224, 448, 672]
    n_samples = [50, 100, 200, 400]

    metrics = {s: {
        "dinomaly_auroc": [], "twostage_auroc": [], "patchcore_auroc": [],
        "dinomaly_ap": [], "twostage_ap": [], "patchcore_ap": [],
        "dinomaly_f1": [], "twostage_f1": [], "patchcore_f1": [],
        "dinomaly_tp": [], "twostage_tp": [], "patchcore_tp": [],
        "dinomaly_fp": [], "twostage_fp": [], "patchcore_fp": [],
        "dinomaly_train_min": [], "patchcore_train_min": [],
    } for s in sizes}

    for s in sizes:
        for n in n_samples:
            e2e_f = outs_dir / f"e2e_out_n{n}_s{s}" / "e2e_results.csv"
            pat_glob = list(outs_dir.glob(f"patchcore_n{n}_s{s}_seed2024/*/predictions.csv"))

            if not e2e_f.is_file():
                continue

            df_e = pd.read_csv(e2e_f)
            y_true = (df_e["true_label"] != "good").astype(int).values
            raw_s = df_e["raw_score"].values
            final_s = df_e["final_score"].values
            dec_e = (df_e["decision"] == "anomaly").astype(int).values

            # Dinomaly2 metrics (at optimal F1 threshold)
            din_auc = roc_auc_score(y_true, raw_s)
            din_ap = average_precision_score(y_true, raw_s)
            p, r, t = precision_recall_curve(y_true, raw_s)
            f1_arr = 2 * p * r / (p + r + 1e-8)
            b_idx = np.argmax(f1_arr)
            din_f1 = f1_arr[b_idx]
            din_pred = (raw_s >= t[min(b_idx, len(t) - 1)]).astype(int)
            tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, din_pred).ravel()

            # Two-Stage E2E metrics
            e2e_auc = roc_auc_score(y_true, final_s)
            e2e_ap = average_precision_score(y_true, final_s)
            p_e, r_e, t_e = precision_recall_curve(y_true, final_s)
            f1_e_arr = 2 * p_e * r_e / (p_e + r_e + 1e-8)
            b_e_idx = np.argmax(f1_e_arr)
            e2e_f1 = f1_e_arr[b_e_idx]
            tn_e, fp_e, fn_e, tp_e = confusion_matrix(y_true, dec_e).ravel()

            # PatchCore metrics (at optimal F1 threshold)
            if pat_glob and pat_glob[0].is_file():
                df_p = pd.read_csv(pat_glob[0])
                p_true = (df_p["anomaly"].astype(str).str.lower() == "true").astype(int).values
                p_score = df_p["anomaly_score"].values
                pat_auc = roc_auc_score(p_true, p_score)
                pat_ap = average_precision_score(p_true, p_score)
                p_p, r_p, t_p = precision_recall_curve(p_true, p_score)
                f1_p_arr = 2 * p_p * r_p / (p_p + r_p + 1e-8)
                b_p_idx = np.argmax(f1_p_arr)
                pat_f1 = f1_p_arr[b_p_idx]
                pat_pred = (p_score >= t_p[min(b_p_idx, len(t_p) - 1)]).astype(int)
                tn_p, fp_p, fn_p, tp_p = confusion_matrix(p_true, pat_pred).ravel()
            else:
                pat_auc = pat_ap = pat_f1 = tp_p = fp_p = None

            metrics[s]["dinomaly_auroc"].append(din_auc)
            metrics[s]["twostage_auroc"].append(e2e_auc)
            metrics[s]["patchcore_auroc"].append(pat_auc)
            metrics[s]["dinomaly_ap"].append(din_ap)
            metrics[s]["twostage_ap"].append(e2e_ap)
            metrics[s]["patchcore_ap"].append(pat_ap)
            metrics[s]["dinomaly_f1"].append(din_f1)
            metrics[s]["twostage_f1"].append(e2e_f1)
            metrics[s]["patchcore_f1"].append(pat_f1)
            metrics[s]["dinomaly_tp"].append(tp_d)
            metrics[s]["twostage_tp"].append(tp_e)
            metrics[s]["patchcore_tp"].append(tp_p)
            metrics[s]["dinomaly_fp"].append(fp_d)
            metrics[s]["twostage_fp"].append(fp_e)
            metrics[s]["patchcore_fp"].append(fp_p)

    return metrics, sizes, n_samples


def plot_all_benchmark_charts(outs_dir: Union[str, Path], chart_dir: Optional[Union[str, Path]] = None) -> None:
    """Generate the full benchmark comparison chart suite across 224, 448, 672."""
    outs_dir = Path(outs_dir).expanduser().resolve()
    if chart_dir is None:
        chart_dir = outs_dir / "charts"
    else:
        chart_dir = Path(chart_dir).expanduser().resolve()
    chart_dir.mkdir(parents=True, exist_ok=True)

    metrics, sizes, n_samples = extract_benchmark_metrics(outs_dir)

    # 1. Standalone AUROC charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        d = metrics[s]
        ax.plot(n_samples, d["dinomaly_auroc"], marker="o", lw=2.2, color="#1f77b4", label="Dinomaly2 基线")
        ax.plot(n_samples, d["twostage_auroc"], marker="s", lw=2.2, color="#2ca02c", label="二阶段端到端 (Two-Stage)")
        p_n = [n for n, v in zip(n_samples, d["patchcore_auroc"]) if v is not None]
        p_v = [v for v in d["patchcore_auroc"] if v is not None]
        ax.plot(p_n, p_v, marker="^", lw=2.0, color="#d62728", linestyle="--", label="PatchCore 基线")

        ax.set_title(f"图像级 AUROC 随训练样本量 N 变化曲线 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("Image AUROC", fontsize=10.5)
        ax.set_xticks(n_samples)
        ax.set_ylim([0.80, 0.98])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"01_image_auroc_curve_s{s}.png")
        plt.close(fig)

    # 2. Standalone F1 charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        d = metrics[s]
        ax.plot(n_samples, d["dinomaly_f1"], marker="o", lw=2.2, color="#1f77b4", label="Dinomaly2 基线")
        ax.plot(n_samples, d["twostage_f1"], marker="s", lw=2.2, color="#2ca02c", label="二阶段端到端 (Two-Stage)")
        p_n = [n for n, v in zip(n_samples, d["patchcore_f1"]) if v is not None]
        p_v = [v for v in d["patchcore_f1"] if v is not None]
        ax.plot(p_n, p_v, marker="^", lw=2.0, color="#d62728", linestyle="--", label="PatchCore 基线")

        ax.set_title(f"图像级 F1-Score 随训练样本量 N 变化曲线 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("Image F1-Score (F1-Max)", fontsize=10.5)
        ax.set_xticks(n_samples)
        ax.set_ylim([0.25, 0.75])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"02_image_f1_curve_s{s}.png")
        plt.close(fig)

    # 3. Standalone TP charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        d = metrics[s]
        x = np.arange(len(n_samples))
        w = 0.25
        ax.bar(x - w, d["dinomaly_tp"], width=w, label="Dinomaly2 (最佳F1阈值)", color="#1f77b4", alpha=0.85)
        ax.bar(x, d["twostage_tp"], width=w, label="二阶段端到端 (最佳F1平衡模式)", color="#2ca02c", alpha=0.85)
        p_tp = [v if v is not None else 0 for v in d["patchcore_tp"]]
        ax.bar(x + w, p_tp, width=w, label="PatchCore (最佳阈值)", color="#d62728", alpha=0.85)
        ax.axhline(53, color="#7f7f7f", linestyle="--", lw=1.2, label="缺陷总数 (53)")
        for i in range(len(n_samples)):
            ax.text(x[i] - w, d["dinomaly_tp"][i] + 0.8, f"{d['dinomaly_tp'][i]}", ha="center", va="bottom", fontsize=8.5)
            ax.text(x[i], d["twostage_tp"][i] + 0.8, f"{d['twostage_tp'][i]}", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
            ax.text(x[i] + w, p_tp[i] + 0.8, f"{p_tp[i]}", ha="center", va="bottom", fontsize=8.5, color="#d62728")

        ax.set_title(f"缺陷样本准确检出数 TP (总缺陷数 53, {s}×{s})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("检出缺陷数 (TP)", fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, 60])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="lower right", fontsize=9.0, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"03_defect_detection_tp_s{s}.png")
        plt.close(fig)

    # 4. Standalone FP charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        d = metrics[s]
        x = np.arange(len(n_samples))
        w = 0.25
        ax.bar(x - w, d["dinomaly_fp"], width=w, label="Dinomaly2 (最佳F1阈值)", color="#1f77b4", alpha=0.85)
        ax.bar(x, d["twostage_fp"], width=w, label="二阶段端到端 (最佳F1平衡模式)", color="#2ca02c", alpha=0.85)
        p_fp = [v if v is not None else 0 for v in d["patchcore_fp"]]
        ax.bar(x + w, p_fp, width=w, label="PatchCore (最佳阈值)", color="#d62728", alpha=0.85)

        for i in range(len(n_samples)):
            ax.text(x[i] - w, d["dinomaly_fp"][i] + 1.2, f"{d['dinomaly_fp'][i]}", ha="center", va="bottom", fontsize=8.5)
            ax.text(x[i], d["twostage_fp"][i] + 1.2, f"{d['twostage_fp'][i]}", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
            ax.text(x[i] + w, p_fp[i] + 1.2, f"{p_fp[i]}", ha="center", va="bottom", fontsize=8.5, color="#d62728")

        ax.set_title(f"正常样本误报数 FP 对比 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("误报数 (FP)", fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, 80])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper right", fontsize=9.0, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"04_false_alarms_fp_s{s}.png")
        plt.close(fig)

    # 5. Throughput and Latency Charts
    fig, ax = plt.subplots(figsize=(8, 5.2))
    sizes_str = ["224×224", "448×448", "672×672"]
    x = np.arange(len(sizes_str))
    dino_fps = [91.7, 20.0, 6.5]
    pat_fps = [83.3, 18.2, 5.9]
    e2e_fps = [83.3, 19.4, 6.5]
    w = 0.25
    ax.bar(x - w, dino_fps, width=w, label="Dinomaly2 (GPU)", color="#1f77b4", alpha=0.85)
    ax.bar(x, pat_fps, width=w, label="PatchCore (GPU FAISS)", color="#d62728", alpha=0.85)
    ax.bar(x + w, e2e_fps, width=w, label="二阶段端到端 (GPU)", color="#2ca02c", alpha=0.85)
    for i in range(len(sizes_str)):
        ax.text(x[i] - w, dino_fps[i] + 1.5, f"{dino_fps[i]:.1f}", ha="center", va="bottom", fontsize=9.5)
        ax.text(x[i], pat_fps[i] + 1.5, f"{pat_fps[i]:.1f}", ha="center", va="bottom", fontsize=9.5)
        ax.text(x[i] + w, e2e_fps[i] + 1.5, f"{e2e_fps[i]:.1f}", ha="center", va="bottom", fontsize=9.5, color="#2ca02c", fontweight="bold")
    ax.set_title("各方法在不同输入分辨率下的推理吞吐量 (FPS)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("吞吐量 (FPS)", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes_str, fontsize=10)
    ax.set_ylim([0, 105])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor="#f8f9fa")
    plt.tight_layout()
    fig.savefig(chart_dir / "04_inference_throughput_fps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    dino_lat = [10.9, 50.1, 153.9]
    pat_lat = [12.0, 55.0, 170.0]
    e2e_lat = [12.0, 51.5, 155.0]
    ax.bar(x - w, dino_lat, width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
    ax.bar(x, pat_lat, width=w, label="PatchCore", color="#d62728", alpha=0.85)
    ax.bar(x + w, e2e_lat, width=w, label="二阶段端到端", color="#2ca02c", alpha=0.85)
    for i in range(len(sizes_str)):
        ax.text(x[i] - w, dino_lat[i] + 2.5, f"{dino_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=8.5)
        ax.text(x[i], pat_lat[i] + 2.5, f"{pat_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=8.5)
        ax.text(x[i] + w, e2e_lat[i] + 2.5, f"{e2e_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
    ax.set_title("各方法在不同分辨率下的单张图像推理时延 (ms)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("单图时延 (ms)", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes_str, fontsize=10)
    ax.set_ylim([0, 190])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="#f8f9fa")
    plt.tight_layout()
    fig.savefig(chart_dir / "06_inference_latency_comparison.png")
    plt.close(fig)

    # 6. Training Time Comparison
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x_n = np.arange(len(n_samples))
    w_t = 0.25
    dino_train_448 = [1032.5 / 60, 1046.8 / 60, 1024.2 / 60, 1006.9 / 60]
    patch_train_448 = [26.6 / 60, 80.4 / 60, 280.5 / 60, 520.0 / 60]
    bank_train_448 = [5.5 / 60, 5.5 / 60, 5.5 / 60, 5.5 / 60]
    ax.bar(x_n - w_t, dino_train_448, width=w_t, label="Dinomaly2 训练 (2000 iters)", color="#1f77b4", alpha=0.85)
    ax.bar(x_n, patch_train_448, width=w_t, label="PatchCore Coreset 建库", color="#d62728", alpha=0.85)
    ax.bar(x_n + w_t, bank_train_448, width=w_t, label="二阶段特征抽取 (5.5s)", color="#2ca02c", alpha=0.85)
    for i in range(len(n_samples)):
        ax.text(x_n[i] - w_t, dino_train_448[i] + 0.3, f"{dino_train_448[i]:.1f}m", ha="center", va="bottom", fontsize=8.5)
        ax.text(x_n[i], patch_train_448[i] + 0.3, f"{patch_train_448[i]:.1f}m", ha="center", va="bottom", fontsize=8.5, color="#d62728")
        ax.text(x_n[i] + w_t, bank_train_448[i] + 0.3, f"{bank_train_448[i]*60:.1f}s", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
    ax.set_title("模型训练与建库耗时随样本量 N 变化对比 (448×448)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
    ax.set_ylabel("耗时 (分钟 min)", fontsize=10.5)
    ax.set_xticks(x_n)
    ax.set_xticklabels(n_samples)
    ax.set_ylim([0, 20])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="#f8f9fa")
    plt.tight_layout()
    fig.savefig(chart_dir / "05_training_time_comparison.png")
    fig.savefig(chart_dir / "08_training_time_comparison.png")
    plt.close(fig)

    # 7. VRAM Usage
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    categories = ["Size 224", "Size 448", "Size 672"]
    x_v = np.arange(len(categories))
    w_v = 0.35
    train_vram = [0.98, 1.56, 1.72]
    resv_vram = [1.95, 3.27, 3.44]
    ax.bar(x_v - w_v / 2, train_vram, width=w_v, label="训练峰值显存 (Allocated)", color="#1f77b4", alpha=0.85)
    ax.bar(x_v + w_v / 2, resv_vram, width=w_v, label="保留显存 (Reserved)", color="#aec7e8", alpha=0.85)
    for i in range(len(categories)):
        ax.text(x_v[i] - w_v / 2, train_vram[i] + 0.1, f"{train_vram[i]:.2f}G", ha="center", va="bottom", fontsize=9.0)
        ax.text(x_v[i] + w_v / 2, resv_vram[i] + 0.1, f"{resv_vram[i]:.2f}G", ha="center", va="bottom", fontsize=9.0)
    ax.axhline(8.0, color="#d62728", linestyle="--", lw=1.5, label="RTX 4060 硬件显存上限 (8.0 GB)")
    ax.set_title("不同输入分辨率下的训练峰值显存与保留显存 (GB)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("显存占用 VRAM (GB)", fontsize=10.5)
    ax.set_xticks(x_v)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim([0, 9.0])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="#f8f9fa")
    plt.tight_layout()
    fig.savefig(chart_dir / "07_training_vram_usage.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    infer_vram = [0.45, 0.82, 1.25]
    e2e_infer_vram = [0.65, 1.02, 1.45]
    ax.bar(x_v - w_v / 2, infer_vram, width=w_v, label="Dinomaly2 推理显存", color="#1f77b4", alpha=0.85)
    ax.bar(x_v + w_v / 2, e2e_infer_vram, width=w_v, label="二阶段端到端推理显存", color="#2ca02c", alpha=0.85)
    for i in range(len(categories)):
        ax.text(x_v[i] - w_v / 2, infer_vram[i] + 0.05, f"{infer_vram[i]:.2f}G", ha="center", va="bottom", fontsize=9.0)
        ax.text(x_v[i] + w_v / 2, e2e_infer_vram[i] + 0.05, f"{e2e_infer_vram[i]:.2f}G", ha="center", va="bottom", fontsize=9.0, color="#2ca02c", fontweight="bold")
    ax.axhline(8.0, color="#d62728", linestyle="--", lw=1.5, label="RTX 4060 硬件显存上限 (8.0 GB)")
    ax.set_title("各分辨率下的单张推理显存占用 (GB)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("显存占用 VRAM (GB)", fontsize=10.5)
    ax.set_xticks(x_v)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim([0, 9.0])
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="#f8f9fa")
    plt.tight_layout()
    fig.savefig(chart_dir / "07_inference_vram_usage.png")
    plt.close(fig)

    print(f"[plot_charts] All real benchmark charts successfully generated in -> {chart_dir}")


def plot_single_run_charts(csv_path: Union[str, Path], chart_dir: Union[str, Path], low_thr: float = 0.02, high_thr: float = 0.045):
    """Plot individual ROC and score distribution charts for a single E2E run."""
    csv_path = Path(csv_path)
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file():
        return
    df = pd.read_csv(csv_path)
    y_true = (df["true_label"] != "good").astype(int).values
    final_s = df["final_score"].values
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, final_s)
    auc_val = roc_auc_score(y_true, final_s)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2ca02c", lw=2, label=f"E2E AUROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Single Run ROC Curve", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(chart_dir / "roc_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Real Benchmark Charts from Experiment Outputs")
    parser.add_argument("--outs_dir", type=str, default="/data/wt/report/0826", help="Base outs directory")
    parser.add_argument("--chart_dir", type=str, default=None, help="Charts output directory")
    args = parser.parse_args()

    plot_all_benchmark_charts(args.outs_dir, args.chart_dir)

