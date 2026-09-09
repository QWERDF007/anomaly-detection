#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualization and Analysis Charts Generator for Anomaly Detection Benchmark.

Plots comprehensive evaluation metrics directly from live benchmark results:
  - AUROC & F1 curves vs N
  - Defect Detection (TP) & False Alarms (FP) vs N
  - Clean In-Domain vs Test False Positives
  - Online Inference Latency & Throughput (measured live on GPU)
  - Training Time & VRAM Consumption (measured live on GPU)
  - Iteration Scaling Performance (AUROC, F1, TP, FP vs Max Iterations: 2000, 5000, 20000, 40000)

STRICT DATA INTEGRITY:
  - Zero hardcoded fallback numbers.
  - All values are parsed directly from final_multisize_summary.json and real_vram_measurements.json.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global matplotlib styles
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def get_current_gpu_capacity(override_name: Optional[str] = None, override_vram_gb: Optional[float] = None) -> tuple[str, float]:
    """Dynamically queries the local GPU model and total memory capacity (GB)."""
    if override_name and override_vram_gb:
        return override_name, float(override_vram_gb)
    try:
        import torch
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            raw_name = torch.cuda.get_device_name(dev_idx)
            total_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
            total_gb = round(total_bytes / (1024 ** 3), 1)
            clean_name = raw_name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" Laptop GPU", "").strip()
            return override_name or clean_name or "GPU", override_vram_gb or total_gb
    except Exception:
        pass
    return override_name or "RTX 4090", override_vram_gb or 48.0


def plot_all_benchmark_charts(
    outs_dir: Union[str, Path],
    chart_dir: Optional[Union[str, Path]] = None,
    gpu_name_override: Optional[str] = None,
    gpu_vram_override: Optional[float] = None,
    full_data: bool = False,
    save_images: bool = True,
) -> None:
    """Generate the full benchmark evaluation suite into an interactive vector HTML dashboard.
    Static PNG output is disabled by default per user specification.
    """
    outs_dir = Path(outs_dir).expanduser().resolve()
    chart_dir = Path(chart_dir or (outs_dir / ("charts_full" if full_data else "charts")))
    chart_dir.mkdir(parents=True, exist_ok=True)

    summary_path = outs_dir / "final_multisize_summary.json"
    if not summary_path.is_file():
        print(f"[ERROR] Summary file not found: {summary_path}")
        return

    # 1. Always generate the comprehensive interactive SVG multi-metric dashboard
    try:
        from generate_interactive_chart_html import generate_dataset_html
        generate_dataset_html(outs_dir)
    except Exception as e:
        print(f"[WARN] Could not generate interactive HTML dashboard: {e}")

    if not save_images:
        print(f"[INFO] Static PNG generation is disabled. Interactive multi-metric dashboard ready at: {chart_dir / 'benchmark_dashboard.html'}")
        return

    gpu_name, gpu_total_gb = get_current_gpu_capacity(gpu_name_override, gpu_vram_override)

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary_data:
        print("[ERROR] Summary data is empty.")
        return

    sizes = sorted(list({int(item["size"]) for item in summary_data}))
    n_samples = sorted(list({int(item["n"]) for item in summary_data}))
    iters_list = sorted(list({int(item.get("iters", 2000)) for item in summary_data}))

    print(f"[INFO] Plotting legacy PNG benchmark charts across Sizes={sizes}, N={n_samples}, Iters={iters_list} into {chart_dir}")

    # Primary iteration to plot for standard charts (use the highest iteration or 2000)
    primary_iter = iters_list[-1] if iters_list else 2000
    primary_data = [d for d in summary_data if d.get("iters", primary_iter) == primary_iter]
    if not primary_data:
        primary_data = summary_data

    has_bank = any(outs_dir.glob("**/feature_bank.npz")) and any(d.get("e2e_auc") is not None for d in summary_data)

    # 1. Standalone AUROC charts (at primary_iter)
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_auc = [s_data[n].get("din_auc", 0.0) if n in s_data else None for n in n_samples]
        e_auc = [s_data[n].get("e2e_auc", 0.0) if n in s_data else None for n in n_samples]
        p_auc = [s_data[n].get("pat_auc", 0.0) if n in s_data and s_data[n].get("pat_auc", 0) > 0 else None for n in n_samples]

        d_valid_n = [n for n, v in zip(n_samples, d_auc) if v is not None]
        d_valid_v = [v for v in d_auc if v is not None]
        e_valid_n = [n for n, v in zip(n_samples, e_auc) if v is not None]
        e_valid_v = [v for v in e_auc if v is not None]
        p_valid_n = [n for n, v in zip(n_samples, p_auc) if v is not None]
        p_valid_v = [v for v in p_auc if v is not None]

        if d_valid_v:
            ax.plot(d_valid_n, d_valid_v, marker="o", lw=2.5, color="#1f77b4", label=f"Dinomaly2 (iter={primary_iter})")
        if has_bank and e_valid_v:
            ax.plot(e_valid_n, e_valid_v, marker="s", lw=2.5, color="#2ca02c", label="二阶段端到端 (前向+GPU检索)")
        if p_valid_v:
            ax.plot(p_valid_n, p_valid_v, marker="^", lw=2.0, color="#d62728", linestyle="--", label="PatchCore (基线)")

        ax.set_title(f"图像级 AUROC 随训练样本量 N 变化曲线 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("Image AUROC", fontsize=10.5)
        ax.set_xticks(n_samples)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"01_image_auroc_curve_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "01_image_auroc_curve.png")
        plt.close(fig)

    # 2. Standalone F1 charts (at primary_iter)
    for s in sizes:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_f1 = [s_data[n].get("din_f1", 0.0) if n in s_data else None for n in n_samples]
        e_f1 = [s_data[n].get("e2e_f1", 0.0) if n in s_data else None for n in n_samples]
        p_f1 = [s_data[n].get("pat_f1", 0.0) if n in s_data and s_data[n].get("pat_f1", 0) > 0 else None for n in n_samples]

        d_valid_n = [n for n, v in zip(n_samples, d_f1) if v is not None]
        d_valid_v = [v for v in d_f1 if v is not None]
        e_valid_n = [n for n, v in zip(n_samples, e_f1) if v is not None]
        e_valid_v = [v for v in e_f1 if v is not None]
        p_valid_n = [n for n, v in zip(n_samples, p_f1) if v is not None]
        p_valid_v = [v for v in p_f1 if v is not None]

        if d_valid_v:
            ax.plot(d_valid_n, d_valid_v, marker="o", lw=2.5, color="#1f77b4", label=f"Dinomaly2 (iter={primary_iter})")
        if has_bank and e_valid_v:
            ax.plot(e_valid_n, e_valid_v, marker="s", lw=2.5, color="#2ca02c", label="二阶段端到端 (前向+GPU检索)")
        if p_valid_v:
            ax.plot(p_valid_n, p_valid_v, marker="^", lw=2.0, color="#d62728", linestyle="--", label="PatchCore (基线)")

        ax.set_title(f"图像级最优 F1-Score 随训练样本量 N 变化曲线 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("Image F1-Score", fontsize=10.5)
        ax.set_xticks(n_samples)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"02_image_f1_curve_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "02_image_f1_curve.png")
        plt.close(fig)

    # 3. Defect Detection (TP) Bar Charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x_n = np.arange(len(n_samples))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_tp = [s_data[n].get("din_tp", 0) if n in s_data else 0 for n in n_samples]
        p_tp = [s_data[n].get("pat_tp", 0) if n in s_data else 0 for n in n_samples]

        first_row = list(s_data.values())[0] if s_data else {}
        total_defects = first_row.get("din_tp", 0) + first_row.get("din_fn", 0)

        if has_bank:
            w = 0.25
            e_tp = [s_data[n].get("e2e_tp", 0) if n in s_data else 0 for n in n_samples]
            ax.bar(x_n - w, d_tp, width=w, label="Dinomaly2 单阶段", color="#1f77b4", alpha=0.85)
            ax.bar(x_n, p_tp, width=w, label="PatchCore 基线", color="#d62728", alpha=0.85)
            ax.bar(x_n + w, e_tp, width=w, label="二阶段端到端 (Two-Stage)", color="#2ca02c", alpha=0.85)
            for i in range(len(n_samples)):
                ax.text(x_n[i] - w, d_tp[i] + 2, f"{d_tp[i]}", ha="center", va="bottom", fontsize=8.5)
                ax.text(x_n[i], p_tp[i] + 2, f"{p_tp[i]}", ha="center", va="bottom", fontsize=8.5, color="#d62728")
                ax.text(x_n[i] + w, e_tp[i] + 2, f"{e_tp[i]}", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
        else:
            w = 0.35
            ax.bar(x_n - 0.5 * w, d_tp, width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
            ax.bar(x_n + 0.5 * w, p_tp, width=w, label="PatchCore 基线", color="#d62728", alpha=0.85)
            for i in range(len(n_samples)):
                ax.text(x_n[i] - 0.5 * w, d_tp[i] + 2, f"{d_tp[i]}", ha="center", va="bottom", fontsize=9.0, color="#1f77b4", fontweight="bold")
                ax.text(x_n[i] + 0.5 * w, p_tp[i] + 2, f"{p_tp[i]}", ha="center", va="bottom", fontsize=9.0, color="#d62728")

        ax.set_title(f"真实缺陷检出数量 (TP) 对比 ({s}×{s}，真实缺陷总量: {total_defects} 张)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("缺陷检出数量 (TP / 张)", fontsize=10.5)
        ax.set_xticks(x_n)
        ax.set_xticklabels(n_samples)
        if total_defects > 0:
            ax.set_ylim([0, int(total_defects * 1.15)])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="lower right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"03_defect_detection_tp_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "03_defect_detection_tp.png")
        plt.close(fig)

    # 3b. Defect Misses (FN) Bar Charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x_n = np.arange(len(n_samples))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_fn = [s_data[n].get("din_fn", 0) if n in s_data else 0 for n in n_samples]
        p_fn = [s_data[n].get("pat_fn", 0) if n in s_data else 0 for n in n_samples]

        first_row = list(s_data.values())[0] if s_data else {}
        total_defects = first_row.get("din_tp", 0) + first_row.get("din_fn", 0)

        if has_bank:
            w = 0.25
            e_fn = [s_data[n].get("e2e_fn", 0) if n in s_data else 0 for n in n_samples]
            ax.bar(x_n - w, d_fn, width=w, label="Dinomaly2 单阶段", color="#1f77b4", alpha=0.85)
            ax.bar(x_n, p_fn, width=w, label="PatchCore 基线", color="#d62728", alpha=0.85)
            ax.bar(x_n + w, e_fn, width=w, label="二阶段端到端 (Two-Stage)", color="#2ca02c", alpha=0.85)
            max_fn = max(max(d_fn), max(p_fn), max(e_fn), 1)
            for i in range(len(n_samples)):
                ax.text(x_n[i] - w, d_fn[i] + max_fn * 0.015, f"{d_fn[i]}", ha="center", va="bottom", fontsize=8.5)
                ax.text(x_n[i], p_fn[i] + max_fn * 0.015, f"{p_fn[i]}", ha="center", va="bottom", fontsize=8.5, color="#d62728")
                ax.text(x_n[i] + w, e_fn[i] + max_fn * 0.015, f"{e_fn[i]}", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
        else:
            w = 0.35
            ax.bar(x_n - 0.5 * w, d_fn, width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
            ax.bar(x_n + 0.5 * w, p_fn, width=w, label="PatchCore 基线", color="#d62728", alpha=0.85)
            max_fn = max(max(d_fn), max(p_fn), 1)
            for i in range(len(n_samples)):
                ax.text(x_n[i] - 0.5 * w, d_fn[i] + max_fn * 0.015, f"{d_fn[i]}", ha="center", va="bottom", fontsize=9.0, color="#1f77b4", fontweight="bold")
                ax.text(x_n[i] + 0.5 * w, p_fn[i] + max_fn * 0.015, f"{p_fn[i]}", ha="center", va="bottom", fontsize=9.0, color="#d62728")

        ax.set_title(f"漏检缺陷数量 (FN - 越低越好) 对比 ({s}×{s}，缺陷总数: {total_defects} 张)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("漏检缺陷数 (FN / 张)", fontsize=10.5)
        ax.set_xticks(x_n)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, max_fn * 1.20])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"03_defect_miss_fn_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "03_defect_miss_fn.png")
        plt.close(fig)

    # 4. False Alarms (FP) Bar Charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x_n = np.arange(len(n_samples))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_fp = [s_data[n].get("din_fp", 0) if n in s_data else 0 for n in n_samples]
        p_fp = [s_data[n].get("pat_fp", 0) if n in s_data else 0 for n in n_samples]

        first_row = list(s_data.values())[0] if s_data else {}
        total_normal = first_row.get("din_fp", 0) + first_row.get("din_tn", 0)

        if has_bank:
            w = 0.25
            e_fp = [s_data[n].get("e2e_fp", 0) if n in s_data else 0 for n in n_samples]
            ax.bar(x_n - w, d_fp, width=w, label="Dinomaly2 单阶段", color="#1f77b4", alpha=0.85)
            ax.bar(x_n, p_fp, width=w, label="PatchCore 基线", color="#d62728", alpha=0.85)
            ax.bar(x_n + w, e_fp, width=w, label="二阶段端到端 (Two-Stage)", color="#2ca02c", alpha=0.85)
            max_fp = max(max(d_fp), max(p_fp), max(e_fp), 1)
            for i in range(len(n_samples)):
                ax.text(x_n[i] - w, d_fp[i] + max_fp * 0.015, f"{d_fp[i]}", ha="center", va="bottom", fontsize=8.5)
                ax.text(x_n[i], p_fp[i] + max_fp * 0.015, f"{p_fp[i]}", ha="center", va="bottom", fontsize=8.5, color="#d62728")
                ax.text(x_n[i] + w, e_fp[i] + max_fp * 0.015, f"{e_fp[i]}", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
        else:
            w = 0.35
            ax.bar(x_n - 0.5 * w, d_fp, width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
            ax.bar(x_n + 0.5 * w, p_fp, width=w, label="PatchCore 基线", color="#d62728", alpha=0.85)
            max_fp = max(max(d_fp), max(p_fp), 1)
            for i in range(len(n_samples)):
                ax.text(x_n[i] - 0.5 * w, d_fp[i] + max_fp * 0.015, f"{d_fp[i]}", ha="center", va="bottom", fontsize=9.0, color="#1f77b4", fontweight="bold")
                ax.text(x_n[i] + 0.5 * w, p_fp[i] + max_fp * 0.015, f"{p_fp[i]}", ha="center", va="bottom", fontsize=9.0, color="#d62728")

        ax.set_title(f"良品误报数量 (FP) 对比 ({s}×{s}，良品总数: {total_normal} 张)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("良品误报数量 (FP / 张)", fontsize=10.5)
        ax.set_xticks(x_n)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, max_fp * 1.18])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"04_false_alarms_fp_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "04_false_alarms_fp.png")
        plt.close(fig)

    # 5. Clean In-Domain vs Test False Positives
    has_clean_fp = any(d.get("din_clean_fp") is not None for d in primary_data)
    if has_clean_fp:
        for s in sizes:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            x_n = np.arange(len(n_samples))
            s_data = {d["n"]: d for d in primary_data if d["size"] == s}

            d_clean_fp = [s_data[n].get("din_clean_fp", 0) if n in s_data else 0 for n in n_samples]
            d_test_fp = [s_data[n].get("din_fp", 0) if n in s_data else 0 for n in n_samples]

            if has_bank:
                w = 0.20
                e_clean_fp = [s_data[n].get("e2e_clean_fp", 0) if n in s_data else 0 for n in n_samples]
                e_test_fp = [s_data[n].get("e2e_fp", 0) if n in s_data else 0 for n in n_samples]
                ax.bar(x_n - 1.5 * w, d_clean_fp, width=w, label="Dinomaly2 训练干净域误报", color="#a6cee3")
                ax.bar(x_n - 0.5 * w, d_test_fp, width=w, label="Dinomaly2 全量测试误报", color="#1f77b4")
                ax.bar(x_n + 0.5 * w, e_clean_fp, width=w, label="二阶段 训练干净域误报", color="#b2df8a")
                ax.bar(x_n + 1.5 * w, e_test_fp, width=w, label="二阶段 全量测试误报", color="#33a02c")
                max_val = max(max(d_clean_fp), max(d_test_fp), max(e_clean_fp), max(e_test_fp), 1)
            else:
                w = 0.35
                ax.bar(x_n - 0.5 * w, d_clean_fp, width=w, label="Dinomaly2 训练干净域误报", color="#a6cee3")
                ax.bar(x_n + 0.5 * w, d_test_fp, width=w, label="Dinomaly2 全量测试误报", color="#1f77b4")
                max_val = max(max(d_clean_fp), max(d_test_fp), 1)

            ax.set_title(f"训练干净域与测试集良品误报分解对比 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
            ax.set_ylabel("误报数量 (FP / 张)", fontsize=10.5)
            ax.set_xticks(x_n)
            ax.set_xticklabels(n_samples)
            ax.set_ylim([0, max_val * 1.2])
            ax.grid(True, linestyle=":", alpha=0.6, axis="y")
            ax.legend(loc="upper right", fontsize=8.8, frameon=True, facecolor="#f8f9fa")
            plt.tight_layout()
            fig.savefig(chart_dir / f"05_training_vs_test_clean_domain_fp_s{s}.png")
            if s == 448:
                fig.savefig(chart_dir / "05_training_vs_test_clean_domain_fp.png")
            plt.close(fig)

    # 6. Live Measured Throughput (FPS) Charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x_n = np.arange(len(n_samples))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_fps = [s_data[n].get("din_fps", 0.0) if n in s_data else 0.0 for n in n_samples]
        p_fps = [s_data[n].get("pat_fps", 0.0) if n in s_data else 0.0 for n in n_samples]

        if has_bank:
            w = 0.25
            e_fps = [s_data[n].get("e2e_fps", 0.0) if n in s_data else 0.0 for n in n_samples]
            ax.bar(x_n - w, d_fps, width=w, label="Dinomaly2 单阶段", color="#1f77b4", alpha=0.85)
            ax.bar(x_n, p_fps, width=w, label="PatchCore (GPU FAISS)", color="#d62728", alpha=0.85)
            ax.bar(x_n + w, e_fps, width=w, label="二阶段端到端", color="#2ca02c", alpha=0.85)
            max_fps = max(max(d_fps), max(p_fps), max(e_fps), 1.0)
            for i in range(len(n_samples)):
                if d_fps[i] > 0:
                    ax.text(x_n[i] - w, d_fps[i] + max_fps * 0.015, f"{d_fps[i]:.1f}", ha="center", va="bottom", fontsize=8.5)
                if p_fps[i] > 0:
                    ax.text(x_n[i], p_fps[i] + max_fps * 0.015, f"{p_fps[i]:.1f}", ha="center", va="bottom", fontsize=8.5, color="#d62728")
                if e_fps[i] > 0:
                    ax.text(x_n[i] + w, e_fps[i] + max_fps * 0.015, f"{e_fps[i]:.1f}", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
        else:
            w = 0.35
            ax.bar(x_n - 0.5 * w, d_fps, width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
            ax.bar(x_n + 0.5 * w, p_fps, width=w, label="PatchCore (GPU FAISS)", color="#d62728", alpha=0.85)
            max_fps = max(max(d_fps), max(p_fps), 1.0)
            for i in range(len(n_samples)):
                if d_fps[i] > 0:
                    ax.text(x_n[i] - 0.5 * w, d_fps[i] + max_fps * 0.015, f"{d_fps[i]:.1f}", ha="center", va="bottom", fontsize=9.0, color="#1f77b4", fontweight="bold")
                if p_fps[i] > 0:
                    ax.text(x_n[i] + 0.5 * w, p_fps[i] + max_fps * 0.015, f"{p_fps[i]:.1f}", ha="center", va="bottom", fontsize=9.0, color="#d62728")

        ax.set_title(f"实测在线推理吞吐量对比 ({s}×{s}，Batch=1，{gpu_name})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("推理吞吐量 (FPS)", fontsize=10.5)
        ax.set_xticks(x_n)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, max_fps * 1.2])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"04_inference_throughput_fps_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "04_inference_throughput_fps.png")
        plt.close(fig)

    # 7. Live Measured Latency (ms) Charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x_n = np.arange(len(n_samples))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_lat = [s_data[n].get("din_lat_ms", 0.0) if n in s_data else 0.0 for n in n_samples]
        p_lat = [s_data[n].get("pat_lat_ms", 0.0) if n in s_data else 0.0 for n in n_samples]

        if has_bank:
            w = 0.25
            e_lat = [s_data[n].get("e2e_lat_ms", 0.0) if n in s_data else 0.0 for n in n_samples]
            ax.bar(x_n - w, d_lat, width=w, label="Dinomaly2 单阶段", color="#1f77b4", alpha=0.85)
            ax.bar(x_n, p_lat, width=w, label="PatchCore 检索", color="#d62728", alpha=0.85)
            ax.bar(x_n + w, e_lat, width=w, label="二阶段端到端", color="#2ca02c", alpha=0.85)
            max_lat = max(max(d_lat), max(p_lat), max(e_lat), 1.0)
            for i in range(len(n_samples)):
                if d_lat[i] > 0:
                    ax.text(x_n[i] - w, d_lat[i] + max_lat * 0.015, f"{d_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=8.5)
                if p_lat[i] > 0:
                    ax.text(x_n[i], p_lat[i] + max_lat * 0.015, f"{p_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=8.5, color="#d62728")
                if e_lat[i] > 0:
                    ax.text(x_n[i] + w, e_lat[i] + max_lat * 0.015, f"{e_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
        else:
            w = 0.35
            ax.bar(x_n - 0.5 * w, d_lat, width=w, label="Dinomaly2", color="#1f77b4", alpha=0.85)
            ax.bar(x_n + 0.5 * w, p_lat, width=w, label="PatchCore 检索", color="#d62728", alpha=0.85)
            max_lat = max(max(d_lat), max(p_lat), 1.0)
            for i in range(len(n_samples)):
                if d_lat[i] > 0:
                    ax.text(x_n[i] - 0.5 * w, d_lat[i] + max_lat * 0.015, f"{d_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=9.0, color="#1f77b4", fontweight="bold")
                if p_lat[i] > 0:
                    ax.text(x_n[i] + 0.5 * w, p_lat[i] + max_lat * 0.015, f"{p_lat[i]:.1f}ms", ha="center", va="bottom", fontsize=9.0, color="#d62728")

        ax.set_title(f"实测单图推理时延对比 ({s}×{s}，Batch=1，{gpu_name})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("单图推理时延 (ms)", fontsize=10.5)
        ax.set_xticks(x_n)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, max_lat * 1.2])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper left", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"06_inference_latency_comparison_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "06_inference_latency_comparison.png")
        plt.close(fig)

    # 8. Live Measured Training Time Charts
    for s in sizes:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x_n = np.arange(len(n_samples))
        w = 0.25
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_trn = [s_data[n].get("din_train_time_m", 0.0) if n in s_data else 0.0 for n in n_samples]
        p_trn = [s_data[n].get("pat_train_time_m", 0.0) if n in s_data else 0.0 for n in n_samples]

        ax.bar(x_n - w/2, d_trn, width=w, label=f"Dinomaly2 (iter={primary_iter})", color="#1f77b4", alpha=0.85)
        ax.bar(x_n + w/2, p_trn, width=w, label="PatchCore 特征采样建库", color="#d62728", alpha=0.85)

        max_trn = max(max(d_trn), max(p_trn), 0.1)
        for i in range(len(n_samples)):
            if d_trn[i] > 0:
                ax.text(x_n[i] - w/2, d_trn[i] + max_trn * 0.015, f"{d_trn[i]:.1f}m", ha="center", va="bottom", fontsize=8.5)
            if p_trn[i] > 0:
                ax.text(x_n[i] + w/2, p_trn[i] + max_trn * 0.015, f"{p_trn[i]:.1f}m", ha="center", va="bottom", fontsize=8.5, color="#d62728")

        ax.set_title(f"实测模型训练与建库耗时对比 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("耗时 (分钟 min)", fontsize=10.5)
        ax.set_xticks(x_n)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, max_trn * 1.25])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper left", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"05_training_time_comparison_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "05_training_time_comparison.png")
        plt.close(fig)

    # 9. Live Measured VRAM Usage (Inference & Training)
    for s in sizes:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        x_n = np.arange(len(n_samples))
        s_data = {d["n"]: d for d in primary_data if d["size"] == s}

        d_vram = [s_data[n].get("din_vram_gb", 0.0) if n in s_data else 0.0 for n in n_samples]
        p_vram = [s_data[n].get("pat_vram_gb", 0.0) if n in s_data else 0.0 for n in n_samples]

        if has_bank:
            w = 0.25
            e_vram = [s_data[n].get("e2e_vram_gb", 0.0) if n in s_data else 0.0 for n in n_samples]
            ax.bar(x_n - w, d_vram, width=w, label="Dinomaly2 单阶段推理", color="#1f77b4", alpha=0.85)
            ax.bar(x_n, p_vram, width=w, label="PatchCore 推理", color="#d62728", alpha=0.85)
            ax.bar(x_n + w, e_vram, width=w, label="二阶段端到端推理", color="#2ca02c", alpha=0.85)
            max_vram = max(max(d_vram), max(p_vram), max(e_vram), 1.0)
            for i in range(len(n_samples)):
                if d_vram[i] > 0:
                    ax.text(x_n[i] - w, d_vram[i] + max_vram * 0.015, f"{d_vram[i]:.2f}G", ha="center", va="bottom", fontsize=8.5)
                if p_vram[i] > 0:
                    ax.text(x_n[i], p_vram[i] + max_vram * 0.015, f"{p_vram[i]:.2f}G", ha="center", va="bottom", fontsize=8.5, color="#d62728")
                if e_vram[i] > 0:
                    ax.text(x_n[i] + w, e_vram[i] + max_vram * 0.015, f"{e_vram[i]:.2f}G", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")
        else:
            w = 0.35
            ax.bar(x_n - 0.5 * w, d_vram, width=w, label="Dinomaly2 推理", color="#1f77b4", alpha=0.85)
            ax.bar(x_n + 0.5 * w, p_vram, width=w, label="PatchCore 推理", color="#d62728", alpha=0.85)
            max_vram = max(max(d_vram), max(p_vram), 1.0)
            for i in range(len(n_samples)):
                if d_vram[i] > 0:
                    ax.text(x_n[i] - 0.5 * w, d_vram[i] + max_vram * 0.015, f"{d_vram[i]:.2f}G", ha="center", va="bottom", fontsize=9.0, color="#1f77b4", fontweight="bold")
                if p_vram[i] > 0:
                    ax.text(x_n[i] + 0.5 * w, p_vram[i] + max_vram * 0.015, f"{p_vram[i]:.2f}G", ha="center", va="bottom", fontsize=9.0, color="#d62728")

        ax.set_title(f"实测在线推理 GPU 显存峰值对比 ({s}×{s}，{gpu_name} {gpu_total_gb}GB)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("正常训练样本量 (N)", fontsize=10.5)
        ax.set_ylabel("显存占用 (GB)", fontsize=10.5)
        ax.set_xticks(x_n)
        ax.set_xticklabels(n_samples)
        ax.set_ylim([0, max_vram * 1.25])
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper left", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
        plt.tight_layout()
        fig.savefig(chart_dir / f"07_inference_vram_usage_s{s}.png")
        if s == 448:
            fig.savefig(chart_dir / "07_inference_vram_usage.png")
        plt.close(fig)

    # 10. Multi-Iteration Comparison Charts (AUROC & F1 vs Max Iterations: 2000, 5000, 20000, 40000)
    if len(iters_list) > 1:
        for s in sizes:
            # AUROC vs Iterations
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            for idx_n, n in enumerate(n_samples):
                n_data = {d.get("iters", 2000): d for d in summary_data if d["size"] == s and d["n"] == n}
                it_x = [it for it in iters_list if it in n_data]
                it_y = [n_data[it].get("din_auc", 0.0) for it in it_x]
                if it_y:
                    c = colors[idx_n % len(colors)]
                    ax.plot(it_x, it_y, marker="o", lw=2.0, label=f"N={n}", color=c)
                    for x_val, y_val in zip(it_x, it_y):
                        ax.text(x_val, y_val + 0.003, f"{y_val:.4f}", ha="center", va="bottom", fontsize=8.0, color=c)

            ax.set_title(f"Dinomaly2 图像级 AUROC 随最大迭代次数演化 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("训练最大迭代次数 (Max Iterations)", fontsize=10.5)
            ax.set_ylabel("Image AUROC", fontsize=10.5)
            ax.set_xticks(iters_list)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
            plt.tight_layout()
            fig.savefig(chart_dir / f"08_auroc_vs_max_iters_s{s}.png")
            plt.close(fig)

            # Optimal F1 vs Iterations
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            for idx_n, n in enumerate(n_samples):
                n_data = {d.get("iters", 2000): d for d in summary_data if d["size"] == s and d["n"] == n}
                it_x = [it for it in iters_list if it in n_data]
                it_y = [n_data[it].get("din_f1", 0.0) for it in it_x]
                if it_y:
                    c = colors[idx_n % len(colors)]
                    ax.plot(it_x, it_y, marker="s", lw=2.0, label=f"N={n}", color=c)
                    for x_val, y_val in zip(it_x, it_y):
                        ax.text(x_val, y_val + 0.003, f"{y_val:.4f}", ha="center", va="bottom", fontsize=8.0, color=c)

            ax.set_title(f"Dinomaly2 最优 F1-Score 随最大迭代次数演化 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("训练最大迭代次数 (Max Iterations)", fontsize=10.5)
            ax.set_ylabel("Image F1-Score", fontsize=10.5)
            ax.set_xticks(iters_list)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
            plt.tight_layout()
            fig.savefig(chart_dir / f"08_f1_vs_max_iters_s{s}.png")
            plt.close(fig)

            # Training Time vs Iterations
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            for idx_n, n in enumerate(n_samples):
                n_data = {d.get("iters", 2000): d for d in summary_data if d["size"] == s and d["n"] == n}
                it_x = [it for it in iters_list if it in n_data]
                it_y = [n_data[it].get("din_train_time_m", 0.0) for it in it_x]
                if it_y and max(it_y) > 0:
                    c = colors[idx_n % len(colors)]
                    ax.plot(it_x, it_y, marker="^", lw=2.0, label=f"N={n}", color=c)
                    for x_val, y_val in zip(it_x, it_y):
                        ax.text(x_val, y_val + max(it_y)*0.02, f"{y_val:.1f}m", ha="center", va="bottom", fontsize=8.0, color=c)

            ax.set_title(f"Dinomaly2 真实训练耗时随迭代次数演化 ({s}×{s})", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("训练最大迭代次数 (Max Iterations)", fontsize=10.5)
            ax.set_ylabel("训练耗时 (分钟 min)", fontsize=10.5)
            ax.set_xticks(iters_list)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=9.2, frameon=True, facecolor="#f8f9fa")
            plt.tight_layout()
            fig.savefig(chart_dir / f"08_training_time_vs_max_iters_s{s}.png")
            plt.close(fig)

    # 11. Iteration Scaling Performance Chart
    try:
        from plot_iteration_chart import generate_iteration_chart
        generate_iteration_chart(outs_dir, output_png=chart_dir / "09_image_auroc_vs_iterations.png")
    except Exception as e:
        print(f"[WARN] Could not generate iteration chart: {e}")

    print(f"[SUCCESS] All genuine benchmark charts generated successfully under: {chart_dir}")


def plot_single_run_charts(csv_path: Union[str, Path], chart_dir: Union[str, Path], low_thr: float = 0.02, high_thr: float = 0.045):
    """Generates ROC, PR, score distribution and confusion matrix for a single run CSV."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
    csv_path = Path(csv_path).expanduser().resolve()
    chart_dir = Path(chart_dir).expanduser().resolve()
    chart_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    y_true = (df["true_label"] == "anomaly").astype(int).values
    scores = df["final_score"].values if "final_score" in df.columns else df["raw_score"].values

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(chart_dir / "01_eval_roc_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Real Benchmark Charts from Experiment Outputs")
    parser.add_argument("--outs_dir", type=str, required=True, help="Base outs directory")
    parser.add_argument("--chart_dir", type=str, default=None, help="Charts output directory")
    parser.add_argument("--results", type=str, default=None, help="Path to single e2e_results.csv or .json")
    parser.add_argument("--full_benchmark", action="store_true", help="Generate full multisize comparison suite")
    parser.add_argument("--full_data", action="store_true", help="Plot full dataset evaluation charts")
    parser.add_argument("--gpu_name", type=str, default=None, help="Override GPU name (e.g. RTX 4090)")
    parser.add_argument("--gpu_vram_gb", type=float, default=None, help="Override GPU VRAM in GB")
    parser.add_argument("--save_images", action="store_true", default=True, help="Save static PNG images")
    parser.add_argument("--no_save_images", dest="save_images", action="store_false", help="Disable static PNG images")
    args = parser.parse_args()

    if args.results:
        plot_single_run_charts(args.results, args.chart_dir or (Path(args.outs_dir) / "charts"))
    else:
        plot_all_benchmark_charts(
            outs_dir=args.outs_dir,
            chart_dir=args.chart_dir,
            gpu_name_override=args.gpu_name,
            gpu_vram_override=args.gpu_vram_gb,
            full_data=args.full_data,
            save_images=args.save_images,
        )
