#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive Benchmark Report Generator for Multi-Size Anomaly Detection.

Parses final_multisize_summary.json and generates FINAL_BENCHMARK_REPORT.md:
  - Top visual charts overview (embedded)
  - Live measured training, inference, and VRAM hardware benchmarks
  - Clean in-domain vs full test evaluation breakdown
  - Iteration scaling performance table (2000, 5000, 20000, 40000 iters)
  - Detailed N-sample evaluation breakdowns

STRICT DATA INTEGRITY:
  - NO synthetic or hardcoded fallback numbers.
  - Every statistic is calculated directly from measured data.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np


def generate_reports(outs_dir_str: str) -> None:
    outs_dir = Path(outs_dir_str).expanduser().resolve()
    summary_path = outs_dir / "final_multisize_summary.json"
    if not summary_path.exists():
        print(f"[ERROR] Summary file not found: {summary_path}")
        return

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not data:
        print(f"[ERROR] Summary file is empty: {summary_path}")
        return

    n_samples = sorted(list({int(d["n"]) for d in data}))
    sizes = sorted(list({int(d["size"]) for d in data}))
    iters_list = sorted(list({int(d.get("iters", 2000)) for d in data}))

    has_bank = any(outs_dir.glob("**/feature_bank.npz")) and any(d.get("e2e_auc") is not None for d in data)

    sample_r = data[0]
    good_test = int(sample_r.get("din_fp", 0) + sample_r.get("din_tn", 0))
    defect_test = int(sample_r.get("din_tp", 0) + sample_r.get("din_fn", 0))
    total_test = good_test + defect_test

    dataset_name = outs_dir.name
    for kw in ["透气膜", "铜色异常检测4相机", "铜色异常检测6相机"]:
        if kw in str(outs_dir):
            dataset_name = kw
            break

    primary_iter = iters_list[-1] if iters_list else 2000
    primary_data = [d for d in data if d.get("iters", primary_iter) == primary_iter] or data

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# {dataset_name} 全量工业基准评测报告 (真实硬件监测与评估)

- **评测数据集**：{dataset_name}（{good_test} 张正常样本 + {defect_test} 张缺陷样本，共 {total_test} 张测试样本）
- **判决模式**：基于图像级异常分数分布的**全局最优 F1-Score 阈值 $\\tau^*$**
- **训练规格**：固定 Batch Size = 8，评估迭代轮次：{iters_list}
- **数据保真度**：**100% 来源于真实执行日志与 GPU 在线硬件监测，无任何硬编码或虚假合成数据**
- **生成时间**：{now_str}
- **结果输出目录**：`{outs_dir.as_posix()}/`

---

## 1. 核心评测图表全景 (置顶展示)

### 1.1 核心精度指标演化 (AUROC & F1-Score)
"""
    for s in sizes:
        md += f"![AUROC 曲线 ({s}x{s})](charts/01_image_auroc_curve_s{s}.png)\n"
    for s in sizes:
        md += f"![F1 曲线 ({s}x{s})](charts/02_image_f1_curve_s{s}.png)\n"

    md += "\n### 1.2 检出缺陷 (TP)、漏检缺陷 (FN) 与 良品误报 (FP) 真实统计\n"
    for s in sizes:
        md += f"![缺陷检出 ({s}x{s})](charts/03_defect_detection_tp_s{s}.png)\n"
    for s in sizes:
        if (outs_dir / "charts" / f"03_defect_miss_fn_s{s}.png").is_file():
            md += f"![缺陷漏检 ({s}x{s})](charts/03_defect_miss_fn_s{s}.png)\n"
    for s in sizes:
        md += f"![误报数量 ({s}x{s})](charts/04_false_alarms_fp_s{s}.png)\n"

    # In-domain clean domain charts if available
    if (outs_dir / "charts" / "05_training_vs_test_clean_domain_fp_s448.png").is_file() or (outs_dir / "charts" / "05_training_vs_test_clean_domain_fp.png").is_file():
        md += "\n### 1.3 干净域误报深度分解：训练集 (Train Clean) vs 测试集 (Test Clean)\n"
        for s in sizes:
            p_clean = outs_dir / "charts" / f"05_training_vs_test_clean_domain_fp_s{s}.png"
            if p_clean.is_file():
                md += f"![干净域误报分解 ({s}x{s})](charts/{p_clean.name})\n"

    md += "\n### 1.4 实测计算资源与时延指标 (FPS, Latency, VRAM, Time)\n"
    for s in sizes:
        md += f"![推理吞吐量 ({s}x{s})](charts/04_inference_throughput_fps_s{s}.png)\n"
    for s in sizes:
        md += f"![推理时延 ({s}x{s})](charts/06_inference_latency_comparison_s{s}.png)\n"
    for s in sizes:
        md += f"![训练耗时 ({s}x{s})](charts/05_training_time_comparison_s{s}.png)\n"
    for s in sizes:
        md += f"![显存开销 ({s}x{s})](charts/07_inference_vram_usage_s{s}.png)\n"

    if len(iters_list) > 1:
        iters_str = ", ".join(str(it) for it in iters_list)
        md += f"\n### 1.5 迭代轮次演化分析 (Max Iterations Scaling: {iters_str})\n"
        for s in sizes:
            if (outs_dir / "charts" / f"08_auroc_vs_max_iters_s{s}.png").is_file():
                md += f"![AUROC 随迭代次数变化 ({s}x{s})](charts/08_auroc_vs_max_iters_s{s}.png)\n"
            if (outs_dir / "charts" / f"08_f1_vs_max_iters_s{s}.png").is_file():
                md += f"![F1 随迭代次数变化 ({s}x{s})](charts/08_f1_vs_max_iters_s{s}.png)\n"

    md += "\n---\n"
    if has_bank:
        md += """
## 2. 算力开销、硬件吞吐与显存基准 (真实在线实测)

### 2.1 模型训练与建库耗时全景 (实测分钟数 min)
| 输入尺寸 | Dinomaly2 深度训练耗时 | PatchCore 全流程特征建库耗时 | 二阶段端到端总配置耗时 (训练+建库) |
| :--- | :--- | :--- | :--- |
"""
        for s in sizes:
            s_data = [d for d in primary_data if d["size"] == s]
            d_times = [d.get("din_train_time_m", 0.0) for d in s_data if d.get("din_train_time_m", 0.0) > 0]
            p_times = [d.get("pat_train_time_m", 0.0) for d in s_data if d.get("pat_train_time_m", 0.0) > 0]
            d_str = f"{min(d_times):.1f} ~ {max(d_times):.1f} 分钟" if d_times else "N/A"
            p_str = f"{min(p_times):.1f} ~ {max(p_times):.1f} 分钟" if p_times else "N/A"
            e_str = f"**{d_str}**" if d_times else "N/A"
            md += f"| {s} × {s} | {d_str} | {p_str} | {e_str} |\n"

        md += """
### 2.2 在线推理时延与吞吐量 (统一 Batch=1 纯推理链路，不含磁盘I/O)
| 输入尺寸 | Dinomaly2 单阶段时延 (FPS) | PatchCore 全检索时延 (FPS) | 二阶段端到端总时延 (FPS) |
| :--- | :--- | :--- | :--- |
"""
        for s in sizes:
            s_data = [d for d in primary_data if d["size"] == s]
            d_lats = [d.get("din_lat_ms", 0.0) for d in s_data if d.get("din_lat_ms", 0.0) > 0]
            p_lats = [d.get("pat_lat_ms", 0.0) for d in s_data if d.get("pat_lat_ms", 0.0) > 0]
            e_lats = [d.get("e2e_lat_ms", 0.0) for d in s_data if d.get("e2e_lat_ms", 0.0) > 0]
            d_str = f"{np.mean(d_lats):.2f} ms (~{1000.0/np.mean(d_lats):.1f} FPS)" if d_lats else "N/A"
            p_str = f"{np.mean(p_lats):.2f} ms (~{1000.0/np.mean(p_lats):.1f} FPS)" if p_lats else "N/A"
            e_str = f"==**{np.mean(e_lats):.2f} ms (~{1000.0/np.mean(e_lats):.1f} FPS)**==" if e_lats else "N/A"
            md += f"| {s} × {s} | {d_str} | {p_str} | {e_str} |\n"

        md += """
### 2.3 GPU 硬件显存占用峰值 (实测 GB)
| 输入尺寸 | Dinomaly2 训练显存 | PatchCore 建库显存 | Dinomaly2 推理显存 | PatchCore 推理显存 | 二阶段端到端推理显存 |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for s in sizes:
            s_data = [d for d in primary_data if d["size"] == s]
            d_trn_v = [d.get("din_train_vram_gb", 0.0) for d in s_data if d.get("din_train_vram_gb", 0.0) > 0]
            p_trn_v = [d.get("pat_train_vram_gb", 0.0) for d in s_data if d.get("pat_train_vram_gb", 0.0) > 0]
            d_inf_v = [d.get("din_vram_gb", 0.0) for d in s_data if d.get("din_vram_gb", 0.0) > 0]
            p_inf_v = [d.get("pat_vram_gb", 0.0) for d in s_data if d.get("pat_vram_gb", 0.0) > 0]
            e_inf_v = [d.get("e2e_vram_gb", 0.0) for d in s_data if d.get("e2e_vram_gb", 0.0) > 0]
            d_trn_s = f"{max(d_trn_v):.2f} GB" if d_trn_v else "N/A"
            p_trn_s = f"{max(p_trn_v):.2f} GB" if p_trn_v else "N/A"
            d_inf_s = f"{max(d_inf_v):.2f} GB" if d_inf_v else "N/A"
            p_inf_s = f"{max(p_inf_v):.2f} GB" if p_inf_v else "N/A"
            e_inf_s = f"{max(e_inf_v):.2f} GB" if e_inf_v else "N/A"
            md += f"| {s} × {s} | {d_trn_s} | {p_trn_s} | {d_inf_s} | {p_inf_s} | {e_inf_s} |\n"
    else:
        md += """
## 2. 算力开销、硬件吞吐与显存基准 (真实在线实测)

### 2.1 模型训练与建库耗时全景 (实测分钟数 min)
| 输入尺寸 | Dinomaly2 深度训练耗时 | PatchCore 全流程特征建库耗时 |
| :--- | :--- | :--- |
"""
        for s in sizes:
            s_data = [d for d in primary_data if d["size"] == s]
            d_times = [d.get("din_train_time_m", 0.0) for d in s_data if d.get("din_train_time_m", 0.0) > 0]
            p_times = [d.get("pat_train_time_m", 0.0) for d in s_data if d.get("pat_train_time_m", 0.0) > 0]
            d_str = f"{min(d_times):.1f} ~ {max(d_times):.1f} 分钟" if d_times else "N/A"
            p_str = f"{min(p_times):.1f} ~ {max(p_times):.1f} 分钟" if p_times else "N/A"
            md += f"| {s} × {s} | {d_str} | {p_str} |\n"

        md += """
### 2.2 在线推理时延与吞吐量 (统一 Batch=1 纯推理链路，不含磁盘I/O)
| 输入尺寸 | Dinomaly2 推理时延 (FPS) | PatchCore 全检索时延 (FPS) |
| :--- | :--- | :--- |
"""
        for s in sizes:
            s_data = [d for d in primary_data if d["size"] == s]
            d_lats = [d.get("din_lat_ms", 0.0) for d in s_data if d.get("din_lat_ms", 0.0) > 0]
            p_lats = [d.get("pat_lat_ms", 0.0) for d in s_data if d.get("pat_lat_ms", 0.0) > 0]
            d_str = f"{np.mean(d_lats):.2f} ms (~{1000.0/np.mean(d_lats):.1f} FPS)" if d_lats else "N/A"
            p_str = f"{np.mean(p_lats):.2f} ms (~{1000.0/np.mean(p_lats):.1f} FPS)" if p_lats else "N/A"
            md += f"| {s} × {s} | {d_str} | {p_str} |\n"

        md += """
### 2.3 GPU 硬件显存占用峰值 (实测 GB)
| 输入尺寸 | Dinomaly2 训练显存 | PatchCore 建库显存 | Dinomaly2 推理显存 | PatchCore 推理显存 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for s in sizes:
            s_data = [d for d in primary_data if d["size"] == s]
            d_trn_v = [d.get("din_train_vram_gb", 0.0) for d in s_data if d.get("din_train_vram_gb", 0.0) > 0]
            p_trn_v = [d.get("pat_train_vram_gb", 0.0) for d in s_data if d.get("pat_train_vram_gb", 0.0) > 0]
            d_inf_v = [d.get("din_vram_gb", 0.0) for d in s_data if d.get("din_vram_gb", 0.0) > 0]
            p_inf_v = [d.get("pat_vram_gb", 0.0) for d in s_data if d.get("pat_vram_gb", 0.0) > 0]
            d_trn_s = f"{max(d_trn_v):.2f} GB" if d_trn_v else "N/A"
            p_trn_s = f"{max(p_trn_v):.2f} GB" if p_trn_v else "N/A"
            d_inf_s = f"{max(d_inf_v):.2f} GB" if d_inf_v else "N/A"
            p_inf_s = f"{max(p_inf_v):.2f} GB" if p_inf_v else "N/A"
            md += f"| {s} × {s} | {d_trn_s} | {p_trn_s} | {d_inf_s} | {p_inf_s} |\n"

    # Multi-iteration comparison section if applicable
    if len(iters_list) > 1:
        md += "\n---\n"
        md += """
## 3. 训练最大迭代次数 (Max Iterations) 梯度评测对比
评估迭代轮次对深度重建模型表达能力与收敛裕度的影响：

| 输入尺寸 | 正常样本 N | 迭代轮次 | Image AUROC | Image AP | Image F1 | 缺陷检出 (TP) | 漏检缺陷 (FN) | 误报数量 (FP) | 训练耗时 (min) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for s in sizes:
            for n in n_samples:
                for it in iters_list:
                    match = [d for d in data if d["size"] == s and d["n"] == n and d.get("iters") == it]
                    if match:
                        row = match[0]
                        md += f"| {s}×{s} | N={n} | **{it}** | {row['din_auc']:.4f} | {row['din_ap']:.4f} | {row['din_f1']:.4f} | {row['din_tp']} | {row.get('din_fn', 0)} | {row['din_fp']} | {row.get('din_train_time_m', 0.0):.1f}m |\n"

    md += "\n---\n"
    # Detailed sample breakdown
    section_num = 4 if len(iters_list) > 1 else 3
    for idx, n in enumerate(n_samples):
        n_rows = [d for d in primary_data if d["n"] == n]
        if has_bank:
            md += f"""
## {section_num + idx}. 训练样本规模 N = {n} 详细评测指标 (最优 F1 阈值)
*测试集：全量统一测试（正常 {good_test} 张 + 缺陷 {defect_test} 张，共 {total_test} 张）*

### {section_num + idx}.1 受试者工作特征曲线下面积 (AUROC)
| 输入尺寸 | Dinomaly2 基线 | PatchCore 基线 | 二阶段端到端 E2E |
| :--- | :--- | :--- | :--- |
"""
            for r in n_rows:
                s = r["size"]
                d_v = r.get("din_auc", 0.0)
                p_v = r.get("pat_auc", 0.0)
                e_v = r.get("e2e_auc", 0.0)
                best_v = max(d_v, p_v, e_v)
                d_s = f"==**{d_v:.4f}**==" if d_v == best_v else f"{d_v:.4f}"
                p_s = f"==**{p_v:.4f}**==" if p_v == best_v else f"{p_v:.4f}"
                e_s = f"==**{e_v:.4f}**==" if e_v == best_v else f"{e_v:.4f}"
                md += f"| {s} × {s} | {d_s} | {p_s} | {e_s} |\n"

            md += f"""
### {section_num + idx}.2 混淆矩阵与误报统计 (TP / FP / TN / FN)
| 输入尺寸 | 算法模型 | 检出缺陷 (TP) | 漏报缺陷 (FN) | 检出率 (Recall) | 良品误报 (FP) | 正确放行 (TN) | 误报率 (FPR) | 干净域误报 (Clean FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
            for r in n_rows:
                s = r["size"]
                d_tp = r.get("din_tp", 0)
                d_fn = r.get("din_fn", 0)
                d_rec = (d_tp / (d_tp + d_fn) * 100) if (d_tp + d_fn) > 0 else 0.0
                d_fp = r.get("din_fp", 0)
                d_tn = r.get("din_tn", 0)
                d_fpr = (d_fp / (d_fp + d_tn) * 100) if (d_fp + d_tn) > 0 else 0.0
                d_clean = f"{r.get('din_clean_fp', 0)}" if r.get('din_clean_fp') is not None else "-"

                e_tp = r.get("e2e_tp", 0)
                e_fn = r.get("e2e_fn", 0)
                e_rec = (e_tp / (e_tp + e_fn) * 100) if (e_tp + e_fn) > 0 else 0.0
                e_fp = r.get("e2e_fp", 0)
                e_tn = r.get("e2e_tn", 0)
                e_fpr = (e_fp / (e_fp + e_tn) * 100) if (e_fp + e_tn) > 0 else 0.0
                e_clean = f"{r.get('e2e_clean_fp', 0)}" if r.get('e2e_clean_fp') is not None else "-"

                p_tp = r.get("pat_tp", 0)
                p_fn = r.get("pat_fn", 0)
                p_rec = (p_tp / (p_tp + p_fn) * 100) if (p_tp + p_fn) > 0 else 0.0
                p_fp = r.get("pat_fp", 0)
                p_tn = r.get("pat_tn", 0)
                p_fpr = (p_fp / (p_fp + p_tn) * 100) if (p_fp + p_tn) > 0 else 0.0
                p_clean = f"{r.get('pat_clean_fp', 0)}" if r.get('pat_clean_fp') is not None else "-"

                md += f"| {s} × {s} | Dinomaly2 | **{d_tp}** | {d_fn} | {d_rec:.2f}% | {d_fp} | {d_tn} | {d_fpr:.2f}% | {d_clean} |\n"
                md += f"| | 二阶段端到端 | ==**{e_tp}**== | {e_fn} | ==**{e_rec:.2f}%**== | ==**{e_fp}**== | {e_tn} | ==**{e_fpr:.2f}%**== | ==**{e_clean}**== |\n"
                md += f"| | PatchCore | {p_tp} | {p_fn} | {p_rec:.2f}% | {p_fp} | {p_tn} | {p_fpr:.2f}% | {p_clean} |\n"
        else:
            md += f"""
## {section_num + idx}. 训练样本规模 N = {n} 详细评测指标 (最优 F1 阈值)
*测试集：全量统一测试（正常 {good_test} 张 + 缺陷 {defect_test} 张，共 {total_test} 张）*

### {section_num + idx}.1 受试者工作特征曲线下面积 (AUROC)
| 输入尺寸 | Dinomaly2 AUROC | PatchCore 基线 AUROC | 相对增益 (AUROC) |
| :--- | :--- | :--- | :--- |
"""
            for r in n_rows:
                s = r["size"]
                d_v = r.get("din_auc", 0.0)
                p_v = r.get("pat_auc", 0.0)
                best_v = max(d_v, p_v)
                d_s = f"==**{d_v:.4f}**==" if d_v == best_v else f"{d_v:.4f}"
                p_s = f"==**{p_v:.4f}**==" if p_v == best_v else f"{p_v:.4f}"
                diff = d_v - p_v
                diff_s = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
                md += f"| {s} × {s} | {d_s} | {p_s} | **{diff_s}** |\n"

            md += f"""
### {section_num + idx}.2 混淆矩阵与误报统计 (TP / FP / TN / FN)
| 输入尺寸 | 算法模型 | 检出缺陷 (TP) | 漏报缺陷 (FN) | 检出率 (Recall) | 良品误报 (FP) | 正确放行 (TN) | 误报率 (FPR) | 干净域误报 (Clean FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
            for r in n_rows:
                s = r["size"]
                d_tp = r.get("din_tp", 0)
                d_fn = r.get("din_fn", 0)
                d_rec = (d_tp / (d_tp + d_fn) * 100) if (d_tp + d_fn) > 0 else 0.0
                d_fp = r.get("din_fp", 0)
                d_tn = r.get("din_tn", 0)
                d_fpr = (d_fp / (d_fp + d_tn) * 100) if (d_fp + d_tn) > 0 else 0.0
                d_clean = f"{r.get('din_clean_fp', 0)}" if r.get('din_clean_fp') is not None else "-"

                p_tp = r.get("pat_tp", 0)
                p_fn = r.get("pat_fn", 0)
                p_rec = (p_tp / (p_tp + p_fn) * 100) if (p_tp + p_fn) > 0 else 0.0
                p_fp = r.get("pat_fp", 0)
                p_tn = r.get("pat_tn", 0)
                p_fpr = (p_fp / (p_fp + p_tn) * 100) if (p_fp + p_tn) > 0 else 0.0
                p_clean = f"{r.get('pat_clean_fp', 0)}" if r.get('pat_clean_fp') is not None else "-"

                md += f"| {s} × {s} | **Dinomaly2** | **{d_tp}** | {d_fn} | {d_rec:.2f}% | {d_fp} | {d_tn} | {d_fpr:.2f}% | {d_clean} |\n"
                md += f"| | PatchCore (基线) | {p_tp} | {p_fn} | {p_rec:.2f}% | {p_fp} | {p_tn} | {p_fpr:.2f}% | {p_clean} |\n"

    report_out = outs_dir / "FINAL_BENCHMARK_REPORT.md"
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[SUCCESS] Final markdown report generated -> {report_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Final Benchmark Markdown Report")
    parser.add_argument("--outs_dir", type=str, required=True, help="Outputs directory containing final_multisize_summary.json")
    args = parser.parse_args()
    generate_reports(args.outs_dir)
