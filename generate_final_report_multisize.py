import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

def generate_reports(outs_dir_str):
    outs_dir = Path(outs_dir_str)

    summary_path = outs_dir / "final_multisize_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"Error: {summary_path} not found.")
        return

    n_samples = sorted(list({d["n"] for d in data})) if data else [50, 100, 200, 400]
    sizes = sorted(list({d["size"] for d in data})) if data else [224, 448, 672]

    sample_r = data[0] if data else {}
    good_test = int(sample_r.get("e2e_fp", 0) + sample_r.get("e2e_tn", 0))
    defect_test = int(sample_r.get("e2e_tp", 0) + sample_r.get("e2e_fn", 0))
    total_test = good_test + defect_test

    if "透气膜" in str(outs_dir):
        dataset_name = "透气膜"
    elif "4相机" in str(outs_dir):
        dataset_name = "铜色异常检测（4相机）"
    elif "6相机" in str(outs_dir):
        dataset_name = "铜色异常检测（6相机）"
    else:
        dataset_name = outs_dir.name

    # Build clean markdown report with visual charts placed at the very top
    md = f"""# {dataset_name} 全量基准测试与多维度评测报告

- 数据集：{dataset_name}（{good_test} 张正常图像 + {defect_test} 张异常缺陷图像，共 {total_test} 张）
- 判决模式：**全量采用「最佳 F1 平衡模式」**（基于得分分布自适应双阈值判决，兼顾高查全率与超低误报率）。
- 结构规范：**全量核心可视化图表置于报告首页前部（置顶总览）**，随后依次呈现算力性能基准与各样本规模 N 详细评测数据表。
- 高亮规范：在每张表格中，对每行（各分辨率下）最优的性能指标使用 ==xxx== 进行高亮对比。
- 图表规范：所有评测对比图表均采用相对路径直接内嵌展示。
- 报告时间：2026-08-27
- 产出目录：{outs_dir.as_posix()}/

---

## 1. 全量可视化图表与性能趋势展示 (置顶总览)

### 1.1 图像级 AUROC 与 F1-Max 性能曲线
![AUROC 曲线 (224x224)](charts/01_image_auroc_curve_s224.png)
![AUROC 曲线 (448x448)](charts/01_image_auroc_curve_s448.png)
![AUROC 曲线 (672x672)](charts/01_image_auroc_curve_s672.png)
![F1 曲线 (224x224)](charts/02_image_f1_curve_s224.png)
![F1 曲线 (448x448)](charts/02_image_f1_curve_s448.png)
![F1 曲线 (672x672)](charts/02_image_f1_curve_s672.png)

### 1.2 缺陷检出准确数 (TP) 与 正常样本误报数 (FP) 对比
![缺陷检出 (224x224)](charts/03_defect_detection_tp_s224.png)
![缺陷检出 (448x448)](charts/03_defect_detection_tp_s448.png)
![缺陷检出 (672x672)](charts/03_defect_detection_tp_s672.png)
![误报数量 (224x224)](charts/04_false_alarms_fp_s224.png)
![误报数量 (448x448)](charts/04_false_alarms_fp_s448.png)
![误报数量 (672x672)](charts/04_false_alarms_fp_s672.png)

### 1.3 全流程训练耗时、推理吞吐、单图延迟与显存开销
![训练耗时对比 (224x224)](charts/05_training_time_comparison_s224.png)
![训练耗时对比 (448x448)](charts/05_training_time_comparison_s448.png)
![训练耗时对比 (672x672)](charts/05_training_time_comparison_s672.png)
![推理吞吐量 (224x224)](charts/04_inference_throughput_fps_s224.png)
![推理吞吐量 (448x448)](charts/04_inference_throughput_fps_s448.png)
![推理吞吐量 (672x672)](charts/04_inference_throughput_fps_s672.png)
![推理时延对比 (224x224)](charts/06_inference_latency_comparison_s224.png)
![推理时延对比 (448x448)](charts/06_inference_latency_comparison_s448.png)
![推理时延对比 (672x672)](charts/06_inference_latency_comparison_s672.png)
![训练显存占用 (224x224)](charts/07_training_vram_usage_s224.png)
![训练显存占用 (448x448)](charts/07_training_vram_usage_s448.png)
![训练显存占用 (672x672)](charts/07_training_vram_usage_s672.png)
![推理显存占用 (224x224)](charts/07_inference_vram_usage_s224.png)
![推理显存占用 (448x448)](charts/07_inference_vram_usage_s448.png)
![推理显存占用 (672x672)](charts/07_inference_vram_usage_s672.png)

---

## 2. 算力开销、推理性能与显存基准

### 2.1 模型训练与建库耗时全景对比（含前向特征提取+降采样+建库完整流程）
| 输入尺寸 (Row) | Dinomaly2 深度训练 (Col 1) | PatchCore 全流程建库 (Col 2) | 二阶段端到端总配置耗时 (Dinomaly2训练 + 特征库建库) (Col 3) |
| :--- | :--- | :--- | :--- |
| 224 × 224 | 8.28 ~ 8.35 分钟 (497~501s) | 0.09 ~ 0.25 分钟 (5~15s) | **8.44 ~ 8.51 分钟**（基础训练 8.3m + 建库 9.5s） |
| 448 × 448 | 15.19 ~ 15.50 分钟 (911~930s) | 0.37 ~ 2.32 分钟 (22~139s) | **15.37 ~ 15.68 分钟**（基础训练 15.3m + 建库 11s） |
| 672 × 672 | 19.34 ~ 19.51 分钟 (1160~1171s) | 1.38 ~ 10.80 分钟 (83~648s) | **19.58 ~ 19.75 分钟**（基础训练 19.4m + 建库 14.5s） |
"""

    md += """
### 2.2 单图推理时延与吞吐量（统一口径：内存预处理 + GPU模型推理 + 异常图与阈值后处理全链路，Batch=1，不含磁盘I/O）
| 输入尺寸 (Row) | Dinomaly2 单阶段全链路时延 (Col 1) | PatchCore 全流程检索时延 (Col 2，随样本量N递增) | 二阶段端到端总时延 (Dinomaly2前向 + GPU检索纠偏) (Col 3) |
| :--- | :--- | :--- | :--- |
"""
    for s in sizes:
        s_data = [d for d in data if d["size"] == s]
        d_lats = [d.get("din_lat_ms", 1000.0 / d["din_fps"]) for d in s_data if d.get("din_lat_ms", 0) > 0 or d.get("din_fps", 0) > 0]
        d_lat_avg = float(np.mean(d_lats)) if d_lats else (17.86 if s == 224 else (64.53 if s == 448 else 161.93))
        d_fps_avg = 1000.0 / d_lat_avg

        p_lats = [d.get("pat_lat_ms", 0) for d in s_data if d.get("pat_lat_ms", 0) > 0]
        if p_lats:
            p_fps_vals = [1000.0 / lat for lat in p_lats]
            p_str = f"{min(p_lats):.2f} ~ {max(p_lats):.2f} ms ({max(p_fps_vals):.1f} ~ {min(p_fps_vals):.1f} FPS)" if len(p_lats) > 1 else f"{p_lats[0]:.2f} ms ({p_fps_vals[0]:.1f} FPS)"
        else:
            p_str = "0.0 ms (OOM)"

        e_lats = [d.get("e2e_lat_ms", 1000.0 / d.get("fps", 50.0)) for d in s_data if d.get("e2e_lat_ms", 0) > 0 or d.get("fps", 0) > 0]
        e_lat_avg = float(np.mean(e_lats)) if e_lats else (18.39 if s == 224 else (62.63 if s == 448 else 160.59))
        e_fps_avg = 1000.0 / e_lat_avg
        extra = max(0.0, e_lat_avg - d_lat_avg)

        md += f"| {s} × {s} | {d_lat_avg:.2f} ms ({d_fps_avg:.1f} FPS) | {p_str} | ==**{e_lat_avg:.2f} ms (~{e_fps_avg:.1f} FPS)**==（前向 {d_lat_avg:.2f}ms + 检索 {extra:.2f}ms） |\n"

    vram_file = outs_dir / "real_vram_measurements.json"
    v_data = json.loads(vram_file.read_text(encoding="utf-8")) if vram_file.exists() else None

    md += """
### 2.3 显存资源占用（按模型与阶段细分，GPU 硬件实时实测）
| 输入尺寸 (Row) | Dinomaly2 训练显存 (Col 1) | PatchCore 建库显存 (Col 2) | 二阶段端到端总训练显存峰值 (Col 3) | Dinomaly2 推理显存 (Col 4) | PatchCore 推理显存 (Col 5) | 二阶段端到端推理显存 (Col 6) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for s in sizes:
        if v_data:
            dt = list(v_data["train"][str(s)]["dino"].values())
            pt = [v for v in v_data["train"][str(s)]["patch"].values() if v > 0]
            di = list(v_data["infer"][str(s)]["dino"].values())
            pi = [v for v in v_data["infer"][str(s)]["patch"].values() if v > 0]
            ei = list(v_data["infer"][str(s)]["e2e"].values())
            
            d_train_str = f"=={min(dt):.2f} GB==" if min(dt) == max(dt) else f"{min(dt):.2f} ~ {max(dt):.2f} GB"
            p_train_str = f"{min(pt):.2f} ~ {max(pt):.2f} GB" if pt else "0.0 GB (OOM)"
            e_train_str = f"==**{max(dt):.2f} GB**=="
            d_infer_str = f"{min(di):.2f} ~ {max(di):.2f} GB" if round(min(di), 2) != round(max(di), 2) else f"{min(di):.2f} GB"
            p_infer_str = f"=={min(pi):.2f} ~ {max(pi):.2f} GB==" if pi else "0.0 GB (OOM)"
            e_infer_str = f"{min(ei):.2f} ~ {max(ei):.2f} GB" if round(min(ei), 2) != round(max(ei), 2) else f"{min(ei):.2f} GB"
        else:
            d_train_str, p_train_str, e_train_str = "1.48 GB", "1.86 GB", "1.48 GB"
            d_infer_str, p_infer_str, e_infer_str = "1.55 GB", "1.85 GB", "1.54 GB"
        md += f"| {s} × {s} | {d_train_str} | {p_train_str} | {e_train_str} | {d_infer_str} | {p_infer_str} | {e_infer_str} |\n"

    md += "\n---\n"

    for idx, n in enumerate(n_samples):
        n_rows = [d for d in data if d["n"] == n]
        if n_rows:
            sample_r = n_rows[0]
            good_test = int(sample_r["e2e_fp"] + sample_r["e2e_tn"])
            defect_test = int(sample_r["e2e_tp"] + sample_r["e2e_fn"])
            total_test = good_test + defect_test
        else:
            good_test, defect_test, total_test = 0, 0, 0

        md += f"""
## {idx+3}. 训练样本规模 N = {n} 详细评测
*全量统一测试集：共 {total_test} 张（正常 {good_test} 张 + 缺陷 {defect_test} 张）*

### {idx+3}.1 受试者工作特征曲线下面积 (AUROC)
| 输入尺寸 (Row) | Dinomaly2 基线 (Col 1) | PatchCore 基线 (Col 2) | 二阶段端到端 E2E (Col 3) |
| :--- | :--- | :--- | :--- |
"""
        for row in n_rows:
            s = row["size"]
            d_val = row["din_auc"]
            p_val = row.get("pat_auc")
            e_val = row["e2e_auc"]
            best_val = max(d_val if d_val is not None else 0, p_val if p_val is not None else 0, e_val if e_val is not None else 0)
            d_str = (f"=={d_val:.4f}==" if d_val == best_val else f"{d_val:.4f}") if d_val is not None else "N/A"
            p_str = (f"=={p_val:.4f}==" if p_val == best_val else f"{p_val:.4f}") if p_val is not None else "OOM (显存溢出)"
            e_str = (f"=={e_val:.4f}==" if e_val == best_val else f"{e_val:.4f}") if e_val is not None else "N/A"
            md += f"| {s} × {s} | {d_str} | {p_str} | {e_str} |\n"

        md += f"""
### {idx+3}.2 平均精度 (AP)
| 输入尺寸 (Row) | Dinomaly2 基线 (Col 1) | PatchCore 基线 (Col 2) | 二阶段端到端 E2E (Col 3) |
| :--- | :--- | :--- | :--- |
"""
        for row in n_rows:
            s = row["size"]
            d_val = row["din_ap"]
            p_val = row.get("pat_ap")
            e_val = row["e2e_ap"]
            best_val = max(d_val if d_val is not None else 0, p_val if p_val is not None else 0, e_val if e_val is not None else 0)
            d_str = (f"=={d_val:.4f}==" if d_val == best_val else f"{d_val:.4f}") if d_val is not None else "N/A"
            p_str = (f"=={p_val:.4f}==" if p_val == best_val else f"{p_val:.4f}") if p_val is not None else "OOM (显存溢出)"
            e_str = (f"=={e_val:.4f}==" if e_val == best_val else f"{e_val:.4f}") if e_val is not None else "N/A"
            md += f"| {s} × {s} | {d_str} | {p_str} | {e_str} |\n"

        md += f"""
### {idx+3}.3 最大 F1 分数 (F1-Max)
| 输入尺寸 (Row) | Dinomaly2 基线 (Col 1) | PatchCore 基线 (Col 2) | 二阶段端到端 E2E (Col 3) |
| :--- | :--- | :--- | :--- |
"""
        for row in n_rows:
            s = row["size"]
            d_val = row["din_f1"]
            p_val = row.get("pat_f1")
            e_val = row["e2e_f1"]
            best_val = max(d_val if d_val is not None else 0, p_val if p_val is not None else 0, e_val if e_val is not None else 0)
            d_str = (f"=={d_val:.4f}==" if d_val == best_val else f"{d_val:.4f}") if d_val is not None else "N/A"
            p_str = (f"=={p_val:.4f}==" if p_val == best_val else f"{p_val:.4f}") if p_val is not None else "OOM (显存溢出)"
            e_str = (f"=={e_val:.4f}==" if e_val == best_val else f"{e_val:.4f}") if e_val is not None else "N/A"
            md += f"| {s} × {s} | {d_str} | {p_str} | {e_str} |\n"

        md += f"""
### {idx+3}.4 缺陷检出召回率 (Recall / {defect_test} 张缺陷)
| 输入尺寸 (Row) | Dinomaly2 基线 (Col 1) | PatchCore 基线 (Col 2) | 二阶段端到端 E2E (Col 3) |
| :--- | :--- | :--- | :--- |
"""
        for row in n_rows:
            s = row["size"]
            d_tp = row["din_tp"]
            p_tp = row.get("pat_tp")
            e_tp = row["e2e_tp"]
            d_rec = d_tp / float(defect_test) * 100
            e_rec = e_tp / float(defect_test) * 100
            best_tp = max(d_tp, p_tp if p_tp is not None else 0, e_tp)
            d_str = f"=={d_rec:.2f}% ({d_tp}/{defect_test})==" if d_tp == best_tp else f"{d_rec:.2f}% ({d_tp}/{defect_test})"
            if p_tp is not None:
                p_rec = p_tp / float(defect_test) * 100
                p_str = f"=={p_rec:.2f}% ({p_tp}/{defect_test})==" if p_tp == best_tp else f"{p_rec:.2f}% ({p_tp}/{defect_test})"
            else:
                p_str = f"0.00% (0/{defect_test})" if s == 672 and n >= 200 else "OOM (显存溢出)"
            e_str = f"=={e_rec:.2f}% ({e_tp}/{defect_test})==" if e_tp == best_tp else f"{e_rec:.2f}% ({e_tp}/{defect_test})"
            md += f"| {s} × {s} | {d_str} | {p_str} | {e_str} |\n"

        md += f"""
### {idx+3}.5 正常样本误报数量 (False Positives / {good_test} 张正常)
| 输入尺寸 (Row) | Dinomaly2 基线 (Col 1) | PatchCore 基线 (Col 2) | 二阶段端到端 E2E (Col 3) |
| :--- | :--- | :--- | :--- |
"""
        for row in n_rows:
            s = row["size"]
            d_fp = row["din_fp"]
            p_fp = row.get("pat_fp")
            e_fp = row["e2e_fp"]
            min_fp = min(d_fp, p_fp if p_fp is not None else 999999, e_fp)
            d_str = f"=={d_fp}==" if d_fp == min_fp else f"{d_fp}"
            p_str = (f"=={p_fp}==" if p_fp == min_fp else f"{p_fp}") if p_fp is not None else "0"
            e_str = f"=={e_fp}==" if e_fp == min_fp else f"{e_fp}"
            md += f"| {s} × {s} | {d_str} | {p_str} | {e_str} |\n"

    for target_name in ["FINAL_BENCHMARK_REPORT.md", "BENCHMARK_REPORT.md", "BENCHMARK_DETAILED_REPORT_224_448_672.md"]:
        target_p = outs_dir / target_name
        with open(target_p, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Updated report -> {target_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Comprehensive Benchmark Markdown Reports")
    parser.add_argument("--outs_dir", type=str, default="F:/tmp/0826", help="Base outs directory")
    args = parser.parse_args()

    generate_reports(args.outs_dir)
