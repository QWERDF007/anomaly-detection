#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Iteration Scaling Performance Chart Generator
Generates clean, elegant multi-model scaling trajectory charts:
  - X-axis: 训练迭代次数 (Training Iterations, Log Scale)
  - Y-axis: 图像级 AUROC (Image-level AUROC)
  - Visual Encodings:
      * Same Model: Same color theme & line trajectory
          - Two-Stage E2E (二阶段端到端): Emerald Green (#059669)
          - Dinomaly2 (单阶段深度重建): Royal Blue (#2563eb)
          - PatchCore (基线): Amber/Orange (#ea580c), fixed at X=1000 iters
      * Different N: Distinct marker shapes ('o', 's', '^', 'D', etc.)
      * Different Image Size: Distinct marker sizes (224=small, 448=medium, 672=large)
      * Strict Data Integrity: 100% genuine measured data from final_multisize_summary.json
      * No artificial envelope or extra lines: only true model trajectories
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Font configuration
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans SC", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Color palette
COLOR_E2E = "#059669"       # Emerald Green
COLOR_DINO = "#2563eb"      # Royal Blue
COLOR_PATCH = "#ea580c"     # Amber / Deep Orange
BG_SHADE = "#ecfdf5"        # Soft Mint Green

MARKER_SHAPES = ["o", "s", "^", "D", "v", "p", "*", "h"]
SIZE_MAP = {
    224: 75,
    448: 165,
    672: 295,
}


def generate_iteration_chart(
    outs_dir: Path,
    output_png: Optional[Path] = None,
    dataset_title: Optional[str] = None,
) -> Path:
    outs_dir = Path(outs_dir).expanduser().resolve()
    summary_path = outs_dir / "final_multisize_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary not found at: {summary_path}")

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary_data:
        raise ValueError(f"Summary data is empty in {summary_path}")

    ds_name = dataset_title or outs_dir.name
    chart_dir = outs_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(output_png) if output_png else chart_dir / "09_image_auroc_vs_iterations.png"

    # Collect unique dimensions
    unique_sizes = sorted(list({int(r["size"]) for r in summary_data}))
    unique_ns = sorted(list({int(r["n"]) for r in summary_data}))
    unique_iters = sorted(list({int(r.get("iters", 2000)) for r in summary_data}))

    # Marker shape mapping for N
    n_marker_map = {}
    for idx, n in enumerate(unique_ns):
        n_marker_map[n] = MARKER_SHAPES[idx % len(MARKER_SHAPES)]

    # Size mapping for image size
    size_marker_map = {}
    for s in unique_sizes:
        if s in SIZE_MAP:
            size_marker_map[s] = SIZE_MAP[s]
        else:
            size_marker_map[s] = int((s / 224.0) ** 1.5 * 75)

    dino_pts = []
    e2e_pts = []
    pat_pts = []
    all_all_pts = []

    pat_seen = set()
    for r in summary_data:
        n = int(r["n"])
        s = int(r["size"])
        it = int(r["iters"])

        d_auc = float(r.get("din_auc", 0.0))
        e_auc = float(r.get("e2e_auc", 0.0))
        p_auc = float(r.get("pat_auc", 0.0))

        # Dinomaly2
        d_item = {
            "model": "Dinomaly2",
            "iters": it,
            "plot_x": it * 0.955,  # slight horizontal offset so overlapping points are visible
            "n": n,
            "size": s,
            "auroc": d_auc,
        }
        dino_pts.append(d_item)
        all_all_pts.append(d_item)

        # Two-Stage E2E
        e_item = {
            "model": "Two-Stage E2E",
            "iters": it,
            "plot_x": it * 1.045,  # slight horizontal offset so overlapping points are visible
            "n": n,
            "size": s,
            "auroc": e_auc,
        }
        e2e_pts.append(e_item)
        all_all_pts.append(e_item)

        # PatchCore (strictly fixed at 1000)
        pat_key = (n, s)
        if pat_key not in pat_seen and p_auc > 0:
            pat_seen.add(pat_key)
            p_item = {
                "model": "PatchCore",
                "iters": 1000,
                "plot_x": 1000.0,
                "n": n,
                "size": s,
                "auroc": p_auc,
            }
            pat_pts.append(p_item)
            all_all_pts.append(p_item)

    # Create widescreen layout figure
    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Determine Y bounds
    all_aucs = [p["auroc"] for p in all_all_pts if p["auroc"] > 0]
    min_auc = min(all_aucs) if all_aucs else 0.70
    max_auc = max(all_aucs) if all_aucs else 1.00

    if max_auc - min_auc > 0.15:
        y_lower = max(0.0, np.floor((min_auc - 0.02) * 20) / 20)
        y_upper = min(1.005, np.ceil((max_auc + 0.015) * 50) / 50)
    else:
        y_lower = max(0.0, np.floor((min_auc - 0.01) * 50) / 50)
        y_upper = min(1.005, np.ceil((max_auc + 0.01) * 50) / 50)
    if y_upper < 1.0:
        y_upper = 1.005

    # 1. Background Shaded Zone: High-Precision Industrial Acceptance Zone
    high_perf_threshold = 0.95 if max_auc >= 0.94 else (0.90 if max_auc >= 0.89 else 0.85)
    if y_upper > high_perf_threshold:
        ax.axhspan(high_perf_threshold, y_upper, xmin=0.0, xmax=1.0,
                   color=BG_SHADE, alpha=0.55, zorder=0)
        ax.text(42000, high_perf_threshold + (y_upper - high_perf_threshold) * 0.88,
                f"★ 高精度工业准入区 (I-AUROC ≥ {high_perf_threshold:.2f})",
                fontsize=9.5, color="#047857", fontweight="bold", va="center", ha="right", zorder=1)

    # 2. Plot Trajectory Lines across Iterations for Dinomaly2 and E2E
    for n in unique_ns:
        for s in unique_sizes:
            # Dinomaly2 trajectory
            d_traj = sorted([p for p in dino_pts if p["n"] == n and p["size"] == s], key=lambda x: x["iters"])
            if len(d_traj) > 1:
                tx = [p["plot_x"] for p in d_traj]
                ty = [p["auroc"] for p in d_traj]
                ax.plot(tx, ty, color=COLOR_DINO, lw=1.2, alpha=0.40, linestyle="-", zorder=2)

            # Two-Stage E2E trajectory
            e_traj = sorted([p for p in e2e_pts if p["n"] == n and p["size"] == s], key=lambda x: x["iters"])
            if len(e_traj) > 1:
                tx = [p["plot_x"] for p in e_traj]
                ty = [p["auroc"] for p in e_traj]
                ax.plot(tx, ty, color=COLOR_E2E, lw=1.5, alpha=0.50, linestyle="-", zorder=2)

    # 3. Scatter Plot Points for Each Model
    # PatchCore points (at iters=1000)
    for p in pat_pts:
        marker = n_marker_map[p["n"]]
        sz = size_marker_map[p["size"]]
        ax.scatter(p["plot_x"], p["auroc"], marker=marker, s=sz,
                   color=COLOR_PATCH, edgecolors="#ffffff", linewidth=1.2,
                   alpha=0.92, zorder=4)

    # Dinomaly2 points
    for p in dino_pts:
        marker = n_marker_map[p["n"]]
        sz = size_marker_map[p["size"]]
        ax.scatter(p["plot_x"], p["auroc"], marker=marker, s=sz,
                   color=COLOR_DINO, edgecolors="#ffffff", linewidth=1.2,
                   alpha=0.88, zorder=4)

    # Two-Stage E2E points
    for p in e2e_pts:
        marker = n_marker_map[p["n"]]
        sz = size_marker_map[p["size"]]
        ax.scatter(p["plot_x"], p["auroc"], marker=marker, s=sz,
                   color=COLOR_E2E, edgecolors="#ffffff", linewidth=1.4,
                   alpha=0.92, zorder=4)

    # 4. Add Clean Point Annotations for Best Standout Models
    # Top overall point
    best_overall = max(all_all_pts, key=lambda x: x["auroc"])
    ax.annotate(
        f"{best_overall['model']}\n(N={best_overall['n']}, s{best_overall['size']})\n{best_overall['auroc']:.4f}",
        xy=(best_overall["plot_x"], best_overall["auroc"]),
        xytext=(best_overall["plot_x"] * 1.14, best_overall["auroc"] + 0.005),
        fontsize=8.5, fontweight="bold", color="#0f172a",
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.0, shrinkA=3, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cbd5e1", lw=0.8, alpha=0.92),
        zorder=7,
    )

    # Best PatchCore point (at iters=1000)
    best_pat = max(pat_pts, key=lambda x: x["auroc"]) if pat_pts else None
    if best_pat and best_pat != best_overall:
        ax.annotate(
            f"PatchCore 最优\n(N={best_pat['n']}, s{best_pat['size']})\n{best_pat['auroc']:.4f}",
            xy=(best_pat["plot_x"], best_pat["auroc"]),
            xytext=(best_pat["plot_x"] * 1.15, best_pat["auroc"] - 0.008),
            fontsize=8.0, color="#c2410c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.0, shrinkA=3, shrinkB=3),
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#fdba74", lw=0.8, alpha=0.92),
            zorder=7,
        )

    # Best 2k model (fast convergence checkpoint)
    best_2k = [p for p in all_all_pts if p["iters"] == 2000]
    if best_2k:
        top_2k = max(best_2k, key=lambda x: x["auroc"])
        if top_2k != best_overall and top_2k != best_pat:
            ax.annotate(
                f"{top_2k['model']} 2k快速收敛\n(N={top_2k['n']}, s{top_2k['size']})\n{top_2k['auroc']:.4f}",
                xy=(top_2k["plot_x"], top_2k["auroc"]),
                xytext=(top_2k["plot_x"] * 1.18, top_2k["auroc"] - 0.016),
                fontsize=8.0, color="#1e3a8a",
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.0, shrinkA=3, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#bfdbfe", lw=0.8, alpha=0.92),
                zorder=7,
            )

    # 5. Axes & Grid Configuration (Log Scale on X)
    ax.set_xscale("log")
    ax.set_xlim(750, 52000)
    ax.set_ylim(y_lower, y_upper)

    # X-Ticks
    x_ticks = [1000, 2000, 5000, 10000, 20000, 40000]
    x_tick_labels = [
        "1,000\n(PatchCore固定)",
        "2,000",
        "5,000",
        "10,000",
        "20,000",
        "40,000",
    ]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=9.5)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=9.5)
    ax.tick_params(axis="y", which="major", left=True, right=False, labelsize=9.5)

    # Labels
    ax.set_xlabel("训练最大迭代轮次 (Training Iterations, Log Scale)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("图像级 AUROC (Image-level AUROC)", fontsize=11, fontweight="bold", labelpad=10)

    # Subtle elegant grid
    ax.grid(True, which="major", axis="both", linestyle=":", color="#e2e8f0", linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Clean Spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    # 6. Title and Branding Header
    ax.set_title(
        f"「{ds_name}」图像级 AUROC 随迭代次数演化对比 (I-AUROC vs. Iterations)",
        fontsize=13.5, fontweight="bold", pad=28, color="#0f172a", loc="left"
    )
    fig.text(0.92, 0.945, "❖ Industrial Anomaly Detection Benchmark",
             fontsize=10.5, color="#64748b", fontweight="bold", ha="right", va="center")

    # 7. Legends: Multi-Dimensional Clean Layout
    # Group A: Model Series
    legend_handles_models = [
        matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_E2E,
                                markersize=9, label="二阶段端到端 (Two-Stage E2E)"),
        matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DINO,
                                markersize=9, label="Dinomaly2 (单阶段深度重建)"),
        matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PATCH,
                                markersize=9, label="PatchCore (特征索引, 固定1000轮)"),
    ]
    leg1 = ax.legend(handles=legend_handles_models, loc="lower left",
                      bbox_to_anchor=(0.0, 1.01, 1.0, 0.05),
                      ncol=3, mode=None, borderaxespad=0., frameon=False, fontsize=9.5)
    ax.add_artist(leg1)

    # Group B & C: Marker Shape (N) & Marker Size (Resolution) in Bottom-Right Inset Legend
    legend_handles_n = [
        matplotlib.lines.Line2D([0], [0], marker=n_marker_map[n], color="#64748b", markerfacecolor="#94a3b8",
                                markersize=7, lw=0, label=f"N={n}")
        for n in unique_ns
    ]
    legend_handles_size = [
        matplotlib.lines.Line2D([0], [0], marker="o", color="#64748b", markerfacecolor="#94a3b8",
                                markersize=int(np.sqrt(size_marker_map[s]) * 0.75), lw=0, label=f"{s}×{s}")
        for s in unique_sizes
    ]
    leg2 = ax.legend(
        handles=legend_handles_n + [matplotlib.lines.Line2D([0], [0], color="w", label="")] + legend_handles_size,
        title="样本量 N (Mark) 与 分辨率 (Size)",
        title_fontsize=8.5,
        loc="lower right",
        ncol=len(unique_ns) + 1 + len(unique_sizes),
        fontsize=8.2,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1"
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Generated Chart -> {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Plot Iteration Performance Chart")
    parser.add_argument("--outs_dir", type=str, default="", help="Experiment output directory for a specific dataset")
    parser.add_argument("--outs_base", type=str, default="/data/wt/exp0906", help="Base directory containing multiple datasets")
    parser.add_argument("--output_png", type=str, default=None, help="Custom output PNG path")
    args = parser.parse_args()

    targets = []
    if args.outs_dir:
        targets.append(Path(args.outs_dir))
    else:
        base = Path(args.outs_base)
        for child in base.iterdir():
            if child.is_dir() and (child / "final_multisize_summary.json").is_file():
                targets.append(child)

    if not targets:
        print(f"[WARN] No valid datasets with final_multisize_summary.json found in {args.outs_base or args.outs_dir}")
        sys.exit(1)

    print(f"Generating Iteration Scaling charts for {len(targets)} dataset(s)...")
    for t in targets:
        try:
            generate_iteration_chart(t, output_png=Path(args.output_png) if args.output_png else None)
        except Exception as e:
            print(f"[ERROR] Failed for {t.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
