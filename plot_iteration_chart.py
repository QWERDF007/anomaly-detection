#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans SC", "SimHei", "WenQuanYi Micro Hei", "DejaVu Sans", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 350
plt.rcParams["savefig.dpi"] = 350

COLOR_E2E = "#059669"       # Emerald Green
COLOR_DINO = "#2563eb"      # Royal Blue
COLOR_PATCH = "#ea580c"     # Amber / Deep Orange

MARKER_SHAPES = ["o", "s", "^", "D", "v", "p", "*", "h"]

# Balanced marker sizes for ultra-HD canvas:
SIZE_MAP = {
    224: 20,
    448: 42,
    672: 76,
}

def generate_iteration_chart(
    outs_dir: Path,
    output_png: Optional[Path] = None,
    dataset_title: Optional[str] = None,
    dpi: int = 350,
    save_png: bool = False,
) -> Path:
    outs_dir = Path(outs_dir).expanduser().resolve()
    summary_path = outs_dir / "final_multisize_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary not found at: {summary_path}")

    chart_dir = outs_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(output_png) if output_png else chart_dir / "09_image_auroc_vs_iterations.png"

    # Always generate the single interactive vector SVG HTML dashboard
    dashboard_html = chart_dir / "benchmark_dashboard.html"
    try:
        from generate_interactive_chart_html import generate_dataset_html
        generate_dataset_html(outs_dir)
    except Exception as e:
        print(f"[WARN] Could not generate interactive HTML: {e}")

    if not save_png:
        print(f"[INFO] Static PNG output disabled. Interactive dashboard available at: {dashboard_html}")
        return dashboard_html

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary_data:
        raise ValueError(f"Summary data is empty in {summary_path}")

    unique_sizes = sorted(list({int(r["size"]) for r in summary_data}))
    unique_ns = sorted(list({int(r["n"]) for r in summary_data}))
    unique_iters = sorted(list({int(r.get("iters", 2000)) for r in summary_data}))

    n_marker_map = {n: MARKER_SHAPES[i % len(MARKER_SHAPES)] for i, n in enumerate(unique_ns)}
    size_marker_map = {s: SIZE_MAP.get(s, int((s / 224.0) ** 1.5 * 20)) for s in unique_sizes}

    # Deterministic micro-dodge offsets based on N and Size
    # Guarantees no two points with the same iteration ever have the same X
    n_offsets = np.linspace(-0.032, 0.032, len(unique_ns)) if len(unique_ns) > 1 else [0.0]
    n_offset_map = {n: n_offsets[i] for i, n in enumerate(unique_ns)}

    size_offsets = np.linspace(-0.010, 0.010, len(unique_sizes)) if len(unique_sizes) > 1 else [0.0]
    size_offset_map = {s: size_offsets[i] for i, s in enumerate(unique_sizes)}

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

        # Dinomaly2: centered around it * 0.94
        d_mult = 0.94 * (1.0 + n_offset_map[n] + size_offset_map[s])
        d_item = {
            "model": "Dinomaly2",
            "iters": it,
            "plot_x": it * d_mult,
            "n": n,
            "size": s,
            "auroc": d_auc,
        }
        dino_pts.append(d_item)
        all_all_pts.append(d_item)

        # Two-Stage E2E: centered around it * 1.06
        e_mult = 1.06 * (1.0 + n_offset_map[n] + size_offset_map[s])
        e_item = {
            "model": "Two-Stage E2E",
            "iters": it,
            "plot_x": it * e_mult,
            "n": n,
            "size": s,
            "auroc": e_auc,
        }
        e2e_pts.append(e_item)
        all_all_pts.append(e_item)

        # Patchcore at 1000
        pat_key = (n, s)
        if pat_key not in pat_seen and p_auc > 0:
            pat_seen.add(pat_key)
            p_mult = 1.0 + (n_offset_map[n] + size_offset_map[s]) * 0.85
            p_item = {
                "model": "PatchCore",
                "iters": 1000,
                "plot_x": 1000.0 * p_mult,
                "n": n,
                "size": s,
                "auroc": p_auc,
            }
            pat_pts.append(p_item)
            all_all_pts.append(p_item)

    fig, ax = plt.subplots(figsize=(19.0, 9.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    all_aucs = [p["auroc"] for p in all_all_pts if p["auroc"] > 0]
    min_auc = min(all_aucs) if all_aucs else 0.70
    max_auc = max(all_aucs) if all_aucs else 1.00

    y_lower = max(0.0, np.floor((min_auc - 0.015) * 50) / 50)
    y_upper = min(1.002, np.ceil((max_auc + 0.012) * 50) / 50)

    # 1. Scatter Points
    for p in pat_pts:
        marker = n_marker_map[p["n"]]
        sz = size_marker_map[p["size"]]
        ax.scatter(p["plot_x"], p["auroc"], marker=marker, s=sz,
                   color=COLOR_PATCH, edgecolors="#ffffff", linewidth=0.7,
                   alpha=0.92, zorder=4)

    for p in dino_pts:
        marker = n_marker_map[p["n"]]
        sz = size_marker_map[p["size"]]
        ax.scatter(p["plot_x"], p["auroc"], marker=marker, s=sz,
                   color=COLOR_DINO, edgecolors="#ffffff", linewidth=0.7,
                   alpha=0.90, zorder=4)

    for p in e2e_pts:
        marker = n_marker_map[p["n"]]
        sz = size_marker_map[p["size"]]
        ax.scatter(p["plot_x"], p["auroc"], marker=marker, s=sz,
                   color=COLOR_E2E, edgecolors="#ffffff", linewidth=0.7,
                   alpha=0.92, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(750, 52000)
    ax.set_ylim(y_lower, y_upper)

    x_ticks = [1000, 2000, 5000, 10000, 20000, 40000]
    x_tick_labels = ["1,000", "2,000", "5,000", "10,000", "20,000", "40,000"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=11.5)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=11.5)
    ax.tick_params(axis="y", which="major", left=True, right=False, labelsize=11.5)

    ax.set_xlabel("Iterations", fontsize=13.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("I-AUROC", fontsize=13.5, fontweight="bold", labelpad=10)

    ax.grid(True, which="major", axis="both", linestyle=":", color="#e2e8f0", linewidth=0.9, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    ax.set_title("I-AUROC vs. Iterations", fontsize=15, fontweight="bold", pad=30, color="#0f172a", loc="left")

    # Legend
    legend_handles_models = [
        matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_E2E,
                                markersize=6.5, lw=0, label="二阶段端到端 (Two-Stage E2E)"),
        matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DINO,
                                markersize=6.5, lw=0, label="Dinomaly2 (单阶段)"),
        matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PATCH,
                                markersize=6.5, lw=0, label="PatchCore (固定1000轮)"),
    ]
    leg1 = ax.legend(handles=legend_handles_models, loc="lower left",
                     bbox_to_anchor=(0.0, 1.02),
                     ncol=3, mode=None, borderaxespad=0., frameon=False, fontsize=10.5,
                     handletextpad=0.38, columnspacing=1.5)
    ax.add_artist(leg1)

    legend_handles_n = [
        matplotlib.lines.Line2D([0], [0], marker=n_marker_map[n], color="#64748b", markerfacecolor="#94a3b8",
                                markersize=5.8, lw=0, label=f"N={n}")
        for n in unique_ns
    ]
    size_legend_sizes = {224: 4.6, 448: 6.6, 672: 8.8}
    legend_handles_size = [
        matplotlib.lines.Line2D([0], [0], marker="o", color="#64748b", markerfacecolor="#94a3b8",
                                markersize=size_legend_sizes.get(s, 6.0), lw=0, label=f"{s}×{s}")
        for s in unique_sizes
    ]
    spacer = [matplotlib.lines.Line2D([0], [0], color="none", marker=None, lw=0, label="   ")]
    all_n_size_handles = legend_handles_n + spacer + legend_handles_size

    leg2 = ax.legend(
        handles=all_n_size_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=len(all_n_size_handles),
        frameon=False,
        fontsize=9.8,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    svg_file = out_file.with_suffix(".svg")
    fig.savefig(svg_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Generated Chart -> {out_file}")
    print(f"[SUCCESS] Generated Vector SVG -> {svg_file}")

    # Also generate the single interactive vector SVG HTML dashboard
    try:
        from generate_interactive_chart_html import generate_dataset_html
        generate_dataset_html(outs_dir)
    except Exception as e:
        print(f"[WARN] Could not generate interactive HTML: {e}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Plot Iteration Performance Chart")
    parser.add_argument("--outs_dir", type=str, default="", help="Experiment output directory for a specific dataset")
    parser.add_argument("--outs_base", type=str, default="/data/wt/exp0906", help="Base directory containing multiple datasets")
    parser.add_argument("--output_png", type=str, default=None, help="Custom output PNG path")
    parser.add_argument("--dpi", type=int, default=350, help="Output image DPI resolution (default: 350)")
    parser.add_argument("--save_png", action="store_true", default=False, help="Save static PNG image (default: False)")
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

    print(f"Generating Iteration Scaling HTML dashboards for {len(targets)} dataset(s)...")
    for t in targets:
        try:
            generate_iteration_chart(
                t,
                output_png=Path(args.output_png) if args.output_png else None,
                dpi=args.dpi,
                save_png=args.save_png,
            )
        except Exception as e:
            print(f"[ERROR] Failed for {t.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
