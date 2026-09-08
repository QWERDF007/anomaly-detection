import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

EXP_DIR = Path("/data/wt/exp0903")
CHARTS_DIR = EXP_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    json_path = EXP_DIR / "benchmark_results.json"
    if not json_path.is_file():
        print(f"Error: {json_path} does not exist!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Separate by dataset
    vc_models = {}
    zs_models = {}

    for k, v in data.items():
        ds = v['dataset']
        m_name = v['model']
        if ds == 'VC':
            vc_models[m_name] = v['metrics']
        else:
            zs_models[m_name] = v['metrics']

    model_order = [
        "PatchCore (Coreset 10%)",
        "Dinomaly2 Base (Baseline)",
        "Source-DG (Adapter)",
        "Source-DG (Full + UOT)",
    ]

    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#D0021B']

    # -------------------------------------------------------------
    # Chart 1: In-Domain Performance (I-AUROC, I-F1, P-AUROC)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax_idx, (ds_name, ds_dict) in enumerate([("VC (N=200)", vc_models), ("中石 (N=56)", zs_models)]):
        ax = axes[ax_idx]
        x = np.arange(len(model_order))
        width = 0.25

        i_auroc = [ds_dict.get(m, {}).get('clean', {}).get('i_auroc', 0) * 100 for m in model_order]
        i_f1 = [ds_dict.get(m, {}).get('clean', {}).get('i_f1', 0) * 100 for m in model_order]
        p_auroc = [ds_dict.get(m, {}).get('clean', {}).get('p_auroc', 0) or 0 for m in model_order]
        p_auroc = [p * 100 for p in p_auroc]

        r1 = ax.bar(x - width, i_auroc, width, label='Image AUROC (%)', color='#2B5B84', alpha=0.9)
        r2 = ax.bar(x, i_f1, width, label='Image F1 (%)', color='#2E9374', alpha=0.9)
        r3 = ax.bar(x + width, p_auroc, width, label='Pixel AUROC (%)', color='#E2843B', alpha=0.9)

        ax.set_title(f'Clean In-Domain Metrics: {ds_name}', fontsize=14, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in model_order], fontsize=10)
        ax.set_ylim(40, 105)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.legend(loc='lower right', frameon=True)

        for rects in [r1, r2, r3]:
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.annotate(f'{h:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    chart1_path = CHARTS_DIR / "clean_benchmark_comparison.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved: {chart1_path}")

    # -------------------------------------------------------------
    # Chart 2: Distribution Shift Robustness (Clean vs Seen vs Unseen FPR)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax_idx, (ds_name, ds_dict) in enumerate([("VC (N=200)", vc_models), ("中石 (N=56)", zs_models)]):
        ax = axes[ax_idx]
        x = np.arange(len(model_order))
        width = 0.25

        clean_fpr = [ds_dict.get(m, {}).get('clean', {}).get('clean_fpr_prod', 0) for m in model_order]
        seen_fpr = [ds_dict.get(m, {}).get('seen_summary', {}).get('mean_fpr_prod', 0) for m in model_order]
        unseen_fpr = [ds_dict.get(m, {}).get('unseen_summary', {}).get('mean_fpr_prod', 0) for m in model_order]

        r1 = ax.bar(x - width, clean_fpr, width, label='Clean Normal FPR (%)', color='#52B788', alpha=0.9)
        r2 = ax.bar(x, seen_fpr, width, label='Seen Shift Mean FPR (%)', color='#F77F00', alpha=0.9)
        r3 = ax.bar(x + width, unseen_fpr, width, label='Unseen Stress Mean FPR (%)', color='#D62828', alpha=0.9)

        ax.set_title(f'False Positive Rate across Distribution Shifts: {ds_name}', fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in model_order], fontsize=10)
        ax.set_ylim(0, max(max(unseen_fpr + seen_fpr) * 1.25, 40))
        ax.set_ylabel('False Positive Rate (%) [Lower is Better]', fontsize=11)
        ax.legend(loc='upper right', frameon=True)

        for rects in [r1, r2, r3]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart2_path = CHARTS_DIR / "distribution_shift_robustness.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved: {chart2_path}")

    # -------------------------------------------------------------
    # Chart 3: Detailed Breakdown across 9 Shift Types
    # -------------------------------------------------------------
    shifts_list = [
        'Seen Photometric', 'Seen Illumination', 'Seen Degradation', 'Seen Geometric',
        'Unseen WhiteBalance', 'Unseen SensorPoisson', 'Unseen DefocusBlur', 'Unseen SCurveTone', 'Unseen SpecularGlare'
    ]
    shift_labels = [s.replace("Seen ", "S: ").replace("Unseen ", "U: ") for s in shifts_list]

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for ax_idx, (ds_name, ds_dict) in enumerate([("VC", vc_models), ("中石", zs_models)]):
        ax = axes[ax_idx]
        x = np.arange(len(shifts_list))
        w = 0.18

        for m_idx, m_name in enumerate(model_order):
            m_metrics = ds_dict.get(m_name, {})
            seen_det = m_metrics.get('seen_details', {})
            unseen_det = m_metrics.get('unseen_details', {})
            
            vals = []
            for s in shifts_list:
                if s in seen_det:
                    vals.append(seen_det[s].get('fpr_prod', 0))
                elif s in unseen_det:
                    vals.append(unseen_det[s].get('fpr_prod', 0))
                else:
                    vals.append(0)

            offset = (m_idx - 1.5) * w
            ax.bar(x + offset, vals, w, label=m_name, color=colors[m_idx], alpha=0.85)

        ax.set_title(f'False Positive Rate by Specific Shift Category: {ds_name} Dataset', fontsize=13, fontweight='bold')
        ax.set_ylabel('FPR (%) [Lower is Better]', fontsize=11)
        ax.legend(loc='upper right', frameon=True)
        ax.set_xticks(x)
        ax.set_xticklabels(shift_labels, rotation=15, ha='right', fontsize=10)

    plt.tight_layout()
    chart3_path = CHARTS_DIR / "shift_breakdown_detail.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved: {chart3_path}")

    # -------------------------------------------------------------
    # Chart 4: Efficiency Benchmark (FPS vs Latency vs VRAM)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Average efficiency across datasets
    fps_list = []
    lat_list = []
    vram_list = []

    for m in model_order:
        fps_val = np.mean([vc_models.get(m, {}).get('efficiency', {}).get('fps', 0),
                           zs_models.get(m, {}).get('efficiency', {}).get('fps', 0)])
        lat_val = np.mean([vc_models.get(m, {}).get('efficiency', {}).get('latency_ms', 0),
                           zs_models.get(m, {}).get('efficiency', {}).get('latency_ms', 0)])
        vram_val = np.mean([vc_models.get(m, {}).get('efficiency', {}).get('vram_gb', 0),
                            zs_models.get(m, {}).get('efficiency', {}).get('vram_gb', 0)])
        fps_list.append(fps_val)
        lat_list.append(lat_val)
        vram_list.append(vram_val)

    y_pos = np.arange(len(model_order))

    # FPS bar chart
    bars = ax1.barh(y_pos, fps_list, color='#3A86FF', alpha=0.85, height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_order, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Throughput (FPS) [Higher is Better]', fontsize=11)
    ax1.set_title('Inference Speed (FPS @ 448x448)', fontsize=13, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        ax1.annotate(f'{w:.1f} FPS',
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     va='center', ha='left', fontsize=9, fontweight='bold')

    # Latency vs Peak VRAM
    ax2_twin = ax2.twinx()
    b1 = ax2.bar(y_pos - 0.2, lat_list, 0.4, label='Latency (ms)', color='#8338EC', alpha=0.85)
    b2 = ax2_twin.bar(y_pos + 0.2, vram_list, 0.4, label='Peak VRAM (GB)', color='#FF006E', alpha=0.85)

    ax2.set_xticks(y_pos)
    ax2.set_xticklabels([m.replace(" ", "\n") for m in model_order], fontsize=9)
    ax2.set_ylabel('Inference Latency (ms)', color='#8338EC', fontsize=11)
    ax2_twin.set_ylabel('Peak VRAM Allocation (GB)', color='#FF006E', fontsize=11)
    ax2.set_title('Latency & GPU Memory Footprint', fontsize=13, fontweight='bold')

    plt.tight_layout()
    chart4_path = CHARTS_DIR / "efficiency_benchmark.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"Saved: {chart4_path}")

if __name__ == '__main__':
    main()
