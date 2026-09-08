import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set style & CJK font
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

EXP_DIR = Path("/data/wt/exp_shimo")
CHARTS_DIR = EXP_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

with open(EXP_DIR / "clean_confusion_matrices.json", "r", encoding="utf-8") as f:
    data = json.load(f)

subdirs = ["ori-10", "ori-21"]
models = [
    "PatchCore (Coreset 10%)",
    "Dinomaly2 Base (Baseline)",
    "Source-DG (Adapter)",
    "Source-DG (Full + UOT)"
]
model_labels = [m.replace(" ", "\n") for m in models]

# -------------------------------------------------------------
# Chart 1: 产线基准阈值 (tau_prod) 下的 TP / FP / TN / FN 堆叠分析图
# -------------------------------------------------------------
def plot_stacked_confusion(mode="at_production", title_suffix="产线基准阈值 (良品放行率 98.7%~99.0%)", filename="confusion_matrices_breakdown_production.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors
    c_tp = '#2a9d8f' # Teal green
    c_fn = '#e76f51' # Coral red
    c_tn = '#457b9d' # Steel blue
    c_fp = '#e9c46a' # Mustard yellow

    for col_idx, sub in enumerate(subdirs):
        # 1. Row 0: Defect Samples Breakdown (Actual Positive = TP + FN)
        ax_pos = axes[0, col_idx]
        tp_vals = []
        fn_vals = []
        total_p = 0
        for m in models:
            item = data[f"{sub}_{m}"][mode]
            tp_vals.append(item['TP'])
            fn_vals.append(item['FN'])
            total_p = data[f"{sub}_{m}"]['actual_positive_NG']
            
        x = np.arange(len(models))
        w = 0.55
        b1 = ax_pos.bar(x, tp_vals, w, label='TP (成功检出缺陷)', color=c_tp, alpha=0.9)
        b2 = ax_pos.bar(x, fn_vals, w, bottom=tp_vals, label='FN (漏检缺陷)', color=c_fn, alpha=0.9)
        
        ax_pos.set_title(f'【{sub}】缺陷样本判定拆解 (真实缺陷总数 = {total_p} 张)', fontsize=13, fontweight='bold', pad=10)
        ax_pos.set_xticks(x)
        ax_pos.set_xticklabels(model_labels, fontsize=10)
        ax_pos.set_ylabel('图像张数 (张)', fontsize=11)
        ax_pos.set_ylim(0, total_p * 1.15)
        ax_pos.legend(loc='upper right', frameon=True)
        
        for idx in range(len(models)):
            tp = tp_vals[idx]
            fn = fn_vals[idx]
            if tp > total_p * 0.08:
                ax_pos.text(x[idx], tp / 2, f'TP: {tp}\n({tp/total_p*100:.1f}%)', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            elif tp > 0:
                ax_pos.text(x[idx], tp + 10, f'TP:{tp}', ha='center', va='bottom', color=c_tp, fontweight='bold', fontsize=9)
            if fn > total_p * 0.08:
                ax_pos.text(x[idx], tp + fn / 2, f'FN: {fn}\n({fn/total_p*100:.1f}%)', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            elif fn > 0:
                ax_pos.text(x[idx], tp + fn + 10, f'FN:{fn}', ha='center', va='bottom', color=c_fn, fontweight='bold', fontsize=9)
                
        # 2. Row 1: Normal Samples Breakdown (Actual Negative = TN + FP)
        ax_neg = axes[1, col_idx]
        tn_vals = []
        fp_vals = []
        total_n = 0
        for m in models:
            item = data[f"{sub}_{m}"][mode]
            tn_vals.append(item['TN'])
            fp_vals.append(item['FP'])
            total_n = data[f"{sub}_{m}"]['actual_negative_OK']
            
        b3 = ax_neg.bar(x, tn_vals, w, label='TN (正确放行良品)', color=c_tn, alpha=0.9)
        b4 = ax_neg.bar(x, fp_vals, w, bottom=tn_vals, label='FP (良品虚警误报)', color=c_fp, alpha=0.9)
        
        ax_neg.set_title(f'【{sub}】良品样本判定拆解 (真实良品总数 = {total_n} 张)', fontsize=13, fontweight='bold', pad=10)
        ax_neg.set_xticks(x)
        ax_neg.set_xticklabels(model_labels, fontsize=10)
        ax_neg.set_ylabel('图像张数 (张)', fontsize=11)
        ax_neg.set_ylim(0, total_n * 1.15)
        ax_neg.legend(loc='upper right', frameon=True)
        
        for idx in range(len(models)):
            tn = tn_vals[idx]
            fp = fp_vals[idx]
            if tn > total_n * 0.1:
                ax_neg.text(x[idx], tn / 2, f'TN: {tn}\n({tn/total_n*100:.1f}%)', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            if fp > 0:
                ax_neg.text(x[idx], tn + max(fp/2, 2), f'FP:{fp}', ha='center', va='center', color='#333333', fontweight='bold', fontsize=9)

    plt.suptitle(f'工业质检模型在 {title_suffix} 下的 TP / FP / TN / FN 详细分解对比', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_p = CHARTS_DIR / filename
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"Saved: {out_p}")

# -------------------------------------------------------------
# Chart 2: 2x2 混淆矩阵热力图网格 (Confusion Matrix Heatmaps Grid)
# -------------------------------------------------------------
def plot_confusion_heatmaps(mode="at_production", filename="confusion_matrix_heatmaps_grid.png"):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    
    for row_idx, sub in enumerate(subdirs):
        for col_idx, m in enumerate(models):
            ax = axes[row_idx, col_idx]
            item = data[f"{sub}_{m}"][mode]
            tp = item['TP']
            fn = item['FN']
            fp = item['FP']
            tn = item['TN']
            
            # Confusion matrix:
            # Row 0: Actual NG (TP, FN)
            # Row 1: Actual OK (FP, TN)
            cm = np.array([[tp, fn], [fp, tn]])
            cm_norm = cm.astype(float)
            cm_norm[0] /= (tp + fn)
            cm_norm[1] /= (fp + tn)
            
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
            
            # Text annotations
            cell_texts = [
                [f"TP: {tp}\n({cm_norm[0,0]*100:.1f}%)", f"FN: {fn}\n({cm_norm[0,1]*100:.1f}%)"],
                [f"FP: {fp}\n({cm_norm[1,0]*100:.1f}%)", f"TN: {tn}\n({cm_norm[1,1]*100:.1f}%)"]
            ]
            for i in range(2):
                for j in range(2):
                    val = cm_norm[i, j]
                    color = "white" if val > 0.55 else "black"
                    ax.text(j, i, cell_texts[i][j], ha="center", va="center", color=color, fontweight='bold', fontsize=11)
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['预测缺陷 (NG)', '预测良品 (OK)'], fontsize=10)
            ax.set_yticklabels(['真实缺陷 (NG)', '真实良品 (OK)'], fontsize=10)
            
            ax.set_title(f"[{sub}] {m.split(' (')[0]}\nRecall: {item['Recall']*100:.1f}% | FPR: {item['FPR']*100:.1f}%", fontsize=11, fontweight='bold', pad=8)
            ax.set_ylabel('真实类别', fontsize=10)
            ax.set_xlabel('模型预测', fontsize=10)

    plt.suptitle('8 组模型在产线基准阈值 (tau_prod) 下的 2x2 混淆矩阵热力图', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_p = CHARTS_DIR / filename
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"Saved: {out_p}")

# -------------------------------------------------------------
# Chart 3: 产线拦截率 (Recall) vs 虚警率 (FPR) 与 精确率 (Precision)
# -------------------------------------------------------------
def plot_production_tradeoffs(filename="production_tradeoff_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    for col_idx, sub in enumerate(subdirs):
        ax = axes[col_idx]
        x = np.arange(len(models))
        w = 0.25
        
        recalls = [data[f"{sub}_{m}"]["at_production"]["Recall"] * 100 for m in models]
        precisions = [data[f"{sub}_{m}"]["at_production"]["Precision"] * 100 for m in models]
        f1s = [data[f"{sub}_{m}"]["at_production"]["F1"] * 100 for m in models]
        
        r1 = ax.bar(x - w, recalls, w, label='缺陷拦截率 Recall (TP率 %)', color='#2A9D8F', alpha=0.9)
        r2 = ax.bar(x, precisions, w, label='查准率 Precision (%)', color='#264653', alpha=0.9)
        r3 = ax.bar(x + w, f1s, w, label='产线综合 F1 (%)', color='#E76F51', alpha=0.9)
        
        ax.set_title(f'【{sub}】产线工作点核心指标表现', fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.set_ylabel('百分比 (%)', fontsize=11)
        ax.set_ylim(0, 115)
        ax.legend(loc='lower right', frameon=True)
        
        for rects in [r1, r2, r3]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.suptitle('工业产线阈值约束下的模型拦截效能对比 (良品误报严格限制在 <= 1.7%)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_p = CHARTS_DIR / filename
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"Saved: {out_p}")

def main():
    plot_stacked_confusion("at_production", "产线基准阈值 (良品放行率 98.7%~99.0%)", "confusion_matrices_breakdown_production.png")
    plot_stacked_confusion("at_best_f1", "理论最优 F1-Score 阈值", "confusion_matrices_breakdown_best_f1.png")
    plot_confusion_heatmaps("at_production", "confusion_matrix_heatmaps_grid.png")
    plot_production_tradeoffs("production_tradeoff_comparison.png")
    print("\n[SUCCESS] All confusion matrix charts generated successfully!")

if __name__ == '__main__':
    main()
