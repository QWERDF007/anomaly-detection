import sys
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

ROOT = Path("/data/wt/anomaly-detection").resolve()
sys.path.insert(0, str(ROOT / "Dinomaly2"))
sys.path.insert(0, str(ROOT / "patchcore-inspection" / "src"))
sys.path.insert(0, str(ROOT))

from dataset import CustomDataset, get_data_transforms
from evaluate_dg_benchmark import (
    load_model as load_dinomaly_model,
    cal_anomaly_maps,
    get_gaussian_kernel,
)
import patchcore.patchcore
import patchcore.common

def get_scores_dinomaly(ckpt_path, test_txt, device="cuda:0"):
    model = load_dinomaly_model(str(ckpt_path), device=device)
    model.eval()
    data_transform, gt_transform = get_data_transforms(448, 448)
    test_data = CustomDataset(root=str(test_txt), transform=data_transform, gt_transform=gt_transform, phase="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    scores = []
    labels = []
    with torch.no_grad():
        for img, _, label, _ in test_loader:
            img = img.to(device)
            en, de = model(img)
            am, _ = cal_anomaly_maps(en, de, 448)
            am = gaussian_kernel(am)
            flat_am = am.flatten(1)
            topk = max(1, int(flat_am.shape[1] * 0.01))
            sp_score = torch.sort(flat_am, dim=1, descending=True)[0][:, :topk].mean(dim=1)
            scores.extend(sp_score.cpu().numpy().tolist())
            labels.extend(label.numpy().tolist())
    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int64)

def get_scores_patchcore(pkl_path, test_txt, device="cuda:0"):
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(
        load_path=str(pkl_path.parent),
        device=device,
        prepend=pkl_path.name[:-len("patchcore_params.pkl")],
        nn_method=patchcore.common.FaissNN(on_gpu=True, device_id=0),
    )
    data_transform, gt_transform = get_data_transforms(448, 448)
    test_data = CustomDataset(root=str(test_txt), transform=data_transform, gt_transform=gt_transform, phase="test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

    scores = []
    labels = []
    with torch.no_grad():
        for img, _, label, _ in test_loader:
            s, _ = model._predict(img.to(device))
            scores.extend(s.tolist() if isinstance(s, np.ndarray) else s)
            labels.extend(label.numpy().tolist())
    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int64)

def calc_confusion(y_true, y_score, tau):
    y_pred = (y_score >= tau).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'tau': float(tau),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'Accuracy': float(acc),
        'FPR': float(fpr)
    }

def main():
    exp_dir = Path("/data/wt/exp0903")
    device = "cuda:0"
    
    models = [
        # VC
        ("VC", "PatchCore (Coreset 10%)", "patchcore", exp_dir / "VC_patchcore_n200/20260903234044/patchcore_params.pkl", exp_dir / "data_splits/VC_test.txt"),
        ("VC", "Dinomaly2 Base (Baseline)", "dinomaly", exp_dir / "VC_dinomaly2_base/20260903234043/model.pth", exp_dir / "data_splits/VC_test.txt"),
        ("VC", "Source-DG (Adapter)", "dinomaly", exp_dir / "VC_source_dg_adapter/20260903234044/model.pth", exp_dir / "data_splits/VC_test.txt"),
        ("VC", "Source-DG (Full + UOT)", "dinomaly", exp_dir / "VC_source_dg_full_uot/20260903234044/model.pth", exp_dir / "data_splits/VC_test.txt"),
        # 中石
        ("中石", "PatchCore (Coreset 10%)", "patchcore", exp_dir / "中石_patchcore_n56/20260903234044/patchcore_params.pkl", exp_dir / "data_splits/中石_test.txt"),
        ("中石", "Dinomaly2 Base (Baseline)", "dinomaly", exp_dir / "中石_dinomaly2_base/20260903234044/model.pth", exp_dir / "data_splits/中石_test.txt"),
        ("中石", "Source-DG (Adapter)", "dinomaly", exp_dir / "中石_source_dg_adapter/20260903234221/model.pth", exp_dir / "data_splits/中石_test.txt"),
        ("中石", "Source-DG (Full + UOT)", "dinomaly", exp_dir / "中石_source_dg_full_uot/20260903234500/model.pth", exp_dir / "data_splits/中石_test.txt"),
    ]
    
    results = {}
    for sub, name, m_type, m_path, test_path in models:
        print(f"Calculating confusion matrix for [{sub}] {name}...")
        if m_type == "patchcore":
            scores, labels = get_scores_patchcore(m_path, test_path, device=device)
        else:
            scores, labels = get_scores_dinomaly(m_path, test_path, device=device)
            
        norm_scores = scores[labels == 0]
        bad_scores = scores[labels == 1]
        
        precs, recs, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * precs * recs / (precs + recs + 1e-8)
        best_idx = np.nanargmax(f1_scores)
        tau_best_f1 = float(thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1])
        tau_prod = float(np.percentile(norm_scores, 99.0))
        tau_rec100 = float(np.min(bad_scores))
        
        results[f"{sub}_{name}"] = {
            'subset': sub,
            'model': name,
            'total_samples': len(labels),
            'actual_positive_NG': int(np.sum(labels == 1)),
            'actual_negative_OK': int(np.sum(labels == 0)),
            'at_best_f1': calc_confusion(labels, scores, tau_best_f1),
            'at_production': calc_confusion(labels, scores, tau_prod),
            'at_rec100': calc_confusion(labels, scores, tau_rec100),
        }
        
    with open(exp_dir / "clean_confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nALL EXP0903 CONFUSION MATRICES CALCULATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
