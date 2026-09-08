import sys
import json
import time
import os
import subprocess
import multiprocessing as mp
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path("/data/wt/anomaly-detection").resolve()
PYTHON = "/home/dell/miniconda3/envs/anomaly/bin/python"
EXP_DIR = Path("/data/wt/exp_shimo")

def run_single(gpu_id, sub, name, m_type, m_path, test_path, out_file):
    cmd = [
        PYTHON, "-c", f"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

ROOT = Path('/data/wt/anomaly-detection').resolve()
sys.path.insert(0, str(ROOT / 'Dinomaly2'))
sys.path.insert(0, str(ROOT / 'patchcore-inspection' / 'src'))
sys.path.insert(0, str(ROOT))

from dataset import CustomDataset, get_data_transforms
from evaluate_dg_benchmark import load_model as load_dinomaly_model, cal_anomaly_maps, get_gaussian_kernel
import patchcore.patchcore
import patchcore.common

sub = '{sub}'
name = '{name}'
m_type = '{m_type}'
m_path = Path('{m_path}')
test_path = Path('{test_path}')
device = 'cuda:0'

data_transform, gt_transform = get_data_transforms(448, 448)
test_data = CustomDataset(root=str(test_path), transform=data_transform, gt_transform=gt_transform, phase='test')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

scores, labels = [], []
if m_type == 'patchcore':
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(
        load_path=str(m_path.parent),
        device=device,
        prepend=m_path.name[:-len('patchcore_params.pkl')],
        nn_method=patchcore.common.FaissNN(on_gpu=True, device_id=0),
    )
    with torch.no_grad():
        for img, _, label, _ in test_loader:
            s, _ = model._predict(img.to(device))
            scores.extend(s.tolist() if isinstance(s, np.ndarray) else s)
            labels.extend(label.numpy().tolist())
else:
    model = load_dinomaly_model(str(m_path), device=device)
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
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

scores = np.array(scores, dtype=np.float64)
labels = np.array(labels, dtype=np.int64)

norm_scores = scores[labels == 0]
bad_scores = scores[labels == 1]

precs, recs, thresholds = precision_recall_curve(labels, scores)
f1_scores = 2 * precs * recs / (precs + recs + 1e-8)
best_idx = np.nanargmax(f1_scores)
tau_best_f1 = float(thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1])
tau_prod = float(np.percentile(norm_scores, 99.0))
tau_rec100 = float(np.min(bad_scores))

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
    return {{
        'tau': float(tau),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'Accuracy': float(acc),
        'FPR': float(fpr)
    }}

res = {{
    'subset': sub,
    'model': name,
    'total_samples': len(labels),
    'actual_positive_NG': int(np.sum(labels == 1)),
    'actual_negative_OK': int(np.sum(labels == 0)),
    'at_best_f1': calc_confusion(labels, scores, tau_best_f1),
    'at_production': calc_confusion(labels, scores, tau_prod),
    'at_rec100': calc_confusion(labels, scores, tau_rec100),
}}

with open('{out_file}', 'w') as f:
    json.dump(res, f, indent=2)
print(f'Done {sub} {name}')
"""
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ret = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"Error in {sub} {name}: {ret.stderr}")

def main():
    exp_dir = Path("/data/wt/exp_shimo")
    tmp_dir = exp_dir / "tmp_conf"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        # (gpu_id, sub, name, m_type, m_path, test_path)
        (0, "ori-10", "PatchCore (Coreset 10%)", "patchcore", exp_dir / "ori-10_patchcore_n200/20260904151728/patchcore_params.pkl", exp_dir / "data_splits/ori-10_test.txt"),
        (1, "ori-10", "Dinomaly2 Base (Baseline)", "dinomaly", exp_dir / "ori-10_dinomaly2_base/20260904151728/model.pth", exp_dir / "data_splits/ori-10_test.txt"),
        (2, "ori-10", "Source-DG (Adapter)", "dinomaly", exp_dir / "ori-10_source_dg_adapter/20260904151728/model.pth", exp_dir / "data_splits/ori-10_test.txt"),
        (3, "ori-10", "Source-DG (Full + UOT)", "dinomaly", exp_dir / "ori-10_source_dg_full_uot/20260904151729/model.pth", exp_dir / "data_splits/ori-10_test.txt"),
        (4, "ori-21", "PatchCore (Coreset 10%)", "patchcore", exp_dir / "ori-21_patchcore_n200/20260904151729/patchcore_params.pkl", exp_dir / "data_splits/ori-21_test.txt"),
        (5, "ori-21", "Dinomaly2 Base (Baseline)", "dinomaly", exp_dir / "ori-21_dinomaly2_base/20260904151730/model.pth", exp_dir / "data_splits/ori-21_test.txt"),
        (6, "ori-21", "Source-DG (Adapter)", "dinomaly", exp_dir / "ori-21_source_dg_adapter/20260904151920/model.pth", exp_dir / "data_splits/ori-21_test.txt"),
        (7, "ori-21", "Source-DG (Full + UOT)", "dinomaly", exp_dir / "ori-21_source_dg_full_uot/20260904151955/model.pth", exp_dir / "data_splits/ori-21_test.txt"),
    ]
    
    procs = []
    for gid, sub, name, m_type, m_path, test_path in models:
        out_f = tmp_dir / f"{sub}_{m_type}_{gid}.json"
        p = mp.Process(target=run_single, args=(gid, sub, name, m_type, m_path, test_path, out_f))
        p.start()
        procs.append((p, sub, name, out_f))
        
    for p, sub, name, out_f in procs:
        p.join()
        
    final_res = {}
    for _, sub, name, out_f in procs:
        if out_f.is_file():
            with open(out_f) as f:
                d = json.load(f)
            final_res[f"{sub}_{name}"] = d
            
    with open(exp_dir / "clean_confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(final_res, f, indent=2, ensure_ascii=False)
        
    print(f"MERGED {len(final_res)} CONFUSION MATRICES!")

if __name__ == "__main__":
    main()
