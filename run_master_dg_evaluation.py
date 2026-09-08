"""Comprehensive Master Evaluator for all Source-DG variants and Baselines."""

import os
import sys
import glob
import json
from pathlib import Path
import torch

_root = Path(__file__).resolve().parent
_din_dir = _root / "Dinomaly2" if (_root / "Dinomaly2").is_dir() else _root
if str(_din_dir) not in sys.path:
    sys.path.insert(0, str(_din_dir))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dataset import CustomDataset, get_data_transforms
from evaluate_dg_benchmark import load_model, evaluate_model_on_distribution_benchmark

def get_latest_checkpoint(root_dir):
    """Find the latest model.pth under a timestamped directory."""
    if os.path.isfile(root_dir):
        return root_dir
    ckpts = glob.glob(os.path.join(root_dir, '**/model.pth'), recursive=True)
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]

def main():
    data_transform, gt_transform = get_data_transforms(448, 448)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    experiments = [
        {
            'dataset_name': '铜色异常检测4相机',
            'test_txt': '/data/wt/exp/data_splits/铜色异常检测4相机_test.txt',
            'has_pixel_gt': False,
            'models': [
                ('A: Baseline (Original Dinomaly2)', '/data/wt/exp/铜色异常检测4相机_baseline/20260901165031/model.pth'),
                ('B: Data Augmentation Baseline', '/data/wt/exp/铜色4相机_dg_aug'),
                ('C: Adapter + Dual-View Consistency', '/data/wt/exp/铜色4相机_dg_adapter'),
                ('D: C + Diverse Hard Shift Mining', '/data/wt/exp/铜色4相机_dg_hard'),
                ('E: Full Source-DG (Adapter+Hard+Preserve)', '/data/wt/exp/铜色4相机_dg_full'),
                ('Legacy: Local Robust Sinkhorn (ws8)', '/data/wt/exp/铜色异常检测4相机_sinkhorn_ws8/20260901174058/model.pth'),
            ]
        },
        {
            'dataset_name': 'leishi_026',
            'test_txt': '/data/wt/exp/data_splits/leishi_026_test.txt',
            'has_pixel_gt': True,
            'models': [
                ('A: Baseline (Original Dinomaly2)', '/data/wt/exp/leishi_026_baseline/20260901170340/model.pth'),
                ('C: Adapter + Dual-View Consistency', '/data/wt/exp/leishi_026_dg_adapter'),
                ('E: Full Source-DG (Adapter+Hard+Preserve)', '/data/wt/exp/leishi_026_dg_full'),
                ('F: Full Source-DG + Cross-View UOT', '/data/wt/exp/leishi_026_dg_full_uot'),
                ('Legacy: Local Robust Sinkhorn (ws8)', '/data/wt/exp/leishi_026_sinkhorn_ws8/20260901180454/model.pth'),
            ]
        }
    ]

    all_results = {}

    for exp in experiments:
        dname = exp['dataset_name']
        test_txt = exp['test_txt']
        has_pixel = exp['has_pixel_gt']

        test_data = CustomDataset(root=test_txt, transform=data_transform, gt_transform=gt_transform, phase='test')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

        print(f"\n{'='*40} {dname} (Test Total: {len(test_data)}) {'='*40}")
        if has_pixel:
            header = f"{'Model Variant':<42} | {'I-AUROC':<7} | {'I-F1':<7} | {'P-F1':<7} | {'Clean FPR@100%':<14} | {'Seen Mean FPR':<13} | {'Unseen Mean FPR':<15} | {'Worst Unseen FPR':<16}"
        else:
            header = f"{'Model Variant':<42} | {'I-AUROC':<7} | {'I-F1':<7} | {'Clean FPR@100%':<14} | {'Clean FPR@99%':<13} | {'Seen Mean FPR':<13} | {'Unseen Mean FPR':<15} | {'Worst Unseen FPR':<16}"
        print(header)
        print('-' * len(header))

        all_results[dname] = []

        for mlabel, mpath in exp['models']:
            ckpt = get_latest_checkpoint(mpath)
            if not ckpt or not os.path.exists(ckpt):
                print(f"{mlabel:<42} | [CHECKPOINT NOT READY]")
                continue
            
            try:
                model = load_model(ckpt, device=device)
                res = evaluate_model_on_distribution_benchmark(model, test_loader, device=device)
                clean = res['clean']
                seen = res['seen_summary']
                unseen = res['unseen_summary']

                row_data = {
                    'label': mlabel,
                    'ckpt': ckpt,
                    'i_auroc': clean['i_auroc'],
                    'i_f1': clean['i_f1'],
                    'p_f1': clean['p_f1'],
                    'clean_fpr_rec100': clean['clean_fpr_rec100'],
                    'clean_fpr_rec99': clean['clean_fpr_rec99'],
                    'seen_mean_fpr': seen['mean_fpr_prod'],
                    'unseen_mean_fpr': unseen['mean_fpr_prod'],
                    'worst_unseen_fpr': unseen['worst_fpr_prod'],
                    'seen_details': res['seen_details'],
                    'unseen_details': res['unseen_details'],
                }
                all_results[dname].append(row_data)

                if has_pixel:
                    pf1_str = f"{clean['p_f1']:.4f}" if clean['p_f1'] is not None else "N/A"
                    print(f"{mlabel:<42} | {clean['i_auroc']:.4f}  | {clean['i_f1']:.4f} | {pf1_str:<7} | {clean['clean_fpr_rec100']:>13.2f}% | {seen['mean_fpr_prod']:>12.2f}% | {unseen['mean_fpr_prod']:>14.2f}% | {unseen['worst_fpr_prod']:>15.2f}%", flush=True)
                else:
                    print(f"{mlabel:<42} | {clean['i_auroc']:.4f}  | {clean['i_f1']:.4f} | {clean['clean_fpr_rec100']:>13.2f}% | {clean['clean_fpr_rec99']:>12.2f}% | {seen['mean_fpr_prod']:>12.2f}% | {unseen['mean_fpr_prod']:>14.2f}% | {unseen['worst_fpr_prod']:>15.2f}%", flush=True)
            except Exception as e:
                print(f"{mlabel:<42} | [ERROR: {e}]", flush=True)

    with open('/data/wt/exp/master_dg_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print("\nBenchmark results saved to /data/wt/exp/master_dg_benchmark_results.json")

if __name__ == '__main__':
    main()
