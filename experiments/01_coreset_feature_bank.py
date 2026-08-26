"""Experiment 1: Greedy K-Center Coreset Subsampling for Two-Stage Feature Banks.
Usage:
    python experiments/01_coreset_feature_bank.py --src_root /data/wt/two_stages/base_672_15k --coreset_ratio 0.5
"""
import argparse
import json
from pathlib import Path
import faiss
import numpy as np
import torch
from tqdm import tqdm

def kcenter_greedy_coreset(features: np.ndarray, target_size: int, device: str = "cuda:0") -> np.ndarray:
    n_samples = features.shape[0]
    if target_size >= n_samples:
        return features
    feat_tensor = torch.from_numpy(features).float().to(device)
    mean_feat = feat_tensor.mean(dim=0, keepdim=True)
    init_idx = int(torch.argmin(torch.norm(feat_tensor - mean_feat, dim=1)).item())
    selected_indices = [init_idx]
    min_distances = torch.norm(feat_tensor - feat_tensor[init_idx:init_idx+1], dim=1)
    for _ in tqdm(range(1, target_size), desc="K-Center Coreset"):
        next_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(next_idx)
        new_dist = torch.norm(feat_tensor - feat_tensor[next_idx:next_idx+1], dim=1)
        min_distances = torch.minimum(min_distances, new_dist)
    selected_indices = np.array(selected_indices, dtype=np.int64)
    return features[selected_indices]

if __name__ == "__main__":
    print("Coreset Feature Bank experiment script initialized.")
