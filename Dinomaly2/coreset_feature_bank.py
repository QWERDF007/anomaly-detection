"""Greedy K-Center Coreset subsampling for two-stage feature memory banks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("coreset")


def kcenter_greedy_coreset(
    features: np.ndarray,
    target_size: int,
    device: str = "cuda:0",
) -> np.ndarray:
    """Select target_size representative features using K-Center Greedy."""
    n_samples = features.shape[0]
    if target_size >= n_samples:
        return features

    LOGGER.info(f"Selecting {target_size} coreset centers from {n_samples} vectors using {device}...")
    feat_tensor = torch.from_numpy(features).float().to(device)

    # 1. Start with the center closest to the mean
    mean_feat = feat_tensor.mean(dim=0, keepdim=True)
    init_idx = int(torch.argmin(torch.norm(feat_tensor - mean_feat, dim=1)).item())

    selected_indices = [init_idx]
    # Initialize min distances to the first selected center
    min_distances = torch.norm(feat_tensor - feat_tensor[init_idx : init_idx + 1], dim=1)

    for _ in tqdm(range(1, target_size), desc="K-Center Coreset"):
        # Select the point furthest from all selected centers
        next_idx = int(torch.argmax(min_distances).item())
        selected_indices.append(next_idx)

        # Update min distances with the newly selected center
        new_dist = torch.norm(feat_tensor - feat_tensor[next_idx : next_idx + 1], dim=1)
        min_distances = torch.minimum(min_distances, new_dist)

    selected_indices = np.array(selected_indices, dtype=np.int64)
    coreset_features = features[selected_indices]
    LOGGER.info(f"Coreset selection finished: {coreset_features.shape[0]} vectors retained.")
    return coreset_features


def build_coreset_library(
    src_dir: Path,
    dst_dir: Path,
    coreset_ratio: float = 0.5,
    min_vectors: int = 100,
    device: str = "cuda:0",
):
    dst_dir.mkdir(parents=True, exist_ok=True)
    index_path = src_dir / "index.faiss"
    meta_path = src_dir / "metadata.json"

    if not (index_path.is_file() and meta_path.is_file()):
        LOGGER.warning(f"Index or metadata missing in {src_dir}")
        return

    index = faiss.read_index(str(index_path))
    n_vecs = index.ntotal
    dim = index.d
    features = index.reconstruct_n(0, n_vecs)

    target_size = max(min_vectors, int(n_vecs * coreset_ratio))
    coreset_feats = kcenter_greedy_coreset(features, min(target_size, n_vecs), device=device)

    # Re-normalize if needed
    faiss.normalize_L2(coreset_feats)
    new_index = faiss.IndexFlatL2(dim)
    new_index.add(coreset_feats)

    faiss.write_index(new_index, str(dst_dir / "index.faiss"))

    meta = json.loads(meta_path.read_text())
    meta["coreset_ratio"] = coreset_ratio
    meta["original_vector_count"] = n_vecs
    meta["coreset_vector_count"] = coreset_feats.shape[0]
    (dst_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    LOGGER.info(f"Saved coreset index to {dst_dir / 'index.faiss'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True, type=Path)
    parser.add_argument("--dst_root", required=True, type=Path)
    parser.add_argument("--coreset_ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    for lib_type in ["good", "anomaly"]:
        s_dir = args.src_root / lib_type
        d_dir = args.dst_root / lib_type
        if s_dir.exists():
            build_coreset_library(s_dir, d_dir, args.coreset_ratio, device=args.device)


if __name__ == "__main__":
    main()
