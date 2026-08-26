"""Experiment 5: Multi-Model Multiplicative Consensus (vit_base_15k * vit_small_5k) with BG Floor Denoising.
Usage:
    python experiments/05_multi_model_consensus.py --p_floor 25
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

def compute_multiplicative_consensus(map_15k: np.ndarray, map_small: np.ndarray, p_floor: float = 25.0) -> np.ndarray:
    consensus = np.sqrt(np.maximum(map_15k, 0.0) * np.maximum(map_small, 0.0))
    if p_floor > 0:
        bg = float(np.percentile(consensus, p_floor))
        consensus = np.maximum(consensus - bg, 0.0)
    return consensus

if __name__ == "__main__":
    print("Multi-Model Consensus experiment script initialized.")
