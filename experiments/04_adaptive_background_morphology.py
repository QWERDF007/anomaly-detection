"""Experiment 4: Adaptive Background Noise Floor Subtraction & Morphological Opening.
Usage:
    python experiments/04_adaptive_background_morphology.py --percentile 20 --kernel_size 3
"""
import argparse
import numpy as np
import cv2

def apply_background_morphology(score_map: np.ndarray, p_floor: float = 20.0, k_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    opened = cv2.morphologyEx(score_map, cv2.MORPH_OPEN, kernel)
    bg_floor = float(np.percentile(opened, p_floor))
    cleaned = np.maximum(opened - bg_floor, 0.0)
    return cleaned

if __name__ == "__main__":
    print("Adaptive Background Morphology experiment script initialized.")
