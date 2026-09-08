"""Master Distribution Generalization Benchmark Engine for Dinomaly2.

Evaluates Clean In-Domain, Seen Distribution Shifts, and Unseen Stress Shifts
following Dinomaly2_Source_Only_Distribution_Generalization_Design.md.
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import partial

# Ensure Dinomaly2 is on sys.path
_root = Path(__file__).resolve().parent
_din_dir = _root / "Dinomaly2" if (_root / "Dinomaly2").is_dir() else _root
if str(_din_dir) not in sys.path:
    sys.path.insert(0, str(_din_dir))

import importlib.util
_spec = importlib.util.spec_from_file_location("din_utils", _din_dir / "utils.py")
_din_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_din_utils)
cal_anomaly_maps = _din_utils.cal_anomaly_maps
get_gaussian_kernel = _din_utils.get_gaussian_kernel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from dataset import CustomDataset, get_data_transforms
from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, LinearAttention2
from models.domain_adapter import CanonicalizationAdapter, FeatureGroupAdapters


def load_model(ckpt_path: str, device: str = 'cuda:0') -> Dinomaly:
    """Load Dinomaly2 model with automatic detection of backbone size and Canonicalization Adapters."""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    embed_dim = 384
    for k, v in state_dict.items():
        if "bottleneck.0.0.weight" in k:
            embed_dim = v.shape[1]
            break

    num_heads = 6 if embed_dim == 384 else (12 if embed_dim == 768 else 16)
    backbone = "dinov2reg_vit_small_14" if embed_dim == 384 else ("dinov2reg_vit_base_14" if embed_dim == 768 else "dinov2reg_vit_large_14")
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9] if embed_dim <= 768 else [4, 6, 8, 10, 12, 14, 16, 18]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(backbone)

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=0.4)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p=0.4),
        ),
    ])

    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=partial(LinearAttention2, eps=1e-8),
        )
        for _ in range(8)
    ])

    has_adapters = any(k.startswith('feature_adapters') for k in state_dict.keys())
    feature_adapters = []
    if has_adapters:
        feature_adapters = [
            CanonicalizationAdapter(embed_dim, bottleneck_ratio=0.25, alpha_init=0.0)
            for _ in fuse_layer_encoder
        ]

    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=1,
        feature_adapters=feature_adapters,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# Normalization constants (ImageNet)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(images.device, dtype=images.dtype)
    std = STD.to(images.device, dtype=images.dtype)
    return (images * std + mean).clamp(0.0, 1.0)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(images.device, dtype=images.dtype)
    std = STD.to(images.device, dtype=images.dtype)
    return (images - mean) / std


class ShiftTransformSuite:
    """Defines calibrated Seen and Unseen distribution shifts for testing."""

    @staticmethod
    def seen_photometric(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Seen: Brightness + Contrast + Gamma."""
        x = denormalize(images)
        b = float(rng.uniform(0.80, 1.20))
        c = float(rng.uniform(0.80, 1.20))
        gamma = float(rng.uniform(0.85, 1.18))
        x = ((x - 0.5) * c + 0.5) * b
        x = x.clamp(1e-5, 1.0).pow(gamma)
        return normalize(x)

    @staticmethod
    def seen_illumination(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Seen: Gradient shadow + Smooth Vignetting."""
        x = denormalize(images)
        B, C, H, W = x.shape
        yy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        grad_strength = float(rng.uniform(-0.18, 0.18))
        vign_strength = float(rng.uniform(0.0, 0.18))
        grad = 1.0 + grad_strength * (0.5 * xx + 0.5 * yy)
        vign = 1.0 - vign_strength * (xx.square() + yy.square()) * 0.5
        x = x * grad * vign
        return normalize(x.clamp(0.0, 1.0))

    @staticmethod
    def seen_degradation(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Seen: Gaussian Sensor Noise + Moderate Blur."""
        x = denormalize(images)
        noise_std = float(rng.uniform(0.008, 0.018))
        noise = torch.randn_like(x) * noise_std
        x = (x + noise).clamp(0.0, 1.0)
        # 3x3 average blur
        x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
        return normalize(x)

    @staticmethod
    def seen_geometric(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Seen: Translation + Minor Rotation."""
        B, C, H, W = images.shape
        angle = float(rng.uniform(-1.8, 1.8)) * math.pi / 180.0
        scale = float(rng.uniform(0.98, 1.02))
        tx = float(rng.uniform(-0.025, 0.025))
        ty = float(rng.uniform(-0.025, 0.025))
        cos, sin = math.cos(angle), math.sin(angle)
        theta = torch.tensor([
            [cos / scale, -sin / scale, tx],
            [sin / scale, cos / scale, ty]
        ], device=images.device, dtype=images.dtype).unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        return F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    # ================= UNSEEN SHIFTS (Out-of-Distribution Stress Test) =================

    @staticmethod
    def unseen_white_balance(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Unseen: Strong Chromatic Temperature & Tint Shift (Color Cast)."""
        x = denormalize(images)
        # Gain per RGB channel
        r_gain = float(rng.uniform(1.20, 1.35))
        b_gain = float(rng.uniform(0.70, 0.85))
        g_gain = float(rng.uniform(0.90, 1.10))
        gains = torch.tensor([r_gain, g_gain, b_gain], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x * gains).clamp(0.0, 1.0)
        return normalize(x)

    @staticmethod
    def unseen_sensor_poisson(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Unseen: Multiplicative Signal-Dependent Poisson & Speckle Noise."""
        x = denormalize(images)
        speckle_noise = torch.randn_like(x) * 0.06
        x = (x + x * speckle_noise).clamp(0.0, 1.0)
        return normalize(x)

    @staticmethod
    def unseen_defocus_blur(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Unseen: Heavy Optical Defocus Blur (5x5 kernel)."""
        x = denormalize(images)
        # 5x5 average pooling simulating deep optical defocus
        x = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), 5, stride=1)
        return normalize(x)

    @staticmethod
    def unseen_s_curve_tone(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Unseen: Non-linear Sigmoidal S-Curve Tone Mapping."""
        x = denormalize(images)
        # Sigmoidal steep contrast curve
        steepness = float(rng.uniform(5.0, 8.0))
        midpoint = 0.5
        x = 1.0 / (1.0 + torch.exp(-steepness * (x - midpoint)))
        return normalize(x.clamp(0.0, 1.0))

    @staticmethod
    def unseen_specular_glare(images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Unseen: Localized high-intensity circular specular hotspot + heavy vignetting."""
        x = denormalize(images)
        B, C, H, W = x.shape
        cx = float(rng.uniform(-0.5, 0.5))
        cy = float(rng.uniform(-0.5, 0.5))
        yy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        dist_sq = (xx - cx).square() + (yy - cy).square()
        glare = 0.35 * torch.exp(-dist_sq / 0.12)
        vignette = 1.0 - 0.35 * (xx.square() + yy.square())
        x = (x * vignette + glare).clamp(0.0, 1.0)
        return normalize(x)


def compute_scores_and_metrics(
    model: nn.Module,
    dataloader,
    device: str,
    shift_fn=None,
    seed: int = 42,
    max_ratio: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Run forward pass and return anomaly scores, labels, and pixel maps."""
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    img_scores = []
    labels = []
    px_preds = []
    px_gts = []
    has_px = False

    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            if shift_fn is not None:
                img = shift_fn(img, rng)

            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            flat_map = anomaly_map.flatten(1)
            topk = max(1, int(flat_map.shape[1] * max_ratio))
            sp_score = torch.sort(flat_map, dim=1, descending=True)[0][:, :topk].mean(dim=1)

            img_scores.extend(sp_score.cpu().numpy().tolist())
            labels.extend(label.numpy().tolist())

            if gt is not None and gt.sum() > 0:
                has_px = True
            if gt is not None:
                gt_down = F.interpolate(gt.float(), size=(112, 112), mode='nearest')
                am_down = F.interpolate(anomaly_map, size=(112, 112), mode='bilinear', align_corners=False)
                px_preds.append(am_down.squeeze(1).cpu().numpy())
                px_gts.append(gt_down.squeeze(1).cpu().numpy())

    res = {
        'scores': np.array(img_scores, dtype=np.float64),
        'labels': np.array(labels, dtype=np.int64),
        'has_px': has_px,
    }
    if has_px and len(px_preds) > 0:
        res['px_preds'] = np.concatenate(px_preds).ravel()
        res['px_gts'] = (np.concatenate(px_gts).ravel() > 0.5).astype(np.uint8)
    return res


def evaluate_model_on_distribution_benchmark(
    model: nn.Module,
    test_loader,
    device: str = 'cuda:0',
    seed: int = 42,
) -> Dict[str, object]:
    """Complete DG Evaluation across Clean, 4 Seen Shifts, and 5 Unseen Shifts."""
    # 1. Clean Benchmark
    clean_res = compute_scores_and_metrics(model, test_loader, device, shift_fn=None, seed=seed)
    clean_scores = clean_res['scores']
    clean_labels = clean_res['labels']

    clean_norm_scores = clean_scores[clean_labels == 0]
    clean_bad_scores = clean_scores[clean_labels == 1]

    i_auroc = float(roc_auc_score(clean_labels, clean_scores))
    i_ap = float(average_precision_score(clean_labels, clean_scores))
    precs, recs, _ = precision_recall_curve(clean_labels, clean_scores)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    i_f1 = float(np.nanmax(f1s))

    # Pixel Metrics
    p_auroc, p_ap, p_f1 = None, None, None
    if clean_res['has_px'] and 'px_preds' in clean_res:
        p_auroc = float(roc_auc_score(clean_res['px_gts'], clean_res['px_preds']))
        p_ap = float(average_precision_score(clean_res['px_gts'], clean_res['px_preds']))
        p_precs, p_recs, _ = precision_recall_curve(clean_res['px_gts'], clean_res['px_preds'])
        p_f1s = 2 * p_precs * p_recs / (p_precs + p_recs + 1e-7)
        p_f1 = float(np.nanmax(p_f1s))

    # Production threshold calibrated at Clean Normal 99% (Clean FPR = 1%)
    tau_prod = float(np.percentile(clean_norm_scores, 99.0)) if len(clean_norm_scores) > 0 else 0.0
    clean_fpr_prod = float(np.mean(clean_norm_scores >= tau_prod) * 100)

    # 100% and 99% recall thresholds on bad samples
    tau_rec100 = float(np.min(clean_bad_scores)) if len(clean_bad_scores) > 0 else 0.0
    clean_fpr_rec100 = float(np.mean(clean_norm_scores >= tau_rec100) * 100) if len(clean_norm_scores) > 0 else 0.0

    tau_rec99 = float(np.percentile(clean_bad_scores, 1.0)) if len(clean_bad_scores) > 0 else 0.0
    clean_fpr_rec99 = float(np.mean(clean_norm_scores >= tau_rec99) * 100) if len(clean_norm_scores) > 0 else 0.0

    # 2. Seen Shifts Evaluation
    seen_shifts = {
        'Seen Photometric': ShiftTransformSuite.seen_photometric,
        'Seen Illumination': ShiftTransformSuite.seen_illumination,
        'Seen Degradation': ShiftTransformSuite.seen_degradation,
        'Seen Geometric': ShiftTransformSuite.seen_geometric,
    }
    seen_metrics = {}
    seen_prod_fprs = []
    seen_rec100_fprs = []
    seen_rec99_fprs = []

    for name, shift_fn in seen_shifts.items():
        s_res = compute_scores_and_metrics(model, test_loader, device, shift_fn=shift_fn, seed=seed)
        s_norm_scores = s_res['scores'][s_res['labels'] == 0]
        fpr_prod = float(np.mean(s_norm_scores >= tau_prod) * 100)
        fpr_rec100 = float(np.mean(s_norm_scores >= tau_rec100) * 100)
        fpr_rec99 = float(np.mean(s_norm_scores >= tau_rec99) * 100)
        seen_metrics[name] = {
            'fpr_prod': fpr_prod,
            'fpr_rec100': fpr_rec100,
            'fpr_rec99': fpr_rec99,
            'mean_score': float(np.mean(s_norm_scores)),
        }
        seen_prod_fprs.append(fpr_prod)
        seen_rec100_fprs.append(fpr_rec100)
        seen_rec99_fprs.append(fpr_rec99)

    # 3. Unseen Shifts Evaluation (OOD Stress Testing)
    unseen_shifts = {
        'Unseen WhiteBalance': ShiftTransformSuite.unseen_white_balance,
        'Unseen SensorPoisson': ShiftTransformSuite.unseen_sensor_poisson,
        'Unseen DefocusBlur': ShiftTransformSuite.unseen_defocus_blur,
        'Unseen SCurveTone': ShiftTransformSuite.unseen_s_curve_tone,
        'Unseen SpecularGlare': ShiftTransformSuite.unseen_specular_glare,
    }
    unseen_metrics = {}
    unseen_prod_fprs = []
    unseen_rec100_fprs = []
    unseen_rec99_fprs = []

    for name, shift_fn in unseen_shifts.items():
        u_res = compute_scores_and_metrics(model, test_loader, device, shift_fn=shift_fn, seed=seed)
        u_norm_scores = u_res['scores'][u_res['labels'] == 0]
        fpr_prod = float(np.mean(u_norm_scores >= tau_prod) * 100)
        fpr_rec100 = float(np.mean(u_norm_scores >= tau_rec100) * 100)
        fpr_rec99 = float(np.mean(u_norm_scores >= tau_rec99) * 100)
        unseen_metrics[name] = {
            'fpr_prod': fpr_prod,
            'fpr_rec100': fpr_rec100,
            'fpr_rec99': fpr_rec99,
            'mean_score': float(np.mean(u_norm_scores)),
        }
        unseen_prod_fprs.append(fpr_prod)
        unseen_rec100_fprs.append(fpr_rec100)
        unseen_rec99_fprs.append(fpr_rec99)

    return {
        'clean': {
            'i_auroc': i_auroc,
            'i_ap': i_ap,
            'i_f1': i_f1,
            'p_auroc': p_auroc,
            'p_ap': p_ap,
            'p_f1': p_f1,
            'clean_fpr_prod': clean_fpr_prod,
            'clean_fpr_rec100': clean_fpr_rec100,
            'clean_fpr_rec99': clean_fpr_rec99,
            'mean_norm': float(np.mean(clean_norm_scores)),
            'mean_bad': float(np.mean(clean_bad_scores)),
            'tau_prod': tau_prod,
            'tau_rec100': tau_rec100,
            'tau_rec99': tau_rec99,
        },
        'seen_summary': {
            'mean_fpr_prod': float(np.mean(seen_prod_fprs)),
            'worst_fpr_prod': float(np.max(seen_prod_fprs)),
            'mean_fpr_rec100': float(np.mean(seen_rec100_fprs)),
            'mean_fpr_rec99': float(np.mean(seen_rec99_fprs)),
        },
        'seen_details': seen_metrics,
        'unseen_summary': {
            'mean_fpr_prod': float(np.mean(unseen_prod_fprs)),
            'worst_fpr_prod': float(np.max(unseen_prod_fprs)),
            'mean_fpr_rec100': float(np.mean(unseen_rec100_fprs)),
            'mean_fpr_rec99': float(np.mean(unseen_rec99_fprs)),
        },
        'unseen_details': unseen_metrics,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate distribution generalization benchmark')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Dinomaly2 checkpoint model.pth')
    parser.add_argument('--test-txt', '--test_txt', type=str, required=True, dest='test_txt', help='Path to test.txt list or dataset directory')
    parser.add_argument('--image_size', type=int, default=448, help='Evaluation image size (default: 448)')
    parser.add_argument('--crop_size', type=int, default=448, help='Evaluation crop size (default: 448)')
    parser.add_argument('--batch_size', type=int, default=16, help='Evaluation batch size (default: 16)')
    parser.add_argument('--output_json', type=str, default=None, help='Optional output json to save detailed benchmark metrics')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    dino_s = (args.image_size // 14) * 14 if args.image_size % 14 != 0 else args.image_size
    dino_crop = (args.crop_size // 14) * 14 if args.crop_size % 14 != 0 else args.crop_size

    data_transform, gt_transform = get_data_transforms(dino_s, dino_crop)
    test_data = CustomDataset(root=args.test_txt, transform=data_transform, gt_transform=gt_transform, phase='test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model(args.checkpoint, device=args.device)
    res = evaluate_model_on_distribution_benchmark(model, test_loader, device=args.device)
    print("=" * 60)
    print(f"Clean I-AUROC:           {res['clean']['i_auroc']:.4f}")
    print(f"Clean I-F1:              {res['clean']['i_f1']:.4f}")
    if res['clean']['p_auroc'] is not None:
        print(f"Clean P-AUROC:           {res['clean']['p_auroc']:.4f}")
        print(f"Clean P-F1:              {res['clean']['p_f1']:.4f}")
    print(f"Clean FPR@100% Recall:   {res['clean']['clean_fpr_rec100']:.2f}%")
    print(f"Clean FPR (Prod Q99):    {res['clean']['clean_fpr_prod']:.2f}%")
    print("-" * 60)
    print(f"Seen Mean FPR (Prod):    {res['seen_summary']['mean_fpr_prod']:.2f}%")
    print(f"Seen Worst FPR (Prod):   {res['seen_summary']['worst_fpr_prod']:.2f}%")
    print(f"Seen Mean FPR@Rec100:    {res['seen_summary']['mean_fpr_rec100']:.2f}%")
    print("-" * 60)
    print(f"Unseen Mean FPR (Prod):  {res['unseen_summary']['mean_fpr_prod']:.2f}%")
    print(f"Unseen Worst FPR (Prod): {res['unseen_summary']['worst_fpr_prod']:.2f}%")
    print(f"Unseen Mean FPR@Rec100:  {res['unseen_summary']['mean_fpr_rec100']:.2f}%")
    print("=" * 60)

    if args.output_json:
        import json
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        print(f"Saved benchmark results to {args.output_json}")
