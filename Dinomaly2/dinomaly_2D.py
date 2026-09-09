# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
_dinomaly_dir = os.path.dirname(os.path.abspath(__file__))
if _dinomaly_dir not in sys.path:
    sys.path.insert(0, _dinomaly_dir)

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_mild_normal_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import json
import csv
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import Dinomaly
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock
from dataset import (
    CustomDataset,
    CustomRAMDataset,
    MVTecDataset,
    MVTecRAMDataset,
)
import torch.backends.cudnn as cudnn
from utils import (
    WarmupCosineScheduler,
    cal_anomaly_maps,
    collect_image_scores,
    evaluation_batch,
    global_cosine,
    global_cosine_hm_percent,
)
from losses import (
    CombinedDinomalyLoss,
    RobustSinkhornLoss,
    anomaly_margin_loss,
    cosine_consistency,
    identity_loss,
    map_consistency,
    preservation_loss,
)
from models.domain_adapter import CanonicalizationAdapter
from augment import NormalShiftGenerator, PseudoAnomalyGenerator
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging
import time
from collections import Counter
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# Keep the legacy and mask-constraint training loops on the same schedule.
TRAIN_BATCH_SIZE = 4
EVAL_EVERY_EPOCHS = -1


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def _sec2hms(s):
    s = int(s)
    return f'{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _feature_image_scores(en, de, max_ratio=0.01):
    """Compute inexpensive image scores for hard normal-shift mining."""

    anomaly_map, _ = cal_anomaly_maps(en, de, out_size=en[0].shape[-1])
    flattened = anomaly_map.flatten(1)
    top_count = max(1, int(flattened.shape[1] * max_ratio))
    return torch.topk(flattened, k=top_count, dim=1).values.mean(dim=1)


def _select_hard_shift(model, images, generator, candidates, device):
    """Generate candidates across diverse physical shift families and select the highest-scoring normal view per sample."""

    candidates = max(1, int(candidates))
    diverse_cats = ['photometric', 'illumination', 'degradation', 'geometry']
    views = [generator(images, force_category=diverse_cats[0] if candidates > 1 else None)]
    if candidates > 1:
        scores = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            scores.append(_feature_image_scores(*model(views[0].images)))
            for i in range(1, candidates):
                cat = diverse_cats[i % len(diverse_cats)]
                view = generator(images, force_category=cat)
                views.append(view)
                scores.append(_feature_image_scores(*model(view.images)))
        model.train(was_training)
        score_matrix = torch.stack(scores, dim=0)
        selected_indices = score_matrix.argmax(dim=0)
    else:
        selected_indices = torch.zeros(images.shape[0], dtype=torch.long, device=device)

    selected_images = torch.stack(
        [views[int(index)].images[row] for row, index in enumerate(selected_indices)], dim=0
    )
    selected_theta = torch.stack(
        [views[int(index)].theta[row] for row, index in enumerate(selected_indices)], dim=0
    )
    selected_families = [views[int(index)].family[row] for row, index in enumerate(selected_indices)]
    return type(views[0])(selected_images, selected_theta, selected_families)


def _phase_ramp(progress, start, end):
    """Return a 0-to-1 linear stage weight for source-DG scheduling."""

    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / max(end - start, 1e-8)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(model, test_data_list, item_list, device, batch_size, epoch, writer=None):
    """Run the shared default Dinomaly2 evaluation for either train mode."""

    if not test_data_list:
        return None

    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
    model.eval()

    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        results = evaluation_batch(
            model,
            test_dataloader,
            device,
            max_ratio=0.01,
            resize_mask=256,
        )
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            '[{}] epoch: {}, {}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
            'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                ts,
                epoch,
                item,
                auroc_sp,
                ap_sp,
                f1_sp,
                auroc_px,
                ap_px,
                f1_px,
                aupro_px,
            ),
            flush=True,
        )

    mean_metrics = np.asarray(
        [
            np.mean(auroc_sp_list),
            np.mean(ap_sp_list),
            np.mean(f1_sp_list),
            np.mean(auroc_px_list),
            np.mean(ap_px_list),
            np.mean(f1_px_list),
            np.mean(aupro_px_list),
        ],
        dtype=np.float64,
    )
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        '[{}] epoch: {}, Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
        'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            ts,
            epoch,
            *mean_metrics,
        ),
        flush=True,
    )

    if writer is not None:
        metric_names = (
            'I-AUROC',
            'I-AP',
            'I-F1',
            'P-AUROC',
            'P-AP',
            'P-F1',
            'P-AUPRO',
        )
        for name, value in zip(metric_names, mean_metrics):
            writer.add_scalar(f'eval/{name}', float(value), epoch)

    model.train()
    return mean_metrics


def _best_f1_threshold(labels, scores):
    """Return the image-score threshold with the best reference F1."""

    from sklearn.metrics import precision_recall_curve

    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.unique(labels).size < 2:
        raise ValueError(
            'automatic Normal Shift threshold selection needs both normal and abnormal '
            'samples in the reference test set; pass --normal-shift-threshold explicitly'
        )
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError('unable to derive a threshold from the reference scores')
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_classification_scores(
    model,
    datasets,
    device,
    batch_size,
    save_dir,
    threshold=None,
    score_filename='evaluation_scores.csv',
):
    """Evaluate image-level metrics and one explicit decision threshold."""

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    parts = []
    for dataset in datasets:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        parts.append(
            collect_image_scores(
                model,
                dataloader,
                device,
                max_ratio=0.01,
                resize_mask=256,
            )
        )
    if not parts:
        raise ValueError('image-level evaluation needs a non-empty test set')

    labels = np.concatenate([part['labels'] for part in parts]).astype(np.int64)
    scores = np.concatenate([part['scores'] for part in parts])
    paths = sum((part['paths'] for part in parts), [])
    mask_ratios = np.concatenate([part['mask_area_ratios'] for part in parts])
    if np.unique(labels).size < 2:
        raise ValueError('image-level evaluation needs both normal and abnormal samples')
    if threshold is None:
        threshold = _best_f1_threshold(labels, scores)

    predictions = (scores >= threshold).astype(np.int64)
    normal = labels == 0
    abnormal = labels == 1
    tn = int(np.sum((predictions == 0) & normal))
    fp = int(np.sum((predictions == 1) & normal))
    tp = int(np.sum((predictions == 1) & abnormal))
    fn = int(np.sum((predictions == 0) & abnormal))
    normal_fpr = float(fp / max(int(normal.sum()), 1))
    defect_recall = float(tp / max(int(abnormal.sum()), 1))

    score_path = os.path.join(save_dir, score_filename)
    with open(score_path, 'w', newline='', encoding='utf-8') as score_file:
        writer = csv.writer(score_file)
        writer.writerow(
            ['path', 'label', 'mask_area_ratio', 'score', 'threshold', 'predicted_anomaly']
        )
        for path, label, mask_ratio, score, prediction in zip(
            paths,
            labels,
            mask_ratios,
            scores,
            predictions,
        ):
            writer.writerow(
                [path, int(label), float(mask_ratio), float(score), threshold, int(prediction)]
            )

    metrics = {
        'I-AUROC': float(roc_auc_score(labels, scores)),
        'I-AP': float(average_precision_score(labels, scores)),
        'I-F1': float(f1_score(labels, predictions)),
        'Decision Threshold': float(threshold),
        'Normal FPR': normal_fpr,
        'Defect Recall': defect_recall,
        'TN': tn,
        'FP': fp,
        'TP': tp,
        'FN': fn,
        'Normal Score Mean': float(scores[normal].mean()),
        'Abnormal Score Mean': float(scores[abnormal].mean()),
    }
    print(
        '[Image Eval] I-AUROC:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, threshold:{:.6f}, '
        'Normal FPR:{:.4f} ({}/{}), Defect Recall:{:.4f} ({}/{})'.format(
            metrics['I-AUROC'],
            metrics['I-AP'],
            metrics['I-F1'],
            threshold,
            normal_fpr,
            fp,
            int(normal.sum()),
            defect_recall,
            tp,
            int(abnormal.sum()),
        ),
        flush=True,
    )
    return metrics


def evaluate_normal_shift_model(
    model,
    reference_data_list,
    shift_path,
    data_transform,
    gt_transform,
    device,
    batch_size,
    save_dir,
    threshold=None,
    small_defect_max_area_ratio=0.01,
    cache=False,
    writer=None,
    production_target_fpr=0.01,
):
    """Measure Normal FPR and Shift Normal FPR at one fixed threshold."""

    reference_parts = []
    for dataset in reference_data_list:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        reference_parts.append(
            collect_image_scores(
                model,
                dataloader,
                device,
                max_ratio=0.01,
                resize_mask=256,
            )
        )
    if not reference_parts:
        raise ValueError('Normal Shift evaluation needs a non-empty reference test set')

    reference_labels = np.concatenate([part['labels'] for part in reference_parts])
    reference_scores = np.concatenate([part['scores'] for part in reference_parts])
    reference_mask_ratios = np.concatenate(
        [part['mask_area_ratios'] for part in reference_parts]
    )
    reference_paths = sum((part['paths'] for part in reference_parts), [])
    normal_scores = reference_scores[reference_labels == 0]
    if normal_scores.size == 0:
        raise ValueError('reference test set does not contain normal samples')
    if threshold is None:
        if production_target_fpr is None or production_target_fpr == 0:
            threshold = _best_f1_threshold(reference_labels, reference_scores)
        else:
            if not 0 < production_target_fpr < 1:
                raise ValueError('production_target_fpr must be in (0, 1)')
            threshold = float(
                np.quantile(normal_scores, 1.0 - production_target_fpr, method='higher')
            )
    normal_fpr = float(np.mean(normal_scores >= threshold))
    small_defect_mask = (
        (reference_labels == 1)
        & (reference_mask_ratios > 0)
        & (reference_mask_ratios <= small_defect_max_area_ratio)
    )
    small_defect_recall = None
    if np.any(small_defect_mask):
        small_defect_recall = float(
            np.mean(reference_scores[small_defect_mask] >= threshold)
        )

    shift_dataset_cls = CustomRAMDataset if cache else CustomDataset
    shift_dataset = shift_dataset_cls(
        root=shift_path,
        transform=data_transform,
        gt_transform=gt_transform,
        phase='test',
    )
    shift_loader = DataLoader(
        shift_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    shift_result = collect_image_scores(
        model,
        shift_loader,
        device,
        max_ratio=0.01,
        resize_mask=256,
    )
    if np.any(shift_result['labels'] != 0):
        raise ValueError(
            'Normal Shift dataset must contain only normal samples under test/good'
        )
    shift_fpr = float(np.mean(shift_result['scores'] >= threshold))

    score_path = os.path.join(save_dir, 'normal_shift_scores.csv')
    with open(score_path, 'w', newline='', encoding='utf-8') as score_file:
        score_writer = csv.writer(score_file)
        score_writer.writerow(
            [
                'split',
                'path',
                'label',
                'mask_area_ratio',
                'score',
                'threshold',
                'predicted_anomaly',
            ]
        )
        for path, label, mask_ratio, score in zip(
            reference_paths,
            reference_labels,
            reference_mask_ratios,
            reference_scores,
        ):
            score_writer.writerow(
                [
                    'reference',
                    path,
                    int(label),
                    float(mask_ratio),
                    float(score),
                    threshold,
                    int(score >= threshold),
                ]
            )
        for path, score in zip(shift_result['paths'], shift_result['scores']):
            score_writer.writerow(
                ['normal_shift', path, 0, 0.0, float(score), threshold, int(score >= threshold)]
            )

    print(
        '[Normal Shift] threshold:{:.6f}, Normal FPR:{:.4f}, Shift Normal FPR:{:.4f}, '
        'reference_normal:{}, shifted_normal:{}'.format(
            threshold,
            normal_fpr,
            shift_fpr,
            normal_scores.size,
            shift_result['scores'].size,
        ),
        flush=True,
    )
    if small_defect_recall is not None:
        print(
            '[Normal Shift] Small Defect Recall:{:.4f} (mask area <= {:.2%}, n={})'.format(
                small_defect_recall,
                small_defect_max_area_ratio,
                int(small_defect_mask.sum()),
            ),
            flush=True,
        )
    if writer is not None:
        writer.add_scalar('robustness/normal_fpr', normal_fpr, 0)
        writer.add_scalar('robustness/shift_normal_fpr', shift_fpr, 0)
        if small_defect_recall is not None:
            writer.add_scalar('robustness/small_defect_recall', small_defect_recall, 0)

    metrics = {
        'Normal Shift Threshold': float(threshold),
        'Production Target FPR': (
            None if production_target_fpr is None else float(production_target_fpr)
        ),
        'Normal FPR': normal_fpr,
        'Shift Normal FPR': shift_fpr,
    }
    if small_defect_recall is not None:
        metrics['Small Defect Recall'] = small_defect_recall
        metrics['Small Defect Max Area Ratio'] = float(small_defect_max_area_ratio)
    return metrics


def train(item_list, args):
    if getattr(args, 'train_mode', 'default') == 'mask_constraint':
        from dinomaly_2D_mask_constraint import train_mask

        return train_mask(item_list, args)

    setup_seed(args.seed)

    max_iters = args.max_iters
    batch_size = getattr(args, 'batch_size', TRAIN_BATCH_SIZE)
    image_size = args.image_size
    crop_size = args.crop_size

    data_transform, gt_transform = get_data_transforms(
        image_size,
        crop_size,
        resize_mode=args.resize_mode,
    )
    if args.train_augmentation == 'mild':
        train_transform = get_mild_normal_transforms(
            image_size,
            crop_size,
            translate_pixels=args.aug_translate,
            rotation_degrees=args.aug_rotation,
            brightness=args.aug_brightness,
            contrast=args.aug_contrast,
            resize_mode=args.resize_mode,
        )
    else:
        train_transform = data_transform

    train_data_list = []
    test_data_list = []
    if args.dataset == 'custom':
        # 4060 8G 单卡：支持 txt 列表（Path 自动处理中文/空格，需 ""）；若 data_path 是 train_*.txt，自动找同目录 test_*.txt 作评估
        from pathlib import Path
        data_path = Path(args.data_path).expanduser()
        test_path_arg = getattr(args, 'test_path', None)
        if test_path_arg:
            test_root = Path(test_path_arg).expanduser()
        elif data_path.is_file() and data_path.suffix.lower() == ".txt" and "train" in data_path.name:
            candidate = data_path.parent / data_path.name.replace("train", "test")
            test_root = candidate if candidate.is_file() else data_path
            if test_root is not candidate:
                print(f"[Dinomaly2 custom] txt test sibling not found: {candidate}, fallback to train txt for test (eval may be train-only)")
            else:
                print(f"[Dinomaly2 custom] txt split: train={data_path} -> test={test_root}")
        else:
            test_root = data_path
        train_data = CustomDataset(
            root=str(data_path),
            transform=train_transform,
            gt_transform=gt_transform,
            phase='train',
        )
        dataset_cls = CustomRAMDataset if args.cache else CustomDataset
        try:
            test_data = dataset_cls(
                root=str(test_root),
                transform=data_transform,
                gt_transform=gt_transform,
                phase='test',
            )
        except Exception as exc:
            print(f"[Dinomaly2 custom] test dataset load failed ({exc}), eval will be skipped")
            test_data = None
        train_data_list.append(train_data)
        if test_data is not None and len(test_data) > 0:
            test_data_list.append(test_data)
        else:
            print("[Dinomaly2 custom] no test data, training only")
    else:
        for i, item in enumerate(item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)

            train_data = ImageFolder(root=train_path, transform=train_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]

            dataset_cls = MVTecRAMDataset if args.cache else MVTecDataset
            test_data = dataset_cls(
                root=test_path,
                transform=data_transform,
                gt_transform=gt_transform,
                phase='test',
            )
            train_data_list.append(train_data)
            test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                                   drop_last=True)

    encoder_name = args.backbone
    # encoder_name = 'dinov2reg_vit_small_14'
    # encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    # encoder_name = 'dinov2_vit_base_14'
    # encoder_name = 'dino_vit_base_16'
    # encoder_name = 'ibot_vit_base_16'
    # encoder_name = 'mae_vit_base_16'
    # encoder_name = 'beitv2_vit_base_16'
    # encoder_name = 'beit_vit_base_16'
    # encoder_name = 'digpt_vit_base_16'
    # encoder_name = 'deit_vit_base_16'

    if args.lc == 0:  # layer to layer
        fuse_layer_encoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
        fuse_layer_decoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
    elif args.lc == 1:  # one group
        fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
    elif args.lc == 2:  # two group
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    elif args.lc == 3:  # three group
        fuse_layer_encoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
    elif args.lc == 4:  # four group
        fuse_layer_encoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
    elif args.lc == 11:  # sparse, one layer
        fuse_layer_encoder = [[7]]
        fuse_layer_decoder = [[7]]
    elif args.lc == 12:  # sparse, two layers
        fuse_layer_encoder = [[3], [7]]
        fuse_layer_decoder = [[3], [7]]
    elif args.lc == 14:  # sparse, four layers
        fuse_layer_encoder = [[1], [3], [5], [7]]
        fuse_layer_decoder = [[1], [3], [5], [7]]
    else:
        raise "loose constraint value not supported"

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    dropout = args.dropout
    bottleneck.append(nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)))
    bottleneck.append(nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
                                    nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout)))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=partial(LinearAttention2, eps=1e-8) if args.la else Attention)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    feature_adapters = []
    if args.source_dg:
        feature_adapters = [
            CanonicalizationAdapter(
                embed_dim,
                bottleneck_ratio=args.adapter_bottleneck_ratio,
                alpha_init=args.adapter_alpha_init,
            )
            for _ in fuse_layer_encoder
        ]

    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                     remove_class_token=False,
                     fuse_layer_encoder=fuse_layer_encoder,
                     fuse_layer_decoder=fuse_layer_decoder,
                     context_aware_recenter=args.cr,
                     feature_adapters=feature_adapters)
    model = model.to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    trainable_modules = [bottleneck, decoder]
    if model.feature_adapters:
        trainable_modules.insert(0, model.feature_adapters)
    trainable = nn.ModuleList(trainable_modules)

    robust_ot_loss = RobustSinkhornLoss(
        window_size=args.cross_view_uot_window_size if args.cross_view_uot else args.ot_window_size,
        stride=args.cross_view_uot_stride if args.cross_view_uot else args.ot_stride,
        epsilon=args.cross_view_uot_epsilon if args.cross_view_uot else args.ot_epsilon,
        rho=args.cross_view_uot_rho if args.cross_view_uot else args.ot_rho,
        iterations=args.cross_view_uot_iters if args.cross_view_uot else args.ot_iters,
        position_weight=args.cross_view_uot_position_weight if args.cross_view_uot else args.ot_position_weight,
        ot_type=args.ot_type,
    ).to(device)
    combined_loss = CombinedDinomalyLoss(
        robust_ot=robust_ot_loss,
        lambda_ot=args.lambda_ot,
        warmup_iters=args.ot_warmup_iters,
        enabled=args.robust_ot and not args.source_dg,
    )
    if args.robust_ot:
        print_fn(
            'robust OT: type={}, lambda={}, warmup={}, window={}, stride={}, '
            'epsilon={}, rho={}, iterations={}, position_weight={}'.format(
                args.ot_type,
                args.lambda_ot,
                args.ot_warmup_iters,
                args.ot_window_size,
                args.ot_stride,
                args.ot_epsilon,
                args.ot_rho,
                args.ot_iters,
                args.ot_position_weight,
            )
        )

    if args.source_dg and args.robust_ot:
        print_fn(
            'warning: --robust-ot is the legacy Encoder-Decoder OT path and is '
            'disabled when --source-dg is enabled; use --cross-view-uot explicitly.'
        )

    normal_shift_generator = None
    pseudo_anomaly_generator = None
    if args.source_dg:
        normal_shift_generator = NormalShiftGenerator(
            geometry_probability=args.geometry_probability,
            photometric_probability=args.photometric_probability,
            degradation_probability=args.degradation_probability,
            illumination_probability=args.illumination_probability,
            max_ops_per_view=args.max_ops_per_view,
            translate_fraction=args.translate_fraction,
            rotation_degrees=args.rotation_degrees,
            noise_std=args.noise_std,
            blur_sigma=args.blur_sigma,
        )
        if args.pseudo_anomaly:
            pseudo_anomaly_generator = PseudoAnomalyGenerator(
                min_area_ratio=args.pseudo_min_area_ratio,
                max_area_ratio=args.pseudo_max_area_ratio,
            )

    model.init_weights()

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError('--eval-only requires --checkpoint')
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=True)
        print_fn(f'evaluation checkpoint loaded: {args.checkpoint}')

        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tb'))
        metrics_dict = {}
        mean_metrics = evaluate_model(
            model,
            test_data_list,
            item_list,
            device,
            batch_size,
            epoch=0,
            writer=writer,
        )
        if mean_metrics is not None:
            metric_names = [
                'I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO'
            ]
            metrics_dict.update(
                {
                    name: round(float(value), 6)
                    for name, value in zip(metric_names, mean_metrics)
                }
            )

        classification_metrics = evaluate_classification_scores(
            model=model,
            datasets=test_data_list,
            device=device,
            batch_size=batch_size,
            save_dir=args.save_dir,
            threshold=args.decision_threshold,
        )
        metrics_dict.update(classification_metrics)

        if args.normal_shift_path:
            robustness_metrics = evaluate_normal_shift_model(
                model=model,
                reference_data_list=test_data_list,
                shift_path=args.normal_shift_path,
                data_transform=data_transform,
                gt_transform=gt_transform,
                device=device,
                batch_size=batch_size,
                save_dir=args.save_dir,
                threshold=classification_metrics['Decision Threshold'],
                small_defect_max_area_ratio=args.small_defect_max_area_ratio,
                cache=args.cache,
                writer=writer,
                production_target_fpr=args.production_target_fpr,
            )
            metrics_dict.update(robustness_metrics)

        with open(os.path.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
            json.dump(metrics_dict, file, indent=2)
        writer.close()
        return

    if args.source_dg:
        optimizer = StableAdamW([
            {'params': model.feature_adapters.parameters(), 'lr': args.adapter_lr},
            {'params': bottleneck.parameters(), 'lr': args.decoder_lr},
            {'params': decoder.parameters(), 'lr': args.decoder_lr},
        ], lr=args.decoder_lr, betas=(0.9, 0.999), weight_decay=1e-4,
           amsgrad=False, eps=1e-10)
    else:
        optimizer = StableAdamW([{'params': bottleneck[0].parameters(), 'lr': 2e-4},
                                 {'params': bottleneck[1].parameters()},
                                 {'params': decoder.parameters()}],
                                lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                                amsgrad=False, eps=1e-10)
    lr_scheduler = WarmupCosineScheduler(optimizer, final_ratio=args.lr_decay_ratio, total_epochs=max_iters,
                                         warmup_epochs=100)

    print_fn('train image number:{}'.format(len(train_data)))

    log_dir = os.path.join(args.save_dir, 'tb')
    writer = SummaryWriter(log_dir=log_dir)

    it = 0
    start_time = time.time()
    total_epochs =int(np.ceil(max_iters / len(train_dataloader)))
    for epoch in range(total_epochs):
        model.train()

        loss_list = []
        cosine_loss_list = []
        ot_loss_list = []
        ot_weight_list = []
        transport_mass_list = []
        dg_loss_lists = {
            'anchor_rec': [],
            'shift_rec': [],
            'inv': [],
            'identity': [],
            'map': [],
            'preserve': [],
            'margin': [],
            'uot': [],
            'hard_score': [],
        }
        hard_shift_families = []
        batch_cost_list = []
        data_cost_list = []
        t_loader = time.time()
        for img, label in train_dataloader:
            t_data = time.time() - t_loader

            img = img.to(device)

            if args.source_dg:
                progress = it / max(max_iters, 1)
                shift_stage = _phase_ramp(progress, 0.10, 0.20)
                consistency_stage = _phase_ramp(progress, 0.10, 0.20)
                map_stage = _phase_ramp(progress, 0.20, 0.60)
                en, de, anchor_details = model(img, return_features=True)
                hard_candidates = (
                    args.hard_shift_candidates
                    if it / max(max_iters, 1) >= args.hard_shift_start_ratio
                    else 1
                )
                shift_batch = _select_hard_shift(
                    model,
                    img,
                    normal_shift_generator,
                    hard_candidates,
                    device,
                )
                shifted_en, shifted_de, shifted_details = model(
                    shift_batch.images,
                    return_features=True,
                )
                hard_shift_families.extend(shift_batch.family)

                p_final = args.ll_ratio
                p = min(p_final * it / 1000, p_final)
                cosine_decoder = [feature.clone() for feature in de]
                anchor_rec = global_cosine_hm_percent(
                    en,
                    cosine_decoder,
                    p=p,
                    factor=args.ll_factor,
                ) if args.ll else global_cosine(en, cosine_decoder)
                shift_rec = global_cosine_hm_percent(
                    shifted_en,
                    shifted_de,
                    p=p,
                    factor=args.ll_factor,
                ) if args.ll else global_cosine(shifted_en, shifted_de)
                loss_cos = anchor_rec + args.lambda_shift * shift_stage * shift_rec
                inverse_warp = shift_batch.inverse_warp
                loss_inv = args.lambda_inv * consistency_stage * cosine_consistency(
                    anchor_details['canonical_en'],
                    shifted_details['canonical_en'],
                    inverse_warp=inverse_warp,
                )
                loss_identity = args.lambda_id * identity_loss(
                    anchor_details['canonical_en'],
                    anchor_details['raw_en'],
                )
                anchor_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
                shifted_map, _ = cal_anomaly_maps(
                    shifted_en,
                    shifted_de,
                    img.shape[-1],
                )
                loss_map = args.lambda_map * map_stage * map_consistency(
                    anchor_map,
                    shifted_map,
                    inverse_warp=inverse_warp,
                )
                loss_uot = loss_cos.new_zeros(())
                if args.cross_view_uot:
                    inverse_shifted = [
                        inverse_warp(feature)
                        for feature in shifted_details['canonical_en']
                    ]
                    loss_uot = args.lambda_uot * robust_ot_loss(
                        anchor_details['canonical_en'],
                        inverse_shifted,
                    )['loss_ot']
                loss_preserve = loss_cos.new_zeros(())
                loss_margin = loss_cos.new_zeros(())
                if (
                    pseudo_anomaly_generator is not None
                    and it / max(max_iters, 1) >= args.pseudo_start_ratio
                ):
                    pseudo_images, pseudo_masks = pseudo_anomaly_generator(img)
                    pseudo_en, pseudo_de, pseudo_details = model(
                        pseudo_images,
                        return_features=True,
                    )
                    loss_preserve = args.lambda_preserve * preservation_loss(
                        anchor_details['canonical_en'],
                        pseudo_details['canonical_en'],
                        pseudo_masks,
                        margin=args.preserve_feature_margin,
                    )
                    if args.lambda_margin > 0:
                        pseudo_map, _ = cal_anomaly_maps(
                            pseudo_en,
                            pseudo_de,
                            img.shape[-1],
                        )
                        loss_margin = args.lambda_margin * anomaly_margin_loss(
                            anchor_map.detach(),
                            pseudo_map,
                            pseudo_masks,
                            margin=args.margin,
                        )
                loss_result = {
                    'loss_total': loss_cos + loss_inv + loss_identity + loss_map + loss_preserve + loss_margin + loss_uot,
                    'loss_cos': loss_cos,
                    'loss_ot': loss_cos.new_zeros(()),
                    'lambda_ot': loss_cos.new_zeros(()),
                    'transport_mass': loss_cos.new_zeros(()),
                }
                dg_loss_lists['anchor_rec'].append(anchor_rec.item())
                dg_loss_lists['shift_rec'].append(shift_rec.item())
                dg_loss_lists['inv'].append(loss_inv.item())
                dg_loss_lists['identity'].append(loss_identity.item())
                dg_loss_lists['map'].append(loss_map.item())
                dg_loss_lists['preserve'].append(loss_preserve.item())
                dg_loss_lists['margin'].append(loss_margin.item())
                dg_loss_lists['uot'].append(loss_uot.item())
                dg_loss_lists['hard_score'].append(_feature_image_scores(shifted_en, shifted_de).mean().item())
            else:
                en, de = model(img)

                p_final = args.ll_ratio
                p = min(p_final * it / 1000, p_final)
                # Loose Loss registers a gradient hook. Route its branch through a
                # clone so the hook does not also attenuate the independent OT
                # gradient arriving at the shared decoder feature.
                use_ot_branch = args.robust_ot and args.lambda_ot > 0
                cosine_decoder = [feature.clone() for feature in de] if use_ot_branch else de
                if args.ll:
                    loss_cos = global_cosine_hm_percent(
                        en, cosine_decoder, p=p, factor=args.ll_factor
                    )
                else:
                    loss_cos = global_cosine(en, cosine_decoder)

                loss_result = combined_loss(loss_cos, en, de, iteration=it)
            loss = loss_result['loss_total']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_batch = time.time() - t_loader
            loss_list.append(loss.item())
            cosine_loss_list.append(loss_result['loss_cos'].item())
            ot_loss_list.append(loss_result['loss_ot'].item())
            ot_weight_list.append(loss_result['lambda_ot'].item())
            transport_mass_list.append(loss_result['transport_mass'].item())
            batch_cost_list.append(t_batch)
            data_cost_list.append(t_data)
            lr_scheduler.step()
            t_loader = time.time()

            it += 1
            if it == max_iters:
                break
            if (it + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (max_iters - it) / it if it > 0 else 0
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                avg_batch = np.mean(batch_cost_list)
                avg_data = np.mean(data_cost_list)
                data_pct = avg_data / avg_batch * 100 if avg_batch > 0 else 0
                mean_loss = np.mean(loss_list)
                mean_cosine_loss = np.mean(cosine_loss_list)
                mean_ot_loss = np.mean(ot_loss_list)
                mean_ot_weight = np.mean(ot_weight_list)
                mean_transport_mass = np.mean(transport_mass_list)
                if args.robust_ot:
                    print('[{}] epoch: [{}/{}], iter [{}/{}], loss_total:{:.4f}, loss_cos:{:.4f}, '
                          'loss_ot:{:.4f}, lambda_ot:{:.5f}, transport_mass:{:.4f}, batch:{:.3f}s, '
                          'data:{:.3f}s({:.1f}%), elapsed:{}, ETA:{}'.format(
                              ts, epoch, total_epochs, it, max_iters, mean_loss,
                              mean_cosine_loss, mean_ot_loss, mean_ot_weight,
                              mean_transport_mass, avg_batch, avg_data, data_pct,
                              _sec2hms(elapsed), _sec2hms(eta)), flush=True)
                else:
                    print('[{}] epoch: [{}/{}], iter [{}/{}], loss:{:.4f}, batch:{:.3f}s, data:{:.3f}s({:.1f}%), elapsed:{}, ETA:{}'.format(
                        ts, epoch, total_epochs, it, max_iters, mean_loss,
                        avg_batch, avg_data, data_pct, _sec2hms(elapsed), _sec2hms(eta)), flush=True)
                # print(f'[{ts}, epoch: [{epoch}/{total_epochs}]], iter: [{it}/{max_iters}], elapsed: {_sec2hms(elapsed)}, ETA: {_sec2hms(eta)}')
                writer.add_scalar('train/loss', mean_loss, it)
                writer.add_scalar('train/loss_total', mean_loss, it)
                writer.add_scalar('train/loss_cos', mean_cosine_loss, it)
                writer.add_scalar('train/loss_ot', mean_ot_loss, it)
                writer.add_scalar('train/lambda_ot', mean_ot_weight, it)
                writer.add_scalar('train/transport_mass', mean_transport_mass, it)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
                writer.add_scalar('train/batch_time', avg_batch, it)
                writer.add_scalar('train/data_pct', data_pct, it)
                if args.source_dg:
                    for name, values in dg_loss_lists.items():
                        if values:
                            writer.add_scalar(f'train/loss_{name}', float(np.mean(values)), it)
                    if hard_shift_families:
                        family_counts = Counter(hard_shift_families)
                        family_text = ', '.join(
                            f'{name}:{count}'
                            for name, count in sorted(family_counts.items())
                        )
                        writer.add_text('train/hard_shift_family', family_text, it)
                    adapter_norm = sum(
                        float(adapter.fc2.weight.detach().abs().mean())
                        for adapter in model.feature_adapters
                    ) / max(len(model.feature_adapters), 1)
                    writer.add_scalar('train/adapter_residual_norm', adapter_norm, it)
                loss_list = []
                cosine_loss_list = []
                ot_loss_list = []
                ot_weight_list = []
                transport_mass_list = []
                batch_cost_list = []
                data_cost_list = []
                dg_loss_lists = {name: [] for name in dg_loss_lists}
                hard_shift_families = []

        if not args.skip_eval and args.eval_interval > 0 and epoch % args.eval_interval == 0:
            try:
                evaluate_model(
                    model,
                    test_data_list,
                    item_list,
                    device,
                    batch_size,
                    epoch,
                    writer,
                )
            except Exception as e:
                print(f"[eval] epoch {epoch} failed: {e}", flush=True)

        if it >= max_iters:
            break

    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    print(f'Peak GPU Memory: {peak_gpu_mem_mb:.1f} MB', flush=True)

    metrics_dict = {"peak_gpu_mem_mb": round(peak_gpu_mem_mb, 2)}
    if not args.skip_eval:
        try:
            mean_metrics = evaluate_model(
                model,
                test_data_list,
                item_list,
                device,
                batch_size,
                total_epochs,
                writer,
            )
            if mean_metrics is not None:
                metric_names = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP', 'P-F1', 'P-AUPRO']
                for name, val in zip(metric_names, mean_metrics):
                    metrics_dict[name] = round(float(val), 6)
        except Exception as e:
            print(f"[eval] final evaluation failed: {e} (跳过，不影响模型保存)", flush=True)

    if args.normal_shift_path:
        try:
            robustness_metrics = evaluate_normal_shift_model(
                model=model,
                reference_data_list=test_data_list,
                shift_path=args.normal_shift_path,
                data_transform=data_transform,
                gt_transform=gt_transform,
                device=device,
                batch_size=batch_size,
                save_dir=args.save_dir,
                threshold=args.normal_shift_threshold,
                small_defect_max_area_ratio=args.small_defect_max_area_ratio,
                cache=args.cache,
                writer=writer,
                production_target_fpr=args.production_target_fpr,
            )
            metrics_dict.update(robustness_metrics)
        except Exception as e:
            print(f"[Normal Shift] evaluation failed: {e}", flush=True)

    try:
        writer.close()
    except Exception:
        pass

    try:
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))
        print(f"[save] model.pth saved to {os.path.join(args.save_dir, 'model.pth')}", flush=True)
    except Exception as e:
        print(f"[save] failed: {e}", flush=True)
        raise

    try:
        with open(os.path.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)
    except Exception:
        pass

    return


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../MPDD',
                        help='Support: mvtec_anomaly_detection, VisA_pytorch/1cls, MPDD, BTech_Dataset_transformed, MIAD, Uni-Medical')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['custom', 'mvtec'],
        default='custom',
        help='Dataset format: custom uses one data_path; mvtec keeps the original multi-category logic.',
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./saved_results',
        help='训练输出根目录；实际输出保存到其下的 YYYYMMDDHHMMSS 子目录。',
    )
    parser.add_argument('--backbone', type=str, default='dinov2reg_vit_small_14')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1).')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate for Noisy Bottleneck')
    parser.add_argument(
        '--train-augmentation', '--train_augmentation',
        choices=['none', 'mild'],
        default='none',
        dest='train_augmentation',
        help='Training-only augmentation; evaluation is always deterministic.',
    )
    parser.add_argument('--aug-translate', type=int, default=4, dest='aug_translate')
    parser.add_argument('--aug-rotation', type=float, default=2.0, dest='aug_rotation')
    parser.add_argument('--aug-brightness', type=float, default=0.05, dest='aug_brightness')
    parser.add_argument('--aug-contrast', type=float, default=0.05, dest='aug_contrast')
    parser.add_argument(
        '--source-dg', '--source_dg',
        action='store_true',
        dest='source_dg',
        help='Enable source-only normal-shift canonicalization and dual-view training.',
    )
    parser.add_argument('--adapter-bottleneck-ratio', '--adapter_bottleneck_ratio', type=float,
                        default=0.25, dest='adapter_bottleneck_ratio')
    parser.add_argument('--adapter-alpha-init', '--adapter_alpha_init', type=float,
                        default=0.0, dest='adapter_alpha_init')
    parser.add_argument('--adapter-lr', '--adapter_lr', type=float, default=2e-4, dest='adapter_lr')
    parser.add_argument('--decoder-lr', '--decoder_lr', type=float, default=5e-5, dest='decoder_lr')
    parser.add_argument('--lambda-shift', '--lambda_shift', type=float, default=1.0, dest='lambda_shift')
    parser.add_argument('--lambda-inv', '--lambda_inv', type=float, default=0.10, dest='lambda_inv')
    parser.add_argument('--lambda-id', '--lambda_id', type=float, default=0.05, dest='lambda_id')
    parser.add_argument('--lambda-map', '--lambda_map', type=float, default=0.05, dest='lambda_map')
    parser.add_argument('--lambda-preserve', '--lambda_preserve', type=float, default=0.05, dest='lambda_preserve')
    parser.add_argument('--lambda-margin', '--lambda_margin', type=float, default=0.0, dest='lambda_margin')
    parser.add_argument(
        '--cross-view-uot', '--cross_view_uot',
        action='store_true',
        dest='cross_view_uot',
        help='Optional Local UOT between inverse-warped anchor/shift canonical features.',
    )
    parser.add_argument('--lambda-uot', '--lambda_uot', type=float, default=0.01, dest='lambda_uot')
    parser.add_argument('--cross-view-uot-window-size', type=int, default=3,
                        dest='cross_view_uot_window_size')
    parser.add_argument('--cross-view-uot-stride', type=int, default=3,
                        dest='cross_view_uot_stride')
    parser.add_argument('--cross-view-uot-epsilon', type=float, default=0.05,
                        dest='cross_view_uot_epsilon')
    parser.add_argument('--cross-view-uot-rho', type=float, default=1.0,
                        dest='cross_view_uot_rho')
    parser.add_argument('--cross-view-uot-iters', type=int, default=20,
                        dest='cross_view_uot_iters')
    parser.add_argument('--cross-view-uot-position-weight', type=float, default=0.10,
                        dest='cross_view_uot_position_weight')
    parser.add_argument('--margin', type=float, default=0.05)
    parser.add_argument('--preserve-feature-margin', '--preserve_feature_margin', type=float,
                        default=0.10, dest='preserve_feature_margin')
    parser.add_argument('--hard-shift-candidates', '--hard_shift_candidates', type=int,
                        default=4, dest='hard_shift_candidates')
    parser.add_argument('--hard-shift-start-ratio', '--hard_shift_start_ratio', type=float,
                        default=0.20, dest='hard_shift_start_ratio')
    parser.add_argument('--geometry-probability', '--geometry_probability', type=float,
                        default=0.30, dest='geometry_probability')
    parser.add_argument('--photometric-probability', '--photometric_probability', type=float,
                        default=0.80, dest='photometric_probability')
    parser.add_argument('--degradation-probability', '--degradation_probability', type=float,
                        default=0.40, dest='degradation_probability')
    parser.add_argument('--illumination-probability', '--illumination_probability', type=float,
                        default=0.40, dest='illumination_probability')
    parser.add_argument('--max-ops-per-view', '--max_ops_per_view', type=int,
                        default=3, dest='max_ops_per_view')
    parser.add_argument('--translate-fraction', '--translate_fraction', type=float,
                        default=0.03, dest='translate_fraction')
    parser.add_argument('--rotation-degrees', '--rotation_degrees', type=float,
                        default=2.0, dest='rotation_degrees')
    parser.add_argument('--noise-std', '--noise_std', type=float, default=0.02, dest='noise_std')
    parser.add_argument('--blur-sigma', '--blur_sigma', type=float, default=1.2, dest='blur_sigma')
    parser.add_argument('--pseudo-anomaly', '--pseudo_anomaly', action='store_true', dest='pseudo_anomaly')
    parser.add_argument('--pseudo-start-ratio', '--pseudo_start_ratio', type=float,
                        default=0.60, dest='pseudo_start_ratio')
    parser.add_argument('--pseudo-min-area-ratio', '--pseudo_min_area_ratio', type=float,
                        default=1e-4, dest='pseudo_min_area_ratio')
    parser.add_argument('--pseudo-max-area-ratio', '--pseudo_max_area_ratio', type=float,
                        default=0.02, dest='pseudo_max_area_ratio')
    parser.add_argument('--la', type=int, default=1,
                        help='Linear Attention. 1 for yes, 0 for no.')
    parser.add_argument('--lc', type=int, default=2,
                        help='Loose Constraint. 1 for 1 group, 2 for 2 group, 0 for layer-to-layer.')
    parser.add_argument('--ll', type=int, default=1,
                        help='Loose Loss. 1 for yes, 0 for no.')
    parser.add_argument('--ll_ratio', type=float, default=0.9,
                        help='The ratio of discarded regions in Loose Loss. 0.9 (90%%) by default.')
    parser.add_argument('--ll_factor', type=float, default=0.1,
                        help='The ratio gradients of the discarded regions. 0.1 by default.')
    parser.add_argument(
        '--robust-ot', '--robust_ot',
        action='store_true',
        dest='robust_ot',
        help='Enable the local robust Sinkhorn auxiliary training loss.',
    )
    parser.add_argument(
        '--ot-type', '--ot_type',
        choices=['balanced', 'unbalanced'],
        default='unbalanced',
        dest='ot_type',
        help='Local transport type (default: unbalanced).',
    )
    parser.add_argument('--lambda-ot', '--lambda_ot', type=float, default=0.05, dest='lambda_ot')
    parser.add_argument(
        '--ot-warmup-iters', '--ot_warmup_iters',
        type=int,
        default=2000,
        dest='ot_warmup_iters',
    )
    parser.add_argument(
        '--ot-window-size', '--ot_window_size',
        type=int,
        default=5,
        dest='ot_window_size',
    )
    parser.add_argument('--ot-stride', '--ot_stride', type=int, default=5, dest='ot_stride')
    parser.add_argument('--ot-epsilon', '--ot_epsilon', type=float, default=0.05, dest='ot_epsilon')
    parser.add_argument('--ot-rho', '--ot_rho', type=float, default=0.5, dest='ot_rho')
    parser.add_argument('--ot-iters', '--ot_iters', type=int, default=20, dest='ot_iters')
    parser.add_argument(
        '--ot-position-weight', '--ot_position_weight',
        type=float,
        default=0.0,
        dest='ot_position_weight',
    )
    parser.add_argument('--cr', type=int, default=1,
                        help='Context-aware recentering. 1 for yes, 0 for no.')
    parser.add_argument('--image_size', type=int, default=672)
    parser.add_argument('--crop_size', type=int, default=672)
    parser.add_argument(
        '--resize-mode', '--resize_mode',
        choices=['stretch', 'center_square'],
        default='stretch',
        dest='resize_mode',
        help='stretch resizes the full frame; center_square crops a centered square first',
    )
    parser.add_argument(
        '--batch-size', '--batch_size',
        type=int,
        default=4,
        dest='batch_size',
        help='Batch size for training and evaluation (default: 4).',
    )
    parser.add_argument('--max-iters', '--max_iters', type=int, default=40000, dest='max_iters')
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=EVAL_EVERY_EPOCHS,
        help=(
            'Evaluate every N epochs in both modes; use -1 to evaluate only '
            'after the final training iteration.'
        ),
    )
    parser.add_argument('--lr_decay_ratio', type=float, default=1.)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--test_path', type=str, default=None,
                        help='可选：显式指定 test 列表/目录（txt 或目录）；未指定时若 --data_path 为 train_*.txt 则自动找同目录 test_*.txt（4060 单卡适配，Path 自动处理中文/空格）')
    parser.add_argument(
        '--skip-eval', '--skip_eval',
        action='store_true',
        dest='skip_eval',
        help='Skip evaluation during and after training.',
    )
    parser.add_argument(
        '--eval-only', '--eval_only',
        action='store_true',
        dest='eval_only',
        help='Load --checkpoint and evaluate without training.',
    )
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument(
        '--decision-threshold', '--decision_threshold',
        type=float,
        default=None,
        dest='decision_threshold',
        help='Fixed image-level decision threshold; default selects best F1 on this split.',
    )
    parser.add_argument(
        '--normal-shift-path', '--normal_shift_path',
        type=str,
        default=None,
        dest='normal_shift_path',
        help='Optional generated Normal Shift dataset root containing test/good.',
    )
    parser.add_argument(
        '--normal-shift-threshold', '--normal_shift_threshold',
        type=float,
        default=None,
        dest='normal_shift_threshold',
        help='Fixed image anomaly threshold; default selects best F1 on the reference test set.',
    )
    parser.add_argument(
        '--production-target-fpr', '--production_target_fpr',
        type=float,
        default=0.01,
        dest='production_target_fpr',
        help='Calibrate Normal Shift threshold from clean normal scores only (default: 1%%). '
             'Set to 0 to restore best-F1 reference calibration.',
    )
    parser.add_argument(
        '--small-defect-max-area-ratio', '--small_defect_max_area_ratio',
        type=float,
        default=0.01,
        dest='small_defect_max_area_ratio',
        help='Maximum foreground-mask ratio counted as a small defect (default: 0.01).',
    )
    parser.add_argument('--cache', action='store_true',
                        help='Cache test dataset in RAM.')
    parser.add_argument(
        '--train_mode',
        choices=['default', 'mask_constraint'],
        default='default',
        help=(
            'default: original Dinomaly2 training on Train/good; '
            'mask_constraint: one-pass training with optional four-value masks '
            'from Train/good and all non-good directories.'
        ),
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        default=None,
        help='Optional root directory used to resolve training masks.',
    )
    parser.add_argument(
        '--good_value',
        type=int,
        default=1,
        help='Integer value representing good/weak_ok pixels in a mask.',
    )
    parser.add_argument(
        '--anomaly_value',
        type=int,
        default=2,
        help='Integer value representing anomaly pixels in a mask.',
    )
    parser.add_argument(
        '--ignore_value',
        type=int,
        default=255,
        help='Integer value representing uncertain/ignored pixels in a mask (default: 255).',
    )
    parser.add_argument(
        '--mask_only',
        action='store_true',
        help='For samples with a Mask, compute the Dinomaly2 loss only on good and anomaly pixels.',
    )
    parser.add_argument(
        '--lambda_good',
        type=float,
        default=0.5,
        help='Weight of the Dinomaly2 loss restricted to good pixels.',
    )
    parser.add_argument(
        '--lambda_anomaly',
        type=float,
        default=0.5,
        help='Weight of the anomaly-region Dinomaly2 loss to maximize.',
    )
    parser.add_argument(
        '--aug_hflip_prob',
        type=float,
        default=0,
        help='Training random horizontal flip probability.',
    )
    parser.add_argument(
        '--aug_hue',
        type=float,
        default=0,
        help='Training hue jitter strength in [0, 0.5].',
    )
    args = parser.parse_args()

    if args.eval_interval == 0 or args.eval_interval < -1:
        parser.error('--eval_interval must be -1 or a positive integer.')
    if args.lambda_ot < 0:
        parser.error('--lambda-ot must be non-negative.')
    if args.ot_warmup_iters < 0:
        parser.error('--ot-warmup-iters must be non-negative.')
    if args.ot_window_size <= 0:
        parser.error('--ot-window-size must be positive.')
    if args.ot_stride <= 0 or args.ot_stride > args.ot_window_size:
        parser.error('--ot-stride must be in [1, --ot-window-size].')
    if args.ot_epsilon <= 0 or args.ot_rho <= 0 or args.ot_iters <= 0:
        parser.error('--ot-epsilon, --ot-rho, and --ot-iters must be positive.')
    if args.ot_position_weight < 0:
        parser.error('--ot-position-weight must be non-negative.')
    if args.aug_translate < 0 or args.aug_rotation < 0:
        parser.error('--aug-translate and --aug-rotation must be non-negative.')
    if args.aug_brightness < 0 or args.aug_contrast < 0:
        parser.error('--aug-brightness and --aug-contrast must be non-negative.')
    if not 0 < args.adapter_bottleneck_ratio <= 1:
        parser.error('--adapter-bottleneck-ratio must be in (0, 1].')
    if args.adapter_lr <= 0 or args.decoder_lr <= 0:
        parser.error('--adapter-lr and --decoder-lr must be positive.')
    if args.lambda_shift < 0 or args.lambda_inv < 0 or args.lambda_id < 0:
        parser.error('--lambda-shift, --lambda-inv, and --lambda-id must be non-negative.')
    if args.lambda_map < 0 or args.lambda_preserve < 0 or args.lambda_margin < 0:
        parser.error('--lambda-map, --lambda-preserve, and --lambda-margin must be non-negative.')
    if args.lambda_uot < 0:
        parser.error('--lambda-uot must be non-negative.')
    if args.cross_view_uot and not args.source_dg:
        parser.error('--cross-view-uot requires --source-dg.')
    if args.cross_view_uot_window_size <= 0 or args.cross_view_uot_stride <= 0:
        parser.error('cross-view UOT window and stride must be positive.')
    if args.cross_view_uot_stride > args.cross_view_uot_window_size:
        parser.error('cross-view UOT stride must not exceed window size.')
    if args.cross_view_uot_epsilon <= 0 or args.cross_view_uot_rho <= 0 or args.cross_view_uot_iters <= 0:
        parser.error('cross-view UOT epsilon, rho, and iterations must be positive.')
    if args.cross_view_uot_position_weight < 0:
        parser.error('cross-view UOT position weight must be non-negative.')
    if args.production_target_fpr < 0 or args.production_target_fpr >= 1:
        parser.error('--production-target-fpr must be in [0, 1).')
    if args.hard_shift_candidates <= 0:
        parser.error('--hard-shift-candidates must be positive.')
    if not 0 <= args.hard_shift_start_ratio <= 1:
        parser.error('--hard-shift-start-ratio must be in [0, 1].')
    for option_name in (
        'geometry_probability', 'photometric_probability',
        'degradation_probability', 'illumination_probability',
    ):
        if not 0 <= getattr(args, option_name) <= 1:
            parser.error(f'--{option_name.replace("_", "-")} must be in [0, 1].')
    if args.max_ops_per_view <= 0 or args.translate_fraction < 0 or args.rotation_degrees < 0:
        parser.error('--max-ops-per-view must be positive and geometry magnitudes non-negative.')
    if args.noise_std < 0 or args.blur_sigma < 0:
        parser.error('--noise-std and --blur-sigma must be non-negative.')
    if not 0 <= args.pseudo_start_ratio <= 1:
        parser.error('--pseudo-start-ratio must be in [0, 1].')
    if not 0 < args.pseudo_min_area_ratio <= args.pseudo_max_area_ratio:
        parser.error('pseudo anomaly area ratios must be positive and ordered.')
    if args.eval_only and not args.checkpoint:
        parser.error('--eval-only requires --checkpoint.')
    if not 0 < args.small_defect_max_area_ratio <= 1:
        parser.error('--small-defect-max-area-ratio must be in (0, 1].')

    if args.dataset == 'custom':
        item_name = os.path.basename(os.path.normpath(args.data_path)) or 'custom'
        item_list = [item_name]
    else:
        if 'mvtec' in args.data_path.lower():
            item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                         'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
        elif 'visa' in args.data_path.lower():
            item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                         'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        elif 'btech' in args.data_path.lower():
            item_list = ['01', '02', '03']
        elif 'mpdd' in args.data_path.lower():
            item_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
        elif 'miad' in args.data_path.lower():
            item_list = ['catenary_dropper', 'electrical_insulator', 'metal_welding', 'nut_and_bolt',
                         'photovoltaic_module', 'wind_turbine', 'witness_mark']
        elif 'medical' in args.data_path.lower():
            item_list = ['BraTS2021', 'LIVER', 'RESC']
        else:
            item_list = os.listdir(args.data_path)

    run_name = datetime.now().strftime('%Y%m%d%H%M%S')
    args.save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger(run_name, args.save_dir)
    print_fn = logger.info

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    print_fn(args)
    train(item_list, args)
