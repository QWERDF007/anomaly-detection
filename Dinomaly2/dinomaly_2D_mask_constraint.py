"""One-pass Dinomaly2 training with optional three-value region masks.

The legacy ``dinomaly_2D.py`` remains the default entry point. It dispatches
here only when ``--train_mode mask_constraint`` is selected.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    CustomDataset,
    CustomRAMDataset,
    MVTecDataset,
    MVTecRAMDataset,
    get_data_transforms,
    get_mask_constraint_train_transform,
)
from dinomaly_2D import (
    TRAIN_BATCH_SIZE,
    _sec2hms,
    evaluate_model,
    setup_seed,
)
from mask_constraint_dataset import MaskConstraintTrainDataset
from mask_constraint_losses import calculate_mask_constraint_losses
from models import vit_encoder
from models.uad import Dinomaly
from models.vision_transformer import Attention, LinearAttention2, Block as VitBlock
from optimizers import StableAdamW
from utils import evaluation_batch
from utils import WarmupCosineScheduler


def _layer_groups(loose_constraint: int):
    if loose_constraint == 0:
        groups = [[i] for i in range(8)]
    elif loose_constraint == 1:
        groups = [list(range(8))]
    elif loose_constraint == 2:
        groups = [list(range(4)), list(range(4, 8))]
    elif loose_constraint == 3:
        groups = [list(range(3)), list(range(3, 6)), [6, 7]]
    elif loose_constraint == 4:
        groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    elif loose_constraint == 11:
        groups = [[7]]
    elif loose_constraint == 12:
        groups = [[3], [7]]
    elif loose_constraint == 14:
        groups = [[1], [3], [5], [7]]
    else:
        raise ValueError(f"Unsupported loose constraint: {loose_constraint}")
    return groups, [list(group) for group in groups]


def build_mask_constraint_model(args, device):
    fuse_layer_encoder, fuse_layer_decoder = _layer_groups(args.lc)
    encoder = vit_encoder.load(args.backbone)

    if "small" in args.backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in args.backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in args.backbone:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported backbone architecture: {args.backbone}")

    bottleneck = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.Dropout(p=args.dropout),
            ),
            nn.Sequential(
                nn.Linear(256, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(p=args.dropout),
            ),
        ]
    )
    decoder = nn.ModuleList(
        [
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=partial(LinearAttention2, eps=1e-8)
                if args.la
                else Attention,
            )
            for _ in range(8)
        ]
    )
    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=args.cr,
    ).to(device)
    model.init_weights()
    return model, bottleneck, decoder


def _build_test_datasets(item_list, args, data_transform, gt_transform):
    test_data_list = []
    if args.dataset == "custom":
        dataset_cls = CustomRAMDataset if args.cache else CustomDataset
        try:
            test_data_list.append(
                dataset_cls(
                    root=args.data_path,
                    transform=data_transform,
                    gt_transform=gt_transform,
                    phase="test",
                )
            )
        except FileNotFoundError as error:
            print(f"Skip default evaluation: {error}", flush=True)
    else:
        for item in item_list:
            test_path = os.path.join(args.data_path, item)
            dataset_cls = MVTecRAMDataset if args.cache else MVTecDataset
            test_data_list.append(
                dataset_cls(
                    root=test_path,
                    transform=data_transform,
                    gt_transform=gt_transform,
                    phase="test",
                )
            )
    return test_data_list


def train_mask(item_list, args):
    setup_seed(1)
    max_iters = int(args.max_iters)
    # Keep the common dataset layout self-contained: when no override is
    # supplied, masks live below the dataset root in ``masks/``.
    mask_dir = args.mask_dir
    if mask_dir is None:
        mask_dir = os.path.join(args.data_path, "masks")
    batch_size = TRAIN_BATCH_SIZE
    data_transform, gt_transform = get_data_transforms(
        args.image_size,
        args.crop_size,
    )
    train_transform = get_mask_constraint_train_transform(
        args.image_size,
        args.crop_size,
        hflip_prob=args.aug_hflip_prob,
        brightness=args.aug_brightness,
        contrast=args.aug_contrast,
        hue=args.aug_hue,
    )
    train_data = MaskConstraintTrainDataset(
        root=args.data_path,
        image_transform=data_transform,
        image_size=args.image_size,
        crop_size=args.crop_size,
        mask_dir=mask_dir,
        good_value=args.good_value,
        anomaly_value=args.anomaly_value,
        joint_transform=train_transform,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    test_data_list = _build_test_datasets(
        item_list,
        args,
        data_transform,
        gt_transform,
    )

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    model, bottleneck, decoder = build_mask_constraint_model(args, device)
    trainable = nn.ModuleList([bottleneck, decoder])
    optimizer = StableAdamW(
        [
            {"params": bottleneck[0].parameters(), "lr": 2e-4},
            {"params": bottleneck[1].parameters()},
            {"params": decoder.parameters()},
        ],
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=False,
        eps=1e-10,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        final_ratio=args.lr_decay_ratio,
        total_epochs=max_iters,
        warmup_epochs=100,
    )
    output_root = os.path.join(args.save_dir, args.save_name)
    os.makedirs(output_root, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_root, "tb"))
    print(f"train image number: {len(train_data)}", flush=True)
    print(
        f"mask values: BG=0, good={args.good_value}, "
        f"anomaly={args.anomaly_value}",
        flush=True,
    )

    iteration = 0
    start_time = time.time()
    loss_window = []
    good_window = []
    anomaly_window = []
    for epoch in range(max(1, math.ceil(max_iters / len(train_loader)))):
        model.train()
        for image, masks, has_mask, _paths in train_loader:
            if iteration >= max_iters:
                break
            batch_start = time.time()
            image = image.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            has_mask = has_mask.to(device, non_blocking=True)
            encoder_features, decoder_features = model(image)
            p_final = args.ll_ratio
            p = min(p_final * iteration / 1000.0, p_final)
            loss_dinomaly, loss_good, loss_anomaly = (
                calculate_mask_constraint_losses(
                    encoder_features,
                    decoder_features,
                    masks,
                    has_mask,
                    good_value=args.good_value,
                    anomaly_value=args.anomaly_value,
                    use_loose_loss=bool(args.ll),
                    p=p,
                    factor=args.ll_factor,
                )
            )
            good_term = args.lambda_good * loss_good
            anomaly_term = -args.lambda_anomaly * loss_anomaly
            total_loss = loss_dinomaly + good_term + anomaly_term
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()

            loss_window.append(float(loss_dinomaly.detach().cpu()))
            good_window.append(float(loss_good.detach().cpu()))
            anomaly_window.append(float(loss_anomaly.detach().cpu()))
            iteration += 1
            torch.cuda.synchronize()

            if iteration % 100 == 0 or iteration == max_iters:
                elapsed = time.time() - start_time
                eta = elapsed * (max_iters - iteration) / max(iteration, 1)
                mean_loss = float(np.mean(loss_window))
                mean_good = float(np.mean(good_window))
                mean_anomaly = float(np.mean(anomaly_window))
                mean_good_term = args.lambda_good * mean_good
                mean_anomaly_term = -args.lambda_anomaly * mean_anomaly
                mean_total = mean_loss + mean_good_term + mean_anomaly_term
                print(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"iter [{iteration}/{max_iters}], "
                    f"loss={mean_total:.4f}, "
                    f"dinomaly={mean_loss:.4f}, good={mean_good:.4f}, "
                    f"anomaly={mean_anomaly:.4f}, "
                    f"batch={time.time() - batch_start:.3f}s, "
                    f"elapsed={_sec2hms(elapsed)}, ETA={_sec2hms(eta)}",
                    flush=True,
                )
                writer.add_scalar("train/loss", mean_loss, iteration)
                writer.add_scalar("train/loss_good", mean_good, iteration)
                writer.add_scalar("train/loss_anomaly", mean_anomaly, iteration)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
                loss_window.clear()
                good_window.clear()
                anomaly_window.clear()
        if iteration >= max_iters:
            break

        if args.eval_interval > 0 and epoch % args.eval_interval == 0:
            evaluate_model(
                model,
                test_data_list,
                item_list,
                device,
                batch_size,
                epoch,
                writer,
            )

    if args.eval_interval == -1:
        evaluate_model(
            model,
            test_data_list,
            item_list,
            device,
            batch_size,
            math.ceil(max_iters / len(train_loader)),
            writer,
        )
    model_path = os.path.join(output_root, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"save to {model_path}", flush=True)
    writer.close()
    return


if __name__ == "__main__":
    # Keep one argument surface for both entry points. Running this file
    # defaults to the new mode, while an explicit --train_mode can still
    # select the legacy training path.
    import runpy
    import sys
    from pathlib import Path

    if "--train_mode" not in sys.argv:
        sys.argv[1:1] = ["--train_mode", "mask_constraint"]
    runpy.run_path(
        str(Path(__file__).with_name("dinomaly_2D.py")),
        run_name="__main__",
    )
