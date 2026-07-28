# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
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
from utils import evaluation_batch, global_cosine, global_cosine_hm_percent, WarmupCosineScheduler
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# Keep the legacy and mask-constraint training loops on the same schedule.
TRAIN_BATCH_SIZE = 8
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


def train(item_list, args):
    if getattr(args, 'train_mode', 'default') == 'mask_constraint':
        from dinomaly_2D_mask_constraint import train_mask

        return train_mask(item_list, args)

    setup_seed(1)

    max_iters = args.max_iters
    batch_size = TRAIN_BATCH_SIZE
    image_size = args.image_size
    crop_size = args.crop_size

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    if args.dataset == 'custom':
        train_data = CustomDataset(
            root=args.data_path,
            transform=data_transform,
            gt_transform=gt_transform,
            phase='train',
        )
        dataset_cls = CustomRAMDataset if args.cache else CustomDataset
        test_data = dataset_cls(
            root=args.data_path,
            transform=data_transform,
            gt_transform=gt_transform,
            phase='test',
        )
        train_data_list.append(train_data)
        test_data_list.append(test_data)
    else:
        for i, item in enumerate(item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)

            train_data = ImageFolder(root=train_path, transform=data_transform)
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

    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                     remove_class_token=False,
                     fuse_layer_encoder=fuse_layer_encoder,
                     fuse_layer_decoder=fuse_layer_decoder,
                     context_aware_recenter=args.cr)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    model.init_weights()

    optimizer = StableAdamW([{'params': bottleneck[0].parameters(), 'lr': 2e-4},
                             {'params': bottleneck[1].parameters()},
                             {'params': decoder.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)
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
        batch_cost_list = []
        data_cost_list = []
        t_loader = time.time()
        for img, label in train_dataloader:
            t_data = time.time() - t_loader

            img = img.to(device)

            en, de = model(img)

            p_final = args.ll_ratio
            p = min(p_final * it / 1000, p_final)
            if args.ll:
                loss = global_cosine_hm_percent(en, de, p=p, factor=args.ll_factor)
            else:
                loss = global_cosine(en, de)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            torch.cuda.synchronize()
            t_batch = time.time() - t_loader
            loss_list.append(loss.item())
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
                print('[{}] epoch: [{}/{}], iter [{}/{}], loss:{:.4f}, batch:{:.3f}s, data:{:.3f}s({:.1f}%), elapsed:{}, ETA:{}'.format(
                    ts, epoch, total_epochs, it, max_iters, mean_loss,
                    avg_batch, avg_data, data_pct, _sec2hms(elapsed), _sec2hms(eta)), flush=True)
                # print(f'[{ts}, epoch: [{epoch}/{total_epochs}]], iter: [{it}/{max_iters}], elapsed: {_sec2hms(elapsed)}, ETA: {_sec2hms(eta)}')
                writer.add_scalar('train/loss', mean_loss, it)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
                writer.add_scalar('train/batch_time', avg_batch, it)
                writer.add_scalar('train/data_pct', data_pct, it)
                loss_list = []
                batch_cost_list = []
                data_cost_list = []

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

        if it >= max_iters:
            break

    if args.eval_interval == -1:
        evaluate_model(
            model,
            test_data_list,
            item_list,
            device,
            batch_size,
            total_epochs,
            writer,
        )

    writer.close()
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))

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
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate for Noisy Bottleneck')
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
    parser.add_argument('--cr', type=int, default=1,
                        help='Context-aware recentering. 1 for yes, 0 for no.')
    parser.add_argument('--image_size', type=int, default=672)
    parser.add_argument('--crop_size', type=int, default=672)
    parser.add_argument('--max-iters', type=int, default=40000)
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
        help=(
            'Optional root directory used to resolve training masks. '
            'Defaults to <data_path>/masks.'
        ),
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
        help=(
            'Integer value representing uncertain/ignored pixels in a mask. '
            'These pixels are excluded from all three mask-constraint losses '
            '(default: 255).'
        ),
    )
    parser.add_argument(
        '--mask_only',
        action='store_true',
        help=(
            'For samples with a Mask, compute the Dinomaly2 loss only on '
            'good and anomaly pixels; BG and ignored pixels are skipped. '
            'Samples without a Mask still use the full-image loss.'
        ),
    )
    parser.add_argument(
        '--aug_hflip_prob',
        type=float,
        default=0,
        help='Training random horizontal flip probability. Set 0 to disable.',
    )
    parser.add_argument(
        '--aug_brightness',
        type=float,
        default=0,
        help='Training brightness jitter strength. Set 0 to disable.',
    )
    parser.add_argument(
        '--aug_contrast',
        type=float,
        default=0,
        help='Training contrast jitter strength. Set 0 to disable.',
    )
    parser.add_argument(
        '--aug_hue',
        type=float,
        default=0,
        help='Training hue jitter strength in [0, 0.5]. Set 0 to disable.',
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
    args = parser.parse_args()

    if args.eval_interval == 0 or args.eval_interval < -1:
        parser.error('--eval_interval must be -1 or a positive integer.')
    if not 0 <= args.aug_hflip_prob <= 1:
        parser.error('--aug_hflip_prob must be in [0, 1].')
    if not 0 <= args.ignore_value <= 255:
        parser.error('--ignore_value must be in [0, 255].')
    if args.ignore_value == 0:
        parser.error('--ignore_value must differ from BG value 0.')
    if args.ignore_value in {args.good_value, args.anomaly_value}:
        parser.error(
            '--ignore_value must differ from --good_value and --anomaly_value.'
        )
    if args.aug_brightness < 0:
        parser.error('--aug_brightness must be non-negative.')
    if args.aug_contrast < 0:
        parser.error('--aug_contrast must be non-negative.')
    if not 0 <= args.aug_hue <= 0.5:
        parser.error('--aug_hue must be in [0, 0.5].')

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
