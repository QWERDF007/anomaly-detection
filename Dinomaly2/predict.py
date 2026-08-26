"""Predict anomaly maps for images using a trained Dinomaly2 model."""
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time
from datetime import datetime
from PIL import Image
from torchvision import transforms
from functools import partial
import cv2

from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from utils import cal_anomaly_maps, min_max_norm, cvt2heatmap, show_cam_on_image, get_gaussian_kernel


def build_model(args, device):
    if args.lc == 0:
        fuse_enc = fuse_dec = [[0], [1], [2], [3], [4], [5], [6], [7]]
    elif args.lc == 1:
        fuse_enc = fuse_dec = [[0, 1, 2, 3, 4, 5, 6, 7]]
    elif args.lc == 2:
        fuse_enc = fuse_dec = [[0, 1, 2, 3], [4, 5, 6, 7]]
    elif args.lc == 3:
        fuse_enc = fuse_dec = [[0, 1, 2], [3, 4, 5], [6, 7]]
    elif args.lc == 4:
        fuse_enc = fuse_dec = [[0, 1], [2, 3], [4, 5], [6, 7]]
    elif args.lc == 11:
        fuse_enc = fuse_dec = [[7]]
    elif args.lc == 12:
        fuse_enc = fuse_dec = [[3], [7]]
    elif args.lc == 14:
        fuse_enc = fuse_dec = [[1], [3], [5], [7]]
    else:
        raise ValueError(f"Unsupported lc: {args.lc}")

    encoder = vit_encoder.load(args.backbone)

    if 'small' in args.backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'base' in args.backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'large' in args.backbone:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unknown backbone size: {args.backbone}")

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=args.dropout)),
        nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=args.dropout),
                      nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=args.dropout)),
    ])

    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=partial(LinearAttention2, eps=1e-8) if args.la else Attention)
        for _ in range(8)
    ])

    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                     target_layers=target_layers, remove_class_token=False,
                     fuse_layer_encoder=fuse_enc, fuse_layer_decoder=fuse_dec,
                     context_aware_recenter=args.cr)
    return model.to(device)


def predict_one(model, img_path, transform, device, threshold=None):
    """Return (anomaly_map_uint8, heatmap_bgr, overlay_bgr, anomaly_score)."""
    t0 = time.time()
    img = Image.open(img_path).convert('RGB')
    orig = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(device)
    t_data = time.time() - t0

    model.eval()
    use_cuda = "cuda" in str(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda, dtype=torch.float16):
        en, de = model(img_tensor)
        anomaly_map, _ = cal_anomaly_maps(en, de, orig.shape[:2])
        anomaly_map = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)(anomaly_map)
    t_infer = time.time() - t0 - t_data

    amap = anomaly_map[0, 0].cpu().numpy()
    if threshold is not None:
        amap_norm = (np.clip(amap, 0, threshold) / threshold * 255).astype(np.uint8)
    else:
        amap_norm = (min_max_norm(amap) * 255).astype(np.uint8)
    heatmap = cvt2heatmap(amap_norm)
    overlay = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
    score = float(amap.max())

    return amap_norm, heatmap, overlay, score, t_data, t_infer,amap


def main():
    parser = argparse.ArgumentParser(description='Predict anomaly maps with Dinomaly2')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model.pth')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--backbone', type=str, default='dinov2reg_vit_small_14')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--cr', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for heatmap normalization. If None, use image-based min-max normalization.')
    args = parser.parse_args()

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = build_model(args, device)
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f'Model loaded from {args.model}')

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(args.crop_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        img_paths = sorted(p for p in os.listdir(args.input)
                          if os.path.splitext(p)[1].lower() in exts)
        img_paths = [os.path.join(args.input, p) for p in img_paths]
    else:
        img_paths = [args.input]

    total = len(img_paths)
    times = []
    scores=[]
    for i, img_path in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]

        t0 = time.time()
        amap, heatmap, overlay, score, t_data, t_infer,amap_s = predict_one(model, img_path, transform, device, args.threshold)
        elapsed = time.time() - t0
        times.append(elapsed)

        # cv2.imwrite(os.path.join(args.output, f'{name}_amap.png'), amap)
        # cv2.imwrite(os.path.join(args.output, f'{name}_heatmap.png'), heatmap)
        # cv2.imwrite(os.path.join(args.output, f'{name}_overlay.jpg'), overlay)

        np.save(os.path.join(args.output, f'{name}.npy'), amap_s)

        scores.append(score)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        eta = np.mean(times) * (total - i - 1)
        print(f'[{ts}] [{i+1}/{total}] {name}: score={score:.4f}, time={elapsed:.3f}s (data:{t_data:.3f}s infer:{t_infer:.3f}s), ETA={eta:.1f}s')

    print(f'Done. avg={np.mean(times):.3f}s/img, total={np.sum(times):.1f}s')

    max_score = np.max(scores)
    mean_score=np.mean(scores)
    std_score=np.std(scores)

    print(f"最大分数: {max_score:.4f}")
    print(f"平均分数: {mean_score:.4f}")
    print(f"分数标准差: {std_score:.4f}")

if __name__ == '__main__':
    main()
