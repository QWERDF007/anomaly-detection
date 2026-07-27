"""Get maximum anomaly scores for images using a trained Dinomaly2 model."""
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
from functools import partial
import matplotlib.pyplot as plt

from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from utils import cal_anomaly_maps, get_gaussian_kernel


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


def get_score(model, img_path, transform, device):
    """Return the maximum anomaly score for an image."""
    img = Image.open(img_path).convert('RGB')
    orig = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        en, de = model(img_tensor)
        anomaly_map, _ = cal_anomaly_maps(en, de, orig.shape[:2])
        anomaly_map = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)(anomaly_map)

    amap = anomaly_map[0, 0].cpu().numpy()
    score = float(amap.max())

    return score


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def iter_image_paths(root):
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {root}')
    return sorted(
        [
            path for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def collect_split_scores(model, split_dir, split_name, transform, device):
    split_dir = Path(split_dir)
    image_paths = iter_image_paths(split_dir)
    grouped_scores = {}

    for image_path in image_paths:
        try:
            relative_parts = image_path.relative_to(split_dir).parts
        except ValueError:
            relative_parts = image_path.parts
        is_good = (
            bool(relative_parts)
            and relative_parts[0].lower() in {'good', 'normal'}
        )
        category = 'Good' if is_good else 'Anomaly'
        label = f'{split_name} / {category}'

        score = get_score(model, str(image_path), transform, device)
        grouped_scores.setdefault(label, []).append(score)
        print(f'{label}/{image_path.stem}: {score:.4f}')

    return grouped_scores


def find_child_directory(root, name):
    root = Path(root)
    if not root.is_dir():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def collect_input_scores(model, input_dir, transform, device):
    input_dir = Path(input_dir)
    train_dir = find_child_directory(input_dir, 'train')
    test_dir = find_child_directory(input_dir, 'test')

    if train_dir is not None or test_dir is not None:
        grouped_scores = {}
        if train_dir is not None:
            grouped_scores.update(
                collect_split_scores(
                    model, train_dir, 'Train', transform, device
                )
            )
        if test_dir is not None:
            grouped_scores.update(
                collect_split_scores(
                    model, test_dir, 'Test', transform, device
                )
            )
        return grouped_scores

    # Backward-compatible layout: input/good and input/<anomaly_type>.
    grouped_scores = {}
    for subdir in sorted(
        [path for path in input_dir.iterdir() if path.is_dir()],
        key=lambda path: str(path).lower(),
    ):
        category = (
            'Good'
            if subdir.name.lower() in {'good', 'normal'}
            else 'Anomaly'
        )
        label = category
        for image_path in iter_image_paths(subdir):
            score = get_score(model, str(image_path), transform, device)
            grouped_scores.setdefault(label, []).append(score)
            print(f'{label}/{image_path.stem}: {score:.4f}')
    return grouped_scores


def plot_score_distributions(grouped_scores, output_path, bins=30):
    all_scores = [
        score
        for scores in grouped_scores.values()
        for score in scores
    ]
    if not all_scores:
        raise ValueError('No valid images were found for score evaluation.')

    low = float(min(all_scores))
    high = float(max(all_scores))
    if high <= low:
        margin = max(abs(low) * 0.05, 1e-6)
        bin_edges = np.linspace(low - margin, high + margin, bins + 1)
    else:
        bin_edges = np.linspace(low, high, bins + 1)

    colors = {
        'Train / Good': 'green',
        'Train / Anomaly': 'orange',
        'Test / Good': 'blue',
        'Test / Anomaly': 'red',
        'Good': 'green',
        'Anomaly': 'red',
    }
    plt.figure(figsize=(10, 6))
    for label, scores in grouped_scores.items():
        if not scores:
            continue
        plt.hist(
            scores,
            bins=bin_edges,
            alpha=0.65,
            color=colors.get(label, 'steelblue'),
            label=f'{label} (n={len(scores)})',
        )

    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_score_summary(grouped_scores):
    print('\n=== Summary ===')
    for label, scores in grouped_scores.items():
        if not scores:
            continue
        print(f'{label}: {len(scores)} samples')
        print(f'  max score: {max(scores):.4f}')
        print(f'  mean score: {np.mean(scores):.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='Get maximum anomaly scores with Dinomaly2'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model.pth'
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help='Legacy input image or dataset directory.'
    )
    parser.add_argument(
        '--train', type=str, default=None,
        help='Train directory, normally containing train/good.'
    )
    parser.add_argument(
        '--test', type=str, default=None,
        help='Test directory, normally containing test/good and test/<anomaly>.'
    )
    parser.add_argument(
        '--plot_output', type=str, default='score_distribution.png',
        help='Output plot path. Default: score_distribution.png'
    )
    parser.add_argument('--bins', type=int, default=30)
    parser.add_argument('--backbone', type=str, default='dinov2reg_vit_small_14')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--cr', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    if args.input is None and args.train is None and args.test is None:
        parser.error('At least one of --input, --train, or --test is required.')
    if args.bins < 1:
        parser.error('--bins must be greater than 0.')

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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if args.train is not None or args.test is not None:
        grouped_scores = {}
        if args.train is not None:
            grouped_scores.update(
                collect_split_scores(
                    model, args.train, 'Train', transform, device
                )
            )
        if args.test is not None:
            grouped_scores.update(
                collect_split_scores(
                    model, args.test, 'Test', transform, device
                )
            )
        plot_score_distributions(
            grouped_scores, args.plot_output, bins=args.bins
        )
        print_score_summary(grouped_scores)
        print(f'\nDistribution plot saved to {args.plot_output}')
        return

    if os.path.isdir(args.input):
        grouped_scores = collect_input_scores(
            model, args.input, transform, device
        )
        plot_score_distributions(
            grouped_scores, args.plot_output, bins=args.bins
        )
        print_score_summary(grouped_scores)
        print(f'\nDistribution plot saved to {args.plot_output}')
    else:
        score = get_score(model, args.input, transform, device)
        name = os.path.splitext(os.path.basename(args.input))[0]
        print(f'{name}: {score:.4f}')


if __name__ == '__main__':
    main()
