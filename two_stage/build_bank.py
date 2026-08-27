#!/usr/bin/env python
"""Build feature bank (good/anomaly) from Dinomaly2 encoder without re-training.

Usage (PowerShell, 路径含空格/中文需 ""):
  D:\\Software\\anaconda3\\envs\\py312\\python.exe two_stage/build_bank.py `
    --model "F:\\tmp\\outs\\dinomaly2_n400_s448_seed2024\\*\\model.pth" `
    --data_dir "F:\\data\\异常检测测试报告数据\\铜色异常检测6相机_建库数据" `
    --save_bank "F:\\tmp\\feature_bank.npz" `
    --image_size 448 --cuda 0

Supports:
  - --model 可为 glob（如 *\\model.pth），自动取最新
  - --data_dir 为扁平 OK/NG 目录或带 train/test 的目录
  - 4060 8G 单卡：自动 batch 4 (448) / 2 (672)，faiss-cpu 可选
  - 路径全部用 Path 处理中文/空格

Output: .npz 包含 good_features, anomaly_features, meta；若指定 --save_dir 则另存 faiss 索引
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Allow import of Dinomaly2 modules
ROOT = Path(__file__).resolve().parents[1]
DINOMALY2 = ROOT / "Dinomaly2"
if str(DINOMALY2) not in sys.path:
    sys.path.insert(0, str(DINOMALY2))

IMAGE_EXTS = {".bmp",".jpeg",".jpg",".png",".tif",".tiff",".webp"}

def resolve_model(path_str: str) -> Path:
    # 支持 glob 与通配
    candidates = glob.glob(path_str, recursive=True)
    if candidates:
        # 取最新修改时间
        candidates = sorted([Path(p) for p in candidates if Path(p).is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            print(f"[build_bank] model glob {path_str!r} -> {candidates[0]} (from {len(candidates)} candidates)")
            return candidates[0]
    p = Path(path_str).expanduser()
    if p.is_file():
        return p
    # 尝试在 save_dir 下递归找 model.pth
    if p.is_dir():
        found = list(p.rglob("model.pth"))
        if found:
            found = sorted(found, key=lambda x: x.stat().st_mtime, reverse=True)
            print(f"[build_bank] found {len(found)} model.pth under {p}, using {found[0]}")
            return found[0]
    raise FileNotFoundError(f"Model not found: {path_str}")

def iter_images(directory: Path):
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: str(p).lower())

def build_parser():
    p = argparse.ArgumentParser(description="Build Dinomaly2 feature bank (good/anomaly) for two-stage")
    p.add_argument("--model", type=str, required=True, help="Dinomaly2 model.pth path or glob (e.g. F:\\tmp\\outs\\...\\*\\model.pth)")
    p.add_argument("--data_dir", type=str, required=True, help="建库数据目录，含 OK/NG 子目录 (中文/空格需 \"\")")
    p.add_argument("--save_bank", type=str, required=True, help="输出 npz 路径，如 F:\\tmp\\feature_bank.npz")
    p.add_argument("--save_dir", type=str, default=None, help="可选：同时保存 faiss 索引的目录")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--crop_size", type=int, default=None, help="默认同 image_size")
    p.add_argument("--batch_size", type=int, default=None, help="默认 448->4, 672->2 (4060 8G)")
    p.add_argument("--backbone", type=str, default="dinov2reg_vit_small_14", help="需与训练时一致")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--ok_names", type=str, nargs="+", default=["OK","ok","good","normal","Good"])
    return p

def main():
    args = build_parser().parse_args()
    model_path = resolve_model(args.model)
    data_dir = Path(args.data_dir).expanduser().resolve()
    save_bank = Path(args.save_bank).expanduser().resolve()
    crop_size = args.crop_size if args.crop_size else args.image_size
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 2 if args.image_size >= 672 else 4
        print(f"[build_bank] 4060 8G auto batch_size={batch_size} for image_size={args.image_size}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    save_bank.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >=0 else "cpu")
    print(f"[build_bank] device={device}, model={model_path}")
    print(f"[build_bank] data_dir={data_dir} -> save_bank={save_bank}")

    # 收集 OK/NG
    ok_lower = {n.lower() for n in args.ok_names}
    ok_images = []
    ng_images = []
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if subdirs:
        for sub in subdirs:
            if sub.name.lower() in ok_lower:
                ok_images.extend(iter_images(sub))
            else:
                # 视为 anomaly（NG）
                ng_images.extend(iter_images(sub))
    else:
        # flat 目录全部视为 good
        ok_images = iter_images(data_dir)

    print(f"[build_bank] OK={len(ok_images)}, NG={len(ng_images)} (建库耗时仅 1s 级)")

    # 构建 Dinomaly 模型
    # 复用 Dinomaly2/dinomaly_two_stage 的 build/test 逻辑，但此处轻量实现：用 encoder 直接提特征
    from models.uad import Dinomaly
    from models import vit_encoder
    from functools import partial
    import torch.nn as nn
    from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2

    # 加载 checkpoint
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict):
        for k in ("state_dict","model_state_dict","model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    # 去掉 module. 前缀
    if ckpt and all(k.startswith("module.") for k in ckpt):
        ckpt = {k[len("module."):]:v for k,v in ckpt.items()}

    backbone = args.backbone
    if "bottleneck.0.0.weight" in ckpt:
        in_dim = ckpt["bottleneck.0.0.weight"].shape[1]
        if in_dim == 384 and "small" not in backbone:
            backbone = "dinov2reg_vit_small_14"
        elif in_dim == 768 and "base" not in backbone:
            backbone = "dinov2reg_vit_base_14"
        elif in_dim == 1024 and "large" not in backbone:
            backbone = "dinov2reg_vit_large_14"

    encoder = vit_encoder.load(backbone)
    if "small" in backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2,3,4,5,6,7,8,9]
    elif "base" in backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2,3,4,5,6,7,8,9]
    elif "large" in backbone:
        embed_dim, num_heads = 1024, 16
        target_layers = [4,6,8,10,12,14,16,18]
    else:
        raise ValueError(f"Unknown backbone {backbone}")

    fuse_layer_encoder = [[0,1,2,3],[4,5,6,7]]
    fuse_layer_decoder = [[0,1,2,3],[4,5,6,7]]
    bottleneck = []
    decoder = []
    dropout=0.4
    bottleneck.append(nn.Sequential(nn.Linear(embed_dim,256), nn.Dropout(p=dropout)))
    bottleneck.append(nn.Sequential(nn.Linear(256, embed_dim*4), nn.GELU(), nn.Dropout(p=dropout), nn.Linear(embed_dim*4, embed_dim), nn.Dropout(p=dropout)))
    bottleneck = nn.ModuleList(bottleneck)
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=partial(LinearAttention2, eps=1e-8))
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)
    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers, remove_class_token=False, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder, context_aware_recenter=1)
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()
    print(f"[build_bank] model loaded: {model_path}")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    def extract_features(image_paths):
        feats = []
        # 4060 8G 单卡：分批、pin_memory False 在 Windows 更稳
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Extract"):
                batch_paths = image_paths[i:i+batch_size]
                imgs = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        imgs.append(transform(img))
                    except Exception as e:
                        print(f"[warn] failed {p}: {e}")
                if not imgs:
                    continue
                batch = torch.stack(imgs).to(device)
                enc_out, _ = model(batch)
                # 取最后一层 encoder feature map -> NCHW, 全局平均/或 patch tokens
                # dinomaly_two_stage 中使用 encoder_outputs[-1] 的 CHW feature，做 ROIAlign 前的库
                # 这里取最后一层特征的全局平均作为图像级向量，兼容两阶段检索
                f = enc_out[-1]
                if isinstance(f, torch.Tensor):
                    if f.ndim == 3:
                        # BxNxC -> 平均
                        f = f.mean(dim=1)
                    elif f.ndim == 4:
                        f = f.mean(dim=(2,3))
                    # F.normalize(p=2)
                    f = torch.nn.functional.normalize(f.float(), p=2, dim=-1)
                else:
                    f = f[0] if isinstance(f, (list,tuple)) else f
                    if isinstance(f, torch.Tensor):
                        if f.ndim == 4:
                            f = f.mean(dim=(2,3))
                        f = torch.nn.functional.normalize(f.float(), p=2, dim=-1)
                feats.append(f.detach().cpu().numpy().astype(np.float32))
        if feats:
            return np.concatenate(feats, axis=0)
        else:
            return np.zeros((0, embed_dim), dtype=np.float32)

    good_feat = extract_features(ok_images) if ok_images else np.zeros((0, embed_dim), dtype=np.float32)
    anomaly_feat = extract_features(ng_images) if ng_images else np.zeros((0, embed_dim), dtype=np.float32)
    print(f"[build_bank] good_features {good_feat.shape}, anomaly_features {anomaly_feat.shape}")

    # 保存 npz (Path 自动处理中文/空格)
    np.savez_compressed(str(save_bank), good_features=good_feat, anomaly_features=anomaly_feat,
                        ok_paths=np.array([str(p) for p in ok_images]), ng_paths=np.array([str(p) for p in ng_images]),
                        image_size=np.array(args.image_size), backbone=np.array(args.backbone))
    print(f"[build_bank] saved -> {save_bank} ({save_bank.stat().st_size/1024:.1f} KB)")

    # 可选 faiss 索引保存
    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            import faiss
            for name, feat in [("good", good_feat), ("anomaly", anomaly_feat)]:
                if feat.shape[0]==0:
                    continue
                dim = feat.shape[1]
                index = faiss.IndexFlatIP(dim)
                faiss.normalize_L2(feat)
                index.add(feat.astype(np.float32))
                faiss.write_index(index, str(save_dir / f"{name}.faiss"))
                print(f"[build_bank] faiss {name}: {feat.shape} -> {save_dir / f'{name}.faiss'}")
        except ImportError:
            print("[build_bank] faiss not installed, skip index save")
        except Exception as e:
            print(f"[build_bank] faiss save failed: {e}")

if __name__ == "__main__":
    main()
