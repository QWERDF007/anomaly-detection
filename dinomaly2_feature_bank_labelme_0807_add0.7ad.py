import os
import cv2
import json
import torch
import numpy as np
import faiss
import argparse
from PIL import Image
from torchvision import transforms
from functools import partial
from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from utils import cal_anomaly_maps, get_gaussian_kernel
import torch.nn as nn
from tqdm import tqdm
import math


# ===================== 1. 模型构建 =====================
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
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

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


# ===================== 2. 特征提取器（ROI 特征提取） =====================
class DiNoMaly2FeatureExtractor:
    def __init__(self, model, device, transform):
        self.model = model
        self.device = device
        self.transform = transform  # 保存 transform
        self.model.eval()

    def extract_feature_map(self, img_tensor):
        """返回 (Hf, Wf, C) 的特征图"""
        with torch.no_grad():
            en, de = self.model(img_tensor)
            feat = en[-1]  # (B, C, Hf, Wf)
            feat = feat[0].cpu().numpy()  # (C, Hf, Wf)
            C, Hf, Wf = feat.shape
            return np.transpose(feat, (1, 2, 0)), Hf, Wf

    # ========== 核心修改：带分数的特征提取 ==========
    def extract_roi_features_with_scores(self, img, polygon_pts, score_map=None):
        """
        提取多边形内的特征，并返回特征和对应的分数

        Args:
            img: PIL Image（原始尺寸，未 Resize）
            polygon_pts: 原图坐标的多边形顶点
            score_map: 异常分数图 (H, W) numpy array，与原始图像同尺寸

        Returns:
            features: (N, C) 特征数组
            scores: (N,) 对应的分数数组
        """
        orig_W, orig_H = img.size
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        feat_map, Hf, Wf = self.extract_feature_map(img_tensor)

        scale_x = Wf / orig_W
        scale_y = Hf / orig_H

        feat_poly = np.array([(round(x * scale_x), round(y * scale_y)) for (x, y) in polygon_pts], dtype=np.int32)

        mask = np.zeros((Hf, Wf), dtype=np.uint8)
        for y in range(Hf):
            for x in range(Wf):
                if cv2.pointPolygonTest(feat_poly, (x, y), False) >= 0:
                    mask[y, x] = 1

        ys, xs = np.where(mask == 1)

        # 如果面积太小，取中心点
        if len(ys) == 0:
            cx = int(np.mean([p[0] for p in feat_poly]))
            cy = int(np.mean([p[1] for p in feat_poly]))
            cx = max(0, min(Wf - 1, cx))
            cy = max(0, min(Hf - 1, cy))
            features = feat_map[cy:cy + 1, cx:cx + 1, :].reshape(1, -1)
            if score_map is not None:
                scores = np.array([score_map[cy, cx]])
            else:
                scores = None
            return features, scores

        features = feat_map[ys, xs, :]  # (N, C)

        # 提取分数
        if score_map is not None:
            scores = []
            for y, x in zip(ys, xs):
                # 计算该特征点对应的原图区域范围
                # 像素坐标 左上角？中心？映射是不是错乱  ，resize  特征图尺寸是不是好？
                x1 = int(x / scale_x)
                x2 = int((x + 1) / scale_x)
                y1 = int(y / scale_y)
                y2 = int((y + 1) / scale_y)
                if y2 > y1 and x2 > x1:
                    region_score = np.max(score_map[y1:y2, x1:x2])
                else:
                    region_score = score_map[y, x]
                scores.append(region_score)
            scores = np.array(scores)
        else:
            scores = None

        return features, scores

    def extract_roi_features(self, img, polygon_pts):
        """兼容原接口"""
        features, _ = self.extract_roi_features_with_scores(img, polygon_pts, score_map=None)
        return features


# ===================== 3. 特征库，余弦相似度方式 =====================
class FeatureBank:
    def __init__(self, dim):
        self.dim = dim
        self.abnormal_features = []
        self.normal_features = []
        self.ab_index = None
        self.nor_index = None

    def _normalize(self, features):
        """L2 归一化，使内积 = 余弦相似度"""
        features = features.astype(np.float32)
        faiss.normalize_L2(features)
        return features

    def add_abnormal(self, features):
        if features is not None and len(features) > 0:
            self.abnormal_features.append(features)

    def add_normal(self, features):
        if features is not None and len(features) > 0:
            self.normal_features.append(features)

    def build(self):
        if self.abnormal_features:
            all_ab = np.vstack(self.abnormal_features).astype(np.float32)
            all_ab = self._normalize(all_ab)
            self.ab_index = faiss.IndexFlatIP(self.dim)
            self.ab_index.add(all_ab)
            print(f"[特征库] 异常特征数: {all_ab.shape[0]}")

        if self.normal_features:
            all_nor = np.vstack(self.normal_features).astype(np.float32)
            all_nor = self._normalize(all_nor)
            self.nor_index = faiss.IndexFlatIP(self.dim)
            self.nor_index.add(all_nor)
            print(f"[特征库] 正常特征数: {all_nor.shape[0]}")

    def query(self, query_feat, k=1):
        q = query_feat.astype(np.float32)
        q = self._normalize(q)

        ab_cos_dist = np.full(len(q), 1.0)
        nor_cos_dist = np.full(len(q), 1.0)

        if self.ab_index is not None and self.ab_index.ntotal > 0:
            ab_ip, _ = self.ab_index.search(q, k)
            ab_cos_dist = 1 - ab_ip[:, 0]

        if self.nor_index is not None and self.nor_index.ntotal > 0:
            nor_ip, _ = self.nor_index.search(q, k)
            nor_cos_dist = 1 - nor_ip[:, 0]

        return ab_cos_dist, nor_cos_dist

    def save(self, path):
        data = {}
        if self.ab_index is not None and self.ab_index.ntotal > 0:
            data['ab_features'] = self.ab_index.reconstruct_n(0, self.ab_index.ntotal)
        if self.nor_index is not None and self.nor_index.ntotal > 0:
            data['nor_features'] = self.nor_index.reconstruct_n(0, self.nor_index.ntotal)
        np.savez(path, **data)
        print(f"[特征库] 保存到: {path}")

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        if 'ab_features' in data:
            ab_feat = data['ab_features'].astype(np.float32)
            ab_feat = self._normalize(ab_feat)
            self.ab_index = faiss.IndexFlatIP(self.dim)
            self.ab_index.add(ab_feat)
            print(f"[特征库] 加载异常特征: {ab_feat.shape[0]}")
        if 'nor_features' in data:
            nor_feat = data['nor_features'].astype(np.float32)
            nor_feat = self._normalize(nor_feat)
            self.nor_index = faiss.IndexFlatIP(self.dim)
            self.nor_index.add(nor_feat)
            print(f"[特征库] 加载正常特征: {nor_feat.shape[0]}")


# ===================== 4. LabelMe JSON 解析 =====================
def parse_labelme_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    polygons = {'ad': [], 'good': []}

    for shape in data.get('shapes', []):
        label = shape.get('label', '').lower()
        if label not in ['ad', 'good']:
            continue

        points = shape.get('points', [])
        if len(points) < 3:
            continue

        polygon = [(int(p[0]), int(p[1])) for p in points]
        polygons[label].append(polygon)

    return polygons


# ========== 新增：带分数排序的数据加载函数 ==========
def load_labelme_dataset_with_scores(data_dir, npy_dir, extractor, keep_ratio=0.5):
    """
    遍历文件夹，加载图像、JSON 和 npy 分数，提取 ROI 特征
    异常多边形：按分数排序，取前 keep_ratio 的特征入库

    Args:
        data_dir: 包含图像和 JSON 的目录
        npy_dir: 包含 npy 分数文件的目录
        extractor: DiNoMaly2FeatureExtractor 实例
        keep_ratio: 保留比例，默认 0.5 (50%)
    """
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = [f for f in os.listdir(data_dir)
                 if os.path.splitext(f)[1].lower() in img_exts]

    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        json_path = os.path.join(data_dir, base_name + '.json')
        img_path = os.path.join(data_dir, img_file)
        npy_path = os.path.join(npy_dir, base_name + '.npy')

        if not os.path.exists(json_path):
            print(f"跳过 {img_file}: 找不到对应的 JSON 文件")
            continue

        # 加载分数图
        score_map = None
        if os.path.exists(npy_path):
            score_map = np.load(npy_path)
        else:
            print(f"警告: 找不到分数文件 {npy_path}，异常多边形将使用全部特征")

        try:
            polygons = parse_labelme_json(json_path)
            img = Image.open(img_path).convert('RGB')

            # ========== 处理异常多边形（ad）：按分数排序，取前 50% ==========
            for poly in polygons['ad']:
                features, scores = extractor.extract_roi_features_with_scores(img, poly, score_map)

                if features is None or len(features) == 0:
                    continue

                # 如果有分数，按分数从高到低排序，取前 keep_ratio
                if scores is not None and len(scores) > 1:
                    # 按分数降序排序
                    sorted_indices = np.argsort(scores)[::-1]  # 从高到低
                    n_total = len(sorted_indices)
                    n_keep = max(1, int(np.ceil(n_total * keep_ratio)))  # 向上取整，至少保留1个
                    keep_indices = sorted_indices[:n_keep]
                    features_filtered = features[keep_indices]
                    # print(f"  异常多边形: 总数 {n_total}, 保留 {len(features_filtered)} ({keep_ratio*100:.0f}%)")
                else:
                    features_filtered = features

                if len(features_filtered) > 0:
                    yield 'abnormal', features_filtered

            # ========== 处理正常多边形（good）：全部入库 ==========
            for poly in polygons['good']:
                features = extractor.extract_roi_features(img, poly)
                if features is not None and len(features) > 0:
                    yield 'normal', features

        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")


# ===================== 5. 推理（带特征库修正） =====================
def predict_with_feature_bank(model, feature_bank, img_path, transform, device,
                              low_thresh=0.25, high_thresh=0.8):
    img = Image.open(img_path).convert('RGB')
    orig = np.array(img)
    H, W = orig.shape[:2]
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        en, de = model(img_tensor)
        anomaly_map, _ = cal_anomaly_maps(en, de, (H, W))
        anomaly_map = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)(anomaly_map)
    amap = anomaly_map[0, 0].cpu().numpy()

    extractor = DiNoMaly2FeatureExtractor(model, device, transform)
    feat_map, Hf, Wf = extractor.extract_feature_map(img_tensor)

    amap_resized = cv2.resize(amap, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
    uncertain_mask = (amap_resized > low_thresh) & (amap_resized < high_thresh)

    if np.any(uncertain_mask) and feature_bank.ab_index is not None:
        uncertain_idx = np.where(uncertain_mask)
        uncertain_feats = feat_map[uncertain_idx]

        ab_dist, nor_dist = feature_bank.query(uncertain_feats)

        for i, (ab_d, nor_d) in enumerate(zip(ab_dist, nor_dist)):
            y, x = uncertain_idx[0][i], uncertain_idx[1][i]
            if ab_d < nor_d:
                amap_resized[y, x] = 1.5 * high_thresh
            else:
                amap_resized[y, x] = low_thresh * 0.5

    final_amap = cv2.resize(amap_resized, (W, H), interpolation=cv2.INTER_LINEAR)
    return final_amap.astype(np.float32)


# ===================== 6. 批量推理 =====================
def batch_inference(model, feature_bank, input_dir, output_dir, transform, device,
                    low_thresh=0.25, high_thresh=0.8):
    os.makedirs(output_dir, exist_ok=True)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_files = [f for f in os.listdir(input_dir)
                 if os.path.splitext(f)[1].lower() in img_exts]

    print(f"\n找到 {len(img_files)} 张图像，开始批量推理...")

    for img_file in tqdm(img_files, desc="推理进度"):
        img_path = os.path.join(input_dir, img_file)
        base_name = os.path.splitext(img_file)[0]

        try:
            final_amap = predict_with_feature_bank(
                model, feature_bank, img_path, transform, device,
                low_thresh=low_thresh, high_thresh=high_thresh
            )

            npy_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(npy_path, final_amap)

        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")


# ===================== 7. 主程序 =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='训练好的 DiNoMaly2 模型')
    parser.add_argument('--data_dir', type=str, default=None, help='LabelMe 标注数据目录（图像+JSON）')
    parser.add_argument('--npy_dir', type=str, default=None, help='npy 分数文件目录')
    parser.add_argument('--input_dir', type=str, required=True, help='待推理的图像文件夹')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='结果输出目录')
    parser.add_argument('--image_size', type=int, default=672)
    parser.add_argument('--crop_size', type=int, default=672)
    parser.add_argument('--backbone', type=str, default='dinov2reg_vit_small_14')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--cr', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--low_thresh', type=float, default=0.25)
    parser.add_argument('--high_thresh', type=float, default=0.8)
    parser.add_argument('--save_bank', type=str, default='feature_bank.npz', help='保存特征库路径')
    parser.add_argument('--load_bank', type=str, default=None, help='加载特征库路径')
    parser.add_argument('--keep_ratio', type=float, default=0.5, help='异常多边形保留比例，默认 0.5')
    args = parser.parse_args()

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    # 构建模型
    model = build_model(args, device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"模型加载完成: {args.model}")

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(args.crop_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 特征提取器（传入 transform）
    extractor = DiNoMaly2FeatureExtractor(model, device, transform)

    # 特征库
    embed_dim = 384 if 'small' in args.backbone else 768
    feature_bank = FeatureBank(dim=embed_dim)

    # 加载或构建特征库
    if args.load_bank and os.path.exists(args.load_bank):
        feature_bank.load(args.load_bank)
    elif args.data_dir and os.path.exists(args.data_dir):
        print(f"\n开始从 {args.data_dir} 构建特征库...")

        for label, feats in load_labelme_dataset_with_scores(
                args.data_dir, args.npy_dir, extractor, args.keep_ratio
        ):
            if label == 'abnormal':
                feature_bank.add_abnormal(feats)
            else:
                feature_bank.add_normal(feats)

        feature_bank.build()
        feature_bank.save(args.save_bank)
    else:
        print("警告: 未提供特征库，推理时模糊区域将无法修正")

    # 批量推理
    batch_inference(
        model=model,
        feature_bank=feature_bank,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        transform=transform,
        device=device,
        low_thresh=args.low_thresh,
        high_thresh=args.high_thresh
    )
    print(f"\n推理完成! 结果保存在: {args.output_dir}")

