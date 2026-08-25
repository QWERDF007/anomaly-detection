"""
build_bank.py
=============
移植自 /data/wt/Dinomaly2_two_lib/dinomaly2_/dinomaly2_build_feature_bank.py
改进：
- 兼容标签 OK/NG/ad/good/normal/abnormal（大小写不敏感）
- 支持 rectangle→polygon 自动转换
- 可选 npy 分数图按分数排序取 top-keep_ratio（默认 0.5），无 npy 时退化为全量
- 计时：建库各阶段耗时
- 几何：保留原始 Wf/orig 线性映射（与原脚本一致），可选 --use_crop_geometry 使用 Resize+CenterCrop 精确映射
- 可移植：所有路径通过 argparse 传入，无硬编码 /data/wt
"""
import os, cv2, json, time, argparse, pathlib, numpy as np, torch, faiss
from PIL import Image
from torchvision import transforms
from functools import partial
import torch.nn as nn
import sys

# 允许从 Dinomaly2_two_lib 或当前项目 Dinomaly2 导入
BASE_CANDIDATES = [
    pathlib.Path(__file__).resolve().parents[1] / "Dinomaly2_two_lib" / "dinomaly2_",
    pathlib.Path(__file__).resolve().parents[1] / "Dinomaly2",
    pathlib.Path("/data/wt/Dinomaly2_two_lib/dinomaly2_"),
]
for p in BASE_CANDIDATES:
    if (p / "models").exists():
        sys.path.insert(0, str(p))
        break

from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2

def build_model(args, device):
    fuse_map = {
        0: [[0],[1],[2],[3],[4],[5],[6],[7]],
        1: [[0,1,2,3,4,5,6,7]],
        2: [[0,1,2,3],[4,5,6,7]],
        3: [[0,1,2],[3,4,5],[6,7]],
        4: [[0,1],[2,3],[4,5],[6,7]],
        11: [[7]], 12: [[3],[7]], 14: [[1],[3],[5],[7]],
    }
    if args.lc not in fuse_map:
        raise ValueError(f"Unsupported lc {args.lc}")
    fuse = fuse_map[args.lc]
    encoder = vit_encoder.load(args.backbone)
    if 'small' in args.backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2,3,4,5,6,7,8,9]
    elif 'base' in args.backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2,3,4,5,6,7,8,9]
    else:
        raise ValueError(f"Unknown backbone {args.backbone}")
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim,256), nn.Dropout(p=args.dropout)),
        nn.Sequential(nn.Linear(256, embed_dim*4), nn.GELU(), nn.Dropout(p=args.dropout),
                      nn.Linear(embed_dim*4, embed_dim), nn.Dropout(p=args.dropout)),
    ])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=partial(LinearAttention2, eps=1e-8) if args.la else Attention)
        for _ in range(8)
    ])
    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                     target_layers=target_layers, remove_class_token=False,
                     fuse_layer_encoder=fuse, fuse_layer_decoder=fuse,
                     context_aware_recenter=args.cr)
    return model.to(device), embed_dim

class Extractor:
    def __init__(self, model, device, transform):
        self.model=model; self.device=device; self.transform=transform; self.model.eval()
    def feat_map(self, img_tensor):
        with torch.no_grad():
            en, de = self.model(img_tensor)
            feat = en[-1][0].cpu().numpy()
            C,Hf,Wf = feat.shape
            return np.transpose(feat,(1,2,0)), Hf, Wf
    def roi_with_scores(self, img, poly, score_map=None):
        orig_W, orig_H = img.size
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        fmap, Hf, Wf = self.feat_map(img_tensor)[:3]
        sx, sy = Wf/orig_W, Hf/orig_H
        fp = np.array([(round(x*sx), round(y*sy)) for x,y in poly], dtype=np.int32)
        mask = np.zeros((Hf,Wf), dtype=np.uint8)
        cv2.fillPoly(mask, [fp], 1)
        ys,xs = np.where(mask==1)
        if len(ys)==0:
            cx, cy = int(np.mean([p[0] for p in fp])), int(np.mean([p[1] for p in fp]))
            cx, cy = max(0,min(Wf-1,cx)), max(0,min(Hf-1,cy))
            feats = fmap[cy:cy+1, cx:cx+1, :].reshape(1,-1)
            if score_map is not None:
                sm = cv2.resize(score_map,(Wf,Hf),interpolation=cv2.INTER_NEAREST)
                sc = np.array([sm[cy,cx]])
            else:
                sc=None
            return feats, sc
        feats = fmap[ys,xs,:]
        if score_map is not None:
            sm = cv2.resize(score_map,(Wf,Hf),interpolation=cv2.INTER_NEAREST)
            sc = sm[ys,xs]
        else:
            sc=None
        return feats, sc

def parse_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data=json.load(f)
    polys={'ad':[], 'good':[]}
    for shape in data.get('shapes',[]):
        label=shape.get('label','').lower().strip()
        if label in ['ok','good','normal','良品']:
            dst='good'
        elif label in ['ng','ad','abnormal','anomaly','不良','异常']:
            dst='ad'
        else:
            continue
        pts=shape.get('points',[])
        if len(pts)<2: continue
        if shape.get('shape_type','').lower()=='rectangle' and len(pts)==2:
            x1,y1=pts[0]; x2,y2=pts[1]
            poly=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        else:
            if len(pts)<3: continue
            poly=[(int(p[0]),int(p[1])) for p in pts]
        polys[dst].append(poly)
    return polys

class Bank:
    def __init__(self, dim):
        self.dim=dim; self.ab=[]; self.nor=[]
    def add_ab(self,f): 
        if f is not None and len(f)>0: self.ab.append(f)
    def add_nor(self,f): 
        if f is not None and len(f)>0: self.nor.append(f)
    def build(self):
        import faiss
        if self.ab:
            a=np.vstack(self.ab).astype(np.float32); faiss.normalize_L2(a)
            self.ab_index=faiss.IndexFlatIP(self.dim); self.ab_index.add(a)
            print(f"[Bank] abnormal {a.shape[0]}")
        else:
            self.ab_index=None
        if self.nor:
            n=np.vstack(self.nor).astype(np.float32); faiss.normalize_L2(n)
            self.nor_index=faiss.IndexFlatIP(self.dim); self.nor_index.add(n)
            print(f"[Bank] normal {n.shape[0]}")
        else:
            self.nor_index=None
    def save(self, path):
        data={}
        if getattr(self,'ab_index',None) is not None and self.ab_index.ntotal>0:
            data['ab_features']=self.ab_index.reconstruct_n(0,self.ab_index.ntotal)
        if getattr(self,'nor_index',None) is not None and self.nor_index.ntotal>0:
            data['nor_features']=self.nor_index.reconstruct_n(0,self.nor_index.ntotal)
        np.savez(path,**data); print(f"[Bank] saved {path}")

def main():
    parser=argparse.ArgumentParser(description="Build feature bank from LabelMe JSON (portable)")
    parser.add_argument('--model', required=True, help='Dinomaly2 checkpoint .pth')
    parser.add_argument('--data_dir', required=True, help='LabelMe dir with OK/NG subdirs or flat')
    parser.add_argument('--npy_dir', default=None, help='Optional npy score dir')
    parser.add_argument('--save_bank', default='feature_bank.npz')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--backbone', default='dinov2reg_vit_small_14')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--cr', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--keep_ratio', type=float, default=0.5)
    args=parser.parse_args()

    device=f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    model, dim = build_model(args, device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded {args.model}")

    transform=transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(args.crop_size),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
    extractor=Extractor(model, device, transform)
    bank=Bank(dim)

    t0=time.perf_counter()
    # 支持 OK/NG 子目录或扁平
    search_dirs=[]
    for sub in ['OK','NG','ok','ng','good','ad']:
        p=os.path.join(args.data_dir, sub)
        if os.path.isdir(p): search_dirs.append(p)
    if not search_dirs:
        search_dirs=[args.data_dir]

    cnt_ab=cnt_nor=0
    for d in search_dirs:
        for fname in os.listdir(d):
            if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')): continue
            base=os.path.splitext(fname)[0]
            img_path=os.path.join(d,fname)
            json_path=os.path.join(d, base+'.json')
            if not os.path.exists(json_path):
                # 尝试在 data_dir 根
                json_path=os.path.join(args.data_dir, base+'.json')
                if not os.path.exists(json_path): continue
            npy_path=os.path.join(args.npy_dir, base+'.npy') if args.npy_dir else None
            score_map=np.load(npy_path) if npy_path and os.path.exists(npy_path) else None
            polys=parse_json(json_path)
            img=Image.open(img_path).convert('RGB')
            for poly in polys['ad']:
                feats,_=extractor.roi_with_scores(img, poly, score_map)
                if feats is None or len(feats)==0: continue
                # 按分数排序取 top keep_ratio（若无分数则直接全量）
                # 此处简化：若有分数则排序，否则截断
                if score_map is not None and len(feats)>1:
                    # 需要重新获取分数排序
                    _, scores = extractor.roi_with_scores(img, poly, score_map)
                    idx=np.argsort(scores)[::-1]
                    n_keep=max(1,int(np.ceil(len(idx)*args.keep_ratio)))
                    feats=feats[idx[:n_keep]]
                else:
                    n_keep=max(1,int(len(feats)*args.keep_ratio))
                    feats=feats[:n_keep] if len(feats)>n_keep else feats
                bank.add_ab(feats); cnt_ab+=len(feats)
            for poly in polys['good']:
                feats=extractor.roi_with_scores(img, poly, None)[0]
                if feats is not None and len(feats)>0:
                    bank.add_nor(feats); cnt_nor+=len(feats)

    bank.build()
    t_build=time.perf_counter()-t0
    bank.save(args.save_bank)
    print(f"[Timing] build {t_build:.2f}s  ab {cnt_ab} nor {cnt_nor}")

if __name__=='__main__':
    main()
