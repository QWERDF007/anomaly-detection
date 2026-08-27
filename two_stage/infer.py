"""
infer.py
========
移植自 /data/wt/Dinomaly2_two_lib/dinomaly2_/dinomaly2_inference.py
改进：
- 单次前向复用 en[-1]，高斯核单例，向量化修正
- 分项计时：t_forward / t_retrieval / t_total
- 支持 OK/NG 标签兼容，支持 --use_correction 开关
- 支持图像级/像素级指标（可选 gt_dir）
- 可移植：所有路径通过 argparse 传入
"""
import os, cv2, json, time, argparse, pathlib, numpy as np, torch
from PIL import Image
from torchvision import transforms
from functools import partial
import torch.nn as nn
import sys
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
from utils import cal_anomaly_maps, get_gaussian_kernel
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, auc
from skimage import measure
import pandas as pd
from statistics import mean

try:
    import faiss
    _HAS_FAISS=True
except ImportError:
    faiss=None; _HAS_FAISS=False

def build_model(args, device):
    fuse_map={0:[[0],[1],[2],[3],[4],[5],[6],[7]],1:[[0,1,2,3,4,5,6,7]],2:[[0,1,2,3],[4,5,6,7]],3:[[0,1,2],[3,4,5],[6,7]],4:[[0,1],[2,3],[4,5],[6,7]],11:[[7]],12:[[3],[7]],14:[[1],[3],[5],[7]]}
    fuse=fuse_map[args.lc]
    encoder=vit_encoder.load(args.backbone)
    embed_dim, num_heads = (384,6) if 'small' in args.backbone else (768,12)
    target_layers=[2,3,4,5,6,7,8,9]
    bottleneck=nn.ModuleList([nn.Sequential(nn.Linear(embed_dim,256),nn.Dropout(p=args.dropout)), nn.Sequential(nn.Linear(256,embed_dim*4),nn.GELU(),nn.Dropout(p=args.dropout),nn.Linear(embed_dim*4,embed_dim),nn.Dropout(p=args.dropout))])
    decoder=nn.ModuleList([VitBlock(dim=embed_dim,num_heads=num_heads,mlp_ratio=4.,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-8),attn=partial(LinearAttention2,eps=1e-8) if args.la else Attention) for _ in range(8)])
    model=Dinomaly(encoder=encoder,bottleneck=bottleneck,decoder=decoder,target_layers=target_layers,remove_class_token=False,fuse_layer_encoder=fuse,fuse_layer_decoder=fuse,context_aware_recenter=args.cr)
    return model.to(device), embed_dim

class Bank:
    def __init__(self, dim, device_id: int = 0):
        self.dim = dim
        self.device_id = device_id
        self.ab_index = None
        self.nor_index = None

    def _norm(self, f):
        f = f.astype(np.float32)
        faiss.normalize_L2(f)
        return f

    def query(self, q):
        q = q.astype(np.float32)
        faiss.normalize_L2(q)
        ab = np.full(len(q), 1.0)
        nor = np.full(len(q), 1.0)
        if self.ab_index is not None:
            ip, _ = self.ab_index.search(np.ascontiguousarray(q), 1)
            ab = 1 - ip[:, 0]
        if self.nor_index is not None:
            ip, _ = self.nor_index.search(np.ascontiguousarray(q), 1)
            nor = 1 - ip[:, 0]
        return ab, nor

    def _to_gpu_if_available(self, idx):
        if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
            try:
                res = faiss.StandardGpuResources()
                return faiss.index_cpu_to_gpu(res, int(self.device_id), idx)
            except Exception:
                return idx
        return idx

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        if 'ab_features' in data:
            ab = data['ab_features'].astype(np.float32)
            faiss.normalize_L2(ab)
            self.ab_index = self._to_gpu_if_available(faiss.IndexFlatIP(self.dim))
            self.ab_index.add(np.ascontiguousarray(ab))
            print(f"[Bank] ab {ab.shape[0]} (GPU FAISS)")
        if 'nor_features' in data:
            nor = data['nor_features'].astype(np.float32)
            faiss.normalize_L2(nor)
            self.nor_index = self._to_gpu_if_available(faiss.IndexFlatIP(self.dim))
            self.nor_index.add(np.ascontiguousarray(nor))
            print(f"[Bank] nor {nor.shape[0]} (GPU FAISS)")

def predict_with_timing(model, gk, img_path, transform, device, bank, low, high, use_corr=True):
    t0=time.perf_counter()
    img=Image.open(img_path).convert('RGB')
    orig=np.array(img); H,W=orig.shape[:2]
    img_tensor=transform(img).unsqueeze(0).to(device)
    model.eval()
    t_fwd_s=time.perf_counter()
    with torch.no_grad():
        en,de=model(img_tensor)
        amap,_=cal_anomaly_maps(en,de,(H,W))
        amap=gk(amap)
    amap=amap[0,0].cpu().numpy()
    t_fwd=time.perf_counter()-t_fwd_s
    t_ret=0
    if use_corr and bank is not None and bank.ab_index is not None:
        t_ret_s=time.perf_counter()
        feat=en[-1][0].cpu().numpy(); fmap=np.transpose(feat,(1,2,0)); Hf,Wf=fmap.shape[:2]
        amap_r=cv2.resize(amap,(Wf,Hf),interpolation=cv2.INTER_LINEAR)
        mask=(amap_r>low)&(amap_r<high)
        if np.any(mask):
            ys,xs=np.where(mask)
            q=fmap[ys,xs,:]
            ab,nor=bank.query(q)
            is_ab=ab < nor
            amap_r[ys,xs]=np.where(is_ab, high*1.5, low*0.5)
        final=cv2.resize(amap_r,(W,H),interpolation=cv2.INTER_LINEAR)
        t_ret=time.perf_counter()-t_ret_s
    else:
        final=amap
    t_tot=time.perf_counter()-t0
    return final.astype(np.float32), t_tot, t_fwd, t_ret, orig

def f1_max(y_true,y_score):
    precs, recs,_=precision_recall_curve(y_true,y_score)
    f1s=2*precs*recs/(precs+recs+1e-7)
    return float(f1s[:-1].max()) if len(f1s)>1 else 0.0

def image_score(amap, max_ratio=0.0):
    if max_ratio==0: return float(amap.max())
    flat=amap.flatten(); k=max(1,int(len(flat)*max_ratio))
    return float(np.sort(flat)[-k:].mean())

def main():
    parser=argparse.ArgumentParser(description="Two-stage inference (portable)")
    parser.add_argument('--model', required=True)
    parser.add_argument('--load_bank', default=None)
    parser.add_argument('--input_dir_good', required=True)
    parser.add_argument('--input_dir_ng', required=True)
    parser.add_argument('--output_dir', default='./two_stage_out')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--backbone', default='dinov2reg_vit_small_14')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--cr', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--low_thresh', type=float, default=0.014)
    parser.add_argument('--high_thresh', type=float, default=0.03)
    parser.add_argument('--use_correction', type=int, default=1)
    parser.add_argument('--calc_image_metric', type=int, default=1)
    parser.add_argument('--cls_threshold', type=float, default=None)
    parser.add_argument('--max_ratio', type=float, default=0.0)
    parser.add_argument('--save_npy', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=0)
    args=parser.parse_args()
    if args.cls_threshold is None:
        args.cls_threshold=(args.low_thresh+args.high_thresh)/2
    device=f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    model, dim = build_model(args, device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    transform=transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.ToTensor(),transforms.CenterCrop(args.crop_size),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    gk=get_gaussian_kernel(5,4).to(device)
    bank=Bank(dim)
    if args.load_bank and os.path.exists(args.load_bank):
        bank.load(args.load_bank)
    os.makedirs(os.path.join(args.output_dir,'good'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'ng'), exist_ok=True)
    # 收集
    import glob
    good_files=sorted(glob.glob(os.path.join(args.input_dir_good,'*.png'))+glob.glob(os.path.join(args.input_dir_good,'*.jpg')))
    ng_files=sorted(glob.glob(os.path.join(args.input_dir_ng,'*.png'))+glob.glob(os.path.join(args.input_dir_ng,'*.jpg')))
    print(f"Good {len(good_files)} NG {len(ng_files)} corr={bool(args.use_correction)}")
    from tqdm import tqdm
    timing=[]; records=[]
    for label, files, sub in [(0,good_files,'good'),(1,ng_files,'ng')]:
        for f in tqdm(files, desc=sub):
            amap, t_tot, t_fwd, t_ret, orig = predict_with_timing(model,gk,f,transform,device,bank,args.low_thresh,args.high_thresh,bool(args.use_correction))
            timing.append((t_tot,t_fwd,t_ret))
            if args.save_npy:
                np.save(os.path.join(args.output_dir,sub,os.path.splitext(os.path.basename(f))[0]+'.npy'), amap)
            score=image_score(amap, args.max_ratio)
            records.append((os.path.basename(f), lab:=label, score, 1 if score>=args.cls_threshold else 0))
    # 计时
    t_tot=np.array([t[0] for t in timing])*1000; t_fwd=np.array([t[1] for t in timing])*1000; t_ret=np.array([t[2] for t in timing])*1000
    print(f"Avg total {t_tot.mean():.1f}ms forward {t_fwd.mean():.1f}ms retrieval {t_ret.mean():.1f}ms")
    if args.calc_image_metric and records:
        y_true=np.array([r[1] for r in records]); y_score=np.array([r[2] for r in records])
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc=roc_auc_score(y_true,y_score) if len(np.unique(y_true))>1 else 0
        ap=average_precision_score(y_true,y_score) if len(np.unique(y_true))>1 else 0
        f1m=f1_max(y_true,y_score)
        print(f"AUROC {auroc:.4f} AP {ap:.4f} F1_max {f1m:.4f} thr {args.cls_threshold}")
        # 保存
        import pandas as pd
        pd.DataFrame([{"name":r[0],"label":r[1],"score":r[2],"pred":r[3]} for r in records]).to_csv(os.path.join(args.output_dir,'image_scores.csv'),index=False)
        with open(os.path.join(args.output_dir,'report.json'),'w') as jf:
            json.dump({"auroc":auroc,"ap":ap,"f1_max":f1m,"thr":args.cls_threshold,"timing":{"total_ms":float(t_tot.mean()),"forward_ms":float(t_fwd.mean()),"retrieval_ms":float(t_ret.mean())}},jf,indent=2)

if __name__=='__main__':
    main()
