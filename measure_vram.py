#!/usr/bin/env python
"""Measure peak VRAM for Dinomaly2 forward per image size / batch size on 4060 8G.
Produces F:\\tmp\\outs\\vram_measure.json
Usage: D:\\Software\\anaconda3\\envs\\py312\\python.exe measure_vram.py --image_sizes 224 448 672
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from functools import partial

ROOT = Path(__file__).resolve().parent
DINOMALY2 = ROOT / "Dinomaly2"
if str(DINOMALY2) not in sys.path:
    sys.path.insert(0, str(DINOMALY2))

def measure_one(image_size, batch_size, backbone="dinov2reg_vit_base_14"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    from models.uad import Dinomaly
    from models import vit_encoder
    from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
    encoder = vit_encoder.load(backbone)
    embed_dim, num_heads = 768,12
    target_layers=[2,3,4,5,6,7,8,9]
    fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]]
    fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]]
    bottleneck=nn.ModuleList([nn.Sequential(nn.Linear(embed_dim,256), nn.Dropout(p=0.4)), nn.Sequential(nn.Linear(256, embed_dim*4), nn.GELU(), nn.Dropout(p=0.4), nn.Linear(embed_dim*4, embed_dim), nn.Dropout(p=0.4))])
    decoder=nn.ModuleList([VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=partial(LinearAttention2, eps=1e-8)) for _ in range(8)])
    model=Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers, remove_class_token=False, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder, context_aware_recenter=1)
    model.to(device).eval()
    # optimizer to measure training VRAM
    from optimizers import StableAdamW
    optimizer = StableAdamW([{'params': bottleneck[0].parameters(), 'lr':2e-4},{'params': bottleneck[1].parameters()},{'params': decoder.parameters()}], lr=2e-3, betas=(0.9,0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)
    # dummy batch
    x = torch.randn(batch_size,3,image_size,image_size, device=device)
    # forward + backward training
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    en,de = model(x)
    from utils import global_cosine
    loss = global_cosine(en,de)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() if device.type=="cuda" else None
    t1=time.perf_counter()
    peak_train = torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0
    reserved_train = torch.cuda.max_memory_reserved()/1024**3 if device.type=="cuda" else 0
    # inference only
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with torch.no_grad():
        t2=time.perf_counter()
        en,de = model(x)
        torch.cuda.synchronize() if device.type=="cuda" else None
        t3=time.perf_counter()
    peak_infer = torch.cuda.max_memory_allocated()/1024**3 if device.type=="cuda" else 0
    return {
        "image_size": image_size,
        "batch_size": batch_size,
        "backbone": backbone,
        "peak_train_gb": round(float(peak_train),3),
        "reserved_train_gb": round(float(reserved_train),3),
        "peak_infer_gb": round(float(peak_infer),3),
        "train_forward_backward_ms": round((t1-t0)*1000,1),
        "infer_ms_batch": round((t3-t2)*1000,1),
        "infer_ms_per_image": round((t3-t2)/batch_size*1000,1)
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[224,448,672])
    parser.add_argument("--outs_dir", type=str, default=r"F:\tmp\outs")
    args=parser.parse_args()
    outs=Path(args.outs_dir).expanduser().resolve()
    outs.mkdir(parents=True, exist_ok=True)
    # 4060 batch mapping
    batch_map={224:8,448:4,672:2}
    results=[]
    for sz in args.image_sizes:
        bs=batch_map.get(sz,4)
        print(f"Measuring {sz} bs={bs} ...")
        try:
            r=measure_one(sz, bs)
            print(r)
            results.append(r)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at {sz} bs={bs}, try bs={bs//2}")
                torch.cuda.empty_cache()
                try:
                    r=measure_one(sz, max(1, bs//2))
                    r["note"]="OOM fallback"
                    results.append(r)
                    print(r)
                except Exception as e2:
                    print(f"Failed {sz}: {e2}")
            else:
                print(f"Failed {sz}: {e}")
        # also measure patchcore backbone VRAM separately
    # PatchCore measure
    import torch
    from patchcore_inspection_mock import measure_patchcore
    for sz in args.image_sizes:
        bs=batch_map.get(sz,4)
        try:
            pr=measure_patchcore(sz, bs)
            results.append(pr)
            print(pr)
        except Exception as e:
            print(f"patchcore measure failed {sz}: {e}")
    out=outs/"vram_measure.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> {out}")

if __name__=="__main__":
    main()
