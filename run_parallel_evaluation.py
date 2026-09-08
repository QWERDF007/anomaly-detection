import json
import multiprocessing as mp
from pathlib import Path
from evaluate_exp0903_all_models import (
    eval_patchcore_variant,
    eval_dinomaly_variant,
    get_latest_checkpoint,
    get_patchcore_params,
)

EXP_DIR = Path("/data/wt/exp0903")

def eval_dataset_worker(ds_name, gpu_id, models_list, result_queue):
    import traceback
    device = f"cuda:{gpu_id}"
    print(f"[Process {ds_name} on GPU {gpu_id}] STARTING ...")
    test_txt = EXP_DIR / "data_splits" / f"{ds_name}_test.txt"
    
    ds_results = {}
    for model_title, m_type, m_dir in models_list:
        key = f"{ds_name}_{model_title}"
        print(f"\n[{ds_name}] >> Evaluating {model_title} on {device} ...")
        
        try:
            if m_type == "patchcore":
                pkl_p = get_patchcore_params(m_dir)
                if not pkl_p:
                    print(f"[{ds_name}] ERROR: No patchcore_params.pkl found in {m_dir}")
                    continue
                res = eval_patchcore_variant(pkl_p, test_txt, device=device)
            else:
                ckpt_p = get_latest_checkpoint(m_dir)
                if not ckpt_p:
                    print(f"[{ds_name}] ERROR: No model.pth found in {m_dir}")
                    continue
                res = eval_dinomaly_variant(ckpt_p, test_txt, device=device)
                
            ds_results[key] = {
                'dataset': ds_name,
                'model': model_title,
                'type': m_type,
                'model_dir': str(m_dir),
                'metrics': res,
            }
            
            cl = res['clean']
            sn = res['seen_summary']
            un = res['unseen_summary']
            eff = res['efficiency']
            print(f"[{ds_name}] {model_title} DONE:")
            print(f"   Clean I-AUROC: {cl['i_auroc']:.4f} | I-F1: {cl['i_f1']:.4f} | P-AUROC: {cl['p_auroc'] or 0:.4f}")
            print(f"   Clean FPR@Q99: {cl['clean_fpr_prod']:.2f}% | Seen Mean FPR: {sn['mean_fpr_prod']:.2f}% | Unseen Mean FPR: {un['mean_fpr_prod']:.2f}%")
            print(f"   FPS: {eff['fps']:.1f} | VRAM: {eff['vram_gb']:.2f} GB")
        except Exception as e:
            print(f"[{ds_name}] EXCEPTION evaluating {model_title}: {e}")
            traceback.print_exc()

    result_queue.put(ds_results)
    print(f"[Process {ds_name}] ALL MODELS FINISHED!")

def main():
    vc_models = [
        ("PatchCore (Coreset 10%)", "patchcore", EXP_DIR / "VC_patchcore_n200"),
        ("Dinomaly2 Base (Baseline)", "dinomaly", EXP_DIR / "VC_dinomaly2_base"),
        ("Source-DG (Adapter)", "dinomaly", EXP_DIR / "VC_source_dg_adapter"),
        ("Source-DG (Full + UOT)", "dinomaly", EXP_DIR / "VC_source_dg_full_uot"),
    ]
    
    zs_models = [
        ("PatchCore (Coreset 10%)", "patchcore", EXP_DIR / "中石_patchcore_n56"),
        ("Dinomaly2 Base (Baseline)", "dinomaly", EXP_DIR / "中石_dinomaly2_base"),
        ("Source-DG (Adapter)", "dinomaly", EXP_DIR / "中石_source_dg_adapter"),
        ("Source-DG (Full + UOT)", "dinomaly", EXP_DIR / "中石_source_dg_full_uot"),
    ]

    result_queue = mp.Queue()

    p_vc = mp.Process(target=eval_dataset_worker, args=("VC", 1, vc_models, result_queue))
    p_zs = mp.Process(target=eval_dataset_worker, args=("中石", 2, zs_models, result_queue))

    p_vc.start()
    p_zs.start()

    p_vc.join()
    p_zs.join()

    all_results = {}
    while not result_queue.empty():
        part = result_queue.get()
        all_results.update(part)

    out_json = EXP_DIR / "benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved complete benchmark results to: {out_json}")

if __name__ == '__main__':
    main()
