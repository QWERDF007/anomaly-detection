import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path

ROOT = Path("/data/wt/anomaly-detection").resolve()
PYTHON = "/home/dell/miniconda3/envs/anomaly/bin/python"
EXP_DIR = Path("/data/wt/exp_shimo")

def run_worker(gpu_id, task_queue, result_queue):
    while True:
        try:
            task = task_queue.get_nowait()
        except:
            break
        name, cmd = task
        print(f"[GPU {gpu_id}] >>> STARTING: {name}", flush=True)
        log_file = EXP_DIR / f"{name}.log"
        t0 = time.perf_counter()
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd_adjusted = []
        for arg in cmd:
            cmd_adjusted.append(arg)
            
        with open(log_file, "w", encoding="utf-8") as f_out:
            proc = subprocess.Popen(
                cmd_adjusted,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(ROOT),
            )
            ret = proc.wait()
        elapsed = time.perf_counter() - t0
        status = "SUCCESS" if ret == 0 else f"FAILED(code={ret})"
        print(f"[GPU {gpu_id}] <<< FINISHED [{status}] in {elapsed:.1f}s: {name}", flush=True)
        result_queue.put((name, ret, elapsed, str(log_file)))

def main():
    gpu_pool = [1, 2, 3, 4, 5, 6]
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = [
        # ori-10 tasks
        ("ori-10_patchcore_n200", [
            PYTHON, str(ROOT / "patchcore-inspection" / "train.py"),
            "--data_path", str(EXP_DIR / "data_splits" / "ori-10_train_n200.txt"),
            "--dataset", "custom",
            "--backbone", "wideresnet50",
            "--layers", "layer2",
            "--layers", "layer3",
            "--sampler", "approx_greedy_coreset",
            "--sampling_percentage", "0.1",
            "--resize", "448",
            "--imagesize", "448",
            "--save_dir", str(EXP_DIR / "ori-10_patchcore_n200"),
            "--skip_eval",
            "--gpu", "0",
        ]),
        ("ori-10_dinomaly2_base", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "ori-10_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "ori-10_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "ori-10_dinomaly2_base"),
            "--eval_interval", "-1",
            "--cuda", "0",
        ]),
        ("ori-10_source_dg_adapter", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "ori-10_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "ori-10_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "ori-10_source_dg_adapter"),
            "--eval_interval", "-1",
            "--source-dg",
            "--cuda", "0",
        ]),
        ("ori-10_source_dg_full_uot", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "ori-10_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "ori-10_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "ori-10_source_dg_full_uot"),
            "--eval_interval", "-1",
            "--source-dg",
            "--pseudo-anomaly",
            "--cross-view-uot",
            "--cuda", "0",
        ]),
        # ori-21 tasks
        ("ori-21_patchcore_n200", [
            PYTHON, str(ROOT / "patchcore-inspection" / "train.py"),
            "--data_path", str(EXP_DIR / "data_splits" / "ori-21_train_n200.txt"),
            "--dataset", "custom",
            "--backbone", "wideresnet50",
            "--layers", "layer2",
            "--layers", "layer3",
            "--sampler", "approx_greedy_coreset",
            "--sampling_percentage", "0.1",
            "--resize", "448",
            "--imagesize", "448",
            "--save_dir", str(EXP_DIR / "ori-21_patchcore_n200"),
            "--skip_eval",
            "--gpu", "0",
        ]),
        ("ori-21_dinomaly2_base", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "ori-21_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "ori-21_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "ori-21_dinomaly2_base"),
            "--eval_interval", "-1",
            "--cuda", "0",
        ]),
        ("ori-21_source_dg_adapter", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "ori-21_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "ori-21_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "ori-21_source_dg_adapter"),
            "--eval_interval", "-1",
            "--source-dg",
            "--cuda", "0",
        ]),
        ("ori-21_source_dg_full_uot", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "ori-21_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "ori-21_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "ori-21_source_dg_full_uot"),
            "--eval_interval", "-1",
            "--source-dg",
            "--pseudo-anomaly",
            "--cross-view-uot",
            "--cuda", "0",
        ]),
    ]
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for t in tasks:
        task_queue.put(t)
        
    workers = []
    for gid in gpu_pool:
        p = mp.Process(target=run_worker, args=(gid, task_queue, result_queue))
        p.start()
        workers.append(p)
        time.sleep(0.5)
        
    for p in workers:
        p.join()
        
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
        
    print("\n=======================================================")
    print("ALL 8 SHIMO TRAINING TASKS COMPLETED! SUMMARY:")
    print("=======================================================")
    for name, code, elapsed, log_p in sorted(results, key=lambda x: x[0]):
        status = "SUCCESS" if code == 0 else f"FAILED(code={code})"
        print(f"  {name:<26}: {status:<15} (took {elapsed:.1f}s) -> Log: {log_p}")

if __name__ == "__main__":
    main()
