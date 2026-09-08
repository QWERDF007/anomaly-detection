import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path

PYTHON = "/home/dell/miniconda3/envs/anomaly/bin/python"
ROOT = Path("/data/wt/anomaly-detection").resolve()
EXP_DIR = Path("/data/wt/exp0903").resolve()

def worker(gpu_id, task_queue, result_queue):
    while True:
        try:
            task = task_queue.get_nowait()
        except:
            break
        name, cmd = task
        print(f"[GPU {gpu_id}] >>> STARTING: {name}", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Note: within the process, CUDA device index will be 0 because of CUDA_VISIBLE_DEVICES
        cmd_adjusted = []
        for i, c in enumerate(cmd):
            if c in ["--cuda", "--gpu"]:
                cmd_adjusted.append(c)
                cmd_adjusted.append("0")
            elif i > 0 and cmd[i-1] in ["--cuda", "--gpu"]:
                continue
            else:
                cmd_adjusted.append(c)
                
        t0 = time.perf_counter()
        log_file = EXP_DIR / f"{name}.log"
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
    
    tasks = [
        # VC tasks
        ("VC_patchcore_n200", [
            PYTHON, str(ROOT / "patchcore-inspection" / "train.py"),
            "--data_path", str(EXP_DIR / "data_splits" / "VC_train_n200.txt"),
            "--dataset", "custom",
            "--backbone", "wideresnet50",
            "--layers", "layer2",
            "--layers", "layer3",
            "--sampler", "approx_greedy_coreset",
            "--sampling_percentage", "0.1",
            "--resize", "448",
            "--imagesize", "448",
            "--save_dir", str(EXP_DIR / "VC_patchcore_n200"),
            "--skip_eval",
            "--gpu", "0",
        ]),
        ("VC_dinomaly2_base", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "VC_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "VC_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "VC_dinomaly2_base"),
            "--eval_interval", "-1",
            "--cuda", "0",
        ]),
        ("VC_source_dg_adapter", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "VC_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "VC_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "VC_source_dg_adapter"),
            "--eval_interval", "-1",
            "--source-dg",
            "--cuda", "0",
        ]),
        ("VC_source_dg_full_uot", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "VC_train_n200.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "VC_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "VC_source_dg_full_uot"),
            "--eval_interval", "-1",
            "--source-dg",
            "--pseudo-anomaly",
            "--cross-view-uot",
            "--cuda", "0",
        ]),
        # 中石 tasks
        ("中石_patchcore_n56", [
            PYTHON, str(ROOT / "patchcore-inspection" / "train.py"),
            "--data_path", str(EXP_DIR / "data_splits" / "中石_train_n56.txt"),
            "--dataset", "custom",
            "--backbone", "wideresnet50",
            "--layers", "layer2",
            "--layers", "layer3",
            "--sampler", "approx_greedy_coreset",
            "--sampling_percentage", "0.1",
            "--resize", "448",
            "--imagesize", "448",
            "--save_dir", str(EXP_DIR / "中石_patchcore_n56"),
            "--skip_eval",
            "--gpu", "0",
        ]),
        ("中石_dinomaly2_base", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "中石_train_n56.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "中石_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "中石_dinomaly2_base"),
            "--eval_interval", "-1",
            "--cuda", "0",
        ]),
        ("中石_source_dg_adapter", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "中石_train_n56.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "中石_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "中石_source_dg_adapter"),
            "--eval_interval", "-1",
            "--source-dg",
            "--cuda", "0",
        ]),
        ("中石_source_dg_full_uot", [
            PYTHON, str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
            "--dataset", "custom",
            "--data_path", str(EXP_DIR / "data_splits" / "中石_train_n56.txt"),
            "--test_path", str(EXP_DIR / "data_splits" / "中石_test.txt"),
            "--backbone", "dinov2reg_vit_base_14",
            "--image_size", "448",
            "--crop_size", "448",
            "--batch_size", "4",
            "--max_iters", "2000",
            "--save_dir", str(EXP_DIR / "中石_source_dg_full_uot"),
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

    processes = []
    for gid in gpu_pool:
        p = mp.Process(target=worker, args=(gid, task_queue, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n=======================================================")
    print("ALL 8 TRAINING TASKS COMPLETED! SUMMARY:")
    print("=======================================================")
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    for name, ret, elapsed, log_p in sorted(results, key=lambda x: x[0]):
        status = "SUCCESS" if ret == 0 else f"FAILED(code={ret})"
        print(f"  {name:<25}: {status:<15} (took {elapsed:.1f}s) -> Log: {log_p}")

if __name__ == '__main__':
    main()
