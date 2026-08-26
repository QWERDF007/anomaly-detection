#!/usr/bin/env python
"""Single-GPU full run: 50/100/200/400 × 224/448/672, max_iters 2000, sequential.
Logs to F:\tmp\outs\full_run.log
"""
import argparse, subprocess, sys, time, json
from pathlib import Path
from datetime import datetime

PYTHON = Path(r"D:\Software\anaconda3\envs\py312\python.exe")
ROOT = Path(__file__).resolve().parent

def build_parser():
    parser = argparse.ArgumentParser(description="Single-GPU full benchmark matrix runner")
    parser.add_argument("--outs_dir", type=str, default=r"F:\tmp\0826", help="Output directory for all benchmark artifacts")
    parser.add_argument("--splits_dir", type=str, default=r"F:\tmp\outs\data_splits", help="Directory containing train_N_seed2024.txt")
    parser.add_argument("--cuda", type=str, default="0", help="GPU index")
    parser.add_argument("--max_iters", type=int, default=2000, help="Stage 1 training iterations")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--image_sizes", type=int, nargs="+", default=[224, 448, 672])
    return parser

def main():
    args = build_parser().parse_args()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    outs_dir.mkdir(parents=True, exist_ok=True)
    log_file = outs_dir / "full_run.log"
    err_file = outs_dir / "full_run.err"
    summary_file = outs_dir / "full_run_summary.json"

    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run_cmd(cmd, cwd=None):
        log(f"RUN: {' '.join(str(c) for c in cmd)} (cwd={cwd})")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, encoding="utf-8", errors="replace")
        t1 = time.perf_counter()
        elapsed = t1 - t0
        log(f"EXIT {proc.returncode} in {elapsed:.1f}s")
        if proc.stdout:
            log(f"STDOUT tail:\n{proc.stdout[-2000:]}")
        if proc.stderr:
            log(f"STDERR tail:\n{proc.stderr[-2000:]}")
        with err_file.open("a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\nCMD: {' '.join(str(c) for c in cmd)}\nElapsed: {elapsed:.1f}s Return: {proc.returncode}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}\n")
        return proc.returncode == 0, elapsed

    log_file.write_text("", encoding="utf-8")
    err_file.write_text("", encoding="utf-8")
    log(f"=== Single-GPU Full Run Start (outs_dir={outs_dir}) ===")
    log(f"PYTHON={PYTHON} exists={PYTHON.is_file()}")
    import torch
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(int(args.cuda)) if torch.cuda.is_available() else 'cpu'}")

    train_sizes = args.train_sizes
    image_sizes = args.image_sizes
    batch_map = {224: 8, 448: 4, 672: 2}
    max_iters = args.max_iters
    splits = Path(args.splits_dir).expanduser().resolve()
    assert splits.is_dir(), f"splits not found {splits}"
    results = []
    total_t0 = time.perf_counter()
    for sz in image_sizes:
        bs = batch_map[sz]
        for n in train_sizes:
            train_txt = splits / f"train_{n}_seed2024.txt"
            test_txt = splits / f"test_{n}_seed2024.txt"
            log(f"\n{'='*80}\nConfig: N={n} SZ={sz} BS={bs} train={train_txt.name} test={test_txt.name}\n{'='*80}")
            # Check if already done
            d_save = outs_dir / f"dinomaly2_n{n}_s{sz}_seed2024"
            p_save = outs_dir / f"patchcore_n{n}_s{sz}_seed2024"
            d_models = list(d_save.rglob("model.pth")) if d_save.is_dir() else []
            p_models = list(p_save.rglob("nnscorer_search_index.faiss")) if p_save.is_dir() else []
            d_done = len(d_models) > 0
            p_done = len(p_models) > 0
            # Dinomaly2
            if d_done:
                log(f"Skip Dinomaly2 N={n} SZ={sz} already has {d_models[0]}")
                d_elapsed = None
            else:
                cmd = [str(PYTHON), str(ROOT / "Dinomaly2" / "dinomaly_2D.py"),
                       "--data_path", str(train_txt),
                       "--dataset", "custom",
                       "--image_size", str(sz),
                       "--crop_size", str(sz),
                       "--batch-size", str(bs),
                       "--max-iters", str(max_iters),
                       "--save_dir", str(d_save),
                       "--cuda", str(args.cuda),
                       "--backbone", "dinov2reg_vit_base_14"]
                ok, elapsed = run_cmd(cmd, cwd=str(ROOT / "Dinomaly2"))
                d_elapsed = elapsed if ok else None
                if not ok:
                    log(f"FAILED Dinomaly2 N={n} SZ={sz}")
                results.append({"step":"dinomaly_train", "n":n, "sz":sz, "elapsed":elapsed, "ok":ok})
            # PatchCore
            if p_done:
                log(f"Skip PatchCore N={n} SZ={sz} already has {p_models[0]}")
                p_elapsed = None
            else:
                cmd = [str(PYTHON), str(ROOT / "patchcore-inspection" / "train.py"),
                       "--data_path", str(train_txt),
                       "--dataset", "custom",
                       "--backbone", "wideresnet50",
                       "-imgsz", str(sz),
                       "-csz", str(sz),
                       "--batch_size", str(bs),
                       "--save_dir", str(p_save),
                       "--gpu", str(args.cuda)]
                ok, elapsed = run_cmd(cmd, cwd=str(ROOT / "patchcore-inspection"))
                p_elapsed = elapsed if ok else None
                if not ok:
                    log(f"FAILED PatchCore N={n} SZ={sz}")
                results.append({"step":"patchcore_train", "n":n, "sz":sz, "elapsed":elapsed, "ok":ok})
            # Build bank (if dinomaly model exists)
            d_models = list(d_save.rglob("model.pth")) if d_save.is_dir() else []
            if d_models:
                model = sorted(d_models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                bank_npz = d_save / "feature_bank.npz"
                if bank_npz.is_file():
                    log(f"Skip bank N={n} SZ={sz} exists {bank_npz}")
                else:
                    cmd = [str(PYTHON), str(ROOT / "two_stage" / "build_bank.py"),
                           "--model", str(model),
                           "--data_dir", r"F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据",
                           "--save_bank", str(bank_npz),
                           "--image_size", str(sz),
                           "--cuda", str(args.cuda)]
                    ok, elapsed = run_cmd(cmd, cwd=str(ROOT))
                    results.append({"step":"build_bank", "n":n, "sz":sz, "elapsed":elapsed, "ok":ok})
            else:
                log(f"No dinomaly model for bank N={n} SZ={sz}")

    # After all trainings, run benchmark_speed and run_e2e for each config
    log("\n=== Benchmark Speed ===")
    cmd = [str(PYTHON), str(ROOT / "benchmark_speed.py"), "--gpus", str(args.cuda), "--outs_dir", str(outs_dir)]
    run_cmd(cmd, cwd=str(ROOT))
    for sz in image_sizes:
        for n in train_sizes:
            d_save = outs_dir / f"dinomaly2_n{n}_s{sz}_seed2024"
            d_models = list(d_save.rglob("model.pth")) if d_save.is_dir() else []
            if not d_models:
                continue
            model = sorted(d_models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            test_txt = splits / f"test_{n}_seed2024.txt"
            out_e2e = outs_dir / f"e2e_out_n{n}_s{sz}"
            cmd = [str(PYTHON), str(ROOT / "run_e2e.py"),
                   "--dinomaly_model", str(model),
                   "--bank_data", r"F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据",
                   "--test_list", str(test_txt),
                   "--output_dir", str(out_e2e),
                   "--cuda", str(args.cuda),
                   "--low", "0.018",
                   "--high", "0.020",
                   "--image_size", str(sz)]
            run_cmd(cmd, cwd=str(ROOT))
    # Final report
    log("\n=== Generate Final Report ===")
    cmd = [str(PYTHON), str(ROOT / "generate_final_report_multisize.py"), "--outs_dir", str(outs_dir)]
    run_cmd(cmd, cwd=str(ROOT))
    # Also run analyze_and_report
    cmd = [str(PYTHON), str(ROOT / "analyze_and_report.py"), "--outs_dir", str(outs_dir)]
    run_cmd(cmd, cwd=str(ROOT))
    # Plot full benchmark comparison suite
    log("\n=== Plot Full Benchmark Suite ===")
    cmd = [str(PYTHON), str(ROOT / "plot_evaluation_charts.py"), "--chart_dir", str(outs_dir / "charts"), "--full_benchmark"]
    run_cmd(cmd, cwd=str(ROOT))
    total = time.perf_counter() - total_t0
    log(f"\n=== ALL DONE in {total:.1f}s ({total/3600:.2f}h) ===")
    summary_file.write_text(json.dumps({"finished": datetime.now().isoformat(), "total_sec": total, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    sys.exit(main())
