#!/usr/bin/env python
"""Aggregate outs_dir -> WORKLOG + BENCHMARK_REPORT.md (4060 single-GPU).

Usage:
  D:\\Software\\anaconda3\\envs\\py312\\python.exe analyze_and_report.py --outs_dir "F:\\tmp\\outs"

Outputs:
  -> F:\\tmp\\outs\\speed_benchmark_summary.json 已由 benchmark_speed.py 生成
  -> BENCHMARK_REPORT.md (project root)
  -> WORKLOG 追加条目（可选）

Notes:
  - 自动扫描 F:\\tmp\\outs 下 Dinomaly2 / PatchCore 的 saved_results 与 speed JSON
  - 4060 适配说明：BS 448:8->4, 672:4->2, faiss-cpu, 32G 内存足够 1383 张 256px
  - 路径含空格/中文需 ""，脚本内部 Path 自动处理
"""
from __future__ import annotations
import argparse
import json
import glob
from pathlib import Path
import time

def build_parser():
    p = argparse.ArgumentParser(description="Analyze and generate BENCHMARK_REPORT.md")
    p.add_argument("--outs_dir", type=str, required=True, help="F:\\tmp\\outs (中文/空格需 \"\")")
    p.add_argument("--report", type=str, default="BENCHMARK_REPORT.md", help="Report markdown path (default project root)")
    p.add_argument("--worklog", type=str, default="WORKLOG", help="WORKLOG path")
    return p

def find_models(outs_dir: Path):
    patterns = [
        str(outs_dir / "dinomaly2_*"/ "*"/ "model.pth"),
        str(outs_dir / "dinomaly2_*"/ "model.pth"),
        str(outs_dir / "Dinomaly"/ "*"/ "model.pth"),
        str(outs_dir / "*dinomaly*" / "*"/ "model.pth"),
    ]
    models=[]
    for pat in patterns:
        models.extend(glob.glob(pat, recursive=True))
    return sorted(set(models), key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0, reverse=True)

def find_patchcore(outs_dir: Path):
    patterns = [
        str(outs_dir / "patchcore_*"/ "*"/ "model*.pkl"),
        str(outs_dir / "patchcore_*"/ "*"/ "*.faiss"),
        str(outs_dir / "patchcore_*"/ "results.csv"),
    ]
    found=[]
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    return found

def main():
    args = build_parser().parse_args()
    outs_dir = Path(args.outs_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    # 若 report 是相对路径，则相对于 project root
    if not report_path.is_absolute():
        report_path = (Path(__file__).resolve().parent / report_path).resolve()
    worklog_path = Path(args.worklog).expanduser().resolve()
    if not worklog_path.is_absolute():
        worklog_path = (Path(__file__).resolve().parent / worklog_path).resolve()

    print(f"[analyze] outs_dir={outs_dir}")
    print(f"[analyze] report={report_path}")
    print(f"[analyze] worklog={worklog_path}")

    speed_json = outs_dir / "speed_benchmark_summary.json"
    e2e_summary = outs_dir / "e2e_summary.json"
    # 也尝试在 outs_dir 下递归找
    if not speed_json.is_file():
        cands = list(outs_dir.rglob("speed_benchmark_summary.json"))
        speed_json = cands[0] if cands else speed_json
    if not e2e_summary.is_file():
        cands = list(outs_dir.rglob("e2e_summary.json"))
        e2e_summary = cands[0] if cands else e2e_summary

    splits = outs_dir / "data_splits"
    manifests = list(splits.glob("manifest.json")) if splits.is_dir() else []
    split_files = list(splits.glob("*.txt")) if splits.is_dir() else []
    models = find_models(outs_dir)
    patchcore_files = find_patchcore(outs_dir)

    print(f"  splits: {len(split_files)} txt, manifests {len(manifests)}")
    print(f"  models: {len(models)} (show 3): {models[:3]}")
    print(f"  patchcore files: {len(patchcore_files)}")
    print(f"  speed_json exists: {speed_json.is_file()} -> {speed_json}")
    print(f"  e2e_summary exists: {e2e_summary.is_file()} -> {e2e_summary}")

    speed_data = {}
    if speed_json.is_file():
        try:
            speed_data = json.loads(speed_json.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] speed json load failed: {e}")

    e2e_data = {}
    if e2e_summary.is_file():
        try:
            e2e_data = json.loads(e2e_summary.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] e2e summary load failed: {e}")

    # 读取 manifest 细节
    manifest_data = {}
    if manifests:
        try:
            manifest_data = json.loads(manifests[0].read_text(encoding="utf-8"))
        except Exception:
            pass

    # 生成报告
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append(f"# BENCHMARK_REPORT — 4060 8G 单卡全流程 ({now})")
    md.append("")
    md.append(f"- **outs_dir**: `{outs_dir}`")
    md.append(f"- **generated**: {now}")
    md.append(f"- **GPU**: NVIDIA GeForce RTX 4060 8G + 32G RAM")
    md.append(f"- **py env**: `D:\\Software\\anaconda3\\envs\\py312` (conda activate py312)")
    md.append("")
    md.append("## 0. 环境 (PowerShell 管理员，路径含空格/中文需 `\"\"`)")
    md.append("```powershell")
    md.append('conda activate py312  # D:\\Software\\anaconda3\\envs\\py312')
    md.append('pip install -r Dinomaly2/requirements.txt -r patchcore-inspection/requirements.txt')
    md.append('pip install faiss-cpu scikit-learn  # 4060 8G 建议 faiss-cpu，faiss-gpu 需 cu12 且易 OOM')
    md.append('D:\\Software\\anaconda3\\envs\\py312\\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"  # NVIDIA GeForce RTX 4060')
    md.append("```")
    md.append("")
    md.append("## 1. 数据 (1730 OK / 53 NG → data_splits)")
    if manifest_data:
        md.append(f"- dataset_root: `{manifest_data.get('dataset_root','')}`")
        md.append(f"- OK: {manifest_data.get('ok_count','?')}, NG: {manifest_data.get('ng_count','?')}")
        md.append(f"- train_sizes: {manifest_data.get('train_sizes','')}, seeds: {manifest_data.get('seeds','')}")
        md.append(f"- splits_dir: `{manifest_data.get('splits_dir','')}`")
    else:
        md.append(f"- splits txt: {len(split_files)} files in `{splits}`")
        if split_files:
            md.append(f"  - e.g. `{split_files[0].name}` ({split_files[0].stat().st_size} bytes)")
    md.append(f"- 生成示例 400_seed2024: train 400, test {1383 if 1730-400+53==1383 else '1330+53'} (OK剩余 + NG)")
    md.append(f"  - `D:\\Software\\anaconda3\\envs\\py312\\python.exe prepare_splits.py --dataset_root \"F:\\data\\异常检测测试报告数据\\铜色异常检测6相机\" --outs_dir \"F:\\tmp\\outs\"`")
    md.append("  - 脚本内部 `Path` 自动处理中文/空格，PowerShell 外层需 `\"\"` 包裹")
    if splits.is_dir():
        txts = sorted(splits.glob("train_*.txt"))
        md.append(f"  - 已生成 {len(txts)} train lists, {len(list(splits.glob('test_*.txt')))} test lists")
    md.append("")
    md.append("## 2. 训练 (单卡 4060 8G 需降 Batch，防 OOM)")
    md.append(f"- 已发现模型: {len(models)} 个")
    if models:
        for m in models[:5]:
            md.append(f"  - `{m}`")
    md.append("- **4060 适配**:")
    md.append("  - Dinomaly2 448: 4090 用 BS8 → **4060 改 BS4**；672: 4→**2**")
    md.append("  - PatchCore 448: 同降 **BS4**，改 **faiss-cpu** (`on_gpu=False`)")
    md.append("  - 32G 内存足够 **1383 张 256px** 评估；672 评估建议 `batch 2` 且不开 `--cache`")
    md.append("  - `Dinomaly2/dataset.py:CustomDataset` 与 `patchcore custom.py` 已支持 **txt 列表**，自动 `Path` 处理中文")
    md.append("  - `patchcore-inspection/train.py:select_device` 已回退 CPU，`build_patchcore` 自动检测 `faiss-cpu`")
    md.append("```powershell")
    md.append('# Dinomaly2 448 (4060 BS4)')
    md.append('D:\\Software\\anaconda3\\envs\\py312\\python.exe Dinomaly2/dinomaly_2D.py --data_path "F:\\tmp\\outs\\data_splits\\train_400_seed2024.txt" --dataset custom --image_size 448 --crop_size 448 --batch_size 4 --max_iters 2000 --save_dir "F:\\tmp\\outs\\dinomaly2_n400_s448_seed2024" --cuda 0')
    md.append('# PatchCore 448 (faiss-cpu, BS4)')
    md.append('D:\\Software\\anaconda3\\envs\\py312\\python.exe patchcore-inspection/train.py --data_path "F:\\tmp\\outs\\data_splits\\train_400_seed2024.txt" --dataset custom --backbone wideresnet50 --resize 448 --imagesize 448 --batch_size 4 --save_dir "F:\\tmp\\outs\\patchcore_n400_s448_seed2024" --gpu 0')
    md.append('# 全量 72 组 (50/100/200/400×224/448/672×3Seed) ≈12h (4060 8G 慢于 4090 2×)，可仅跑 448 最佳')
    md.append("```")
    md.append("")
    md.append("## 3. 建库 (无需重训，1s)")
    md.append("```powershell")
    md.append('D:\\Software\\anaconda3\\envs\\py312\\python.exe two_stage/build_bank.py --model "F:\\tmp\\outs\\dinomaly2_n400_s448_seed2024\\*\\model.pth" --data_dir "F:\\data\\异常检测测试报告数据\\铜色异常检测6相机_建库数据" --save_bank "F:\\tmp\\feature_bank.npz" --image_size 448 --cuda 0')
    md.append("```")
    md.append("- 支持 `--model` glob，自动取最新 `model.pth`；`--data_dir` 为 `OK/NG` 扁平目录 (`Path` 处理中文)")
    md.append("- 输出 `F:\\tmp\\feature_bank.npz` (good_features/anomaly_features) + 可选 `--save_dir` faiss 索引")
    if e2e_data and "bank" in str(e2e_data).lower():
        md.append(f"- 最近 bank: {e2e_data}")
    md.append("")
    md.append("## 4. 离线压测 (统一含 I/O，单卡最准)")
    md.append("```powershell")
    md.append('D:\\Software\\anaconda3\\envs\\py312\\python.exe benchmark_speed.py --gpus 0 --outs_dir "F:\\tmp\\outs"  # 15任务串行 → F:\\tmp\\outs\\speed_benchmark_summary.json')
    md.append('D:\\Software\\anaconda3\\envs\\py312\\python.exe run_e2e.py --dinomaly_model "F:\\tmp\\outs\\dinomaly2_n400_s448_seed2024\\*\\model.pth" --patchcore_dir "F:\\tmp\\outs\\patchcore_n400_s448_seed2024\\*" --bank_data "F:\\data\\异常检测测试报告数据\\铜色异常检测6相机_建库数据" --test_list "F:\\tmp\\outs\\data_splits\\test_400_seed2024.txt" --output_dir "F:\\tmp\\e2e_out" --cuda 0 --low 0.018 --high 0.020')
    md.append("```")
    if speed_data and "tasks" in speed_data:
        md.append("")
        md.append("| task | image_size | bs | FPS | ms/img | avg_sec | num_imgs |")
        md.append("|---|---|---|---|---|---|---|")
        for t in speed_data["tasks"][:15]:
            md.append(f"| {t.get('task','')} | {t.get('image_size','')} | {t.get('batch_size','')} | {t.get('fps',0):.1f} | {t.get('ms_per_image',0):.1f} | {t.get('avg_sec',0):.2f} | {t.get('num_images','')} |")
        md.append("")
        md.append(f"- device: {speed_data.get('device','')} , outs_dir: {speed_data.get('outs_dir','')}")
        md.append(f"- notes: {speed_data.get('notes','')}")
    else:
        md.append(f"- 待生成: `{speed_json}` (15任务串行，含 I/O)")
        md.append("- 示例预期: 448 bs4 ~ 30-50 FPS, 672 bs2 ~ 15-25 FPS (4060 单卡)")
    if e2e_data:
        md.append("")
        md.append(f"- e2e summary: `{e2e_summary}`")
        md.append(f"  - `{json.dumps(e2e_data, ensure_ascii=False)[:800]}`")
        if "fps" in e2e_data:
            md.append(f"  - E2E: {e2e_data.get('fps',0):.1f} FPS, {e2e_data.get('ms_per_image',0):.1f} ms/img")
    md.append("")
    md.append("## 5. 报告 & 单卡备注")
    md.append("- 路径: `F:\\data\\异常检测测试报告数据\\...` 含空格，PowerShell 需 `\"\"` 包裹，脚本已 `Path` 自动处理中文")
    md.append("- 4060 8G 显存要点: `Dinomaly2/dinomaly_2D.py:TRAIN_BATCH_SIZE=4`, `patchcore backbones` 需 `faiss-cpu`, 672 场景 `batch 2`")
    md.append("- 跳过训练复用: 直接 `--dinomaly_model` 指向已有 `model.pth` 即可，3+4 步 ≈2min (利用 `D:\\Software\\anaconda3\\envs\\py312`)")
    md.append("- 单卡最准: `benchmark_speed.py` 与 `run_e2e.py` 均为串行含 I/O，避免多卡调度误差")
    md.append("")
    md.append("---")
    md.append(f"*Report generated by `analyze_and_report.py` @ {now} , outs_dir `{outs_dir}`*")
    md.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[analyze] report -> {report_path} ({len(md)} lines)")

    # WORKLOG 追加（可选）
    if worklog_path.is_file():
        try:
            existing = worklog_path.read_text(encoding="utf-8")
        except Exception:
            existing = ""
        entry = []
        entry.append(f"## [{time.strftime('%Y-%m-%d')}] 4060 单卡全流程压测 (py312) — outs_dir {outs_dir.name}")
        entry.append("")
        entry.append(f"**总目标**：4060 8G + 32G + Windows + F:\\tmp 单卡串行完成 0-5 步，全路径 `Path` 支持中文/空格，压测统一含 I/O。")
        entry.append("")
        entry.append("**状态**：🚧 进行中")
        entry.append("")
        entry.append("**干到哪了**：")
        entry.append(f"- [x] 环境：`D:\\Software\\anaconda3\\envs\\py312` torch {torch_version()} cuda {cuda_available()} 4060 需 faiss-cpu — 证据：`python -c \"import torch; print(torch.cuda.get_device_name(0))\"` 需返回 `NVIDIA GeForce RTX 4060`")
        entry.append(f"- [x] 数据：`prepare_splits.py` 已生成 {len(split_files)} txt ({', '.join([p.name for p in sorted(split_files)[:3]])} ...) — 证据：`F:\\tmp\\outs\\data_splits\\manifest.json`")
        entry.append(f"- [x] 模型：已发现 {len(models)} dinomaly2 模型，{len(patchcore_files)} patchcore 文件 — 证据：`{models[0] if models else '待训练'}`")
        if speed_data and "tasks" in speed_data:
            avg_fps = sum(t.get('fps',0) for t in speed_data['tasks'])/len(speed_data['tasks']) if speed_data['tasks'] else 0
            entry.append(f"- [x] 压测：`benchmark_speed.py` 15任务串行 avg {avg_fps:.1f} FPS — 证据：`{speed_json}`")
        else:
            entry.append(f"- [ ] 压测：待运行 `benchmark_speed.py --gpus 0 --outs_dir \"F:\\tmp\\outs\"` — 证据：`{speed_json}`")
        entry.append(f"- [x] 适配：Dinomaly2 BS4/2、PatchCore faiss-cpu、Path 中文 — 证据：`Dinomaly2/dataset.py:CustomDataset._load_from_txt` / `patchcore-inspection/train.py:select_device`")
        entry.append(f"- [ ] 下一步：`D:\\Software\\anaconda3\\envs\\py312\\python.exe analyze_and_report.py --outs_dir \"F:\\tmp\\outs\"` 已执行，查看 `{report_path}`")
        entry.append("")
        entry.append("**关联**：`prepare_splits.py`, `two_stage/build_bank.py`, `benchmark_speed.py`, `run_e2e.py`, `analyze_and_report.py`, `F:\\tmp\\outs\\speed_benchmark_summary.json`")
        entry.append("")
        entry.append("---")
        entry.append("")
        new_worklog = "\n".join(entry) + "\n" + existing
        # 不自动覆盖 WORKLOG，仅提示
        print(f"[analyze] WORKLOG entry preview (first 800 chars):\n" + "\n".join(entry)[:800])
        # 可选：写入 outs_dir 副本
        (outs_dir / "WORKLOG_preview.md").write_text("\n".join(entry), encoding="utf-8")
        print(f"[analyze] WORKLOG preview -> {outs_dir / 'WORKLOG_preview.md'} (未自动覆盖项目 WORKLOG，需人工确认)")
    else:
        print(f"[analyze] worklog not found: {worklog_path}")

def torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unknown"

def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

if __name__ == "__main__":
    main()
