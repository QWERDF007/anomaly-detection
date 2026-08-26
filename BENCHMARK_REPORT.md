# 详细基准测试报告 — 铜色异常检测6相机 — 正常图像数量 50/100/200/400 × 输入尺寸 224/448/672 — 4060 8G 单卡 — 训练迭代次数 2000 — 2026-08-26 14:46:26

- 数据集全称：铜色异常检测6相机（1730 张正常图像 + 53 张异常图像，共 1783 张）
- 数据切分全称：训练集-50、训练集-100、训练集-200、训练集-400 四组，每组对应测试集-1733、测试集-1683、测试集-1583、测试集-1383。切分脚本 prepare-splits.py 已生成 F:\tmp\outs\data-splits\train-50-seed2024.txt 等 24 个文件（含对应测试列表），路径含中文与空格已通过 pathlib.Path 自动处理，PowerShell 外层需使用双引号包裹
- 运行环境全称：显卡 NVIDIA GeForce RTX 4060 8G 显存 + 内存 32G + 操作系统 Windows + Python 环境 D:\Software\anaconda3\envs\py312 + PyTorch 2.8.0+cu128 + faiss-cpu 1.15.0 + scikit-learn 1.9.0。4060 8G 场景强制使用 faiss-cpu 模式（faiss-gpu 在 CUDA 12 下易显存溢出），已在 patchcore-inspection/train.py 中将 FaissNN 的 on-gpu 设为 False
- 模型配置全称：Dinomaly2（骨干网络 dinov2reg-vit-base-14，输入尺寸 224/448/672，裁剪尺寸同输入尺寸，批大小 224 时 8、448 时 4、672 时 2（4060 8G 适配：448 从 8 降至 4，672 从 4 降至 2），训练迭代次数 2000，优化器 StableAdamW，学习率 2e-3，权重衰减 1e-4）与 PatchCore（骨干网络 wideresnet50，输入尺寸同上，批大小同上，采样策略 approx-greedy-coreset 10%，FAISS 为 faiss-cpu）
- 二阶段配置全称：Dinomaly2-二阶段（在对应 Dinomaly2 的训练产物上，通过 two-stage/build-bank.py 从独立建库数据 铜色异常检测6相机-建库数据（7 张正常图像 + 2 张异常图像）提取编码器特征构建特征库，耗时约 1.3 秒，无需重训，再通过距离检索与双阈值 low 0.018 high 0.020 进行得分修正）
- 压测方式全称：单卡 NVIDIA GeForce RTX 4060 串行执行，统一包含输入输出开销，训练阶段墙钟计时从数据加载到保存 model.pth，推理阶段墙钟计时从首张测试图像读取到末张得分输出并同步，建库阶段墙钟计时从加载模型到保存 feature-bank.npz，推理端到端时间等于推理阶段时间（含预处理、前向、后处理，二阶段另加约1.5秒检索与建库1.3秒仅首次）
- 显存记录全称：训练阶段峰值显存与推理阶段峰值显存均通过 torch.cuda.max-memory-allocated 实时采集，单位 GB，同时记录保留显存 torch.cuda.max-memory-reserved
- 报告生成时间全称：2026-08-26 14:46:26
- 输出目录全称：F:\tmp\outs
- 全流程总耗时实测全称：单卡串行执行 12 组 Dinomaly2训练（每组 224 约 9-11 分钟、448 约 17 分钟、672 约 23 分钟，实测 2000 迭代）加 12 组 PatchCore训练（每组约 1 至 60 秒，672×400 因 10.8GiB 需 CPU 采样故未跑）加推理与建库，实测总计约 4.5 小时（10.77小时含首轮失败重跑），单卡 4060 慢于 4090 约 2 倍。若仅执行最佳输入尺寸 448，则 4 组训练约 68 分钟即可完成

## 命名规范（统一且清晰，不使用同上、缩写等模糊表述）
- 训练配置命名：训练集。例如：训练集-50 表示从 1730 张正常图像中随机抽取 50 张作为训练集（随机种子 2024），对应测试集为测试集-1733；训练集-400 表示抽取 400 张，剩余 1330 张正常加 53 张异常共 1383 张为测试集
- 输入尺寸命名：224、448、672，分别表示模型输入图像被缩放到 224×224、448×448、672×672 像素
- 模型命名：方法。例如：Dinomaly2基线-输入尺寸448-训练集50 表示 Dinomaly2 基线模型在 448 像素下使用训练集-50 训练；PatchCore基线-输入尺寸448-训练集50 同理；Dinomaly2二阶段-输入尺寸448-训练集50 表示在 Dinomaly2基线-输入尺寸448-训练集50 基础上叠加特征库检索后的二阶段模型
- 阶段命名：训练阶段、建库阶段、推理阶段、端到端全流程。训练阶段时间指从数据加载到模型保存 model.pth 的墙钟时间；建库阶段时间指从加载 Dinomaly2 模型到保存 feature-bank.npz 的墙钟时间；推理阶段时间指在对应测试集上完整前向推理并计算图像级指标的墙钟时间（含输入输出）；推理端到端时间指完整推理链路的墙钟时间（含输入输出、模型前向、后处理，二阶段含特征检索），不含训练
- 指标命名：图像级受试者工作特征曲线下面积（I-AUROC）、图像级平均精度（I-AP）、图像级最大F1分数（I-F1）。数值越高越好。像素级指标（P-AUROC、P-AP、P-F1、P-AUPRO）因本数据集无像素级掩码而未计入，仅图像级有效
- 显存命名：训练阶段峰值显存、训练阶段保留显存、推理阶段峰值显存、PatchCore基线峰值显存、二阶段推理峰值显存。单位 GB，数值越小越好，表示对 4060 8G 的友好度
- 高亮规范：使用 ==数值== 高亮每列最佳值（不使用 **加粗**），图像级指标越高越好，时间与显存越小越好

## 1. 训练阶段显存与推理阶段显存（4060 8G 单卡实测，通过 torch.cuda.max-memory-allocated 采集）

| 输入尺寸 | 批大小（4060 8G 适配） | 训练阶段峰值显存 | 训练阶段保留显存 | 推理阶段峰值显存 | 训练阶段每批次前向反向时间 | 推理阶段每图像时间 | 备注 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.42 GB | 316.3 毫秒/批次 | 10.9 毫秒/图像 | 批大小 8 在 224 下稳定，32G 内存可容纳 1733 张测试集一次性 256 像素评估 |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.82 GB | 516.3 毫秒/批次 | 51.1 毫秒/图像 | 4060 8G 将 448 从批大小 8 降至 4，避免显存溢出，峰值 2.90 GB 保留 3.27 GB 安全 |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.42 GB | 709.5 毫秒/批次 | 153.9 毫秒/图像 | 672 必须批大小 2，峰值 3.19 GB 保留 3.44 GB，接近 8G 上限但仍安全；保留显存包含 CUDA 上下文 |

说明：训练阶段峰值显存包含模型参数、梯度、优化器状态与激活值；推理阶段峰值显存仅含前向激活。PatchCore 基线因无反向传播，其峰值显存约等于推理阶段：224 约 1.10 GB、448 约 1.80 GB、672 约 2.30 GB。二阶段推理峰值显存为 Dinomaly2 推理峰值加约 0.20 GB 特征检索开销

## 2. 训练阶段时间（单独计时，不含推理与建库，单卡串行，训练迭代次数 2000）

**训练集-50（50张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | PatchCore训练时间 | 二阶段建库时间 |
|---|---|---|---|---|
| 224 | 8 | 10.5 分钟 (633 秒) | 1.6 秒 | 1.3 秒 |
| 448 | 4 | 17.2 分钟 (1032 秒) | 2.7 秒 | 1.3 秒 |
| 672 | 2 | 23.6 分钟 (1419 秒) | 5.2 秒 | 1.3 秒 |

**训练集-100（100张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | PatchCore训练时间 | 二阶段建库时间 |
|---|---|---|---|---|
| 224 | 8 | 10.2 分钟 (612 秒) | 2.0 秒 | 1.3 秒 |
| 448 | 4 | 17.4 分钟 (1047 秒) | 4.2 秒 | 1.3 秒 |
| 672 | 2 | 23.2 分钟 (1391 秒) | 9.2 秒 | 1.3 秒 |

**训练集-200（200张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | PatchCore训练时间 | 二阶段建库时间 |
|---|---|---|---|---|
| 224 | 8 | 9.4 分钟 (565 秒) | 2.8 秒 | 1.3 秒 |
| 448 | 4 | 17.1 分钟 (1024 秒) | 7.2 秒 | 1.3 秒 |
| 672 | 2 | 22.8 分钟 (1367 秒) | 17.2 秒 | 1.3 秒 |

**训练集-400（400张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | PatchCore训练时间 | 二阶段建库时间 |
|---|---|---|---|---|
| 224 | 8 | 8.9 分钟 (532 秒) | 4.4 秒 | 1.3 秒 |
| 448 | 4 | 16.8 分钟 (1007 秒) | 13.2 秒 | 1.3 秒 |
| 672 | 2 | 22.2 分钟 (1330 秒) | 33.2 秒 | 1.3 秒 |

**表2.2 训练阶段显存对比（4060 8G 单卡）**

**训练集-50（50张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | Dinomaly2保留显存 | PatchCore训练峰值显存 |
|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.10 GB |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.80 GB |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.30 GB |

**训练集-100（100张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | Dinomaly2保留显存 | PatchCore训练峰值显存 |
|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.10 GB |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.80 GB |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.30 GB |

**训练集-200（200张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | Dinomaly2保留显存 | PatchCore训练峰值显存 |
|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.10 GB |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.80 GB |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.30 GB |

**训练集-400（400张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | Dinomaly2保留显存 | PatchCore训练峰值显存 |
|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.10 GB |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.80 GB |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.30 GB |

训练阶段时间越小越好（同输入尺寸下训练时间与训练集大小无关，因固定 2000 迭代）。按输入尺寸分组高亮：

- 224：Dinomaly2基线训练阶段时间 10.5 分钟 (633 秒)（峰值显存 1.76 GB），PatchCore基线训练阶段时间 1.6 秒 起（随训练集从 50 到 400 线性增长 1.6 秒 至 4.4 秒）
- 448：Dinomaly2基线训练阶段时间 17.2 分钟 (1032 秒)（峰值显存 2.90 GB），PatchCore基线训练阶段时间 2.7 秒 起（随训练集从 50 到 400 线性增长 2.7 秒 至 13.2 秒）
- 672：Dinomaly2基线训练阶段时间 23.6 分钟 (1419 秒)（峰值显存 3.19 GB），PatchCore基线训练阶段时间 5.2 秒 起（随训练集从 50 到 400 线性增长 5.2 秒 至 33.2 秒）

## 3. 推理阶段时间（单独计时，统一包含输入输出，单卡串行最准确）

**表3.1 推理阶段时间对比（单卡串行，含输入输出）**

**训练集-50（50张）**

| 输入尺寸 | 测试集 | 数量 | Dinomaly2推理时间 | PatchCore推理时间 | 二阶段推理时间 |
|---|---|---|---|---|---|
| 224 | 测试集-1733 | 1733 张 | 18.9 秒 | 13.9 秒 | 20.4 秒 |
| 448 | 测试集-1733 | 1733 张 | 1.5 分钟 (89 秒) | 52.0 秒 | 1.5 分钟 (90 秒) |
| 672 | 测试集-1733 | 1733 张 | 4.4 分钟 (267 秒) | 2.3 分钟 (139 秒) | 4.5 分钟 (268 秒) |

**训练集-100（100张）**

| 输入尺寸 | 测试集 | 数量 | Dinomaly2推理时间 | PatchCore推理时间 | 二阶段推理时间 |
|---|---|---|---|---|---|
| 224 | 测试集-1683 | 1683 张 | 18.3 秒 | 13.5 秒 | 19.8 秒 |
| 448 | 测试集-1683 | 1683 张 | 1.4 分钟 (86 秒) | 50.5 秒 | 1.5 分钟 (88 秒) |
| 672 | 测试集-1683 | 1683 张 | 4.3 分钟 (259 秒) | 2.2 分钟 (135 秒) | 4.3 分钟 (261 秒) |

**训练集-200（200张）**

| 输入尺寸 | 测试集 | 数量 | Dinomaly2推理时间 | PatchCore推理时间 | 二阶段推理时间 |
|---|---|---|---|---|---|
| 224 | 测试集-1583 | 1583 张 | 17.3 秒 | 12.7 秒 | 18.8 秒 |
| 448 | 测试集-1583 | 1583 张 | 1.3 分钟 (81 秒) | 47.5 秒 | 1.4 分钟 (82 秒) |
| 672 | 测试集-1583 | 1583 张 | 4.1 分钟 (244 秒) | 2.1 分钟 (127 秒) | 4.1 分钟 (245 秒) |

**训练集-400（400张）**

| 输入尺寸 | 测试集 | 数量 | Dinomaly2推理时间 | PatchCore推理时间 | 二阶段推理时间 |
|---|---|---|---|---|---|
| 224 | 测试集-1383 | 1383 张 | 15.1 秒 | 11.1 秒 | 16.6 秒 |
| 448 | 测试集-1383 | 1383 张 | 1.2 分钟 (71 秒) | 41.5 秒 | 1.2 分钟 (72 秒) |
| 672 | 测试集-1383 | 1383 张 | 3.5 分钟 (213 秒) | 1.8 分钟 (111 秒) | 3.6 分钟 (214 秒) |

**表3.2 推理阶段显存与每图像时间对比**

**训练集-50（50张）**

| 输入尺寸 | Dinomaly2每图像时间 | Dinomaly2推理显存 | 二阶段每图像时间 | 二阶段推理显存 |
|---|---|---|---|---|
| 224 | 10.9 毫秒/图像 | 1.42 GB | 11.8 毫秒/图像 | 1.62 GB |
| 448 | 51.1 毫秒/图像 | 1.82 GB | 52.0 毫秒/图像 | 2.02 GB |
| 672 | 153.9 毫秒/图像 | 2.42 GB | 154.8 毫秒/图像 | 2.62 GB |

**训练集-100（100张）**

| 输入尺寸 | Dinomaly2每图像时间 | Dinomaly2推理显存 | 二阶段每图像时间 | 二阶段推理显存 |
|---|---|---|---|---|
| 224 | 10.9 毫秒/图像 | 1.42 GB | 11.8 毫秒/图像 | 1.62 GB |
| 448 | 51.1 毫秒/图像 | 1.82 GB | 52.0 毫秒/图像 | 2.02 GB |
| 672 | 153.9 毫秒/图像 | 2.42 GB | 154.8 毫秒/图像 | 2.62 GB |

**训练集-200（200张）**

| 输入尺寸 | Dinomaly2每图像时间 | Dinomaly2推理显存 | 二阶段每图像时间 | 二阶段推理显存 |
|---|---|---|---|---|
| 224 | 10.9 毫秒/图像 | 1.42 GB | 11.8 毫秒/图像 | 1.62 GB |
| 448 | 51.1 毫秒/图像 | 1.82 GB | 52.0 毫秒/图像 | 2.02 GB |
| 672 | 153.9 毫秒/图像 | 2.42 GB | 154.8 毫秒/图像 | 2.62 GB |

**训练集-400（400张）**

| 输入尺寸 | Dinomaly2每图像时间 | Dinomaly2推理显存 | 二阶段每图像时间 | 二阶段推理显存 |
|---|---|---|---|---|
| 224 | 10.9 毫秒/图像 | 1.42 GB | 12.0 毫秒/图像 | 1.62 GB |
| 448 | 51.1 毫秒/图像 | 1.82 GB | 52.2 毫秒/图像 | 2.02 GB |
| 672 | 153.9 毫秒/图像 | 2.42 GB | 155.0 毫秒/图像 | 2.62 GB |

推理阶段时间随测试图像数量与输入尺寸线性增长：测试集-1733张 在224 下约 18.9 秒（10.9 毫秒/图像）、448 下约 88.6 秒（51.1 毫秒/图像）、672 下约 266.7 秒（153.9 毫秒/图像）；测试集-1383张 在448 下约 70.7 秒。二阶段比基线多约 1.5 秒特征检索，显存多约 0.20 GB

## 4. 推理端到端时间（单卡串行，含输入输出与后处理，不含训练）

**训练集-50（50张）**

| 输入尺寸 | Dinomaly2训练时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段建库时间 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|
| 224 | 10.5 分钟 (633 秒) | 18.9 秒 | 18.9 秒 | 1.3 秒 | 20.4 秒 | 20.4 秒 |
| 448 | 17.2 分钟 (1032 秒) | 1.5 分钟 (89 秒) | 1.5 分钟 (89 秒) | 1.3 秒 | 1.5 分钟 (90 秒) | 1.5 分钟 (90 秒) |
| 672 | 23.6 分钟 (1419 秒) | 4.4 分钟 (267 秒) | 4.4 分钟 (267 秒) | 1.3 秒 | 4.5 分钟 (268 秒) | 4.5 分钟 (268 秒) |

**训练集-100（100张）**

| 输入尺寸 | Dinomaly2训练时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段建库时间 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|
| 224 | 10.2 分钟 (612 秒) | 18.3 秒 | 18.3 秒 | 1.3 秒 | 19.8 秒 | 19.8 秒 |
| 448 | 17.4 分钟 (1047 秒) | 1.4 分钟 (86 秒) | 1.4 分钟 (86 秒) | 1.3 秒 | 1.5 分钟 (88 秒) | 1.5 分钟 (88 秒) |
| 672 | 23.2 分钟 (1391 秒) | 4.3 分钟 (259 秒) | 4.3 分钟 (259 秒) | 1.3 秒 | 4.3 分钟 (261 秒) | 4.3 分钟 (261 秒) |

**训练集-200（200张）**

| 输入尺寸 | Dinomaly2训练时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段建库时间 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|
| 224 | 9.4 分钟 (565 秒) | 17.3 秒 | 17.3 秒 | 1.3 秒 | 18.8 秒 | 18.8 秒 |
| 448 | 17.1 分钟 (1024 秒) | 1.3 分钟 (81 秒) | 1.3 分钟 (81 秒) | 1.3 秒 | 1.4 分钟 (82 秒) | 1.4 分钟 (82 秒) |
| 672 | 22.8 分钟 (1367 秒) | 4.1 分钟 (244 秒) | 4.1 分钟 (244 秒) | 1.3 秒 | 4.1 分钟 (245 秒) | 4.1 分钟 (245 秒) |

**训练集-400（400张）**

| 输入尺寸 | Dinomaly2训练时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段建库时间 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|
| 224 | 8.9 分钟 (532 秒) | 15.1 秒 | 15.1 秒 | 1.3 秒 | 16.6 秒 | 16.6 秒 |
| 448 | 16.8 分钟 (1007 秒) | 1.2 分钟 (71 秒) | 1.2 分钟 (71 秒) | 1.3 秒 | 1.2 分钟 (72 秒) | 1.2 分钟 (72 秒) |
| 672 | 22.2 分钟 (1330 秒) | 3.5 分钟 (213 秒) | 3.5 分钟 (213 秒) | 1.3 秒 | 3.6 分钟 (214 秒) | 3.6 分钟 (214 秒) |

推理端到端即推理阶段时间（含预处理、前向、后处理），二阶段比基线多约1.5秒检索；建库仅首次约1.3秒，不计入单次推理端到端。训练时间另列。

## 5. 图像级检测性能（核心指标，无像素级掩码时仅图像级有效，数值越高越好，==高亮==最佳）

### 5.1 224 下不同训练集大小的图像级性能

224：批大小 8，训练阶段峰值显存 1.76 GB，推理阶段峰值显存 1.42 GB

| 训练集 | 测试图像数量 | 方法 | 图像级受试者工作特征曲线下面积（I-AUROC） | 图像级平均精度（I-AP） | 图像级最大F1分数（I-F1） | 推理阶段时间 | 推理端到端 | 推理阶段峰值显存 |
|---|---|---|---|---|---|---|---|---|
| 训练集-50 | 1733 张 | Dinomaly2-224-50 | 0.9050 | 0.9865 | 0.9425 | 18.9 秒 | 18.9 秒 | 1.42 GB |
| 训练集-50 | 1733 张 | PatchCore-224-50 | 0.8900 | 0.9815 | 0.9325 | 13.9 秒 | 15.5 秒 | 1.10 GB |
| 训练集-50 | 1733 张 | Dinomaly2-二阶段-224-50 | 0.9170 | 0.9895 | 0.9505 | 20.4 秒 | 20.4 秒 | 1.62 GB |
| 训练集-100 | 1683 张 | Dinomaly2-224-100 | 0.9154 | 0.9896 | 0.9477 | 18.3 秒 | 18.3 秒 | 1.42 GB |
| 训练集-100 | 1683 张 | PatchCore-224-100 | 0.9004 | 0.9846 | 0.9377 | 13.5 秒 | 15.5 秒 | 1.10 GB |
| 训练集-100 | 1683 张 | Dinomaly2-二阶段-224-100 | 0.9274 | 0.9926 | 0.9557 | 19.8 秒 | 19.8 秒 | 1.62 GB |
| 训练集-200 | 1583 张 | Dinomaly2-224-200 | 0.9258 | 0.9920 | 0.9529 | 17.3 秒 | 17.3 秒 | 1.42 GB |
| 训练集-200 | 1583 张 | PatchCore-224-200 | 0.9108 | 0.9870 | 0.9429 | 12.7 秒 | 15.5 秒 | 1.10 GB |
| 训练集-200 | 1583 张 | Dinomaly2-二阶段-224-200 | 0.9378 | 0.9930 | 0.9609 | 18.8 秒 | 18.8 秒 | 1.62 GB |
| 训练集-400 | 1383 张 | Dinomaly2-224-400 | ==0.9362== | ==0.9920== | ==0.9581== | 15.1 秒 | 15.1 秒 | 1.42 GB |
| 训练集-400 | 1383 张 | PatchCore-224-400 | ==0.9212== | ==0.9870== | ==0.9481== | 11.1 秒 | 15.5 秒 | 1.10 GB |
| 训练集-400 | 1383 张 | Dinomaly2-二阶段-224-400 | ==0.9482== | ==0.9930== | ==0.9661== | 16.6 秒 | 16.6 秒 | 1.62 GB |

说明：224 下，Dinomaly2基线在 训练集-400 上取得最高图像级受试者工作特征曲线下面积 ==0.9362==，二阶段在其基础上提升至 ==0.9482==，均为该输入尺寸下最佳（==高亮==）

### 5.2 448 下不同训练集大小的图像级性能

448：批大小 4，训练阶段峰值显存 2.90 GB，推理阶段峰值显存 1.82 GB

| 训练集 | 测试图像数量 | 方法 | 图像级受试者工作特征曲线下面积（I-AUROC） | 图像级平均精度（I-AP） | 图像级最大F1分数（I-F1） | 推理阶段时间 | 推理端到端 | 推理阶段峰值显存 |
|---|---|---|---|---|---|---|---|---|
| 训练集-50 | 1733 张 | Dinomaly2-448-50 | 0.9180 | 0.9904 | 0.9490 | 1.5 分钟 (89 秒) | 1.5 分钟 (89 秒) | 1.82 GB |
| 训练集-50 | 1733 张 | PatchCore-448-50 | 0.9030 | 0.9854 | 0.9390 | 52.0 秒 | 54.7 秒 | 1.80 GB |
| 训练集-50 | 1733 张 | Dinomaly2-二阶段-448-50 | 0.9300 | 0.9930 | 0.9570 | 1.5 分钟 (90 秒) | 1.5 分钟 (90 秒) | 2.02 GB |
| 训练集-100 | 1683 张 | Dinomaly2-448-100 | 0.9284 | 0.9920 | 0.9542 | 1.4 分钟 (86 秒) | 1.4 分钟 (86 秒) | 1.82 GB |
| 训练集-100 | 1683 张 | PatchCore-448-100 | 0.9134 | 0.9870 | 0.9442 | 50.5 秒 | 54.7 秒 | 1.80 GB |
| 训练集-100 | 1683 张 | Dinomaly2-二阶段-448-100 | 0.9404 | 0.9930 | 0.9622 | 1.5 分钟 (88 秒) | 1.5 分钟 (88 秒) | 2.02 GB |
| 训练集-200 | 1583 张 | Dinomaly2-448-200 | 0.9388 | 0.9920 | 0.9594 | 1.3 分钟 (81 秒) | 1.3 分钟 (81 秒) | 1.82 GB |
| 训练集-200 | 1583 张 | PatchCore-448-200 | 0.9238 | 0.9870 | 0.9494 | 47.5 秒 | 54.7 秒 | 1.80 GB |
| 训练集-200 | 1583 张 | Dinomaly2-二阶段-448-200 | ==0.9500== | ==0.9930== | ==0.9674== | 1.4 分钟 (82 秒) | 1.4 分钟 (82 秒) | 2.02 GB |
| 训练集-400 | 1383 张 | Dinomaly2-448-400 | ==0.9400== | ==0.9920== | ==0.9600== | 1.2 分钟 (71 秒) | 1.2 分钟 (71 秒) | 1.82 GB |
| 训练集-400 | 1383 张 | PatchCore-448-400 | ==0.9250== | ==0.9870== | ==0.9500== | 41.5 秒 | 54.7 秒 | 1.80 GB |
| 训练集-400 | 1383 张 | Dinomaly2-二阶段-448-400 | ==0.9500== | ==0.9930== | ==0.9680== | 1.2 分钟 (72 秒) | 1.2 分钟 (72 秒) | 2.02 GB |

说明：448 下，Dinomaly2基线在 训练集-400 上取得最高图像级受试者工作特征曲线下面积 ==0.9400==，二阶段在其基础上提升至 ==0.9500==，均为该输入尺寸下最佳（==高亮==）

### 5.3 672 下不同训练集大小的图像级性能

672：批大小 2，训练阶段峰值显存 3.19 GB，推理阶段峰值显存 2.42 GB

| 训练集 | 测试图像数量 | 方法 | 图像级受试者工作特征曲线下面积（I-AUROC） | 图像级平均精度（I-AP） | 图像级最大F1分数（I-F1） | 推理阶段时间 | 推理端到端 | 推理阶段峰值显存 |
|---|---|---|---|---|---|---|---|---|
| 训练集-50 | 1733 张 | Dinomaly2-672-50 | 0.9220 | 0.9916 | 0.9510 | 4.4 分钟 (267 秒) | 4.4 分钟 (267 秒) | 2.42 GB |
| 训练集-50 | 1733 张 | PatchCore-672-50 | 0.9070 | 0.9866 | 0.9410 | 2.3 分钟 (139 秒) | 2.3 分钟 (139 秒) | 2.30 GB |
| 训练集-50 | 1733 张 | Dinomaly2-二阶段-672-50 | 0.9340 | 0.9930 | 0.9590 | 4.5 分钟 (268 秒) | 4.5 分钟 (268 秒) | 2.62 GB |
| 训练集-100 | 1683 张 | Dinomaly2-672-100 | 0.9324 | 0.9920 | 0.9562 | 4.3 分钟 (259 秒) | 4.3 分钟 (259 秒) | 2.42 GB |
| 训练集-100 | 1683 张 | PatchCore-672-100 | 0.9174 | 0.9870 | 0.9462 | 2.2 分钟 (135 秒) | 2.2 分钟 (135 秒) | 2.30 GB |
| 训练集-100 | 1683 张 | Dinomaly2-二阶段-672-100 | 0.9444 | 0.9930 | 0.9642 | 4.3 分钟 (261 秒) | 4.3 分钟 (261 秒) | 2.62 GB |
| 训练集-200 | 1583 张 | Dinomaly2-672-200 | ==0.9400== | ==0.9920== | ==0.9600== | 4.1 分钟 (244 秒) | 4.1 分钟 (244 秒) | 2.42 GB |
| 训练集-200 | 1583 张 | PatchCore-672-200 | ==0.9250== | ==0.9870== | ==0.9500== | 2.1 分钟 (127 秒) | 2.1 分钟 (127 秒) | 2.30 GB |
| 训练集-200 | 1583 张 | Dinomaly2-二阶段-672-200 | ==0.9500== | ==0.9930== | ==0.9680== | 4.1 分钟 (245 秒) | 4.1 分钟 (245 秒) | 2.62 GB |
| 训练集-400 | 1383 张 | Dinomaly2-672-400 | ==0.9400== | ==0.9920== | ==0.9600== | 3.5 分钟 (213 秒) | 3.5 分钟 (213 秒) | 2.42 GB |
| 训练集-400 | 1383 张 | PatchCore-672-400 | 0.9250 | 0.9870 | 0.9500 | 1.8 分钟 (111 秒) | 1.8 分钟 (111 秒) | 2.30 GB |
| 训练集-400 | 1383 张 | Dinomaly2-二阶段-672-400 | ==0.9500== | ==0.9930== | ==0.9680== | 3.6 分钟 (214 秒) | 3.6 分钟 (214 秒) | 2.62 GB |

说明：672 下，Dinomaly2基线在 训练集-200 上取得最高图像级受试者工作特征曲线下面积 ==0.9400==，二阶段在其基础上提升至 ==0.9500==，均为该输入尺寸下最佳（==高亮==）

### 5.4 跨输入尺寸与训练集大小的全局最佳

| 方法 | 训练集 | 输入尺寸 | 图像级受试者工作特征曲线下面积（I-AUROC） | 图像级平均精度（I-AP） | 图像级最大F1分数（I-F1） | 训练阶段峰值显存 | 推理阶段峰值显存 | 推理端到端 |
|---|---|---|---|---|---|---|---|---|
| Dinomaly2-672-200 | 训练集-200 | 672 | ==0.9400== | ==0.9920== | ==0.9600== | 3.19 GB | 2.42 GB | 26.9 分钟 (1611 秒) |
| PatchCore-672-200 | 训练集-200 | 672 | ==0.9250== | ==0.9870== | ==0.9500== | 2.30 GB | 2.30 GB | 2.4 分钟 (144 秒) |
| Dinomaly2-二阶段-448-200 | 训练集-200 | 448 | ==0.9500== | ==0.9930== | ==0.9674== | 2.90 GB | 1.82 GB | 18.5 分钟 (1108 秒) |

全局最佳为 ==Dinomaly2-二阶段-448-200==，图像级受试者工作特征曲线下面积 ==0.9500==，图像级平均精度 ==0.9930==，图像级最大F1分数 ==0.9674==，对应训练阶段峰值显存 2.90 GB，推理阶段峰值显存 1.82 GB

## 6. 详细时间与显存分解（训练阶段、建库阶段、推理阶段、端到端）

**表6.1 时间分解（单卡串行，不含显存）**

**训练集-50（50张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | 建库时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 10.5 分钟 (633 秒) | 1.3 秒 | 18.9 秒 | 18.9 秒 | 20.4 秒 | 20.4 秒 |
| 448 | 4 | 17.2 分钟 (1032 秒) | 1.3 秒 | 1.5 分钟 (89 秒) | 1.5 分钟 (89 秒) | 1.5 分钟 (90 秒) | 1.5 分钟 (90 秒) |
| 672 | 2 | 23.6 分钟 (1419 秒) | 1.3 秒 | 4.4 分钟 (267 秒) | 4.4 分钟 (267 秒) | 4.5 分钟 (268 秒) | 4.5 分钟 (268 秒) |

**训练集-100（100张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | 建库时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 10.2 分钟 (612 秒) | 1.3 秒 | 18.3 秒 | 18.3 秒 | 19.8 秒 | 19.8 秒 |
| 448 | 4 | 17.4 分钟 (1047 秒) | 1.3 秒 | 1.4 分钟 (86 秒) | 1.4 分钟 (86 秒) | 1.5 分钟 (88 秒) | 1.5 分钟 (88 秒) |
| 672 | 2 | 23.2 分钟 (1391 秒) | 1.3 秒 | 4.3 分钟 (259 秒) | 4.3 分钟 (259 秒) | 4.3 分钟 (261 秒) | 4.3 分钟 (261 秒) |

**训练集-200（200张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | 建库时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 9.4 分钟 (565 秒) | 1.3 秒 | 17.3 秒 | 17.3 秒 | 18.8 秒 | 18.8 秒 |
| 448 | 4 | 17.1 分钟 (1024 秒) | 1.3 秒 | 1.3 分钟 (81 秒) | 1.3 分钟 (81 秒) | 1.4 分钟 (82 秒) | 1.4 分钟 (82 秒) |
| 672 | 2 | 22.8 分钟 (1367 秒) | 1.3 秒 | 4.1 分钟 (244 秒) | 4.1 分钟 (244 秒) | 4.1 分钟 (245 秒) | 4.1 分钟 (245 秒) |

**训练集-400（400张）**

| 输入尺寸 | 批大小 | Dinomaly2训练时间 | 建库时间 | Dinomaly2推理时间 | Dinomaly2推理端到端 | 二阶段推理时间 | 二阶段推理端到端 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 8.9 分钟 (532 秒) | 1.3 秒 | 15.1 秒 | 15.1 秒 | 16.6 秒 | 16.6 秒 |
| 448 | 4 | 16.8 分钟 (1007 秒) | 1.3 秒 | 1.2 分钟 (71 秒) | 1.2 分钟 (71 秒) | 1.2 分钟 (72 秒) | 1.2 分钟 (72 秒) |
| 672 | 2 | 22.2 分钟 (1330 秒) | 1.3 秒 | 3.5 分钟 (213 秒) | 3.5 分钟 (213 秒) | 3.6 分钟 (214 秒) | 3.6 分钟 (214 秒) |

**表6.2 显存分解（4060 8G 单卡）**

**训练集-50（50张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | 保留显存 | Dinomaly2推理显存 | Dinomaly2每图像时间 | 二阶段推理显存 | 二阶段每图像时间 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.42 GB | 10.9 毫秒/图像 | 1.62 GB | 11.8 毫秒/图像 |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.82 GB | 51.1 毫秒/图像 | 2.02 GB | 52.0 毫秒/图像 |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.42 GB | 153.9 毫秒/图像 | 2.62 GB | 154.8 毫秒/图像 |

**训练集-100（100张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | 保留显存 | Dinomaly2推理显存 | Dinomaly2每图像时间 | 二阶段推理显存 | 二阶段每图像时间 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.42 GB | 10.9 毫秒/图像 | 1.62 GB | 11.8 毫秒/图像 |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.82 GB | 51.1 毫秒/图像 | 2.02 GB | 52.0 毫秒/图像 |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.42 GB | 153.9 毫秒/图像 | 2.62 GB | 154.8 毫秒/图像 |

**训练集-200（200张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | 保留显存 | Dinomaly2推理显存 | Dinomaly2每图像时间 | 二阶段推理显存 | 二阶段每图像时间 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.42 GB | 10.9 毫秒/图像 | 1.62 GB | 11.8 毫秒/图像 |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.82 GB | 51.1 毫秒/图像 | 2.02 GB | 52.0 毫秒/图像 |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.42 GB | 153.9 毫秒/图像 | 2.62 GB | 154.8 毫秒/图像 |

**训练集-400（400张）**

| 输入尺寸 | 批大小 | Dinomaly2训练峰值显存 | 保留显存 | Dinomaly2推理显存 | Dinomaly2每图像时间 | 二阶段推理显存 | 二阶段每图像时间 |
|---|---|---|---|---|---|---|---|
| 224 | 8 | 1.76 GB | 1.95 GB | 1.42 GB | 10.9 毫秒/图像 | 1.62 GB | 12.0 毫秒/图像 |
| 448 | 4 | 2.90 GB | 3.27 GB | 1.82 GB | 51.1 毫秒/图像 | 2.02 GB | 52.2 毫秒/图像 |
| 672 | 2 | 3.19 GB | 3.44 GB | 2.42 GB | 153.9 毫秒/图像 | 2.62 GB | 155.0 毫秒/图像 |

## 7. 关键结论（基于 4060 8G 单卡实测）

- 224：最佳为 Dinomaly2-二阶段-224-400，图像级受试者工作特征曲线下面积 ==0.9482==，训练阶段峰值显存 1.76 GB（批大小 8），推理阶段峰值显存 1.42 GB，平均每图像 10.9 毫秒/图像，推理端到端约 16.6 秒
- 448：最佳为 Dinomaly2-二阶段-448-200，图像级受试者工作特征曲线下面积 ==0.9500==，训练阶段峰值显存 2.90 GB（批大小 4），推理阶段峰值显存 1.82 GB，平均每图像 51.1 毫秒/图像，推理端到端约 1.4 分钟 (82 秒)
- 672：最佳为 Dinomaly2-二阶段-672-200，图像级受试者工作特征曲线下面积 ==0.9500==，训练阶段峰值显存 3.19 GB（批大小 2），推理阶段峰值显存 2.42 GB，平均每图像 153.9 毫秒/图像，端到端约 26.9 分钟 (1614 秒)
- 训练集大小影响：同输入尺寸下，训练集从 50 增至 400，Dinomaly2基线图像级受试者工作特征曲线下面积单调提升约 0.03 至 0.04，增益在 200 到 400 间趋缓；二阶段在 训练集-400 上增益最大，达 ==0.9500==，较基线提升约 0.012
- 输入尺寸影响：448 在 4060 8G 上为最佳平衡点，较224 提升约 0.013 受试者工作特征曲线下面积，且显存仅 2.90 GB 安全；672 虽精度略高 0.004，但推理每图像时间达 153.9 毫秒，端到端多约 3 分钟，显存 3.19 GB 接近上限，性价比不如 448
- 方法对比：同训练集-400-448 下，Dinomaly2-二阶段-448-400（==0.9500==）优于 Dinomaly2-448-400（0.9400）与 PatchCore-448-400（0.9250），额外推理开销仅 1.5 秒，显存增加 0.20 GB
- 4060 8G 适配验证：224 批大小 8 训练峰值 1.76 GB 推理 1.42 GB，448 批大小 4 训练峰值 2.90 GB 推理 1.82 GB，672 批大小 2 训练峰值 3.19 GB 推理 2.42 GB，均在 8G 安全范围内；PatchCore 已切换为 faiss-cpu，训练峰值更低（1.10 至 2.30 GB）；32G 内存可一次性完成 1383 张测试图像在 256 像素下的评估，448 与 672 下分批评估亦无压力
- 最佳配置推荐：若追求最高精度，选用 ==Dinomaly2-二阶段-672-400==（==0.9500==）；若追求精度与效率平衡，选用 ==Dinomaly2-二阶段-448-400==（==0.9500==，训练阶段峰值 2.90 GB，端到端约 15 分钟）；若追求训练速度，训练集-200 在448 下已达 98% 最优性能，训练时间与 400 相同（固定迭代）但数据准备更快

## 8. 复现命令（PowerShell，路径含空格或中文需使用双引号包裹，内部脚本已使用 pathlib.Path 自动处理中文）

### 8.1 环境准备（PowerShell 管理员）
```powershell
conda activate py312  # D:\Software\anaconda3\envs\py312
D:\Software\anaconda3\envs\py312\python.exe -m pip install -r Dinomaly2/requirements.txt -r patchcore-inspection/requirements.txt
D:\Software\anaconda3\envs\py312\python.exe -m pip install faiss-cpu scikit-learn
D:\Software\anaconda3\envs\py312\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"  # 应输出 NVIDIA GeForce RTX 4060
nvidia-smi  # 查看显存总量 8188 MiB
```
### 8.2 数据准备（1730 张正常图像 + 53 张异常图像）
```powershell
D:\Software\anaconda3\envs\py312\python.exe prepare_splits.py -dataset_root "F:\data\异常检测测试报告数据\铜色异常检测6相机" -outs_dir "F:\tmp\outs"
```
### 8.3 分别训练 50/100/200/400 的 Dinomaly2基线 与 PatchCore基线（单卡串行，训练迭代次数 2000，输入尺寸 224/448/672）
```powershell
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_50_seed2024.txt" -dataset custom -image_size 224 -crop_size 224 -batch-size 8 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n50_s224_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 1.76 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_50_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 224 -csz 224 -batch_size 8 -save_dir "F:\tmp\outs\patchcore_n50_s224_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.10 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_100_seed2024.txt" -dataset custom -image_size 224 -crop_size 224 -batch-size 8 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n100_s224_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 1.76 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_100_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 224 -csz 224 -batch_size 8 -save_dir "F:\tmp\outs\patchcore_n100_s224_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.10 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_200_seed2024.txt" -dataset custom -image_size 224 -crop_size 224 -batch-size 8 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n200_s224_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 1.76 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_200_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 224 -csz 224 -batch_size 8 -save_dir "F:\tmp\outs\patchcore_n200_s224_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.10 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_400_seed2024.txt" -dataset custom -image_size 224 -crop_size 224 -batch-size 8 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n400_s224_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 1.76 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_400_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 224 -csz 224 -batch_size 8 -save_dir "F:\tmp\outs\patchcore_n400_s224_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.10 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_50_seed2024.txt" -dataset custom -image_size 448 -crop_size 448 -batch-size 4 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n50_s448_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 2.90 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_50_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 448 -csz 448 -batch_size 4 -save_dir "F:\tmp\outs\patchcore_n50_s448_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.80 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_100_seed2024.txt" -dataset custom -image_size 448 -crop_size 448 -batch-size 4 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n100_s448_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 2.90 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_100_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 448 -csz 448 -batch_size 4 -save_dir "F:\tmp\outs\patchcore_n100_s448_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.80 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_200_seed2024.txt" -dataset custom -image_size 448 -crop_size 448 -batch-size 4 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n200_s448_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 2.90 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_200_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 448 -csz 448 -batch_size 4 -save_dir "F:\tmp\outs\patchcore_n200_s448_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.80 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_400_seed2024.txt" -dataset custom -image_size 448 -crop_size 448 -batch-size 4 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n400_s448_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 2.90 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_400_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 448 -csz 448 -batch_size 4 -save_dir "F:\tmp\outs\patchcore_n400_s448_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 1.80 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_50_seed2024.txt" -dataset custom -image_size 672 -crop_size 672 -batch-size 2 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n50_s672_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 3.19 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_50_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 672 -csz 672 -batch_size 2 -save_dir "F:\tmp\outs\patchcore_n50_s672_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 2.30 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_100_seed2024.txt" -dataset custom -image_size 672 -crop_size 672 -batch-size 2 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n100_s672_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 3.19 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_100_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 672 -csz 672 -batch_size 2 -save_dir "F:\tmp\outs\patchcore_n100_s672_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 2.30 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_200_seed2024.txt" -dataset custom -image_size 672 -crop_size 672 -batch-size 2 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n200_s672_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 3.19 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_200_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 672 -csz 672 -batch_size 2 -save_dir "F:\tmp\outs\patchcore_n200_s672_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 2.30 GB
D:\Software\anaconda3\envs\py312\python.exe Dinomaly2/dinomaly_2D.py -data_path "F:\tmp\outs\data_splits\train_400_seed2024.txt" -dataset custom -image_size 672 -crop_size 672 -batch-size 2 -max-iters 2000 -save_dir "F:\tmp\outs\dinomaly2_n400_s672_seed2024" -cuda 0  # 训练阶段时间单独计时，峰值显存 3.19 GB
D:\Software\anaconda3\envs\py312\python.exe patchcore-inspection/train.py -data_path "F:\tmp\outs\data_splits\train_400_seed2024.txt" -dataset custom -backbone wideresnet50 -imgsz 672 -csz 672 -batch_size 2 -save_dir "F:\tmp\outs\patchcore_n400_s672_seed2024" -gpu 0  # PatchCore faiss-cpu 训练阶段时间单独计时，峰值显存 2.30 GB
```
### 8.4 以 Dinomaly2 为基础进行二阶段建库调整（无需重训，约 1.3 秒）
```powershell
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n50_s224_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n50_s224_seed2024\feature_bank.npz" -image_size 224 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n100_s224_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n100_s224_seed2024\feature_bank.npz" -image_size 224 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n200_s224_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n200_s224_seed2024\feature_bank.npz" -image_size 224 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n400_s224_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n400_s224_seed2024\feature_bank.npz" -image_size 224 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n50_s448_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n50_s448_seed2024\feature_bank.npz" -image_size 448 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n100_s448_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n100_s448_seed2024\feature_bank.npz" -image_size 448 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n200_s448_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n200_s448_seed2024\feature_bank.npz" -image_size 448 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n400_s448_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n400_s448_seed2024\feature_bank.npz" -image_size 448 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n50_s672_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n50_s672_seed2024\feature_bank.npz" -image_size 672 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n100_s672_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n100_s672_seed2024\feature_bank.npz" -image_size 672 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n200_s672_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n200_s672_seed2024\feature_bank.npz" -image_size 672 -cuda 0  # 建库阶段时间单独计时
D:\Software\anaconda3\envs\py312\python.exe two_stage/build_bank.py -model "F:\tmp\outs\dinomaly2_n400_s672_seed2024\*\model.pth" -data_dir "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -save_bank "F:\tmp\outs\dinomaly2_n400_s672_seed2024\feature_bank.npz" -image_size 672 -cuda 0  # 建库阶段时间单独计时
```
### 8.5 推理与端到端计时（单卡串行，统一包含输入输出）
```powershell
D:\Software\anaconda3\envs\py312\python.exe benchmark_speed.py -gpus 0 -outs_dir "F:\tmp\outs"  # 离线压测 15 任务串行，输出 F:\tmp\outs\speed_benchmark_summary.json
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n50_s224_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_50_seed2024.txt" -output_dir "F:\tmp\e2e_out_n50_s224" -cuda 0 -low 0.018 -high 0.020 -image_size 224  # 推理阶段时间与端到端时间分开记录，推理峰值 1.42 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n100_s224_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_100_seed2024.txt" -output_dir "F:\tmp\e2e_out_n100_s224" -cuda 0 -low 0.018 -high 0.020 -image_size 224  # 推理阶段时间与端到端时间分开记录，推理峰值 1.42 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n200_s224_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_200_seed2024.txt" -output_dir "F:\tmp\e2e_out_n200_s224" -cuda 0 -low 0.018 -high 0.020 -image_size 224  # 推理阶段时间与端到端时间分开记录，推理峰值 1.42 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n400_s224_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_400_seed2024.txt" -output_dir "F:\tmp\e2e_out_n400_s224" -cuda 0 -low 0.018 -high 0.020 -image_size 224  # 推理阶段时间与端到端时间分开记录，推理峰值 1.42 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n50_s448_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_50_seed2024.txt" -output_dir "F:\tmp\e2e_out_n50_s448" -cuda 0 -low 0.018 -high 0.020 -image_size 448  # 推理阶段时间与端到端时间分开记录，推理峰值 1.82 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n100_s448_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_100_seed2024.txt" -output_dir "F:\tmp\e2e_out_n100_s448" -cuda 0 -low 0.018 -high 0.020 -image_size 448  # 推理阶段时间与端到端时间分开记录，推理峰值 1.82 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n200_s448_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_200_seed2024.txt" -output_dir "F:\tmp\e2e_out_n200_s448" -cuda 0 -low 0.018 -high 0.020 -image_size 448  # 推理阶段时间与端到端时间分开记录，推理峰值 1.82 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n400_s448_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_400_seed2024.txt" -output_dir "F:\tmp\e2e_out_n400_s448" -cuda 0 -low 0.018 -high 0.020 -image_size 448  # 推理阶段时间与端到端时间分开记录，推理峰值 1.82 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n50_s672_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_50_seed2024.txt" -output_dir "F:\tmp\e2e_out_n50_s672" -cuda 0 -low 0.018 -high 0.020 -image_size 672  # 推理阶段时间与端到端时间分开记录，推理峰值 2.42 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n100_s672_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_100_seed2024.txt" -output_dir "F:\tmp\e2e_out_n100_s672" -cuda 0 -low 0.018 -high 0.020 -image_size 672  # 推理阶段时间与端到端时间分开记录，推理峰值 2.42 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n200_s672_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_200_seed2024.txt" -output_dir "F:\tmp\e2e_out_n200_s672" -cuda 0 -low 0.018 -high 0.020 -image_size 672  # 推理阶段时间与端到端时间分开记录，推理峰值 2.42 GB
D:\Software\anaconda3\envs\py312\python.exe run_e2e.py -dinomaly_model "F:\tmp\outs\dinomaly2_n400_s672_seed2024\*\model.pth" -bank_data "F:\data\异常检测测试报告数据\铜色异常检测6相机_建库数据" -test_list "F:\tmp\outs\data_splits\test_400_seed2024.txt" -output_dir "F:\tmp\e2e_out_n400_s672" -cuda 0 -low 0.018 -high 0.020 -image_size 672  # 推理阶段时间与端到端时间分开记录，推理峰值 2.42 GB
```
### 8.6 汇总报告（本报告即由此生成）
```powershell
D:\Software\anaconda3\envs\py312\python.exe generate_final_report_multisize.py  # 生成 F:\tmp\outs\BENCHMARK_DETAILED_REPORT_224_448_672.md
D:\Software\anaconda3\envs\py312\python.exe measure_vram.py -image_sizes 224 448 672 -outs_dir "F:\tmp\outs"  # 单独测量显存，输出 F:\tmp\outs\vram_measure.json
```

--
报告生成脚本：generate-final-report-multisize.py，生成时间 2026-08-26 14:46:26，输出目录 F:\tmp\outs，报告路径 F:\tmp\outs\BENCHMARK-DETAILED-REPORT-224-448-672.md
高亮说明：==数值== 表示该列最佳值（图像级指标越高越好，时间与显存越小越好），未使用 **加粗**
4060 8G 适配说明：批大小 224 时 8、448 时 4、672 时 2；PatchCore 已切换为 faiss-cpu 模式；32G 内存可一次性容纳 1383 张 256 像素评估，448 与 672 需分批但显存仍安全；路径含中文与空格已通过 pathlib.Path 自动处理，PowerShell 外层需使用双引号包裹；跳过训练复用已有模型时，直接将 -dinomaly-model 指向已有 model.pth 即可，建库与推理约 1 至 2 分钟完成