# 672 两阶段工业异常检测系统深度优化与前沿探索计划 (plan.md)

## 📌 1. 项目总目标与核心约束
- **基础架构**：严格保持 **Dinomaly2 粗筛 + 特征库二阶段调整** 系统架构；
- **输入分辨率**：统一基于 **672 分辨率** (`image_size=672, crop_size=672`)；
- **数据源约束**：严格仅使用 `/data/wt/ramdisk/leishi_026/`（良品训练与评测数据）与 `/data/wt/ramdisk/test_export_03/`（标注建库数据）；
- **优化核心指标**：
  - **最大化降低误报区域数 (`R-FP-RegionCount`)**：从原始 ~23,000 处断崖式压降至数千乃至千处以内；
  - **最大化提升定位与分类精度**：提升 `P-AP`, `P-AUPRO`, `P-AUROC`, `I-AUROC`, `I-AP`, `I-F1`；
  - **维持极低漏检率**：缺陷漏检率维持在 `R-MissRate <= 0.3%`（真实缺陷召回率 $\ge 99.7\%$）。

---

## 📋 2. 方案探索计划表与状态跟踪 (Progress Tracker)

| 阶段 / 任务编号 | 方案名称 | 核心算法与探索内容 | 对应独立实验脚本 | 显卡分配 | 当前状态 | 核心成果指标 |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Stage 0** | **深度收敛基础模型训练** | 15,000 iter 深度充分收敛 `vit_base_14` (672) | `/data/wt/trainlogs/.../vit_base_672_15k/` | GPU 7 | ✅ 已完成 | I-AUROC: 0.9228, I-AP: 0.9899 |
| **Track 1** | **极小极大子集采样 (Coreset)** | GPU 加速 K-Center Coreset (50% 向量提纯) | `experiments/01_coreset_feature_bank.py` | GPU 6 | ✅ 已完成 | I-AUROC: 0.9203, FP Count: 21,108 |
| **Track 2** | **密集三层多尺度特征金字塔** | Layer 6+8+11 级联 ($2304$ 维) + 100% Patch 覆盖 | `experiments/02_dense_multi_layer.py` | GPU 0 | ✅ 已完成 | MissRate: 0.21%, PixelCov: 98.44% |
| **Track 3** | **三重特征库决策系统** | 引入高频反光边缘困难负样本库 (Hard-Negative Bank) | `experiments/03_hard_negative_three_bank.py` | GPU 2, 3 | ✅ 已完成 | I-AUROC: 0.9208, I-AP: 0.9897 |
| **Track 4** | **空间坐标感知先验加权** | $D_{\text{spatial}} = D_{\text{feat}} + \lambda \|(u_p,v_p)-(u_q,v_q)\|^2$ | `experiments/06_spatial_prior_knn.py` | GPU 0, 1 | ✅ 已完成 | I-AUROC: 0.9309, I-AP: 0.9909 |
| **Track 5** | **非对称置信度标定** | Sigmoid 温度缩放 ($\tau_g=0.2, \tau_a=0.05$) 软门限 | `experiments/07_asymmetric_margin_scaling.py` | GPU 2, 3 | ✅ 已完成 | P-AUPRO: 0.9411, I-AUROC: 0.9186 |
| **Track 6** | **多尺度引导滤波与保边锐化** | RGB 多通道导向图 ($r=2, \epsilon=10^{-4}$) 边缘对齐 | `experiments/08_multiscale_guided_bilateral.py` | GPU 4, 5 | ✅ 已完成 | P-AP: 0.4965, 误报直降 -52% |
| **Track 7** | **自适应背景基底扣除与开运算** | 背景百分位数自适应扣除 ($p=20\%$) + 开运算 ($k=3$) | `experiments/04_adaptive_background_morphology.py` | GPU 6, 7 | ✅ 已完成 | P-AUPRO: 0.9446, FP: 14,657 (-38%) |
| **Track 8** | **跨模型乘性共识集成** | 15k Base 与 5k Small 几何平均乘性消除非相关底噪 | `experiments/05_multi_model_consensus.py` | GPU 6, 7 | ✅ 已完成 | 误报直降至 11,674 处 (-51%) |
| **Track A** | **连通域拓扑几何极值滤波** | 多维面积/紧密度 (Solidity)/离心率多重拓扑去噪 | `experiments/10_topological_morphological_denoising.py` | GPU 0 | ✅ 已完成 | 面积与紧密度有效滤除孤立噪点 |
| **Track B** | **动态双阈值自适应切割** | 局部统计量动态切割与连续软调制函数 (tanh) | `experiments/11_adaptive_threshold_soft_modulation.py` | GPU 1 | ✅ 已完成 | 消除硬截断伪连通域 |
| **Track C** | **三重库决策权重矩阵搜索** | 良品/异常/困难负样本最优距离加权矩阵搜索 (234组) | `experiments/12_triple_bank_optimal_weighting.py` | GPU 2 | ✅ 已完成 | `12_triple_bank_optimal_weighting_results.csv` |
| **Track E** | **多距离度量与流形对齐** | 余弦相似度、马氏距离协方差缩放 vs L2 (95组) | `experiments/14_metric_manifold_alignment.py` | GPU 3 | ✅ 已完成 | `14_metric_manifold_alignment_results.csv` |
| **Track F** | **局部上下文空间聚合** | $3 \times 3$ 空间邻域池化与多尺度感受野拼接 | `experiments/15_patch_context_expansion.py` | GPU 4 | ✅ 已完成 | 增强微细瑕疵空间一致性 |
| **Track G** | **GMM 对数似然比打分** | 基于高斯混合模型对数似然比替代简单 KNN 距离 (44组) | `experiments/16_gmm_kde_likelihood_scoring.py` | GPU 5 | ✅ 已完成 | PCA(64)+GMM(4) I-AUROC: 0.9242, P-AUPRO: 0.9433 |
| **Track H** | **边缘曲率自适应加权衰减** | 工件倒角与高频反光边缘曲率 Sobel 衰减 (111组) | `experiments/17_edge_saliency_attenuation.py` | GPU 6 | ✅ 已完成 | `17_edge_saliency_attenuation_results.csv` |
| **Track D** | **终极超参全局网格搜索** | Hard Trigger + Good Supp + BG Floor + Opening + Area (48组) | `experiments/13_deep_fusion_grid_search.py` | GPU 7 | ✅ 已完成 | P-AUPRO 达 0.9466 (历史最高) |
| **Final** | **终极方案提炼与总结** | 提炼最优架构与完整指标对比，撰写 FINAL.md | `FINAL.md` | 全局 | ✅ 已完成 | 交付完整技术方案与复现指南 |

---

## 🔄 3. 流水线执行与持续推进机制
1. **多卡无缝流水**：每个任务完成后，立即释放 GPU 资源并衔接后续排队方案；
2. **严格记录纪律**：每个方案产出后，将量化指标严格记录至 [`WORKLOG`](file:///data/wt/anomaly-detection/WORKLOG)；
3. **安全提交规范**：**未经用户明确指令，绝不执行自动 git commit**；
4. **终极交付**：所有方案评测完毕后，全面汇总并生成 [`FINAL.md`](file:///data/wt/anomaly-detection/FINAL.md)。
