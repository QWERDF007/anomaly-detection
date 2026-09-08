# Dinomaly2 零目标数据分布漂移鲁棒化方案

> 适用约束：训练阶段只有单一源域正常图像；部署前没有目标域有标签或无标签图像；默认推理阶段不更新模型、不累计目标域统计量。

---

## 1. 目标、边界与核心结论

### 1.1 目标

在不依赖目标域数据的条件下，降低以下正常变化造成的误报：

- 光照、亮度、对比度、Gamma、白平衡变化；
- 轻度噪声、失焦、压缩、锐化和分辨率变化；
- 小范围平移、旋转、缩放、透视变化；
- 平滑反光、渐变阴影、暗角；
- 不改变产品结构的轻微纹理或成像变化。

同时满足：

- 小划痕、针孔、小缺口、溢胶、局部白边等真实缺陷不能被不变性模块抹除；
- 原始干净分布上的检测能力基本不下降；
- 默认部署仍为单次前向推理，不引入目标域训练和 Sinkhorn 推理模块。

### 1.2 能力边界

单源正常数据无法自动区分“从未出现过的正常变化”和“从未出现过的缺陷”。因此本方案必须显式定义：

| 集合 | 含义 | 示例 |
|---|---|---|
| 正常扰动集合 `T_normal` | 应当被模型忽略的变化 | 全局亮度、平滑反光、小角度位姿变化 |
| 异常扰动集合 `T_anomaly` | 必须继续保持敏感的局部变化 | 裂纹、缺口、局部白边、异物、结构缺失 |
| 未定义区域 | 无法仅凭源域数据判断 | 与真实缺陷外观高度相似的新工艺变化 |

如果某种局部白边在业务上既可能是 OK 反光，也可能是 NG 溢胶，且源数据与规则都不能区分，则不能只靠损失函数保证正确分类。

### 1.3 核心结论

停止将以下损失作为主要方向：

```text
Local UOT(Encoder Feature, Decoder Feature)
```

改为：

```text
源域正常图 x
   ├── Anchor View: x
   └── Shift View: T_normal(x)
             ↓
冻结 DINO Encoder
             ↓
轻量 Normality Canonicalization Adapter
             ↓
Dinomaly2 Decoder + Loose Reconstruction
             ↓
双视图一致性 + 原分布锚定 + 异常保留约束
```

Local UOT 只作为跨视图残余局部错位的可选损失，不用于 Encoder—Decoder 重建匹配。

---

## 2. 旧方案问题与保留项

### 2.1 根本问题

旧方案假设输入发生正常位移后，Encoder 与 Decoder 特征会形成纯 token 重排：

```text
Encoder: A B C D
Decoder: X A B C
```

实际两者处理同一张输入，输入位移会同时影响 Encoder 特征和 Decoder 的空间坐标。误报的主要原因通常是：新域特征超出 Decoder 学到的源域正常流形，而不是两组特征仅发生了排列变化。

因此 `UOT(F, F_hat)` 主要降低重建约束，并不会产生关于未来正常域的监督。Dinomaly2 本身已有 Loose Reconstruction，再叠加 UOT 的新增信息有限。

### 2.2 保留内容

- Local Window、Unbalanced OT、Spatial Cost 的实现；
- Normal Shift Test；
- Shift Normal FPR、Small Defect Recall；
- 正常分数与异常分数必须分开分析；
- Sinkhorn 使用 FP32、log-domain、独立日志；
- 推理阶段优先保持不变。

### 2.3 删除或降级内容

- 删除 `UOT(F, F_hat)` 作为主训练损失；
- 暂停搜索 `lambda_ot / rho / window_size`；
- 暂不开发 Encoder—Decoder OT Anomaly Map；
- 不将“整体更宽松”视为成功；
- 不使用需要目标域样本或目标域统计量的 RoDA 训练流程。

---

## 3. 总体架构

### 3.1 模块组成

| 模块 | 是否训练 | 作用 |
|---|---:|---|
| DINOv2/DINOv3 Encoder | 冻结 | 提取通用特征，避免源域小数据破坏预训练表示 |
| Canonicalization Adapter | 训练 | 将正常成像扰动映射到稳定的正常特征空间 |
| Dinomaly2 Bottleneck/Decoder | 训练或低学习率微调 | 重建标准化后的正常特征 |
| Normal Shift Generator | 无参数或可选对抗参数 | 从源图生成正常伪域 |
| Hard Shift Miner | 无独立推理开销 | 选择当前模型最容易误报的正常视图 |
| Pseudo Anomaly Generator | 仅训练 | 防止 Adapter 和 Decoder 抹除局部缺陷 |
| Cross-view Local UOT | 可选 | 处理无法被已知逆变换完全对齐的局部正常形变 |

### 3.2 Adapter 位置

默认在 Dinomaly2 两个特征组之后加入 Adapter：

```text
Group 1: layer 3~6
Group 2: layer 7~10
```

每组 Adapter：

$$
C_{\phi,k}(g_k)=g_k+\alpha_k W_{2,k}\,\mathrm{GELU}(W_{1,k}\,\mathrm{LN}(g_k))
$$

建议：

- 瓶颈比例：`1/4` 或 `1/8`；
- `W2` 零初始化；
- 可学习缩放 `alpha` 初始为 0；
- Adapter 输入输出维度相同；
- 不修改冻结 DINO 参数。

零初始化保证初始模型严格接近原始 Dinomaly2，便于定位性能变化来源。

### 3.3 推理数据流

```text
Input
  ↓
Frozen DINO
  ↓
Feature Grouping
  ↓
Canonicalization Adapter
  ↓
Dinomaly2 Bottleneck + Decoder
  ↓
Token-wise Cosine Anomaly Map
```

默认不包含：目标域缓存、在线训练、目标 Batch、Sinkhorn、目标域阈值重标定。

---

## 4. 正常伪域生成

### 4.1 设计原则

正常增强必须满足：

1. 不增加或删除产品结构；
2. 不产生尖锐、局部、高频的缺陷形状；
3. 几何变换参数可记录，便于特征逆变换对齐；
4. 强度上限由最小目标缺陷尺寸约束；
5. 训练增强族与验收增强族不能完全相同。

### 4.2 初始正常增强池

以下范围是第一轮起点，不是最终固定值：

| 类别 | 操作 | 初始范围 | 约束 |
|---|---|---:|---|
| 光度 | Brightness | `0.75~1.25` | 全局或低频变化 |
| 光度 | Contrast | `0.75~1.25` | 避免饱和裁剪 |
| 光度 | Gamma | `0.80~1.25` | 保留缺陷可见性 |
| 颜色 | RGB Gain | `0.85~1.15/通道` | 模拟白平衡、色温 |
| 颜色 | Hue | `±0.03` | 纹理颜色不是缺陷定义时使用 |
| 颜色 | Saturation | `0.80~1.20` | 颜色缺陷类应缩小范围 |
| 噪声 | Gaussian/Poisson | 归一化 `σ≤0.02` | 不覆盖最小缺陷 |
| 成像 | Gaussian Blur | `σ≤1.2 px` | 同时受最小缺陷宽度约束 |
| 成像 | JPEG | `quality 60~100` | 不生成明显块状伪缺陷 |
| 成像 | Sharpen | 轻度 | 不产生强 halo 白边 |
| 照明 | Vignetting | `±20%` | 低频平滑蒙版 |
| 照明 | Gradient Shadow | `±20%` | 禁止硬边界 |
| 照明 | Specular Field | 平滑椭圆/曲面 | 禁止模拟成局部异物 |
| 几何 | Translation | 初始 `±0.5 patch` | 最大不超过已知装夹容差 |
| 几何 | Rotation | 初始 `±2°` | 记录逆变换矩阵 |
| 几何 | Scale | `0.97~1.03` | 记录逆变换矩阵 |
| 几何 | Perspective | 轻度 | 不改变可见零件拓扑 |

模糊上限建议满足：

$$
\sigma_{blur} < \frac{d_{min}}{4}
$$

其中 `d_min` 是必须检出的最小缺陷宽度。若最小缺陷只有 2~3 px，不应加入会抹掉该缺陷的模糊作为正常增强。

### 4.3 特征级风格随机化

只在浅层组或 Adapter 输入处使用：

- Channel mean/std 随机化；
- MixStyle；
- Fourier amplitude 低频扰动；
- 平滑低秩风格偏移。

限制：

- 不扰动 token 空间拓扑；
- 不在深层语义组使用大幅随机化；
- 每次最多启用一种特征风格增强；
- 必须单独做 Small Defect Recall 消融。

### 4.4 禁止作为正常增强的操作

- CutPaste、局部 Copy-Paste；
- Perlin 局部遮挡；
- 局部锐利白边；
- 随机擦除零件；
- Patch Shuffle；
- 会改变孔、槽、边界、螺纹数量的形变；
- 强 Motion Blur、强失焦、严重过曝；
- 与业务真实缺陷外观相同的局部增强。

---

## 5. 双视图训练

### 5.1 输入构造

每张源域正常图生成：

```text
x_anchor = x
x_shift  = T_normal(x)
```

记录几何变换 `W_T`。纯光度增强时 `W_T` 为恒等变换。

### 5.2 原始重建损失

Anchor 与 Shift View 均执行 Dinomaly2 Loose Reconstruction：

$$
\mathcal L_{rec}
=
\mathcal L_{loose}(x_{anchor})
+
\lambda_{shift}\mathcal L_{loose}(x_{shift})
$$

初始：

```yaml
lambda_shift: 1.0
```

作用：明确告诉 Decoder，`T_normal` 生成的样本仍属于正常分布。

### 5.3 Canonical Consistency Loss

对增强特征做逆变换后，与 Anchor 特征比较：

$$
\mathcal L_{inv}
=
\frac{1}{|\mathcal G|}
\sum_{k\in\mathcal G}
d_{cos}
\left(
C_{\phi,k}(g_k(x)),
W_T^{-1}C_{\phi,k}(g_k(T(x)))
\right)
$$

规则：

- 光度、颜色、噪声：同位置比较；
- 平移、旋转、缩放：先按记录参数逆变换；
- 只在双方均有效的 token mask 内计算；
- 边界填充 token 不参与损失；
- 默认 Point-wise Cosine，不先使用 OT。

### 5.4 Source Identity Anchor

防止 Adapter 改写干净源域特征：

$$
\mathcal L_{id}
=
\frac{1}{|\mathcal G|}
\sum_k
d_{cos}(C_{\phi,k}(g_k(x)),g_k(x))
$$

如果 Clean 指标下降，优先增大 `lambda_id`，而不是降低整个方案的训练轮数。

### 5.5 Map Equivariance Loss

对最终异常图做几何一致性约束：

$$
\mathcal L_{map}
=
\left\|
A(x)-W_T^{-1}A(T(x))
\right\|_1
$$

仅对正常样本使用，目标不是把所有分数压成零，而是要求同一物理位置对正常扰动响应一致。

---

## 6. 异常信息保护

### 6.1 目的

仅优化正常不变性容易产生：

```text
正常变化分数下降
真实缺陷分数也下降
```

因此需要约束 Adapter 不得消除局部结构变化。

### 6.2 伪异常生成

伪异常只用于边界保护，不作为主要异常分布模型。建议包含：

- 小面积纹理替换；
- 细划痕、针孔、小缺口；
- 局部边缘增生或缺失；
- 局部颜色变化；
- 小面积结构遮挡。

必须保存伪异常 mask `M_a`，面积分布覆盖业务最小缺陷尺寸。

### 6.3 Adapter Preservation Loss

要求伪异常区域在 Canonical Space 中仍与原正常区域保持差异：

$$
\mathcal L_{preserve}
=
\max\left(
0,
m_f-
d_{mask}
\left(
C_\phi(E(T_a(x))),
C_\phi(E(x));M_a
\right)
\right)
$$

### 6.4 Anomaly Score Margin

可选地要求伪异常区域的分数高于原正常区域：

$$
\mathcal L_{margin}
=
\max\left(
0,
m_s+
\operatorname{mean}_{M_a}A(x)
-
\operatorname{mean}_{M_a}A(T_a(x))
\right)
$$

该损失在主方案稳定后再启用，避免伪异常质量较差时主导训练。

---

## 7. Hard Shift Mining

### 7.1 策略

每张源图随机生成 `K` 个正常视图：

$$
\{T_1(x),T_2(x),...,T_K(x)\}
$$

用当前模型计算正常误报分数，选择 Top-K hard views：

$$
T^*=\arg\max_{T_i\in\mathcal T_{normal}}s_{anom}(T_i(x))
$$

初始配置：

```yaml
hard_shift:
  candidates: 4
  select_topk: 1
  start_iter_ratio: 0.20
```

### 7.2 防止增强族塌缩

- 每个 Batch 至少覆盖两个增强大类；
- 对连续入选次数过多的增强族降采样；
- 几何、光度、成像、照明分别统计 hard ratio；
- 记录每种增强的误报分数和入选率；
- 不允许增强器通过制造局部缺陷来提高分数。

### 7.3 可选对抗域随机化

Hard Mining 有效后，可引入受约束的可学习增强器：

$$
\max_T
s_{anom}(T(x))
-\mu_1 d_{structure}(T(x),x)
-\mu_2 d_{local}(T(x),x)
$$

只允许其控制全局颜色、低频照明、平滑相机响应和小范围几何参数。第一阶段不开发该模块。

---

## 8. Cross-view Local UOT：可选模块

### 8.1 使用条件

只有同时满足以下条件才启用：

- 双视图 Point-wise Consistency 已有收益；
- 已知几何逆变换后仍存在局部形变残差；
- 残余误报主要集中在结构边缘；
- Small Defect Recall 有足够安全裕量。

### 8.2 正确作用对象

$$
\mathcal L_{uot-view}
=
\operatorname{LocalUOT}
\left(
C_\phi(E(x)),
W_T^{-1}C_\phi(E(T(x)))
\right)
$$

不再使用：

$$
\operatorname{UOT}(F,\hat F)
$$

### 8.3 初始限制

```yaml
cross_view_uot:
  enabled: false
  lambda: 0.01
  window_size: 3
  stride: 3
  epsilon: 0.05
  rho: 1.0
  position_weight: 0.10
  iterations: 20
```

与旧方案相比：

- 更小的窗口；
- 更强的空间代价；
- 更接近 Balanced OT；
- 更低的损失权重；
- 只处理跨正常视图残余错位。

---

## 9. 总损失与训练调度

### 9.1 总损失

$$
\begin{aligned}
\mathcal L_{total}={}&
\mathcal L_{rec}
+\lambda_{inv}\mathcal L_{inv}
+\lambda_{id}\mathcal L_{id}\\
&+\lambda_{map}\mathcal L_{map}
+\lambda_{preserve}\mathcal L_{preserve}
+\lambda_{margin}\mathcal L_{margin}\\
&+\lambda_{uot}\mathcal L_{uot-view}
\end{aligned}
$$

### 9.2 第一轮建议权重

```yaml
loss:
  lambda_shift: 1.0
  lambda_inv: 0.10
  lambda_id: 0.05
  lambda_map: 0.05
  lambda_preserve: 0.05
  lambda_margin: 0.00
  lambda_uot: 0.00
```

权重搜索采用单变量消融，不做全排列。

### 9.3 训练阶段

| 阶段 | 迭代占比 | 训练内容 |
|---|---:|---|
| S0 | 基线 | 复现原 Dinomaly2，保存固定 checkpoint |
| S1 | `0~10%` | Adapter Identity + Anchor Reconstruction；不启用强增强 |
| S2 | `10~20%` | 加入 Shift Reconstruction 和 `L_inv` |
| S3 | `20~60%` | 启用 Hard Shift Mining 和 `L_map` |
| S4 | `60~100%` | 加入 `L_preserve`；确认稳定后才启用 `L_margin` |
| S5 | 独立实验 | 只有满足第 8.1 节条件时测试 Cross-view UOT |

### 9.4 参数更新

默认：

- DINO Encoder：冻结；
- Adapter：正常学习率；
- Bottleneck/Decoder：Adapter 学习率的 `0.25~0.5`；
- Sinkhorn：FP32、log-domain；
- AMP：除 Sinkhorn 外可使用 BF16/FP16。

如果从原始 Dinomaly2 checkpoint 微调，建议起点：

```yaml
optimizer:
  adapter_lr: 2.0e-4
  decoder_lr: 5.0e-5
  weight_decay: 1.0e-4
```

最终学习率应根据原始 Batch Size 线性缩放。

---

## 10. 实验协议

### 10.1 数据划分

源域正常数据：

```text
80% 训练
20% Clean Normal Validation
```

测试数据分为：

1. Clean OK；
2. Seen Shift OK：增强机制相同、参数区间不同；
3. Unseen Shift OK：训练未使用的增强机制；
4. Composed Shift OK：两到三种轻度变化组合；
5. Clean NG；
6. Shifted NG；
7. Small Defect 子集；
8. Structure Defect 子集。

### 10.2 Seen/Unseen 隔离示例

| 训练使用 | Unseen 验收使用 |
|---|---|
| Brightness、Gamma | White Balance、Tone Curve |
| Gaussian Noise | Poisson、Speckle Noise |
| Gaussian Blur | Defocus、轻 Motion Blur |
| Translation、Rotation | Scale、Perspective |
| Gradient Shadow | Vignetting、平滑 Specular Field |

第二轮交换两组，排除增强族偶然偏置。

### 10.3 公平性要求

- 所有模型使用相同训练/测试划分；
- 相同输入分辨率、Encoder、训练迭代数；
- 至少 3 个随机种子；
- 不用 NG 测试集调超参数；
- 超参数只依据 Clean Validation 和模拟 Shift Validation；
- 生产阈值只能由源域 Clean Normal Validation 标定；
- Baseline 与改造版都重新按同一规则标定阈值。

---

## 11. 指标定义

### 11.1 主指标

#### Shift-FPR@NG-TPR95

固定真实 NG 召回率为 95%，统计 Shifted OK 的误报率：

$$
\mathrm{ShiftFPR@TPR95}
=
P(s(x_{shift-ok})>\tau_{TPR95})
$$

这是主验收指标。

#### Production Shift FPR

阈值仅由 Clean Normal Validation 标定，例如目标 Clean FPR=1%，然后直接应用于所有 Shift OK：

$$
\tau_{prod}=Q_{0.99}(s(X_{clean-val}))
$$

### 11.2 辅助指标

- Clean Good FPR；
- Worst-Shift FPR；
- Mean Shift FPR；
- Small Defect Recall；
- Shifted NG Recall；
- I-AUROC、I-AP；
- P-AUROC、P-AP、P-AUPRO；
- 每类 Shift 的 OK 分数 P50/P95/P99；
- NG 分数 P05/P50；
- Clean/Shift/NG 分数分布重叠率。

### 11.3 可靠性统计

- 报告 3 个随机种子的均值和标准差；
- 对主指标执行 bootstrap 95% 置信区间；
- 分类别报告，不只报告所有类别平均；
- 单独报告最差类别和最差 Shift 类型。

---

## 12. 实验矩阵

| 编号 | 正常增强训练 | Adapter | 双视图一致性 | Hard Mining | 异常保护 | Cross-view UOT |
|---|---:|---:|---:|---:|---:|---:|
| A |  |  |  |  |  |  |
| B | ✓ |  |  |  |  |  |
| C | ✓ | ✓ | ✓ |  |  |  |
| D | ✓ | ✓ | ✓ | ✓ |  |  |
| E | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

说明：

- A：原始 Dinomaly2；
- B：验证“把伪域作为正常样本训练”是否已经有效；
- C：验证 Adapter 和显式一致性；
- D：验证困难增强挖掘；
- E：验证异常保护是否恢复小缺陷召回；
- F：只有 D/E 已通过验收且存在局部错位残差时测试。

额外对照：

- GNL/ADShift 风格的增强不变性基线；
- FiCo 风格补偿/过滤基线；
- 如果允许使用源域正常模板，可增加 CostFilter-AD 类参考匹配基线。

---

## 13. 阶段验收标准

以下为建议起始门槛，最终应根据漏检与误报成本调整。

### 13.1 Baseline 验收

进入改造前必须满足：

- 3 个随机种子 Clean 指标标准差不超过 `0.3` 个百分点；
- 原始论文/仓库配置可复现到合理误差范围；
- Clean OK、Shift OK、NG 分数分布和示例图已保存；
- 所有后续实验严格复用相同评价代码。

### 13.2 MVP 验收：B/C/D

必须同时满足：

| 指标 | Go 条件 |
|---|---:|
| Mean Seen-Shift FPR | 相对 Baseline 下降 `≥25%` |
| Mean Unseen-Shift FPR | 相对 Baseline 下降 `≥10%` |
| Worst-Shift FPR | 不允许比 Baseline 增加 `>2` 个百分点 |
| Clean Good FPR | 增加 `≤1` 个百分点 |
| Clean I-AUROC | 下降 `≤0.3` 个百分点 |
| P-AUROC/P-AUPRO | 下降 `≤0.5` 个百分点 |
| Small Defect Recall | 下降 `≤2` 个百分点 |

如果只改善 Seen Shift、Unseen Shift 无改善，判定为增强记忆，不进入部署候选。

### 13.3 最终算法验收：E 或 F

必须同时满足：

| 指标 | 最终 Go 条件 |
|---|---:|
| Mean Shift-FPR@TPR95 | 相对 Baseline 下降 `≥30%` |
| Mean Unseen-Shift FPR | 相对 Baseline 下降 `≥20%` |
| Production Shift FPR | 绝对下降 `≥3` 个百分点，或相对下降 `≥30%` |
| Worst-Shift FPR | 不高于 Baseline，容许统计波动 `≤1` 个百分点 |
| Shifted NG Recall | 下降 `≤1` 个百分点 |
| Small Defect Recall | 下降 `≤1` 个百分点 |
| Clean I-AUROC | 下降 `≤0.2` 个百分点 |
| P-AUPRO | 下降 `≤0.3` 个百分点 |
| 统计可靠性 | 主指标改善的 bootstrap 95% CI 下界 `>0` |

如果业务以零漏检优先，可将 Small Defect Recall 的允许下降改为 0，并降低对 FPR 改善幅度的要求。

### 13.4 性能验收

| 项目 | 要求 |
|---|---:|
| 推理延迟 | 相对 Baseline 增加 `≤5%` |
| 推理显存 | 增加 `≤10%` |
| 训练时间 | 增加 `≤60%` |
| TensorRT 改造 | 仅增加 Adapter 的 Linear/LN/GELU，禁止 Sinkhorn 进入默认图 |
| 数值一致性 | PyTorch 与 TensorRT Anomaly Map 最大绝对误差满足项目既有标准 |

---

## 14. 失败模式与决策规则

| 现象 | 结论 | 处理 |
|---|---|---|
| B 对 Seen Shift 都无提升 | 正常增强没有覆盖模型敏感因素 | 检查增强强度、DINO 层和分数来源，不进入 UOT |
| Seen 改善、Unseen 不改善 | 记忆增强类型 | 扩展低频风格随机化和 Hard Mining，增加跨族验证 |
| Shift OK 与 NG 分数同步下降 | 模型整体变宽松 | 降低 `lambda_shift/inv`，提高 `lambda_id/preserve` |
| Clean 指标下降 | Adapter 改写源域特征 | 检查零初始化，提高 `lambda_id`，降低 Decoder LR |
| 小缺陷召回下降 | 正常增强或一致性覆盖缺陷线索 | 收紧 Blur/Color 范围，提高异常保护，移除冲突增强 |
| 边缘位移误报仍高 | 几何等变性不足 | 检查逆变换与 token 坐标；最后才测试 Cross-view UOT |
| F 相比 E 无显著提升 | UOT 没有新增信息 | 删除 UOT，保持静态简单部署 |
| 某种真实变化无法模拟且像缺陷 | 单源数据不可辨识 | 需要业务规则、成像约束或未来生产反馈，不继续调损失掩盖 |

### 14.1 明确停止条件

满足任一条件即停止继续堆叠模块：

- C/D 在 3 个随机种子上均未改善 Unseen Shift FPR；
- F 相比 E 的主指标提升小于 `5%` 相对值；
- F 导致 Small Defect Recall 下降超过 `1` 个百分点；
- F 的计算代价超过其 FPR 收益；
- 正常增强与真实缺陷定义发生不可消除的语义冲突。

---

## 15. 推荐配置起点

```yaml
source_dg:
  encoder_frozen: true

  adapter:
    enabled: true
    position: group_output
    bottleneck_ratio: 0.25
    zero_init: true
    learnable_scale_init: 0.0

  normal_shift:
    candidates_per_image: 4
    select_topk: 1
    hard_mining_start_ratio: 0.20
    max_ops_per_view: 3
    geometry_probability: 0.30
    photometric_probability: 0.80
    degradation_probability: 0.40
    illumination_probability: 0.40

  loss:
    lambda_shift: 1.0
    lambda_inv: 0.10
    lambda_id: 0.05
    lambda_map: 0.05
    lambda_preserve: 0.05
    lambda_margin: 0.00
    lambda_uot: 0.00

  pseudo_anomaly:
    enabled: true
    start_iter_ratio: 0.60
    min_area_ratio: 0.0001
    max_area_ratio: 0.02

  cross_view_uot:
    enabled: false
    window_size: 3
    stride: 3
    epsilon: 0.05
    rho: 1.0
    position_weight: 0.10
    iterations: 20
```

---

## 16. 代码结构

```text
Dinomaly2/
├── models/
│   └── domain_adapter.py
├── augment/
│   ├── normal_shift.py
│   ├── hard_shift_mining.py
│   └── pseudo_anomaly.py
├── losses/
│   ├── source_dg_loss.py
│   ├── consistency_loss.py
│   ├── anomaly_preserve_loss.py
│   └── cross_view_uot.py
├── evaluation/
│   ├── shift_protocol.py
│   ├── score_distribution.py
│   └── acceptance_report.py
├── configs/
│   └── source_dg.yaml
└── train_source_dg.py
```

日志至少包含：

```text
loss_total
loss_anchor_rec
loss_shift_rec
loss_inv
loss_id
loss_map
loss_preserve
loss_margin
adapter_residual_norm
hard_shift_family
hard_shift_score
clean_ok_score_p95
shift_ok_score_p95
pseudo_anomaly_score_mean
```

---

## 17. 开发顺序与交付物

### P0：评价协议

交付：

- Clean/Seen/Unseen/Composed Shift 数据生成器；
- Shift-FPR@TPR95；
- Small Defect 子集；
- 三随机种子 Baseline 报告；
- 分类别、分 Shift 的分数分布。

验收：第 13.1 节。

### P1：正常增强训练

交付：实验 B。

目的：确认不增加新模块时，训练增强是否能降低误报。

### P2：Adapter + 双视图一致性

交付：实验 C。

目的：建立明确的正常域标准化空间。

### P3：Hard Shift Mining

交付：实验 D。

目的：提高对困难和组合变化的覆盖。

### P4：异常保护

交付：实验 E。

目的：恢复或保护小缺陷、局部边缘缺陷召回。

### P5：可选 Cross-view UOT

交付：实验 F。

前置条件：E 已通过最终主要指标，且剩余误报可确认来自逆变换后的局部错位。

### P6：部署验证

交付：

- Adapter TensorRT 网络；
- PyTorch/TensorRT 数值一致性；
- 延迟、显存、吞吐报告；
- 固定源域阈值下的最终验收报告。

---

## 18. 最终推荐

主线：

```text
Dinomaly2
+ 源域正常伪域生成
+ 零初始化 Canonicalization Adapter
+ 双视图特征一致性
+ Hard Shift Mining
+ 局部异常信息保护
```

可选项：

```text
Cross-view Local Spatial UOT
```

明确不采用：

```text
目标域适配
目标域 Memory Bank
目标域统计对齐
UOT(Encoder, Decoder) 作为主损失
默认推理 Sinkhorn
```

最终是否成功不以“总体 AUROC 有小幅提升”为标准，而以以下组合为准：

```text
Unseen Shift FPR 明显下降
+ Worst-Shift 不恶化
+ Small Defect Recall 基本不下降
+ Clean 性能基本持平
+ 静态推理成本可接受
```

---

## 19. 参考方向

- [Dinomaly2: One Dinomaly2 Detect Them All](https://arxiv.org/abs/2510.17611)
- [Anomaly Detection under Distribution Shift / GNL](https://arxiv.org/abs/2303.13845)
- [FiCo: Filter or Compensate](https://arxiv.org/abs/2412.10115)
- [MMR: Industrial Anomaly Detection with Domain Shift](https://arxiv.org/abs/2304.02216)
- [CostFilter-AD](https://arxiv.org/abs/2505.01476)
- [RoDA](https://arxiv.org/abs/2503.14910)：需要目标域适配数据，不属于本方案约束下的主方法。
