# Dinomaly2 + Triplet Constraint 工业异常检测最终方案

## 1. 目标与兼容性原则

在保留 Dinomaly2 原始网络结构、异常图算法、checkpoint 键和推理接口的前提下，引入 BG/normal、good/weak_ok、anomaly 和待定/忽略四种 Mask 状态，以及 Triplet 式排序约束。待定状态只用于屏蔽不确定标注，不产生新的损失。

标注和训练约束以 Mask 区域为单位，不以整张图的类别为单位。同一张图可以同时存在 BG/normal、`good`、`anomaly` 和待定区域。其中 `good` 表示业务允许的 Weak OK，完全正常区域由值为 0 的 BG 表示。待定区域表示无法可靠判断为 BG、good 或 anomaly 的区域。

最终部署 checkpoint 仍为原始 `Dinomaly.state_dict()`：

```text
model.pth    结构和 state-dict 键与默认 Dinomaly2 完全一致
```

现有 `predict.py` 和 `dinomaly_roi_pipeline.py` 可直接加载 `model.pth`，不增加推理分支、参数量和推理耗时。

必须满足：

```text
model.pth.keys() == Dinomaly(...).state_dict().keys()
```

---

## 2. 数据定义

每个训练样本只包含图像和可选的 mask，不使用互斥的图像级 `sample_type`：

```text
image
mask（可选）
```

Mask 是四值语义标注：

```text
0                 BG，完全正常区域（normal）
GOOD_VALUE        good，业务允许的 Weak OK 区域
ANOMALY_VALUE     anomaly，必须检出的异常区域
IGNORE_VALUE      待定区域，不参与任何训练损失
```

`GOOD_VALUE`、`ANOMALY_VALUE` 和 `IGNORE_VALUE` 由参数明确指定，默认值分别为 `1`、`2`、`255`，且必须互不相同并都不等于 0。数据加载时必须校验 Mask 只包含这四种值。

进入损失计算前，将 Mask 转换为三个监督区域 Mask 和一个有效区域 Mask：

```text
M_n    mask == 0                 # BG / normal
M_g    mask == GOOD_VALUE        # good / weak_ok
M_a    mask == ANOMALY_VALUE     # anomaly
M_v    mask != IGNORE_VALUE      # 参与默认 Dinomaly2 损失的有效区域
M_{ga} (M=GOOD_VALUE) or (M=ANOMALY_VALUE)
                  # 启用 --mask_only 时的有效区域
```

同一张图允许以下任意组合：

```text
没有 Mask（整张图为 BG/normal）
Mask 中只有 BG
Mask 中同时存在 BG 和 good
Mask 中同时存在 BG 和 anomaly
Mask 中同时存在 BG、good、anomaly
Mask 中同时存在 BG、good、anomaly 和待定
```

完全正常图像不需要 Mask。若样本没有 Mask，则定义整张有效图像为 BG/normal：

$$
M_n=\mathbf{1},\qquad M_g=\mathbf{0},\qquad M_a=\mathbf{0}
$$

此时 $M_i=\mathbf{0}$、$M_v=\mathbf{1}$。

若样本包含 Mask，则直接按各像素值构造区域：

$$
M_n=[M=0]
$$

$$
M_g=[M=GOOD\_VALUE]
$$

$$
M_a=[M=ANOMALY\_VALUE]
$$

待定区域为：

$$
M_i=[M=IGNORE_VALUE]
$$

有效区域为：

$$
M_v=[M\ne IGNORE_VALUE]
$$

四种值互斥并覆盖整张 Mask；BG 的值固定为 0。待定区域不转换为 BG，也不参与 $L_{dinomaly}$、$L_{good}$ 或 $L_{anomaly}$。为了避免 anomaly 边缘插值污染 normal/good 监督，在映射到特征图后，可从 $M_n$ 和 $M_g$ 中排除膨胀后的 anomaly 边界，但不修改原始 Mask 和评估 GT。

训练集、验证集和测试集如何划分由用户自行组织；方案和训练代码不对工件、生产批次、采集日期或 ROI 的跨集合分布做强制检查和限制。

新训练流程的数据目录支持：

```text
dataset/
└── Train/
    ├── good/
    ├── bad/
    ├── anomaly_type_a/
    └── xxx/
```

`Train/good` 是默认正常目录；`Train/` 下所有名称不等于 `good` 的子目录都是新流程可读取的训练目录。非 `good` 目录名称只用于组织数据，不代替四值 Mask 标签，区域语义仍由 Mask 中的 0、`GOOD_VALUE`、`ANOMALY_VALUE` 和 `IGNORE_VALUE` 决定。

---

## 3. 网络结构

Dinomaly2 主体保持不变：

```text
Image
  ↓
Frozen DINO Encoder
  ↓
Multi-layer Encoder Features e
  ↓
Bottleneck + Decoder
  ↓
Reconstructed Features d
  ↓
Cosine Anomaly Map
```

Encoder 全程冻结，仅允许更新 Bottleneck 和 Decoder。

训练和推理都使用 Dinomaly2 原有 Encoder、Bottleneck 和 Decoder，不增加 Projection Head 或 Pixel Head。

---

## 4. 训练模式参数

使用显式参数选择旧训练流程或新训练流程，不根据目录或 Mask 自动切换：

```text
--train_mode default
--train_mode mask_constraint
```

默认值为 `default`，以保持现有 Dinomaly2 行为。

### 4.1 `--train_mode default`

完全执行默认 Dinomaly2 训练，只读取 `Train/good`，忽略 `Train/` 下其他目录，不读取 Mask，也不计算 $L_{good}$ 或 $L_{anomaly}$：

$$
L=L_{dinomaly}=L_{default}(e,d)
$$

模型初始化、优化器、学习率调度、训练迭代数和 checkpoint 保存方式均沿用默认实现。

### 4.2 `--train_mode mask_constraint`

启用新的四值 Mask 训练流程，递归读取 `Train/` 下 `good` 和所有非 `good` 子目录，从标准 Dinomaly2 初始化开始，在一次训练中联合完成所有约束。待定值只表示忽略区域，不增加新的损失：

若需要让有 Mask 样本只在 good/anomaly 区域计算损失，增加：

```bash
--mask_only
```

```text
标准 Dinomaly2 初始化
        ↓
每个训练样本：图像 + 可选 Mask
无 Mask：只计算默认 Dinomaly2 损失
有 Mask：计算默认、good、anomaly 三个损失；待定区域从三者中排除
        ↓
默认 Dinomaly2 损失 + good 区域损失 + anomaly 区域损失
        ↓
model.pth
```

不加载预先训练的基础 checkpoint，不创建 Teacher，也不进行第二阶段微调。训练从第一个 iteration 到最后一个 iteration 都属于同一次运行。

现有 Dinomaly2 Encoder 的冻结策略保持不变；Bottleneck 和 Decoder 在同一次反向传播中联合优化。

---

## 5. 原始异常图

继续使用 Dinomaly2 的 cosine distance：

$$
A_l=1-\cos(e_l,d_l)
$$

多层平均：

$$
A=\frac{1}{L}\sum_l A_l
$$

训练使用未平滑的特征图分数；推理继续使用现有插值、Gaussian smoothing、图像分数和 ROI 过滤流程。

---

## 6. 统一的 Dinomaly2 区域损失

三个损失都调用同一种 Dinomaly2 loss。区别仅在于参与损失计算的区域。待定区域不会进入任何一个损失。

定义 Mask $M$ 下的 Dinomaly2 loss：

$$
L_D(e,d;M)=L_{default}(e\odot M,d\odot M)
$$

其中 $L_{default}$ 是当前配置实际使用的 Dinomaly2 loss：

```text
global_cosine
或
global_cosine_hm_percent（启用原有 --ll 时）
```

Mask 使用 nearest interpolation 映射到每层特征尺寸，并广播到通道维。区域外特征乘 0，区域内特征保持不变。Encoder 特征的 detach 行为、层融合方式和多层平均方式完全沿用默认 Dinomaly2。

以 `global_cosine` 为例：

$$
L_D(e,d;M)=\frac{1}{L}\sum_l
\left[1-\cos\left(
\operatorname{vec}(e_l\odot M_l),
\operatorname{vec}(d_l\odot M_l)
\right)\right]
$$

区域 Mask 为空时，不计算该区域损失并返回 0。

当一个有 Mask 的样本包含待定值时，默认损失使用有效区域 Mask $M_v$：

$$
L_{dinomaly}=L_D(e,d;M_v),\qquad M_v=[M\ne IGNORE_VALUE]
$$

没有 Mask 的样本使用全 1 Mask，仍然计算完整的默认 Dinomaly2 损失。若一个样本全部为待定区域，则该样本对三个损失均不产生梯度。

---

## 7. 三个损失

### 7.1 Dinomaly2 默认损失

没有待定值时，全图使用值为 1 的 Mask，完全等价于默认 Dinomaly2：

$$
L_{dinomaly}=L_D(e,d;\mathbf{1})=L_{default}(e,d)
$$

该损失对每个训练样本都计算。包含待定值的有 Mask 样本只计算 $M_v$ 有效区域；没有待定值时保持默认全图计算方式。若启用 `--mask_only`，有 Mask 样本改为只计算 $M_{ga}$，BG 和待定区域均不参与。

### 7.2 good 区域损失

只保留 Mask 中 `GOOD_VALUE` 对应的区域，并使用完全相同的 Dinomaly2 loss：

$$
L_{good}=L_D(e,d;M_g)
$$

训练时最小化 $L_{good}$，使 good/weak_ok 区域保持较小的 Encoder/Decoder 重建差异。当前 batch 没有 good 区域时，$L_{good}=0$。

### 7.3 anomaly 区域损失

只保留 Mask 中 `ANOMALY_VALUE` 对应的区域，并使用完全相同的 Dinomaly2 loss：

$$
L_{anomaly}=L_D(e,d;M_a)
$$

训练时最大化 $L_{anomaly}$，使 anomaly 区域的 Encoder/Decoder 重建差异增大。因此该分量在最小化的总目标中使用负号。当前 batch 没有 anomaly 区域时，$L_{anomaly}=0$。

---

## 8. 损失启用规则

每个样本根据 Mask 是否存在决定损失：

```text
无 Mask：L_sample = L_dinomaly

有 Mask：L_sample = L_dinomaly
                   + lambda_good * L_good
                   - lambda_anomaly * L_anomaly

有待定区域的 Mask：先从该样本中去除 `IGNORE_VALUE` 区域，再按上式计算；待定区域不参与三个损失。
```

一个 batch 可以同时包含有 Mask 和无 Mask 的图像。无待定值且未启用 BG 忽略时，`L_dinomaly` 保持默认 Dinomaly2 的整图计算；存在待定值时，包含待定值的样本使用有效区域 Mask。启用 `--mask_only` 后，有 Mask 样本的 `L_dinomaly` 只使用 good/anomaly 区域。`L_good` 和 `L_anomaly` 只使用有 Mask 图像中的对应区域。

---

## 9. Mask 映射规则

Mask 用于计算三个损失。没有待定值且未启用 BG 忽略时不改变
$L_{dinomaly}$ 的默认全图定义；包含待定值时，从 $L_{dinomaly}$ 中排除
`IGNORE_VALUE` 区域。启用 `--mask_only` 后，有 Mask 样本的
$L_{dinomaly}$ 只使用 `GOOD_VALUE` 和 `ANOMALY_VALUE` 区域。

- Mask 缺失：整图视为 BG/normal，但只计算 $L_{dinomaly}$。
- Mask 存在：解析值 0、`GOOD_VALUE`、`ANOMALY_VALUE` 和 `IGNORE_VALUE`。
- `IGNORE_VALUE` 只表示待定/忽略，不转换为 BG、good 或 anomaly。
- Mask 缩放到 anomaly map 或特征图尺寸时只使用 nearest interpolation。
- 不对原始 Mask 做膨胀、平滑或重新赋值。
- 不额外计算 BCE、Dice、Pixel Loss 或蒸馏损失。

---

## 10. Loss 数量约束

训练代码中只允许出现以下三个顶层 loss：

```text
loss_dinomaly
loss_good
loss_anomaly
```

不再添加独立的 Triplet Loss、Pixel Loss、Background Loss、Distillation Loss 或分类损失。三个分量都采用默认 Dinomaly2 loss；good 项最小化，anomaly 项最大化。

---

## 11. 单次联合反向传播

有 Mask 的训练流程不使用 Teacher 或蒸馏。每个 iteration 只执行一次 Student 前向，将当前 batch 中存在的损失项加权求和后执行一次反向传播：

```text
images + optional masks
        ↓
一次 Dinomaly 前向
        ↓
计算当前 batch 可用的全部损失
        ↓
一次 zero_grad / backward / clip_grad_norm / optimizer.step
```

缺少 good 或 anomaly 区域时，只将对应损失项置为 0；待定区域从三个损失的有效 Mask 中排除。不切换训练模式，也不启动额外训练阶段。

---

## 12. 总损失

$$
L=
L_{dinomaly}
+\lambda_gL_{good}
-\lambda_aL_{anomaly}
$$

| 损失 | 作用范围 |
|---|---|
| $L_{dinomaly}$ | 每张图；默认包含 BG/good/anomaly、排除待定；启用 `--mask_only` 后，有 Mask 样本只包含 good/anomaly |
| $L_{good}$ | 有 Mask 且存在 good 区域时；最小化 |
| $L_{anomaly}$ | 有 Mask 且存在 anomaly 区域时；最大化 |

$\lambda_g$ 和 $\lambda_a$ 是新训练流程的显式参数。方案不额外引入其他 loss，也不强制使用损失预热或分阶段权重。

---

## 13. Batch 采样

Batch 以图像为单位加载，再从每张图的 Mask 中提取区域，不按互斥的图像类别固定分组。

- 无 Mask 图像提供全图 BG/normal 区域。
- 有 Mask 图像可同时提供 BG、good 和 anomaly 区域。
- 启用 `--mask_only` 后，有 Mask 图像的 BG 区域不提供任何损失监督；无 Mask 图像仍提供全图默认损失。
- 待定区域不提供任何损失监督；如果整张图都是待定，该图在当前 iteration 中不产生有效梯度。
- 若某一类区域缺失，只跳过依赖该类的损失项，其他损失继续计算。
- 图像增强必须同步作用于 image 和四值 Mask；Mask 只使用 nearest interpolation。
- Batch size、shuffle 和样本组织由用户配置，不强制固定类别比例。

---

## 14. 有 Mask 数据的训练流程

有 Mask 时只进行一次训练，不设置预热、适配或稳定微调阶段：

```text
iterations = 与用户配置的 max_iters 一致
冻结 = Encoder（保持默认 Dinomaly2 行为）
训练 = Bottleneck、Decoder
loss = L_dinomaly + lambda_good * L_good - lambda_anomaly * L_anomaly
       （待定区域不参与三个分量；启用 --mask_only 时，BG 也不参与）
```

优化器、参数组、学习率调度、梯度裁剪、训练迭代数和 checkpoint 周期均沿用默认 Dinomaly2 配置。唯一变化是数据加载器可返回 Mask，并按第 12 节组合三个损失；`IGNORE_VALUE` 区域始终从三个损失中排除，`--mask_only` 决定是否同时排除 BG。

---

## 15. Checkpoint 设计

```text
checkpoints/
└── model.pth
```

只保存 `model.pth`。该文件只包含原始 `Dinomaly.state_dict()`，因此现有推理代码仍可使用 `strict=True` 加载。

---

## 16. 推理

最终部署流程：

```text
Image
  ↓
model.pth
  ↓
Encoder / Decoder cosine distance
  ↓
原始 anomaly map
  ↓
原始 Gaussian smoothing
  ↓
原始 image score / ROI filtering
```

最终异常图：

$$
A=A_{rec}
$$

不使用额外 Metric Head，不融合第二张异常图，不改变现有阈值和后处理接口。因此推理结构、参数量、速度、`strict=True` 加载方式和 ROI pipeline 接口均保持不变。

---

## 17. 验证指标

完全保留 Dinomaly2 默认评估函数、异常图、后处理和指标，不增加四值区域专用指标。训练 Mask 中的 `IGNORE_VALUE` 只影响训练损失，不自动改写 `Test/ground_truth`：

```text
I-AUROC
I-AP
I-F1
P-AUROC
P-AP
P-F1
P-AUPRO
```

如果用户明确使用训练 Mask 作为额外 GT，`ANOMALY_VALUE` 可转为异常标签，`BG` 和 `GOOD_VALUE` 可转为正常标签；`IGNORE_VALUE` 必须在评估中排除，不能当作正常或异常标签。默认评估仍使用 `Test/ground_truth`，不读取训练 Mask：

```text
ANOMALY_VALUE    → 1
0 / GOOD_VALUE   → 0
IGNORE_VALUE     → evaluation ignore
```

图像中只要存在 `ANOMALY_VALUE`，默认图像级 label 为 1，否则为 0。无 Mask 图像的像素 GT 全 0、图像级 label 为 0。除此以外不修改 Dinomaly2 默认验证逻辑。

---

## 18. 验收条件

有 Mask 训练输出的 `model.pth` 使用第 17 节默认 Dinomaly2 指标验收，并满足：

1. checkpoint 与原模型 state-dict 键完全一致。
2. 现有 `predict.py` 使用 `strict=True` 加载成功。
3. 现有 ROI pipeline 无需修改即可推理。
4. 最终指标门槛由用户根据任务设置，方案不增加强制阈值。

---

## 19. 最终目录规划

```text
Dinomaly2/
├── dinomaly_2D.py                       # 原始训练，不修改
├── models/uad.py                        # 原始模型，不修改
├── dinomaly_2D_mask_constraint.py       # 有 Mask 时的新训练入口
├── mask_constraint_losses.py            # 三个 Dinomaly2 损失和待定区域屏蔽
└── checkpoints/
    └── model.pth
```

---

## 20. 最终结论

最终采用“训练期多约束、推理期零结构变化”的设计：

```text
无 Mask：
    L = L_dinomaly

有 Mask：
    L = L_dinomaly
      + lambda_good * L_good
      - lambda_anomaly * L_anomaly

有待定区域：
    三个损失均排除 IGNORE_VALUE 区域

启用 --mask_only：
    有 Mask 样本的三个损失只使用 GOOD_VALUE 和 ANOMALY_VALUE 区域
```

`--train_mode default` 执行原始 Dinomaly2 训练；`--train_mode mask_constraint` 读取 `Train/` 下全部目录，并在一次训练中联合优化三个损失。训练和推理都不增加额外 Head，只保存与原 Dinomaly2 完全兼容的 `model.pth`。

该方案支持同一张图同时存在 BG/normal、good/weak_ok、anomaly 和待定区域，也支持完全正常图像不提供 Mask。通过 `--mask_only` 可以进一步只使用有 Mask 样本的 good/anomaly 区域，同时保持原始网络结构、像素级异常定位方式、checkpoint 格式、推理接口和推理速度。







Dinomaly2 的特征**余弦距离损失**，不是 BCE、MSE 或 Dice。

  模型先得到：

  - en：冻结的 DINO Encoder 特征
  - de：Bottleneck + Decoder 输出特征

  每个特征层的损失为：

$$
L_l=\frac{1}{B}\sum_{b=1}^{B}
  \left(1-\cos(\operatorname{vec}(en_{b,l}),\operatorname{vec}(de_{b,l}))\right)
$$

  最终损失是所有特征层的平均：

$$
L=\frac{1}{L}\sum_l L_l
$$

  Encoder 特征会 detach，实际只训练 Bottleneck 和 Decoder。实现见 Dinomaly2/utils.py:31 和训练循环 Dinomaly2/
  dinomaly_2D.py:337。

  当前默认参数 --ll 1，实际使用 global_cosine_hm_percent：

  - 前向计算的损失值仍然是上述全局余弦距离；
  - 每个特征点计算局部余弦距离；
  - 前 1000 次迭代将 p 从 0 增加到 0.9；
  - 默认只重点保留距离较大的约 10% 区域梯度；
  - 其余低距离区域的梯度乘以 ll_factor=0.1。

  如果指定 --ll 0，则直接使用普通的 global_cosine。

mask_constraint，总损失为：

$$
L_{\text{total}} = 

  L_{\text{dinomaly}}

  + \lambda_{\text{good}}L_{\text{good}}

  - \lambda_{\text{anomaly}}L_{\text{anomaly}}
$$

  默认 lambda_good=0.5、lambda_anomaly=0.5：

  - L_dinomaly：默认使用 BG/good/anomaly 有效区域，忽略值为 255 的区域；启用 `--mask_only` 后，有 Mask 样本只使用 good/anomaly 区域；
  - L_good：只计算 Mask 值为 1 的 good 区域，并最小化；
  - L_anomaly：只计算 Mask 值为 2 的 anomaly 区域，由于前面的负号，实际最大化异常区域的特征差异。



• 在当前代码中，P-* 是像素级指标。流程是：

  1. 计算 anomaly map；
  2. resize 到 256×256；
  3. 进行 Gaussian smoothing；
  4. 将 GT 转成二值 Mask：非零为异常，零为正常；
  5. 展平所有图像的像素后计算指标。

  实现见 Dinomaly2/utils.py:142。

| 指标    | 计算方式                                                  | GT 无异常但预测异常的影响                                    |
| ------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| P-AUROC | 所有阈值下绘制 TPR-FPR 曲线并计算面积                     | 预测分数高于部分真实异常像素时降低；如果误检分数很低，影响可能较小 |
| P-AP    | 所有阈值下计算 Precision-Recall 面积                      | 误检会增加 FP、降低 Precision，通常会降低 P-AP               |
| P-F1    | 遍历阈值，取最大 F1                                       | 误检降低 Precision；但这是“最佳阈值”下的最大值，低分误检可能被阈值过滤 |
| P-AUPRO | 计算 GT 异常连通区域的覆盖率 PRO，并对 FPR≤0.3 的曲线积分 | 误检区域不增加 PRO，但会增加背景 FPR，从而降低 P-AUPRO       |

具体实现中：

  - P-AUROC、P-AP、P-F1 在 Dinomaly2/utils.py:189 计算；
  - P-F1 是所有阈值中的最大值，见 Dinomaly2/utils.py:125；
  - P-AUPRO 使用 GT 连通区域和背景误检率计算，见 Dinomaly2/utils.py:389。

所以，GT 没有标注的区域会被视为正常区域。该区域被检测为异常，就是像素级 FP，会影响这些指标。预测区域越大、分数越高，影响通常越明显。

需要注意，训练评估中的 P-F1 不是固定 score_threshold 下的 F1，而是自动寻找最佳阈值。因此可能出现：

P-F1 较高，但使用实际部署阈值时仍有较多误检

此外，当前评估会先进行双线性插值和 Gaussian smoothing，异常边界附近的分数可能扩散到 GT 外部，也会被计为误检。若 GT 标注不完整，未标注但实际异常的区域同样会被当作 FP。



如果关心“预测出的异常区域是否真的对应 GT”，应使用区域级误判率，而非仅看像素 FPR。

  计算流程：
  3. 将预测区域与 GT 区域做一对一匹配；通常要求 IoU ≥ 阈值（如 0.1 或 0.3）。
  4. 未匹配到任何 GT 的预测区域记为 FP。

  区域误判率（更准确说是误报占比 / FDR）：

$$
  \text{Region-FDR}=\frac{FP}{TP+FP}=1-\text{Region-Precision}
$$

  同时可输出：

$$
  \text{Region-Recall}=\frac{TP}{TP+FN}
$$

  其中 FN 是未被任何预测区域匹配到的 GT 区域。

  补充：现有 P-AUPRO 中的 FPR 是像素级：

$$
  \text{Pixel-FPR}=\frac{\text{GT 外被预测为异常的像素数}}{\text{全部 GT 外像素数}}
$$

  它衡量背景像素被要判断“预测区域是否真的和 GT 匹配”，应使用区域级误判率，而不仅是像素 FPR。

  流程是：

  1. 用分数阈值把 score map 二值化，得到预测区域。
  2. 对预测和 GT 分别做连通域分析。
  3. 逐个预测区域与 GT 区域计算 IoU。
  4. 当 IoU ≥ 阈值（常用 0.1、0.3 或 0.5）时，视为匹配；建议一对一匹配，避免多个预测区域重复匹配同一个 GT。

  得到：

  - TP：匹配到 GT 的预测区域
  - FP：没有匹配到任何 GT 的预测区域
  - FN：没有被预测区域匹配到的 GT 区域

  区域误判率通常定义为：

$$
  \text{Region FDR} = \frac{FP}{TP + FP}
$$

  也就是“所有预测异常区域中，实际不对应任何 GT 的比例”。对应的区域准确率是：

$$
  \text{Region Precision} = \frac{TP}{TP + FP}
$$

  而当前常见的像素误判率是：

$$
  \text{Pixel FPR} = \frac{\text{预测为异常但 GT 为正常的像素数}}{\text{所有 GT 正常像素数}}
$$

  P-AUPRO 使用的是后者的像素 FPR，不等同于“预测区域是否匹配 GT”。如果你要评估 ROI/连通区域是否误报，建议在新脚本中增加
  R-Precision、R-Recall、R-F1 和 R-FDR。
