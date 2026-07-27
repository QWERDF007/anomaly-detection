# Dinomaly2 + Triplet Constraint 工业异常检测最终方案

## 1. 目标与兼容性原则

在保留 Dinomaly2 原始网络结构、异常图算法、checkpoint 键和推理接口的前提下，引入 BG/normal、good/weak_ok、anomaly 三值区域和 Triplet 式排序约束。

标注和训练约束以 Mask 区域为单位，不以整张图的类别为单位。同一张图可以同时存在 BG/normal、`good` 和 `anomaly` 区域。其中 `good` 表示业务允许的 Weak OK，完全正常区域由值为 0 的 BG 表示。

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

Mask 是三值语义标注：

```text
0                 BG，完全正常区域（normal）
GOOD_VALUE        good，业务允许的 Weak OK 区域
ANOMALY_VALUE     anomaly，必须检出的异常区域
```

`GOOD_VALUE` 和 `ANOMALY_VALUE` 由数据配置明确指定，不在方案中猜测其数值。数据加载时必须校验 Mask 只包含 `0`、`GOOD_VALUE`、`ANOMALY_VALUE` 三种值。

进入损失计算前，将 Mask 转换为三个互斥的 boolean mask：

```text
M_n    mask == 0                 # BG / normal
M_g    mask == GOOD_VALUE        # good / weak_ok
M_a    mask == ANOMALY_VALUE     # anomaly
```

同一张图允许以下任意组合：

```text
没有 Mask（整张图为 BG/normal）
Mask 中只有 BG
Mask 中同时存在 BG 和 good
Mask 中同时存在 BG 和 anomaly
Mask 中同时存在 BG、good、anomaly
```

完全正常图像不需要 Mask。若样本没有 Mask，则定义整张有效图像为 BG/normal：

$$
M_n=\mathbf{1},\qquad M_g=\mathbf{0},\qquad M_a=\mathbf{0}
$$

若样本包含 Mask，则直接按三种像素值构造区域：

$$
M_n=[M=0]
$$

$$
M_g=[M=GOOD\_VALUE]
$$

$$
M_a=[M=ANOMALY\_VALUE]
$$

三种区域必须互斥并覆盖整张 Mask；BG 的值固定为 0。为了避免 anomaly 边缘插值污染 normal/good 监督，在映射到特征图后，可从 $M_n$ 和 $M_g$ 中排除膨胀后的 anomaly 边界，但不修改原始 Mask 和评估 GT。

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

`Train/good` 是默认正常目录；`Train/` 下所有名称不等于 `good` 的子目录都是新流程可读取的训练目录。非 `good` 目录名称只用于组织数据，不代替三值 Mask 标签，区域语义仍由 Mask 中的 0、`GOOD_VALUE`、`ANOMALY_VALUE` 决定。

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

启用新的三值 Mask 训练流程，递归读取 `Train/` 下 `good` 和所有非 `good` 子目录，从标准 Dinomaly2 初始化开始，在一次训练中联合完成所有约束：

```text
标准 Dinomaly2 初始化
        ↓
每个训练样本：图像 + 可选 Mask
无 Mask：只计算默认 Dinomaly2 损失
有 Mask：计算默认、good、anomaly 三个损失
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

三个损失都调用同一种 Dinomaly2 loss。区别仅在于参与损失计算的区域。

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

---

## 7. 三个损失

### 7.1 Dinomaly2 默认损失

全图使用值为 1 的 Mask，完全等价于默认 Dinomaly2：

$$
L_{dinomaly}=L_D(e,d;\mathbf{1})=L_{default}(e,d)
$$

该损失对每个训练样本都计算。有 Mask 时也保持默认全图计算方式。

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
```

一个 batch 可以同时包含有 Mask 和无 Mask 的图像。`L_dinomaly` 对整个 batch 计算；`L_good` 和 `L_anomaly` 只使用有 Mask 图像中的对应区域。

---

## 9. Mask 映射规则

Mask 只用于计算 $L_{good}$ 和 $L_{anomaly}$，不改变 $L_{dinomaly}$ 的默认全图定义。

- Mask 缺失：整图视为 BG/normal，但只计算 $L_{dinomaly}$。
- Mask 存在：解析值 0、`GOOD_VALUE`、`ANOMALY_VALUE`。
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

缺少 good 或 anomaly 区域时，只将对应损失项置为 0，不切换训练模式，也不启动额外训练阶段。

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
| $L_{dinomaly}$ | 每张图，定义与默认 Dinomaly2 完全相同 |
| $L_{good}$ | 有 Mask 且存在 good 区域时；最小化 |
| $L_{anomaly}$ | 有 Mask 且存在 anomaly 区域时；最大化 |

$\lambda_g$ 和 $\lambda_a$ 是新训练流程的显式参数。方案不额外引入其他 loss，也不强制使用损失预热或分阶段权重。

---

## 13. Batch 采样

Batch 以图像为单位加载，再从每张图的 Mask 中提取区域，不按互斥的图像类别固定分组。

- 无 Mask 图像提供全图 BG/normal 区域。
- 有 Mask 图像可同时提供 BG、good 和 anomaly 区域。
- 若某一类区域缺失，只跳过依赖该类的损失项，其他损失继续计算。
- 图像增强必须同步作用于 image 和三值 Mask；Mask 只使用 nearest interpolation。
- Batch size、shuffle 和样本组织由用户配置，不强制固定类别比例。

---

## 14. 有 Mask 数据的训练流程

有 Mask 时只进行一次训练，不设置预热、适配或稳定微调阶段：

```text
iterations = 与用户配置的 max_iters 一致
冻结 = Encoder（保持默认 Dinomaly2 行为）
训练 = Bottleneck、Decoder
loss = L_dinomaly + lambda_good * L_good - lambda_anomaly * L_anomaly
```

优化器、参数组、学习率调度、梯度裁剪、训练迭代数和 checkpoint 周期均沿用默认 Dinomaly2 配置。唯一变化是数据加载器可返回 Mask，并按第 12 节组合三个损失。

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

完全保留 Dinomaly2 默认评估函数、异常图、后处理和指标，不增加三值区域专用指标：

```text
I-AUROC
I-AP
I-F1
P-AUROC
P-AP
P-F1
P-AUPRO
```

用于默认像素指标时，将三值训练 Mask 转成二值异常 GT：

```text
ANOMALY_VALUE    → 1
0 / GOOD_VALUE   → 0
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
├── mask_constraint_losses.py            # good/anomaly 两个区域损失
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
```

`--train_mode default` 执行原始 Dinomaly2 训练；`--train_mode mask_constraint` 读取 `Train/` 下全部目录，并在一次训练中联合优化三个损失。训练和推理都不增加额外 Head，只保存与原 Dinomaly2 完全兼容的 `model.pth`。

该方案支持同一张图同时存在 BG/normal、good/weak_ok 和 anomaly 区域，也支持完全正常图像不提供 Mask，同时保持原始网络结构、像素级异常定位方式、checkpoint 格式、推理接口和推理速度。
