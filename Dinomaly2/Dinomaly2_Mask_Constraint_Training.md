# Dinomaly2 Mask Constraint 训练说明

本文只说明 `mask_constraint` 模式的实际训练流程、数据组织、Mask 生成和运行命令。

## 1. 训练入口

使用以下脚本启动训练：

```text
Dinomaly2/dinomaly_2D.py
```

通过参数显式启用新流程：

```text
--train_mode mask_constraint
```

也可以直接运行 `dinomaly_2D_mask_constraint.py`；该脚本在没有指定训练模式时会自动使用 `mask_constraint`。

## 2. 数据组织

最小目录结构如下：

```text
dataset/
├── Train/
│   ├── good/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   └── anomaly/
│       └── image_003.jpg
├── Test/
│   ├── good/
│   │   └── image_101.jpg
│   └── anomaly/
│       └── image_102.jpg
├── ground_truth/
│   └── anomaly/
│       └── image_102.png
└── masks/
    ├── good/
    │   └── image_002.png
    └── anomaly/
        └── image_003.png
```

说明：

- 必须存在 `Train/`，目录名大小写不敏感。
- `Train/` 下的图像会递归读取，目录名不决定区域类别。
- `weak_ok/` 目录不是必需的，可以只使用 `good/`、`anomaly/`，也可以使用其他目录名称。
- 训练样本是否包含 Mask，以及 Mask 中是什么区域，完全由 Mask 文件决定。
- `Test/` 和 `ground_truth/` 用于训练结束后的 Dinomaly2 默认指标评估，不参与三个训练损失。
- 如果没有测试集或测试异常 GT，训练仍可执行，但默认评估可能被跳过。

例如，以下结构也是有效的：

```text
dataset/
└── Train/
    ├── good/
    └── production_batch_02/
```

`production_batch_02` 只是目录名，里面图像的 good/anomaly 区域仍然根据对应 Mask 的像素值判断。

## 3. Mask 语义

默认使用 8 位 PNG Mask，像素值为：

```text
0 = BG / normal
1 = good / weak_ok
2 = anomaly
255 = 待定/忽略
```

同一张图可以同时包含四种值：

```text
0、1、2、255
```

因此不需要通过图像所在目录区分 `good`、`weak_ok` 和 `anomaly`。

训练流程对每个样本的处理方式：

| 样本状态 | 使用的损失 |
|---|---|
| 没有 Mask | 只使用默认 Dinomaly2 损失 |
| 有 Mask，但只有 0 | 默认损失；good/anomaly 区域损失为 0 |
| 有 good 区域 | 额外计算 good 区域损失 |
| 有 anomaly 区域 | 额外计算 anomaly 区域损失 |
| 同时有 good 和 anomaly | 同时计算两个区域损失 |
| 有待定区域 | 待定区域不参与默认、good、anomaly 三个损失 |

Mask 缺失和全 0 Mask 的区别是：缺失 Mask 会被标记为无标注样本；全 0 Mask 是显式标注的正常区域。两者都不会产生 good/anomaly 区域损失。

待定值默认是 `255`，表示无法明确判断为 BG、good 或 anomaly 的区域。待定值不会转换为 BG，也不会参与任何损失。如果整张 Mask 都是待定，该样本在当前 iteration 中不产生有效梯度。

默认情况下，有 Mask 样本的 `L_dinomaly` 使用 BG、good、anomaly 三类有效区域，待定区域除外。如果指定 `--mask_only`，有 Mask 样本的三个损失只使用 good 和 anomaly 区域，BG 和待定区域都不计算损失。没有 Mask 的样本始终使用整图默认损失。

## 4. LabelMe 标注转换

转换脚本为：

```text
Dinomaly2/convert_labelme_to_mask.py
```

LabelMe 标签规则固定为：

```text
标签为 good       → Mask 值 1
其他所有标签      → Mask 值 2
未标注区域         → Mask 值 0
```

不需要配置标签映射。例如 `scratch`、`crack`、`defect`、`weak_ok` 等标签都会被转换为 anomaly；只有 `good` 标签转换为 good。标签比较会忽略首尾空格和大小写。

当前 LabelMe 转换脚本没有把任何标签自动转换为待定值，保持“非 `good` 标签都是 anomaly”的约定。若需要待定区域，可在转换后将对应 Mask 像素设为 `255`，或使用其他能够直接生成四值 Mask 的标注流程。

如果标注 JSON 位于 `annotations/Train/` 下，可以执行：

```bash
python Dinomaly2/convert_labelme_to_mask.py \
    --input_dir /path/to/annotations/Train \
    --output_dir /path/to/dataset/masks \
    --image_root /path/to/dataset \
    --good_value 1 \
    --anomaly_value 2 \
    --overwrite
```

脚本会递归读取 JSON，并保持相对目录结构输出 PNG。例如：

```text
annotations/Train/good/image_002.json
        ↓
masks/good/image_002.png
```

若 JSON 中已经包含有效的 `imageWidth` 和 `imageHeight`，`--image_root` 可以省略。脚本支持 LabelMe 的 polygon、rectangle、circle、line、linestrip 和 point 标注。

## 5. Mask 与图像匹配

默认 Mask 根目录固定为 `--data_path/masks`，因此通常不需要指定 `--mask_dir`：

```text
Train/good/image_002.jpg
masks/good/image_002.png
```

训练程序会按相对路径查找对应 Mask。例如：

```text
--data_path /path/to/dataset
        ↓
/path/to/dataset/masks/good/image_002.png
```

`--mask_dir` 仍然保留，用于 Mask 不在默认目录时覆盖默认位置：

```bash
--mask_dir /another/path/masks
```

也支持以下方式：

```text
Train/good/image_002.jpg
Train/good/image_002_mask.png
```

训练流程只从默认的 `<data_path>/masks/` 读取训练 Mask（允许 `0/1/2/255` 四种值），不会自动读取
`ground_truth/`、`mask/` 或 `annotations/`。其中 `ground_truth/` 仅用于训练结束后的
默认评估；如果训练 Mask 位于其他位置，必须通过 `--mask_dir` 指定。

## 6. 启动训练

在 `Dinomaly2` 目录下执行：

```bash
cd /path/to/anomaly-detection/Dinomaly2

python dinomaly_2D.py \
    --data_path /path/to/dataset \
    --dataset custom \
    --train_mode mask_constraint \
    --save_dir /path/to/trainlogs \
    --backbone dinov2reg_vit_small_14 \
    --image_size 448 \
    --crop_size 392 \
    --max-iters 40000 \
    --cuda 0 \
    --good_value 1 \
    --anomaly_value 2 \
    --ignore_value 255 \
    --mask_only \
    --lambda_good 1.0 \
    --lambda_anomaly 1.0
```

常用参数：

| 参数 | 作用 | 默认值 |
|---|---|---:|
| `--data_path` | 数据集根目录，内部必须有 `Train/` | `../MPDD` |
| `--train_mode` | 训练模式 | `default` |
| `--mask_dir` | 覆盖 Mask 根目录 | `<data_path>/masks` |
| `--backbone` | DINO Encoder | `dinov2reg_vit_small_14` |
| `--max-iters` | 最大训练 iteration 数 | `40000` |
| `--image_size` | 图像缩放尺寸 | `448` |
| `--crop_size` | 中心裁剪尺寸 | `392` |
| `--cuda` | GPU 编号 | `0` |
| `--eval_interval` | 评估周期，单位为 epoch；`-1` 表示只在最终评估 | `50` |
| `--good_value` | good 的 Mask 值 | `1` |
| `--anomaly_value` | anomaly 的 Mask 值 | `2` |
| `--ignore_value` | 待定/忽略的 Mask 值；该区域不参与三个损失 | `255` |
| `--mask_only` | 有 Mask 时只计算 good/anomaly 区域，忽略 BG | 关闭 |
| `--lambda_good` | good 损失权重 | `1.0` |
| `--lambda_anomaly` | anomaly 损失权重 | `1.0` |

如果 Mask 使用的值就是 `0/1/2`，不需要额外指定 `--good_value` 和 `--anomaly_value`。

不启用 BG 忽略时，使用默认行为；需要只训练标注为 good/anomaly 的区域时，添加：

```bash
--mask_only
```

## 7. 单次训练流程

每个 iteration 执行以下流程：

```text
读取图像和可选 Mask
        ↓
图像变换；Mask 使用最近邻插值
        ↓
Dinomaly2 Encoder 前向
        ↓
Bottleneck + Decoder
        ↓
计算默认全图损失 L_dinomaly
        ↓
按配置确定 L_dinomaly 的有效区域：默认包含 BG/good/anomaly；启用
`--mask_only` 时只包含 good/anomaly；待定像素始终排除
        ↓
按 Mask 值计算 L_good 和 L_anomaly；BG/待定像素不进入区域损失
        ↓
L = L_dinomaly
    + lambda_good × L_good
    - lambda_anomaly × L_anomaly
        ↓
一次 backward 和 optimizer.step
```

`L_anomaly` 不取倒数。Dinomaly2 的 cosine distance 是有界量，使用负号可以直接实现“最大化 anomaly 区域距离”；倒数在 `L_anomaly` 接近 0 时会产生很大的梯度，训练不稳定。日志中的 `anomaly_term` 为 `-lambda_anomaly × L_anomaly`，因此最终 `loss` 是三个有符号项的和。

Encoder 保持冻结，只训练 Dinomaly2 的 Bottleneck 和 Decoder。整个过程不加载第二个阶段的模型、不增加额外 Head，也不需要单独的 `weak_ok` 训练阶段。

训练 DataLoader 与默认 Dinomaly2 流程保持一致，使用 batch size 8、`shuffle=True`、4 个 worker，并使用 `drop_last=True`。

评估周期由 `--eval_interval` 控制，两个训练模式使用同一参数：

```text
--eval_interval 50     每 50 个 epoch 评估一次
--eval_interval -1     跳过中途评估，只在最终 iteration 后评估一次
```

## 8. 训练输出

训练输出目录为：

```text
save_dir/
└── YYYYMMDDHHMMSS/
    ├── model.pth
    └── tb/
```

每次启动 `dinomaly_2D.py` 都会自动使用启动时刻生成一个
`YYYYMMDDHHMMSS` 目录，不再传入 `--save_name`。

只保存一个模型文件：

```text
model.pth
```

训练期间每 100 个 iteration 打印：

```text
总损失
默认 Dinomaly2 损失
good 区域损失
anomaly 区域损失
当前学习率
耗时和 ETA
```

按照 `--eval_interval`，如果 `Test/` 和对应的 `ground_truth/` 可用，会打印默认 Dinomaly2 指标：

```text
I-AUROC
I-AP
I-F1
P-AUROC
P-AP
P-F1
P-AUPRO
```

## 9. 训练前检查

启动前建议确认：

1. `--data_path/Train/` 存在，并且其中包含图像。
2. `<data_path>/masks` 下的 Mask 与训练图像能够按相对路径或文件名匹配；如果使用其他目录，再检查 `--mask_dir`。
3. Mask 只包含 `0`、`1`、`2`、`255`，或与 `--good_value`、`--anomaly_value`、`--ignore_value` 参数一致的四个值。
4. 图像和 Mask 的原始尺寸对应。
5. `--dataset custom` 用于这种单数据集目录结构。
6. 如果需要训练结束后的默认评估，准备好 `Test/` 和异常图对应的 `ground_truth/`。

## 10. 训练数据增强

`mask_constraint` 训练默认启用以下增强：

| 增强 | 默认参数 | 作用对象 |
|---|---:|---|
| 随机左右翻转 | `--aug_hflip_prob 0.5` | 图像和 Mask 同步 |
| 随机亮度调整 | `--aug_brightness 0.2` | 仅图像 |
| 随机对比度调整 | `--aug_contrast 0.2` | 仅图像 |
| 随机色调调整 | `--aug_hue 0.1` | 仅图像 |

Mask 不执行颜色变换。左右翻转使用同一个随机决定同步变换图像和 Mask，因此不会破坏区域对应关系。

本流程不执行以下增强：

```text
上下翻转
旋转
随机裁剪
```

关闭某项增强时，将对应参数设为 0。例如完全关闭这些增强：

```bash
--aug_hflip_prob 0 \
--aug_brightness 0 \
--aug_contrast 0 \
--aug_hue 0
```

也可以只调整强度：

```bash
--aug_hflip_prob 0.5 \
--aug_brightness 0.1 \
--aug_contrast 0.1 \
--aug_hue 0.05
```

增强只应用于训练图像，训练结束后的 Test 默认评估不使用随机增强。
