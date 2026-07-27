# Dinomaly2 ROI 特征检索综合流程

本文档说明 `dinomaly_roi_pipeline.py` 的使用方法。该脚本将以下步骤
合并为一次运行：

1. 使用 Dinomaly2 对 Train/good、Test/good、Test/anomaly 推理并生成 score map；
2. 从 Dinomaly2 的 DINO encoder 提取 patch token 特征，保存为 NCHW 格式的 `.npy`；
3. 读取 Train/good 的 Labelme polygon，映射到特征图后进行 ROIAlign，建立 FAISS 索引；
4. 根据 Dinomaly2 分数图找到异常 ROI，使用 DINO ROI 特征搜索 FAISS，得到 ROI 距离；
5. 按距离阈值过滤 ROI，并输出过滤前后的分布图和评估指标。

## 一、数据目录组织

推荐目录结构如下：

```text
/data/dataset/
├── train/
│   └── good/
│       ├── 0001.jpg
│       └── 0002.jpg
├── test/
│   ├── good/
│   │   └── 1001.jpg
│   ├── bad/
│   │   └── 2001.jpg
│   └── gg/
│       └── 3001.jpg
├── labelme/
│   └── train/
│       └── good/
│           ├── 0001.json
│           └── 0002.json
└── ground_truth/
    ├── bad/2001.png
    └── gg/3001.png
```

说明：

- `train/good` 和 `test/good` 固定从 `--data_root` 下查找；
- `test` 下除名称为 `good` 外的所有一级子目录都会自动作为异常目录，例如 `test/bad`、`test/gg`；
- `--train_annotation_dir` 指向 Labelme JSON 目录，JSON 文件名需要和 Train/good 图像文件名对应；
- Labelme JSON 使用 `imageWidth`、`imageHeight` 和 `shapes` 中的 `polygon`；每个 polygon 建立一个 ROI；
- `ground_truth` 默认是 `data_root/ground_truth`，需要提供所有非 good 测试目录的像素掩码；Test/good 自动使用全零掩码；
- Ground Truth 文件可以保持相对目录结构，也可以使用图像同名文件，或使用 `<图像名>_mask.png`；
- 为保证 Labelme 坐标与特征图准确对应，建议 `--image_size` 与 `--crop_size` 设置为相同值，例如都设置为 `672`。

## 二、运行方式

进入脚本所在目录后，一次性设置全部参数：

```bash
cd /data/wt/anomaly-detection/Dinomaly2

CUDA_VISIBLE_DEVICES=0 python dinomaly_roi_pipeline.py \
  --data_root /data/dataset \
  --model /data/trainlogs/Dinomaly/model.pth \
  --output_dir /data/roi_pipeline \
  --train_annotation_dir /data/dataset/labelme/train/good \
  --backbone dinov2reg_vit_small_14 \
  --image_size 672 \
  --crop_size 672 \
  --layers 2 3 4 5 6 7 8 9 \
  --feature_merge mean \
  --roi_size 7 \
  --gpu 0
```

也可以先查看中文参数说明：

```bash
python dinomaly_roi_pipeline.py --help
```

## 三、缓存复用

脚本会优先读取：

```text
<output_dir>/scores/
<output_dir>/features/
```

如果所有图像对应的 score `.npy` 和 feature `.npy` 都存在，则直接复用，不重新加载
Dinomaly2 模型，也不会重复推理。第一次运行或缓存不完整时，才会加载 `--model` 并补齐
缺失文件。若修改了模型、输入尺寸或 DINO 层，请使用新的 `--output_dir`，或清理对应
缓存后重新运行。

特征文件保存为 `[1, C, H, W]`，其中只包含 DINO patch token，不包含 CLS token。

推理/缓存读取、ROI 建库和 ROI 查询都会使用 `tqdm` 显示进度、处理速度和预计剩余时间。
控制台不输出逐图像、逐 ROI 的详细日志；缺失标注、ROI 距离、阈值和指标等详细信息写入
`<output_dir>/pipeline.log`。

## 四、输出内容

```text
<output_dir>/
├── scores/
│   ├── train_good/
│   ├── test_good/
│   └── test_anomaly/
│       ├── bad/              test/bad 的缓存
│       └── gg/               test/gg 的缓存
├── features/
│   ├── train_good/
│   ├── test_good/
│   └── test_anomaly/
├── roi_index/
│   ├── roi_index.faiss
│   └── roi_index.json
├── distance_distribution.png
├── score_distribution_comparison.png
├── score_values.csv
├── pipeline.log
├── metrics.json
└── metrics.csv
```

`metrics.json` 和 `metrics.csv` 包含距离过滤前后的：

```text
I-AUROC, I-AP, I-F1,
P-AUROC, P-AP, P-F1, P-AUPRO
```

## 五、参数说明

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--data_root` | 必填 | 数据集根目录。 |
| `--model` | 必填 | Dinomaly2 模型权重 `.pth`；缓存完整时不会重新加载，但参数仍需保留。 |
| `--output_dir` | 必填 | score、feature、FAISS、分布图和指标的输出目录。 |
| `--ground_truth_dir` | `data_root/ground_truth` | 所有非 good 测试图像的像素 Ground Truth 掩码目录。 |
| `--train_annotation_dir` | 必填 | Train/good 的 Labelme JSON 目录。 |
| `--backbone` | `dinov2reg_vit_small_14` | Dinomaly2 使用的 DINOv2 backbone。 |
| `--image_size` | `672` | 输入图像缩放后的正方形边长。 |
| `--crop_size` | `672` | 缩放后的中心裁剪边长；建议和 image_size 相同。 |
| `--dropout` | `0.4` | Dinomaly2 解码器 dropout，必须和训练时一致。 |
| `--la` | `1` | Dinomaly2 解码器参数，必须和训练时一致。 |
| `--lc` | `2` | Dinomaly2 解码器参数，必须和训练时一致。 |
| `--cr` | `1` | Dinomaly2 解码器参数，必须和训练时一致。 |
| `--layers` | `2 3 4 5 6 7 8 9` | 提取 DINO patch token 的 Transformer 层编号。 |
| `--feature_merge` | `mean` | 多层特征合并方式：`mean` 平均，`concat` 通道拼接。 |
| `--roi_size` | `7` | ROIAlign 输出尺寸为 `roi_size × roi_size`，之后池化为 ROI 向量。 |
| `--bins` | `30` | 密度曲线的采样分辨率基数；数值越大，波谷定位越细。 |
| `--metric_size` | `256` | 计算像素指标前统一缩放的图像边长。 |
| `--score_threshold` | 自动 | Dinomaly2 score map 异常阈值；自动时从正常+异常分数分布波谷选择。 |
| `--distance_threshold` | 自动 | FAISS 距离阈值；自动时从正常+异常 ROI 距离分布波谷选择。 |
| `--gpu` / `--cuda` | `0` | Dinomaly2 和 FAISS 使用的 GPU 编号；`-1` 表示 CPU。 |
| `--no-normalize` | 关闭开关 | 不对 ROI 特征进行 L2 归一化；默认会归一化。 |

短参数别名：`-i`=`--data_root`、`-m`=`--model`、`-o`=`--output_dir`、
`-gt`=`--ground_truth_dir`、`-ann`=`--train_annotation_dir`、
`-imgsz`=`--image_size`、`-csz`=`--crop_size`、`-msz`=`--metric_size`。

## 六、阈值选择规则

- 未指定 `--score_threshold` 时，将 Train/good、Test/good 作为正常分数，所有非 good 测试目录作为异常分数，
  在合并密度分布的两个峰之间寻找波谷；
- 未指定 `--distance_threshold` 时，对正常和异常 ROI 距离使用同样的波谷方法；
- 如果两个分布峰重叠，脚本依次使用最大间隔或两组中位数中点作为退化情况下的分界；
- 距离过滤保留 `distance >= distance_threshold` 的 ROI；
- `distance_distribution.png` 和 `score_distribution_comparison.png` 使用密度曲线，不使用直方图；
- `score_distribution_comparison.png` 的上半部分是距离过滤前分布，下半部分是过滤后分布。
