# Dinomaly2 双库二阶段推理

`dinomaly_two_stage.py` 实现以下流程：

1. 用 Dinomaly2 计算原始异常分数和原始分辨率 score map；
2. 原始分数严格大于 `--score_threshold` 时才进入第二阶段，否则直接判定为正常；
3. 将 score map 中的异常连通区域映射到 Dinomaly2 的编码器特征图，做带 Mask 的 ROIAlign；
4. 分别搜索良品库和异常库，得到 `good_distance` 与 `anomaly_distance`；
5. 用两者的相对距离差生成偏移值。与良品库更近时减去偏移值，与异常库更近时加上偏移值；
6. 用修正后的分数再次和同一个阈值比较，输出最终判定。

距离处理公式为：

```text
confidence = (farther_distance - nearer_distance)
             / (farther_distance + nearer_distance + eps)
offset = min(offset_scale * confidence, max_offset)  # max_offset 可选
```

FAISS 距离越小表示越相似。一个图像有多个候选区域时，使用置信度最大的区域产生图像级偏移；所有区域的距离仍会写入该图像的 JSON 明细。距离相同或无效时不修正分数。阈值比较使用严格 `>`，因此等于阈值时判定为正常。

## 1. 输入 Mask

`--masks` 可以是目录或单个文件，支持：

- 二值/灰度 PNG、JPG、BMP、TIFF、WebP；非零像素为 ROI；
- 2D `.npy` Mask；
- Labelme `.json`，脚本会把 polygon、rectangle、circle 和 line/linestrip 栅格化。

图像和 Mask 优先按相对路径匹配，也支持同名、`<stem>_mask`、`good/<relative>` 等常见目录布局。Mask 坐标会和图像一起经过 Dinomaly2 的 `Resize(image_size)` 与 `CenterCrop(crop_size)`，避免 `crop_size < image_size` 时 ROI 偏移。

## 2. 建立两个特征库

良品库和异常库需要分别建立。两次命令的模型、输入尺寸、`feature_merge` 和 `roi_size` 必须保持一致：

```bash
cd /path/to/anomaly-detection/Dinomaly2

python dinomaly_two_stage.py build-library \
  --model /path/to/model.pth \
  --images /path/to/library_images/good \
  --masks /path/to/library_masks/good \
  --library good \
  --output_dir /path/to/libraries/good \
  --backbone dinov2reg_vit_small_14 \
  --image_size 672 \
  --crop_size 672 \
  --feature_merge mean \
  --roi_size 7 \
  --gpu 0

python dinomaly_two_stage.py build-library \
  --model /path/to/model.pth \
  --images /path/to/library_images/anomaly \
  --masks /path/to/library_masks/anomaly \
  --library anomaly \
  --output_dir /path/to/libraries/anomaly \
  --backbone dinov2reg_vit_small_14 \
  --image_size 672 \
  --crop_size 672 \
  --feature_merge mean \
  --roi_size 7 \
  --gpu 0
```

如果 good 和 anomaly 区域位于同一批 Labelme JSON 中，可以让脚本按
`shape["label"]` 自动分库：

```bash
python dinomaly_two_stage.py build-libraries \
  --model /path/to/model.pth \
  --images /path/to/library_images \
  --masks /path/to/labelme_annotations \
  --output_dir /path/to/libraries \
  --good_labels good normal OK \
  --ignore_labels ignore \
  --backbone dinov2reg_vit_small_14 \
  --image_size 672 \
  --crop_size 672 \
  --feature_merge mean \
  --roi_size 7 \
  --gpu 0
```

规则为：Labelme 的 `label` 会去除首尾空格并忽略大小写；匹配
`--good_labels` 的 shape 写入 `libraries/good`，匹配 `--ignore_labels` 的
shape 跳过，其他 label（例如 `scratch`、`crack`、`anomaly`）写入
`libraries/anomaly`。同一个 JSON 可以同时包含 good 和 anomaly shape。
没有有效标注区域的图像会被跳过；因此要建立 good 库，至少需要一个
`good`（或通过 `--good_labels` 指定的正常标签）区域。

每个库目录包含：

```text
index.faiss       # IndexFlatL2
metadata.json     # 特征维度、参数和每个向量的图像/Mask/区域记录
vectors.npy       # 与 FAISS 行号一致的向量备份
id_mapping.csv    # vector_id -> image_id/roi_id/图像路径/ROI 坐标
id_mapping.json   # 同一映射的 JSON 版本
```

每个 Mask 的连通区域作为一个 ROI 向量入库。默认对向量做 L2 归一化；如需关闭，在两个建库命令中都添加 `--no-normalize`。

`image_id` 根据图像相对特征库根目录的路径生成，`roi_id` 由 `image_id` 和 Mask 连通区域 ID 生成；这两个 ID 在同一批图像/Mask 下重建时保持一致。`vector_id` 是当前 FAISS 索引的行号，用于查询后读取映射表。

## 3. 二阶段预测

```bash
python dinomaly_two_stage.py predict \
  --model /path/to/model.pth \
  --input /path/to/test/images \
  --good_library /path/to/libraries/good \
  --anomaly_library /path/to/libraries/anomaly \
  --score_threshold 0.35 \
  --output_dir /path/to/two_stage_result \
  --backbone dinov2reg_vit_small_14 \
  --image_size 672 \
  --crop_size 672 \
  --feature_merge mean \
  --roi_size 7 \
  --offset_scale 1.0 \
  --gpu 0
```

`--threshold` 是 `--score_threshold` 的别名。`--roi_dilation 1` 可在 ROIAlign 时向候选区域外扩一圈，但输出的 candidate mask 仍是原始 score 阈值区域。

输出目录包括：

```text
results.csv                 # 每张图的 raw/adjusted 分数、两次判定、距离和偏移
roi_results.csv             # 每个候选 ROI 的良品/异常距离和偏移方向
run.json                    # 全部运行参数和每个 ROI 的明细
score_maps/<image>.npy      # Dinomaly2 原始 score map
candidate_regions/<image>.png
details/<image>.json        # 每个 ROI 的两个距离、匹配行号和偏移方向
```

`stage1_label` 是原始分数判定，`final_label` 是二阶段修正后的最终判定；`stage2_applied` 只有在第一阶段为异常且至少找到一个有效候选 ROI 时才为真。

## 4. 双阈值特征库预测

如果希望把分数划分为 good、区间候选和 anomaly 三段，可以使用
`dinomaly_two_threshold_predict.py`：

```powershell
python .\dinomaly_two_threshold_predict.py `
  --model D:\model\model.pth `
  --data_root D:\query\data_root `
  --good_library D:\libraries\good `
  --anomaly_library D:\libraries\anomaly `
  --good_threshold 0.20 `
  --anomaly_threshold 0.45 `
  --output_dir D:\dual_threshold_result `
  --gpu 0
```

`data_root` 的一级子目录用于记录数据标签，例如：

```text
D:\query\data_root\good\*.jpg       # dataset_label=good
D:\query\data_root\scratch\*.jpg   # dataset_label=anomaly
D:\query\data_root\crack\*.jpg     # dataset_label=anomaly
```

脚本会递归搜索这些子目录；结果 CSV 中的 `dataset_label` 是目录标签，
`final_label` 是模型和特征库的预测结果。

判定规则：

1. 原始分数 `< good_threshold`，直接判为 good；
2. 原始分数 `> anomaly_threshold`，直接判为 anomaly；
3. 两个阈值之间（包括等于阈值）使用 `good_threshold` 提取候选异常区域，分别搜索 good 和 anomaly 特征库；
4. 如果 `good_distance < anomaly_distance`，说明最近的是 good 库，使用负偏移值（减去偏移）；如果 `anomaly_distance < good_distance`，使用正偏移值（加上偏移）；
5. 偏移后越过阈值时按修正分数判定；仍处于中间区间时按最近特征库判定，距离相同或无有效 ROI 时使用两个阈值的中点作为兜底。

输出包括 `results.csv`、`roi_results.csv`、`run.json`、`score_density.png`、每张图的 score map、候选区域和明细 JSON。`score_density.png` 使用 KDE 连续密度曲线：上图为偏移前 raw score，下图为偏移后 adjusted score；good 使用绿色，anomaly 使用红色。可用 `--density_points` 调整曲线采样点数。

## 5. 反查特征库中的原始图像和 ROI

如果已经有一张输入图像和异常区域 Mask，可以使用新增的
`query_feature_library.py`：

```bash
python query_feature_library.py \
  --model /path/to/model.pth \
  --input /path/to/query/image.jpg \
  --region_mask /path/to/query/anomaly_mask.png \
  --good_library /path/to/libraries/good \
  --anomaly_library /path/to/libraries/anomaly \
  --top_k 3 \
  --output_dir /path/to/lookup_result \
  --backbone dinov2reg_vit_small_14 \
  --image_size 672 \
  --crop_size 672 \
  --feature_merge mean \
  --roi_size 7 \
  --gpu 0
```

也可以直接输入 score map，由脚本按阈值生成异常区域：

```bash
python query_feature_library.py \
  --model /path/to/model.pth \
  --input /path/to/query/image.jpg \
  --score_map /path/to/query/score.npy \
  --score_threshold 0.35 \
  --library /path/to/libraries/good /path/to/libraries/anomaly \
  --output_dir /path/to/lookup_result \
  --gpu 0
```

结果写入 `lookup_results.csv` 和 `lookup_results.json`，包含查询 ROI、库类型、距离、`vector_id`、`image_id`、`roi_id`、原始图像路径、Mask 路径和库中 ROI 坐标。

## 6. PySide6 图形界面反查

安装依赖后可以启动 GUI。左侧打开输入图像并选择矩形或多边形，在图像上绘制异常区域；多边形左键依次点击顶点后，点击右键即可完成多边形（也可以双击或点击“完成多边形”按钮）。点击“查询特征库”后，右侧显示匹配的原始图像和对应 ROI。下方结果表只显示原始图像路径和距离，选中不同结果即可切换右侧图像。

```powershell
python .\query_feature_library_gui.py `
  --model D:\model\model.pth `
  --good_library D:\libraries\good `
  --anomaly_library D:\libraries\anomaly `
  --gpu 0 `
  --output_dir D:\gui_lookup_results
```

也可以用 `--input D:\query\image.jpg` 预先打开输入图像。GUI 会在后台调用
`query_feature_library.py`，查询完成后会把 `image_id`、`roi_id`、距离、图像路径和
ROI bbox 打印到启动 GUI 的命令行窗口；完整 CSV/JSON 结果保存在每次查询的输出子目录中。
