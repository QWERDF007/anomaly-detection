# Custom datasets

The standalone `train.py` and `predict.py` scripts use the same directory
convention as Dinomaly2 and MVTec AD:

```text
widget/
├── train/
│   └── good/                 # normal images used to build the memory bank
├── test/
│   ├── good/
│   └── scratch/              # any directory other than good/normal/ok is anomalous
└── ground_truth/
    └── scratch/
        └── image_001.png     # optional mask: image_001.png or image_001_mask.png
```

`train` may also contain images directly, without a `good` directory.  Test
images may be nested below their class directories.  Supported image formats
include PNG, JPG/JPEG, BMP, TIFF and WEBP.  If a test image has no mask, a zero
mask is used and image-level prediction still works.

Train and save a model:

```bash
python train.py --data_path /path/to/widget \
  --dataset custom --backbone wideresnet50 \
  --save_dir ./saved_results --save_name widget
```

The model is written to
`saved_results/widget/models/widget/`.  To train several category directories
under one root, repeat `--category` (or pass a comma-separated list):

```bash
python train.py --data_path /path/to/datasets --dataset custom \
  --category widget_a --category widget_b
```

Run prediction on one image or a directory:

```bash
python predict.py \
  --model ./saved_results/widget/models/widget \
  --input /path/to/images --output ./predictions
```

The output directory contains `results.csv`, one raw anomaly map (`.npy`) per
image, and heatmap/overlay PNGs.  Add `--threshold <value>` to also save binary
prediction masks.

All training and prediction parameters are provided directly to `train.py` and `predict.py`; no chained CLI is required.
