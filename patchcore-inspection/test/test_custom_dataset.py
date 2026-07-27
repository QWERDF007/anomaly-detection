from pathlib import Path

import numpy as np
from PIL import Image

from patchcore.datasets.custom import CustomDataset
from patchcore.datasets.mvtec import DatasetSplit


def _write_image(path: Path, value: int, size: int = 24):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size, size, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_custom_dataset_mvtec_layout(tmp_path):
    root = tmp_path / "widget"
    for index in range(3):
        _write_image(root / "train" / "good" / f"train_{index}.png", index)
    _write_image(root / "test" / "good" / "good.png", 10)
    _write_image(root / "test" / "scratch" / "sample.png", 200)

    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:12, 5:14] = 255
    (root / "ground_truth" / "scratch").mkdir(parents=True)
    Image.fromarray(mask).save(root / "ground_truth" / "scratch" / "sample.png")

    train = CustomDataset(root=root, resize=32, imagesize=16, split=DatasetSplit.TRAIN)
    assert len(train) == 3
    assert train[0]["image"].shape == (3, 16, 16)
    assert train[0]["is_anomaly"] == 0

    test = CustomDataset(root=root, resize=32, imagesize=16, split="test")
    assert len(test) == 2
    assert [item[1] != "good" for item in test.data_to_iterate] == [False, True]
    assert test[1]["mask"].shape == (1, 16, 16)
    assert float(test[1]["mask"].sum()) > 0


def test_custom_dataset_train_validation_split(tmp_path):
    root = tmp_path / "widget"
    for index in range(4):
        _write_image(root / "train" / "normal" / f"train_{index}.jpg", index)

    train = CustomDataset(root=root, split=DatasetSplit.TRAIN, train_val_split=0.5)
    validation = CustomDataset(root=root, split=DatasetSplit.VAL, train_val_split=0.5)
    assert len(train) == 2
    assert len(validation) == 2
    assert set(item[1] for item in validation.data_to_iterate) == {"good"}




