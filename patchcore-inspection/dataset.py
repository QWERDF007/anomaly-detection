"""Public dataset helpers for the standalone PatchCore scripts.

This compatibility module mirrors the small ``dataset.py`` entry point used by
Dinomaly2.  The actual implementation lives in ``patchcore.datasets.custom``
so it can also be reused by the original PatchCore command-line tools.
"""

from pathlib import Path
import sys


_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from patchcore.datasets.custom import (  # noqa: E402
    CustomDataset,
    CustomImageDataset,
    ImageInferenceDataset,
    get_data_transforms,
)
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset  # noqa: E402


__all__ = [
    "CustomDataset",
    "CustomImageDataset",
    "DatasetSplit",
    "ImageInferenceDataset",
    "MVTecDataset",
    "get_data_transforms",
]
