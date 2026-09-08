import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


DINOMALY_ROOT = Path(__file__).resolve().parents[1]
if str(DINOMALY_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOMALY_ROOT))

from prepare_data.generate_normal_shift_test import build_transforms, generate


class NormalShiftGeneratorTests(unittest.TestCase):
    def test_generates_expected_custom_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / 'normal'
            output_root = root / 'shifted'
            input_root.mkdir()
            source = np.zeros((24, 32, 3), dtype=np.uint8)
            source[6:18, 8:24] = (120, 180, 220)
            Image.fromarray(source).save(input_root / 'sample.png')

            generated = generate(input_root, output_root)

            expected = len(build_transforms())
            self.assertEqual(generated, expected)
            images = list((output_root / 'test' / 'good').rglob('*.png'))
            self.assertEqual(len(images), expected)
            with (output_root / 'manifest.csv').open(encoding='utf-8') as manifest:
                rows = list(csv.DictReader(manifest))
            self.assertEqual(len(rows), expected)
            self.assertEqual(
                {row['shift'] for row in rows},
                {
                    'translation',
                    'rotation',
                    'brightness',
                    'contrast',
                    'gamma',
                    'gaussian_blur',
                    'hue',
                    'saturation',
                },
            )


if __name__ == '__main__':
    unittest.main()
