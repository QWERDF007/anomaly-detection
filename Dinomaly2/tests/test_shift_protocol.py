import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import production_threshold, shift_fpr_at_tpr, summarize_shift_scores


class ShiftProtocolTests(unittest.TestCase):
    def test_production_threshold_uses_normal_tail(self):
        threshold = production_threshold([0.1, 0.2, 0.3, 0.4], target_fpr=0.25)
        self.assertEqual(threshold, 0.4)

    def test_shift_fpr_at_tpr(self):
        fpr, threshold, recall = shift_fpr_at_tpr(
            [0.1, 0.2, 0.8],
            [0.7, 0.9, 1.0, 1.2],
            target_tpr=0.75,
        )
        self.assertEqual(threshold, 0.7)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(fpr, 1 / 3)

    def test_summary_reports_family_distribution(self):
        summary = summarize_shift_scores(
            [0.1, 0.2, 0.3, 0.4],
            {'brightness': [0.2, 0.3], 'blur': [0.5, 0.6]},
            [0.8, 0.9, 1.0, 1.1],
            target_fpr=0.25,
        )
        self.assertEqual(set(summary['families']), {'brightness', 'blur'})
        self.assertTrue(np.isfinite(summary['worst_production_shift_fpr']))


if __name__ == '__main__':
    unittest.main()
