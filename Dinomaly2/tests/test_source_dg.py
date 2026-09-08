import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from augment import NormalShiftGenerator, PseudoAnomalyGenerator
from losses import cosine_consistency, identity_loss, map_consistency, preservation_loss
from dinomaly_2D import _phase_ramp
from models.domain_adapter import CanonicalizationAdapter


class SourceDGTests(unittest.TestCase):
    def test_phase_ramp_matches_staged_schedule(self):
        self.assertEqual(_phase_ramp(0.05, 0.10, 0.20), 0.0)
        self.assertAlmostEqual(_phase_ramp(0.15, 0.10, 0.20), 0.5)
        self.assertEqual(_phase_ramp(0.60, 0.20, 0.60), 1.0)

    def test_adapter_is_identity_and_trainable_at_initialization(self):
        torch.manual_seed(3)
        adapter = CanonicalizationAdapter(8, bottleneck_ratio=0.25)
        features = torch.randn(2, 5, 8, requires_grad=True)
        output = adapter(features)
        self.assertTrue(torch.equal(output, features))
        output.square().mean().backward()
        self.assertIsNotNone(adapter.fc2.weight.grad)
        self.assertTrue(torch.isfinite(adapter.fc2.weight.grad).all())

    def test_normal_shift_preserves_shape_and_inverse_identity(self):
        torch.manual_seed(4)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        images = (torch.rand(2, 3, 32, 32) - mean) / std
        generator = NormalShiftGenerator(
            geometry_probability=0.0,
            photometric_probability=0.0,
            degradation_probability=0.0,
            illumination_probability=0.0,
        )
        shifted = generator(images)
        self.assertEqual(tuple(shifted.images.shape), tuple(images.shape))
        self.assertTrue(torch.allclose(shifted.images, images, atol=1e-5))
        self.assertTrue(torch.allclose(shifted.inverse_warp(images), images, atol=1e-5))

    def test_normal_shift_generator_is_reproducible_with_torch_generator(self):
        images = torch.zeros(2, 3, 16, 16)
        generator = NormalShiftGenerator()
        first = generator(images, generator=torch.Generator().manual_seed(11))
        second = generator(images, generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(first.images, second.images))
        self.assertEqual(first.family, second.family)

    def test_pseudo_anomaly_generates_local_mask(self):
        images = torch.zeros(2, 3, 32, 32)
        pseudo, masks = PseudoAnomalyGenerator(0.01, 0.02)(images)
        self.assertEqual(tuple(pseudo.shape), tuple(images.shape))
        self.assertEqual(tuple(masks.shape), (2, 1, 32, 32))
        self.assertGreater(float(masks.sum()), 0.0)
        self.assertGreater(float((pseudo - images).abs().sum()), 0.0)

    def test_source_losses_are_finite(self):
        torch.manual_seed(5)
        anchor = [torch.randn(2, 4, 8, 8, requires_grad=True)]
        shifted = [torch.randn(2, 4, 8, 8, requires_grad=True)]
        masks = torch.zeros(2, 1, 8, 8)
        masks[:, :, 2:4, 2:4] = 1
        losses = [
            cosine_consistency(anchor, shifted),
            identity_loss(anchor, shifted),
            map_consistency(anchor[0], shifted[0]),
            preservation_loss(anchor, shifted, masks),
        ]
        self.assertTrue(all(torch.isfinite(value) for value in losses))
        sum(losses).backward()
        self.assertIsNotNone(shifted[0].grad)


if __name__ == '__main__':
    unittest.main()
