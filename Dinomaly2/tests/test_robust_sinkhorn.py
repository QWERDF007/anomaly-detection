import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


DINOMALY_ROOT = Path(__file__).resolve().parents[1]
if str(DINOMALY_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOMALY_ROOT))

from losses import CombinedDinomalyLoss, RobustSinkhornLoss


class RobustSinkhornLossTests(unittest.TestCase):
    def test_non_divisible_feature_map_keeps_every_token(self):
        criterion = RobustSinkhornLoss(window_size=5, stride=5)
        feature = torch.randn(2, 3, 6, 7)

        windows, valid = criterion._extract_windows(feature)

        self.assertEqual(tuple(windows.shape), (2, 4, 25, 3))
        self.assertTrue(torch.equal(valid.sum(dim=(1, 2)), torch.tensor([42, 42])))

    def test_balanced_identical_features_have_unit_mass_and_zero_cost(self):
        criterion = RobustSinkhornLoss(
            window_size=3,
            stride=3,
            epsilon=0.05,
            iterations=30,
            ot_type='balanced',
        )
        feature = torch.ones(2, 4, 5, 4)

        result = criterion([feature], [feature.clone()])

        self.assertLess(result['loss_ot'].item(), 1e-6)
        self.assertAlmostEqual(result['transport_mass'].item(), 1.0, places=5)

    def test_local_transport_handles_a_within_window_permutation(self):
        criterion = RobustSinkhornLoss(
            window_size=5,
            stride=5,
            epsilon=0.03,
            iterations=40,
            ot_type='balanced',
        )
        encoder = torch.eye(25).T.reshape(1, 25, 5, 5)
        decoder = encoder.flatten(2).roll(shifts=1, dims=2).reshape_as(encoder)
        pointwise = (1 - F.cosine_similarity(encoder, decoder, dim=1)).mean()

        result = criterion([encoder], [decoder])

        self.assertGreater(pointwise.item(), 0.9)
        self.assertLess(result['loss_ot'].item(), 1e-5)

    def test_spatial_cost_penalizes_long_permutations(self):
        encoder = torch.eye(25).T.reshape(1, 25, 5, 5)
        decoder = encoder.flatten(2).flip(2).reshape_as(encoder)
        plain = RobustSinkhornLoss(
            window_size=5,
            stride=5,
            epsilon=0.03,
            iterations=40,
            position_weight=0.0,
            ot_type='balanced',
        )
        spatial = RobustSinkhornLoss(
            window_size=5,
            stride=5,
            epsilon=0.03,
            iterations=40,
            position_weight=0.5,
            ot_type='balanced',
        )

        plain_loss = plain([encoder], [decoder])['loss_ot']
        spatial_loss = spatial([encoder], [decoder])['loss_ot']

        self.assertGreater(spatial_loss.item(), plain_loss.item() + 0.1)

    def test_unbalanced_transport_reduces_mass_for_an_outlier(self):
        encoder = torch.eye(25).T.reshape(1, 25, 5, 5)
        decoder = encoder.clone()
        decoder[:, :, 0, 0] = -encoder[:, :, 0, 0]
        balanced = RobustSinkhornLoss(
            window_size=5,
            stride=5,
            epsilon=0.05,
            iterations=40,
            ot_type='balanced',
        )
        unbalanced = RobustSinkhornLoss(
            window_size=5,
            stride=5,
            epsilon=0.05,
            rho=0.2,
            iterations=40,
            ot_type='unbalanced',
        )

        balanced_mass = balanced([encoder], [decoder])['transport_mass']
        unbalanced_mass = unbalanced([encoder], [decoder])['transport_mass']

        self.assertAlmostEqual(balanced_mass.item(), 1.0, places=4)
        self.assertLess(unbalanced_mass.item(), balanced_mass.item())

    def test_encoder_is_stop_gradient_and_decoder_receives_gradient(self):
        torch.manual_seed(7)
        encoder = torch.randn(1, 6, 5, 5, requires_grad=True)
        decoder = torch.randn(1, 6, 5, 5, requires_grad=True)
        criterion = RobustSinkhornLoss(window_size=5, stride=5, iterations=10)

        criterion([encoder], [decoder])['loss_ot'].backward()

        self.assertIsNone(encoder.grad)
        self.assertIsNotNone(decoder.grad)
        self.assertTrue(torch.isfinite(decoder.grad).all())

    def test_combined_loss_warmup_and_zero_lambda(self):
        torch.manual_seed(9)
        encoder = torch.randn(1, 4, 3, 3)
        decoder = torch.randn(1, 4, 3, 3, requires_grad=True)
        robust = RobustSinkhornLoss(window_size=3, stride=3, iterations=10)
        combined = CombinedDinomalyLoss(
            robust,
            lambda_ot=0.1,
            warmup_iters=10,
            enabled=True,
        )
        base = decoder.sum() * 0 + 2.0

        result = combined(base, [encoder], [decoder], iteration=5)

        self.assertAlmostEqual(result['lambda_ot'].item(), 0.05, places=6)
        expected = base + 0.05 * result['loss_ot']
        self.assertTrue(torch.allclose(result['loss_total'], expected))

        baseline = CombinedDinomalyLoss(
            robust,
            lambda_ot=0.0,
            warmup_iters=10,
            enabled=True,
        )(base, [encoder], [decoder], iteration=5)
        self.assertIs(baseline['loss_total'], base)
        self.assertEqual(baseline['loss_ot'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
