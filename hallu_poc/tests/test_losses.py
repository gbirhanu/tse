"""Character coverage and useful gradients for the optional soft-F1 surrogate."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallu.losses import character_targets, soft_span_f1_loss


class CharacterTargetTests(unittest.TestCase):
    def test_character_lengths_partial_tokens_and_gold_union(self):
        weights, gold = character_targets('abcdefghij', [(1, 3), (2, 5), (1, 3)],
                                          [(0, 1), (1, 4), (4, 10)])
        np.testing.assert_array_equal(weights, [1, 3, 6])
        np.testing.assert_allclose(gold, [0, 1, 1 / 6])
        self.assertAlmostEqual(float((weights * gold).sum()), 4.0)

    def test_partial_overlap_splits_both_character_and_gold_mass(self):
        weights, gold = character_targets('abcdef', [(0, 3)], [(0, 4), (2, 6)])
        np.testing.assert_array_equal(weights, [3, 3])
        np.testing.assert_allclose(gold, [5 / 6, 1 / 6])
        np.testing.assert_allclose(weights * gold, [2.5, 0.5])
        self.assertEqual(float(weights.sum()), 6.0)

    def test_duplicate_offsets_do_not_double_count_characters(self):
        weights, gold = character_targets('abcd', [(0, 1)], [(0, 2), (0, 2), (2, 4)])
        np.testing.assert_array_equal(weights, [1, 1, 2])
        np.testing.assert_array_equal(gold, [0.5, 0.5, 0])
        self.assertEqual(float(weights.sum()), 4.0)
        self.assertEqual(float((weights * gold).sum()), 1.0)

    def test_empty_reversed_and_out_of_bounds_offsets(self):
        weights, gold = character_targets('abcd', [(-5, 2), (3, 99)],
                                          [(-10, 1), (1, 99), (2, 2), (4, 2), (8, 9)])
        np.testing.assert_array_equal(weights, [1, 3, 0, 0, 0])
        np.testing.assert_allclose(gold, [1, 2 / 3, 0, 0, 0])
        weights, gold = character_targets('', [(0, 1)], [(0, 0)])
        np.testing.assert_array_equal(weights, [0])
        np.testing.assert_array_equal(gold, [0])

    def test_uncovered_gold_characters_have_no_prediction_mass(self):
        weights, gold = character_targets('abcdef', [(2, 4)], [(0, 1), (5, 6)])
        np.testing.assert_array_equal(weights, [1, 1])
        np.testing.assert_array_equal(gold, [0, 0])

    def test_noninteger_offsets_fail_instead_of_silently_truncating(self):
        with self.assertRaisesRegex(ValueError, 'integers'):
            character_targets('abc', [], [(0.5, 2)])


class SoftSpanF1Tests(unittest.TestCase):
    def test_positive_and_negative_gradients_push_toward_gold(self):
        logits = torch.tensor([-0.3, 0.7], requires_grad=True)
        loss = soft_span_f1_loss(logits.sigmoid(), torch.tensor([1., 2.]), torch.tensor([1., 0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1].item(), 0)

    def test_overlapping_token_loss_matches_explicit_character_average(self):
        weights, gold = character_targets('abcdef', [(0, 3)], [(0, 4), (2, 6)])
        probabilities = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
        actual = soft_span_f1_loss(probabilities, torch.tensor(weights), torch.tensor(gold))
        char_probs = torch.stack([probabilities[0], probabilities[0], probabilities.mean(),
                                  probabilities.mean(), probabilities[1], probabilities[1]])
        expected = 1 - (2 * char_probs[:3].sum() + 1e-8) / (char_probs.sum() + 3 + 1e-8)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
        actual_grad, = torch.autograd.grad(actual, probabilities, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, probabilities)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-7)

    def test_batch_is_character_pooled_instead_of_mean_token_or_sample_f1(self):
        probabilities = torch.tensor([0.9, 0.4])
        weights = torch.tensor([1., 9.])
        gold = torch.tensor([1., 0.])
        expected = 1 - (2 * 0.9 + 1e-8) / (0.9 + 9 * 0.4 + 1 + 1e-8)
        self.assertAlmostEqual(soft_span_f1_loss(probabilities, weights, gold).item(), expected, places=6)

    def test_all_negative_batch_has_useful_false_positive_gradients(self):
        for values in ([0.2, 0.8], [0.0, 0.0]):
            with self.subTest(values=values):
                probabilities = torch.tensor(values, requires_grad=True)
                loss = soft_span_f1_loss(probabilities, torch.tensor([1., 3.]), torch.zeros(2))
                self.assertAlmostEqual(loss.item(), (values[0] + 3 * values[1]) / 4, places=6)
                loss.backward()
                torch.testing.assert_close(probabilities.grad, torch.tensor([0.25, 0.75]))

    def test_no_covered_characters_and_empty_batch_are_differentiable_zero(self):
        for values in ([0.2, 0.8], []):
            with self.subTest(values=values):
                probabilities = torch.tensor(values, requires_grad=True)
                loss = soft_span_f1_loss(probabilities, torch.zeros_like(probabilities),
                                         torch.zeros_like(probabilities))
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                torch.testing.assert_close(probabilities.grad, torch.zeros_like(probabilities))

    def test_perfect_predictions_have_zero_loss(self):
        loss = soft_span_f1_loss(torch.tensor([1., 0.]), torch.tensor([1., 7.]), torch.tensor([1., 0.]))
        self.assertEqual(loss.item(), 0.0)

    def test_fractional_gold_gradient_matches_finite_differences(self):
        probabilities = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
        weights, gold = torch.tensor([3., 2.], dtype=torch.float64), torch.tensor([0.7, 0.1], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda p: soft_span_f1_loss(p, weights, gold),
                                               (probabilities,), eps=1e-6, atol=1e-5))

    def test_half_precision_is_accumulated_in_float32(self):
        probabilities = torch.tensor([0.2, 0.8], dtype=torch.float16, requires_grad=True)
        loss = soft_span_f1_loss(probabilities, torch.tensor([1., 2.]), torch.tensor([1., 0.]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    def test_shapes_cannot_silently_broadcast(self):
        with self.assertRaisesRegex(ValueError, 'identical shapes'):
            soft_span_f1_loss(torch.zeros(2), torch.zeros(2, 1), torch.zeros(2))


if __name__ == '__main__':
    unittest.main()
