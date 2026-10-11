"""Regression checks for the LD metric and paired source bootstrap."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallu.scoring import bootstrap, counts, ld_counts, score, set_counts


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {'source_id': 'a', 'task': 'qa', 'spans': [(0, 4)]},
            {'source_id': 'a', 'task': 'qa', 'spans': [(2, 6)]},
            {'source_id': 'b', 'task': 'summary', 'spans': [(0, 2)]},
        ]
        self.preds = [[(0, 3)], [(1, 5)], [(0, 1)]]
        self.base = [[(0, 2)], [(0, 3)], []]

    def test_ld_retains_pairwise_overlap_semantics(self):
        self.assertEqual(ld_counts([(0, 5)], [(0, 4), (2, 6)]), (7, 5, 8))
        self.assertEqual(set_counts([(0, 5)], [(0, 4), (2, 6)]), (5, 5, 6))

    def test_rejects_length_mismatch_including_paired_reference(self):
        with self.assertRaises(ValueError):
            score(self.preds[:-1], self.samples)
        with self.assertRaises(ValueError):
            bootstrap(self.preds, self.samples, self.base[:-1])
        with self.assertRaises(ValueError):
            counts(self.preds, self.samples, 'typo')

    def test_default_rounding_unchanged_and_raw_precision_available(self):
        rounded = score(self.preds, self.samples)
        raw = score(self.preds, self.samples, digits=None)
        self.assertEqual(rounded['all'], (87.5, 70.0, 77.78))
        self.assertAlmostEqual(raw['all'][2], 100 * 14 / 18)
        self.assertNotEqual(raw['all'][2], rounded['all'][2])
        raw_ci = bootstrap(self.preds, self.samples, self.base, digits=None)
        self.assertEqual(bootstrap(self.preds, self.samples, self.base),
                         tuple(round(value, 2) for value in raw_ci))

    def test_bootstrap_resamples_whole_sources_on_same_draws(self):
        n, seed = 101, 7
        # Independent manual source aggregation: both answers from source a
        # always travel together, with identical draws for candidate/reference.
        draws = np.random.default_rng(seed).multinomial(2, [0.5, 0.5], size=n)
        candidate = np.array([[6, 7, 8], [1, 1, 2]], dtype=float)
        reference = np.array([[3, 5, 8], [0, 0, 2]], dtype=float)
        cand_total, base_total = draws @ candidate, draws @ reference
        gains = 100 * (2 * cand_total[:, 0] / (cand_total[:, 1] + cand_total[:, 2])
                       - 2 * base_total[:, 0] / (base_total[:, 1] + base_total[:, 2]))
        expected = np.percentile(gains, [2.5, 97.5])
        np.testing.assert_allclose(bootstrap(self.preds, self.samples, self.base,
                                            n=n, seed=seed, digits=None), expected)
        self.assertEqual(bootstrap(self.preds, self.samples, self.preds), (0.0, 0.0))

    def test_empty_bootstrap_is_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, 'at least one sample'):
            bootstrap([], [])
        for n in (0, -1, 1.5, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                bootstrap(self.preds, self.samples, n=n)


if __name__ == '__main__':
    unittest.main()
