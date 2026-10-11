"""CPU-only contracts for cache alignment and probability averaging."""
import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallu.ensemble import ensemble_tokens, fingerprint_tokenizer


class FakeBackend:
    def __init__(self, definition):
        self.definition = definition

    def to_str(self):
        return json.dumps(self.definition)


class FakeTokenizer:
    def __init__(self, definition, special_tokens=None):
        self.backend_tokenizer = FakeBackend(definition)
        self.special_tokens_map = special_tokens or {'pad_token': '[PAD]'}


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.samples = [{'id': 'a', 'answer': 'hello world'}, {'id': 'b', 'answer': ''}]
        self.members = [
            [[(0, 5, 0.2), (6, 11, 0.8)], []],
            [[(0, 5, 0.8), (6, 11, 0.4)], []],
        ]

    def test_equal_average_preserves_inputs_and_empty_answers(self):
        before = copy.deepcopy(self.members)
        result = ensemble_tokens(self.members, self.samples)
        self.assertEqual(result[1], [])
        self.assertEqual([t[:2] for t in result[0]], [(0, 5), (6, 11)])
        self.assertAlmostEqual(result[0][0][2], 0.5)
        self.assertAlmostEqual(result[0][1][2], 0.6)
        self.assertEqual(self.members, before)

    def test_normalizes_predeclared_weights(self):
        result = ensemble_tokens(self.members, self.samples, [3, 1])
        self.assertAlmostEqual(result[0][0][2], 0.35)
        self.assertAlmostEqual(result[0][1][2], 0.7)
        self.assertEqual(ensemble_tokens(self.members, self.samples, [1, 0]), self.members[0])
        self.assertEqual(ensemble_tokens(self.members, self.samples, [1e308, 1e308]),
                         ensemble_tokens(self.members, self.samples))

    def test_repeated_and_overlapping_byte_offsets_are_preserved(self):
        members = [[[(0, 2, 0.0), (0, 2, 1.0), (1, 3, 0.2)]],
                   [[(0, 2, 1.0), (0, 2, 0.0), (1, 3, 0.8)]]]
        self.assertEqual(ensemble_tokens(members, [{'answer': 'abc'}]),
                         [[(0, 2, 0.5), (0, 2, 0.5), (1, 3, 0.5)]])

    def test_rejects_missing_members_answers_and_tokens(self):
        cases = [[], [self.members[0][:-1], self.members[1]],
                 [self.members[0], [None, []]],
                 [self.members[0], [[(0, 5, 0.3)], []]]]
        for members in cases:
            with self.subTest(members=members), self.assertRaises(ValueError):
                ensemble_tokens(members, self.samples)

    def test_rejects_changed_offsets_despite_equal_token_count(self):
        self.members[1][0][0] = (0, 4, 0.8)
        with self.assertRaisesRegex(ValueError, 'offsets do not match'):
            ensemble_tokens(self.members, self.samples)

    def test_rejects_invalid_offsets_and_probabilities(self):
        invalid_tokens = [(-1, 5, 0.1), (0, 12, 0.1), (5, 0, 0.1),
                          (0.0, 5, 0.1), (False, 5, 0.1),
                          (0, 5, float('nan')), (0, 5, float('inf')),
                          (0, 5, -0.1), (0, 5, 1.1), (0, 5, '0.5'), (0, 5)]
        for token in invalid_tokens:
            members = copy.deepcopy(self.members)
            members[1][0][0] = token
            with self.subTest(token=token), self.assertRaises(ValueError):
                ensemble_tokens(members, self.samples, [1, 0])

    def test_rejects_invalid_weights(self):
        for weights in ([0, 0], [-1, 2], [1], [1, 2, 3],
                        [float('nan'), 1], [float('inf'), 1], ['1', 1], [True, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                ensemble_tokens(self.members, self.samples, weights)


class TokenizerFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.definition = {'model': {'vocab': {'hello': 1, '[PAD]': 0}, 'type': 'BPE'},
                           'added_tokens': [{'id': 0, 'content': '[PAD]', 'special': True}],
                           'padding': None, 'truncation': None}

    def test_ignores_runtime_padding_truncation_and_json_key_order(self):
        other = dict(reversed(list(copy.deepcopy(self.definition).items())))
        other['padding'] = {'length': 32}
        other['truncation'] = {'max_length': 8192, 'strategy': 'OnlyFirst'}
        self.assertEqual(fingerprint_tokenizer(FakeTokenizer(self.definition)),
                         fingerprint_tokenizer(FakeTokenizer(other)))

    def test_vocabulary_and_special_role_changes_change_identity(self):
        reference = fingerprint_tokenizer(FakeTokenizer(self.definition))
        changed = copy.deepcopy(self.definition)
        changed['model']['vocab']['hello'] = 2
        self.assertNotEqual(reference, fingerprint_tokenizer(FakeTokenizer(changed)))
        self.assertNotEqual(reference, fingerprint_tokenizer(
            FakeTokenizer(self.definition, {'unk_token': '[PAD]'})))

    def test_requires_serializable_fast_tokenizer(self):
        with self.assertRaisesRegex(ValueError, 'fast tokenizer'):
            fingerprint_tokenizer(object())


if __name__ == '__main__':
    unittest.main()
