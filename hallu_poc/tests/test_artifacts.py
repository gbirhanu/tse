"""Prediction cache safety, alignment, and resumable completion contracts."""
import copy
import pickle
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallu import artifacts


def samples():
    return [{'id': str(i), 'source_id': 'source-' + str(i), 'task': 'qa', 'split': 'test',
             'prompt': 'The supported source.', 'answer': 'A false answer.', 'spans': [(2, 7)]}
            for i in range(3)]


class ExecutablePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        expression = "__import__('pathlib').Path(" + repr(str(self.marker)) + ").write_text('executed')"
        return eval, (expression,)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / 'work_test' / ('artifacts-' + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.addCleanup(self.clean_scratch)

    def clean_scratch(self):
        for path in sorted(self.scratch.rglob('*'), key=lambda path: len(path.parts), reverse=True):
            if not path.resolve().is_relative_to(self.scratch.resolve()):
                raise ValueError('Scratch cleanup escaped its test directory')
            path.unlink() if path.is_file() else path.rmdir()
        self.scratch.rmdir()

    def path(self, name):
        return str(self.scratch / name)

    def test_primitive_pickle_roundtrip_preserves_legacy_offsets(self):
        data = {'ids': ['a'], 'tokens': [[(0, 5, 0.75)]],
                'metadata': {'flag': True, 'count': 2, 'none': None, 'bytes': b'abc'}}
        artifacts.atomic_write(self.path('predictions.pkl'), data, binary=True)
        self.assertEqual(artifacts.read_pickle(self.path('predictions.pkl')), data)

    def test_pickle_cannot_execute_reduce_payload(self):
        marker = self.scratch / 'must-not-exist.txt'
        payload = self.scratch / 'untrusted.pkl'
        payload.write_bytes(pickle.dumps(ExecutablePayload(marker), protocol=4))
        with self.assertRaisesRegex(pickle.UnpicklingError, 'primitive'):
            artifacts.read_pickle(str(payload))
        self.assertFalse(marker.exists())

    def test_alignment_reorders_by_id_with_content_validation(self):
        rows = samples()
        tokens = [[(2, 7, probability)] for probability in (0.1, 0.2, 0.3)]
        record = artifacts.prediction_record(rows, tokens, {'training_scope': 'train_without_dev'})
        self.assertEqual(artifacts.aligned(record, [rows[2], rows[0]]), [tokens[2], tokens[0]])
        self.assertEqual(record['ids'], ['0', '1', '2'])

    def test_duplicate_missing_and_mismatched_prediction_lengths_are_rejected(self):
        rows = samples()
        record = artifacts.prediction_record(rows, [[], [], []], {})
        bad_records = [dict(record, ids=['0', '0', '2']),
                       dict(record, tokens=[[], []]),
                       dict(record, ids=['0', '1', 'other'])]
        for bad in bad_records:
            with self.subTest(record=bad), self.assertRaises(ValueError):
                artifacts.aligned(bad, rows)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            artifacts.aligned(record, [rows[0], rows[0]])
        with self.assertRaisesRegex(ValueError, 'count'):
            artifacts.prediction_record(rows, [[], []], {})

    def test_reused_id_with_changed_content_or_labels_is_rejected(self):
        rows = samples()
        record = artifacts.prediction_record(rows, [[], [], []], {})
        changed_values = {'source_id': 'other-source', 'task': 'summary', 'split': 'train',
                          'prompt': 'Changed source', 'answer': 'Changed answer', 'spans': [(1, 4)]}
        for field, value in changed_values.items():
            altered = copy.deepcopy(rows)
            altered[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'differs'):
                artifacts.aligned(record, altered)

    def test_completion_skips_only_matching_config_and_intact_outputs(self):
        output, marker = self.path('output.json'), self.path('complete.json')
        config = {'seed': 42, 'targets': 'hard', 'limit': 300}
        self.assertFalse(artifacts.is_finished(marker, config))
        artifacts.atomic_write(output, {'tokens': [0.5]})
        artifacts.mark_finished(marker, config, [output])
        self.assertTrue(artifacts.is_finished(marker, dict(config)))
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            artifacts.is_finished(marker, dict(config, seed=43))
        artifacts.atomic_write(output, {'tokens': [0.9]})
        self.assertFalse(artifacts.is_finished(marker, config))
        Path(output).unlink()
        self.assertFalse(artifacts.is_finished(marker, config))

    def test_failed_atomic_replacement_preserves_previous_complete_file(self):
        path = self.path('atomic.json')
        artifacts.atomic_write(path, {'generation': 1})
        with patch.object(artifacts.os, 'replace', side_effect=OSError('simulated disconnect')):
            with self.assertRaisesRegex(OSError, 'simulated disconnect'):
                artifacts.atomic_write(path, {'generation': 2})
        self.assertEqual(artifacts.read_json(path), {'generation': 1})
        self.assertEqual(artifacts.read_json(path + '.tmp'), {'generation': 2})

    def test_missing_output_never_creates_completion_marker(self):
        marker = self.path('complete.json')
        with self.assertRaises(FileNotFoundError):
            artifacts.mark_finished(marker, {'seed': 42}, [self.path('missing.json')])
        self.assertFalse(Path(marker).exists())


if __name__ == '__main__':
    unittest.main()
