"""Leakage, exact evaluation population, and declared-reference acceptance gates."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallu import artifacts, experiments


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / 'work_test' / ('protocol-' + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.addCleanup(self.clean_scratch)
        self.paths = {'preds': str(self.scratch / 'preds'), 'results': str(self.scratch / 'results')}
        self.test = [{'id': str(i), 'source_id': 'source-' + str(i // 2), 'task': 'qa',
                      'split': 'test', 'prompt': 'source', 'answer': 'abc', 'spans': [(0, 1)]}
                     for i in range(8)]
        self.perfect = [[(0, 1)] for _ in self.test]
        self.empty = [[] for _ in self.test]
        self.reference(self.empty)

    def clean_scratch(self):
        for path in sorted(self.scratch.rglob('*'), key=lambda path: len(path.parts), reverse=True):
            if not path.resolve().is_relative_to(self.scratch.resolve()):
                raise ValueError('Scratch cleanup escaped its test directory')
            path.unlink() if path.is_file() else path.rmdir()
        self.scratch.rmdir()

    def reference(self, spans, **overrides):
        record = {'ids': [s['id'] for s in self.test], 'spans': {'baseline': spans},
                  'test_fingerprint': artifacts.sample_fingerprint(self.test)}
        record.update(overrides)
        artifacts.atomic_write(str(Path(self.paths['preds']) / 'reference_spans.pkl'), record, binary=True)

    def rows(self, spans=None, smoke=False):
        return experiments.result_rows('candidate', 'fixed', spans or {'baseline': self.perfect},
                                       self.test, self.paths, 'reference:baseline', smoke)

    def legacy(self, name='ld_large'):
        ids = [s['id'] for s in self.test]
        records = [(self.paths['preds'], name + '_spans.pkl',
                    {'ids': ids, 'spans': {'baseline': self.perfect}}, True),
                   (self.paths['preds'], name + '_test.pkl',
                    {'ids': ids, 'tokens': [[(0, 1, 0.9), (1, 3, 0.1)] for _ in ids]}, True),
                   (self.paths['results'], name + '_levers.json',
                    {'rows': [{'setting': 'baseline', 'ld': experiments.score(self.perfect, self.test)}]}, False)]
        if name in ('mb_hard', 'mb_soft'):
            records.append((self.paths['results'], name + '_training.json',
                            {'base': 'answerdotai/ModernBERT-base', 'seed': 42,
                             'targets': 'hard' if name == 'mb_hard' else 'soft'}, False))
        for folder, filename, record, binary in records:
            artifacts.atomic_write(str(Path(folder) / filename), record, binary=binary)

    def test_own_models_allow_only_unseen_dev_or_fixed_decoding(self):
        own = {'model': 'answerdotai/ModernBERT-base', 'training_scope': 'train_without_dev'}
        experiments.require_fit(own, 'dev')
        experiments.require_fit(own, 'dev', ensemble=True)
        experiments.require_fit(own, 'fixed')
        for metadata in (own, dict(own, training_scope='ragtruth_all_train')):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, 'Illegal calibration'):
                experiments.require_fit(metadata, 'crossfit')

    def test_crossfit_is_reserved_for_single_published_lettucedetect(self):
        published = {'model': experiments.PUBLISHED_LD, 'training_scope': 'ragtruth_all_train'}
        experiments.require_fit(published, 'crossfit')
        experiments.require_fit(published, 'fixed')
        for fit, ensemble in (('dev', False), ('dev', True), ('crossfit', True)):
            with self.subTest(fit=fit, ensemble=ensemble), self.assertRaises(ValueError):
                experiments.require_fit(published, fit, ensemble=ensemble)
        with self.assertRaises(ValueError):
            experiments.require_fit(dict(published, model='my-finetuned-ld'), 'crossfit')

    def test_mixed_or_unknown_provenance_cannot_fit_dev_or_crossfit(self):
        for scope in ('mixed', 'unknown', None):
            metadata = {'training_scope': scope}
            experiments.require_fit(metadata, 'fixed', ensemble=True)
            for fit in ('dev', 'crossfit'):
                with self.subTest(scope=scope, fit=fit), self.assertRaises(ValueError):
                    experiments.require_fit(metadata, fit, ensemble=True)

    def test_reference_requires_exact_ids_and_content(self):
        self.assertEqual(experiments.reference_spans(self.paths, 'reference:baseline', self.test), self.empty)
        changes = {'prompt': 'different source', 'answer': 'different answer',
                   'spans': [(1, 2)], 'source_id': 'different source ID'}
        for field, value in changes.items():
            changed = copy.deepcopy(self.test)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'different test data'):
                experiments.reference_spans(self.paths, 'reference:baseline', changed)
        for ids in ([s['id'] for s in self.test[:-1]],
                    [s['id'] for s in self.test] + ['extra']):
            self.reference(self.empty, ids=ids)
            with self.assertRaisesRegex(ValueError, 'exactly the same'):
                experiments.reference_spans(self.paths, 'reference:baseline', self.test)
        self.reference(self.empty + [[]], ids=[s['id'] for s in self.test] + ['0'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            experiments.reference_spans(self.paths, 'reference:baseline', self.test)

    def test_unknown_reference_rejects_unverifiable_legacy_test_identity(self):
        path = str(Path(self.paths['preds']) / 'reference_spans.pkl')
        record = artifacts.read_pickle(path)
        record.pop('test_fingerprint')
        artifacts.atomic_write(path, record, binary=True)
        with self.assertRaisesRegex(ValueError, 'fingerprint|identity|provenance'):
            experiments.reference_spans(self.paths, 'reference:baseline', self.test)

    def test_documented_legacy_references_require_reproduced_spans_and_scores(self):
        for name in ('ld_large', 'mb_hard', 'mb_soft'):
            self.legacy(name)
            with self.subTest(name=name):
                actual = experiments.reference_spans(self.paths, name + ':baseline', self.test)
                self.assertEqual(actual, self.perfect)

    def test_legacy_reference_rejects_changed_baseline_reconstruction(self):
        self.legacy()
        path = str(Path(self.paths['preds']) / 'ld_large_spans.pkl')
        record = artifacts.read_pickle(path)
        record['spans']['baseline'][0] = [(1, 2)]
        artifacts.atomic_write(path, record, binary=True)
        with self.assertRaisesRegex(ValueError, 'baseline disagrees'):
            experiments.reference_spans(self.paths, 'ld_large:baseline', self.test)

    def test_legacy_reference_rejects_changed_recorded_score(self):
        self.legacy()
        path = str(Path(self.paths['results']) / 'ld_large_levers.json')
        record = artifacts.read_json(path)
        record['rows'][0]['ld']['all'][2] = 99.0
        artifacts.atomic_write(path, record)
        with self.assertRaisesRegex(ValueError, 'recorded LD scores'):
            experiments.reference_spans(self.paths, 'ld_large:baseline', self.test)

    def test_legacy_reference_rejects_missing_token_ids(self):
        self.legacy()
        path = str(Path(self.paths['preds']) / 'ld_large_test.pkl')
        record = artifacts.read_pickle(path)
        record['ids'].pop()
        record['tokens'].pop()
        artifacts.atomic_write(path, record, binary=True)
        with self.assertRaisesRegex(ValueError, 'no cached prediction'):
            experiments.reference_spans(self.paths, 'ld_large:baseline', self.test)

    def test_legacy_own_model_rejects_wrong_seed_provenance(self):
        self.legacy('mb_hard')
        path = str(Path(self.paths['results']) / 'mb_hard_training.json')
        record = artifacts.read_json(path)
        record['seed'] = 43
        artifacts.atomic_write(path, record)
        with self.assertRaisesRegex(ValueError, 'documented Part B'):
            experiments.reference_spans(self.paths, 'mb_hard:baseline', self.test)

    def test_reference_requires_explicit_run_and_setting(self):
        with self.assertRaisesRegex(ValueError, 'RUN:SETTING'):
            experiments.reference_spans(self.paths, 'reference', self.test)

    def test_real_paired_gain_keeps_only_positive_interval(self):
        row = self.rows()[0]
        self.assertEqual(row['decision'], 'keep')
        self.assertEqual(row['reference_gain'], 100.0)
        self.assertEqual(row['reference_gain_ci'], (100.0, 100.0))
        self.assertEqual(row['test_fingerprint'], artifacts.sample_fingerprint(self.test))

    def test_zero_or_negative_lower_bound_rejects_even_with_positive_gain(self):
        bootstrap = experiments.bootstrap
        for interval, decision in (((0.0, 1.0), 'reject'), ((-0.0001, 1.0), 'reject'),
                                   ((0.00001, 1.0), 'keep')):
            paired_calls = []

            def controlled(predicted, test, base_preds=None, **kwargs):
                if base_preds is not None and kwargs.get('digits', 'default') is None:
                    paired_calls.append((test, base_preds))
                    return interval
                return bootstrap(predicted, test, base_preds, **kwargs)

            with self.subTest(interval=interval), patch.object(experiments, 'bootstrap', side_effect=controlled):
                row = self.rows()[0]
            self.assertEqual(row['decision'], decision)
            self.assertEqual(row['reference_gain_ci'], interval)
            self.assertEqual(paired_calls, [(self.test, self.empty)])

    def test_nonpositive_gain_rejects_even_if_interval_is_positive(self):
        self.reference(self.perfect)
        with patch.object(experiments, 'bootstrap', return_value=(0.1, 1.0)):
            row = self.rows({'baseline': self.empty})[0]
        self.assertLess(row['reference_gain'], 0)
        self.assertEqual(row['decision'], 'reject')

    def test_smoke_run_can_never_be_kept(self):
        row = self.rows(smoke=True)[0]
        self.assertEqual(row['reference_gain_ci'], (100.0, 100.0))
        self.assertEqual(row['decision'], 'smoke-only')

    def test_only_declared_final_setting_receives_reference_decision(self):
        rows = self.rows({'baseline': self.empty, 'tuned': self.perfect})
        self.assertEqual([row['decision'] for row in rows], ['not-tested', 'keep'])
        self.assertNotIn('reference_gain_ci', rows[0])
        self.assertEqual(rows[1]['reference'], 'reference:baseline')


if __name__ == '__main__':
    unittest.main()
