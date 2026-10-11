"""CPU checks for exact training recovery; no model download is required."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallu import encoder


class TinyTokenizer:
    pad_token_id = 0

    def save_pretrained(self, directory):
        Path(directory, 'tokenizer_config.json').write_text('{}', encoding='utf-8')


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(8, 2)
        self.config = SimpleNamespace(to_dict=lambda: {'tiny': True})
        self.training_calls = 0
        self.checkpointing = False

    def forward(self, input_ids, attention_mask):
        self.training_calls += int(self.training)
        return SimpleNamespace(logits=self.classifier(self.dropout(self.embedding(input_ids))))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.checkpointing = gradient_checkpointing_kwargs == {'use_reentrant': False}

    def save_pretrained(self, directory):
        torch.save(self.state_dict(), Path(directory, 'pytorch_model.bin'))


def tiny_encode(tok, prompt, answer, max_length):
    ids = [1] + [ord(c) % 29 + 2 for c in answer] + [2]
    return ids, list(range(1, len(answer) + 1)), [(i, i + 1) for i in range(len(answer))]


def samples():
    return [{'id': str(i), 'source_id': str(i), 'task': 'qa', 'split': 'train',
             'prompt': 'A source', 'answer': 'abcde'[:3 + i % 3], 'spans': [(1, 2)]}
            for i in range(8)]


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parents[1] / 'work_test' / ('resume-' + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)
        self.addCleanup(self.clean_scratch)
        self.created = []

        def tiny_load(name, device, fp16_weights=False):
            model = TinyClassifier().to(device)
            self.created.append(model)
            return TinyTokenizer(), model

        self.load_patch = patch.object(encoder, 'load', side_effect=tiny_load)
        self.encode_patch = patch.object(encoder, 'encode', side_effect=tiny_encode)
        self.load_patch.start()
        self.encode_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.encode_patch.stop)
        self.args = dict(name='tiny', train_samples=samples(), dev_samples=samples()[:2],
                         epochs=2, batch_tokens=14, fp16=False, checkpoint_every=1,
                         seed=11, log=lambda message: None)

    def directory(self, name):
        return self.tmp / name

    def clean_scratch(self):
        for path in sorted(self.tmp.rglob('*'), key=lambda path: len(path.parts), reverse=True):
            if not path.resolve().is_relative_to(self.tmp.resolve()):
                raise ValueError('Scratch cleanup escaped its test directory')
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.tmp.rmdir()

    def state(self, name):
        return torch.load(self.directory(name) / 'latest.pt', map_location='cpu', weights_only=False)

    def test_mid_epoch_resume_is_exact_and_completed_run_skips(self):
        save = encoder._atomic_checkpoint

        def capture_full(path, payload):
            save(path, payload)
            if payload['epoch'] == self.args['epochs'] and not payload.get('complete'):
                save(Path(path).parent / 'last-active.pt', payload)

        with patch.object(encoder, '_atomic_checkpoint', side_effect=capture_full):
            _, expected_model, expected_history = encoder.train(
                **self.args, checkpoint_dir=self.directory('baseline'))

        def interrupt(path, payload):
            capture_full(path, payload)
            if payload['epoch'] == 0 and payload['next_batch'] == 2:
                raise RuntimeError('simulated disconnect')

        with patch.object(encoder, '_atomic_checkpoint', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated disconnect'):
                encoder.train(**self.args, checkpoint_dir=self.directory('resumed'))
        with patch.object(encoder, '_atomic_checkpoint', side_effect=capture_full):
            _, actual_model, actual_history = encoder.train(
                **self.args, checkpoint_dir=self.directory('resumed'))
        for key, expected in expected_model.state_dict().items():
            torch.testing.assert_close(actual_model.state_dict()[key], expected, rtol=0, atol=0)
        for expected, actual in zip(expected_history, actual_history):
            for key in ('epoch', 'loss', 'dev_span_f1'):
                self.assertEqual(expected[key], actual[key])
        baseline, resumed = self.state('baseline'), self.state('resumed')
        for key, expected in baseline['best_state'].items():
            torch.testing.assert_close(resumed['best_state'][key], expected, rtol=0, atol=0)
        for state in (baseline, resumed):
            self.assertTrue(state['complete'])
            self.assertEqual(state['epoch'], self.args['epochs'])
            for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'batches'):
                self.assertNotIn(key, state)
        baseline_full, resumed_full = [torch.load(self.directory(name) / 'last-active.pt',
                                                  map_location='cpu', weights_only=False)
                                       for name in ('baseline', 'resumed')]
        for key, expected in baseline_full['model'].items():
            torch.testing.assert_close(resumed_full['model'][key], expected, rtol=0, atol=0)
        self.assertEqual(baseline_full['scheduler'], resumed_full['scheduler'])
        torch.testing.assert_close(baseline_full['rng']['torch'], resumed_full['rng']['torch'], rtol=0, atol=0)
        for parameter, state in baseline_full['optimizer']['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(resumed_full['optimizer']['state'][parameter][key],
                                           value, rtol=0, atol=0)
        self.assertTrue((self.directory('resumed') / 'best' / 'pytorch_model.bin').exists())
        encoder.train(**self.args, checkpoint_dir=self.directory('resumed'))
        self.assertEqual(self.created[-1].training_calls, 0)

    def test_interrupted_dev_repeats_evaluation_without_retraining(self):
        args = dict(self.args, epochs=1)
        with patch.object(encoder, 'predict', side_effect=RuntimeError('evaluation interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'evaluation interrupted'):
                encoder.train(**args, checkpoint_dir=self.directory('dev'))
        state = self.state('dev')
        self.assertEqual(state['next_batch'], len(state['batches']))
        predict = encoder.predict
        budgets = []

        def inspect_budget(*positional, **kwargs):
            budgets.append(kwargs['batch_tokens'])
            return predict(*positional, **kwargs)

        with patch.object(encoder, 'predict', side_effect=inspect_budget):
            encoder.train(**args, checkpoint_dir=self.directory('dev'))
        self.assertEqual(self.created[-1].training_calls, 0)
        self.assertEqual(budgets, [args['batch_tokens']])

    def test_resume_rejects_changed_data_or_settings(self):
        directory = self.directory('identity')
        encoder.train(**self.args, checkpoint_dir=directory)
        with self.assertRaisesRegex(ValueError, 'targets'):
            encoder.train(**dict(self.args, targets='soft'), checkpoint_dir=directory)
        changed = samples()
        changed[0] = dict(changed[0], prompt='Different source with the same row ID')
        with self.assertRaisesRegex(ValueError, 'train'):
            encoder.train(**dict(self.args, train_samples=changed), checkpoint_dir=directory)

    def test_gradient_checkpointing_is_enabled(self):
        _, model, _ = encoder.train(**self.args, gradient_checkpointing=True)
        self.assertTrue(model.checkpointing)

    def test_completed_export_recovery_does_not_restore_optimizer_or_train(self):
        directory = self.directory('export')
        with patch.object(encoder, '_export_best', side_effect=RuntimeError('export interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'export interrupted'):
                encoder.train(**self.args, checkpoint_dir=directory)
        state = self.state('export')
        self.assertTrue(state['complete'])
        with patch.object(torch.optim.AdamW, 'load_state_dict', side_effect=AssertionError('optimizer restored')):
            _, model, history = encoder.train(**self.args, checkpoint_dir=directory)
        self.assertEqual(model.training_calls, 0)
        self.assertEqual(history, state['history'])
        for key, expected in state['best_state'].items():
            torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
        self.assertTrue((directory / 'best' / 'pytorch_model.bin').exists())

    def test_old_full_completed_checkpoint_is_recovered_and_compacted(self):
        directory = self.directory('legacy')
        save = encoder._atomic_checkpoint
        old_path = self.tmp / 'old-complete.pt'

        def capture_complete(path, payload):
            save(path, payload)
            if payload['epoch'] == self.args['epochs'] and not payload.get('complete'):
                save(old_path, payload)

        with patch.object(encoder, '_atomic_checkpoint', side_effect=capture_complete):
            _, expected, expected_history = encoder.train(**self.args, checkpoint_dir=directory)
        old = torch.load(old_path, map_location='cpu', weights_only=False)
        self.assertIn('optimizer', old)
        save(directory / 'latest.pt', old)
        with patch.object(torch.optim.AdamW, 'load_state_dict', side_effect=AssertionError('optimizer restored')):
            _, actual, history = encoder.train(**self.args, checkpoint_dir=directory)
        self.assertEqual(actual.training_calls, 0)
        self.assertEqual(history, expected_history)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
        compact = self.state('legacy')
        self.assertTrue(compact['complete'])
        self.assertNotIn('optimizer', compact)

    def test_resume_cleans_only_recognized_interrupted_save_artifacts(self):
        directory = self.directory('staging')
        encoder.train(**self.args, checkpoint_dir=directory)
        orphan_file = directory / '.checkpoint-interrupted.tmp'
        orphan_file.write_bytes(b'incomplete checkpoint')
        orphan_export = directory / ('.best-' + 'a' * 32)
        orphan_export.mkdir()
        (orphan_export / 'model.safetensors').write_bytes(b'incomplete weights')
        (orphan_export / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
        unfamiliar_export = directory / ('.best-' + 'b' * 32)
        unfamiliar_export.mkdir()
        preserved = unfamiliar_export / 'user-notes.txt'
        preserved.write_text('preserve me', encoding='utf-8')
        ordinary_file = directory / 'unrelated.tmp'
        ordinary_file.write_text('preserve me', encoding='utf-8')
        encoder.train(**self.args, checkpoint_dir=directory)
        self.assertFalse(orphan_file.exists())
        self.assertFalse(orphan_export.exists())
        self.assertTrue(preserved.exists())
        self.assertTrue(ordinary_file.exists())
        self.assertEqual(self.created[-1].training_calls, 0)


if __name__ == '__main__':
    unittest.main()
