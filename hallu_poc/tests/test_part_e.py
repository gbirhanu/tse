"""Part E checks on CPU: weight averaging, NLI starting point, synthetic summaries."""
from pathlib import Path
import re
import shutil
import sys
import unittest
import uuid

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (ModernBertConfig, ModernBertForSequenceClassification,
                          ModernBertForTokenClassification, PreTrainedTokenizerFast)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallu import encoder, soup, synth


def tiny_tokenizer():
    words = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]', 'a', 'b', 'c']
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', cls_token='[CLS]',
                                   sep_token='[SEP]', pad_token='[PAD]', mask_token='[MASK]')


def tiny_config(num_labels=2):
    return ModernBertConfig(vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, max_position_embeddings=64, pad_token_id=3,
                            cls_token_id=1, sep_token_id=2, num_labels=num_labels)


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parents[1] / 'work_test' / ('part-e-' + uuid.uuid4().hex)
        self.tmp.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def save(self, model, name):
        directory = self.tmp / name
        model.save_pretrained(directory)
        tiny_tokenizer().save_pretrained(directory)
        return directory


class SoupTest(Scratch):
    def test_average_is_the_mean_of_member_weights(self):
        dirs, states = [], []
        for seed in (1, 2, 3):
            torch.manual_seed(seed)
            model = ModernBertForTokenClassification(tiny_config())
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            dirs.append(self.save(model, f'm{seed}'))
        _, averaged = soup.average_models(dirs, log=lambda _: None)
        for key, value in averaged.state_dict().items():
            expected = sum(state[key].float() for state in states) / 3
            torch.testing.assert_close(value.float(), expected)

    def test_different_shapes_are_refused(self):
        a = self.save(ModernBertForTokenClassification(tiny_config()), 'a')
        wider = tiny_config()
        wider.intermediate_size = 48
        b = self.save(ModernBertForTokenClassification(wider), 'b')
        with self.assertRaises(ValueError):
            soup.average_models([a, b], log=lambda _: None)


class NliStartTest(Scratch):
    def test_nli_checkpoint_keeps_encoder_and_gets_a_two_class_layer(self):
        config = tiny_config(num_labels=3)
        nli = ModernBertForSequenceClassification(config)
        directory = self.save(nli, 'nli')
        _, model = encoder.load(directory, 'cpu', new_head=True)
        self.assertEqual(model.classifier.out_features, 2)
        saved = nli.state_dict()
        for key, value in model.state_dict().items():
            if key.startswith(('model.', 'head.')):
                torch.testing.assert_close(value, saved[key])

    def test_two_class_checkpoint_loads_unchanged(self):
        torch.manual_seed(0)
        base = ModernBertForTokenClassification(tiny_config())
        directory = self.save(base, 'base')
        _, plain = encoder.load(directory, 'cpu')
        _, flagged = encoder.load(directory, 'cpu', new_head=True)
        self.assertEqual(plain.config.to_dict(), flagged.config.to_dict())
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, flagged.state_dict()[key])


ARTICLE = ('Summarize the following news within 60 words:\n'
           'John Smith visited Paris on Monday, March 3, 2015, with 1,200 people. '
           'Mary Jones stayed in London for 12 days.\n\noutput:')


def fake_ner(texts):
    labels = {'John Smith': 'PERSON', 'Mary Jones': 'PERSON', 'Paris': 'GPE', 'London': 'GPE'}
    return [[(m.start(), m.end(), labels[m.group()])
             for m in re.finditer('|'.join(labels), text)] for text in texts]


def sample(i, answer, spans=(), source='s1', split='train'):
    return {'id': str(i), 'source_id': source, 'task': 'summary', 'split': split,
            'prompt': ARTICLE, 'answer': answer, 'spans': list(spans)}


class SynthTest(unittest.TestCase):
    def data(self):
        moved = 'The visit was cancelled by the city council.'
        sibling = 'Mary Jones stayed in London. ' + moved
        clean = [sample(i, 'John Smith visited Paris on Monday with 1,200 people in 2015. '
                           'Mary Jones stayed in London for 12 days.') for i in range(1, 7)]
        return clean + [sample(9, sibling, [(sibling.index(moved), len(sibling))])], moved

    def test_spans_mark_exactly_the_changed_text(self):
        train, moved = self.data()
        extra, stats = synth.make(train, seed=0, ner=fake_ner)
        self.assertEqual(stats['made'], 6)
        originals = {s['id']: s['answer'] for s in train}
        for s in extra:
            self.assertNotEqual(s['answer'], originals[s['synth']['from']])
            for (a, b), edit in zip(s['spans'], s['synth']['edits']):
                self.assertEqual(s['answer'][a:b], edit['new'])
                if edit['type'] == 'move':
                    self.assertEqual(edit['new'], moved)
                elif edit['type'] == 'name':
                    self.assertIn(edit['new'], ('John Smith', 'Mary Jones', 'Paris', 'London'))
        self.assertEqual(extra, synth.make(train, seed=0, ner=fake_ner)[0])

    def test_only_error_free_training_summaries_are_used(self):
        train, _ = self.data()
        with self.assertRaises(ValueError):
            synth.make(train + [sample(20, 'x', split='test')], seed=0, ner=fake_ner)

    def test_extra_answers_from_dev_or_test_sources_are_refused(self):
        train, _ = self.data()
        extra, _ = synth.make(train, seed=0, ner=fake_ner)
        dev = [sample(30, 'y', source='s1')]
        with self.assertRaises(ValueError):
            synth.check_extra(extra, train, dev, [])
        checked = synth.check_extra(extra, train, [], [])
        self.assertTrue(all(isinstance(span, tuple) for s in checked for span in s['spans']))

    def test_lead_in_line_is_never_edited(self):
        text = "Sure! Here's a summary in 49 words:\n\nJohn Smith visited Paris in 2015."
        candidates = synth._swap_candidates(text, synth.article_text(ARTICLE),
                                            fake_ner([text])[0],
                                            fake_ner([synth.article_text(ARTICLE)])[0],
                                            __import__('random').Random(0))
        self.assertTrue(candidates)
        self.assertTrue(all(c[1] > text.index('\n') for c in candidates))


if __name__ == '__main__':
    unittest.main()
