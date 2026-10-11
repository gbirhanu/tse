"""
Local demo: give a tweet and its sentiment, get the phrase that carries the sentiment.

    python demo.py --pretrained ROBERTA_DIR --work-dir work
    python demo.py --pretrained ROBERTA_DIR --work-dir work \
                   --text "my boss is bullying me..." --sentiment negative

Only config.json, vocab.json and merges.txt are read from ROBERTA_DIR. The
weights come from work/models/model_XY.bin (level one) and work/level2/
(level two). Any number of level-one models works: one fold of one seed is
enough for a light demo; all 25 give the competition setup.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from first_level_model import config as c1, helper as h1             # noqa: E402
from first_level_model.dataset import TweetDataset, load_tokenizer   # noqa: E402
from first_level_model.model import load_fold_model                  # noqa: E402
from paths import Paths                                               # noqa: E402
from second_level_model import config as c2, helper as h2            # noqa: E402
from second_level_model.dataset import TweetCharDataset              # noqa: E402
from second_level_model.train_routine import new_model, predict      # noqa: E402


class Extractor:
    def __init__(self, pretrained, work_dir, seeds=None, device='cpu',
                 max_len=c1.MAX_LEN, models_dir=None, folds=None, data_dir=None):
        self.paths = Paths(None, work_dir, pretrained, models_dir)
        self.device = torch.device(device)
        self.tokenizer = load_tokenizer(pretrained)
        self.max_len = max_len

        seeds = range(len(c1.SEEDS)) if seeds is None else seeds
        folds = range(c1.N_FOLDS) if folds is None else folds
        self.level1 = {}
        for s in seeds:
            files = [self.paths.model_bin(s, f) for f in folds
                     if os.path.exists(self.paths.model_bin(s, f))]
            if files:
                self.level1[s] = [load_fold_model(pretrained, f, self.device) for f in files]
        if not self.level1:
            sys.exit(f'No level-one models found in {self.paths.models_dir}')
        n1 = sum(len(m) for m in self.level1.values())
        print(f'Level one: {n1} model(s) from seed(s) {list(self.level1)}')

        self.level2 = []
        if os.path.exists(self.paths.char_vocab_json):
            self.char_tok = h2.CharTokenizer.load(self.paths.char_vocab_json)
            for k in range(c2.N_FOLDS):
                path = self.paths.level2_model(k)
                if os.path.exists(path):
                    m = new_model(len(self.char_tok), self.device)
                    h2.load_model_weights(m, path, verbose=0)
                    m.eval()
                    self.level2.append(m)
        print(f'Level two: {len(self.level2)} model(s)')

        scores = {}
        if os.path.exists(self.paths.scores_json):
            with open(self.paths.scores_json) as fh:
                scores = json.load(fh)
        if self.level2 and 'level2_confidence_table' not in scores:
            scores = build_score_table(work_dir, data_dir, scores, self.device) or scores
        self.confidence_table = scores.get('level2_confidence_table') if self.level2 else None
        self.info = {'seeds': list(self.level1), 'level1_models': n1,
                     'level2_models': len(self.level2), 'device': str(self.device),
                     'validation': validation_scores(scores, list(self.level1))}

    def _level1(self, text, sentiment):
        item = TweetDataset([text], [sentiment], [text], self.tokenizer, self.max_len)[0]
        ids = item['ids'][None].to(self.device)
        mask = item['mask'][None].to(self.device)
        tt = item['token_type_ids'][None].to(self.device)
        offsets = item['offsets'].numpy()

        seed_start, seed_end, char_start, char_end = [], [], [], []
        with torch.no_grad():
            for models in self.level1.values():
                s_sum, e_sum = 0, 0
                for m in models:
                    s, e = m(ids=ids, mask=mask, token_type_ids=tt)
                    s_sum, e_sum = s_sum + s, e_sum + e
                s_avg, e_avg = s_sum / len(models), e_sum / len(models)
                seed_start.append(s_avg)
                seed_end.append(e_avg)
                char_start.append(h1.token_level_to_char_level(
                    text, offsets, torch.softmax(s_avg, -1)[0].cpu().numpy()))
                char_end.append(h1.token_level_to_char_level(
                    text, offsets, torch.softmax(e_avg, -1)[0].cpu().numpy()))

        start_logits = torch.stack(seed_start).mean(0)[0].cpu().numpy()
        end_logits = torch.stack(seed_end).mean(0)[0].cpu().numpy()
        answer = h1.decode_selected_text(text, start_logits, end_logits,
                                         item['orig_start'], item['orig_end'], offsets)
        s_idx, e_idx = h1.get_best_start_end_idx(start_logits, end_logits,
                                                 item['orig_start'], item['orig_end'])
        confidence = float(_softmax(start_logits)[s_idx] * _softmax(end_logits)[e_idx])
        return answer, confidence, np.stack(char_start, 1), np.stack(char_end, 1)

    def extract(self, text, sentiment):
        """
        Returns a dict:
          selected_text      the answer (level two if loaded, else level one)
          level1_text        the answer of the RoBERTa models alone
          level2_used        True if level two made the answer
          confidence         P(start) * P(end) of the answer, 0..1 (not an accuracy)
          estimated_jaccard  mean Jaccard of held-out tweets with the same sentiment
                             and a similar confidence (level two only), or None
        """
        answer1, conf1, cs, ce = self._level1(text, sentiment)
        out = {'selected_text': answer1, 'level1_text': answer1, 'level2_used': False,
               'confidence': conf1, 'estimated_jaccard': None}
        if not self.level2:
            return out
        df = pd.DataFrame({'text': [text], 'sentiment': [sentiment], 'selected_text': ['']})
        X = h2.to_object_array(self.char_tok.texts_to_sequences([text]))
        ds = TweetCharDataset(df, X, h2.to_object_array([cs]), h2.to_object_array([ce]),
                              n_models=cs.shape[1], max_len=c2.MAX_LEN, train=False)
        preds = [predict(m, ds, self.device, batch_size=1) for m in self.level2]
        texts, confs = h2.spans_from_preds(ds, preds, test=True)
        out.update(selected_text=texts[0], level2_used=True, confidence=confs[0],
                   estimated_jaccard=h2.estimate_jaccard(self.confidence_table,
                                                         sentiment, confs[0]))
        return out


def build_score_table(work_dir, data_dir, scores, device):
    """
    Runs made before the confidence table existed have none in scores.json.
    Builds it once from the saved level-two models (same as main.py score2;
    no training). Needs train.csv: DATA_DIR, or a 'data' folder next to WORK_DIR.
    Returns the new scores, or None if it could not be built.
    """
    from types import SimpleNamespace
    from second_level_model.lt_run import rebuild_scores

    candidates = [data_dir] if data_dir else [
        os.path.join(os.path.dirname(os.path.abspath(work_dir)), 'data'), 'data']
    found = [d for d in candidates if os.path.exists(os.path.join(d, 'train.csv'))]
    seeds = scores.get('level2_seeds')
    if not found or not seeds:
        print('No score table for the estimated Jaccard (train.csv or level2_seeds not found). '
              'Run: python main.py score2 --data-dir DATA --work-dir WORK')
        return None
    print('Building the score table from the saved level-two models (once, no training)...')
    try:
        rebuild_scores(SimpleNamespace(limit=None, seeds=seeds),
                       Paths(found[0], work_dir), device)
    except Exception as err:   # the page still works, only without the estimate
        print(f'Could not build the score table: {err}')
        return None
    with open(Paths(None, work_dir).scores_json) as fh:
        return json.load(fh)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def validation_scores(scores, seeds):
    """Jaccard on held-out training tweets, from scores.json."""
    level1 = [v for s in seeds for v in scores.get(f'level1_seed{s}', {}).values()]
    return {'level1_jaccard': float(np.mean(level1)) if level1 else None,
            'level2_jaccard': scores.get('level2_cv_jaccard'),
            'level2_by_sentiment': scores.get('level2_by_sentiment')}


def level2_seeds(work_dir):
    """The seeds level two was trained on (from scores.json), or None."""
    path = Paths(None, work_dir).scores_json
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh).get('level2_seeds')
    return None


def show(extractor, text, sentiment, answer=None):
    r = extractor.extract(text, sentiment)
    print(f'\nTweet:      {text}\nSentiment:  {sentiment}')
    print(f'Level one:  {r["level1_text"]}')
    if r['level2_used']:
        print(f'Final:      {r["selected_text"]}')
    line = f'Confidence: {r["confidence"]:.2f}'
    if r['estimated_jaccard'] is not None:
        line += f'   estimated Jaccard: {r["estimated_jaccard"]:.2f}'
    print(line)
    if answer:
        print(f'Jaccard with your answer: {h2.jaccard(r["selected_text"], answer):.2f}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--work-dir', default='work')
    p.add_argument('--models-dir', default=None, help='default WORK_DIR/models')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--folds', type=int, nargs='+', default=None,
                   help='load only these level-one folds (fewer = faster)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--text', default=None)
    p.add_argument('--sentiment', default=None, choices=['positive', 'negative', 'neutral'])
    p.add_argument('--answer', default=None,
                   help='the correct phrase, if you know it: prints its Jaccard score')
    p.add_argument('--data-dir', default=None,
                   help='folder with train.csv, used once to build the score table '
                        '(default: a data folder next to WORK_DIR)')
    args = p.parse_args()

    seeds = args.seeds if args.seeds is not None else level2_seeds(args.work_dir)
    extractor = Extractor(args.pretrained, args.work_dir, seeds, args.device,
                          models_dir=args.models_dir, folds=args.folds, data_dir=args.data_dir)
    v = extractor.info['validation']
    best = v['level2_jaccard'] if extractor.level2 else v['level1_jaccard']
    if best is not None:
        print(f'Jaccard on held-out training tweets: {best:.4f}')
    if args.text:
        show(extractor, args.text, args.sentiment or 'neutral', args.answer)
        return
    while True:
        text = input('\nTweet (empty line to quit): ').strip()
        if not text:
            break
        sentiment = input('Sentiment [positive/negative/neutral]: ').strip().lower()
        if sentiment not in ('positive', 'negative', 'neutral'):
            print('Please type positive, negative or neutral.')
            continue
        show(extractor, text, sentiment)


if __name__ == '__main__':
    main()
