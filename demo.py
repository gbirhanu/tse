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
                 max_len=c1.MAX_LEN, models_dir=None):
        self.paths = Paths(None, work_dir, pretrained, models_dir)
        self.device = torch.device(device)
        self.tokenizer = load_tokenizer(pretrained)
        self.max_len = max_len

        seeds = range(len(c1.SEEDS)) if seeds is None else seeds
        self.level1 = {}
        for s in seeds:
            files = [self.paths.model_bin(s, f) for f in range(c1.N_FOLDS)
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
        return answer, np.stack(char_start, 1), np.stack(char_end, 1)

    def extract(self, text, sentiment):
        """Returns (level-one answer, level-one + level-two answer or None)."""
        answer1, cs, ce = self._level1(text, sentiment)
        if not self.level2:
            return answer1, None
        df = pd.DataFrame({'text': [text], 'sentiment': [sentiment], 'selected_text': ['']})
        X = h2.to_object_array(self.char_tok.texts_to_sequences([text]))
        ds = TweetCharDataset(df, X, h2.to_object_array([cs]), h2.to_object_array([ce]),
                              n_models=cs.shape[1], max_len=c2.MAX_LEN, train=False)
        preds = [predict(m, ds, self.device, batch_size=1) for m in self.level2]
        answer2 = h2.string_from_preds_char_level(ds, preds, test=True)[0]
        return answer1, answer2


def show(extractor, text, sentiment):
    a1, a2 = extractor.extract(text, sentiment)
    print(f'\nTweet:      {text}\nSentiment:  {sentiment}')
    print(f'Level one:  {a1}')
    if a2 is not None:
        print(f'Final:      {a2}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--work-dir', default='work')
    p.add_argument('--models-dir', default=None, help='default WORK_DIR/models')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--text', default=None)
    p.add_argument('--sentiment', default=None, choices=['positive', 'negative', 'neutral'])
    args = p.parse_args()

    seeds = args.seeds
    scores_json = Paths(None, args.work_dir).scores_json
    if seeds is None and os.path.exists(scores_json):
        # use the same seeds as level two, if it has been trained
        with open(scores_json) as fh:
            seeds = json.load(fh).get('level2_seeds')
    extractor = Extractor(args.pretrained, args.work_dir, seeds, args.device,
                          models_dir=args.models_dir)
    if args.text:
        show(extractor, args.text, args.sentiment or 'neutral')
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
