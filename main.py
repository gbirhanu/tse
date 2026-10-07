"""
Single entry point for the whole pipeline.

    python main.py all     --data-dir DATA --pretrained ROBERTA_DIR --work-dir work
    python main.py folds   ...   make the 5-fold split for each seed
    python main.py train1  ...   train level one (RoBERTa), saves models + OOF
    python main.py oof1    ...   recompute OOF from saved level-one models
    python main.py test1   ...   level-one predictions on test.csv
    python main.py train2  ...   train level two (character RNN)
    python main.py predict ...   write submission.csv

DATA must hold train.csv and test.csv from the Kaggle competition.
ROBERTA_DIR must hold config.json, vocab.json, merges.txt and the weights,
for example a download of deepset/roberta-base-squad2.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from first_level_model import config as c1          # noqa: E402
from paths import Paths                               # noqa: E402

STEPS = ['folds', 'train1', 'oof1', 'test1', 'train2', 'predict', 'all']


def default_data_dir():
    kaggle = '/kaggle/input/tweet-sentiment-extraction'
    return kaggle if os.path.isdir(kaggle) else 'data'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=STEPS)
    p.add_argument('--data-dir', default=default_data_dir(),
                   help='folder with train.csv and test.csv')
    p.add_argument('--work-dir', default='work', help='folder for all outputs')
    p.add_argument('--pretrained', default=None,
                   help='RoBERTa folder (config.json, vocab.json, merges.txt, weights)')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(len(c1.SEEDS))),
                   help='which of the 5 seeds to use (0-4), default all')
    p.add_argument('--folds', type=int, nargs='+', default=None,
                   help='train only these level-one folds (to split work across sessions)')
    p.add_argument('--epochs', type=int, default=c1.EPOCHS, help='level-one epochs')
    p.add_argument('--batch-size', type=int, default=c1.TRAIN_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=c1.LEARNING_RATE)
    p.add_argument('--max-len', type=int, default=c1.MAX_LEN)
    p.add_argument('--level2-epochs', type=int, default=20)
    p.add_argument('--fp16', action='store_true', help='mixed precision on GPU (faster)')
    p.add_argument('--device', default='auto', help="'cuda', 'cpu' or 'auto'")
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--limit', type=int, default=None,
                   help='use only the first N training rows (quick tests)')
    p.add_argument('--test-limit', type=int, default=None,
                   help='use only the first N test rows (quick tests)')
    p.add_argument('--legacy-target-shift', action='store_true',
                   help='reproduce the 2021 one-character target shift')
    p.add_argument('--overwrite', action='store_true',
                   help='redo steps whose outputs already exist')
    return p.parse_args(argv)


def get_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def main(argv=None):
    args = parse_args(argv)
    if args.legacy_target_shift:
        c1.LEGACY_TARGET_SHIFT = True
    paths = Paths(args.data_dir, args.work_dir, args.pretrained)
    paths.make_dirs()
    device = get_device(args.device)
    print(f'Device: {device} | seeds: {args.seeds} | work dir: {args.work_dir}')

    for f in (paths.train_csv, paths.test_csv):
        if not os.path.exists(f):
            sys.exit(f'Missing {f}. Put the Kaggle competition files in --data-dir.')
    needs_roberta = args.step in ('train1', 'oof1', 'test1', 'all')
    if needs_roberta and not args.pretrained:
        sys.exit('This step needs --pretrained (a RoBERTa folder).')

    tokenizer = None
    if needs_roberta:
        from first_level_model.dataset import load_tokenizer
        tokenizer = load_tokenizer(args.pretrained)

    step = args.step
    if step in ('folds', 'all'):
        from first_level_model.run import ensure_folds
        for s in args.seeds:
            ensure_folds(s, args, paths)

    if step in ('train1', 'all'):
        from first_level_model.run import run_different_seed
        scores = {}
        for s in args.seeds:
            fold_scores = run_different_seed(s, args, paths, tokenizer, device, args.folds)
            scores[f'level1_seed{s}'] = {str(k): v for k, v in fold_scores.items()}
        _save_scores(paths, scores)

    if step == 'oof1':
        from second_level_model.run_oof import run_oof
        for s in args.seeds:
            run_oof(s, args, paths, tokenizer, device)

    if step in ('test1', 'all'):
        from first_level_model.lo_inference import run_inference as level1_test
        for s in args.seeds:
            level1_test(s, args, paths, tokenizer, device)

    if step in ('train2', 'all'):
        from second_level_model.lt_run import run_train as level2_train
        level2_train(args, paths, device)

    if step in ('predict', 'all'):
        from second_level_model.final_inference import run_inference as final_predict
        final_predict(args, paths, device)


def _save_scores(paths, new):
    scores = {}
    if os.path.exists(paths.scores_json):
        with open(paths.scores_json) as fh:
            scores = json.load(fh)
    scores.update(new)
    with open(paths.scores_json, 'w') as fh:
        json.dump(scores, fh, indent=2)


if __name__ == '__main__':
    main()
