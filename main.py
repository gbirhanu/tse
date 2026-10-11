"""
Single entry point for the whole pipeline.

    python main.py all     --data-dir DATA --pretrained ROBERTA_DIR --work-dir work
    python main.py folds   ...   make the 5-fold split for each seed
    python main.py train1  ...   train level one (RoBERTa), saves models + OOF
    python main.py oof1    ...   recompute OOF from saved level-one models
    python main.py test1   ...   level-one predictions on test.csv
    python main.py train2  ...   train level two (character RNN)
    python main.py score2  ...   level-two scores from the saved models (no training)
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

STEPS = ['folds', 'train1', 'oof1', 'test1', 'train2', 'score2', 'predict', 'all']


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
    p.add_argument('--models-dir', default=None,
                   help='where level-one models go (default WORK_DIR/models); on Colab '
                        'this can be local disk to save Drive space')
    p.add_argument('--pretrained', default=None,
                   help='RoBERTa folder (config.json, vocab.json, merges.txt, weights)')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='which of the 5 seeds to use (0-4). Default: all five for level one; '
                        'for train2 and predict, the seeds whose level-one outputs are ready')
    p.add_argument('--folds', type=int, nargs='+', default=None,
                   help='train only these level-one folds (to split work across sessions)')
    p.add_argument('--epochs', type=int, default=c1.EPOCHS, help='level-one epochs')
    p.add_argument('--batch-size', type=int, default=c1.TRAIN_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=c1.LEARNING_RATE)
    p.add_argument('--max-len', type=int, default=c1.MAX_LEN)
    p.add_argument('--level2-epochs', type=int, default=20)
    p.add_argument('--fp16', action='store_true', help='mixed precision on GPU (faster)')
    p.add_argument('--save-fp16-weights', action='store_true',
                   help='save level-one models in half precision (half the disk space)')
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
    paths = Paths(args.data_dir, args.work_dir, args.pretrained, args.models_dir)
    paths.make_dirs()
    device = get_device(args.device)
    if args.step in ('train2', 'score2', 'predict'):
        args.seeds = pick_level2_seeds(args, paths)
    elif args.seeds is None:
        args.seeds = list(range(len(c1.SEEDS)))
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

    if step == 'score2':
        from second_level_model.lt_run import rebuild_scores
        rebuild_scores(args, paths, device)

    if step in ('predict', 'all'):
        from second_level_model.final_inference import run_inference as final_predict
        final_predict(args, paths, device)


def ready_seeds(paths):
    """Seeds whose level-one OOF (all 5 folds) and test predictions both exist."""
    return [s for s in range(len(c1.SEEDS))
            if os.path.exists(paths.oof_pkl(s)) and os.path.exists(paths.test_pkl(s))]


def pick_level2_seeds(args, paths):
    """Level two uses the seeds that are ready, unless --seeds says otherwise."""
    ready = ready_seeds(paths)
    trained_with = _load_scores(paths).get('level2_seeds')
    if args.seeds is not None:
        not_ready = [s for s in args.seeds if s not in ready]
        if not_ready:
            sys.exit(f'Seed(s) {not_ready} have not finished train1 and test1 yet. '
                     f'Ready: {ready or "none"}. Leave out --seeds to use the ready ones.')
        seeds = args.seeds
    elif args.step in ('predict', 'score2') and trained_with:
        seeds = trained_with      # the seeds level two was trained on
    else:
        if not ready:
            sys.exit('No seed has finished level one yet. Run train1 and test1 first.')
        seeds = ready
        print(f'Using the seeds that have finished level one: {seeds}')
    if args.step == 'predict' and trained_with and sorted(seeds) != sorted(trained_with):
        print(f'Note: level two was trained on seeds {trained_with}, '
              f'predicting with {seeds}. Run train2 again to match.')
    return seeds


def _load_scores(paths):
    if os.path.exists(paths.scores_json):
        with open(paths.scores_json) as fh:
            return json.load(fh)
    return {}


def _save_scores(paths, new):
    scores = _load_scores(paths)
    scores.update(new)
    with open(paths.scores_json, 'w') as fh:
        json.dump(scores, fh, indent=2)


if __name__ == '__main__':
    main()
