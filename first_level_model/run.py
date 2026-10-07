import os
import pickle

import numpy as np

from create_fold import create_fold
from first_level_model import config, helper
from first_level_model.run_train import run_train


def ensure_folds(seed_idx, args, paths):
    folds_csv = paths.folds_csv(seed_idx)
    if not os.path.exists(folds_csv) or args.overwrite:
        create_fold(paths.train_csv, folds_csv, config.SEEDS[seed_idx],
                    n_splits=config.N_FOLDS, limit=args.limit)
    return folds_csv


def merge_oof(seed_idx, paths):
    """Joins the 5 per-fold OOF files of one seed into one file."""
    merged = {'textID': [], 'start': [], 'end': []}
    for fold in range(config.N_FOLDS):
        with open(paths.oof_fold_pkl(seed_idx, fold), 'rb') as handle:
            part = pickle.load(handle)
        for key in merged:
            merged[key].extend(part[key])
    with open(paths.oof_pkl(seed_idx), 'wb') as handle:
        pickle.dump(merged, handle)
    return paths.oof_pkl(seed_idx)


def run_different_seed(seed_idx, args, paths, tokenizer, device, folds=None):
    """Trains the 5 folds of one seed. Folds already done are skipped."""
    folds_csv = ensure_folds(seed_idx, args, paths)
    folds = list(range(config.N_FOLDS)) if folds is None else folds
    fold_scores = {}
    for fold in folds:
        done = (os.path.exists(paths.model_bin(seed_idx, fold)) and
                os.path.exists(paths.oof_fold_pkl(seed_idx, fold)))
        if done and not args.overwrite:
            with open(paths.oof_fold_pkl(seed_idx, fold), 'rb') as handle:
                fold_scores[fold] = pickle.load(handle)['jaccard']
            print(f'Seed {seed_idx} fold {fold} already trained, skipping')
            continue
        # seeding per fold, so an interrupted run can resume the same way
        helper.seed_everything(config.SEEDS[seed_idx] + fold)
        fold_scores[fold] = run_train(fold, folds_csv, seed_idx, args, paths,
                                      tokenizer, device)

    for fold, score in fold_scores.items():
        print(f'Seed={seed_idx} Fold={fold}, Jaccard = {score:.5f}')
    if len(fold_scores) == config.N_FOLDS:
        print(f'Seed {seed_idx}: mean = {np.mean(list(fold_scores.values())):.5f}, '
              f'std = {np.std(list(fold_scores.values())):.5f}')
        merge_oof(seed_idx, paths)
    return fold_scores
