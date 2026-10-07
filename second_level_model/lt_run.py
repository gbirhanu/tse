"""Trains the level-two (character RNN) model on the level-one OOF predictions."""
import json
import os

import numpy as np
import pandas as pd

from first_level_model.lo_inference import read_test
from second_level_model import config, dataset, helper, train_routine


def read_train(paths, limit=None):
    df_train = pd.read_csv(paths.train_csv).dropna().reset_index(drop=True)
    if limit:
        df_train = df_train.head(limit)
    return df_train


def run_train(args, paths, device):
    df_train = read_train(paths, args.limit)
    df_test = read_test(paths, args.test_limit)
    df_test['selected_text'] = ''
    seeds = args.seeds

    oof_start, oof_end = helper.load_char_preds(paths, seeds, 'oof', list(df_train.textID))
    test_start, test_end = helper.load_char_preds(paths, seeds, 'test', list(df_test.textID))

    tokenizer = helper.CharTokenizer()
    tokenizer.fit_on_texts(df_train['text'].values)
    tokenizer.save(paths.char_vocab_json)
    len_voc = len(tokenizer)

    X_train = helper.to_object_array(tokenizer.texts_to_sequences(df_train['text'].values))
    X_test = helper.to_object_array(tokenizer.texts_to_sequences(df_test['text'].values))
    preds = {'test_start': helper.to_object_array(test_start),
             'test_end': helper.to_object_array(test_end),
             'oof_start': helper.to_object_array(oof_start),
             'oof_end': helper.to_object_array(oof_end)}

    pred_oof, pred_tests, fold_avg = train_routine.k_fold(
        df_train, df_test, X_train, X_test, preds, len_voc, len(seeds), paths, device,
        k=config.N_FOLDS, fold_seed=config.FOLD_SEED, model_seed=config.MODEL_SEED,
        epochs=args.level2_epochs, num_workers=args.num_workers)

    train_dataset = dataset.TweetCharDataset(df_train, X_train,
                                             preds['oof_start'], preds['oof_end'],
                                             max_len=config.MAX_LEN, train=True,
                                             n_models=len(seeds))
    selected_texts_oof = helper.string_from_preds_char_level(
        train_dataset, pred_oof, test=False, remove_neutral=config.REMOVE_NEUTRAL)
    score = float(np.mean([helper.jaccard(pred, truth) for pred, truth in
                           zip(selected_texts_oof, df_train['selected_text'])]))
    print(f'Local CV score of level two (all OOF rows): {score:.4f}')

    scores = {}
    if os.path.exists(paths.scores_json):
        with open(paths.scores_json) as fh:
            scores = json.load(fh)
    scores['level2_cv_jaccard'] = score
    scores['level2_seeds'] = list(seeds)
    with open(paths.scores_json, 'w') as fh:
        json.dump(scores, fh, indent=2)
    return score
