"""Trains the level-two (character RNN) model on the level-one OOF predictions."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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

    return save_scores(df_train, X_train, preds, pred_oof, seeds, paths)


def rebuild_scores(args, paths, device):
    """
    The same scores as train2, from the saved level-two models, without training:
    each fold model predicts the tweets it did not see. Fast; for runs made
    before the confidence table was added.
    """
    df_train = read_train(paths, args.limit)
    seeds = args.seeds
    oof_start, oof_end = helper.load_char_preds(paths, seeds, 'oof', list(df_train.textID))
    tokenizer = helper.CharTokenizer.load(paths.char_vocab_json)
    X_train = helper.to_object_array(tokenizer.texts_to_sequences(df_train['text'].values))
    preds = {'oof_start': helper.to_object_array(oof_start),
             'oof_end': helper.to_object_array(oof_end)}

    splits = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                             random_state=config.FOLD_SEED).split(X=df_train, y=df_train['sentiment'])
    pred_oof = [None] * len(df_train)
    for i, (_, val_idx) in enumerate(splits):
        model = train_routine.new_model(len(tokenizer), device)
        helper.load_model_weights(model, paths.level2_model(i), verbose=0)
        val_dataset = dataset.TweetCharDataset(df_train.iloc[val_idx], X_train[val_idx],
                                               preds['oof_start'][val_idx],
                                               preds['oof_end'][val_idx],
                                               max_len=config.MAX_LEN, n_models=len(seeds))
        start, end = train_routine.predict(model, val_dataset, device,
                                           batch_size=config.VALID_BATCH_SIZE)
        for j, idx in enumerate(val_idx):
            pred_oof[idx] = [start[j], end[j]]
        print(f'Fold {i + 1}/{config.N_FOLDS} done')

    return save_scores(df_train, X_train, preds, pred_oof, seeds, paths)


def save_scores(df_train, X_train, preds, pred_oof, seeds, paths):
    """Level-two CV Jaccard, per sentiment, and the confidence table, into scores.json."""
    train_dataset = dataset.TweetCharDataset(df_train, X_train,
                                             preds['oof_start'], preds['oof_end'],
                                             max_len=config.MAX_LEN, train=True,
                                             n_models=len(seeds))
    selected_texts_oof, confidences = helper.spans_from_preds(
        train_dataset, pred_oof, test=False, remove_neutral=config.REMOVE_NEUTRAL)
    row_scores = np.array([helper.jaccard(pred, truth) for pred, truth in
                           zip(selected_texts_oof, df_train['selected_text'])])
    score = float(row_scores.mean())
    print(f'Local CV score of level two (all OOF rows): {score:.4f}')
    by_sentiment = {s: float(row_scores[(df_train['sentiment'] == s).values].mean())
                    for s in sorted(df_train['sentiment'].unique())}
    print('By sentiment: ' + ', '.join(f'{s} {v:.4f}' for s, v in by_sentiment.items()))

    scores = {}
    if os.path.exists(paths.scores_json):
        with open(paths.scores_json) as fh:
            scores = json.load(fh)
    scores['level2_cv_jaccard'] = score
    scores['level2_by_sentiment'] = by_sentiment
    # used by demo.py / serve.py to estimate the Jaccard of a new tweet
    scores['level2_confidence_table'] = helper.confidence_table(
        df_train['sentiment'].values, confidences, row_scores)
    scores['level2_seeds'] = list(seeds)
    with open(paths.scores_json, 'w') as fh:
        json.dump(scores, fh, indent=2)
    return score
