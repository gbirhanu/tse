"""
Recomputes the out-of-fold char probabilities from saved level-one models.
Only needed when the models were trained elsewhere (for example the 2021
models on Google Drive); `main.py train1` already writes these files.
"""
import pickle

import pandas as pd

from first_level_model import config
from first_level_model.model import load_fold_model
from first_level_model.run import merge_oof
from first_level_model.run_train import make_loader
from first_level_model.train_routine_helper import eval_fn


def run_oof_pred(seed_idx, fold, args, paths, tokenizer, device):
    dfx = pd.read_csv(paths.folds_csv(seed_idx))
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    loader = make_loader(df_valid, tokenizer, args.max_len,
                         config.VALID_BATCH_SIZE, args.num_workers, False)
    model = load_fold_model(paths.pretrained, paths.model_bin(seed_idx, fold), device)
    jaccard, char_start, char_end = eval_fn(loader, model, device, fp16=args.fp16)
    with open(paths.oof_fold_pkl(seed_idx, fold), 'wb') as handle:
        pickle.dump({'textID': list(df_valid.textID.values),
                     'start': char_start, 'end': char_end,
                     'jaccard': jaccard}, handle)
    return jaccard


def run_oof(seed_idx, args, paths, tokenizer, device):
    for fold in range(config.N_FOLDS):
        run_oof_pred(seed_idx, fold, args, paths, tokenizer, device)
    return merge_oof(seed_idx, paths)
