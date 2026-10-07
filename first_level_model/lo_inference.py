"""Level-one predictions on test.csv: the 5 fold models of a seed, logits averaged."""
import pickle

import pandas as pd
import torch

from first_level_model import config, dataset, train_routine_helper
from first_level_model.model import load_fold_model


def read_test(paths, test_limit=None):
    df_test = pd.read_csv(paths.test_csv).fillna('')
    if test_limit:
        df_test = df_test.head(test_limit)
    return df_test


def run_inference(seed_idx, args, paths, tokenizer, device):
    df_test = read_test(paths, args.test_limit)
    df_test.loc[:, 'selected_text'] = df_test.text.values

    fold_models = [load_fold_model(paths.pretrained, paths.model_bin(seed_idx, f), device)
                   for f in range(config.N_FOLDS)]

    test_dataset = dataset.TweetDataset(tweets=df_test.text.values,
                                        sentiments=df_test.sentiment.values,
                                        selected_texts=df_test.selected_text.values,
                                        tokenizer=tokenizer, max_len=args.max_len)
    data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                              batch_size=config.VALID_BATCH_SIZE,
                                              num_workers=args.num_workers)

    char_start, char_end = train_routine_helper.predict_logits(
        data_loader, fold_models, device, fp16=args.fp16)

    with open(paths.test_pkl(seed_idx), 'wb') as handle:
        pickle.dump({'textID': list(df_test.textID.values),
                     'start': char_start, 'end': char_end}, handle)
    print(f'Test predictions of seed {seed_idx}: {paths.test_pkl(seed_idx)}')
