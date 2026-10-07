import pickle

import pandas as pd
import torch

from first_level_model import config, dataset, helper, train_routine_helper
from first_level_model.model import TweetModel


def make_loader(df, tokenizer, max_len, batch_size, num_workers, shuffle):
    ds = dataset.TweetDataset(tweets=df.text.values,
                              sentiments=df.sentiment.values,
                              selected_texts=df.selected_text.values,
                              tokenizer=tokenizer, max_len=max_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                       num_workers=num_workers, shuffle=shuffle)


def run_train(fold, folds_csv, seed_idx, args, paths, tokenizer, device):
    """Trains one fold, saves the model and its out-of-fold char probabilities."""
    dfx = pd.read_csv(folds_csv)
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_data_loader = make_loader(df_train, tokenizer, args.max_len,
                                    args.batch_size, args.num_workers, True)
    valid_data_loader = make_loader(df_valid, tokenizer, args.max_len,
                                    config.VALID_BATCH_SIZE, args.num_workers, False)

    model = TweetModel(paths.pretrained).to(device)

    num_train_steps = int(len(df_train) / args.batch_size * args.epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': config.WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr,
                                  eps=config.ADAM_EPS)
    scheduler = helper.linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * config.WARMUP_RATIO),
        num_training_steps=num_train_steps)
    scaler = None
    if args.fp16 and device.type == 'cuda':
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):   # torch older than 2.3
            scaler = torch.cuda.amp.GradScaler()

    print(f'Training is starting for seed={seed_idx} and fold={fold} '
          f'({len(df_train)} train, {len(df_valid)} valid)')

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_routine_helper.train_fn(train_data_loader, model, optimizer, device,
                                      scheduler=scheduler, fp16=args.fp16,
                                      scaler=scaler)
        jaccard, char_start, char_end = train_routine_helper.eval_fn(
            valid_data_loader, model, device, fp16=args.fp16)

    torch.save(model.state_dict(), paths.model_bin(seed_idx, fold))
    with open(paths.oof_fold_pkl(seed_idx, fold), 'wb') as handle:
        pickle.dump({'textID': list(df_valid.textID.values),
                     'start': char_start, 'end': char_end,
                     'jaccard': jaccard}, handle)

    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return jaccard
