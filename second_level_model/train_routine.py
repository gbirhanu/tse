import gc
import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from first_level_model import helper as h
from second_level_model import config, dataset, helper
from second_level_model import model as md


def ce_loss(pred, truth, smoothing=False, trg_pad_idx=-1, eps=0.1):
    """Cross entropy with label smoothing (https://arxiv.org/pdf/1701.06548.pdf)."""
    truth = truth.contiguous().view(-1)
    one_hot = torch.zeros_like(pred).scatter(1, truth.view(-1, 1), 1)
    if smoothing:
        n_class = pred.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    loss = -one_hot * F.log_softmax(pred, dim=1)
    if trg_pad_idx >= 0:
        loss = loss.sum(dim=1)
        non_pad_mask = truth.ne(trg_pad_idx)
        loss = loss.masked_select(non_pad_mask)
    return loss.sum()


def loss_fn(start_logits, end_logits, start_positions, end_positions, config):
    bs = start_logits.size(0)
    start_loss = ce_loss(start_logits, start_positions,
                         smoothing=config['smoothing'], eps=config['eps'])
    end_loss = ce_loss(end_logits, end_positions,
                       smoothing=config['smoothing'], eps=config['eps'])
    return (start_loss + end_loss) / bs


def new_model(len_voc, device):
    return md.TweetCharModel(len_voc,
                             use_msd=config.USE_MSD,
                             lstm_dim=config.LSTM_DIM,
                             ft_lstm_dim=config.FT_LSTM_DIM,
                             char_embed_dim=config.CHAR_EMBED_DIM,
                             sent_embed_dim=config.SENT_EMBED_DIM).to(device)


def _forward(model, data, device):
    return model(data['ids'].to(device),
                 data['sentiment_input'].to(device),
                 data['probas_start'].to(device),
                 data['probas_end'].to(device))


def predict(model, dataset, device, batch_size=32, num_workers=0):
    model.eval()
    start_probas = []
    end_probas = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            start_logits, end_logits = _forward(model, data, device)
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()
            for s, e in zip(start_probs, end_probs):
                start_probas.append(list(s))
                end_probas.append(list(e))
    return start_probas, end_probas


def fit(model, train_dataset, val_dataset, loss_config, device,
        epochs=5, batch_size=8, acc_steps=1, warmup_prop=0.0, lr=5e-4,
        num_workers=0):
    """
    Trains with Adam and a linear schedule. From SWA_FIRST_EPOCH on, weights
    are averaged at the end of every epoch (SWA, done with torchcontrib in
    2021); validation uses the averaged weights, and the averaged weights are
    what the model keeps at the end.
    """
    len_sampler = helper.LenMatchBatchSampler(
        torch.utils.data.RandomSampler(train_dataset),
        batch_size=batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=len_sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_first_epoch = config.SWA_FIRST_EPOCH

    n_steps = max(1, int(epochs * len(train_loader) / acc_steps))
    scheduler = h.linear_schedule_with_warmup(optimizer, int(warmup_prop * n_steps), n_steps)

    total_steps = 0
    val_jac = 0.0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        optimizer.zero_grad()
        avg_loss = 0

        for step, data in enumerate(train_loader):
            total_steps += 1
            start_logits, end_logits = _forward(model, data, device)
            loss = loss_fn(start_logits, end_logits,
                           data['target_start'].to(device),
                           data['target_end'].to(device),
                           config=loss_config)
            avg_loss += loss.item() / len(train_loader)
            loss.backward()
            if (step + 1) % acc_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        use_swa = epoch >= swa_first_epoch
        if use_swa:
            swa_model.update_parameters(model)
        eval_model = swa_model.module if use_swa else model
        eval_model.eval()

        avg_val_loss = 0.0
        val_jac = 0.0
        with torch.no_grad():
            for data in val_loader:
                start_logits, end_logits = _forward(eval_model, data, device)
                loss = loss_fn(start_logits, end_logits,
                               data["target_start"].to(device),
                               data["target_end"].to(device),
                               config=loss_config)
                avg_val_loss += loss.item() / len(val_loader)
                val_jac += helper.jaccard_from_logits_string(
                    data, start_logits, end_logits) / len(val_dataset)

        dt = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}/{epochs} step {total_steps} lr={lr_now:.1e} '
              f't={dt:.0f}s loss={avg_loss:.3f} val_loss={avg_val_loss:.3f} '
              f'val_jaccard={val_jac:.4f}')

    if epochs > swa_first_epoch:
        model.load_state_dict(swa_model.module.state_dict())

    del train_loader, val_loader
    gc.collect()
    return val_jac


def k_fold(df_train, df_test, X_train, X_test, preds, len_voc, n_models,
           paths, device, k=5, fold_seed=42, model_seed=42, epochs=config.EPOCHS,
           num_workers=0, save=True):
    score = 0
    splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=fold_seed).split(
        X=df_train, y=df_train['sentiment']))

    pred_oof = [[[], []] for _ in range(len(df_train))]
    pred_tests = []

    test_dataset = dataset.TweetCharDataset(df_test, X_test,
                                            preds['test_start'], preds['test_end'],
                                            max_len=config.MAX_LEN, train=False,
                                            n_models=n_models)

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n-------------   Fold {i + 1}  -------------")
        h.seed_everything(model_seed)
        model = new_model(len_voc, device)
        model.zero_grad()

        train_dataset = dataset.TweetCharDataset(df_train.iloc[train_idx], X_train[train_idx],
                                                 preds['oof_start'][train_idx],
                                                 preds['oof_end'][train_idx],
                                                 max_len=config.MAX_LEN, n_models=n_models)
        val_dataset = dataset.TweetCharDataset(df_train.iloc[val_idx], X_train[val_idx],
                                               preds['oof_start'][val_idx],
                                               preds['oof_end'][val_idx],
                                               max_len=config.MAX_LEN, n_models=n_models)

        helper.unfreeze(model)
        print(f'    -> {helper.count_parameters(model)} trainable parameters\n')

        fold_score = fit(model, train_dataset, val_dataset, config.loss_config, device,
                         epochs=epochs, batch_size=config.TRAIN_BATCH_SIZE,
                         lr=config.LR, warmup_prop=config.WARMUP_PROP,
                         num_workers=num_workers)
        score += fold_score / k

        pred_val_start, pred_val_end = predict(model, val_dataset, device,
                                               batch_size=config.VALID_BATCH_SIZE)
        for j, idx in enumerate(val_idx):
            pred_oof[idx] = [pred_val_start[j], pred_val_end[j]]

        pred_tests.append(predict(model, test_dataset, device,
                                  batch_size=config.VALID_BATCH_SIZE))
        if save:
            helper.save_model_weights(model, paths.level2_model(i), verbose=0)

        del model, train_dataset, val_dataset
        gc.collect()

    print(f'\n Level-two CV jaccard (fold average) is {score:.4f}')
    return pred_oof, pred_tests, score


def infer(df_test, X_test, preds, len_voc, n_models, paths, device, k=5):
    """Test predictions of the k saved level-two models."""
    test_dataset = dataset.TweetCharDataset(df_test, X_test,
                                            preds['test_start'], preds['test_end'],
                                            max_len=config.MAX_LEN, train=False,
                                            n_models=n_models)
    pred_tests = []
    for m in range(k):
        model = new_model(len_voc, device)
        helper.load_model_weights(model, paths.level2_model(m), verbose=0)
        model.to(device)
        pred_tests.append(predict(model, test_dataset, device,
                                  batch_size=config.VALID_BATCH_SIZE))
    return pred_tests, test_dataset
