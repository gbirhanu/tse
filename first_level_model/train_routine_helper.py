import warnings

import numpy as np
import torch
from tqdm import tqdm

from first_level_model import helper

warnings.filterwarnings('ignore', message='.*reduction.*mean.*')


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Kullback-Leibler divergence between the predicted start/end
    distributions and the soft Jaccard targets.
    """
    m = torch.nn.LogSoftmax(dim=1)
    loss_fct = torch.nn.KLDivLoss(reduction='mean')
    start_loss = loss_fct(m(start_logits), start_positions)
    end_loss = loss_fct(m(end_logits), end_positions)
    return start_loss + end_loss


def _autocast(device, fp16):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=fp16 and device.type == 'cuda')


def train_fn(data_loader, model, optimizer, device, scheduler=None,
             fp16=False, scaler=None):
    model.train()
    losses = helper.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), mininterval=5)

    for d in tk0:
        ids = d['ids'].to(device, dtype=torch.long)
        token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
        mask = d['mask'].to(device, dtype=torch.long)
        start_labels = d['start_labels'].to(device, dtype=torch.float)
        end_labels = d['end_labels'].to(device, dtype=torch.float)

        model.zero_grad()
        with _autocast(device, fp16):
            outputs_start, outputs_end = model(ids=ids, mask=mask,
                                               token_type_ids=token_type_ids)
        loss = loss_fn(outputs_start.float(), outputs_end.float(),
                       start_labels, end_labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device, fp16=False):
    """Validation loss and Jaccard; also returns char-level probabilities."""
    model.eval()
    losses = helper.AverageMeter()
    jaccards = helper.AverageMeter()
    char_start, char_end = [], []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), mininterval=5)
        for d in tk0:
            ids = d['ids'].to(device, dtype=torch.long)
            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
            mask = d['mask'].to(device, dtype=torch.long)
            start_labels = d['start_labels'].to(device, dtype=torch.float)
            end_labels = d['end_labels'].to(device, dtype=torch.float)
            orig_start = d['orig_start']
            orig_end = d['orig_end']
            orig_selected = d['orig_selected']
            orig_tweet = d['orig_tweet']
            offsets = d['offsets'].numpy()

            with _autocast(device, fp16):
                outputs_start, outputs_end = model(ids=ids, mask=mask,
                                                   token_type_ids=token_type_ids)
            outputs_start = outputs_start.float()
            outputs_end = outputs_end.float()
            loss = loss_fn(outputs_start, outputs_end, start_labels, end_labels)

            probs_start = torch.softmax(outputs_start, dim=-1).cpu().numpy()
            probs_end = torch.softmax(outputs_end, dim=-1).cpu().numpy()
            outputs_start = outputs_start.cpu().numpy()
            outputs_end = outputs_end.cpu().numpy()

            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                jaccard_scores.append(helper.calculate_jaccard(
                    original_tweet=tweet,
                    target_string=orig_selected[px],
                    start_logits=outputs_start[px, :],
                    end_logits=outputs_end[px, :],
                    orig_start=int(orig_start[px]),
                    orig_end=int(orig_end[px]),
                    offsets=offsets[px]))
                # fold-level char predictions, saved for the level-two model
                char_start.append(helper.token_level_to_char_level(
                    tweet, offsets[px], probs_start[px]))
                char_end.append(helper.token_level_to_char_level(
                    tweet, offsets[px], probs_end[px]))

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    print(f'Jaccard = {jaccards.avg:.5f}')
    return jaccards.avg, char_start, char_end


def predict_logits(data_loader, models, device, fp16=False):
    """
    Averaged start/end logits of several fold models (as in the 2021
    inference), turned into char-level probabilities.
    """
    char_start, char_end, texts = [], [], []
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader), mininterval=5):
            ids = d['ids'].to(device, dtype=torch.long)
            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
            mask = d['mask'].to(device, dtype=torch.long)
            offsets = d['offsets'].numpy()
            sum_start, sum_end = 0, 0
            for m in models:
                with _autocast(device, fp16):
                    s, e = m(ids=ids, mask=mask, token_type_ids=token_type_ids)
                sum_start = sum_start + s.float()
                sum_end = sum_end + e.float()
            probs_start = torch.softmax(sum_start / len(models), dim=-1).cpu().numpy()
            probs_end = torch.softmax(sum_end / len(models), dim=-1).cpu().numpy()
            for px, tweet in enumerate(d['orig_tweet']):
                char_start.append(helper.token_level_to_char_level(
                    tweet, offsets[px], probs_start[px]))
                char_end.append(helper.token_level_to_char_level(
                    tweet, offsets[px], probs_end[px]))
                texts.append(tweet)
    return char_start, char_end
