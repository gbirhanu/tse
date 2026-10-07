import os
import random

import numpy as np
import torch

"""
Helper functions for different purposes.
"""


def seed_everything(seed_value):
    """Fix random states so runs repeat."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def token_level_to_char_level(text, offsets, preds):
    """
    Gives every character of the tweet the probability of the token it
    belongs to (start or end probability).
    """
    offsets = np.asarray(offsets)
    character_level_probablity = np.zeros(len(text))
    for loc, (o0, o1) in enumerate(offsets):
        if o0 or o1:
            character_level_probablity[o0:o1] = preds[loc]
    return character_level_probablity


def jaccard(str1, str2):
    """The similarity metric given in the competition."""
    try:
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except ZeroDivisionError:
        return 0


def get_best_start_end_idx(start_logits, end_logits, orig_start, orig_end):
    """Best start and end indices (start <= end), following the BERT paper."""
    best_logit = -np.inf
    best_idxs = None
    start_logits = start_logits[orig_start:orig_end + 1]
    end_logits = end_logits[orig_start:orig_end + 1]
    for start_idx, start_logit in enumerate(start_logits):
        for end_idx, end_logit in enumerate(end_logits[start_idx:]):
            logit_sum = start_logit + end_logit
            if logit_sum > best_logit:
                best_logit = logit_sum
                best_idxs = (orig_start + start_idx,
                             orig_start + start_idx + end_idx)
    return best_idxs


def decode_selected_text(original_tweet, start_logits, end_logits,
                         orig_start, orig_end, offsets):
    """Turns start/end token scores into the selected phrase."""
    offsets = np.asarray(offsets)
    start_idx, end_idx = get_best_start_end_idx(
        start_logits, end_logits, orig_start, orig_end)

    filtered_output = ''
    for ix in range(start_idx, end_idx + 1):
        filtered_output += original_tweet[offsets[ix][0]:offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += ' '

    # Return orig tweet if it has less then 2 words
    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    if len(filtered_output.split()) == 1:
        filtered_output = filtered_output.replace('!!!!', '!')
        filtered_output = filtered_output.replace('..', '.')
        filtered_output = filtered_output.replace('...', '.')

    filtered_output = filtered_output.replace('ïï', 'ï')
    filtered_output = filtered_output.replace('¿¿', '¿')
    return filtered_output.strip()


def calculate_jaccard(original_tweet, target_string,
                      start_logits, end_logits,
                      orig_start, orig_end,
                      offsets, verbose=False):
    """Jaccard of one prediction, for local CV (cross validation)."""
    pred = decode_selected_text(original_tweet, start_logits, end_logits,
                                orig_start, orig_end, offsets)
    return jaccard(target_string.strip(), pred)


class AverageMeter:
    """Running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Same schedule as transformers.get_linear_schedule_with_warmup."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) /
                   float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
