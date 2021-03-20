import torch
import random
import os 
import numpy as np

"""
This are helper function for diffrent purposes 
"""
def seed_everything(seed_value): 
    """
    Use seed for random state to make results deterministic in diffrent 
    execution time. 
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
def token_level_to_char_level(text, offsets, preds):
    """
    This function accepts text(tweet) and offset of selected_text and probablity of each
    token for being start and end tokend and give that to each character 
    in the token
    
    """
    character_level_probablity  = np.zeros(len(text))
    for loc, offset in enumerate(offsets):
        if offset[0] or offset[1]:
            character_level_probablity[offset[0]:offset[1]] = preds[loc]

    return character_level_probablity


def jaccard(str1, str2):
    """This function implements the similarity metrics given in the competition"""
    try:
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except ZeroDivisionError:
        return 0



def get_best_start_end_idx(start_logits, end_logits,
                           orig_start, orig_end):
    """Return best start and end indices following BERT paper."""
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


def calculate_jaccard(original_tweet, target_string,
                      start_logits, end_logits,
                      orig_start, orig_end,
                      offsets, 
                      verbose=False):
    """Calculate jaccard for local CV(cross validation)"""
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

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac


class AverageMeter:
    """helps to calculate average of diffrent metrics and stores them"""

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