import os

import numpy as np
import tokenizers
import torch

from first_level_model import config

"""
This program has its base in the notebook shared by abhishek on kaggle
https://www.kaggle.com/abhishek/roberta-inference-5-folds
"""

SENTIMENTS = ('positive', 'negative', 'neutral')


def load_tokenizer(pretrained_dir):
    """RoBERTa byte-level BPE tokenizer from vocab.json and merges.txt."""
    vocab = os.path.join(pretrained_dir, 'vocab.json')
    merges = os.path.join(pretrained_dir, 'merges.txt')
    for path in (vocab, merges):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} not found. The pretrained folder needs vocab.json and merges.txt')
    return tokenizers.ByteLevelBPETokenizer(vocab=vocab, merges=merges,
                                            lowercase=True, add_prefix_space=True)


def sentiment_token_ids(tokenizer):
    """Vocab id of each sentiment word (1313, 2430, 7974 for roberta-base)."""
    ids = {}
    for s in SENTIMENTS:
        enc = tokenizer.encode(s).ids
        if len(enc) != 1:
            raise ValueError(f'"{s}" is not a single token in this vocabulary')
        ids[s] = enc[0]
    return ids


def jaccard_array(a, b):
    """Calculates Jaccard on arrays."""
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_selected_text_start_end_index(selected_text, tweet, legacy_shift=False):
    """
    Start and end character index (both inclusive) of selected_text in tweet.
    Spaces are kept as they are in the tweet.
    """
    start = tweet.find(selected_text)
    if start < 0:
        return None, None
    end = start + len(selected_text) - 1
    if legacy_shift:
        # the 2021 code returned positions in ' ' + tweet, one char to the right
        start, end = start + 1, end + 1
    return start, end


def get_ids_within_tweet_with_target_char(tweet_offsets, char_targets):
    """Ids of the tokens that contain at least one character of selected_text."""
    target_ids = []
    for i, (offset_0, offset_1) in enumerate(tweet_offsets):
        if sum(char_targets[offset_0:offset_1]) > 0:
            target_ids.append(i)
    if not target_ids:          # should not happen; fall back to the whole tweet
        target_ids = list(range(len(tweet_offsets)))
    return target_ids[0], target_ids[-1], target_ids


def get_soft_labels(targets_start, targets_end, answer, sentence, n):
    """Soft start/end targets: Jaccard of each candidate span with the answer."""
    start_labels = np.zeros(n)
    for i in range(targets_end + 1):
        jac = jaccard_array(answer, sentence[i:targets_end + 1])
        start_labels[i] = jac + jac ** 2
    start_labels = (1 - config.SOFT_ALPHA) * start_labels / start_labels.sum()
    start_labels[targets_start] += config.SOFT_ALPHA

    end_labels = np.zeros(n)
    for i in range(targets_start, n):
        jac = jaccard_array(answer, sentence[targets_start:i + 1])
        end_labels[i] = jac + jac ** 2
    end_labels = (1 - config.SOFT_ALPHA) * end_labels / end_labels.sum()
    end_labels[targets_end] += config.SOFT_ALPHA

    start_labels = [0, 0, 0, 0] + list(start_labels) + [0]
    end_labels = [0, 0, 0, 0] + list(end_labels) + [0]
    return start_labels, end_labels


def process_data(tweet, selected_text, sentiment, tokenizer, max_len, sentiment_ids,
                 legacy_shift=False):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    idx_0, idx_1 = get_selected_text_start_end_index(selected_text, tweet, legacy_shift)
    # Assign 1 as target for each char in sel_text
    char_targets = [0] * len(tweet)
    if idx_0 is not None and idx_1 is not None:
        for cr in range(idx_0, min(idx_1 + 1, len(tweet))):
            char_targets[cr] = 1

    tokenized_tweet = tokenizer.encode(tweet)
    input_ids_original = tokenized_tweet.ids
    tweet_offsets = tokenized_tweet.offsets

    targets_start, targets_end, target_ids = \
        get_ids_within_tweet_with_target_char(tweet_offsets, char_targets)

    # Soft Jaccard labels
    n = len(input_ids_original)
    sentence = np.arange(n)
    answer = sentence[targets_start:targets_end + 1]
    start_labels, end_labels = get_soft_labels(targets_start, targets_end,
                                               answer, sentence, n)

    # Input for RoBERTa: <s> sentiment </s> </s> tweet </s>
    input_ids = [0] + [sentiment_ids[sentiment]] + [2] + \
                [2] + input_ids_original + [2]
    if len(input_ids) > max_len:
        raise ValueError(f'Tweet needs {len(input_ids)} tokens but max_len is '
                         f'{max_len}. Increase --max-len.')
    # No token types in RoBERTa
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_original) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + list(tweet_offsets) + [(0, 0)]
    orig_start = 4
    orig_end = len(input_ids_original) + 3

    padding_len = max_len - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([1] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_len)
        start_labels = start_labels + ([0] * padding_len)
        end_labels = end_labels + ([0] * padding_len)

    targets_select = [0] * len(token_type_ids)
    for i in target_ids:
        targets_select[i + 4] = 1

    return {'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'start_labels': start_labels,
            'end_labels': end_labels,
            'orig_start': orig_start,
            'orig_end': orig_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets,
            'targets_select': targets_select}


class TweetDataset(torch.utils.data.Dataset):
    """Prepares tweets for the RoBERTa model."""

    def __init__(self, tweets, sentiments, selected_texts, tokenizer,
                 max_len=config.MAX_LEN):
        self.tweets = tweets
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sentiment_ids = sentiment_token_ids(tokenizer)
        # read here (not inside workers) so --legacy-target-shift reaches them
        self.legacy_shift = config.LEGACY_TARGET_SHIFT

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        data = process_data(str(self.tweets[item]),
                            str(self.selected_texts[item]),
                            self.sentiments[item],
                            self.tokenizer,
                            self.max_len,
                            self.sentiment_ids,
                            self.legacy_shift)

        return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                'mask': torch.tensor(data['mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(data['token_type_ids'],
                                               dtype=torch.long),
                'start_labels': torch.tensor(data['start_labels'],
                                             dtype=torch.float),
                'end_labels': torch.tensor(data['end_labels'],
                                           dtype=torch.float),
                'orig_start': data['orig_start'],
                'orig_end': data['orig_end'],
                'orig_tweet': data['orig_tweet'],
                'orig_selected': data['orig_selected'],
                'sentiment': data['sentiment'],
                'offsets': torch.tensor(data['offsets'], dtype=torch.long),
                'targets_select': torch.tensor(data['targets_select'],
                                               dtype=torch.float)}
