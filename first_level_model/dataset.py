import re
from first_level_model  import helper
from first_level_model import config
import torch
import numpy as np

"""
This program have its bases in notebook shared by abhishek on kaggle
https://www.kaggle.com/abhishek/roberta-inference-5-folds

"""
def jaccard_array(a, b):
    """Calculates Jaccard on arrays."""
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_selected_text_start_end_index(selected_text,tweet):
    """
    this function get the start and end index of selected_text in a given tweet
    """
    selected_text = ' ' + ' '.join(re.split(r' ',selected_text))
    tweet = ' ' + ' '.join(re.split(r' ',tweet))
    len_sel_text = len(selected_text) - 1

    # Get sel_text start and end idx
    idx_0 = None
    idx_1 = None
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if ' ' + tweet[ind:ind + len_sel_text] == selected_text:
            idx_0 = ind
            idx_1 = ind + len_sel_text - 1
            break
    return idx_0, idx_1 

def get_ids_within_tweet_with_target_char(tweet_offsets,char_targets):
    """
    This function returns ids of target (selected_text) character.
    """
    target_ids = []
    for i, (offset_0, offset_1) in enumerate(tweet_offsets):
        if sum(char_targets[offset_0:offset_1]) > 0:
            target_ids.append(i)

    targets_start = target_ids[0]
    targets_end = target_ids[-1]
    return targets_start, targets_end,target_ids

def get_soft_labels(targets_start,targets_end,answer,sentence,n):
    start_labels = np.zeros(n)
    for i in range(targets_end + 1):
        jac = jaccard_array(answer, sentence[i:targets_end + 1])
        start_labels[i] = jac + jac ** 2
    start_labels = (1 - helper.SOFT_ALPHA) * start_labels / start_labels.sum()
    start_labels[targets_start] += helper.SOFT_ALPHA

    end_labels = np.zeros(n)
    for i in range(targets_start, n):
        jac = jaccard_array(answer, sentence[targets_start:i + 1])
        end_labels[i] = jac + jac ** 2
    end_labels = (1 - helper.SOFT_ALPHA) * end_labels / end_labels.sum()
    end_labels[targets_end] += helper.SOFT_ALPHA

    start_labels = [0, 0, 0, 0] + list(start_labels) + [0]
    end_labels = [0, 0, 0, 0] + list(end_labels) + [0]
def process_data(tweet, selected_text, sentiment,
                 tokenizer, max_len):
    data = {}
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    idx_0, idx_1 = get_selected_text_start_end_index(selected_text, tweet)
    # Assign 1 as target for each char in sel_text
    char_targets = [0] * len(tweet)
    if idx_0 is not None and idx_1 is not None:
        for cr in range(idx_0, idx_1 + 1):
            char_targets[cr] = 1

    tokenized_tweet = tokenizer.encode(tweet)
    # Vocab ids
    input_ids_original = tokenized_tweet.ids
    # Start and end char
    tweet_offsets = tokenized_tweet.offsets

    # Get ids within tweet of words that have target char
    targets_start, targets_end,target_ids = get_ids_within_tweet_with_target_char(tweet_offsets,char_targets)

    # Sentiment 'word' id in vocab
    # tokenizer.encode("positive").ids = 1313
    # tokenizer.encode("negative").ids = 2430
    # tokenizer.encode("neutral").ids = 7974
    sentiment_id = {'positive': 1313,
                    'negative': 2430,
                    'neutral': 7974}

    # Soft Jaccard labels
    # ----------------------------------
    n = len(input_ids_original)
    sentence = np.arange(n)
    answer = sentence[targets_start:targets_end + 1]

    start_labels, end_labels = get_soft_labels(targets_start,targets_end,answer,sentence,n)
    # ----------------------------------

    # Input for RoBERTa
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + \
                [2] + input_ids_original + [2]
    # No token types in RoBERTa
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_original) + 1)
    # Mask of input without padding
    mask = [1] * len(token_type_ids)
    # Start and end char ids for each word including new tokens
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    # Ids within tweet of words that have target char including new tokens
    targets_start += 4
    targets_end += 4
    orig_start = 4
    orig_end = len(input_ids_original) + 3

    # Input padding: new mask, token type ids, tweet offsets
    padding_len = max_len - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([1] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_len)
        start_labels = start_labels + ([0] * padding_len)
        end_labels = end_labels + ([0] * padding_len)

    targets_select = [0] * len(token_type_ids)
    for i in range(len(targets_select)):
        if i in target_ids:
            targets_select[i + 4] = 1
    data = {'ids': input_ids,
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
    return data


class TweetDataset:
    """
    This class prepares dataset for our roberta model roberta 
    """
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = tweets
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        data = process_data(self.tweets[item],
                            self.selected_texts[item],
                            self.sentiments[item],
                            self.tokenizer,
                            self.max_len)

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