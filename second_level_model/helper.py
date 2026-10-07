import json
import os
import pickle

import numpy as np
import torch


class CharTokenizer:
    """
    Character tokenizer. Replaces keras Tokenizer(char_level=True,
    oov_token='UNK', lower=True), which is gone from Keras 3. Same ids:
    index 1 = unknown, then characters from most to least frequent.
    """

    def __init__(self, oov_token='UNK'):
        self.oov_token = oov_token
        self.word_index = {}

    def fit_on_texts(self, texts):
        counts = {}
        for text in texts:
            for ch in str(text).lower():
                counts[ch] = counts.get(ch, 0) + 1
        ordered = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        vocab = [self.oov_token] + [ch for ch, _ in ordered]
        self.word_index = {ch: i + 1 for i, ch in enumerate(vocab)}

    def texts_to_sequences(self, texts):
        oov = self.word_index[self.oov_token]
        return [[self.word_index.get(ch, oov) for ch in str(text).lower()]
                for text in texts]

    def __len__(self):
        return len(self.word_index) + 1   # + padding index 0

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'oov_token': self.oov_token, 'word_index': self.word_index},
                      fh, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        tok = cls(data['oov_token'])
        tok.word_index = data['word_index']
        return tok


def pad_sequences(sequences, maxlen):
    """Post padding and post truncating with 0, as keras pad_sequences."""
    out = np.zeros((len(sequences), maxlen), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = list(seq)[:maxlen]
        out[i, :len(seq)] = seq
    return out


def to_object_array(items):
    """1-D numpy array of variable-length items (np.array fails on ragged lists)."""
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def save_model_weights(model, path, verbose=1):
    if verbose:
        print(f"\n -> Saving weights to {path}\n")
    torch.save(model.state_dict(), path)


def load_model_weights(model, path, verbose=1):
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    if verbose:
        print(f"\n -> Loading weights from {path}\n")
    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model, all=False):
    if all:
        return sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def jaccard(str1, str2):
    """Original metric implementation."""
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(a) + len(b) - len(c) == 0:
        return 0.0
    return float(len(c)) / (len(a) + len(b) - len(c))


def jaccard_from_logits_string(data, start_logits, end_logits):
    n = start_logits.size(0)
    score = 0
    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
    for i in range(n):
        start_idx = np.argmax(start_logits[i])
        end_idx = np.argmax(end_logits[i])
        text = data["text"][i]
        pred = text[start_idx: end_idx]
        score += jaccard(data["selected_text"][i], pred)
    return score


def load_char_preds(paths, seed_idxs, kind, text_ids):
    """
    Char-level probabilities of several level-one seeds, matched by textID.
    Returns two lists (start, end) with one (len(text), n_seeds) array per text.
    The 2021 code matched rows by position and re-sorted them; matching by
    textID removes that step.
    """
    per_seed = []
    for s in seed_idxs:
        path = paths.oof_pkl(s) if kind == 'oof' else paths.test_pkl(s)
        if not os.path.exists(path):
            step = 'train1 (all 5 folds)' if kind == 'oof' else 'test1'
            raise FileNotFoundError(f'{path} is missing: seed {s} has not finished {step}.')
        with open(path, 'rb') as handle:
            d = pickle.load(handle)
        lookup = {tid: (st, en) for tid, st, en in zip(d['textID'], d['start'], d['end'])}
        missing = [t for t in text_ids if t not in lookup]
        if missing:
            raise KeyError(f'{len(missing)} texts have no {kind} prediction in {path}')
        per_seed.append([lookup[t] for t in text_ids])

    n = len(seed_idxs)
    starts = [np.stack([per_seed[m][i][0] for m in range(n)], 1) for i in range(len(text_ids))]
    ends = [np.stack([per_seed[m][i][1] for m in range(n)], 1) for i in range(len(text_ids))]
    return starts, ends


def string_from_preds_char_level(dataset, preds, test=False,
                                 remove_neutral=False):
    selected_texts = []
    n_models = len(preds)

    for idx in range(len(dataset)):
        data = dataset[idx]
        if test:
            start_probas = np.mean([preds[i][0][idx] for i in range(n_models)], 0)
            end_probas = np.mean([preds[i][1][idx] for i in range(n_models)], 0)
        else:
            start_probas = preds[idx][0]
            end_probas = preds[idx][1]

        start_idx = np.argmax(start_probas)
        end_idx = np.argmax(end_probas)

        if end_idx < start_idx:
            selected_text = data["text"]
        elif remove_neutral and data["sentiment"] == "neutral":
            selected_text = data["text"]
        else:
            selected_text = data["text"][start_idx: end_idx]

        selected_texts.append(selected_text.strip())

    return selected_texts


class LenMatchBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Custom PyTorch Sampler that generates batches of similar length.
    """

    def __iter__(self):
        buckets = [[] for _ in range(100)]
        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx]['ids'] == 0)
            count_zeros = int(count_zeros / 64)
            buckets[count_zeros].append(idx)
            if len(buckets[count_zeros]) == self.batch_size:
                yield list(buckets[count_zeros])
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch
