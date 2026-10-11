"""
Lever 3: a character-level model stacked on the token classifier (the 2021
"level two", changed from one start/end per tweet to tagging every character).

Each answer character gets: its character id, the token classifier's
probability for the token it sits in (spaces take the mean of their two
neighbours), that probability as a logit, a space flag and the task. A small
BiGRU reads the characters and predicts, per character, whether it is inside a
hallucinated span. Its character probabilities are averaged back to tokens so
the same decoder (and lever-1 tuning) can be used.

It must be trained on predictions the token classifier made on data it did not
train on. `fit_and_apply` also tunes the decoder on out-of-fold stacker output.
"""
import random

import numpy as np
import torch
import torch.nn as nn

from hallu import data as hdata
from hallu.decoding import decode_all, tune

TASK_IDS = {'qa': 0, 'summary': 1, 'data2text': 2}


def char_features(answer, tokens):
    n = len(answer)
    p = np.full(n, np.nan)
    for s, e, prob in tokens:
        if e > s:
            p[s:e] = prob
    idx = np.where(~np.isnan(p))[0]
    if len(idx) == 0:
        p[:] = 0.0
    else:     # fill gaps (spaces) with the mean of the nearest probabilities on each side
        left = np.interp(np.arange(n), idx, p[idx], left=p[idx[0]], right=p[idx[-1]])
        p = np.where(np.isnan(p), left, p)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    space = np.array([c.isspace() for c in answer], dtype=np.float32)
    return np.stack([p, np.log(p / (1 - p)) / 5.0, space], 1).astype(np.float32)


def char_labels(answer, spans):
    y = np.zeros(len(answer), dtype=np.float32)
    for s, e in spans:
        y[s:e] = 1.0
    return y


class Vocab:
    def __init__(self, texts):
        counts = {}
        for t in texts:
            for c in t.lower():
                counts[c] = counts.get(c, 0) + 1
        chars = sorted(counts, key=lambda c: -counts[c])
        self.index = {c: i + 2 for i, c in enumerate(chars)}     # 0 pad, 1 unknown

    def __len__(self):
        return len(self.index) + 2

    def ids(self, text):
        return np.array([self.index.get(c, 1) for c in text.lower()], dtype=np.int64)


class CharTagger(nn.Module):
    def __init__(self, n_chars, hidden=64):
        super().__init__()
        self.char = nn.Embedding(n_chars, 16, padding_idx=0)
        self.task = nn.Embedding(3, 4)
        self.inp = nn.Sequential(nn.Linear(16 + 4 + 3, 48), nn.ReLU())
        self.rnn = nn.GRU(48, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, chars, feats, task):
        t = self.task(task)[:, None, :].expand(-1, chars.shape[1], -1)
        x = self.inp(torch.cat([self.char(chars), t, feats], -1))
        return self.out(self.rnn(x)[0]).squeeze(-1)


def _tensors(vocab, samples, token_lists, with_labels):
    items = []
    for s, toks in zip(samples, token_lists):
        items.append((vocab.ids(s['answer']), char_features(s['answer'], toks),
                      TASK_IDS[s['task']],
                      char_labels(s['answer'], s['spans']) if with_labels else None))
    return items


def _batch(items, idx, device):
    L = max(len(items[i][0]) for i in idx)
    chars = torch.zeros((len(idx), L), dtype=torch.long)
    feats = torch.zeros((len(idx), L, 3))
    y = torch.zeros((len(idx), L))
    mask = torch.zeros((len(idx), L))
    for r, i in enumerate(idx):
        c, f, _, lab = items[i]
        n = len(c)
        chars[r, :n] = torch.from_numpy(c)
        feats[r, :n] = torch.from_numpy(f)
        mask[r, :n] = 1
        if lab is not None:
            y[r, :n] = torch.from_numpy(lab)
    task = torch.tensor([items[i][2] for i in idx])
    return chars.to(device), feats.to(device), task.to(device), y.to(device), mask.to(device)


def train_tagger(samples, token_lists, device, epochs=15, lr=3e-3, batch_size=32, seed=0,
                 smoothing=0.05):
    random.seed(seed)
    torch.manual_seed(seed)
    vocab = Vocab([s['answer'] for s in samples])
    model = CharTagger(len(vocab)).to(device)
    items = _tensors(vocab, samples, token_lists, True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * ((len(items) + batch_size - 1) // batch_size))
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    for _ in range(epochs):
        model.train()
        random.shuffle(batches)
        for b in batches:
            chars, feats, task, y, mask = _batch(items, b, device)
            y = y * (1 - smoothing) + smoothing / 2
            logits = model(chars, feats, task)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
                    * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return vocab, model


@torch.no_grad()
def apply_tagger(vocab, model, samples, token_lists, device, batch_size=64):
    """Stacker probabilities averaged back onto the classifier's tokens."""
    model.eval()
    items = _tensors(vocab, samples, token_lists, False)
    out = [None] * len(samples)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for k in range(0, len(order), batch_size):
        b = order[k:k + batch_size]
        chars, feats, task, _, _ = _batch(items, b, device)
        probs = torch.sigmoid(model(chars, feats, task)).cpu().numpy()
        for r, i in enumerate(b):
            out[i] = [(s, e, float(probs[r, s:e].mean()) if e > s else 0.0)
                      for s, e, _ in token_lists[i]]
    return out


def fit_and_apply(fit_samples, fit_tokens, eval_samples, eval_tokens, device, seed=0, log=print):
    """
    1. Out-of-fold stacker output on the fit set (2 folds by source) -> tune the decoder.
    2. Stacker trained on the whole fit set -> applied to the eval set, decoded.
    """
    folds = hdata.source_folds(fit_samples, k=2, seed=seed + 1)
    oof = [None] * len(fit_samples)
    for f in (0, 1):
        tr = [i for i, x in enumerate(folds) if x != f]
        te = [i for i, x in enumerate(folds) if x == f]
        vocab, model = train_tagger([fit_samples[i] for i in tr], [fit_tokens[i] for i in tr],
                                    device, seed=seed)
        pred = apply_tagger(vocab, model, [fit_samples[i] for i in te],
                            [fit_tokens[i] for i in te], device)
        for i, p in zip(te, pred):
            oof[i] = p
    params = tune(oof, fit_samples)
    log(f'    stacker decoder settings: {params}')
    vocab, model = train_tagger(fit_samples, fit_tokens, device, seed=seed)
    stacked = apply_tagger(vocab, model, eval_samples, eval_tokens, device)
    return decode_all(stacked, eval_samples, params), params
