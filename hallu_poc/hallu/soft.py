"""
Lever 2: soft span targets, carried over from the 2021 tweet model.

In 2021 the start (and end) target was not one position but a spread over
nearby positions, each weighted by J + J^2, where J is the Jaccard between the
span it gives and the true span; the true position kept SOFT_ALPHA = 0.6 of
the mass. Here an answer can hold several spans and we tag tokens, so for each
gold span we move its start (end fixed) and its end (start fixed) by up to K
words, weight each moved span by J + J^2, and give every word the weighted
share of moved spans that contain it. A token's target is then
ALPHA * (0/1 gold label) + (1 - ALPHA) * that share, taking the highest over
the gold spans. Inside tokens stay near 1; words just outside get a little.
"""
import re

import numpy as np

ALPHA = 0.6
K = 5
_WORD = re.compile(r'\S+')


def word_coverage(answer, spans, k=K):
    words = [(m.start(), m.end()) for m in _WORD.finditer(answer)]
    n = len(words)
    best = np.zeros(n)
    for gs, ge in spans:
        inside = [i for i, (s, e) in enumerate(words) if e > gs and s < ge]
        if not inside:
            continue
        a, b = inside[0], inside[-1]
        cov = np.zeros(n)
        for moving_start in (True, False):
            part, total = np.zeros(n), 0.0
            cands = range(max(0, a - k), b + 1) if moving_start else range(a, min(n - 1, b + k) + 1)
            for x in cands:
                lo, hi = (x, b) if moving_start else (a, x)
                inter = min(b, hi) - max(a, lo) + 1
                union = max(b, hi) - min(a, lo) + 1
                j = inter / union
                w = j + j * j
                part[lo:hi + 1] += w
                total += w
            cov += 0.5 * part / total
        best = np.maximum(best, cov)
    return words, best


def token_targets(answer, spans, offsets, kind='hard', alpha=ALPHA):
    """Target per answer token. offsets = (start, end) of each answer token in the answer."""
    hard = np.array([1.0 if any(e > gs and s < ge for gs, ge in spans) else 0.0
                     for s, e in offsets])
    if kind == 'hard' or not spans:
        return hard
    words, cov = word_coverage(answer, spans)
    soft = np.zeros(len(offsets))
    w = 0
    for t, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        while w < len(words) and words[w][1] <= s:
            w += 1
        j, val = w, 0.0
        while j < len(words) and words[j][0] < e:
            val = max(val, cov[j])
            j += 1
        soft[t] = val
    return alpha * hard + (1 - alpha) * soft
