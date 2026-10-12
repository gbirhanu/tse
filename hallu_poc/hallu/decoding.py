"""
Turning token probabilities into character spans.

decode() with t=0.5, gap=0, snap=False is LettuceDetect's own rule (a token is
hallucinated when its class-1 probability wins; runs of such tokens form a span).
Lever 1 tunes t, gap and snap per task on data the model did not train on.
"""
import itertools

import numpy as np

from hallu.scoring import ld_counts, prf
from hallu.progress import progress

GRID_T = [0.03, 0.05, 0.08] + [round(float(x), 2) for x in np.arange(0.1, 0.71, 0.05)]
GRID_GAP = [0, 1, 2, 3]
GRID_SNAP = [False, True]
BASELINE = {'t': 0.5, 'gap': 0, 'snap': False}


def decode(tokens, answer, t=0.5, gap=0, snap=False):
    """tokens: list of (start, end, prob) for the answer, in order."""
    spans, cur, neg = [], None, 0
    for s, e, p in tokens:
        if e <= s:
            continue
        if p > t:
            if cur is not None and neg <= gap:
                cur[1] = e
            else:
                if cur is not None:
                    spans.append((cur[0], cur[1]))
                cur = [s, e]
            neg = 0
        elif cur is not None:
            neg += 1
    if cur is not None:
        spans.append((cur[0], cur[1]))
    if snap:
        spans = _snap(spans, answer)
    return spans


def _snap(spans, answer):
    """Trim spaces at the span edges (some tokenizers put the leading space in the token),
    then extend both edges to whole words."""
    out = []
    for s, e in spans:
        while s < e and answer[s].isspace():
            s += 1
        while e > s and answer[e - 1].isspace():
            e -= 1
        if e <= s:
            continue
        while s > 0 and not answer[s - 1].isspace():
            s -= 1
        while e < len(answer) and not answer[e].isspace():
            e += 1
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def decode_all(token_lists, samples, params):
    """params: one dict, or {task: dict}."""
    out = []
    for toks, s in zip(token_lists, samples):
        p = params.get(s['task'], BASELINE) if 't' not in params else params
        out.append(decode(toks, s['answer'], **p))
    return out


def tune(token_lists, samples):
    """Best (t, gap, snap) per task by span F1 (LettuceDetect scorer) on these samples."""
    best = {}
    for task in sorted({s['task'] for s in samples}):
        idx = [i for i, s in enumerate(samples) if s['task'] == task]
        top = (-1.0, BASELINE)
        grid = itertools.product(GRID_T, GRID_GAP, GRID_SNAP)
        for t, gap, snap in progress(grid, total=len(GRID_T) * len(GRID_GAP) * len(GRID_SNAP),
                                     desc=f'Tuning decoder ({task})', unit='settings'):
            c = np.zeros(3)
            for i in idx:
                c += ld_counts(decode(token_lists[i], samples[i]['answer'], t, gap, snap),
                               samples[i]['spans'])
            f = float(prf(c)[2])
            if f > top[0]:
                top = (f, {'t': t, 'gap': gap, 'snap': snap})
        best[task] = top[1]
    return best
