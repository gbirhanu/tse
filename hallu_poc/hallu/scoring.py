"""
Character-level span precision, recall and F1, summed over the whole set.

ld:  the LettuceDetect scorer (evaluate_detector_char_level), copied exactly:
     overlap is summed over every (predicted, gold) pair and gold length is the
     sum of gold span lengths, so overlapping gold spans count twice.
set: the same idea on character sets: each character counts once.
"""
import numpy as np


def ld_counts(pred, gold):
    o = 0
    for ps, pe in pred:
        for gs, ge in gold:
            a, b = max(ps, gs), min(pe, ge)
            if b > a:
                o += b - a
    return o, sum(e - s for s, e in pred), sum(e - s for s, e in gold)


def set_counts(pred, gold):
    p, g = set(), set()
    for s, e in pred:
        p.update(range(s, e))
    for s, e in gold:
        g.update(range(s, e))
    return len(p & g), len(p), len(g)


def counts(preds, samples, kind='ld'):
    if len(preds) != len(samples):
        raise ValueError(f'{len(preds)} predictions do not match {len(samples)} samples.')
    if kind not in ('ld', 'set'):
        raise ValueError("Scorer kind must be 'ld' or 'set'.")
    fn = ld_counts if kind == 'ld' else set_counts
    return np.array([fn(p, s['spans']) for p, s in zip(preds, samples)],
                    dtype=np.float64).reshape(-1, 3)


def prf(c):
    o, p, g = c[..., 0], c[..., 1], c[..., 2]
    prec = np.where(p > 0, o / np.maximum(p, 1), 0.0)
    rec = np.where(g > 0, o / np.maximum(g, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return prec, rec, f1


def score(preds, samples, kind='ld', digits=2):
    """Scores in percent; ``digits=None`` keeps raw precision for acceptance gates."""
    c = counts(preds, samples, kind)
    out = {}
    groups = {'all': np.ones(len(samples), bool)}
    for task in sorted({s['task'] for s in samples}):
        groups[task] = np.array([s['task'] == task for s in samples])
    for name, mask in groups.items():
        p, r, f = prf(c[mask].sum(0))
        values = (100 * float(p), 100 * float(r), 100 * float(f))
        out[name] = values if digits is None else tuple(round(value, digits) for value in values)
    return out


def _source_index(samples):
    sources = sorted({s['source_id'] for s in samples})
    pos = {src: i for i, src in enumerate(sources)}
    return np.array([pos[s['source_id']] for s in samples]), len(sources)


def bootstrap(preds, samples, base_preds=None, kind='ld', n=1000, seed=0, digits=2):
    """
    95% interval of F1 (in points) by resampling sources with replacement.
    With base_preds: interval of F1(preds) - F1(base_preds) on the same resamples.
    ``digits=None`` returns unrounded limits for strict positive-gain decisions.
    """
    if not samples:
        raise ValueError('Source bootstrap requires at least one sample.')
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError('The number of bootstrap resamples must be a positive integer.')
    idx, n_src = _source_index(samples)
    c = counts(preds, samples, kind)
    per_src = np.zeros((n_src, 3))
    np.add.at(per_src, idx, c)
    if base_preds is not None:
        cb = counts(base_preds, samples, kind)
        per_src_b = np.zeros((n_src, 3))
        np.add.at(per_src_b, idx, cb)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n_src, np.ones(n_src) / n_src, size=n)        # (n, n_src) resample counts
    f = prf(w @ per_src)[2]
    if base_preds is not None:
        f = f - prf(w @ per_src_b)[2]
    lo, hi = np.percentile(100 * f, [2.5, 97.5])
    limits = float(lo), float(hi)
    return limits if digits is None else tuple(round(value, digits) for value in limits)
