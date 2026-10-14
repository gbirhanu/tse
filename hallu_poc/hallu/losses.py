"""Differentiable character-coverage F1 auxiliary for answer token probabilities.

This is a training surrogate, not the LettuceDetect-compatible evaluation metric.
It uses the union of covered gold characters and averages probabilities when
token offsets overlap. Decoded span boundaries, gaps/merges, thresholding, and
the LD scorer's repeated counting of overlapping gold spans are not modeled.
Characters without a token offset have no prediction and contribute no mass.
"""
import operator

import numpy as np
import torch

SPAN_F1_VERSION = 'char_union_f1_v1'


def _interval(start, end, length):
    """Clip integer character offsets to the answer, preserving empty spans."""
    try:
        start, end = operator.index(start), operator.index(end)
    except TypeError as exc:
        raise ValueError('Character offsets must be integers') from exc
    return min(length, max(0, start)), min(length, max(0, end))


def character_targets(answer, spans, offsets):
    """Return token character weights and gold fractions as float32 arrays.

    Every covered character contributes total weight one. If k tokens cover a
    character, each receives 1/k of its mass and of its gold mass. Thus duplicate
    byte-token offsets never count characters twice. Fractions use the union of
    gold spans and handle partially labeled tokens. Empty/reversed offsets have
    zero weight; offsets beyond the answer are clipped to valid characters.
    """
    length = len(answer)
    intervals = [_interval(start, end, length) for start, end in offsets]
    difference = np.zeros(length + 1, dtype=np.int64)
    for start, end in intervals:
        if end > start:
            difference[start] += 1
            difference[end] -= 1
    counts = np.cumsum(difference[:-1])
    inverse = np.divide(1.0, counts, out=np.zeros(length, dtype=np.float64), where=counts > 0)
    gold = np.zeros(length, dtype=bool)
    for start, end in spans:
        start, end = _interval(start, end, length)
        if end > start:
            gold[start:end] = True
    mass_prefix = np.concatenate(([0.0], np.cumsum(inverse)))
    gold_prefix = np.concatenate(([0.0], np.cumsum(inverse * gold)))
    weights = np.zeros(len(intervals), dtype=np.float32)
    fractions = np.zeros(len(intervals), dtype=np.float32)
    for i, (start, end) in enumerate(intervals):
        if end > start:
            weight = mass_prefix[end] - mass_prefix[start]
            weights[i] = weight
            if weight > 0:
                fractions[i] = (gold_prefix[end] - gold_prefix[start]) / weight
    return weights, fractions


def soft_span_f1_loss(probabilities, weights, gold_fractions):
    """Batch-pooled soft F1 over answer characters; inputs have identical shape.

    With P=sum(w*p), G=sum(w*g), I=sum(w*p*g), the auxiliary is
    1-(2*I+eps)/(P+G+eps). When G=0, use the weighted mean probability instead,
    retaining useful gradients against false positives in all-negative batches.
    No represented characters yields differentiable zero. Computation uses at
    least float32 even under mixed precision; gold/weights are fixed targets.
    """
    if probabilities.shape != weights.shape or probabilities.shape != gold_fractions.shape:
        raise ValueError('Probabilities, character weights, and gold fractions must have identical shapes')
    p = probabilities.float() if probabilities.dtype in (torch.float16, torch.bfloat16) else probabilities
    w = weights.to(device=p.device, dtype=p.dtype).detach()
    g = gold_fractions.to(device=p.device, dtype=p.dtype).detach()
    predicted_mass = (w * p).sum()
    gold_mass = (w * g).sum()
    overlap = (w * p * g).sum()
    eps = 1e-8
    positive_loss = 1.0 - (2.0 * overlap + eps) / (predicted_mass + gold_mass + eps)
    negative_loss = predicted_mass / w.sum().clamp_min(eps)
    return torch.where(gold_mass > 0, positive_loss, negative_loss)
