"""Validated probability averaging for runs with the same answer tokenization.

This module never chooses members or weights from labels. The caller is
responsible for run provenance, matching sample IDs and calibration eligibility.
"""
import hashlib
import json
import math
from numbers import Integral, Real


def fingerprint_tokenizer(tok):
    """Hash a fast tokenizer's complete, canonical tokenization definition.

    Padding and truncation are mutable backend settings changed by calls to the
    tokenizer, so they are excluded. Special-token assignments remain part of
    the identity, in addition to the backend's added-token definitions.
    """
    backend = getattr(tok, 'backend_tokenizer', None)
    if backend is None or not callable(getattr(backend, 'to_str', None)):
        raise ValueError('A fast tokenizer with a serializable backend is required.')
    definition = json.loads(backend.to_str())
    definition.pop('padding', None)
    definition.pop('truncation', None)

    def special_value(value):
        if isinstance(value, (list, tuple)):
            return [special_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # AddedToken flags are already represented by backend added_tokens;
        # here we record which token is assigned to each special-token role.
        return str(value)

    payload = {
        'backend': definition,
        'special_tokens': {
            key: special_value(value)
            for key, value in getattr(tok, 'special_tokens_map', {}).items()
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def ensemble_tokens(member_tokens, samples, weights=None):
    """Average aligned per-answer ``(start, end, probability)`` token triples.

    Weights must be predeclared, finite and nonnegative, with positive total
    weight. Every member is validated, including zero-weight members. Repeated
    or overlapping offsets are legal (byte-level tokenizers may produce them),
    but every member must have exactly the same ordered offset sequence.
    """
    if not member_tokens:
        raise ValueError('An ensemble needs at least one member.')
    n_members = len(member_tokens)
    if weights is None:
        weights = [1.0] * n_members
    if len(weights) != n_members:
        raise ValueError('Provide exactly one weight per ensemble member.')
    if any(isinstance(w, bool) or not isinstance(w, Real)
           or not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('Ensemble weights must be finite, nonnegative numbers.')
    largest = max(weights)
    if largest <= 0:
        raise ValueError('At least one ensemble weight must be positive.')
    # Scaling first avoids overflow for large but finite predeclared weights.
    scaled = [float(w / largest) for w in weights]
    total = math.fsum(scaled)
    normalized = [w / total for w in scaled]

    for m, predictions in enumerate(member_tokens):
        if len(predictions) != len(samples):
            raise ValueError(f'Member {m} has {len(predictions)} answers; '
                             f'expected {len(samples)}.')

    output = []
    for i, sample in enumerate(samples):
        answer_len = len(sample['answer'])
        reference = None
        probabilities = []
        for m, predictions in enumerate(member_tokens):
            tokens = predictions[i]
            if tokens is None:
                raise ValueError(f'Member {m}, answer {i} has no token predictions.')
            offsets, probs = [], []
            previous_start = -1
            for j, token in enumerate(tokens):
                if not isinstance(token, (tuple, list)) or len(token) != 3:
                    raise ValueError(f'Member {m}, answer {i}, token {j}: '
                                     'expected (start, end, probability).')
                start, end, probability = token
                if (isinstance(start, bool) or isinstance(end, bool)
                        or not isinstance(start, Integral) or not isinstance(end, Integral)
                        or not 0 <= start <= end <= answer_len or start < previous_start):
                    raise ValueError(f'Member {m}, answer {i}, token {j}: invalid offsets.')
                if (isinstance(probability, bool) or not isinstance(probability, Real)
                        or not math.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError(f'Member {m}, answer {i}, token {j}: '
                                     'probability must be finite and between 0 and 1.')
                offsets.append((int(start), int(end)))
                probs.append(float(probability))
                previous_start = start
            if reference is None:
                reference = offsets
            elif offsets != reference:
                raise ValueError(f'Member {m}, answer {i}: token offsets do not match '
                                 'the first member; token probabilities cannot be averaged.')
            probabilities.append(probs)
        averaged = []
        for j, (start, end) in enumerate(reference):
            probability = math.fsum(normalized[m] * probabilities[m][j]
                                    for m in range(n_members))
            # Remove only floating-point roundoff outside the probability range.
            averaged.append((start, end, min(1.0, max(0.0, probability))))
        output.append(averaged)
    return output
