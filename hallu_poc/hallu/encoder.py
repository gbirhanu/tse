"""
Token classifier (ModernBERT and friends): inputs, prediction and training.

The input is built the way LettuceDetect builds it: tokenizer(prompt, answer)
with only the prompt truncated, and a label only on answer tokens.
"""
import random
import time

import numpy as np
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from hallu import soft
from hallu.decoding import BASELINE, decode_all
from hallu.scoring import score


def load(name, device, fp16_weights=False):
    tok = AutoTokenizer.from_pretrained(name)
    kwargs = {'num_labels': 2}
    try:      # ModernBERT can torch.compile parts of itself; turn that off (old GPUs, Colab)
        model = AutoModelForTokenClassification.from_pretrained(name, reference_compile=False, **kwargs)
    except (TypeError, ValueError):
        model = AutoModelForTokenClassification.from_pretrained(name, **kwargs)
    if fp16_weights:
        model = model.half()
    return tok, model.to(device)


def encode(tok, prompt, answer, max_length):
    enc = tok(prompt, answer, truncation='only_first', max_length=max_length,
              return_offsets_mapping=True)
    seq = enc.sequence_ids()
    ans = [i for i, sid in enumerate(seq) if sid == 1]
    offs = [tuple(enc['offset_mapping'][i]) for i in ans]
    return enc['input_ids'], ans, offs


def _batches(lengths, order, batch_tokens, max_rows=64):
    out, cur, longest = [], [], 0
    for i in order:
        L = lengths[i]
        if cur and (max(longest, L) * (len(cur) + 1) > batch_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        out.append(cur)
    return out


def _pad(tok, seqs, device):
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def predict(model, tok, samples, max_length=4096, batch_tokens=16384, fp16=True, encs=None):
    """Per sample: list of (start, end, p_hallucinated) for each answer token."""
    model.eval()
    device = next(model.parameters()).device
    encs = encs or [encode(tok, s['prompt'], s['answer'], max_length) for s in samples]
    lengths = [len(e[0]) for e in encs]
    order = sorted(range(len(samples)), key=lambda i: lengths[i])
    out = [None] * len(samples)
    use_amp = fp16 and device.type == 'cuda'
    for b in _batches(lengths, order, batch_tokens):
        ids, mask = _pad(tok, [encs[i][0] for i in b], device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits.float(), -1)[..., 1].cpu().numpy()
        for r, i in enumerate(b):
            _, ans, offs = encs[i]
            out[i] = [(int(s), int(e), float(probs[r, j])) for j, (s, e) in zip(ans, offs)]
    return out


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(name, train_samples, dev_samples, targets='hard', epochs=3, lr=2e-5,
          max_length=4096, batch_tokens=8192, seed=42, device='cpu', fp16=True,
          warmup=0.1, log=print):
    """
    Fine-tunes `name` as a 2-class token classifier on answer tokens.
    targets='hard': 0/1 labels (LettuceDetect). targets='soft': lever 2.
    The loss is the same in both runs: cross-entropy against the target
    probability y, -(y log p1 + (1 - y) log p0), averaged over answer tokens.
    Keeps the epoch with the best dev span F1 (LettuceDetect scorer, t = 0.5).
    """
    seed_all(seed)
    device = torch.device(device)
    tok, model = load(name, device)
    t0 = time.time()
    data = []
    for s in train_samples:
        ids, ans, offs = encode(tok, s['prompt'], s['answer'], max_length)
        y = soft.token_targets(s['answer'], s['spans'], offs, kind=targets)
        data.append((ids, ans, y))
    dev_encs = [encode(tok, s['prompt'], s['answer'], max_length) for s in dev_samples]
    lengths = [len(d[0]) for d in data]
    log(f'Encoded {len(data)} training answers in {time.time() - t0:.0f}s '
        f'(median {int(np.median(lengths))} tokens, max {max(lengths)})')

    rng = random.Random(seed)

    def epoch_batches():
        order = list(range(len(data)))
        rng.shuffle(order)
        chunk = 2000      # sort inside chunks: similar lengths together, still random overall
        order = [i for c in range(0, len(order), chunk)
                 for i in sorted(order[c:c + chunk], key=lambda i: lengths[i])]
        bs = _batches(lengths, order, batch_tokens)
        rng.shuffle(bs)
        return bs

    n_steps = len(epoch_batches()) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * n_steps), n_steps)
    use_amp = fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_f1, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        batches = epoch_batches()
        for step, b in enumerate(batches):
            ids, mask = _pad(tok, [data[i][0] for i in b], device)
            rows = torch.cat([torch.full((len(data[i][1]),), r, dtype=torch.long)
                              for r, i in enumerate(b)]).to(device)
            cols = torch.cat([torch.tensor(data[i][1], dtype=torch.long) for i in b]).to(device)
            y = torch.cat([torch.tensor(data[i][2], dtype=torch.float32) for i in b]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            logp = torch.log_softmax(logits.float()[rows, cols], -1)
            loss = -(y * logp[:, 1] + (1 - y) * logp[:, 0]).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
            seen += len(b)
            if (step + 1) % 200 == 0:
                log(f'  epoch {ep + 1} step {step + 1}/{len(batches)} loss {total / seen:.4f} '
                    f'({time.time() - t0:.0f}s)')
        dev_tokens = predict(model, tok, dev_samples, max_length, fp16=fp16, encs=dev_encs)
        f1 = score(decode_all(dev_tokens, dev_samples, BASELINE), dev_samples)['all'][2]
        history.append({'epoch': ep + 1, 'loss': round(total / max(seen, 1), 4), 'dev_span_f1': f1,
                        'seconds': round(time.time() - t0)})
        log(f'Epoch {ep + 1}/{epochs}: loss {total / max(seen, 1):.4f}, dev span F1 {f1:.2f} '
            f'({time.time() - t0:.0f}s)')
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return tok, model, history
