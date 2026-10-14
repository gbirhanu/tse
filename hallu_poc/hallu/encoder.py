"""
Token classifier (ModernBERT and friends): inputs, prediction and training.

The input is built the way LettuceDetect builds it: tokenizer(prompt, answer)
with only the prompt truncated, and a label only on answer tokens.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
import uuid

import numpy as np
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from hallu import soft
from hallu.decoding import BASELINE, decode_all
from hallu.losses import SPAN_F1_VERSION, character_targets, soft_span_f1_loss
from hallu.progress import progress
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
def predict(model, tok, samples, max_length=4096, batch_tokens=16384, fp16=True, encs=None,
            progress_desc='Predicting', on_batch=None):
    """Per sample: list of (start, end, p_hallucinated) for each answer token."""
    model.eval()
    device = next(model.parameters()).device
    encs = encs or [encode(tok, s['prompt'], s['answer'], max_length) for s in samples]
    lengths = [len(e[0]) for e in encs]
    order = sorted(range(len(samples)), key=lambda i: lengths[i])
    out = [None] * len(samples)
    use_amp = fp16 and device.type == 'cuda'
    with progress(total=len(samples), desc=progress_desc or 'Predicting', unit='answers',
                  disable=progress_desc is None) as bar:
        for b in _batches(lengths, order, batch_tokens):
            ids, mask = _pad(tok, [encs[i][0] for i in b], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits.float(), -1)[..., 1].cpu().numpy()
            for r, i in enumerate(b):
                _, ans, offs = encs[i]
                out[i] = [(int(s), int(e), float(probs[r, j])) for j, (s, e) in zip(ans, offs)]
            bar.update(len(b))
            if on_batch is not None:
                on_batch(len(b))
    return out


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _identity_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else str(obj))


def _fingerprint(value):
    """Stable identities include text/labels, not just potentially reused row IDs."""
    return hashlib.sha256(_identity_json(value).encode('utf-8')).hexdigest()


def _rows_fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_identity_json(row).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _sample_fingerprint(samples):
    fields = ('id', 'source_id', 'task', 'split', 'prompt', 'answer', 'spans')
    return _rows_fingerprint({k: s.get(k) for k in fields} for s in samples)


def _rng_state(batch_rng):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'batches': batch_rng.getstate()}


def _restore_rng(state, batch_rng):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    batch_rng.setstate(state['batches'])


def _atomic_checkpoint(path, payload):
    """A killed save leaves the previous complete checkpoint available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-',
                                         suffix='.tmp', delete=False) as fh:
            tmp = Path(fh.name)
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _cleanup_staging(directory):
    """Remove only recognized interrupted-save files from this run directory.

    Checkpoint directories have one writer. A killed process cannot execute its
    finally blocks, so repeated reconnects otherwise accumulate large temp files.
    """
    directory = Path(directory)
    if not directory.exists():
        return
    root = directory.resolve()
    export_names = {'config.json', 'generation_config.json', 'model.safetensors',
                    'pytorch_model.bin', 'model.safetensors.index.json',
                    'pytorch_model.bin.index.json', 'tokenizer.json',
                    'tokenizer_config.json', 'special_tokens_map.json',
                    'vocab.json', 'vocab.txt', 'merges.txt', 'added_tokens.json',
                    'spiece.model', 'tokenizer.model', 'sentencepiece.bpe.model'}
    shard_pattern = r'(?:model-\d+-of-\d+\.safetensors|pytorch_model-\d+-of-\d+\.bin)'
    for path in directory.iterdir():
        if path.is_symlink() or path.resolve().parent != root:
            continue
        if path.is_file() and re.fullmatch(r'\.checkpoint-[A-Za-z0-9_-]+\.tmp', path.name):
            path.unlink()
        elif path.is_dir() and re.fullmatch(r'\.best-[0-9a-f]{32}', path.name):
            entries = list(path.iterdir())
            if all(entry.is_file() and not entry.is_symlink()
                   and entry.resolve().parent == path.resolve()
                   and (entry.name in export_names or re.fullmatch(shard_pattern, entry.name))
                   for entry in entries):
                for entry in entries:
                    entry.unlink()
                path.rmdir()


def _completed_checkpoint(identity, epochs, best_state, best_f1, history):
    """Finished training needs only its best model and provenance to recover."""
    return {'identity': identity, 'epoch': epochs, 'complete': True,
            'best_state': best_state, 'best_f1': best_f1, 'history': history}


def _export_best(tok, model, directory):
    """Export can be retried from latest.pt if interrupted during finalization."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Inherit the destination permissions. TemporaryDirectory's private Windows
    # ACL can exclude a restricted-process token even when its parent is writable.
    tmp = directory.parent / ('.best-' + uuid.uuid4().hex)
    tmp.mkdir()
    try:
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        for source in tmp.iterdir():
            if source.is_file():
                os.replace(source, directory / source.name)
    finally:
        # save_pretrained creates only files; never recursively remove a path.
        for source in tmp.iterdir():
            if source.is_file():
                source.unlink()
        tmp.rmdir()


def train(name, train_samples, dev_samples, targets='hard', epochs=3, lr=2e-5,
          max_length=4096, batch_tokens=8192, seed=42, device='cpu', fp16=True,
          warmup=0.1, log=print, checkpoint_dir=None, checkpoint_every=200,
          gradient_checkpointing=False, span_f1_weight=0.0):
    """
    Fine-tunes `name` as a 2-class token classifier on answer tokens.
    targets='hard': 0/1 labels (LettuceDetect). targets='soft': lever 2.
    The loss is the same in both runs: cross-entropy against the target
    probability y, -(y log p1 + (1 - y) log p0), averaged over answer tokens.
    A positive span_f1_weight adds that weight times the batch-pooled,
    character-weighted soft-F1 surrogate from hallu.losses. This auxiliary uses
    gold character unions even when CE uses soft targets. It does not replace
    the LD evaluator. The default zero preserves the original CE path and
    checkpoint identities, including compatibility with earlier checkpoints.
    Keeps the epoch with the best dev span F1 (LettuceDetect scorer, t = 0.5).

    With checkpoint_dir, latest.pt holds the complete training state. Repeating
    the same call resumes at its next unsaved batch (or repeats interrupted dev
    evaluation), and a finished run skips training. Changes to training settings,
    ordered data, or tokenization require a separate checkpoint directory. The
    returned best model is also exported to checkpoint_dir/best. Only load
    checkpoint directories you trust: the state contains Python RNG objects.
    Use one training process per checkpoint directory. Finished checkpoints drop
    optimizer/current-model/RNG state to reduce Drive storage.
    """
    if epochs < 1 or not train_samples or not dev_samples:
        raise ValueError('Training requires epochs >= 1 and non-empty train/dev samples')
    if not math.isfinite(span_f1_weight) or span_f1_weight < 0:
        raise ValueError('span_f1_weight must be finite and nonnegative')
    if batch_tokens < 1:
        raise ValueError('batch_tokens must be positive')
    if checkpoint_dir is not None and checkpoint_every < 1:
        raise ValueError('checkpoint_every must be positive')
    checkpoint_path = Path(checkpoint_dir) / 'latest.pt' if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        _cleanup_staging(checkpoint_path.parent)
    seed_all(seed)
    device = torch.device(device)
    tok, model = load(name, device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
    t0 = time.time()
    data = []
    character_data = [] if span_f1_weight > 0 else None
    for s in progress(train_samples, desc='Preparing training data', unit='answers'):
        ids, ans, offs = encode(tok, s['prompt'], s['answer'], max_length)
        y = soft.token_targets(s['answer'], s['spans'], offs, kind=targets)
        data.append((ids, ans, y))
        if character_data is not None:
            character_data.append(character_targets(s['answer'], s['spans'], offs))
    dev_encs = [encode(tok, s['prompt'], s['answer'], max_length)
                for s in progress(dev_samples, desc='Preparing dev data', unit='answers')]
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
    first_epoch, next_batch, batches = 0, 0, None
    total, seen, elapsed = 0.0, 0, 0.0
    identity = None
    if checkpoint_path is not None:
        identity = {'format': 1, 'name': str(name), 'targets': targets, 'epochs': epochs,
                    'lr': lr, 'max_length': max_length, 'batch_tokens': batch_tokens,
                    'seed': seed, 'device_type': device.type, 'fp16': fp16,
                    'warmup': warmup, 'gradient_checkpointing': gradient_checkpointing,
                    'n_steps': n_steps, 'model_config': _fingerprint(model.config.to_dict()),
                    'train': _sample_fingerprint(train_samples),
                    'dev': _sample_fingerprint(dev_samples),
                    'train_encodings': _rows_fingerprint(data), 'dev_encodings': _rows_fingerprint(dev_encs)}
        if span_f1_weight > 0:
            identity.update({'span_f1_weight': span_f1_weight, 'span_f1_version': SPAN_F1_VERSION,
                             'character_targets': _rows_fingerprint(character_data)})
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            previous = state.get('identity', {})
            changed = sorted(k for k in set(identity) | set(previous)
                             if identity.get(k) != previous.get(k))
            if changed:
                raise ValueError('Checkpoint does not match this run (' + ', '.join(changed)
                                 + '); use a separate checkpoint directory')
            if state['epoch'] == epochs:
                # Also accepts the older full completed checkpoints. No optimizer
                # restore is needed to retry the final export or predictions.
                best_state, best_f1, history = state['best_state'], state['best_f1'], state['history']
                compact = state.get('complete', False)
                del state
                model.load_state_dict(best_state)
                if not compact:
                    _atomic_checkpoint(checkpoint_path, _completed_checkpoint(
                        identity, epochs, best_state, best_f1, history))
                log(f'Training already complete ({epochs} epochs)')
                _export_best(tok, model, checkpoint_path.parent / 'best')
                return tok, model, history
            model.load_state_dict(state['model'])
            opt.load_state_dict(state['optimizer'])
            sched.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            first_epoch, next_batch = state['epoch'], state['next_batch']
            batches, total, seen = state['batches'], state['total'], state['seen']
            elapsed = state['elapsed']
            best_f1, best_state, history = state['best_f1'], state['best_state'], state['history']
            _restore_rng(state['rng'], rng)
            del state
            log(f'Resuming epoch {first_epoch + 1}/{epochs}, next batch {next_batch + 1}'
                if first_epoch < epochs else f'Training already complete ({epochs} epochs)')

    def save_checkpoint(epoch, next_offset, epoch_batches_, loss_total, rows_seen, seconds):
        if checkpoint_path is not None:
            _atomic_checkpoint(checkpoint_path, {
                'identity': identity, 'model': model.state_dict(),
                'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                'scaler': scaler.state_dict(), 'rng': _rng_state(rng),
                'epoch': epoch, 'next_batch': next_offset, 'batches': epoch_batches_,
                'total': loss_total, 'seen': rows_seen, 'elapsed': seconds,
                'best_f1': best_f1, 'best_state': best_state, 'history': history})

    for ep in range(first_epoch, epochs):
        model.train()
        t0 = time.time()
        if batches is None:
            batches = epoch_batches()
            next_batch, total, seen, elapsed = 0, 0.0, 0, 0.0
            save_checkpoint(ep, 0, batches, total, seen, elapsed)
        for step in progress(range(next_batch, len(batches)), total=len(batches),
                             initial=next_batch, desc=f'Epoch {ep + 1}/{epochs} training', unit='batches'):
            b = batches[step]
            ids, mask = _pad(tok, [data[i][0] for i in b], device)
            rows = torch.cat([torch.full((len(data[i][1]),), r, dtype=torch.long)
                              for r, i in enumerate(b)]).to(device)
            cols = torch.cat([torch.tensor(data[i][1], dtype=torch.long) for i in b]).to(device)
            y = torch.cat([torch.tensor(data[i][2], dtype=torch.float32) for i in b]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            logp = torch.log_softmax(logits.float()[rows, cols], -1)
            loss = -(y * logp[:, 1] + (1 - y) * logp[:, 0]).mean()
            if span_f1_weight > 0:
                weights = torch.cat([torch.tensor(character_data[i][0], dtype=torch.float32)
                                     for i in b]).to(device)
                gold_fractions = torch.cat([torch.tensor(character_data[i][1], dtype=torch.float32)
                                            for i in b]).to(device)
                loss = loss + span_f1_weight * soft_span_f1_loss(logp[:, 1].exp(), weights, gold_fractions)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
            seen += len(b)
            if checkpoint_path is not None and ((step + 1) % checkpoint_every == 0
                                                 or step + 1 == len(batches)):
                # Save before dev evaluation so a disconnect there never retrains an epoch.
                save_checkpoint(ep, step + 1, batches, total, seen, elapsed + time.time() - t0)
            if (step + 1) % 200 == 0:
                log(f'  epoch {ep + 1} step {step + 1}/{len(batches)} loss {total / seen:.4f} '
                    f'({time.time() - t0:.0f}s)')
        dev_tokens = predict(model, tok, dev_samples, max_length, batch_tokens=batch_tokens,
                             fp16=fp16, encs=dev_encs, progress_desc=f'Epoch {ep + 1}/{epochs} validation')
        f1 = score(decode_all(dev_tokens, dev_samples, BASELINE), dev_samples)['all'][2]
        history.append({'epoch': ep + 1, 'loss': round(total / max(seen, 1), 4), 'dev_span_f1': f1,
                        'seconds': round(elapsed + time.time() - t0)})
        log(f'Epoch {ep + 1}/{epochs}: loss {total / max(seen, 1):.4f}, dev span F1 {f1:.2f} '
            f'({time.time() - t0:.0f}s)')
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
        save_checkpoint(ep + 1, 0, None, 0.0, 0, 0.0)
        batches = None
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        _atomic_checkpoint(checkpoint_path, _completed_checkpoint(
            identity, epochs, best_state, best_f1, history))
        _export_best(tok, model, checkpoint_path.parent / 'best')
    return tok, model, history
