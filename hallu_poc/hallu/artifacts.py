"""Atomic, content-checked experiment artifacts (old primitive pickles remain readable)."""
import hashlib
import json
import os
import pickle


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def sample_fingerprint(samples):
    return fingerprint([{k: s[k] for k in ('id', 'source_id', 'task', 'split',
                                          'prompt', 'answer', 'spans')} for s in samples])


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, value, binary=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb' if binary else 'w', **({} if binary else {'encoding': 'utf-8'})) as fh:
        if binary:
            pickle.dump(value, fh, protocol=4)
        else:
            json.dump(value, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Prediction caches must contain primitive values only')


def read_pickle(path):
    with open(path, 'rb') as fh:
        return _PrimitiveUnpickler(fh).load()


def read_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def aligned(record, samples, key='tokens'):
    ids = record['ids']
    values = record[key]
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError('Cache has duplicate IDs or unequal ID/value lengths')
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Requested data has duplicate IDs')
    by = dict(zip(ids, values))
    missing = [s['id'] for s in samples if s['id'] not in by]
    if missing:
        raise ValueError(f'{len(missing)} requested answers have no cached prediction')
    if 'samples' in record:
        for s in samples:
            if record['samples'].get(s['id']) != sample_fingerprint([s]):
                raise ValueError(f'Cached data differs for answer {s["id"]}')
    return [by[s['id']] for s in samples]


def prediction_record(samples, tokens, metadata):
    if len(samples) != len(tokens):
        raise ValueError('Prediction count differs from sample count')
    return {'ids': [s['id'] for s in samples], 'tokens': tokens, 'metadata': metadata,
            'samples': {s['id']: sample_fingerprint([s]) for s in samples}}


def is_finished(path, config):
    if not os.path.exists(path):
        return False
    saved = read_json(path)
    if saved['config'] != config:
        raise ValueError('Run configuration changed; choose a new --name or --work-dir')
    return all(os.path.exists(f) and file_hash(f) == digest
               for f, digest in saved['outputs'].items())


def mark_finished(path, config, outputs):
    atomic_write(path, {'config': config, 'outputs': {f: file_hash(f) for f in outputs}})
