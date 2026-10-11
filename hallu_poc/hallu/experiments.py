"""Leakage guards and declared-reference, source-paired experiment decisions."""
import os

from hallu.artifacts import (aligned, atomic_write, read_json, read_pickle,
                            sample_fingerprint)
from hallu.decoding import BASELINE, decode_all
from hallu.scoring import bootstrap, score

PUBLISHED_LD = 'KRLabsOrg/lettucedect-large-modernbert-en-v1'


def require_fit(metadata, fit, ensemble=False):
    scope = metadata.get('training_scope')
    if fit == 'fixed':
        return
    if fit == 'dev' and scope == 'train_without_dev':
        return
    if (fit == 'crossfit' and not ensemble and scope == 'ragtruth_all_train'
            and metadata.get('model') == PUBLISHED_LD):
        return
    raise ValueError(f'Illegal calibration: {fit} for {scope}. Dev must be unseen by every '
                     'member; crossfit is reserved for the published LettuceDetect model. '
                     'Use fixed decoding for mixed LD ensembles.')


def reference_spans(p, reference, test):
    try:
        run, setting = reference.rsplit(':', 1)
    except ValueError as exc:
        raise ValueError('--against must be RUN:SETTING, e.g. ld_large:baseline') from exc
    record = read_pickle(os.path.join(p['preds'], f'{run}_spans.pkl'))
    if 'test_fingerprint' in record and record['test_fingerprint'] != sample_fingerprint(test):
        raise ValueError('Reference uses different test data; do not compare full and smoke runs')
    if set(record['ids']) != {s['id'] for s in test}:
        raise ValueError('Reference and candidate must have exactly the same test answers')
    if 'test_fingerprint' not in record:
        legacy_metadata(p, run)
        cached = read_pickle(os.path.join(p['preds'], f'{run}_test.pkl'))
        baseline = decode_all(aligned(cached, test), test, BASELINE)
        saved_baseline = aligned({'ids': record['ids'], 'tokens': record['spans']['baseline']}, test)
        if baseline != saved_baseline:
            raise ValueError('Legacy reference baseline disagrees with its token predictions')
        result = read_json(os.path.join(p['results'], f'{run}_levers.json'))
        old = next(r for r in result['rows'] if r['setting'] == setting)
        predicted = aligned({'ids': record['ids'], 'tokens': record['spans'][setting]}, test)
        current = {k: list(v) for k, v in score(predicted, test).items()}
        if current != old['ld']:
            raise ValueError('Legacy reference no longer reproduces its recorded LD scores')
    return aligned({'ids': record['ids'], 'tokens': record['spans'][setting]}, test)


def result_rows(name, fit, spans, test, p, against=None, smoke=False):
    rows = []
    base = reference_spans(p, against, test) if against else None
    selected = 'tuned' if 'tuned' in spans else 'baseline'
    for setting, predicted in spans.items():
        ld, st = score(predicted, test, 'ld'), score(predicted, test, 'set')
        row = {'run': name, 'setting': setting, 'fit': fit, 'n_test': len(test),
               'test_fingerprint': sample_fingerprint(test), 'smoke': smoke,
               'ld': ld, 'set': st, 'ld_f1_ci': bootstrap(predicted, test),
               'set_task_macro': sum(st[t][2] for t in st if t != 'all') / (len(st) - 1)}
        if setting != 'baseline':
            row['ld_gain_ci'] = bootstrap(predicted, test, spans['baseline'])
            row['ld_gain'] = ld['all'][2] - score(spans['baseline'], test)['all'][2]
        if setting == selected and base is not None:
            gain = (score(predicted, test, digits=None)['all'][2]
                    - score(base, test, digits=None)['all'][2])
            ci = bootstrap(predicted, test, base, digits=None)
            row.update({'reference': against, 'reference_gain': gain,
                        'reference_gain_ci': ci,
                        'decision': 'smoke-only' if smoke else
                        ('keep' if gain > 0 and ci[0] > 0 else 'reject')})
        else:
            row['decision'] = 'smoke-only' if smoke else 'not-tested'
        rows.append(row)
    return rows


def write_results(p, name, fit, spans, test, settings, against=None, smoke=False):
    rows = result_rows(name, fit, spans, test, p, against, smoke)
    atomic_write(os.path.join(p['preds'], f'{name}_spans.pkl'),
                 {'ids': [s['id'] for s in test], 'spans': spans,
                  'test_fingerprint': sample_fingerprint(test)}, binary=True)
    atomic_write(os.path.join(p['results'], f'{name}_levers.json'),
                 {'run': name, 'fit': fit, 'rows': rows, 'settings': settings})
    return rows


def legacy_metadata(p, name):
    """Only the three explicitly documented prior runs can be adopted implicitly.

    The caller must additionally validate every cached offset against a supplied
    tokenizer. Historical training provenance is the user's declaration, not
    something recoverable from probabilities.
    """
    if name == 'ld_large':
        return {'model': PUBLISHED_LD, 'training_scope': 'ragtruth_all_train',
                'provenance': 'declared original Part A run'}
    if name in ('mb_hard', 'mb_soft'):
        history = read_json(os.path.join(p['results'], f'{name}_training.json'))
        if (history.get('base') != 'answerdotai/ModernBERT-base' or history.get('seed') != 42
                or history.get('targets') != name.removeprefix('mb_')):
            raise ValueError('Legacy run does not match the documented Part B configuration')
        return {'model': history['base'], 'training_scope': 'train_without_dev',
                'targets': history['targets'], 'seed': history['seed'],
                'provenance': 'declared original Part B source split (seed=0, dev=10%)',
                'soft_version': 'legacy'}
    raise ValueError(f'{name} has no provenance. Recreate its predictions with this pipeline.')
