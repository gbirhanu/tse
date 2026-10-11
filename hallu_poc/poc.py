"""RAGTruth span detection: predict, train, ensemble, levers, report.

Own-model decoders are fitted on the source-held-out dev split. Only the
published LettuceDetect model may use two-fold source cross-fitting. An
ensemble containing that model uses fixed decoding because its training
included our dev set. Member lists and weights must be chosen before testing.

All new runs save checked, resumable outputs in --work-dir. Repeating the same
command skips finished work; changed configurations require a new run name.
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hallu import data as hdata
from hallu.artifacts import (aligned, atomic_write, file_hash, fingerprint, is_finished,
                            mark_finished, prediction_record, read_json, read_pickle,
                            sample_fingerprint)
from hallu.decoding import BASELINE, decode_all, tune
from hallu.ensemble import ensemble_tokens, fingerprint_tokenizer
from hallu.experiments import (PUBLISHED_LD, legacy_metadata, reference_spans,
                              require_fit, write_results)
from hallu.scoring import bootstrap, score


def get_device(name):
    if name == 'auto':
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def paths(work):
    work = os.path.abspath(work)
    p = {key: os.path.join(work, key) for key in ('preds', 'results', 'models')}
    for directory in p.values():
        os.makedirs(directory, exist_ok=True)
    return p


def splits(args):
    samples = hdata.load_ragtruth(args.data_dir)
    train = [s for s in samples if s['split'] == 'train']
    test = [s for s in samples if s['split'] == 'test']
    train, dev = hdata.split_dev(train, frac=0.1, seed=0)
    train = hdata.limit_by_source(train, args.limit)
    dev = hdata.limit_by_source(dev, args.limit and max(args.limit // 5, 20))
    test = hdata.limit_by_source(test, args.test_limit)
    if not test or not dev:
        raise ValueError('The requested dataset must have nonempty dev and test splits.')
    return train, dev, test


def _pred_path(p, name, split):
    return os.path.join(p['preds'], f'{name}_{split}.pkl')


def save_preds(p, name, split, samples, tokens, metadata):
    atomic_write(_pred_path(p, name, split), prediction_record(samples, tokens, metadata),
                 binary=True)


def load_preds(p, name, split, samples):
    return aligned(read_pickle(_pred_path(p, name, split)), samples)


def _metadata(p, name, record):
    return record.get('metadata') or legacy_metadata(p, name)


def _start(p, name, step, config, protected=()):
    """Freeze a configuration before work and reject accidental cache reuse."""
    config_path = os.path.join(p['results'], f'{name}_{step}_config.json')
    done_path = os.path.join(p['results'], f'{name}_{step}_done.json')
    if os.path.exists(config_path):
        if read_json(config_path) != config:
            raise ValueError(f'{name}/{step} has different settings or inputs; use a new --name.')
    elif any(os.path.exists(path) for path in protected):
        raise ValueError(f'{name} has existing outputs without this run configuration. '
                         'They are preserved; use a new --name.')
    else:
        atomic_write(config_path, config)
    if is_finished(done_path, config):
        print(f'{name} {step}: already finished; verified saved outputs.')
        return done_path, True
    if os.path.exists(done_path):
        saved = read_json(done_path)
        if saved.get('config') != config:
            raise ValueError(f'{name}/{step} completion record has different settings.')
        for path, digest in saved.get('outputs', {}).items():
            if os.path.exists(path) and file_hash(path) != digest:
                raise ValueError(f'Saved output changed after completion: {path}. '
                                 'Preserve this run and use a new --name.')
    return done_path, False


def _model_identity(name):
    """Detect changes to a local checkpoint even when its directory is reused."""
    if not os.path.isdir(name):
        return {'name': name}
    files = {}
    for filename in sorted(os.listdir(name)):
        path = os.path.join(name, filename)
        if os.path.isfile(path) and filename.endswith(('.json', '.safetensors', '.bin', '.txt', '.model')):
            files[filename] = file_hash(path)
    return {'name': os.path.abspath(name), 'files': files}


def _reference_input(p, against, test):
    if not against:
        return None
    reference_spans(p, against, test)
    run, _ = against.rsplit(':', 1)
    path = os.path.join(p['preds'], f'{run}_spans.pkl')
    return {'reference': against, 'sha256': file_hash(path)}


def _predict_resumable(args, p, split, samples, tok, model, metadata):
    """Save every 64 answers; interruptions repeat at most the current shard."""
    from hallu import encoder
    path = _pred_path(p, args.name, split)
    partial_path = path + '.partial'
    identity = {'metadata': metadata, 'data': sample_fingerprint(samples),
                'max_length': args.max_length, 'batch_tokens': args.batch_tokens,
                'fp16': not args.no_fp16}
    if os.path.exists(path):
        record = read_pickle(path)
        if record.get('metadata') != metadata or set(record['ids']) != {s['id'] for s in samples}:
            raise ValueError(f'{path} belongs to a different run; choose a new --name.')
        return ensemble_tokens([aligned(record, samples)], samples)
    tokens = []
    if os.path.exists(partial_path):
        partial = read_pickle(partial_path)
        if partial.get('config') != identity:
            raise ValueError('Partial prediction configuration changed; use a new --name.')
        count = len(partial['ids'])
        if count > len(samples) or partial['ids'] != [s['id'] for s in samples[:count]]:
            raise ValueError('Partial predictions no longer match the requested answer order.')
        tokens = aligned(partial, samples[:count])
        ensemble_tokens([tokens], samples[:count])
        print(f'{args.name} {split}: resuming after {count}/{len(samples)} answers.')
    for start in range(len(tokens), len(samples), 64):
        batch = samples[start:start + 64]
        predicted = encoder.predict(model, tok, batch, args.max_length,
                                    batch_tokens=args.batch_tokens, fp16=not args.no_fp16)
        ensemble_tokens([predicted], batch)
        tokens.extend(predicted)
        record = prediction_record(samples[:len(tokens)], tokens, metadata)
        record['config'] = identity
        atomic_write(partial_path, record, binary=True)
        print(f'{args.name} {split}: saved {len(tokens)}/{len(samples)} answers.')
    save_preds(p, args.name, split, samples, tokens, metadata)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return tokens


def cmd_predict(args, p):
    _, dev, test = splits(args)
    jobs = [('test', test)] + ([('dev', dev)] if args.with_dev else [])
    config = {'format': 2, 'model': _model_identity(args.model),
              'splits': {split: sample_fingerprint(samples) for split, samples in jobs},
              'max_length': args.max_length, 'batch_tokens': args.batch_tokens,
              'fp16': not args.no_fp16, 'device': args.device}
    outputs = [_pred_path(p, args.name, split) for split, _ in jobs]
    done, finished = _start(p, args.name, 'predict', config, outputs)
    if finished:
        return
    from hallu import encoder
    device = get_device(args.device)
    tok, model = encoder.load(args.model, device)
    local_meta = os.path.join(args.model, 'hallu_run.json')
    metadata = read_json(local_meta) if os.path.isfile(local_meta) else {
        'model': args.model,
        'training_scope': 'ragtruth_all_train' if args.model == PUBLISHED_LD else 'unknown',
    }
    metadata.update({'tokenizer_hash': fingerprint_tokenizer(tok),
                     'max_length': args.max_length, 'run_config': fingerprint(config)})
    for split, samples in jobs:
        t0 = time.time()
        tokens = _predict_resumable(args, p, split, samples, tok, model, metadata)
        f = score(decode_all(tokens, samples, BASELINE), samples)
        print(f'{args.name} {split}: {len(samples)} answers in {time.time() - t0:.0f}s; '
              f'span F1 (LD) {f["all"][2]:.2f}.')
    mark_finished(done, config, outputs)


def cmd_train(args, p):
    train, dev, test = splits(args)
    config = {'format': 2, 'base': _model_identity(args.base), 'targets': args.targets,
              'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
              'max_length': args.max_length, 'batch_tokens': args.batch_tokens,
              'fp16': not args.no_fp16, 'device': args.device,
              'gradient_checkpointing': args.gradient_checkpointing,
              'data': {split: sample_fingerprint(samples)
                       for split, samples in [('train', train), ('dev', dev), ('test', test)]}}
    training_path = os.path.join(p['results'], f'{args.name}_training.json')
    outputs = [_pred_path(p, args.name, split) for split in ('dev', 'test')] + [training_path]
    done, finished = _start(p, args.name, 'train', config, outputs)
    if finished:
        return
    from hallu import encoder
    device = get_device(args.device)
    checkpoint_dir = os.path.join(p['models'], args.name)
    print(f'Training {args.base}, {args.targets}, seed {args.seed}: {len(train)} train, '
          f'{len(dev)} dev, {len(test)} test answers on {device}.')
    tok, model, history = encoder.train(
        args.base, train, dev, targets=args.targets, epochs=args.epochs, lr=args.lr,
        max_length=args.max_length, batch_tokens=args.batch_tokens, seed=args.seed,
        device=device, fp16=not args.no_fp16, checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every, gradient_checkpointing=args.gradient_checkpointing)
    metadata = {'model': args.base, 'training_scope': 'train_without_dev',
                'targets': args.targets, 'seed': args.seed, 'max_length': args.max_length,
                'tokenizer_hash': fingerprint_tokenizer(tok), 'run_config': fingerprint(config),
                'train_fingerprint': sample_fingerprint(train), 'dev_fingerprint': sample_fingerprint(dev),
                'soft_version': 'legacy', 'kind': 'trained'}
    best_dir = os.path.join(checkpoint_dir, 'best')
    atomic_write(os.path.join(best_dir, 'hallu_run.json'), metadata)
    atomic_write(training_path, {'base': args.base, 'targets': args.targets,
                                'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
                                'history': history, 'config': config, 'metadata': metadata})
    for split, samples in [('dev', dev), ('test', test)]:
        _predict_resumable(args, p, split, samples, tok, model, metadata)
    outputs.extend(os.path.join(best_dir, filename) for filename in sorted(os.listdir(best_dir))
                   if os.path.isfile(os.path.join(best_dir, filename)))
    mark_finished(done, config, outputs)
    print(f'Best dev checkpoint saved to {best_dir}.')


def _load_member(args, p, name, split, samples, legacy_tok=None):
    record = read_pickle(_pred_path(p, name, split))
    tokens = aligned(record, samples)
    metadata = _metadata(p, name, record)
    if not record.get('metadata'):
        if legacy_tok is None:
            raise ValueError(f'{name} is a legacy cache. Supply --legacy-tokenizer to validate '
                             'its exact answer offsets; originals will remain unchanged.')
        for i, sample in enumerate(samples):
            encoded = legacy_tok(sample['prompt'], sample['answer'], truncation='only_first',
                                 max_length=args.max_length, return_offsets_mapping=True)
            offsets = [tuple(offset) for offset, sid in
                       zip(encoded['offset_mapping'], encoded.sequence_ids()) if sid == 1]
            if offsets != [tuple(token[:2]) for token in tokens[i]]:
                raise ValueError(f'{name} {split}, answer {sample["id"]}: supplied tokenizer '
                                 'does not reproduce cached offsets.')
        metadata = dict(metadata, tokenizer_hash=fingerprint_tokenizer(legacy_tok))
    if not metadata.get('tokenizer_hash'):
        raise ValueError(f'{name} has no tokenizer fingerprint; recreate its predictions.')
    return tokens, metadata


def cmd_ensemble(args, p):
    _, dev, test = splits(args)
    legacy_tok = None
    if args.legacy_tokenizer:
        from transformers import AutoTokenizer
        legacy_tok = AutoTokenizer.from_pretrained(args.legacy_tokenizer)
    jobs = [('test', test)] + ([('dev', dev)] if args.fit == 'dev' else [])
    inputs, split_tokens, metas = {}, {}, {}
    for split, samples in jobs:
        members = []
        for name in args.members:
            tokens, metadata = _load_member(args, p, name, split, samples, legacy_tok)
            if name in metas and (metas[name].get('tokenizer_hash') != metadata.get('tokenizer_hash')
                                  or metas[name].get('training_scope') != metadata.get('training_scope')
                                  or metas[name].get('run_config') != metadata.get('run_config')):
                raise ValueError(f'{name} dev and test predictions come from different runs.')
            metas[name] = metadata
            members.append(tokens)
            inputs[f'{name}_{split}'] = file_hash(_pred_path(p, name, split))
        split_tokens[split] = members
    tokenizer_hashes = {meta['tokenizer_hash'] for meta in metas.values()}
    if len(tokenizer_hashes) != 1:
        raise ValueError('Ensemble members do not share an identical tokenizer definition.')
    scope = ('train_without_dev' if all(meta.get('training_scope') == 'train_without_dev'
                                      for meta in metas.values()) else 'mixed_or_seen_dev')
    metadata = {'kind': 'ensemble', 'members': args.members, 'weights': args.weights,
                'training_scope': scope, 'tokenizer_hash': next(iter(tokenizer_hashes)),
                'member_metadata': metas}
    require_fit(metadata, args.fit, ensemble=True)
    averaged_by_split = {split: ensemble_tokens(split_tokens[split], samples, args.weights)
                         for split, samples in jobs}
    config = {'format': 2, 'members': args.members, 'weights': args.weights, 'inputs': inputs,
              'fit': args.fit, 'stacker': args.stacker, 'seed': args.seed,
              'legacy_tokenizer': args.legacy_tokenizer, 'max_length': args.max_length,
              'data': {split: sample_fingerprint(samples) for split, samples in jobs},
              'reference': _reference_input(p, args.against, test)}
    outputs = [_pred_path(p, args.name, split) for split, _ in jobs]
    done, finished = _start(p, args.name, 'ensemble', config, outputs)
    if finished:
        print_table(read_json(os.path.join(p['results'], f'{args.name}_levers.json'))['rows'])
        return
    metadata['run_config'] = fingerprint(config)
    for split, samples in jobs:
        save_preds(p, args.name, split, samples, averaged_by_split[split], metadata)
    cmd_levers(args, p)
    outputs.extend([os.path.join(p['results'], f'{args.name}_levers.json'),
                    os.path.join(p['preds'], f'{args.name}_spans.pkl')])
    mark_finished(done, config, outputs)


def cmd_levers(args, p):
    _, dev, test = splits(args)
    test_path = _pred_path(p, args.name, 'test')
    record = read_pickle(test_path)
    metadata = _metadata(p, args.name, record)
    require_fit(metadata, args.fit, ensemble=metadata.get('kind') == 'ensemble')
    result_path = os.path.join(p['results'], f'{args.name}_levers.json')
    spans_path = os.path.join(p['preds'], f'{args.name}_spans.pkl')
    config_path = os.path.join(p['results'], f'{args.name}_levers_config.json')
    # Previously reported experiments remain unchanged and serve as references.
    if not record.get('metadata') and os.path.exists(result_path) and not os.path.exists(config_path):
        old = read_json(result_path)
        if args.against or args.stacker or old.get('fit') != args.fit or not os.path.exists(spans_path):
            raise ValueError('Existing legacy results are preserved. Use them as a reference '
                             'for a new named run instead of rewriting them.')
        reference_spans(p, args.name + ':baseline', test)
        print(f'{args.name}: preserving existing legacy results.')
        print_table(old['rows'])
        return
    test_tokens = ensemble_tokens([aligned(record, test)], test)
    inputs = {'test': file_hash(test_path)}
    dev_tokens = None
    if args.fit == 'dev':
        dev_path = _pred_path(p, args.name, 'dev')
        dev_record = read_pickle(dev_path)
        dev_meta = _metadata(p, args.name, dev_record)
        if any(metadata.get(key) != dev_meta.get(key)
               for key in ('tokenizer_hash', 'run_config', 'training_scope')):
            raise ValueError('Dev and test predictions belong to different model runs.')
        dev_tokens = ensemble_tokens([aligned(dev_record, dev)], dev)
        inputs['dev'] = file_hash(dev_path)
    config = {'format': 2, 'fit': args.fit, 'stacker': args.stacker, 'seed': args.seed,
              'inputs': inputs, 'test': sample_fingerprint(test),
              'dev': sample_fingerprint(dev) if args.fit == 'dev' else None,
              'reference': _reference_input(p, args.against, test),
              'smoke': bool(args.limit is not None or args.test_limit is not None)}
    done, finished = _start(p, args.name, 'levers', config, [result_path, spans_path])
    if finished:
        print_table(read_json(result_path)['rows'])
        return
    spans = {'baseline': decode_all(test_tokens, test, BASELINE)}
    settings = {'metadata': metadata, 'baseline': BASELINE}
    jobs = []
    if args.fit == 'dev':
        jobs = [(dev, dev_tokens, list(range(len(test))))]
    elif args.fit == 'crossfit':
        folds = hdata.source_folds(test, k=2, seed=0)
        for fold in (0, 1):
            fit_idx = [i for i, value in enumerate(folds) if value != fold]
            jobs.append(([test[i] for i in fit_idx], [test_tokens[i] for i in fit_idx],
                         [i for i, value in enumerate(folds) if value == fold]))
    if jobs:
        spans['tuned'], settings['tuned'] = [None] * len(test), []
    if args.stacker:
        from hallu import stacker
        spans['stacker'], settings['stacker'] = [None] * len(test), []
    for fit_samples, fit_tokens, eval_idx in jobs:
        if not fit_samples or not eval_idx:
            raise ValueError('Calibration requires nonempty fitting and evaluation folds.')
        eval_samples = [test[i] for i in eval_idx]
        eval_tokens = [test_tokens[i] for i in eval_idx]
        params = tune(fit_tokens, fit_samples)
        settings['tuned'].append(params)
        print(f'{args.name}: fitted on {len(fit_samples)} answers; decoder {params}')
        for i, prediction in zip(eval_idx, decode_all(eval_tokens, eval_samples, params)):
            spans['tuned'][i] = prediction
        if args.stacker:
            predicted, params = stacker.fit_and_apply(fit_samples, fit_tokens,
                eval_samples, eval_tokens, get_device(args.device), seed=args.seed, log=print)
            settings['stacker'].append(params)
            for i, prediction in zip(eval_idx, predicted):
                spans['stacker'][i] = prediction
    rows = write_results(p, args.name, args.fit, spans, test, settings,
                         against=args.against, smoke=config['smoke'])
    mark_finished(done, config, [result_path, spans_path])
    print_table(rows)


def print_table(rows):
    print('\n| Run | Setting | N | Fit | Span F1 (LD) | 95% CI | P | R | Reference | Gain [95% CI] | Decision | F1 (set) | Set task macro | QA | Summary | Data2text |')
    print('|' + '|'.join(['---'] * 16) + '|')
    for row in rows:
        ld = row['ld']
        ci = row['ld_f1_ci']
        reference, gain = row.get('reference', ''), ''
        if 'reference_gain_ci' in row:
            lo, hi = row['reference_gain_ci']
            gain = f'{row["reference_gain"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'
        elif 'ld_gain_ci' in row:
            reference = row['run'] + ':baseline'
            baseline = next((r for r in rows if r['run'] == row['run']
                             and r['setting'] == 'baseline'), None)
            value = row.get('ld_gain', ld['all'][2] - baseline['ld']['all'][2] if baseline else 0)
            lo, hi = row['ld_gain_ci']
            gain = f'{value:+.2f} [{lo:+.2f}, {hi:+.2f}]'
        st = row['set']
        task_scores = [values[2] for task, values in st.items() if task != 'all']
        macro = row.get('set_task_macro', sum(task_scores) / len(task_scores) if task_scores else 0)
        decision = row.get('decision', 'legacy')
        if row.get('smoke'):
            decision = 'smoke-only'
        tasks = ' | '.join(f'{ld[task][2]:.2f}' if task in ld else ''
                           for task in ('qa', 'summary', 'data2text'))
        print(f'| {row["run"]} | {row["setting"]} | {row["n_test"]} | {row["fit"]} '
              f'| {ld["all"][2]:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] '
              f'| {ld["all"][0]:.2f} | {ld["all"][1]:.2f} | {reference} | {gain} '
              f'| {decision} | {st["all"][2]:.2f} | {macro:.2f} | {tasks} |')


def cmd_report(args, p):
    rows = []
    for filename in sorted(os.listdir(p['results'])):
        if filename.endswith('_levers.json'):
            rows.extend(read_json(os.path.join(p['results'], filename))['rows'])
    if not rows:
        print('No results yet. Run levers or ensemble first.')
        return
    print_table(rows)
    if args.compare:
        a, b = args.compare
        _, _, test = splits(args)
        print(f'\n{b} minus {a}, same test answers (span F1, LD scorer):')
        for setting in ('baseline', 'tuned'):
            try:
                pa, pb = (reference_spans(p, f'{run}:{setting}', test) for run in (a, b))
            except KeyError:
                continue
            gain = score(pb, test, digits=None)['all'][2] - score(pa, test, digits=None)['all'][2]
            lo, hi = bootstrap(pb, test, pa)
            print(f'  {setting:8s} {gain:+.2f}; 95% CI [{lo:+.2f}, {hi:+.2f}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['predict', 'train', 'ensemble', 'levers', 'report'])
    ap.add_argument('--data-dir', default='RAGTruth')
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--name', help='unique run name; changed settings require a new name')
    ap.add_argument('--model', default=PUBLISHED_LD)
    ap.add_argument('--base', default='answerdotai/ModernBERT-base')
    ap.add_argument('--targets', choices=['hard', 'soft'], default='hard')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-length', type=int, default=4096)
    ap.add_argument('--batch-tokens', type=int)
    ap.add_argument('--fit', choices=['dev', 'crossfit', 'fixed'], default='dev')
    ap.add_argument('--members', nargs='+', help='ensemble: predeclared unique run names')
    ap.add_argument('--weights', nargs='+', type=float, help='ensemble: predeclared nonnegative weights')
    ap.add_argument('--legacy-tokenizer', help='tokenizer used to verify original prediction caches')
    ap.add_argument('--against', metavar='RUN:SETTING', help='declared previous best for paired acceptance')
    ap.add_argument('--with-dev', action='store_true')
    ap.add_argument('--stacker', action='store_true', help='also fit the optional character stacker')
    ap.add_argument('--save-model', action='store_true', help='compatibility flag; training always saves its best model')
    ap.add_argument('--checkpoint-every', type=int, default=200)
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--no-fp16', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--limit', type=int, help='about N training answers; marks results as smoke tests')
    ap.add_argument('--test-limit', type=int, help='about N test answers; marks results as smoke tests')
    ap.add_argument('--compare', nargs=2, metavar=('RUN_A', 'RUN_B'))
    args = ap.parse_args()
    if args.step != 'report' and not args.name:
        ap.error('--name is required')
    names = ([args.name] if args.name else []) + (args.members or []) + (args.compare or [])
    if args.against:
        if ':' not in args.against:
            ap.error('--against must be RUN:SETTING')
        run, setting = args.against.rsplit(':', 1)
        if setting not in ('baseline', 'tuned', 'stacker') or run == args.name:
            ap.error('--against must identify another run and baseline, tuned, or stacker')
        names.append(run)
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) for name in names):
        ap.error('Run names must use letters, digits, periods, underscores, and hyphens.')
    if args.step == 'ensemble':
        if not args.members or len(args.members) < 2 or len(set(args.members)) != len(args.members):
            ap.error('ensemble requires at least two unique --members')
        if args.name in args.members:
            ap.error('The ensemble name must differ from its member names')
        if args.fit == 'crossfit':
            ap.error('Cross-fitting is reserved for the published LD model, never an ensemble')
    if args.stacker and args.fit == 'fixed':
        ap.error('--stacker cannot be combined with fixed decoding')
    if args.batch_tokens is None:
        args.batch_tokens = 16384 if args.step == 'predict' else 8192
    if any(value is not None and value < 1 for value in
           (args.limit, args.test_limit, args.batch_tokens, args.max_length, args.epochs, args.checkpoint_every)):
        ap.error('Limits, epochs, context, batch budget and checkpoint interval must be positive')
    try:
        p = paths(args.work_dir)
        {'predict': cmd_predict, 'train': cmd_train, 'ensemble': cmd_ensemble,
         'levers': cmd_levers, 'report': cmd_report}[args.step](args, p)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        ap.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
