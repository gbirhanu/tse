"""
Proof of concept: can the 2021 tweet-span tricks raise span F1 on RAGTruth?

    python poc.py predict --model KRLabsOrg/lettucedect-large-modernbert-en-v1 --name ld_large
    python poc.py levers  --name ld_large --fit crossfit          # Part A
    python poc.py train   --base answerdotai/ModernBERT-base --targets hard --name mb_hard
    python poc.py train   --base answerdotai/ModernBERT-base --targets soft --name mb_soft
    python poc.py levers  --name mb_hard --fit dev
    python poc.py levers  --name mb_soft --fit dev                # Part B
    python poc.py report  --compare mb_hard mb_soft

Common options: --data-dir (RAGTruth: response.jsonl, source_info.jsonl),
--work-dir (outputs), --limit / --test-limit (quick tests), --device.

Levers: baseline = LettuceDetect's decoding; tuned = lever 1 (threshold, gap
merge, word snapping per task); stacker = lever 3 (character BiGRU + tuned
decoding). They are fitted on data the classifier did not train on: the dev
split (10% of training sources) for our own models, or, for the published
LettuceDetect model (trained on all of RAGTruth train), the other half of the
test set ("crossfit", 2 folds by source). Scores: character-level span F1 with
LettuceDetect's scorer (ld) and with character sets (set), 95% bootstrap
intervals over test sources.
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hallu import data as hdata                                   # noqa: E402
from hallu.decoding import BASELINE, decode_all, tune             # noqa: E402
from hallu.scoring import bootstrap, score                        # noqa: E402


def get_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def paths(work):
    p = {k: os.path.join(work, k) for k in ('preds', 'results', 'models')}
    for d in p.values():
        os.makedirs(d, exist_ok=True)
    return p


def splits(args):
    samples = hdata.load_ragtruth(args.data_dir)
    train = [s for s in samples if s['split'] == 'train']
    test = [s for s in samples if s['split'] == 'test']
    train, dev = hdata.split_dev(train, frac=0.1, seed=0)
    train = hdata.limit_by_source(train, args.limit)
    dev = hdata.limit_by_source(dev, args.limit and max(args.limit // 5, 20))
    test = hdata.limit_by_source(test, args.test_limit)
    return train, dev, test


def save_preds(p, name, split, samples, tokens):
    with open(os.path.join(p['preds'], f'{name}_{split}.pkl'), 'wb') as fh:
        pickle.dump({'ids': [s['id'] for s in samples], 'tokens': tokens}, fh)


def load_preds(p, name, split, samples):
    path = os.path.join(p['preds'], f'{name}_{split}.pkl')
    if not os.path.exists(path):
        sys.exit(f'{path} is missing. Run predict (or train) for {name} first.')
    with open(path, 'rb') as fh:
        d = pickle.load(fh)
    by_id = dict(zip(d['ids'], d['tokens']))
    missing = [s['id'] for s in samples if s['id'] not in by_id]
    if missing:
        sys.exit(f'{len(missing)} {split} answers have no prediction in {path} '
                 f'(was it made with a different --limit?)')
    return [by_id[s['id']] for s in samples]


def cmd_predict(args, p):
    from hallu import encoder
    _, dev, test = splits(args)
    device = get_device(args.device)
    tok, model = encoder.load(args.model, device)
    for split, samples in (('test', test), ('dev', dev)) if args.with_dev else (('test', test),):
        t0 = time.time()
        tokens = encoder.predict(model, tok, samples, args.max_length, args.batch_tokens,
                                 fp16=(device == 'cuda'))
        save_preds(p, args.name, split, samples, tokens)
        f = score(decode_all(tokens, samples, BASELINE), samples)
        print(f'{args.name} {split}: {len(samples)} answers in {time.time() - t0:.0f}s; '
              f'span F1 (LettuceDetect scorer, its own decoding) {f["all"][2]:.2f}  '
              f'P {f["all"][0]:.2f} R {f["all"][1]:.2f}')


def cmd_train(args, p):
    from hallu import encoder
    train, dev, test = splits(args)
    device = get_device(args.device)
    print(f'Training {args.base} with {args.targets} targets on {len(train)} answers '
          f'(dev {len(dev)}, test {len(test)}) on {device}')
    tok, model, history = encoder.train(
        args.base, train, dev, targets=args.targets, epochs=args.epochs, lr=args.lr,
        max_length=args.max_length, batch_tokens=args.batch_tokens, seed=args.seed,
        device=device, fp16=not args.no_fp16)
    for split, samples in (('dev', dev), ('test', test)):
        tokens = encoder.predict(model, tok, samples, args.max_length, fp16=not args.no_fp16)
        save_preds(p, args.name, split, samples, tokens)
    with open(os.path.join(p['results'], f'{args.name}_training.json'), 'w') as fh:
        json.dump({'base': args.base, 'targets': args.targets, 'epochs': args.epochs, 'lr': args.lr,
                   'seed': args.seed, 'history': history}, fh, indent=2)
    if args.save_model:
        out = os.path.join(p['models'], args.name)
        model.half().save_pretrained(out)
        tok.save_pretrained(out)
        print(f'Model saved to {out}')


def cmd_levers(args, p):
    from hallu import stacker
    _, dev, test = splits(args)
    device = get_device(args.device)
    test_tokens = load_preds(p, args.name, 'test', test)
    spans = {'baseline': decode_all(test_tokens, test, BASELINE),
             'tuned': [None] * len(test), 'stacker': [None] * len(test)}
    settings = {'tuned': [], 'stacker': []}
    if args.fit == 'dev':
        dev_tokens = load_preds(p, args.name, 'dev', dev)
        jobs = [(dev, dev_tokens, list(range(len(test))))]
    else:
        folds = hdata.source_folds(test, k=2, seed=0)
        jobs = []
        for f in (0, 1):
            fit_idx = [i for i, x in enumerate(folds) if x != f]
            jobs.append(([test[i] for i in fit_idx], [test_tokens[i] for i in fit_idx],
                         [i for i, x in enumerate(folds) if x == f]))
    for n, (fit_s, fit_t, eval_idx) in enumerate(jobs):
        print(f'  fitting levers on {len(fit_s)} answers, applying to {len(eval_idx)} '
              f'({"dev" if args.fit == "dev" else f"test fold {n + 1}/2"})')
        ev_s = [test[i] for i in eval_idx]
        ev_t = [test_tokens[i] for i in eval_idx]
        params = tune(fit_t, fit_s)
        settings['tuned'].append(params)
        print(f'    tuned decoder settings: {params}')
        for i, sp in zip(eval_idx, decode_all(ev_t, ev_s, params)):
            spans['tuned'][i] = sp
        st_spans, st_params = stacker.fit_and_apply(fit_s, fit_t, ev_s, ev_t, device,
                                                    seed=args.seed, log=print)
        settings['stacker'].append(st_params)
        for i, sp in zip(eval_idx, st_spans):
            spans['stacker'][i] = sp

    rows = []
    for name in ('baseline', 'tuned', 'stacker'):
        ld, st = score(spans[name], test, 'ld'), score(spans[name], test, 'set')
        row = {'run': args.name, 'setting': name, 'fit': args.fit, 'n_test': len(test),
               'ld': ld, 'set': st, 'ld_f1_ci': bootstrap(spans[name], test, kind='ld')}
        if name != 'baseline':
            row['ld_gain_ci'] = bootstrap(spans[name], test, spans['baseline'], kind='ld')
        rows.append(row)
    out = {'run': args.name, 'fit': args.fit, 'rows': rows, 'settings': settings}
    with open(os.path.join(p['results'], f'{args.name}_levers.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(p['preds'], f'{args.name}_spans.pkl'), 'wb') as fh:
        pickle.dump({'ids': [s['id'] for s in test], 'spans': spans}, fh)
    print_table(rows)


def print_table(rows):
    print('\n| Run | Setting | Span F1 (LD) | 95% CI | P | R | Gain vs baseline [95% CI] | Span F1 (set) | QA | Summary | Data2text |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        ld = r['ld']
        gain = (f'{ld["all"][2] - next(x for x in rows if x["run"] == r["run"] and x["setting"] == "baseline")["ld"]["all"][2]:+.2f} '
                f'[{r["ld_gain_ci"][0]:+.2f}, {r["ld_gain_ci"][1]:+.2f}]') if 'ld_gain_ci' in r else ''
        print(f'| {r["run"]} | {r["setting"]} | {ld["all"][2]:.2f} | {r["ld_f1_ci"][0]:.1f}–{r["ld_f1_ci"][1]:.1f} '
              f'| {ld["all"][0]:.1f} | {ld["all"][1]:.1f} | {gain} | {r["set"]["all"][2]:.2f} '
              f'| {ld.get("qa", ("", "", ""))[2]} | {ld.get("summary", ("", "", ""))[2]} '
              f'| {ld.get("data2text", ("", "", ""))[2]} |')


def cmd_report(args, p):
    rows = []
    for f in sorted(os.listdir(p['results'])):
        if f.endswith('_levers.json'):
            with open(os.path.join(p['results'], f)) as fh:
                rows += json.load(fh)['rows']
    if not rows:
        sys.exit('No results yet. Run levers first.')
    print_table(rows)
    if args.compare:
        a, b = args.compare
        _, _, test = splits(args)
        sp = {}
        for run in (a, b):
            with open(os.path.join(p['preds'], f'{run}_spans.pkl'), 'rb') as fh:
                d = pickle.load(fh)
            by = {k: dict(zip(d['ids'], v)) for k, v in d['spans'].items()}
            sp[run] = {k: [v[s['id']] for s in test] for k, v in by.items()}
        print(f'\n{b} minus {a}, same test answers (span F1, LettuceDetect scorer):')
        for setting in ('baseline', 'tuned', 'stacker'):
            fa = score(sp[a][setting], test)['all'][2]
            fb = score(sp[b][setting], test)['all'][2]
            lo, hi = bootstrap(sp[b][setting], test, sp[a][setting])
            print(f'  {setting:8s} {fb - fa:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['predict', 'train', 'levers', 'report'])
    ap.add_argument('--data-dir', default='RAGTruth')
    ap.add_argument('--work-dir', default='work')
    ap.add_argument('--name', default=None, help='run name (files in work/preds, work/results)')
    ap.add_argument('--model', default='KRLabsOrg/lettucedect-large-modernbert-en-v1', help='predict: model to run')
    ap.add_argument('--base', default='answerdotai/ModernBERT-base', help='train: model to fine-tune')
    ap.add_argument('--targets', choices=['hard', 'soft'], default='hard')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-length', type=int, default=4096)
    ap.add_argument('--batch-tokens', type=int, default=None)
    ap.add_argument('--fit', choices=['dev', 'crossfit'], default='dev')
    ap.add_argument('--with-dev', action='store_true', help='predict: also predict the dev split')
    ap.add_argument('--save-model', action='store_true')
    ap.add_argument('--no-fp16', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--limit', type=int, default=None, help='about N training answers (quick tests)')
    ap.add_argument('--test-limit', type=int, default=None, help='about N test answers (quick tests)')
    ap.add_argument('--compare', nargs=2, default=None, metavar=('RUN_A', 'RUN_B'))
    args = ap.parse_args()
    if args.step in ('predict', 'train', 'levers') and not args.name:
        ap.error('--name is required')
    if args.batch_tokens is None:
        args.batch_tokens = 16384 if args.step == 'predict' else 8192
    p = paths(args.work_dir)
    {'predict': cmd_predict, 'train': cmd_train, 'levers': cmd_levers, 'report': cmd_report}[args.step](args, p)


if __name__ == '__main__':
    main()
