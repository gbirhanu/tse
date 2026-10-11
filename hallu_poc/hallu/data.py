"""RAGTruth loading and splits."""
import json
import os
import random

TASKS = {'QA': 'qa', 'Summary': 'summary', 'Data2txt': 'data2text'}


def _find(data_dir, name):
    for d in (data_dir, os.path.join(data_dir, 'dataset')):
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{name} not found in {data_dir} or {data_dir}/dataset')


def load_ragtruth(data_dir):
    """
    One dict per response: id, source_id, task, split, prompt, answer, spans.
    spans = list of (start, end) character offsets in the answer, as annotated.
    The prompt is RAGTruth's full prompt, as LettuceDetect uses it.
    """
    with open(_find(data_dir, 'source_info.jsonl'), encoding='utf-8') as fh:
        sources = {s['source_id']: s for s in map(json.loads, fh)}
    samples = []
    with open(_find(data_dir, 'response.jsonl'), encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            s = sources[r['source_id']]
            samples.append({'id': str(r['id']), 'source_id': str(r['source_id']),
                            'task': TASKS[s['task_type']], 'split': r['split'],
                            'prompt': s['prompt'], 'answer': r['response'],
                            'spans': [(int(x['start']), int(x['end'])) for x in r['labels']]})
    return samples


def split_dev(train, frac=0.1, seed=0):
    """Hold out a fraction of the training *sources* (all 6 responses of a source go together)."""
    sources = sorted({s['source_id'] for s in train})
    random.Random(seed).shuffle(sources)
    dev_sources = set(sources[:int(len(sources) * frac)])
    return ([s for s in train if s['source_id'] not in dev_sources],
            [s for s in train if s['source_id'] in dev_sources])


def source_folds(samples, k=2, seed=0):
    """Fold number per sample; all responses of a source share a fold."""
    sources = sorted({s['source_id'] for s in samples})
    random.Random(seed).shuffle(sources)
    fold_of = {src: i % k for i, src in enumerate(sources)}
    return [fold_of[s['source_id']] for s in samples]


def limit_by_source(samples, n, seed=0):
    """About n samples, taking whole sources (for quick tests)."""
    if not n:
        return samples
    sources = sorted({s['source_id'] for s in samples})
    random.Random(seed).shuffle(sources)
    keep, count = set(), 0
    per = {}
    for s in samples:
        per[s['source_id']] = per.get(s['source_id'], 0) + 1
    for src in sources:
        if count >= n:
            break
        keep.add(src)
        count += per[src]
    return [s for s in samples if s['source_id'] in keep]
