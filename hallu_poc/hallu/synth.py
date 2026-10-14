"""
Extra training summaries with known errors (Part E, step 3). No language model is used.

Summaries are the weakest task and have the fewest marked errors in RAGTruth's
training data (about 1,800 spans, against 8,200 for data-to-text). From each
error-free training summary we make one copy with a known error:

* swap: change one number, year, month, weekday, or person/place name so it no
  longer matches the article. A replacement name or number is often taken from
  the same article, so the model must check the context, not just the word.
  The new characters are the hallucinated span.
* move: copy a whole sentence that annotators marked as hallucinated in another
  model's summary of the same article into this summary, at a sentence break.
  The copied sentence is the hallucinated span.

Names are found with spaCy's small English model (en_core_web_sm), and a name
is only swapped for another name of the same kind (person, place, organisation,
...) from the same article. Only training sources are used (never dev or test).
Results are deterministic for a given seed, answer id and spaCy model version.
"""
import random
import re

SYNTH_VERSION = 'summary_swap_move_v1'
NAME_LABELS = ('PERSON', 'GPE', 'LOC', 'ORG', 'NORP', 'FAC', 'EVENT', 'PRODUCT')

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
NUMBER_WORDS = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                'eleven', 'twelve']

_NUMBER = re.compile(r'\d+(?:,\d{3})*(?:\.\d+)?')
_WORD_NUMBER = re.compile(r'\b(' + '|'.join(NUMBER_WORDS) + r')\b', re.IGNORECASE)
_MONTH = re.compile(r'\b(' + '|'.join(MONTHS) + r')\b')
_WEEKDAY = re.compile(r'\b(' + '|'.join(WEEKDAYS) + r')\b')
_SENTENCE_BREAK = re.compile(r'(?<=[.!?])(?=\s+["“]?[A-Z])')


def article_text(prompt):
    """RAGTruth summary prompts wrap the article in an instruction and 'output:'."""
    lines = prompt.split('\n')
    if lines and lines[0].lower().startswith('summarize'):
        lines = lines[1:]
    while lines and lines[-1].strip().lower() in ('', 'output:'):
        lines = lines[:-1]
    return '\n'.join(lines)


def _sentence_start(text, i):
    j = i - 1
    while j >= 0 and text[j] in ' \t"“\'(':
        j -= 1
    return j < 0 or text[j] in '.!?\n:'


def _numbers(text):
    """(start, end, string) of standalone numbers; skips times, dates like 3/4, ids, ordinals."""
    for match in _NUMBER.finditer(text):
        a, b = match.span()
        before = text[a - 1] if a else ' '
        after = text[b] if b < len(text) else ' '
        after2 = text[b + 1] if b + 1 < len(text) else ' '
        if before.isalnum() or before in '.,:/-_':
            continue
        if after.isalnum() or after in ':/_-' or (after in '.,' and after2.isdigit()):
            continue
        yield a, b, match.group()


def _is_year(value):
    return re.fullmatch(r'(19|20)\d\d', value) is not None


def _format_like(original, number):
    if ',' in original:
        return f'{number:,}'
    return str(number)


def _new_number(original, rng, pool):
    """A different, plausible value in the same written style."""
    if _is_year(original):
        year = int(original) + rng.choice([-3, -2, -1, 1, 2, 3])
        return str(year)
    plain = original.replace(',', '')
    if '.' in plain:
        decimals = len(plain.split('.')[1])
        value = float(plain)
        for _ in range(10):
            new = round(value * rng.choice([0.5, 0.7, 0.8, 1.25, 1.5, 2.0]), decimals)
            text = f'{new:.{decimals}f}'
            if text != plain and new > 0:
                return text
        return None
    value = int(plain)
    same_size = [p for p in pool if p != value and len(str(p)) == len(str(value))]
    if same_size and rng.random() < 0.5:
        return _format_like(original, rng.choice(sorted(same_size)))
    if value <= 12:
        return str(rng.choice([n for n in range(1, 13) if n != value]))
    zeros = len(plain) - len(plain.rstrip('0'))
    step = 10 ** min(zeros, len(plain) - 1)
    for _ in range(10):
        new = int(round(value * rng.choice([0.5, 0.6, 0.75, 1.25, 1.5, 2.0]) / step)) * step
        if new > 0 and new != value:
            return _format_like(original, new)
    return None


def _preamble_end(answer):
    """End of a lead-in line such as "Sure! Here's a summary of the news in 49 words:"."""
    first = answer.split('\n', 1)[0]
    if first.rstrip().endswith(':') and re.search(r'summar|words', first, re.IGNORECASE):
        return len(first)
    return 0


def spacy_ner(model='en_core_web_sm'):
    """Returns ner(texts) -> list of [(start, end, label), ...] and a version string."""
    try:
        import spacy
        nlp = spacy.load(model, disable=['parser', 'lemmatizer', 'tagger', 'attribute_ruler'])
    except (ImportError, OSError) as exc:
        raise RuntimeError('Step 3 needs spaCy and its small English model: '
                           'pip install spacy && python -m spacy download en_core_web_sm') from exc

    def ner(texts):
        return [[(e.start_char, e.end_char, e.label_) for e in doc.ents]
                for doc in nlp.pipe(texts, batch_size=32)]
    return ner, f'spacy-{spacy.__version__}/{model}-{nlp.meta.get("version")}'


def _clean_entity(text, a, b):
    """Drop a leading 'the' and a trailing possessive so swaps keep the grammar."""
    if text[a:b].lower().startswith('the '):
        a += 4
    for suffix in ("'s", "’s"):
        if text[a:b].endswith(suffix):
            b -= len(suffix)
    return a, b


def _same_shape(entity, other, label):
    """Keep the sentence readable: letters only, same capitalization, same plural form."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'’&-]*", other):
        return False
    if (len(entity) > 3 and entity.isupper()) != (len(other) > 3 and other.isupper()):
        return False
    if label == 'NORP' and entity.endswith('s') != other.endswith('s'):
        return False
    return True


def _swap_candidates(answer, article, answer_ents, article_ents, rng):
    """Every possible single swap: (type, start, end, new_text)."""
    protected = _preamble_end(answer)
    candidates = []
    pool = {int(n.replace(',', '')) for _, _, n in _numbers(article)
            if '.' not in n and not _is_year(n)}
    pool -= {int(n.replace(',', '')) for _, _, n in _numbers(answer)
             if '.' not in n and not _is_year(n)}
    for a, b, value in _numbers(answer):
        new = _new_number(value, rng, pool)
        if new and new != value:
            candidates.append(('year' if _is_year(value) else 'number', a, b, new))
    for match in _WORD_NUMBER.finditer(answer):
        old = match.group()
        new = rng.choice([w for w in NUMBER_WORDS if w != old.lower()])
        candidates.append(('number_word', *match.span(), new.capitalize() if old[0].isupper() else new))
    for kind, pattern, names in (('month', _MONTH, MONTHS), ('weekday', _WEEKDAY, WEEKDAYS)):
        for match in pattern.finditer(answer):
            candidates.append((kind, *match.span(), rng.choice([n for n in names if n != match.group()])))
    by_label = {}
    for a, b, label in article_ents:
        if label in NAME_LABELS:
            a, b = _clean_entity(article, a, b)
            name = article[a:b]
            if len(name) > 1 and name[0].isupper():
                by_label.setdefault(label, set()).add(name)
    for a, b, label in answer_ents:
        if label not in NAME_LABELS:
            continue
        a, b = _clean_entity(answer, a, b)
        entity = answer[a:b]
        if len(entity) < 2 or not entity[0].isupper() or entity not in article:
            continue
        words = set(entity.split())
        options = set()
        for other in by_label.get(label, ()):
            if label == 'PERSON' and len(entity.split()) == 1:
                other = other.split()[-1]       # surname only, like the answer
            if (len(other.split()) == len(entity.split()) and not words & set(other.split())
                    and other not in answer and entity not in other and other not in entity
                    and _same_shape(entity, other, label)):
                options.add(other)
        if options:
            candidates.append(('name', a, b, rng.choice(sorted(options))))
    return [c for c in candidates if c[1] >= protected]


def _apply(text, edits):
    """edits: non-overlapping (start, end, new). Returns new text and new spans."""
    pieces, spans, last, position = [], [], 0, 0
    for a, b, new in sorted(edits):
        if a < last:
            raise ValueError('Overlapping edits')
        pieces.append(text[last:a])
        position += a - last
        spans.append((position, position + len(new)))
        pieces.append(new)
        position += len(new)
        last = b
    pieces.append(text[last:])
    return ''.join(pieces), spans


def _full_sentence(answer, a, b):
    """The annotated span as a whole sentence, or None."""
    if b < len(answer) and answer[b] in '.!?' and not answer[a:b].rstrip().endswith(('.', '!', '?')):
        b += 1
    while a < b and answer[a].isspace():
        a += 1
    while b > a and answer[b - 1].isspace():
        b -= 1
    sentence = answer[a:b]
    if (len(sentence.split()) < 5 or not sentence[0].isupper()
            or not sentence.endswith(('.', '!', '?')) or not _sentence_start(answer, a)
            or (b < len(answer) and not answer[b].isspace())):
        return None
    return sentence


def _sentence_breaks(answer):
    """Places to insert a sentence: after a sentence end, never inside a lead-in line."""
    protected = _preamble_end(answer)
    breaks = [m.start() for m in _SENTENCE_BREAK.finditer(answer)]
    end = len(answer.rstrip())
    if end and answer[end - 1] in '.!?':
        breaks.append(end)
    return sorted({b for b in breaks if b > protected})


def make(train, seed=0, ner=None, move_share=0.5, second_swap=0.25):
    """Synthetic summaries from error-free training summaries. Returns (samples, stats).

    ner(texts) -> per text a list of (start, end, label); defaults to spaCy.
    """
    summaries = [s for s in train if s['task'] == 'summary']
    if any(s['split'] != 'train' for s in summaries):
        raise ValueError('Synthetic data may only come from training answers')
    ner_version = 'given'
    if ner is None:
        ner, ner_version = spacy_ner()
    hallucinated = {}
    for s in summaries:
        for a, b in s['spans']:
            sentence = _full_sentence(s['answer'], a, b)
            if sentence:
                hallucinated.setdefault(s['source_id'], []).append((s['id'], sentence))
    clean = [s for s in summaries if not s['spans']]
    articles = {}
    for s in clean:
        articles.setdefault(s['source_id'], article_text(s['prompt']))
    article_ids = sorted(articles)
    article_ents = dict(zip(article_ids, ner([articles[i] for i in article_ids])))
    answer_ents = ner([s['answer'] for s in clean])
    out = []
    stats = {'version': SYNTH_VERSION, 'seed': seed, 'ner': ner_version,
             'error_free_summaries': len(clean), 'made': 0, 'skipped': 0,
             'swap': 0, 'move': 0, 'swap_types': {}}
    for s, ents in zip(clean, answer_ents):
        rng = random.Random(f'{seed}:{s["id"]}')
        answer = s['answer']
        moves = sorted({sentence for other, sentence in hallucinated.get(s['source_id'], [])
                        if other != s['id'] and sentence not in answer})
        breaks = _sentence_breaks(answer)
        swaps = _swap_candidates(answer, articles[s['source_id']], ents,
                                 article_ents[s['source_id']], rng)
        use_move = bool(moves and breaks) and (not swaps or rng.random() < move_share)
        if use_move:
            sentence = rng.choice(moves)
            at = rng.choice(breaks)
            new_answer, spans = _apply(answer, [(at, at, ' ' + sentence)])
            spans = [(spans[0][0] + 1, spans[0][1])]
            edits = [{'type': 'move', 'new': sentence}]
        elif swaps:
            first_type = rng.choice(sorted({c[0] for c in swaps}))
            first = rng.choice([c for c in swaps if c[0] == first_type])
            chosen = [first]
            others = [c for c in swaps if c[2] <= first[1] - 20 or c[1] >= first[2] + 20]
            if others and rng.random() < second_swap:
                chosen.append(rng.choice(others))
            chosen.sort(key=lambda c: c[1])
            new_answer, spans = _apply(answer, [(a, b, new) for _, a, b, new in chosen])
            edits = [{'type': t, 'old': answer[a:b], 'new': new} for t, a, b, new in chosen]
        else:
            stats['skipped'] += 1
            continue
        for (a, b), edit in zip(spans, edits):
            if new_answer[a:b] != edit['new']:
                raise AssertionError('Synthetic span does not match its edit')
        kind = 'move' if use_move else 'swap'
        stats[kind] += 1
        for edit in edits:
            if edit['type'] != 'move':
                stats['swap_types'][edit['type']] = stats['swap_types'].get(edit['type'], 0) + 1
        out.append({'id': f'syn-{s["id"]}', 'source_id': s['source_id'], 'task': 'summary',
                    'split': 'train', 'prompt': s['prompt'], 'answer': new_answer,
                    'spans': spans, 'synth': {'kind': kind, 'from': s['id'], 'edits': edits,
                                              'version': SYNTH_VERSION}})
    stats['made'] = len(out)
    return out, stats


def check_extra(extra, train, dev, test):
    """Synthetic answers must come from training sources only and have valid spans."""
    held_out = {s['source_id'] for s in dev} | {s['source_id'] for s in test}
    known = {s['id'] for s in train} | {s['id'] for s in dev} | {s['id'] for s in test}
    ids = set()
    for s in extra:
        if s.get('split') != 'train' or s['source_id'] in held_out:
            raise ValueError(f'Extra answer {s["id"]} uses a dev or test source')
        if s['id'] in known or s['id'] in ids:
            raise ValueError(f'Extra answer id {s["id"]} is not unique')
        ids.add(s['id'])
        for a, b in s['spans']:
            if not 0 <= a < b <= len(s['answer']):
                raise ValueError(f'Extra answer {s["id"]} has an invalid span')
    return [dict(s, spans=[tuple(span) for span in s['spans']]) for s in extra]
