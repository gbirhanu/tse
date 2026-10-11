# Hallucination span detection: proof of concept

Tests three of the levers from the concept note on RAGTruth, with span F1 as the measure.

| Lever | What it does | Where |
| --- | --- | --- |
| 1. Tuned span decoding | Threshold, merging of spans split by 1-3 tokens, and snapping to whole words, chosen per task | `hallu/decoding.py` |
| 2. Soft span targets | 2021-style soft targets: span edges moved by up to 5 words, weighted by J + J^2, mixed 0.6 / 0.4 with the 0/1 label | `hallu/soft.py` |
| 3. Character-level stacker | A small BiGRU over the answer's characters, fed the classifier's probabilities (the 2021 level two) | `hallu/stacker.py` |

## Run it on Colab

1. Put this folder in My Drive as `hallu_poc`. The RAGTruth data is downloaded from GitHub if `hallu_poc/RAGTruth` is not there.
2. `colab/part_a.ipynb` (about 15-20 minutes on a T4): the published LettuceDetect-large, plus levers 1 and 3. No training.
3. `colab/part_b.ipynb` (about 1-1.5 hours on a T4): ModernBERT-base trained twice, with 0/1 targets and with soft targets (lever 2), then levers 1 and 3 on each.

Results go to `hallu_poc/work/results`; `python poc.py report` prints them all as one table.

## How it is measured

- Span F1 is character-level and summed over the test set, computed exactly as LettuceDetect's `evaluate_detector_char_level` does ("LD" column). A second column counts each character once ("set"), because 125 pairs of RAGTruth gold spans overlap.
- Levers are fitted only on data the classifier did not train on: the dev split (10% of training sources) for our own models. The published LettuceDetect model was trained on all of RAGTruth train, so for it the levers are fitted on one half of the test set and scored on the other half, then the other way round. That is fine for a proof of concept, not for a paper.
- 95% intervals come from 1,000 bootstrap resamples of test sources. Gains are paired: the same resamples for both settings.

## Commands

```
python poc.py predict --model KRLabsOrg/lettucedect-large-modernbert-en-v1 --name ld_large
python poc.py levers  --name ld_large --fit crossfit
python poc.py train   --base answerdotai/ModernBERT-base --targets hard --name mb_hard
python poc.py train   --base answerdotai/ModernBERT-base --targets soft --name mb_soft
python poc.py levers  --name mb_hard --fit dev
python poc.py levers  --name mb_soft --fit dev
python poc.py report  --compare mb_hard mb_soft
```

All take `--data-dir` (RAGTruth) and `--work-dir`. For a CPU check without internet:
`python tools/make_tiny_modernbert.py --data-dir RAGTruth --out work_test/tiny`, then the commands above with
`--model work_test/tiny` / `--base work_test/tiny --lr 1e-3 --epochs 2 --limit 300 --test-limit 120`. Scores from the
tiny model mean nothing.
