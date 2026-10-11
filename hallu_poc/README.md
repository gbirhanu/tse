# Hallucination span detection: proof of concept

Span detection on RAGTruth, with LettuceDetect-compatible character span F1 as the main measure.

## Next run: Part C

Open `colab/part_c.ipynb` after pushing the reviewed commits to **fix/runnable-pipeline**.
It clones that branch into `/content/tse` and calls `hallu_poc/poc.py` from there.
Predictions, checkpoints and results remain in `/content/drive/MyDrive/hallu_poc/work`.
Parts A and B remain historical notebooks using the existing Drive copy.

Run Part C's experiment cells individually:

1. Average the existing `mb_hard` and `mb_soft` token probabilities, with decoding fitted on dev.
2. Average `ld_large`, `mb_hard` and `mb_soft` with fixed decoding. Published LD has seen our dev
   sources, so neither dev tuning nor ensemble test cross-fitting is allowed for this mixture.
3. Train hard and soft targets at seeds 43 and 44 in four separate cells, keeping Part B's
   three epochs, 4,096 context and other training settings. Each cell has a 110-minute subprocess
   limit; rerun after a timeout or disconnect. T4 runtime is a budget, not a measured guarantee.
4. Evaluate the predeclared three-seed hard, three-seed soft, and six-model combined ensembles.
   Print each matched-seed soft-minus-hard comparison as well as the ensemble comparison.

Each candidate names its accepted reference before evaluation. The reference advances only when
the candidate's gain is positive and the lower bound of its paired 95% source-bootstrap interval
is strictly above zero. Rejected runs stay in the table for transparency and are not promoted.
Members and equal weights are fixed in advance; do not search combinations using test scores.
No new loss, longer context, PsiloQA pretraining or summary intervention is enabled until these
results have been reviewed. Each of those is a separate subsequent experiment.

### Resuming and old predictions

- Training writes model, optimizer, scheduler, scaler, RNG and batch position every 200 steps
  and at epoch boundaries. The selected dev model is saved before final predictions.
- Predictions checkpoint every 64 answers. Completion manifests validate configuration, input
  data and output hashes; changed settings require a new run name. Finished runs are skipped.
- Checkpoints and best models are under `work/models/<run>/`. Completed checkpoints discard
  optimizer state to reduce Drive usage; unfinished ones preserve it. Temporary saves can require
  another checkpoint's size, and recognized abandoned staging files are cleaned on resume.
- Legacy Part A/B files have no tokenizer fingerprint. `--legacy-tokenizer` re-tokenizes every
  requested answer and verifies its cached offsets. Historical training provenance is explicitly
  inherited from the documented experiment; it cannot be recovered from probabilities alone.
- Legacy prediction/result files are preserved. New records include data and tokenizer fingerprints.
- The character stacker is now opt-in (`--stacker`), because the prior runs showed no gain.

### Soft-target version

Part C intentionally retains the legacy soft-target implementation for matched-seed replication.
Its inward edge shifts can exceed five words on long spans, despite the original description.
Correcting the ranges is a separate ablation; it must not be silently mixed into the two new
soft seeds. Existing scores therefore describe **legacy soft targets**. Source-bootstrap intervals
measure test-source uncertainty, not all uncertainty across training seeds.

| Lever | What it does | Where |
| --- | --- | --- |
| 1. Tuned span decoding | Threshold, merging of spans split by 1-3 tokens, and snapping to whole words, chosen per task | `hallu/decoding.py` |
| 2. Soft span targets | 2021-style soft targets: span edges moved by up to 5 words, weighted by J + J^2, mixed 0.6 / 0.4 with the 0/1 label | `hallu/soft.py` |
| 3. Character-level stacker | A small BiGRU over the answer's characters, fed the classifier's probabilities (the 2021 level two) | `hallu/stacker.py` |

## Historical Parts A and B

1. Put this folder in My Drive as `hallu_poc`. The RAGTruth data is downloaded from GitHub if `hallu_poc/RAGTruth` is not there.
2. `colab/part_a.ipynb` (about 15-20 minutes on a T4): the published LettuceDetect-large, plus levers 1 and 3. No training.
3. `colab/part_b.ipynb` (about 1-1.5 hours on a T4): ModernBERT-base trained twice, with 0/1 targets and with soft targets (lever 2), then levers 1 and 3 on each.

Results go to `hallu_poc/work/results`; `python poc.py report` prints them all as one table.
With the updated CLI, use fresh run names for new experiments instead of overwriting historical runs.

## How it is measured

- Span F1 is character-level and summed over the test set, computed exactly as LettuceDetect's `evaluate_detector_char_level` does ("LD" column). A second column counts each character once ("set"), because 125 pairs of RAGTruth gold spans overlap.
- Levers are fitted only on data the classifier did not train on: the dev split (10% of training sources) for our own models. The published LettuceDetect model was trained on all of RAGTruth train, so for it the levers are fitted on one half of the test set and scored on the other half, then the other way round. That is fine for a proof of concept, not for a paper.
- 95% intervals come from 1,000 bootstrap resamples of test sources. Gains are paired: the same resamples for both settings.
- Keep/reject decisions use unrounded gains and interval bounds. CPU smoke results are labeled
  `smoke-only` and cannot become accepted research results.
- The secondary `Set task macro` column averages task F1 values. RLSeek's 60.1 is a task-macro
  result, not our corpus-micro LD F1; crossing 60.1 in the LD column alone is not an apples-to-apples
  claim of beating that system. See [RLSeek, section 5.1 and Table 1](https://aclanthology.org/2026.acl-long.1492.pdf)
  and [RL4HS, Table 1](https://arxiv.org/html/2510.02173v2). Confirm the complete evaluation protocol
  before making a paper-level comparison.

## Commands

```
python poc.py predict --model KRLabsOrg/lettucedect-large-modernbert-en-v1 --name ld_large
python poc.py levers  --name ld_large --fit crossfit
python poc.py train   --base answerdotai/ModernBERT-base --targets hard --name mb_hard
python poc.py train   --base answerdotai/ModernBERT-base --targets soft --name mb_soft
python poc.py levers  --name mb_hard --fit dev
python poc.py levers  --name mb_soft --fit dev
python poc.py report  --compare mb_hard mb_soft

python poc.py ensemble --name ens_hard_soft_s42 --members mb_hard mb_soft --fit dev \
    --legacy-tokenizer answerdotai/ModernBERT-base --against ld_large:baseline
python poc.py ensemble --name ens_ld_hard_soft_s42 --members ld_large mb_hard mb_soft --fit fixed \
    --legacy-tokenizer answerdotai/ModernBERT-base --against ld_large:baseline
```

The second ensemble command shows the initial reference. Part C automatically uses the accepted
previous reference if the first ensemble passes its gate. `--weights` accepts predeclared weights;
omitting it gives equal weights. Exact tokenizer definitions and ordered token offsets must match.

All take `--data-dir` (RAGTruth) and `--work-dir`. For a CPU check without internet:
`python tools/make_tiny_modernbert.py --data-dir RAGTruth --out work_test/tiny`, then the commands above with
`--model work_test/tiny` / `--base work_test/tiny --lr 1e-3 --epochs 2 --limit 300 --test-limit 120`. Scores from the
tiny model mean nothing.

The complete CPU integration check creates that tiny model, runs predict, hard/soft training,
dev tuning, ensembling, reporting and repeated completed commands, with `--limit 300 --test-limit 120`:

```
python tools/smoke_pipeline.py --data-dir RAGTruth --work-dir work_test/smoke
python -m pytest tests -q -p no:cacheprovider
```

It uses 1,024 context and one epoch to keep the CPU check short; it is not a T4 memory benchmark.
The tests require `torch>=2.5`, `transformers==4.57.6`, NumPy and pytest. Part C pins Transformers
and uses Colab's installed GPU PyTorch. The original corpus-level LD count formula is unchanged.
