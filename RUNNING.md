# Running the code

Everything runs from one file, `main.py`. All paths are given on the command line,
so the same code runs on Kaggle, Colab or a local machine.

## What you need

- `train.csv` and `test.csv` from the Kaggle competition *Tweet Sentiment Extraction*.
- A RoBERTa folder with `config.json`, `vocab.json`, `merges.txt` and the weights.
  The 2021 run used `deepset/roberta-base-squad2` from Hugging Face.
- A GPU for training level one (25 RoBERTa models). Level two and the demo run on a CPU.
- `pip install -r requirements.txt` (Kaggle and Colab already have these packages).

## The pipeline

```
python main.py all --data-dir DATA --pretrained ROBERTA_DIR --work-dir work
```

`all` runs these steps in order. Each step can also be run on its own:

| Step | What it does | Writes to `work/` |
| --- | --- | --- |
| `folds` | 5-fold split of train.csv for each seed | `folds/` |
| `train1` | Level one: RoBERTa, 5 folds per seed | `models/`, `char_level/*-oof.pkl` |
| `oof1` | Out-of-fold predictions from saved models (only for models trained elsewhere) | `char_level/*-oof.pkl` |
| `test1` | Level one on test.csv (5 folds averaged per seed) | `char_level/*-test.pkl` |
| `train2` | Level two: character RNN on the level-one predictions, 5 folds | `level2/`, `scores.json` |
| `predict` | Final predictions for test.csv | `submission.csv` |

Useful options:

- `--seeds 0 1` uses only some of the 5 seeds (default: all five).
- `--folds 0 1 2` trains only some level-one folds, to split work across sessions.
  Finished folds are skipped when you run again.
- `--fp16` uses mixed precision on a GPU (faster, less memory).
- `--limit 1000 --test-limit 200` uses only the first rows, for a quick check.
- `--legacy-target-shift` reproduces a 2021 bug (see below), for comparison.

## On Kaggle

Open `kaggle/train_on_kaggle.ipynb` in a Kaggle notebook, add the competition data,
turn on a GPU and the internet, and run all cells.

## Quick test without a GPU

This checks that every step runs. It uses a tiny random model, so the scores mean nothing.

```
python tools/make_tiny_pretrained.py --train-csv DATA/train.csv --out work_test/tiny-roberta
python main.py all --data-dir DATA --work-dir work_test/run --pretrained work_test/tiny-roberta \
    --seeds 0 1 --limit 1000 --test-limit 200 --epochs 1 --level2-epochs 7 --max-len 160 --num-workers 0
```

## Local demo

After training, copy `work/` (or only some of `work/models/model_XY.bin` plus `work/level2/`)
to your computer, together with the RoBERTa folder (only `config.json`, `vocab.json` and
`merges.txt` are read from it):

```
python demo.py --pretrained ROBERTA_DIR --work-dir work
python demo.py --pretrained ROBERTA_DIR --work-dir work --text "my boss is bullying me..." --sentiment negative
```

The demo prints the level-one answer and the final answer after level two.

## What changed from the 2021 code

The repository did not run as committed. Fixed:

- One entry point (`main.py`) and one place for paths (`paths.py`) instead of Colab
  (`/content/drive/...`) and Kaggle (`../input/...`) paths spread over two config files.
- Crashes: calls to modules instead of functions (`run_train`, `run_oof_pred`),
  `TweetModel` looked up in the wrong module, `get_soft_labels` returned nothing,
  `SOFT_ALPHA` read from the wrong module, a 2-value unpack of a single number in `eval_fn`,
  a local variable hiding the `model` module, `pad_sequences` missing, and weights loaded
  from a file path string.
- Level-two row order: `DataFrame.append` results were thrown away and four seeds used
  fold 0 only, so the re-ordering of predictions did not work. Predictions are now matched by `textID`.
- Libraries that no longer exist or work: `keras.preprocessing.text` (replaced by a small
  character tokenizer with the same ids), `torchcontrib` SWA (replaced by
  `torch.optim.swa_utils`), `transformers.AdamW` (now `torch.optim.AdamW` with the same eps),
  ragged `np.array`.
- The character vocabulary of level two is saved at training time and loaded at prediction
  time; before, it was rebuilt from a differently ordered file.
- Target bug: level-one targets were placed one character to the right of the selected
  text, for every training tweet. Fixed; `--legacy-target-shift` brings the old behaviour back.
- Level-one out-of-fold predictions are written right after each fold, so no reload is needed.
- Runs on CPU or GPU (it was CUDA-only), with optional mixed precision.
