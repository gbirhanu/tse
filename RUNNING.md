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
| `score2` | Level-two scores again from the saved models, no training (only for older runs) | `scores.json` |
| `predict` | Final predictions for test.csv | `submission.csv` |

Useful options:

- `--seeds 0 1` uses only some of the 5 seeds (default: all five).
- `--folds 0 1 2` trains only some level-one folds, to split work across sessions.
  Finished folds are skipped when you run again.
- `--fp16` uses mixed precision on a GPU (faster, less memory).
- `--limit 1000 --test-limit 200` uses only the first rows, for a quick check.
- `--legacy-target-shift` reproduces a 2021 bug (see below), for comparison.
- `--save-fp16-weights` saves level-one models at half size (about 250 MB instead of 500 MB).
- `--models-dir DIR` puts level-one models somewhere other than `work/models`.

## On Google Colab with Google Drive

1. Upload the whole `tse` folder to My Drive (the data is already in `tse/data`).
2. Open `colab/train_on_colab.ipynb` in Colab, choose a T4 GPU, and run the cells.

Outputs are written to `tse/work` on Drive, so nothing is lost when Colab disconnects;
run the cells again and finished folds are skipped. Train one seed per session
(`SEEDS = '0'`, then `'1'`, ...), then run level two once for all trained seeds.

Drive space: 25 full-size models need about 12 GB, close to the free 15 GB.
The notebook saves them at half size (about 6 GB). To keep them off Drive
completely, set `MODELS = '/content/models'` and run `train1` and `test1` for a
seed in the same session.

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

The demo prints the level-one answer, the final answer after level two, the confidence and
the estimated Jaccard. Add `--answer "bullying me"` to also print the real Jaccard.

## Web page and API

`serve.py` loads the same models and starts a web server (needs `pip install fastapi uvicorn`):

```
python serve.py --pretrained ROBERTA_DIR --work-dir work --port 8000
```

- `http://localhost:8000` is a web page: type a tweet, pick its sentiment, and the phrase is marked.
- `POST /api/extract` with `{"text": "...", "sentiment": "negative"}` returns the phrase, its start and
  end position in the text, the level-one answer, the confidence, the estimated Jaccard and the time taken.
  Add `"answer": "..."` (the correct phrase) to also get its real Jaccard score.
- `POST /api/extract_batch` with `{"items": [...]}` handles up to 64 tweets at once.
- `GET /api/health` says which models are loaded and their Jaccard on held-out training tweets;
  `/docs` lists every endpoint.
- Set the environment variable `TSE_API_KEY` to require an `X-API-Key` header.
- `--folds 0` loads one model per seed: faster to start, uses less memory.

What the scores mean:

- **Real Jaccard**: the real score for one tweet. Only possible when you give the correct phrase.
- **Estimated Jaccard**: `train2` keeps, for each sentiment, the average Jaccard of held-out tweets at
  10 confidence levels. A new tweet gets the average of its level. It needs level two. For a run made
  before this was added, `serve.py` and `demo.py` build the table once when they start, from the saved
  models without training. They need `train.csv`: in a `data` folder next to `work`, or give `--data-dir`.
  `python main.py score2 --data-dir DATA --work-dir work` does the same by hand.
- **Confidence**: P(start) × P(end) of the chosen phrase. It shows how sure the model is; it is not a score.
- **Held-out score**: the level-two CV Jaccard from `scores.json`, overall and per sentiment.

On Colab, `colab/serve_on_colab.ipynb` starts the server, gives you a private link, and
optionally a public link through a Cloudflare quick tunnel. Colab is not meant for running
web services, so use the public link for short demos only.

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
