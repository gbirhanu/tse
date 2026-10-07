import pandas as pd
from sklearn import model_selection


def create_fold(train_csv, out_csv, seed, n_splits=5, limit=None):
    """Shuffles train.csv with the given seed and adds a stratified kfold column."""
    df = pd.read_csv(train_csv)
    df = df.dropna().reset_index(drop=True)
    if limit:
        df = df.head(limit)
    df["kfold"] = -1

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=n_splits)
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv(out_csv, index=False)
    print(f'Folds for seed {seed}: {out_csv} ({len(df)} rows)')
    return out_csv
