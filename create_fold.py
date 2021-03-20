
import pandas as pd
from sklearn import model_selection
def create_fold(seed,id_):
    df = pd.read_csv("/content/train/train.csv")
    df = df.dropna().reset_index(drop=True)
    df["kfold"] = -1

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):
        print(len(trn_), len(val_))
        df.loc[val_, 'kfold'] = fold

    df.to_csv(f"/content/train/train_folds{id_}.csv", index=False)
