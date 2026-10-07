"""
All file locations in one place. main.py and demo.py build a Paths object
from the command line, and every step reads and writes through it.

work_dir/
    folds/train_folds{seed}.csv          fold split per seed
    models/model_{seed}{fold}.bin        level-one RoBERTa models
    char_level/roberta{seed}-oof.pkl     out-of-fold char probabilities
    char_level/roberta{seed}-test.pkl    test char probabilities
    level2/rnn_model_{fold}.pt           level-two models
    level2/char_vocab.json               character vocabulary of level two
    scores.json, submission.csv
"""
import os


class Paths:
    def __init__(self, data_dir=None, work_dir='work', pretrained=None, models_dir=None):
        self.data_dir = data_dir
        self.work_dir = work_dir
        self.pretrained = pretrained
        self.folds_dir = os.path.join(work_dir, 'folds')
        self.models_dir = models_dir or os.path.join(work_dir, 'models')
        self.char_dir = os.path.join(work_dir, 'char_level')
        self.level2_dir = os.path.join(work_dir, 'level2')

    def make_dirs(self):
        for d in (self.folds_dir, self.models_dir, self.char_dir, self.level2_dir):
            os.makedirs(d, exist_ok=True)

    # input data
    @property
    def train_csv(self):
        return os.path.join(self.data_dir, 'train.csv')

    @property
    def test_csv(self):
        return os.path.join(self.data_dir, 'test.csv')

    # level one
    def folds_csv(self, seed_idx):
        return os.path.join(self.folds_dir, f'train_folds{seed_idx}.csv')

    def model_bin(self, seed_idx, fold):
        return os.path.join(self.models_dir, f'model_{seed_idx}{fold}.bin')

    def oof_fold_pkl(self, seed_idx, fold):
        return os.path.join(self.char_dir, f'roberta{seed_idx}-oof-fold{fold}.pkl')

    def oof_pkl(self, seed_idx):
        return os.path.join(self.char_dir, f'roberta{seed_idx}-oof.pkl')

    def test_pkl(self, seed_idx):
        return os.path.join(self.char_dir, f'roberta{seed_idx}-test.pkl')

    # level two
    def level2_model(self, fold):
        return os.path.join(self.level2_dir, f'rnn_model_{fold}.pt')

    @property
    def char_vocab_json(self):
        return os.path.join(self.level2_dir, 'char_vocab.json')

    @property
    def scores_json(self):
        return os.path.join(self.work_dir, 'scores.json')

    @property
    def submission_csv(self):
        return os.path.join(self.work_dir, 'submission.csv')
