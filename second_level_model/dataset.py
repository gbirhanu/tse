import numpy as np
import torch

from second_level_model import helper


def get_start_end_string(text, selected_text):
    """Start (inclusive) and end (exclusive) character index of selected_text."""
    idx_start = text.find(selected_text) if selected_text else -1
    if idx_start < 0:
        return 0, 0
    return idx_start, idx_start + len(selected_text)


class TweetCharDataset(torch.utils.data.Dataset):
    def __init__(self, df, X, start_probas, end_probas,
                 n_models=1, max_len=150, train=True):
        self.max_len = max_len

        self.X = helper.pad_sequences(X, maxlen=max_len)

        self.start_probas = np.zeros((len(df), max_len, n_models), dtype=float)
        for i, p in enumerate(start_probas):
            len_ = min(len(p), max_len)
            self.start_probas[i, :len_] = p[:len_]
        # the seeds are averaged into one start and one end channel
        self.start_probas = np.mean(self.start_probas, axis=-1, keepdims=True)

        self.end_probas = np.zeros((len(df), max_len, n_models), dtype=float)
        for i, p in enumerate(end_probas):
            len_ = min(len(p), max_len)
            self.end_probas[i, :len_] = p[:len_]
        self.end_probas = np.mean(self.end_probas, axis=-1, keepdims=True)

        self.sentiments_list = ['positive', 'neutral', 'negative']

        self.texts = df['text'].values
        if train:
            self.selected_texts = df['selected_text'].values
        else:
            self.selected_texts = [''] * len(df)
        self.sentiments = df['sentiment'].values
        self.sentiments_input = [self.sentiments_list.index(s)
                                 for s in self.sentiments]

        # Targets (end is exclusive, so at most max_len - 1 to stay in range)
        if train:
            self.start_idx = []
            self.end_idx = []
            for text, sel_text in zip(df['text'].values,
                                      df['selected_text'].values):
                start, end = get_start_end_string(text, str(sel_text).strip())
                self.start_idx.append(min(start, max_len - 1))
                self.end_idx.append(min(end, max_len - 1))
        else:
            self.start_idx = [0] * len(df)
            self.end_idx = [0] * len(df)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'ids': torch.tensor(self.X[idx], dtype=torch.long),
                'probas_start': torch.tensor(self.start_probas[idx]).float(),
                'probas_end': torch.tensor(self.end_probas[idx]).float(),
                'target_start': torch.tensor(self.start_idx[idx], dtype=torch.long),
                'target_end': torch.tensor(self.end_idx[idx], dtype=torch.long),
                'text': self.texts[idx],
                'selected_text': self.selected_texts[idx],
                'sentiment': self.sentiments[idx],
                'sentiment_input': torch.tensor(self.sentiments_input[idx])}
